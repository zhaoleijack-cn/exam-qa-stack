"""答题流水线：OCR → 检索 → 生成 → 校验（低置信度复核）。"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import config

# 全局 LLM 并发闸：多个批次/多人同时上传时，避免把上游打爆（429）
_LLM_SEM = threading.BoundedSemaphore(max(1, int(os.environ.get('LLM_MAX_CONCURRENCY', '8'))))
MAX_CONCURRENCY = 8

SYSTEM = ('你是银行考试答题助手。只能依据用户给出的【资料】作答，不得使用资料以外的知识。'
          '资料不足时降低 confidence，不要猜。只输出 JSON，不要解释。')

PROMPT = '''【题目】{stem}
【选项】
{options}
【资料】
{context}

请判断正确选项，严格输出 JSON：
{{"answer":"A","confidence":0.9,"evidence":[1,2],"reason":"40字以内，说明依据"}}
要求：
- answer 只能是 {letters} 中的一个（多选写连续字母，如 "ABD"；判断题写 "正确" 或 "错误"）
- confidence 为 0~1，资料不支持时给 0.3 以下
- evidence 是【资料】编号数组，必须来自上面
- reason 不超过 40 字'''

PROMPT_RETRY = '''只输出 JSON，不要任何其他文字。
题目：{stem}
选项：{options}
资料：
{context}
格式：{{"answer":"A","confidence":0.8,"evidence":[1],"reason":"..."}}
answer 必须是 {letters} 中的字母（多选可多个）。'''


def build_query(q):
    opts = q.get('options') or {}
    return q.get('stem', '') + '\n' + '\n'.join(f'{k}. {v}' for k, v in opts.items())


def format_context(hits):
    lines = []
    for i, h in enumerate(hits, 1):
        body = re.sub(r'\s+', ' ', h['text'])[:420]
        src = f"{h['doc']}" + (f" 第{h['page']}页" if h.get('page') else '')
        lines.append(f'[{i}]（{src}）{body}')
    return '\n'.join(lines)


def _json_obj(s):
    s = re.sub(r'^```(?:json)?|```$', '', s.strip(), flags=re.M).strip()
    start = s.find('{')
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except Exception:
                    return None
    return None


def norm_answer(raw, qtype, letters):
    """把模型输出归一为合法答案，返回 (answer, ok)。

    v1.0.2 严格化：只接受"紧凑"的字母答案（去除空白标点后仅由合法选项字母组成），
    不再从长文本里捞字母——此前 "The answer is B" 会被洗成 A、"cannot determine" 变 C，
    无效输出绕过重试直接变确定答案（2026-11-02 外部审查复现）。中文容错保留（"我选C"→C）。"""
    if raw is None:
        return '', False
    s = re.sub(r'[\s。．.！!？?，,：:；;（）()【】\[\]「」、·"\'“”‘’]', '', str(raw)).upper()
    compact = re.sub(r'[^A-Z]', '', s)
    if qtype == '判断':
        if s in ('A', 'T', 'TRUE', '正确', '对', '√', '是', 'YES'):
            return '正确', True
        if s in ('B', 'F', 'FALSE', '错误', '错', '×', '否', 'NO'):
            return '错误', True
        if '不正确' in s or '错误' in s or '错' in s:
            return '错误', True
        if '正确' in s or '对' in s:
            return '正确', True
        return '', False
    # 单选：紧凑后必须是恰好一个合法字母
    if qtype != '多选':
        return (compact, True) if len(compact) == 1 and compact in letters else ('', False)
    # 多选：紧凑后必须是 2~N 个不重复的合法字母（乱序归一为升序）
    uniq = ''.join(sorted(dict.fromkeys(compact)))
    if 2 <= len(uniq) <= len(letters) and all(c in letters for c in uniq):
        return uniq, True
    return '', False


def generate(q, hits, retry=False, temperature=0, model_id=''):
    opts = q.get('options') or {}
    letters = ','.join(opts.keys()) if opts else '正确,错误'
    opt_txt = '\n'.join(f'{k}. {v}' for k, v in opts.items()) or '（判断题，无选项）'
    if q.get('type') == '判断':
        opt_txt += '\n说明：这是判断题，answer 必须写 "正确" 或 "错误"，不要写字母。'
    tpl = PROMPT_RETRY if retry else PROMPT
    user = tpl.format(stem=q.get('stem', ''), options=opt_txt,
                      context=format_context(hits), letters=letters)
    t0 = time.time()
    with _LLM_SEM:
        raw = config.chat([{'role': 'system', 'content': SYSTEM},
                           {'role': 'user', 'content': user}], temperature=temperature, model_id=model_id)
    obj = _json_obj(raw) or {}
    ans, ok = norm_answer(obj.get('answer'), q.get('type', '单选'), set(opts.keys()) or {'正确', '错误'})
    conf = obj.get('confidence', 0.5)
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except Exception:
        conf = 0.5
    ev = obj.get('evidence') if isinstance(obj.get('evidence'), list) else []
    ev = [int(x) for x in ev if str(x).isdigit() and 1 <= int(x) <= len(hits)][:3]
    return {'answer': ans, 'ok': ok, 'confidence': conf,
            'reason': str(obj.get('reason', ''))[:120], 'evidence': ev,
            'raw': raw if not ok else '', 'ms': round((time.time() - t0) * 1000)}


def answer_one(q, dataset, topk=6, verify_below=0.6, model_id=''):
    t0 = time.time()
    hits = []
    err = ''
    try:
        hits = config.retrieve(dataset['collection'], build_query(q), topk)
    except Exception as e:
        hits = []
        err = f'检索失败: {e}'
    else:
        err = ''

    if not hits:
        return {'type': q.get('type', '单选'), 'stem': q.get('stem', ''),
                'options': q.get('options') or {}, 'answer': '', 'confidence': 0.0,
                'reason': err or '知识库没有检索到相关内容', 'evidence': [],
                'needs_review': True, 'ms': {'total': round((time.time() - t0) * 1000)}}

    g = generate(q, hits, model_id=model_id)
    if not g['ok']:
        g = generate(q, hits, retry=True, model_id=model_id)

    # 低置信度复核：再跑一次，一致则加分，不一致则标记
    if g['ok'] and g['confidence'] < verify_below:
        g2 = generate(q, hits, retry=True, temperature=0.3, model_id=model_id)
        if g2['ok'] and g2['answer'] == g['answer']:
            g['confidence'] = min(1.0, max(g['confidence'], g2['confidence']) + 0.08)
            g['reason'] = g['reason'] or g2['reason']
        elif g2['ok']:
            if g2['confidence'] > g['confidence']:
                g = g2
            g['confidence'] = min(g['confidence'], 0.45)

    best = hits[0]['score'] if hits else 0
    # v1.0.2: ①空引用不得自动通过 ②OCR 质量低（如双跑冲突）强制复核；警告附进 reason
    ocr_conf = q.get('_ocr_confidence', 'high')
    ocr_warns = q.get('_ocr_warnings') or []
    ocr_note = ''
    if ocr_conf == 'low':
        ocr_note = 'OCR识别质量低'
    elif ocr_warns:
        ocr_note = 'OCR警告: ' + '；'.join(str(w) for w in ocr_warns[:2])
    needs_review = (not g['ok']) or not g['evidence'] or g['confidence'] < 0.55 or best < 0.3 or ocr_conf == 'low'
    reason = g['reason']
    if ocr_note:
        reason = (reason + '　' if reason else '') + '⚠️ ' + ocr_note
    return {
        'type': q.get('type', '单选'), 'stem': q.get('stem', ''),
        'options': q.get('options') or {}, 'answer': g['answer'],
        'confidence': round(g['confidence'], 2), 'reason': reason,
        'evidence': [{'n': n, **{k: hits[n - 1][k] for k in ('doc', 'page', 'score')},
                      'text': hits[n - 1]['text'][:300]} for n in g['evidence']],
        'top_score': best, 'needs_review': needs_review,
        'ms': {'total': round((time.time() - t0) * 1000), 'gen': g['ms']},
        '_raw': g['raw'],
    }


def solve_images(images, dataset, concurrency=5, only_ocr=False, model_id=''):
    """images: [(filename, bytes)]；返回 {"questions":[...], "meta":{...}}"""
    t0 = time.time()
    def do_ocr(item):
        name, data = item
        t = time.time()
        try:
            r = config.ocr_image(data, name)
        except Exception as e:  # 单图异常（网络/超时）不炸整批
            r = {'ok': False, 'error': f'OCR 异常: {e}'}
        return name, r, round((time.time() - t) * 1000)

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(images)))) as ex:
        ocr_results = list(ex.map(do_ocr, images))

    questions, errors = [], []
    for name, r, ms in ocr_results:
        if not r.get('ok'):
            errors.append({'image': name, 'error': (r.get('detail') or r.get('error') or 'OCR 失败')})
            continue
        for q in r.get('questions', []):
            q['_image'] = name
            q['_ocr_ms'] = ms
            # v1.0.2: OCR 质量贯穿到每题——low/警告不再被答题端高置信掩盖
            q['_ocr_confidence'] = r.get('confidence', 'high')
            # 部分题目校验失败时，OCR 会返回可用题目 + errors；把 errors 也带进警告
            q['_ocr_warnings'] = list(r.get('warnings') or []) + list(r.get('errors') or [])
            questions.append(q)

    if only_ocr:
        return {'questions': questions, 'errors': errors,
                'meta': {'ocr_ms': round((time.time() - t0) * 1000), 'count': len(questions)}}

    t1 = time.time()
    def _answer_safe(q):
        try:
            return answer_one(q, dataset, model_id=model_id)
        except Exception as e:  # 单题异常（生成超时/接口错误）不炸整批，占位待复核
            return {'type': q.get('type', '单选'), 'stem': q.get('stem', ''),
                    'options': q.get('options') or {}, 'answer': '', 'confidence': 0.0,
                    'reason': f'答题异常: {e}', 'evidence': [],
                    'needs_review': True,
                    'ms': {'total': 0}}

    concurrency = max(1, min(MAX_CONCURRENCY, int(concurrency or 1)))
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        answers = list(ex.map(_answer_safe, questions))

    out = []
    for i, (q, a) in enumerate(zip(questions, answers), 1):
        a['index'] = i
        a['image'] = q.get('_image', '')
        out.append(a)

    need = sum(1 for a in out if a['needs_review'])
    return {
        'questions': out, 'errors': errors,
        'meta': {
            'count': len(out), 'needs_review': need,
            'avg_confidence': round(sum(a['confidence'] for a in out) / len(out), 2) if out else 0,
            'ocr_ms': round((t1 - t0) * 1000), 'answer_ms': round((time.time() - t1) * 1000),
            'total_ms': round((time.time() - t0) * 1000),
        },
    }


def to_markdown(result):
    lines = []
    for a in result['questions']:
        lines.append(f"### {a['index']}. [{a['type']}] {a['stem']}")
        for k, v in a['options'].items():
            mark = ' ✅' if k in (a['answer'] or '') else ''
            lines.append(f"- {k}. {v}{mark}")
        lines.append(f"\n**答案：{a['answer'] or '（未确定）'}**　置信度 {a['confidence']}"
                     + ('　⚠️ 建议复核' if a['needs_review'] else ''))
        if a['reason']:
            lines.append(f"> {a['reason']}")
        for e in a['evidence'][:2]:
            lines.append(f"> 依据：{e['doc']} 第{e['page']}页（{e['score']}）")
        lines.append('')
    return '\n'.join(lines)
