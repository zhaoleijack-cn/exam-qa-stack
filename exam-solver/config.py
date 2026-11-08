"""配置与服务封装：知识库、检索、OCR、大模型。"""
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
# 旧版 material-service 时代的遗留常量，仅 healthz 展示用，留空即表示未配置
META_DB = Path(os.environ.get('META_DB', ''))
MILVUS = os.environ.get('MILVUS_URL', '')  # 已弃用（检索走 kb-service）
KB_SERVICE = os.environ.get('KB_SERVICE', 'http://kb-service:8090')
KB_API_KEY = os.environ.get('KB_API_KEY', '')
EMBED_URL = os.environ.get('EMBED_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings')
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'text-embedding-v4')

OCR_URL = os.environ.get('OCR_URL', 'http://127.0.0.1:8090/ocr')
ARK_URL = os.environ.get('ARK_URL', 'https://ark.cn-beijing.volces.com/api/plan/v3/chat/completions')
LLM_MODEL = os.environ.get('LLM_MODEL', 'deepseek-v4.1-flash')


def dashscope_key():
    return os.environ.get('DASHSCOPE_API_KEY', '')


def ark_key():
    return os.environ.get('HS_PLAN_API_KEY', '')


def _post(url, body, headers=None, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ---- 知识库清单：10 秒 TTL 缓存，避免每次请求都打 kb-service ----
_DS_CACHE = {'at': 0.0, 'data': []}
_DS_LOCK = threading.Lock()
DS_TTL = float(os.environ.get('KB_LIST_TTL', '10'))


def datasets(force=False):
    """v3: 知识库清单来自 kb-service（不再依赖 material-service 的 meta.db）。"""
    with _DS_LOCK:
        now = time.time()
        if not force and _DS_CACHE['data'] and now - _DS_CACHE['at'] < DS_TTL:
            return list(_DS_CACHE['data'])
    try:
        req = urllib.request.Request(f'{KB_SERVICE}/datasets',
                                     headers={'X-API-Key': KB_API_KEY})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        data = [{'id': x['dataset'], 'name': x['dataset'], 'collection': x['dataset'],
                 'chunks': x['chunks']} for x in d.get('datasets', [])]
    except Exception as e:
        print(f'[config] 读取知识库清单失败: {e}', flush=True)
        data = []
    with _DS_LOCK:
        _DS_CACHE['at'] = time.time()
        _DS_CACHE['data'] = data
    return list(data)


def embed(texts):
    d = _post(EMBED_URL, {'model': EMBED_MODEL, 'input': texts},
              {'Authorization': 'Bearer ' + dashscope_key()}, timeout=90)
    return [x['embedding'] for x in d['data']], d.get('usage', {}).get('total_tokens', 0)


def retrieve(collection, query, topk=6):
    """统一走 kb-service 检索（dataset=collection），返回结构与原 Milvus 版一致。"""
    d = _post(f'{KB_SERVICE}/search', {'query': query, 'dataset': collection, 'top_k': topk},
              {'X-API-Key': KB_API_KEY}, timeout=60)
    return [{'text': h.get('text', ''), 'doc': h.get('title', ''), 'heading': '',
             'page': 0, 'score': h.get('score', 0)} for h in d.get('results', [])]


def ocr_image(data, filename='q.jpg', strict=False):
    boundary = '----examsolver' + str(int(time.time() * 1000))
    parts = []
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="strict"\r\n\r\n'
                 f'{"true" if strict else "false"}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                 f'filename="{filename}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode() + data + b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    body = b''.join(parts)
    req = urllib.request.Request(OCR_URL, data=body,
                                 headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode())
        except Exception:
            return {'ok': False, 'error': f'OCR HTTP {e.code}'}


LLM_MODELS_FILE = BASE / 'data' / 'llm_models.json'
_LLM_LOCK = threading.Lock()


def llm_models():
    try:
        data = json.loads(LLM_MODELS_FILE.read_text('utf-8'))
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        return []
    except Exception as e:
        # 解析失败不再静默：坏文件留证据，避免"配置凭空消失"查不出原因
        print(f'[config] llm_models.json 解析失败: {e}', flush=True)
        return []


def save_llm_models(models):
    """原子写 + 进程内锁：并发编辑不会互相覆盖，写一半崩溃也不会留坏文件。"""
    with _LLM_LOCK:
        LLM_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(models, ensure_ascii=False, indent=1)
        fd, tmp = tempfile.mkstemp(dir=str(LLM_MODELS_FILE.parent), prefix='.llm_models.')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(text)
            # mkstemp 默认 0600 root:root；改为"数据目录所有者 + 0600"，
            # 宿主机备份可读、又不会把模型 Key 摊给所有人
            try:
                dirst = os.stat(str(LLM_MODELS_FILE.parent))
                os.chown(tmp, dirst.st_uid, dirst.st_gid)
            except Exception:
                pass
            os.chmod(tmp, 0o600)
            os.replace(tmp, LLM_MODELS_FILE)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

def active_llm():
    models = [m for m in llm_models() if m.get('role', 'answer') == 'answer']
    act = [m for m in models if m.get('active')]
    if act:
        return act[0]
    return {'id': 'env-default', 'name': '环境默认(火山ARK)', 'base_url': ARK_URL,
            'api_key': ark_key(), 'model': LLM_MODEL, 'active': True, 'builtin': True}

def llm_test(m):
    url = m['base_url'].strip()
    if not url.rstrip('/').endswith('/chat/completions'):
        url = url.rstrip('/') + '/chat/completions'
    payload = {'model': m['model'], 'messages': [{'role': 'user', 'content': '只回复: OK'}], 'max_tokens': 300}
    if m.get('extra'): payload.update(m['extra'])
    t0 = time.time()
    d = _post(url, payload, {'Authorization': 'Bearer ' + str(m.get('api_key', ''))}, timeout=60)
    reply = (d.get('choices') or [{}])[0].get('message', {}).get('content', '')
    return {'ok': True, 'ms': round((time.time() - t0) * 1000), 'reply': str(reply)[:60]}


def chat(messages, temperature=0, max_tokens=600, model_id=''):
    if model_id:
        hit = [x for x in llm_models() if x.get('id') == model_id]
        m = hit[0] if hit else active_llm()
    else:
        m = active_llm()
    url = m['base_url'].strip()
    if not url.rstrip('/').endswith('/chat/completions'):
        url = url.rstrip('/') + '/chat/completions'
    payload = {'model': m['model'], 'messages': messages,
               'temperature': temperature, 'max_tokens': max_tokens}
    if m.get('extra'): payload.update(m['extra'])
    d = _post(url, payload, {'Authorization': 'Bearer ' + str(m.get('api_key', ''))}, timeout=120)
    return d['choices'][0]['message']['content']
