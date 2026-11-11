"""试题图片 → 识别 → 检索知识库 → 生成答案（FastAPI）。"""
import base64
import hashlib
import json
import os
import secrets as pysecrets
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import config
import solver
import store

BASE = Path(__file__).resolve().parent
app = FastAPI(title='试题答题服务')
app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')

# 上传/写操作要鉴权；/live 只读看板公开（无鉴权、只读）
AUTH_USER = os.environ.get('EXAM_AUTH_USER', 'exam')
AUTH_PASS = os.environ.get('EXAM_AUTH_PASS', '')
# 只读看板用独立接口 /api/live/items，避免把管理用的 /api/items 整个暴露出去
PUBLIC_PATHS = {'/live', '/static/live.html', '/api/live/items', '/healthz', '/favicon.ico'}


def _is_public(path, method):
    return path in PUBLIC_PATHS and method in ('GET', 'HEAD')


def _authorized(request):
    if not AUTH_PASS:
        return False
    h = request.headers.get('authorization', '')
    if not h.lower().startswith('basic '):
        return False
    try:
        raw = base64.b64decode(h.split(' ', 1)[1]).decode('utf-8')
        user, _, pwd = raw.partition(':')
    except Exception:
        return False
    return (pysecrets.compare_digest(user, AUTH_USER) and
            pysecrets.compare_digest(pwd, AUTH_PASS))


@app.middleware('http')
async def auth_middleware(request: Request, call_next):
    # CSRF：Basic Auth 会被浏览器自动带上，跨站写请求必须挡掉（非浏览器客户端无此头，不受影响）
    if request.method not in ('GET', 'HEAD', 'OPTIONS') and \
            request.headers.get('sec-fetch-site') == 'cross-site':
        return JSONResponse({'detail': '拒绝跨站请求'}, status_code=403)
    if not _is_public(request.url.path, request.method) and not _authorized(request):
        return Response(status_code=401,
                        headers={'WWW-Authenticate': 'Basic realm="exam-upload", charset="UTF-8"'},
                        content='需要登录：只读看板请访问 /live')
    return await call_next(request)


@app.on_event('startup')
def startup():
    store.init()


@app.get('/')
def index():
    return FileResponse(BASE / 'static' / 'index.html', headers={'Cache-Control': 'no-store'})


@app.get('/live')
def live():
    """只读展示页：自动刷新试题与答案，无上传功能。"""
    return FileResponse(BASE / 'static' / 'live.html', headers={'Cache-Control': 'no-store'})


@app.get('/healthz')
def healthz():
    return {'ok': True, 'model': config.LLM_MODEL, 'ocr': config.OCR_URL,
            'milvus': config.MILVUS, 'datasets': len(config.datasets()),
            'items': len(store.all_items()),
            'keys': {'ark': bool(config.ark_key()), 'dashscope': bool(config.dashscope_key())}}


@app.get('/api/datasets')
def datasets():
    # 交互式读取：绕过 10 秒 TTL，新建的库立刻出现在下拉里
    return config.datasets(force=True)


def _dataset(ds_id):
    for d in config.datasets():
        if d['id'] == ds_id:
            return d
    raise HTTPException(404, '知识库不存在')


@app.post('/api/solve')
def solve(files: list[UploadFile] = File(...),
          dataset_id: str = Form(...),
          concurrency: int = Form(5),
          only_ocr: bool = Form(False),
          force: bool = Form(False),
          model_id: str = Form('')):
    """同步 def 路由：FastAPI 自动放入线程池，长任务不再阻塞事件循环。
    force=True 跳过去重缓存（换知识库重答 / 资料更新后重答）。"""
    ds = _dataset(dataset_id)
    concurrency = max(1, min(8, int(concurrency)))
    images, hashes, skipped = [], {}, []
    seen_hashes, total_bytes = set(), 0
    for f in files:
        data = f.file.read()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, f'{f.filename} 超过 20MB')
        total_bytes += len(data)
        if total_bytes > 120 * 1024 * 1024:
            raise HTTPException(413, '单次上传总量超过 120MB')
        h = hashlib.sha256(data).hexdigest()[:16]
        if not force and (h in seen_hashes or store.image_seen(h, ds['id'])):
            skipped.append(f.filename)
            continue
        seen_hashes.add(h)
        name = f.filename or 'q.jpg'
        while name in hashes:
            name = '_' + name
        images.append((name, data))
        hashes[name] = h
    if not images:
        return {'questions': [], 'errors': [], 'skipped': skipped, 'added': 0,
                'meta': {'count': 0, 'total_ms': 0},
                'dataset': {'id': ds['id'], 'name': ds['name']},
                'total': len(store.all_items())}

    result = solver.solve_images(images, ds, concurrency=concurrency, only_ocr=only_ocr, model_id=model_id)
    if not only_ocr and result['questions']:
        bid, added, n = store.save_batch(ds, result['questions'], hashes)
        result['batch'] = {'id': bid, 'added': added, 'images': n}
    result['skipped'] = skipped
    result['dataset'] = {'id': ds['id'], 'name': ds['name']}
    result['total'] = len(store.all_items())
    return result


@app.post('/api/solve/json')
def solve_json(payload: dict):
    import base64
    ds = _dataset(payload.get('dataset_id', ''))
    try:
        concurrency = int(payload.get('concurrency') or 5)
    except (TypeError, ValueError):
        concurrency = 5
    concurrency = max(1, min(8, concurrency))
    images, hashes, skipped = [], {}, []
    force = bool(payload.get('force'))
    seen_hashes, total_bytes = set(), 0
    for it in payload.get('images', []):
        b = it.get('base64', '')
        if ',' in b[:80]:
            b = b.split(',', 1)[1]
        data = base64.b64decode(b)
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, '单张图片超过 20MB')
        total_bytes += len(data)
        if total_bytes > 120 * 1024 * 1024:
            raise HTTPException(413, '单次请求总量超过 120MB')
        h = hashlib.sha256(data).hexdigest()[:16]
        if not force and (h in seen_hashes or store.image_seen(h, ds['id'])):
            skipped.append(it.get('name', 'q.jpg'))
            continue
        seen_hashes.add(h)
        name = it.get('name', 'q.jpg')
        while name in hashes:
            name = '_' + name
        images.append((name, data))
        hashes[name] = h
    if not images:
        return {'questions': [], 'errors': [], 'skipped': skipped, 'added': 0,
                'meta': {'count': 0, 'total_ms': 0}}
    result = solver.solve_images(images, ds, concurrency=concurrency,
                                 only_ocr=bool(payload.get('only_ocr')))
    if not payload.get('only_ocr') and result['questions']:
        bid, added, n = store.save_batch(ds, result['questions'], hashes)
        result['batch'] = {'id': bid, 'added': added, 'images': n}
    result['skipped'] = skipped
    result['total'] = len(store.all_items())
    return result


def _items_payload(brief=False):
    stats = store.stats()          # SQL 聚合：总数/最新时间/需复核/平均分
    if brief:
        return {'total': stats['total'], 'latest': stats['latest']}
    rows = store.all_items()
    return {'items': rows, 'total': stats['total'], 'latest': stats['latest'],
            'needs_review': stats['needs_review'],
            'avg_confidence': stats['avg_confidence'], 'batches': store.batches()}


@app.get('/api/items')
def items(brief: int = 0):
    return _items_payload(bool(brief))


@app.get('/api/live/items')
def live_items(brief: int = 0):
    """只读看板专用：与 /api/items 同结构，公开可读。"""
    return _items_payload(bool(brief))


@app.delete('/api/items/{item_id}')
def del_item(item_id: str):
    store.delete_item(item_id)
    return {'ok': True}


@app.delete('/api/batches/{batch_id}')
def del_batch(batch_id: str):
    store.delete_batch(batch_id)
    return {'ok': True}


@app.post('/api/clear')
def clear():
    store.clear()
    return {'ok': True}


@app.post('/api/markdown')
def markdown(payload: dict):
    rows = payload.get('items') or store.all_items()
    return {'markdown': solver.to_markdown({'questions': rows})}

def _mask(k):
    k = str(k or '')
    return (k[:6] + '...' + k[-4:]) if len(k) > 12 else '***'


@app.get('/api/llm/models')
def llm_list():
    models = config.llm_models()
    if not models:
        models = [config.active_llm()]
    grouped = {'answer': [], 'ocr': []}
    for m in models:
        d = dict(m)
        d['api_key_masked'] = _mask(m.get('api_key'))
        d.pop('api_key', None)
        grouped.setdefault(d.get('role', 'answer'), []).append(d)
    return grouped


@app.post('/api/llm/models')
def llm_add(payload: dict):
    role = payload.get('role') or 'answer'
    if role not in ('answer', 'ocr'):
        raise HTTPException(400, 'role 必须是 answer 或 ocr')
    name = (payload.get('name') or '').strip()
    base_url = (payload.get('base_url') or '').strip()
    api_key = (payload.get('api_key') or '').strip()
    model = (payload.get('model') or '').strip()
    if not (name and base_url and model):
        raise HTTPException(400, 'name / base_url / model 必填；编辑时 api_key 留空=沿用原值')
    models = config.llm_models()
    mid = (payload.get('id') or '').strip() or (role + '-' + name.strip().lower().replace(' ', '_')[:20])
    existing = [m for m in models if m.get('id') == mid and m.get('role', 'answer') == role]
    entry = {'id': mid, 'role': role, 'name': name, 'base_url': base_url,
             'api_key': api_key or (existing[0].get('api_key', '') if existing else ''),
             'model': model, 'active': bool(payload.get('active'))}
    if payload.get('extra'):
        entry['extra'] = payload['extra']
    elif existing and existing[0].get('extra'):
        entry['extra'] = existing[0]['extra']   # 编辑时未带 extra 则沿用原值
    if not entry['api_key']:
        raise HTTPException(400, '新模型必须提供 api_key')
    models = [m for m in models if not (m.get('id') == mid and m.get('role', 'answer') == role)]
    same = [m for m in models if m.get('role', 'answer') == role]
    if entry['active'] or not same:
        for m in same: m['active'] = False
        entry['active'] = True
    models.append(entry)
    config.save_llm_models(models)
    return {'ok': True, 'id': mid, 'role': role, 'active': entry['active'], 'updated': bool(existing)}


@app.get('/api/llm/models/{role}/{mid}/raw')
def llm_raw(role: str, mid: str):
    hit = [m for m in config.llm_models() if m.get('id') == mid and m.get('role', 'answer') == role]
    if not hit: raise HTTPException(404, 'model not found')
    d = dict(hit[0])
    d['api_key'] = ''            # 不回显明文；编辑时留空=沿用原值
    d['api_key_set'] = bool(hit[0].get('api_key'))
    return d


@app.post('/api/llm/models/{role}/{mid}/active')
def llm_active(role: str, mid: str):
    if role not in ('answer', 'ocr'):
        raise HTTPException(400, 'bad role')
    models = config.llm_models()
    hit = False
    for m in models:
        if m.get('role', 'answer') == role:
            m['active'] = (m.get('id') == mid)
            if m.get('id') == mid: hit = True
    if not hit: raise HTTPException(404, 'model not found')
    config.save_llm_models(models)
    return {'ok': True, 'role': role, 'active': mid}


@app.post('/api/llm/models/{role}/{mid}/test')
def llm_test_route(role: str, mid: str):
    hit = [m for m in config.llm_models() if m.get('id') == mid and m.get('role', 'answer') == role]
    if not hit: raise HTTPException(404, 'model not found')
    return config.llm_test(hit[0])


@app.delete('/api/llm/models/{role}/{mid}')
def llm_del(role: str, mid: str):
    models = config.llm_models()
    models2 = [m for m in models if not (m.get('id') == mid and m.get('role', 'answer') == role)]
    if len(models2) == len(models): raise HTTPException(404, 'model not found')
    config.save_llm_models(models2)
    return {'ok': True}
