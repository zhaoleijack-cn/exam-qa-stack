"""知识库服务：正文（SQLite+FTS5）+ 向量（.npy/.usearch）+ RRF 混合检索。"""
import glob
import io
import json
import os
import re
import secrets
import threading
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from dataset import Dataset
from db import SCHEMA_VERSION
import search as searchmod

DATA_DIR = os.environ.get('KB_DATA', '/data')
VEC_DIR = os.environ.get('KB_VEC_DIR', os.path.join(DATA_DIR, 'vec'))
CFG = json.load(open(os.path.join(DATA_DIR, 'config.json')))
API_KEY = str(CFG.get('api_key') or '').strip()
if not API_KEY:
    # 空 key 会让 `x_api_key != API_KEY` 对无头请求成立（false 变成 pass），必须拒绝启动
    raise RuntimeError('data/kb/config.json 的 api_key 为空：拒绝启动（否则等于无鉴权）')
EMBED_MODEL = CFG.get('embed_model', 'text-embedding-v4')
DIMS = int(CFG.get('embed_dims', 1536))
CHUNK_SIZE = max(50, int(CFG.get('chunk_size', 800)))
CHUNK_OVERLAP = max(0, min(CHUNK_SIZE - 50, int(CFG.get('chunk_overlap', 100))))
DS_RE = re.compile(r'^[a-z0-9_]{1,32}$')

app = FastAPI(title='Knowledge Base Service', version='3.0',
              description='知识库服务：正文 SQLite + FTS5，向量 HNSW 独立文件，混合检索 RRF。')

STORES = {}
STORE_LOCK = threading.Lock()


def auth(x_api_key: str = Header(default='')):
    if not API_KEY or not secrets.compare_digest(str(x_api_key), API_KEY):
        raise HTTPException(401, 'invalid api key')


def check_ds(dataset: str) -> str:
    if not dataset or not DS_RE.match(dataset):
        raise HTTPException(400, 'dataset 必填且只能含小写字母/数字/下划线（1-32位）')
    return dataset


def get_store(dataset: str, create: bool = False) -> Optional[Dataset]:
    check_ds(dataset)
    with STORE_LOCK:
        if dataset not in STORES:
            path = os.path.join(DATA_DIR, 'kb_%s.db' % dataset)
            if not os.path.exists(path) and not create:
                return None
            STORES[dataset] = Dataset(dataset, DATA_DIR, VEC_DIR, CFG)
        return STORES[dataset]


# ---------- 分块与解析 ----------
def _split_sentences(line: str) -> List[str]:
    """按句末标点切成完整句子；单句仍超 CHUNK_SIZE 的极端情况才带重叠硬切。"""
    segs, buf = [], ''
    for ch in line:
        buf += ch
        if ch in '。！？；!?':
            if buf.strip():
                segs.append(buf.strip())
            buf = ''
    if buf.strip():
        segs.append(buf.strip())
    out = []
    for s in segs:
        if len(s) <= CHUNK_SIZE:
            out.append(s)
        else:
            start = 0
            while start < len(s):
                out.append(s[start:start + CHUNK_SIZE])
                start += CHUNK_SIZE - CHUNK_OVERLAP
    return [s for s in out if s]


def chunk_text(text: str) -> List[str]:
    """句子边界感知切块：任何块都不从句子中间开始/结束，相邻块保留一个完整句重叠。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]
    segs = []
    for line in text.replace('\r\n', '\n').split('\n'):
        line = line.strip()
        if not line:
            continue
        if len(line) <= CHUNK_SIZE:
            segs.append(line)
        else:
            segs.extend(_split_sentences(line))
    chunks, cur = [], ''
    for s in segs:
        cand = (cur + '\n' + s) if cur else s
        if len(cand) <= CHUNK_SIZE:
            cur = cand
            continue
        if cur:
            chunks.append(cur)
        tail = cur.split('\n')[-1] if cur else ''
        cur = tail if 0 < len(tail) <= CHUNK_OVERLAP * 2 else ''
        cand = (cur + '\n' + s) if cur else s
        cur = cand if len(cand) <= CHUNK_SIZE else s
    if cur:
        chunks.append(cur)
    return chunks

def parse_pdf(b):
    from pypdf import PdfReader
    return '\n'.join((p.extract_text() or '') for p in PdfReader(io.BytesIO(b)).pages)


def parse_docx(b):
    import docx
    return '\n'.join(p.text for p in docx.Document(io.BytesIO(b)).paragraphs if p.text.strip())


def parse_txt(b):
    for enc in ('utf-8', 'gb18030'):
        try:
            return b.decode(enc)
        except Exception:
            pass
    return b.decode('utf-8', 'replace')


def parse_file(name, b):
    low = (name or '').lower()
    if low.endswith('.pdf'):
        return parse_pdf(b)
    if low.endswith('.docx'):
        return parse_docx(b)
    return parse_txt(b)


# ---------- 模型 ----------
class SearchReq(BaseModel):
    query: str
    dataset: str
    top_k: int = 8
    min_score: float = 0.0
    mode: Optional[str] = None           # embedding | fulltext | hybrid
    embedding_weight: Optional[float] = None


class BulkItem(BaseModel):
    title: str
    text: str


class BulkReq(BaseModel):
    dataset: str
    items: List[BulkItem]


class TextReq(BaseModel):
    dataset: str
    title: str
    text: str


# ---------- 路由 ----------
@app.post('/search')
def search(req: SearchReq, _: None = Depends(auth)):
    q = (req.query or '').strip()
    if not q:
        raise HTTPException(400, 'empty query')
    if len(q) > 2000:
        raise HTTPException(400, 'query 过长（>2000 字）')
    top_k = max(1, min(50, int(req.top_k or 8)))
    ds = get_store(req.dataset, create=False)
    if ds is None:
        raise HTTPException(404, '知识库不存在')
    try:
        results, info = ds.search(q, top_k=top_k, min_score=req.min_score,
                                  mode=req.mode or searchmod.MODE_DEFAULT,
                                  embedding_weight=req.embedding_weight)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, str(e)[:200])
    return {'results': results, 'top_k': top_k, 'dataset': ds.name,
            'score_fields': 'score=向量余弦(仅全文命中时为0.5)；融合排序看 rrf_score',
            **info}


@app.post('/ingest/file')
def ingest_file(file: UploadFile = File(...), dataset: str = Form(...),
                title: Optional[str] = Form(None), _: None = Depends(auth)):
    # 同步 def：PDF 解析 + embedding 是重活，走线程池，不阻塞事件循环
    ds = get_store(dataset, create=True)
    if ds is None:
        raise HTTPException(400, 'dataset 非法')
    b = file.file.read()
    if len(b) > 30 * 1024 * 1024:
        raise HTTPException(400, 'file too large (max 30MB)')
    title = title or file.filename
    try:
        text = parse_file(file.filename, b)
    except Exception as e:
        raise HTTPException(400, 'parse failed: ' + str(e)[:150])
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, 'no content extracted')
    try:
        r = ds.ingest(title, chunks, source=file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, str(e)[:200])
    return {**r, 'chars': len(text)}


@app.post('/ingest/text')
def ingest_text(req: TextReq, _: None = Depends(auth)):
    ds = get_store(req.dataset, create=True)
    if ds is None:
        raise HTTPException(400, 'dataset 非法')
    chunks = chunk_text(req.text)
    if not chunks:
        raise HTTPException(400, 'empty text')
    try:
        r = ds.ingest(req.title, chunks, source='text')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, str(e)[:200])
    return r


@app.post('/ingest/bulk')
def ingest_bulk(req: BulkReq, _: None = Depends(auth)):
    ds = get_store(req.dataset, create=True)
    if ds is None:
        raise HTTPException(400, 'dataset 非法')
    grouped, order = {}, []
    for it in req.items:
        if it.title not in grouped:
            grouped[it.title] = []
            order.append(it.title)
        grouped[it.title].append(it.text)
    out = []
    for title in order:
        chunks = chunk_text('\n'.join(grouped[title]))
        if not chunks:
            continue
        r = ds.ingest(title, chunks, source='bulk')
        out.append({'doc_id': r['doc_id'], 'title': title, 'chunks': r['chunks']})
    return {'imported': out, 'total_chunks': sum(o['chunks'] for o in out)}


@app.get('/datasets')
def datasets(_: None = Depends(auth)):
    out = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, 'kb_*.db'))):
        name = os.path.basename(p)[3:-3]
        try:
            ds = get_store(name)
            if ds is not None:
                out.append(ds.stats())
        except Exception as e:
            out.append({'dataset': name, 'error': str(e)[:120]})
    return {'datasets': out}


@app.get('/documents/{dataset}')
def documents(dataset: str, _: None = Depends(auth)):
    ds = get_store(dataset, create=False)
    if ds is None:
        raise HTTPException(404, '知识库不存在')
    return {'documents': ds.documents()}


@app.get('/documents/{dataset}/{doc_id}/chunks')
def get_chunks(dataset: str, doc_id: str, _: None = Depends(auth)):
    ds = get_store(dataset, create=False)
    if ds is None:
        raise HTTPException(404, '知识库不存在')
    return {'doc_id': doc_id, 'chunks': ds.list_chunks(doc_id)}


@app.put('/documents/{dataset}/chunk/{rowid}')
def edit_chunk(dataset: str, rowid: int, body: dict, _: None = Depends(auth)):
    text = str(body.get('text') or '').strip()
    if not text:
        raise HTTPException(400, '块内容不能为空')
    ds = get_store(dataset, create=False)
    if ds is None:
        raise HTTPException(404, '知识库不存在')
    chars = ds.edit_chunk(rowid, text)
    return {'ok': True, 'chars': chars}


@app.delete('/documents/{dataset}/chunk/{rowid}')
def del_chunk(dataset: str, rowid: int, _: None = Depends(auth)):
    ds = get_store(dataset, create=False)
    if ds is None:
        raise HTTPException(404, '知识库不存在')
    ds.delete_chunk(rowid)
    return {'ok': True}


@app.post('/documents/{dataset}/merge-chunks')
def merge_chunks(dataset: str, body: dict, _: None = Depends(auth)):
    ds = get_store(dataset, create=False)
    if ds is None:
        raise HTTPException(404, '知识库不存在')
    try:
        chars = ds.merge_chunks(body.get('rowids') or [])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {'ok': True, 'chars': chars}


@app.get('/config')
def get_cfg(_: None = Depends(auth)):
    return {'chunk_size': CHUNK_SIZE, 'chunk_overlap': CHUNK_OVERLAP,
            'embed_model': CFG.get('embed_model'), 'embed_dims': CFG.get('embed_dims')}


@app.post('/config')
def set_cfg(body: dict, _: None = Depends(auth)):
    global CHUNK_SIZE, CHUNK_OVERLAP
    try:
        size = int(body.get('chunk_size', CHUNK_SIZE))
        overlap = int(body.get('chunk_overlap', CHUNK_OVERLAP))
    except Exception:
        raise HTTPException(400, 'chunk_size/chunk_overlap 必须是整数')
    size = max(200, min(6000, size))
    overlap = max(0, min(size // 2, overlap))
    CFG['chunk_size'] = size
    CFG['chunk_overlap'] = overlap
    # 原子写：config.json 含 api_key/embed_token，写一半损坏会导致下次启动 fail-closed
    cfg_path = os.path.join(DATA_DIR, 'config.json')
    tmp = cfg_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(CFG, f, ensure_ascii=False, indent=1)
    os.replace(tmp, cfg_path)
    CHUNK_SIZE = size
    CHUNK_OVERLAP = overlap
    return {'ok': True, 'chunk_size': size, 'chunk_overlap': overlap}


@app.delete('/documents/{dataset}/{doc_id}')
def delete_document(dataset: str, doc_id: str, _: None = Depends(auth)):
    ds = get_store(dataset, create=False)
    if ds is None:
        raise HTTPException(404, '知识库不存在')
    n = ds.delete_document(doc_id)
    return {'deleted': doc_id, 'vectors_removed': n}


@app.post('/compact/{dataset}')
def compact(dataset: str, _: None = Depends(auth)):
    ds = get_store(dataset, create=False)
    if ds is None:
        raise HTTPException(404, '知识库不存在')
    ds.compact()
    return {'ok': True, **ds.stats()}


@app.on_event('startup')
def warmup():
    """启动时把所有库打开一遍：触发迁移、加载索引，问题尽早暴露在日志里。"""
    for p in sorted(glob.glob(os.path.join(DATA_DIR, 'kb_*.db'))):
        name = os.path.basename(p)[3:-3]
        try:
            ds = get_store(name)
            if ds is not None:
                print('[startup] %s: %s' % (name, ds.stats()), flush=True)
        except Exception as e:
            print('[startup] %s 打开失败: %s' % (name, e), flush=True)


@app.get('/', response_class=HTMLResponse)
def ui():
    return open(os.path.join(DATA_DIR, 'index.html'), encoding='utf-8').read()


@app.get('/health')
def health():
    out = {'status': 'ok', 'datasets': len(glob.glob(os.path.join(DATA_DIR, 'kb_*.db'))),
           'dim': DIMS, 'model': EMBED_MODEL, 'schema': SCHEMA_VERSION,
           'mode_default': searchmod.MODE_DEFAULT,
           'vectors': 0, 'tombstones': 0}
    for ds in list(STORES.values()):
        out['vectors'] += ds.vec.count - ds.vec.tombstones
        out['tombstones'] += ds.vec.tombstones
    return out
