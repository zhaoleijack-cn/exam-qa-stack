import os, io, json, re, glob, sqlite3, time, threading, hashlib
from typing import List, Optional
import numpy as np
import requests
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends
from pydantic import BaseModel

DATA_DIR = os.environ.get('KB_DATA', '/data')
CFG = json.load(open(os.path.join(DATA_DIR, 'config.json')))
API_KEY = CFG['api_key']
EMBED_URL = CFG.get('embed_url', 'http://fastgpt-aiproxy:3000/v1/embeddings')
EMBED_TOKEN = CFG.get('embed_token', '')
EMBED_MODEL = CFG.get('embed_model', 'text-embedding-v4')
DIMS = int(CFG.get('embed_dims', 1536))
CHUNK_SIZE = int(CFG.get('chunk_size', 800))
CHUNK_OVERLAP = int(CFG.get('chunk_overlap', 100))
DS_RE = re.compile(r'^[a-z0-9_]{1,32}$')

app = FastAPI(title='Knowledge Base Service', version='2.0',
              description='多知识库服务：每个知识库独立存储、完全隔离。检索接口供大模型工具调用，dataset 必填。')

db_lock = threading.Lock()
STORES = {}
STORE_LOCK = threading.Lock()

def auth(x_api_key: str = Header(default='')):
    if x_api_key != API_KEY:
        raise HTTPException(401, 'invalid api key')

def check_ds(dataset: str) -> str:
    if not dataset or not DS_RE.match(dataset):
        raise HTTPException(400, 'dataset 必填且只能含小写字母/数字/下划线（1-32位）')
    return dataset

class Store:
    def __init__(self, name: str):
        self.name = name
        self.path = os.path.join(DATA_DIR, 'kb_%s.db' % name)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS documents(
              id TEXT PRIMARY KEY, title TEXT, dataset TEXT, source TEXT, created_at REAL);
            CREATE TABLE IF NOT EXISTS chunks(
              id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INTEGER, text TEXT, embedding BLOB);
            CREATE INDEX IF NOT EXISTS idx_c_doc ON chunks(doc_id);
        ''')
        self.mlock = threading.Lock()
        self._load()

    def _load(self):
        rows = self.db.execute('SELECT id, embedding FROM chunks').fetchall()
        self.ids = [r[0] for r in rows]
        self.mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows \
            else np.zeros((0, DIMS), dtype=np.float32)

    def reload(self):
        with self.mlock: self._load()

    def add(self, doc_id, title, chunks, vecs):
        with self.mlock:
            self.db.execute('INSERT INTO documents(id,title,dataset,source,created_at) VALUES(?,?,?,?,?) '
                            'ON CONFLICT(id) DO UPDATE SET title=excluded.title',
                            (doc_id, title, self.name, title, time.time()))
            self.db.execute('DELETE FROM chunks WHERE doc_id=?', (doc_id,))
            for ci, (ch, v) in enumerate(zip(chunks, vecs)):
                cid = hashlib.sha1((doc_id + '|' + str(ci) + '|' + ch[:64]).encode()).hexdigest()[:16] + str(ci)
                self.db.execute('INSERT OR REPLACE INTO chunks(id,doc_id,chunk_index,text,embedding) VALUES(?,?,?,?,?)',
                                (cid, doc_id, ci, ch, v.tobytes()))
            self.db.commit()
        self.reload()

    def search(self, qvec, top_k, min_score):
        with self.mlock:
            if len(self.ids) == 0: return []
            scores = self.mat @ qvec
        order = np.argsort(-scores)
        out = []
        for i in order:
            if len(out) >= top_k or scores[i] < min_score: break
            row = self.db.execute('SELECT c.title, ch.chunk_index, ch.text, ch.doc_id '
                                  'FROM chunks ch JOIN documents c ON c.id = ch.doc_id WHERE ch.id = ?',
                                  (self.ids[i],)).fetchone()
            if row: out.append({'title': row[0], 'dataset': self.name, 'chunk_index': row[1],
                                'text': row[2], 'doc_id': row[3], 'score': round(float(scores[i]), 4)})
        return out

    def delete_doc(self, doc_id):
        with self.mlock:
            self.db.execute('DELETE FROM chunks WHERE doc_id=?', (doc_id,))
            self.db.execute('DELETE FROM documents WHERE id=?', (doc_id,))
            self.db.commit()
        self.reload()

    def stats(self):
        d = self.db.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
        c = self.db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        return {'dataset': self.name, 'documents': d, 'chunks': c}

def get_store(dataset: str) -> Store:
    check_ds(dataset)
    with STORE_LOCK:
        if dataset not in STORES:
            STORES[dataset] = Store(dataset)
        return STORES[dataset]

def embed(texts: List[str]) -> np.ndarray:
    out: List[Optional[np.ndarray]] = [None] * len(texts)
    B = 32
    todo = []
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        try:
            r = requests.post(EMBED_URL, timeout=90,
                headers={'Authorization': 'Bearer ' + EMBED_TOKEN},
                json={'model': EMBED_MODEL, 'input': batch, 'dimensions': DIMS})
            if r.status_code != 200: raise RuntimeError(r.text[:150])
            for item in r.json()['data']:
                out[i + item['index']] = np.array(item['embedding'], dtype=np.float32)
        except Exception as e:
            print('[embed] batch %d failed (%s), fallback single' % (i, str(e)[:80]))
            todo.extend(range(i, min(i + B, len(texts))))
    for i in todo:
        try:
            r = requests.post(EMBED_URL, timeout=90,
                headers={'Authorization': 'Bearer ' + EMBED_TOKEN},
                json={'model': EMBED_MODEL, 'input': [texts[i]], 'dimensions': DIMS})
            if r.status_code != 200: raise RuntimeError(r.text[:150])
            out[i] = np.array(r.json()['data'][0]['embedding'], dtype=np.float32)
        except Exception as e:
            print('[embed] single %d failed: %s' % (i, str(e)[:100]))
    failed = [i for i, v in enumerate(out) if v is None]
    if failed:
        raise HTTPException(502, 'embedding failed for %d texts' % len(failed))
    return np.vstack(out)

def chunk_text(text: str) -> List[str]:
    text = text.strip()
    if not text: return []
    if len(text) <= CHUNK_SIZE: return [text]
    paras = [p.strip() for p in text.replace('\r\n', '\n').split('\n') if p.strip()]
    chunks, cur = [], ''
    for p in paras:
        if len(cur) + len(p) + 1 <= CHUNK_SIZE:
            cur = (cur + '\n' + p).strip(); continue
        if cur: chunks.append(cur)
        if len(p) <= CHUNK_SIZE: cur = p; continue
        start = 0
        while start < len(p):
            chunks.append(p[start:start + CHUNK_SIZE]); start += CHUNK_SIZE - CHUNK_OVERLAP
        cur = ''
    if cur: chunks.append(cur)
    return chunks

def parse_pdf(b): 
    from pypdf import PdfReader
    return '\n'.join((p.extract_text() or '') for p in PdfReader(io.BytesIO(b)).pages)
def parse_docx(b):
    import docx
    return '\n'.join(p.text for p in docx.Document(io.BytesIO(b)).paragraphs if p.text.strip())
def parse_txt(b):
    for enc in ('utf-8', 'gb18030'):
        try: return b.decode(enc)
        except Exception: pass
    return b.decode('utf-8', 'replace')
def parse_file(name, b):
    low = name.lower()
    if low.endswith('.pdf'): return parse_pdf(b)
    if low.endswith('.docx'): return parse_docx(b)
    return parse_txt(b)

def make_doc_id(dataset, title):
    return hashlib.sha1((dataset + '|' + title).encode()).hexdigest()[:16]

class SearchReq(BaseModel):
    query: str
    dataset: str
    top_k: int = 8
    min_score: float = 0.0

@app.post('/search')
def search(req: SearchReq, _: None = Depends(auth)):
    if not req.query.strip(): raise HTTPException(400, 'empty query')
    qv = embed([req.query.strip()])[0]
    st = get_store(req.dataset)
    return {'results': st.search(qv, req.top_k, req.min_score), 'total': req.top_k}

class BulkItem(BaseModel):
    title: str
    text: str

class BulkReq(BaseModel):
    dataset: str
    items: List[BulkItem]

@app.post('/ingest/bulk')
def ingest_bulk(req: BulkReq, _: None = Depends(auth)):
    get_store(req.dataset)
    titles = []
    seen = set()
    for it in req.items:
        if it.title not in seen:
            seen.add(it.title); titles.append(it.title)
    out = []
    for title in titles:
        texts = []
        for it in req.items:
            if it.title == title: texts.extend(chunk_text(it.text))
        if not texts: continue
        doc_id = make_doc_id(req.dataset, title)
        get_store(req.dataset).add(doc_id, title, texts, embed(texts))
        out.append({'doc_id': doc_id, 'title': title, 'chunks': len(texts)})
    return {'imported': out, 'total_chunks': sum(o['chunks'] for o in out)}

@app.post('/ingest/file')
async def ingest_file(file: UploadFile = File(...), dataset: str = Form(...),
                      title: Optional[str] = Form(None), _: None = Depends(auth)):
    get_store(dataset)
    b = await file.read()
    if len(b) > 30 * 1024 * 1024: raise HTTPException(400, 'file too large (max 30MB)')
    title = title or file.filename
    try: text = parse_file(file.filename, b)
    except Exception as e: raise HTTPException(400, 'parse failed: ' + str(e)[:150])
    chunks = chunk_text(text)
    if not chunks: raise HTTPException(400, 'no content extracted')
    doc_id = make_doc_id(dataset, title)
    get_store(dataset).add(doc_id, title, chunks, embed(chunks))
    return {'doc_id': doc_id, 'title': title, 'chunks': len(chunks), 'chars': len(text)}

class TextReq(BaseModel):
    dataset: str
    title: str
    text: str

@app.post('/ingest/text')
def ingest_text(req: TextReq, _: None = Depends(auth)):
    chunks = chunk_text(req.text)
    if not chunks: raise HTTPException(400, 'empty text')
    doc_id = make_doc_id(req.dataset, req.title)
    get_store(req.dataset).add(doc_id, req.title, chunks, embed(chunks))
    return {'doc_id': doc_id, 'title': req.title, 'chunks': len(chunks)}

@app.get('/datasets')
def datasets(_: None = Depends(auth)):
    out = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, 'kb_*.db'))):
        name = os.path.basename(p)[3:-3]
        try:
            out.append(get_store(name).stats())
        except Exception: pass
    return {'datasets': out}

@app.get('/documents/{dataset}')
def documents(dataset: str, _: None = Depends(auth)):
    st = get_store(dataset)
    return {'documents': [{'doc_id': r[0], 'title': r[1], 'created_at': r[2], 'chunks': r[3]}
            for r in st.db.execute('SELECT d.id, d.title, d.created_at, COUNT(ch.id) '
                                   'FROM documents d LEFT JOIN chunks ch ON ch.doc_id=d.id '
                                   'GROUP BY d.id ORDER BY d.created_at DESC')]}

@app.delete('/documents/{dataset}/{doc_id}')
def delete_document(dataset: str, doc_id: str, _: None = Depends(auth)):
    get_store(dataset).delete_doc(doc_id)
    return {'deleted': doc_id}

@app.get('/', response_class=HTMLResponse)
def ui():
    return open(os.path.join(DATA_DIR, 'index.html'), encoding='utf-8').read()

@app.get('/health')
def health():
    return {'status': 'ok', 'datasets': len(glob.glob(os.path.join(DATA_DIR, 'kb_*.db'))), 'dim': DIMS, 'model': EMBED_MODEL}
