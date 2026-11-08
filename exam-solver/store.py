"""答题结果的持久化（SQLite）：一张总清单，按批次增量追加。"""
import json
import sqlite3
import time
import uuid
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
DATA.mkdir(exist_ok=True)
DB = DATA / 'answers.db'


def conn():
    c = sqlite3.connect(DB, timeout=15)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL')
    return c


def init():
    with conn() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS batches(
            id TEXT PRIMARY KEY, dataset_id TEXT, dataset_name TEXT,
            created_at INTEGER, images INTEGER, added INTEGER);
        CREATE TABLE IF NOT EXISTS items(
            id TEXT PRIMARY KEY, batch_id TEXT, image_name TEXT, image_hash TEXT,
            type TEXT, stem TEXT, options TEXT, answer TEXT, confidence REAL,
            reason TEXT, evidence TEXT, needs_review INTEGER, seq INTEGER, created_at INTEGER);
        CREATE INDEX IF NOT EXISTS idx_items_batch ON items(batch_id);
        CREATE INDEX IF NOT EXISTS idx_items_hash ON items(image_hash);
        ''')


def image_seen(h):
    """该图片是否已有"答出过答案"的成功记录；空答案/失败不算，允许自动重试。"""
    if not h:
        return False
    with conn() as c:
        return c.execute("SELECT 1 FROM items WHERE image_hash=? AND answer<>'' LIMIT 1",
                         (h,)).fetchone() is not None


def save_batch(dataset, questions, hashes):
    """questions: solver 的结果（含 image 字段）。返回 (batch_id, added, skipped)。"""
    bid = uuid.uuid4().hex[:12]
    now = int(time.time())
    added = 0
    with conn() as c:
        # 先一次性取出已有成功答案的图片指纹，再逐题写入；
        # 不能在循环里边写边查，否则同一图片的第 1 题入库后，
        # 后续题目会查到自己刚写的记录而被当成重复图片跳过。
        # 只把"有答案"的记录当缓存：空答案/失败记录不挡后续重答。
        existing = {r['image_hash'] for r in c.execute(
            "SELECT DISTINCT image_hash FROM items WHERE image_hash<>'' AND answer<>''")}
        for seq, q in enumerate(questions):
            h = hashes.get(q.get('image', ''), '')
            if h and h in existing:
                continue
            c.execute('''INSERT INTO items(id,batch_id,image_name,image_hash,type,stem,options,answer,
                         confidence,reason,evidence,needs_review,seq,created_at)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                      (uuid.uuid4().hex[:12], bid, q.get('image', ''), h, q.get('type', ''),
                       q.get('stem', ''), json.dumps(q.get('options') or {}, ensure_ascii=False),
                       q.get('answer', ''), float(q.get('confidence') or 0), q.get('reason', ''),
                       json.dumps(q.get('evidence') or [], ensure_ascii=False),
                       1 if q.get('needs_review') else 0, seq, now))
            added += 1
        c.execute('INSERT INTO batches(id,dataset_id,dataset_name,created_at,images,added) VALUES(?,?,?,?,?,?)',
                  (bid, dataset.get('id', ''), dataset.get('name', ''), now, len(hashes), added))
    return bid, added, len(hashes)


def all_items():
    with conn() as c:
        rows = c.execute('SELECT * FROM items ORDER BY created_at, seq, rowid').fetchall()
    out = []
    for i, r in enumerate(rows, 1):
        d = dict(r)
        d['options'] = json.loads(d['options'] or '{}')
        d['evidence'] = json.loads(d['evidence'] or '[]')
        d['needs_review'] = bool(d['needs_review'])
        d['index'] = i
        out.append(d)
    return out


def batches():
    with conn() as c:
        return [dict(r) for r in c.execute('SELECT * FROM batches ORDER BY created_at DESC').fetchall()]


def delete_item(item_id):
    with conn() as c:
        c.execute('DELETE FROM items WHERE id=?', (item_id,))


def delete_batch(batch_id):
    with conn() as c:
        c.execute('DELETE FROM items WHERE batch_id=?', (batch_id,))
        c.execute('DELETE FROM batches WHERE id=?', (batch_id,))


def clear():
    with conn() as c:
        c.execute('DELETE FROM items')
        c.execute('DELETE FROM batches')
