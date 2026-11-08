"""答题结果的持久化（SQLite）：一张总清单，按批次增量追加。"""
import contextlib
import json
import sqlite3
import time
import uuid
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
DATA.mkdir(exist_ok=True)
DB = DATA / 'answers.db'


@contextlib.contextmanager
def conn():
    """每次用完显式关闭，避免长时间运行后连接/句柄累积。"""
    c = sqlite3.connect(DB, timeout=15)
    c.row_factory = sqlite3.Row
    try:
        c.execute('PRAGMA journal_mode=WAL')
        yield c
        c.commit()
    finally:
        c.close()


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


def image_seen(h, dataset_id=''):
    """该图片在**同一个知识库**里是否已有答出过答案的记录。

    v1.0.3: 去重按库隔离——同一张图换库重答不会再被静默跳过；空答案/失败不算，允许重试。
    """
    if not h:
        return False
    with conn() as c:
        if dataset_id:
            row = c.execute("SELECT 1 FROM items i JOIN batches b ON b.id=i.batch_id "
                            "WHERE i.image_hash=? AND i.answer<>'' AND b.dataset_id=? LIMIT 1",
                            (h, dataset_id)).fetchone()
        else:
            row = c.execute("SELECT 1 FROM items WHERE image_hash=? AND answer<>'' LIMIT 1",
                            (h,)).fetchone()
        return row is not None


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
            "SELECT DISTINCT i.image_hash FROM items i JOIN batches b ON b.id=i.batch_id "
            "WHERE i.image_hash<>'' AND i.answer<>'' AND b.dataset_id=?",
            (dataset.get('id', ''),))}
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
        if added:      # 空批次不落库，避免删空后堆一堆 0 题记录
            c.execute('INSERT INTO batches(id,dataset_id,dataset_name,created_at,images,added) '
                      'VALUES(?,?,?,?,?,?)',
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


def stats():
    """SQL 聚合，替代过去"全表读出来再算"的轮询开销。"""
    with conn() as c:
        row = c.execute('SELECT COUNT(*), COALESCE(MAX(created_at),0), '
                        'COALESCE(SUM(needs_review),0), COALESCE(AVG(confidence),0) '
                        'FROM items').fetchone()
        return {'total': row[0], 'latest': row[1], 'needs_review': row[2],
                'avg_confidence': round(row[3], 2)}


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
