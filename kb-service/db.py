"""正文库：文档、分块、FTS5 全文索引、向量映射（与向量文件解耦）。

设计要点
--------
- 正文与元数据只存在 SQLite（一个库一个 db 文件），备份=拷文件。
- 向量不进这张表：由 vec.py 写到独立 .npy / .usearch 文件，本表只存 label→rowid 映射。
- 全文索引用 FTS5 external content + 触发器自动同步，中文用 trigram 分词。
"""
import hashlib
import os
import re
import sqlite3
import threading
import time

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS documents(
  id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT, sha256 TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS chunks(
  rowid INTEGER PRIMARY KEY,
  id TEXT UNIQUE NOT NULL,
  doc_id TEXT NOT NULL,
  chunk_index INTEGER,
  text TEXT NOT NULL,
  created_at REAL);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, content='chunks', content_rowid='rowid', tokenize='trigram');
CREATE TABLE IF NOT EXISTS vec_map(
  label INTEGER PRIMARY KEY,
  rowid INTEGER UNIQUE NOT NULL,
  model TEXT);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
  INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""


def chunk_id(doc_id, index, text):
    return hashlib.sha1(('%s|%d|%s' % (doc_id, index, text[:64])).encode()).hexdigest()[:16]


def is_legacy_db(path):
    """旧方案把 embedding 直接塞在 chunks 表里；有这一列就是要迁移。"""
    if not os.path.exists(path):
        return False
    try:
        c = sqlite3.connect(path, timeout=5)
        cols = [r[1] for r in c.execute('PRAGMA table_info(chunks)')]
        c.close()
        return 'embedding' in cols
    except Exception:
        return False


def schema_version(path):
    if not os.path.exists(path):
        return 0
    try:
        c = sqlite3.connect(path, timeout=5)
        try:
            row = c.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        except sqlite3.OperationalError:
            row = None
        c.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def create_new(path, model, dim):
    """新建一个空库（新 schema）。"""
    c = sqlite3.connect(path)
    try:
        c.executescript(SCHEMA)
        c.executemany('INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)', [
            ('schema', str(SCHEMA_VERSION)),
            ('model', model),
            ('dim', str(dim)),
            ('created_at', str(time.time())),
        ])
        c.commit()
    finally:
        c.close()


class ContentDB:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=15)
        self.db.row_factory = sqlite3.Row
        with self.lock:
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=NORMAL')

    # ---------- 元信息 ----------
    def meta(self, key, default=None):
        with self.lock:
            try:
                row = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
            except sqlite3.OperationalError:
                return default
        return row[0] if row else default

    def set_meta(self, key, value):
        with self.lock, self.db:
            self.db.execute('INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)', (key, str(value)))

    # ---------- 写入 ----------
    def write_document(self, title, doc_id, source, sha256, chunks):
        """整篇替换：返回 (rowids, old_labels)。旧向量 label 由调用方从索引里移除。"""
        now = time.time()
        with self.lock, self.db:
            old_labels = [r[0] for r in self.db.execute(
                'SELECT v.label FROM vec_map v JOIN chunks c ON c.rowid=v.rowid WHERE c.doc_id=?',
                (doc_id,))]
            self.db.execute('DELETE FROM vec_map WHERE rowid IN '
                            '(SELECT rowid FROM chunks WHERE doc_id=?)', (doc_id,))
            self.db.execute('DELETE FROM chunks WHERE doc_id=?', (doc_id,))
            self.db.execute(
                'INSERT INTO documents(id, title, source, sha256, created_at) VALUES(?,?,?,?,?) '
                'ON CONFLICT(id) DO UPDATE SET title=excluded.title, source=excluded.source, '
                'sha256=excluded.sha256, created_at=excluded.created_at',
                (doc_id, title, source, sha256, now))
            rowids = []
            for i, text in enumerate(chunks):
                cur = self.db.execute(
                    'INSERT INTO chunks(id, doc_id, chunk_index, text, created_at) VALUES(?,?,?,?,?)',
                    (chunk_id(doc_id, i, text), doc_id, i, text, now))
                rowids.append(cur.lastrowid)
        return rowids, old_labels

    def map_vectors(self, pairs, model):
        """pairs: [(rowid, label)]"""
        with self.lock, self.db:
            self.db.executemany(
                'INSERT OR REPLACE INTO vec_map(label, rowid, model) VALUES(?,?,?)',
                [(label, rowid, model) for rowid, label in pairs])

    def delete_document(self, doc_id):
        with self.lock, self.db:
            labels = [r[0] for r in self.db.execute(
                'SELECT v.label FROM vec_map v JOIN chunks c ON c.rowid=v.rowid WHERE c.doc_id=?',
                (doc_id,))]
            self.db.execute('DELETE FROM vec_map WHERE rowid IN '
                            '(SELECT rowid FROM chunks WHERE doc_id=?)', (doc_id,))
            self.db.execute('DELETE FROM chunks WHERE doc_id=?', (doc_id,))
            self.db.execute('DELETE FROM documents WHERE id=?', (doc_id,))
        return labels

    def clear(self):
        with self.lock, self.db:
            labels = [r[0] for r in self.db.execute('SELECT label FROM vec_map')]
            self.db.execute('DELETE FROM vec_map')
            self.db.execute('DELETE FROM chunks')
            self.db.execute('DELETE FROM documents')
        return labels

    # ---------- 映射 ----------
    def vec_pairs(self):
        with self.lock:
            return [(r[0], r[1]) for r in
                    self.db.execute('SELECT label, rowid FROM vec_map ORDER BY label')]

    def rowids_for_labels(self, labels):
        if not labels:
            return {}
        qs = ','.join('?' * len(labels))
        with self.lock:
            rows = self.db.execute('SELECT label, rowid FROM vec_map WHERE label IN (%s)' % qs,
                                   list(labels)).fetchall()
        return {r[0]: r[1] for r in rows}

    def drop_vec_map(self, labels):
        if not labels:
            return
        qs = ','.join('?' * len(labels))
        with self.lock, self.db:
            self.db.execute('DELETE FROM vec_map WHERE label IN (%s)' % qs, list(labels))

    def replace_vec_map(self, labels_rowids, model):
        """按顺序重编号：labels_rowids 是 [(rowid, label)] 的新映射。"""
        with self.lock, self.db:
            self.db.execute('DELETE FROM vec_map')
            self.db.executemany(
                'INSERT INTO vec_map(label, rowid, model) VALUES(?,?,?)',
                [(label, rowid, model) for rowid, label in labels_rowids])

    # ---------- 读 ----------
    def documents(self):
        with self.lock:
            rows = self.db.execute(
                'SELECT d.id, d.title, d.created_at, COUNT(c.rowid) AS chunks '
                'FROM documents d LEFT JOIN chunks c ON c.doc_id=d.id '
                'GROUP BY d.id ORDER BY d.created_at DESC').fetchall()
        return [{'doc_id': r[0], 'title': r[1], 'created_at': r[2], 'chunks': r[3]} for r in rows]

    def stats(self):
        with self.lock:
            d = self.db.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
            c = self.db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
            v = self.db.execute('SELECT COUNT(*) FROM vec_map').fetchone()[0]
        return {'documents': d, 'chunks': c, 'vectors': v}

    def chunk_rows(self, rowids):
        if not rowids:
            return {}
        qs = ','.join('?' * len(rowids))
        with self.lock:
            rows = self.db.execute(
                'SELECT c.rowid, c.text, c.doc_id, c.chunk_index, d.title '
                'FROM chunks c JOIN documents d ON d.id=c.doc_id '
                'WHERE c.rowid IN (%s)' % qs, list(rowids)).fetchall()
        return {r[0]: {'text': r[1], 'doc_id': r[2], 'chunk_index': r[3], 'title': r[4]}
                for r in rows}

    @staticmethod
    def _fts_terms(query, loose=False):
        """把整句拆成片段。

        strict：每段整段加引号，= 精确子串（原文照抄的查询最准）；
        loose：长片段再补 3 字滑窗，= 容忍用户改写（字面不连续也能召回）。
        """
        q = re.sub(r'[\x00]', ' ', query or '').strip()
        seps = re.compile(r'[\s，。、；：？！,.;:?!()（）\[\]【】《》“”‘’·]+')
        segments = [x for x in seps.split(q) if x]
        terms, seen = [], set()

        def add(t):
            if len(t) >= 3 and t not in seen:
                seen.add(t)
                terms.append('"%s"' % t.replace('"', ' '))

        for seg in segments:
            add(seg)
            if not loose or len(seg) <= 8:
                continue
            grams = [seg[i:i + 3] for i in range(0, len(seg) - 2)]
            if len(grams) > 10:
                grams = grams[::max(1, len(grams) // 10)][:10]
            for g in grams:
                add(g)
        joined = ''.join(segments)
        if len(joined) >= 3 and not terms:
            # 全部是 2 字以内的短词（如「灾备 切换」）：trigram 匹配不到单段，
            # 退化为去分隔符整句的 3 字滑窗，至少能召回字面连续的文本。
            grams = [joined[i:i + 3] for i in range(0, len(joined) - 2)]
            if len(grams) > 10:
                grams = grams[::max(1, len(grams) // 10)][:10]
            for g in grams:
                add(g)
        return terms[:32]

    @classmethod
    def _fts_expr(cls, query, loose=False):
        terms = cls._fts_terms(query, loose=loose)
        return ' OR '.join(terms) if terms else None

    def _fts_query(self, expr, limit):
        with self.lock:
            return self.db.execute(
                'SELECT c.rowid, bm25(chunks_fts) AS bm '
                'FROM chunks_fts JOIN chunks c ON c.rowid=chunks_fts.rowid '
                'WHERE chunks_fts MATCH ? ORDER BY bm LIMIT ?',
                (expr, int(limit))).fetchall()

    def search_fts(self, query, limit):
        """先精确匹配；一个字都没命中再用宽松（3 字滑窗）召回。"""
        strict = self._fts_expr(query)
        if not strict:
            return []       # 少于 3 个字符：trigram 无法匹配，只走向量路
        try:
            rows = self._fts_query(strict, limit)
        except sqlite3.OperationalError as e:
            print('[fts] 精确查询失败 %r: %s' % (query, e), flush=True)
            rows = []
        if not rows:
            loose = self._fts_expr(query, loose=True)
            if loose and loose != strict:
                try:
                    rows = self._fts_query(loose, limit)
                except sqlite3.OperationalError as e:
                    print('[fts] 宽松查询失败 %r: %s' % (query, e), flush=True)
                    rows = []
        return [(r[0], float(r[1])) for r in rows]

    def close(self):
        with self.lock:
            self.db.close()
