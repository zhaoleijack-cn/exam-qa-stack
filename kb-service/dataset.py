"""Dataset 编排：正文库 + 向量库 + 迁移 + 一致性修复。"""
import hashlib
import os
import shutil
import sqlite3
import threading
import time

import glob

import numpy as np

import db as dbmod
import search as searchmod
from db import ContentDB, chunk_id, create_new, is_legacy_db, schema_version
from vec import VectorStore


def make_doc_id(dataset, title):
    return hashlib.sha1((dataset + '|' + title).encode()).hexdigest()[:16]


# ---------- 迁移：把 chunks.embedding 搬到独立向量文件 ----------
def migrate_legacy(path, vec_dir, name, cfg):
    model = cfg.get('embed_model', 'text-embedding-3')
    dim = int(cfg.get('embed_dims', 1536))
    stamp = time.strftime('%Y%m%d-%H%M%S')
    backup = path + '.pre-vecsplit-' + stamp
    shutil.copy2(path, backup)
    print('[migrate] %s: 备份 %s' % (name, os.path.basename(backup)), flush=True)

    old = sqlite3.connect(path)
    old.row_factory = sqlite3.Row
    try:
        docs = old.execute('SELECT id, title, source, created_at FROM documents').fetchall()
        chunks = old.execute('SELECT rowid, id, doc_id, chunk_index, text, embedding '
                             'FROM chunks ORDER BY rowid').fetchall()
    finally:
        old.close()

    tmp_path = path + '.new'
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
    create_new(tmp_path, model, dim)

    for p in (os.path.join(vec_dir, name + '.npy'),
              os.path.join(vec_dir, name + '.usearch'),
              os.path.join(vec_dir, name + '.meta.json')):
        if os.path.exists(p):
            os.unlink(p)

    vec = VectorStore(name, vec_dir, model, dim)
    pairs, vectors, skipped = [], [], 0
    newc = sqlite3.connect(tmp_path)
    try:
        newc.executemany(
            'INSERT OR REPLACE INTO documents(id,title,source,sha256,created_at) VALUES(?,?,?,?,?)',
            [(d['id'], d['title'], d['source'], '', d['created_at'] or time.time()) for d in docs])
        for ch in chunks:
            new_rowid = newc.execute(
                'INSERT INTO chunks(id,doc_id,chunk_index,text,created_at) VALUES(?,?,?,?,?)',
                (ch['id'], ch['doc_id'], ch['chunk_index'], ch['text'], time.time())).lastrowid
            blob = ch['embedding']
            try:
                v = np.frombuffer(blob, dtype=np.float32) if blob else None
            except Exception:
                v = None
            if v is None or len(v) != dim:
                skipped += 1
                continue
            n = np.linalg.norm(v)
            if n:
                v = v / n
            pairs.append((new_rowid, len(vectors)))
            vectors.append(v.astype(np.float32))
        newc.commit()
    finally:
        newc.close()

    if vectors:
        labels = vec.append(np.vstack(vectors))
        c = sqlite3.connect(tmp_path)
        try:
            c.executemany('INSERT INTO vec_map(label,rowid,model) VALUES(?,?,?)',
                          [(lab, rid, model) for lab, (rid, _) in zip(labels, pairs)])
            c.commit()
        finally:
            c.close()

    os.replace(tmp_path, path)
    print('[migrate] %s: %d 文档 / %d 块 / %d 向量（%d 块无向量，仅全文可检索）'
          % (name, len(docs), len(chunks), len(vectors), skipped), flush=True)
    return {'documents': len(docs), 'chunks': len(chunks), 'vectors': len(vectors)}


class Dataset:
    def __init__(self, name, data_dir, vec_dir, cfg):
        self.name = name
        self.cfg = cfg
        self.model = cfg.get('embed_model', 'text-embedding-3')
        self.dim = int(cfg.get('embed_dims', 1536))
        self.path = os.path.join(data_dir, 'kb_%s.db' % name)
        self.lock = threading.RLock()
        if is_legacy_db(self.path):
            migrate_legacy(self.path, vec_dir, name, cfg)
        elif not os.path.exists(self.path) or schema_version(self.path) == 0:
            create_new(self.path, self.model, self.dim)
        self.db = ContentDB(self.path)
        self.db.name = name
        self.vec = VectorStore(name, vec_dir, self.model, self.dim)
        self._reconcile()
        self._fix_owner()

    def _fix_owner(self):
        if os.geteuid() != 0:
            return
        try:
            st = os.stat(os.path.dirname(self.path))
        except Exception:
            return
        import glob as _glob
        for p in ([self.path, self.path + '-wal', self.path + '-shm']
                  + _glob.glob(self.path + '.pre-vecsplit-*')):
            try:
                os.chown(p, st.st_uid, st.st_gid)
            except Exception:
                pass

    def _reconcile(self):
        """启动自检：映射里有、索引里没有的 label 丢弃（该块退化为仅全文可检索）。"""
        pairs = self.db.vec_pairs()
        if not pairs:
            if self.vec.count:
                self.vec.sync_tombstones(self.vec.count)
            return
        labels = [l for l, _ in pairs]
        valid = self.vec.contains(labels)
        bad = [l for l, ok in zip(labels, valid) if not ok]
        if bad:
            print('[ds %s] %d 个向量映射在索引中不存在，已降级为仅全文可检索'
                  % (self.name, len(bad)), flush=True)
            self.db.drop_vec_map(bad)
        live = [l for l, ok in zip(labels, valid) if ok]
        self.vec.sync_tombstones(max(0, self.vec.count - len(live)))

    # ---------- 写 ----------
    def ingest(self, title, chunks, source='', sha256=''):
        chunks = [c for c in chunks if c.strip()]
        if not chunks:
            return {'doc_id': '', 'chunks': 0}
        with self.lock:
            vecs = searchmod.embed(chunks, self.cfg)
            doc_id = make_doc_id(self.name, title)
            rowids, old_labels = self.db.write_document(title, doc_id, source, sha256, chunks)
            if old_labels:
                self.vec.remove(old_labels)
            labels = self.vec.append(vecs)
            self.db.map_vectors(list(zip(rowids, labels)), self.model)
            self._maybe_compact()
            return {'doc_id': doc_id, 'title': title, 'chunks': len(chunks), 'vectors': len(labels)}

    def delete_document(self, doc_id):
        with self.lock:
            labels = self.db.delete_document(doc_id)
            if labels:
                self.vec.remove(labels)
            self._maybe_compact()
            return len(labels)

    def clear(self):
        with self.lock:
            labels = self.db.clear()
            if labels:
                self.vec.remove(labels)
            self.vec.compact([])
            self.vec.sync_tombstones(0)

    def _maybe_compact(self):
        if self.vec.should_compact():
            self.compact()

    def compact(self):
        with self.lock:
            pairs = self.db.vec_pairs()
            new_pairs = self.vec.compact(pairs)
            self.db.replace_vec_map(new_pairs, self.model)
            print('[ds %s] 向量压缩完成：%d 条' % (self.name, len(new_pairs)), flush=True)

    # ---------- 读 ----------
    def search(self, query, top_k=8, min_score=0.0, mode='hybrid', embedding_weight=None):
        w = searchmod.DEFAULT_EMBED_WEIGHT if embedding_weight is None else embedding_weight
        return searchmod.search(self.db, self.vec, query, top_k, min_score, mode, w, self.cfg)

    def stats(self):
        s = self.db.stats()
        s.update({'dataset': self.name, 'model': self.model, 'dim': self.dim,
                  'indexed': self.vec.count, 'tombstones': self.vec.tombstones})
        return s

    def documents(self):
        return self.db.documents()

    def info(self):
        return {'schema': schema_version(self.path), 'model': self.model, 'dim': self.dim,
                'vectors': self.vec.count, 'tombstones': self.vec.tombstones,
                'path': os.path.basename(self.path)}
