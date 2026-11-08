"""向量存储：.npy 是源（append-only），.usearch 是派生 HNSW 索引（可 mmap、可重建）。

- label = .npy 行号，label→chunk rowid 的映射存在正文库的 vec_map 表。
- 删除不做物理搬移，只从索引移除并累加墓碑；墓碑超过阈值时 compact 重写。
- 索引文件损坏/缺失时可以从 .npy 重建，不需要重新调用 embedding。
"""
import json
import os
import tempfile
import threading
import time

import numpy as np
from usearch.index import Index

HNSW_CONNECTIVITY = 32
HNSW_EXPANSION_ADD = 128
HNSW_EXPANSION_SEARCH = int(os.environ.get('HNSW_EF_SEARCH', '128'))
EXACT_MAX = int(os.environ.get('VEC_EXACT_MAX', '20000'))  # 小库直接精确扫描，保证与暴力检索等价
BASE_CAPACITY = 64
COMPACT_TOMBSTONE_RATIO = 0.3
COMPACT_MIN_TOMBSTONES = 64


def _atomic_write_json(path, data):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.meta.')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class VectorStore:
    def __init__(self, name, vec_dir, model, dim):
        self.name = name
        self.dir = vec_dir
        self.model = model
        self.dim = int(dim)
        self.npy_path = os.path.join(vec_dir, name + '.npy')
        self.idx_path = os.path.join(vec_dir, name + '.usearch')
        self.meta_path = os.path.join(vec_dir, name + '.meta.json')
        self.lock = threading.RLock()
        self.mat = None          # np.memmap (capacity, dim)
        self.index = None        # usearch Index
        self._index_mode = None  # 'view' | 'rw'
        self.count = 0
        self.capacity = 0
        self.tombstones = 0
        os.makedirs(vec_dir, exist_ok=True)
        self._load()
        self._fix_owner(self.dir, self.npy_path, self.idx_path, self.meta_path)

    def _fix_owner(self, *paths):
        """容器以 root 跑，写出的文件仍归数据目录所有者，宿主机备份不用 sudo。"""
        if os.geteuid() != 0:
            return
        try:
            st = os.stat(os.path.dirname(self.dir) or '/data')
        except Exception:
            return
        for p in paths:
            try:
                os.chown(p, st.st_uid, st.st_gid)
            except Exception:
                pass

    # ---------- 载入 ----------
    def _load(self):
        meta = {}
        if os.path.exists(self.meta_path):
            try:
                meta = json.load(open(self.meta_path, encoding='utf-8'))
            except Exception as e:
                print('[vec] meta 读取失败 %s: %s' % (self.meta_path, e), flush=True)
        if meta and (meta.get('model') != self.model or int(meta.get('dim', 0)) != self.dim):
            print('[vec] %s 模型/维度变化（%s/%s → %s/%s），向量与正文不匹配，需重建'
                  % (self.name, meta.get('model'), meta.get('dim'), self.model, self.dim), flush=True)
            raise RuntimeError('vector model mismatch for %s' % self.name)
        self.count = int(meta.get('count', 0)) if meta else 0
        self.capacity = int(meta.get('capacity', 0)) if meta else 0
        self.tombstones = int(meta.get('tombstones', 0)) if meta else 0
        if self.count and os.path.exists(self.npy_path):
            self.mat = np.load(self.npy_path, mmap_mode='r+')
            if self.mat.shape[0] < self.count or self.mat.shape[1] != self.dim:
                print('[vec] %s npy 形状异常 %s，按 0 条处理' % (self.name, self.mat.shape), flush=True)
                self.mat, self.count = None, 0
            else:
                self.capacity = int(self.mat.shape[0])
        if self.count:
            if not os.path.exists(self.idx_path):
                print('[vec] %s 索引文件缺失，将从 npy 重建' % self.name, flush=True)
                self._rebuild_index()
            else:
                try:
                    self._open_view()
                except Exception as e:
                    print('[vec] %s 索引加载失败，将从 npy 重建: %s' % (self.name, e), flush=True)
                    self._rebuild_index()

    def _ensure_capacity(self, need):
        if self.mat is not None and need <= self.capacity:
            return
        new_cap = max(BASE_CAPACITY, self.capacity or 0)
        if new_cap < need:
            new_cap = 1 << (int(need) - 1).bit_length()
        new_cap = max(new_cap, need)
        tmp = self.npy_path + '.tmp'
        new_mat = np.lib.format.open_memmap(tmp, mode='w+', dtype='float32',
                                            shape=(new_cap, self.dim))
        new_mat[:] = 0
        if self.mat is not None and self.count:
            new_mat[:self.count] = self.mat[:self.count]
            del self.mat
        os.replace(tmp, self.npy_path)
        self.mat = np.load(self.npy_path, mmap_mode='r+')
        self.capacity = new_cap

    # ---------- 索引 ----------
    def _open_view(self):
        if self.index is not None and self._index_mode == 'view':
            return
        self._drop_index()
        self.index = Index(ndim=self.dim, metric='ip', dtype='f32',
                           connectivity=HNSW_CONNECTIVITY,
                           expansion_add=HNSW_EXPANSION_ADD,
                           expansion_search=HNSW_EXPANSION_SEARCH,
                           path=self.idx_path, view=True)
        self._index_mode = 'view'

    def _open_rw(self):
        if self.index is not None and self._index_mode == 'rw':
            return
        self._drop_index()
        self.index = Index(ndim=self.dim, metric='ip', dtype='f32',
                           connectivity=HNSW_CONNECTIVITY,
                           expansion_add=HNSW_EXPANSION_ADD,
                           expansion_search=HNSW_EXPANSION_SEARCH,
                           path=self.idx_path, view=False)
        self._index_mode = 'rw'

    def _drop_index(self):
        self.index = None
        self._index_mode = None

    def _save_meta(self):
        _atomic_write_json(self.meta_path, {
            'model': self.model, 'dim': self.dim, 'count': self.count,
            'capacity': self.capacity, 'tombstones': self.tombstones,
            'updated_at': int(time.time()),
        })
        self._fix_owner(self.meta_path)

    def _save_index(self):
        self._open_rw()
        self.index.save(self.idx_path)
        self._drop_index()

    def _rebuild_index(self):
        """只用 npy 里的有效向量重建索引（不调 embedding）。"""
        with self.lock:
            if not self.count:
                return
            labels = np.arange(self.count, dtype=np.int64)
            vecs = np.asarray(self.mat[:self.count], dtype=np.float32)
            self._drop_index()
            self.index = Index(ndim=self.dim, metric='ip', dtype='f32',
                               connectivity=HNSW_CONNECTIVITY,
                               expansion_add=HNSW_EXPANSION_ADD,
                               expansion_search=HNSW_EXPANSION_SEARCH)
            self.index.add(labels, vecs, copy=False)
            self.index.save(self.idx_path)
            self._drop_index()

    # ---------- 增删 ----------
    def append(self, vectors):
        """vectors: (n, dim) float32，已归一化。返回新 label 列表。"""
        if vectors is None or len(vectors) == 0:
            return []
        vecs = np.asarray(vectors, dtype=np.float32, order='C')
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError('向量维度不符：%s != %d' % (vecs.shape, self.dim))
        with self.lock:
            need = self.count + len(vecs)
            self._ensure_capacity(need)
            self.mat[self.count:need] = vecs
            self.mat.flush()
            labels = list(range(self.count, need))
            self._open_rw()
            self.index.add(np.asarray(labels, dtype=np.int64), vecs, copy=False)
            self.index.save(self.idx_path)
            self._drop_index()
            self.count = need
            self._save_meta()
            self._fix_owner(self.npy_path, self.idx_path, self.meta_path)
        return labels

    def remove(self, labels):
        if not labels:
            return
        with self.lock:
            self._open_rw()
            try:
                self.index.remove(np.asarray(labels, dtype=np.int64))
            except Exception as e:
                print('[vec] 索引移除失败（可由 compact 修复）: %s' % e, flush=True)
            self.index.save(self.idx_path)
            self._drop_index()
            self.tombstones += len(labels)
            self._save_meta()
            self._fix_owner(self.idx_path, self.meta_path)

    def sync_tombstones(self, n):
        with self.lock:
            self.tombstones = max(0, int(n))
            self._save_meta()

    def should_compact(self):
        return self.tombstones >= max(COMPACT_MIN_TOMBSTONES,
                                      int(self.count * COMPACT_TOMBSTONE_RATIO))

    # ---------- 读 ----------
    def search(self, qvec, k):
        if not self.count or not os.path.exists(self.idx_path):
            return []
        with self.lock:
            self._open_view()
            # 小库用 exact（线性扫描，几百条时 <1ms）：近似索引在近重复向量上会有边界抖动，
            # 这个量级下精确扫描既更准也没有性能代价；超过阈值才走 HNSW 近似。
            exact = self.count <= EXACT_MAX
            try:
                m = self.index.search(np.asarray(qvec, dtype=np.float32), count=int(k), exact=exact)
            except Exception as e:
                print('[vec] 检索失败: %s' % e, flush=True)
                return []
            keys = np.asarray(m.keys).astype(np.int64).tolist()
            dists = np.asarray(m.distances).astype(float).tolist()
        # usearch 的 ip 度量返回距离 = 1 - 内积；换回余弦相似度给上层用
        return [(int(k_), float(1.0 - d)) for k_, d in zip(keys, dists)]

    def get_vectors(self, labels):
        with self.lock:
            return np.asarray([self.mat[l] for l in labels], dtype=np.float32)

    def contains(self, labels):
        if not labels or not self.count:
            return [False] * len(labels)
        with self.lock:
            self._open_view()
            try:
                return [bool(x) for x in self.index.contains(np.asarray(labels, dtype=np.int64))]
            except Exception:
                return [True] * len(labels)

    # ---------- 压缩 ----------
    def compact(self, live_rowids):
        """live_rowids 按旧 label 升序：[(old_label, rowid)]；重写后 label 变成 0..n-1。"""
        with self.lock:
            if not live_rowids:
                self.count = 0
                self.tombstones = 0
                self._drop_index()
                self.mat = None
                self.capacity = 0
                for p in (self.npy_path, self.idx_path):
                    if os.path.exists(p):
                        os.unlink(p)
                self._save_meta()
                return []
            vecs = np.asarray([self.mat[l] for l, _ in live_rowids], dtype=np.float32)
            new_cap = max(BASE_CAPACITY, 1 << (len(vecs) - 1).bit_length())
            tmp = self.npy_path + '.tmp'
            new_mat = np.lib.format.open_memmap(tmp, mode='w+', dtype='float32',
                                                shape=(new_cap, self.dim))
            new_mat[:] = 0
            new_mat[:len(vecs)] = vecs
            new_mat.flush()
            del new_mat
            del self.mat
            os.replace(tmp, self.npy_path)
            self.mat = np.load(self.npy_path, mmap_mode='r+')
            self.capacity = new_cap
            self.count = len(vecs)
            self.tombstones = 0
            self._drop_index()
            self.index = Index(ndim=self.dim, metric='ip', dtype='f32',
                               connectivity=HNSW_CONNECTIVITY,
                               expansion_add=HNSW_EXPANSION_ADD,
                               expansion_search=HNSW_EXPANSION_SEARCH)
            self.index.add(np.arange(self.count, dtype=np.int64), vecs, copy=False)
            self.index.save(self.idx_path)
            self._drop_index()
            self._save_meta()
            self._fix_owner(self.npy_path, self.idx_path, self.meta_path)
            return [(rid, label) for label, (_, rid) in enumerate(live_rowids)]
