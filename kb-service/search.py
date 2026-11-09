"""检索层：embedding 查询 + 向量召回 + 全文召回 + RRF 融合。

RRF 只用名次，天然免疫余弦分与 BM25 分量纲不同的问题。
"""
import os
import re
import numpy as np
import requests

RRF_K = int(os.environ.get('RRF_K', '60'))
DEFAULT_EMBED_WEIGHT = float(os.environ.get('HYBRID_EMBED_WEIGHT', '0.5'))
VEC_RECALL = int(os.environ.get('VEC_RECALL', '80'))
FTS_RECALL = int(os.environ.get('FTS_RECALL', '60'))
MODE_DEFAULT = os.environ.get('SEARCH_MODE_DEFAULT', 'hybrid')
MODES = ('embedding', 'fulltext', 'hybrid')
EMBED_BATCH = int(os.environ.get('EMBED_BATCH', '64'))   # 单次请求的 input 上限，防大文档一批被上游拒



def embed(texts, cfg):
    """返回已 L2 归一化的向量矩阵；行数与输入一致。分批请求，避免单批过大被上游拒绝。"""
    if not len(texts):
        return np.zeros((0, int(cfg.get('embed_dims', 1536))), dtype=np.float32)
    dims = int(cfg.get('embed_dims', 1536))
    out = [None] * len(texts)
    for start in range(0, len(texts), EMBED_BATCH):
        part = list(texts[start:start + EMBED_BATCH])
        r = requests.post(cfg['embed_url'], timeout=90,
                          headers={'Authorization': 'Bearer ' + cfg.get('embed_token', '')},
                          json={'model': cfg['embed_model'], 'input': part,
                                'dimensions': dims})
        if r.status_code != 200:
            raise RuntimeError('embedding HTTP %s: %s' % (r.status_code, r.text[:150]))
        data = r.json()['data']
        if len(data) != len(part):
            raise RuntimeError('embedding 返回条数不足（%d/%d）' % (len(data), len(part)))
        for item in data:
            v = np.array(item['embedding'], dtype=np.float32)
            if len(v) != dims:
                raise RuntimeError('embedding 维度 %d 与配置 %d 不一致' % (len(v), dims))
            out[start + int(item['index'])] = v
    if any(v is None for v in out):
        raise RuntimeError('embedding 返回条数不足')
    mat = np.vstack(out)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _rrf(lists, weights, k=RRF_K):
    """lists: [(rank1, rank2, ...)] 由调用方拼装；返回 {rowid: {...}}。"""
    fused = {}
    for (kind, weight), ranked in zip(weights, lists):
        for rank, item in enumerate(ranked, 1):
            rowid = item[0]
            rec = fused.setdefault(rowid, {'rrf': 0.0, 'vec': None, 'rv': None, 'rf': None})
            rec['rrf'] += weight / (k + rank)
            rec[kind] = rank if kind == 'rv' else rec[kind]
            if kind == 'rv':
                rec['vec'] = item[1]
            if kind == 'rf':
                rec['rf'] = rank
    return fused


def search(db, vec, query, top_k, min_score, mode, embedding_weight, cfg):
    q = (query or '').strip()
    if mode not in MODES:
        mode = MODE_DEFAULT
    qv, embed_err = None, None
    if mode != 'fulltext':
        try:
            qv = embed([q], cfg)[0]
        except Exception as e:
            if mode == 'embedding':   # 显式要向量路的，如实报错
                raise
            # hybrid：embedding 故障时降级为仅全文，答题链路不至于整体不可用
            embed_err = str(e)[:150]
            print('[search] embedding 失败，hybrid 降级为仅全文: %s' % embed_err, flush=True)
            mode = 'fulltext'

    vec_ranked, fts_ranked = [], []
    if mode in ('embedding', 'hybrid'):
        raw = vec.search(qv, max(VEC_RECALL, top_k * 4))
        labels = [lab for lab, _ in raw]
        rowmap = db.rowids_for_labels(labels)
        vec_ranked = [(rowmap[lab], score) for lab, score in raw if lab in rowmap]

    if mode in ('fulltext', 'hybrid') and len(q) >= 3:
        fts_ranked = db.search_fts(q, FTS_RECALL)
    # 短查询（<3 字）trigram 无法匹配，退化为向量路

    if mode == 'embedding':
        fused = {rid: {'rrf': 1.0 / (RRF_K + i + 1), 'vec': score, 'rv': i + 1, 'rf': None}
                 for i, (rid, score) in enumerate(vec_ranked)}
    elif mode == 'fulltext':
        fused = {rid: {'rrf': 1.0 / (RRF_K + i + 1), 'vec': None, 'rv': None, 'rf': i + 1}
                 for i, (rid, _bm) in enumerate(fts_ranked)}
    else:
        w = max(0.0, min(1.0, float(embedding_weight)))
        fused = _rrf([vec_ranked, fts_ranked], [('rv', w), ('rf', 1.0 - w)])

    ranked = sorted(fused.items(), key=lambda kv: -kv[1]['rrf'])[:top_k]
    rowids = [rid for rid, _ in ranked]
    rows = db.chunk_rows(rowids)
    out = []
    for rid, rec in ranked:
        row = rows.get(rid)
        if not row:
            continue
        if rec['vec'] is not None:
            score, score_type = round(rec['vec'], 4), 'vector'
        else:
            # 仅全文命中：给一个 0~1 的弱可比分数，避免调用方把 BM25 原始分当余弦
            score, score_type = 0.5, 'fulltext'
        if min_score and score < min_score:
            continue    # 阈值对两路统一：仅全文命中占位 0.5 同样受 min_score 约束
        out.append({
            'title': row['title'],
            'dataset': getattr(db, 'name', ''),
            'chunk_index': row['chunk_index'],
            'text': row['text'],
            'doc_id': row['doc_id'],
            'score': score,
            'score_type': score_type,
            'rrf_score': round(rec['rrf'], 6),
            'rank_vector': rec['rv'],
            'rank_fulltext': rec['rf'],
        })
    info = {'mode': mode, 'vector_hits': len(vec_ranked), 'fulltext_hits': len(fts_ranked)}
    if embed_err:
        info['degraded'] = True
        info['embed_error'] = embed_err
    return out, info
