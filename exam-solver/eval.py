#!/usr/bin/env python3
"""批量评测答题准确率。

用法：
  python3 eval.py --dataset <id> --dir /path/to/images --answers answers.json
  python3 eval.py --dataset <id> --result result.json --answers answers.json

answers.json 支持两种写法：
  {"微信图片_xxx.jpg": ["B"], "另一张.jpg": ["A", "C"]}          # 按图片顺序给答案
  {"微信图片_xxx.jpg": [{"answer": "B"}, {"answer": "AC"}]}       # 显式对象
"""
import argparse
import json
import time
from pathlib import Path

import config
import solver

IMG_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def load_answers(path):
    d = json.loads(Path(path).read_text('utf-8'))
    out = {}
    for k, v in d.items():
        seq = []
        for item in v:
            seq.append(item['answer'] if isinstance(item, dict) else str(item))
        out[k] = seq
    return out


def norm(x):
    return ''.join(sorted(c for c in str(x).upper() if c.isalpha())) or str(x).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--dir')
    ap.add_argument('--result')
    ap.add_argument('--answers', required=True)
    ap.add_argument('--concurrency', type=int, default=5)
    ap.add_argument('--out', default='eval_report.json')
    args = ap.parse_args()

    ds = next((d for d in config.datasets() if d['id'] == args.dataset), None)
    if not ds:
        raise SystemExit('知识库不存在')

    if args.result:
        result = json.loads(Path(args.result).read_text('utf-8'))
    else:
        files = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in IMG_EXT)
        images = [(p.name, p.read_bytes()) for p in files]
        print(f'共 {len(images)} 张图片，开始识别与作答…')
        result = solver.solve_images(images, ds, concurrency=args.concurrency)

    gold = load_answers(args.answers)
    rows, correct, total = [], 0, 0
    for q in result['questions']:
        seq = gold.get(q.get('image', ''), [])
        idx = q['index'] - 1
        # 同一张图多题时按出现顺序对应
        same = [q2 for q2 in result['questions'] if q2.get('image') == q.get('image')]
        idx = same.index(q)
        exp = seq[idx] if idx < len(seq) else ''
        ok = bool(exp) and norm(exp) == norm(q['answer'])
        total += 1
        correct += int(ok)
        rows.append({'index': q['index'], 'image': q.get('image', ''), 'type': q['type'],
                     'stem': q['stem'], 'pred': q['answer'], 'gold': exp, 'ok': ok,
                     'confidence': q['confidence'], 'needs_review': q['needs_review']})

    acc = correct / total if total else 0
    report = {
        'dataset': ds['name'], 'total': total, 'correct': correct, 'accuracy': round(acc, 4),
        'needs_review': sum(1 for r in rows if r['needs_review']),
        'avg_confidence': round(sum(r['confidence'] for r in rows) / total, 3) if total else 0,
        'elapsed_ms': result['meta']['total_ms'], 'rows': rows,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), 'utf-8')
    print(f"准确率 {acc*100:.1f}% （{correct}/{total}），需复核 {report['needs_review']} 题")
    for r in rows:
        if not r['ok']:
            print(f"  ✗ 第{r['index']}题 预测={r['pred']} 标准={r['gold']} | {r['stem'][:45]}")
    print('报告已写入', args.out)


if __name__ == '__main__':
    main()
