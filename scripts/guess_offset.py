#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict

def load_det_counts(csv_path):
    counts = defaultdict(int)
    with open(csv_path, 'r', newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                idx = int(float(row.get('frame_idx', row.get('frame', '0'))))
            except Exception:
                continue
            counts[idx] += 1
    return counts

def load_gt_counts(gt_json):
    with open(gt_json, 'r') as f:
        gt = json.load(f)
    # Build image order list and annotation counts by image_id
    images = gt.get('images', [])
    order = [img.get('id') for img in images]
    ann_counts = defaultdict(int)
    for ann in gt.get('annotations', []):
        ann_counts[int(ann.get('image_id'))] += 1
    gt_counts = [ann_counts.get(int(img_id), 0) for img_id in order]
    return gt_counts

def score_offset(det_counts, gt_counts, offset):
    # similarity: sum over overlapping indices of det_count[i] * gt_count[i+offset]
    # handles negative offsets
    score = 0
    min_i = max(0, -offset)
    max_i = min(max(det_counts.keys(), default=-1), len(gt_counts)-1 - offset)
    if max_i < min_i:
        return 0
    for i in range(min_i, max_i+1):
        dc = det_counts.get(i, 0)
        gc = gt_counts[i+offset] if 0 <= (i+offset) < len(gt_counts) else 0
        if dc and gc:
            score += dc * gc
    return score

def main():
    ap = argparse.ArgumentParser(description='Guess best frame index offset between CSV detections and GT order (Argoverse/COCO).')
    ap.add_argument('--csv', required=True, help='Detections CSV (out_A.csv or out_B.csv)')
    ap.add_argument('--gt', required=True, help='GT JSON (e.g., Argoverse-HD val.json)')
    ap.add_argument('--range', type=int, default=1000, help='Search offsets in [-range, +range]')
    ap.add_argument('--topk', type=int, default=5, help='Show top-K candidate offsets')
    args = ap.parse_args()

    det_counts = load_det_counts(args.csv)
    if not det_counts:
        raise SystemExit('No detections found in CSV.')
    gt_counts = load_gt_counts(args.gt)
    if not gt_counts:
        raise SystemExit('GT has no images/annotations.')

    best = []  # list of (score, offset)
    for off in range(-args.range, args.range+1):
        s = score_offset(det_counts, gt_counts, off)
        if s > 0:
            best.append((s, off))
    best.sort(reverse=True)
    print(f"Candidates (score, offset) top-{args.topk} out of {len(best)} with positive score:")
    for s, off in best[:args.topk]:
        print(f"  score={s}  offset={off}")
    if best:
        print(f"Suggested --first-frame-offset {best[0][1]}")
    else:
        print("No positive-scoring offset found. Try expanding --range or ensure the CSV/GT correspond to the same split.")

if __name__ == '__main__':
    main()

