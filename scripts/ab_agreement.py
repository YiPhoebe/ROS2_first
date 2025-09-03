#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict

def parse_csv(csv_path):
    by_frame = defaultdict(list)
    with open(csv_path, 'r', newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                idx = int(float(row.get('frame_idx', row.get('frame', '0'))))
                x1 = int(float(row.get('xmin', 0)))
                y1 = int(float(row.get('ymin', 0)))
                x2 = int(float(row.get('xmax', 0)))
                y2 = int(float(row.get('ymax', 0)))
                cls = str(row.get('class', row.get('class_id', '')))
            except Exception:
                continue
            by_frame[idx].append((x1,y1,x2,y2,cls))
    return by_frame

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0, ax2-ax1) * max(0, ay2-ay1)
    b_area = max(0, bx2-bx1) * max(0, by2-by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union

def match_sets(A, B, iou_th=0.5, class_aware=False):
    matched = 0
    used_b = set()
    for i, (ax1,ay1,ax2,ay2,ac) in enumerate(A):
        best_j = -1
        best_iou = 0.0
        for j, (bx1,by1,bx2,by2,bc) in enumerate(B):
            if j in used_b:
                continue
            if class_aware and (ac != bc):
                continue
            iou_val = iou((ax1,ay1,ax2,ay2), (bx1,by1,bx2,by2))
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_j >= 0 and best_iou >= iou_th:
            matched += 1
            used_b.add(best_j)
    return matched

def main():
    ap = argparse.ArgumentParser(description='Compute A<->B IoU agreement per frame and overall.')
    ap.add_argument('--csvA', required=True, help='Path to out_A.csv')
    ap.add_argument('--csvB', required=True, help='Path to out_B.csv')
    ap.add_argument('--iou', type=float, default=0.5, help='IoU threshold for matching')
    ap.add_argument('--class-aware', action='store_true', help='Require same class label to match')
    ap.add_argument('--out', default='ab_agreement.csv', help='Output CSV for per-frame metrics')
    args = ap.parse_args()

    A = parse_csv(args.csvA)
    B = parse_csv(args.csvB)
    frames = sorted(set(A.keys()) | set(B.keys()))

    rows = []
    sum_a_cov = 0.0
    sum_b_cov = 0.0
    sum_f1 = 0.0
    for f in frames:
        detA = A.get(f, [])
        detB = B.get(f, [])
        mAB = match_sets(detA, detB, iou_th=args.iou, class_aware=args.class_aware)
        mBA = mAB  # greedy is symmetric enough for rates here
        nA = len(detA)
        nB = len(detB)
        covA = (mAB / nA) if nA > 0 else (1.0 if nB == 0 else 0.0)
        covB = (mBA / nB) if nB > 0 else (1.0 if nA == 0 else 0.0)
        f1 = (2*mAB / (nA + nB)) if (nA + nB) > 0 else 1.0
        rows.append({
            'frame_idx': f,
            'A_boxes': nA,
            'B_boxes': nB,
            'matched': mAB,
            'A_covered': round(covA, 6),
            'B_covered': round(covB, 6),
            'sym_F1_like': round(f1, 6),
        })
        sum_a_cov += covA
        sum_b_cov += covB
        sum_f1 += f1

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['frame_idx'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n = max(1, len(frames))
    print(f'[AGREEMENT] frames={len(frames)}  A_covered_mean={sum_a_cov/n:.3f}  B_covered_mean={sum_b_cov/n:.3f}  sym_F1_like_mean={sum_f1/n:.3f}')
    print(f'[AGREEMENT] wrote per-frame CSV -> {args.out}')

if __name__ == '__main__':
    main()

