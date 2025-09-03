#!/usr/bin/env python3
import argparse
import json
import csv
import os
import re
from collections import defaultdict

# Optional heavy deps are imported lazily in eval step

def load_gt(gt_path):
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    # COCO typically uses 'file_name'; Argoverse-HD uses 'name'. Support both.
    images = gt.get('images', [])
    images_by_fname = {}
    images_order = []  # list of (image_id, file_name)
    images_time = []   # list of (image_id, timestamp_ns or None)
    for img in images:
        img_id = img.get('id')
        fname = img.get('file_name') or img.get('name') or ''
        images_by_fname[fname] = img_id
        images_order.append((img_id, fname))
        # Try extract integer timestamp from fname (Argoverse: ring_front_center_XXXXXXXXXXXXXXX.jpg)
        ts_ns = None
        try:
            base = os.path.splitext(os.path.basename(fname))[0]
            # take last continuous digits as timestamp
            import re as _re
            m = _re.search(r'(\d{12,})$', base)
            if m:
                ts_ns = int(m.group(1))
        except Exception:
            ts_ns = None
        images_time.append((img_id, ts_ns))
    cats_by_name = {}
    for c in gt.get('categories', []):
        name = str(c.get('name','')).strip().lower()
        cats_by_name[name] = c.get('id')
        # common alias: spaces to underscore
        cats_by_name[name.replace(' ', '_')] = c.get('id')
    return gt, images_by_fname, cats_by_name, images_order, images_time

def normalize_class(name: str):
    if name is None:
        return ''
    s = str(name).strip().lower()
    s = s.replace('-', '_').replace(' ', '_')
    # simple synonyms
    if s in ('traffic_light', 'trafficlight', 'signal'):
        s = 'traffic light'.replace(' ', '_')
    return s

def build_class_map(cats_by_name, override_map_json=None):
    m = {}
    if override_map_json:
        with open(override_map_json,'r') as f:
            raw = json.load(f)
        for k,v in raw.items():
            m[normalize_class(k)] = int(v)
    # fall back: map by name from GT
    for n, cid in cats_by_name.items():
        m[normalize_class(n)] = cid
    return m

def parse_csv(csv_path):
    """Yield dict rows for each detection from our overlay CSV."""
    with open(csv_path, 'r', newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                frame_idx = int(float(row.get('frame_idx', row.get('frame', '0'))))
                x1 = int(float(row.get('xmin', 0)))
                y1 = int(float(row.get('ymin', 0)))
                x2 = int(float(row.get('xmax', 0)))
                y2 = int(float(row.get('ymax', 0)))
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                score = float(row.get('prob', row.get('score', 0.0)))
                cls = row.get('class', row.get('class_id', ''))
            except Exception:
                continue
            yield {
                'frame_idx': frame_idx,
                'bbox': [x1, y1, w, h],
                'score': score,
                'class': cls,
            }

def map_image_id(det, images_by_fname, filename_template=None, id_offset=0, fallback_image_id=None):
    if filename_template:
        fname = filename_template.format(frame_idx=det['frame_idx'])
        if fname in images_by_fname:
            return images_by_fname[fname]
    if fallback_image_id is not None:
        return fallback_image_id
    return det['frame_idx'] + id_offset

def csv_to_coco_results(csv_path, images_by_fname, class_map, filename_template=None, id_offset=0, fallback_image_id=None, images_order=None, map_by_order=False, first_frame_offset=0, map_by_time=False, images_time=None, time_tol_ns=100_000_000):
    results = []
    for det in parse_csv(csv_path):
        cname = normalize_class(det['class'])
        cat_id = class_map.get(cname)
        if cat_id is None:
            # skip classes unknown to GT
            continue
        if map_by_time and images_time:
            # Need timestamp in CSV
            # We extended parse_csv to only read boxes; fetch stamps by re-reading row? Simpler: open CSV again for stamps
            pass
        if map_by_order and images_order:
            idx = det['frame_idx'] + int(first_frame_offset)
            if 0 <= idx < len(images_order):
                image_id = images_order[idx][0]
            else:
                # out of range -> skip
                continue
        else:
            image_id = map_image_id(det, images_by_fname, filename_template, id_offset, fallback_image_id)
        results.append({
            'image_id': int(image_id),
            'category_id': int(cat_id),
            'bbox': [float(det['bbox'][0]), float(det['bbox'][1]), float(det['bbox'][2]), float(det['bbox'][3])],
            'score': float(det['score'])
        })
    return results

def run_eval(gt_path, results, iou_type='bbox', pr_out=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    cocoGt = COCO(gt_path)
    # Some datasets (e.g., Argoverse-HD annotations) may miss optional COCO keys.
    # pycocotools expects these keys to exist when cloning GT into result dataset.
    if 'info' not in cocoGt.dataset:
        cocoGt.dataset['info'] = {}
    if 'licenses' not in cocoGt.dataset:
        cocoGt.dataset['licenses'] = []
    cocoDt = cocoGt.loadRes(results)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType=iou_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    if pr_out:
        os.makedirs(pr_out, exist_ok=True)
        # precision: [TxRxKxAxM] (IoU x recall x categories x area x maxDet)
        precisions = cocoEval.eval['precision']  # shape: T,R,K,A,M
        recalls = cocoEval.params.recThrs          # R
        iouThrs = cocoEval.params.iouThrs         # T
        catIds = cocoEval.params.catIds           # K
        # overall (all cats, area=0, maxDet last) mean over IoU thresholds
        import csv as _csv
        # Per-category PR at IoU=0.50 and 0.75 (common)
        for k_idx, catId in enumerate(catIds):
            for iou_target in (0.50, 0.75):
                # find closest IoU index
                t_idx = int(np.argmin(np.abs(iouThrs - iou_target)))
                pr = precisions[t_idx, :, k_idx, 0, -1]
                out_csv = os.path.join(pr_out, f'pr_cat{catId}_iou{int(iou_target*100)}.csv')
                with open(out_csv, 'w', newline='') as f:
                    w = _csv.writer(f)
                    w.writerow(['recall','precision'])
                    for r, p in zip(recalls, pr):
                        val = None if p == -1 else float(p)
                        w.writerow([float(r), val])
        # Also write mean PR across categories at IoU=0.50:0.95
        mean_pr = precisions.mean(axis=2)  # avg over categories
        out_csv = os.path.join(pr_out, f'pr_mean_iou50_95.csv')
        with open(out_csv, 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['iou_index','recall','precision'])
            for t_idx in range(mean_pr.shape[0]):
                pr = mean_pr[t_idx, :, 0, -1]
                for r, p in zip(recalls, pr):
                    val = None if p == -1 else float(p)
                    w.writerow([t_idx, float(r), val])

def _auto_search_offset(gt, csv_path, max_frames=200, search_range=100, iou_thr=0.3, step=1):
    """Heuristically search first_frame_offset that maximizes rough IoU matches.

    Returns best_offset or None if unable to estimate.
    """
    from collections import defaultdict

    if search_range <= 0:
        return None

    # Build GT lookup: per (image_id, category_id) -> list of bboxes
    anns_by_img_cat = defaultdict(list)
    for a in gt.get('annotations', []):
        anns_by_img_cat[(a['image_id'], a['category_id'])].append(a['bbox'])
    # Build category id by normalized name
    cat_id_by_name = {}
    for c in gt.get('categories', []):
        name = normalize_class(c.get('name',''))
        cat_id_by_name[name] = c.get('id')

    # Load a subset of CSV grouped by frame_idx
    csv_by_frame = defaultdict(list)
    frames = []
    with open(csv_path, 'r', newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                fi = int(float(row.get('frame_idx', row.get('frame', '0'))))
                x1 = int(float(row.get('xmin', 0))); y1 = int(float(row.get('ymin', 0)))
                x2 = int(float(row.get('xmax', 0))); y2 = int(float(row.get('ymax', 0)))
                cls = normalize_class(row.get('class', row.get('class_id', '')))
            except Exception:
                continue
            if fi not in csv_by_frame and len(frames) < max_frames:
                frames.append(fi)
            if fi in csv_by_frame or len(frames) <= max_frames:
                csv_by_frame[fi].append((x1, y1, max(0, x2-x1), max(0, y2-y1), cls))
            if len(frames) >= max_frames and fi > frames[-1] and len(csv_by_frame) >= max_frames:
                # collected enough distinct frames (approx)
                pass

    if not frames:
        return None
    frames = sorted(frames)

    def iou_xywh(b1, b2):
        x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
        xa = max(x1, x2); ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2); yb = min(y1 + h1, y2 + h2)
        iw = max(0.0, xb - xa); ih = max(0.0, yb - ya)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        uni = w1 * h1 + w2 * h2 - inter
        return inter / uni if uni > 0 else 0.0

    best_off = None
    best_score = -1
    for off in range(-int(search_range), int(search_range) + 1, max(1, int(step))):
        total = 0
        for fidx in frames:
            img_id = fidx + off
            for (x, y, w, h, cls) in csv_by_frame.get(fidx, []):
                cid = cat_id_by_name.get(cls)
                if cid is None:
                    continue
                gts = anns_by_img_cat.get((img_id, cid), [])
                if not gts:
                    continue
                best_iou = 0.0
                for g in gts:
                    iou = iou_xywh((x, y, w, h), g)
                    if iou > best_iou:
                        best_iou = iou
                        if best_iou >= iou_thr:
                            break
                if best_iou >= iou_thr:
                    total += 1
        if total > best_score:
            best_score = total
            best_off = off

    return best_off

def main():
    ap = argparse.ArgumentParser(description='Convert overlay CSV to COCO results and evaluate with pycocotools.')
    ap.add_argument('--csv', required=True, help='Path to detections CSV (e.g., out_A.csv)')
    ap.add_argument('--gt', required=True, help='Path to COCO GT annotations JSON')
    ap.add_argument('--results-out', default=None, help='Write COCO results JSON to this path')
    ap.add_argument('--filename-template', default='frame_{frame_idx:06d}.jpg', help='Template to map frame_idx to GT file_name (uses images.file_name/name)')
    ap.add_argument('--image-id-offset', type=int, default=0, help='Fallback: image_id = frame_idx + offset')
    ap.add_argument('--class-map-json', default=None, help='Optional JSON mapping {"class_name": category_id}')
    ap.add_argument('--map-by-order', action='store_true', help='Map detections to GT images by index order (frame_idx + offset -> images order)')
    ap.add_argument('--first-frame-offset', type=int, default=0, help='Index offset when using --map-by-order (use negative to align if CSV starts later)')
    ap.add_argument('--pr-out', default=None, help='Directory to write PR curve CSVs')
    ap.add_argument('--map-by-time', action='store_true', help='Map by timestamp: CSV (sec,nsec) -> GT filename timestamp')
    ap.add_argument('--time-tol-sec', type=float, default=0.1, help='Time tolerance in seconds for --map-by-time')
    ap.add_argument('--auto-offset-range', type=int, default=0, help='If >0 and using --map-by-order, search offsets in [-range, +range] and pick the best')
    ap.add_argument('--auto-offset-frames', type=int, default=200, help='How many early frames to sample for auto-offset search')
    ap.add_argument('--auto-offset-iou', type=float, default=0.3, help='IoU threshold used in auto-offset search heuristic')
    ap.add_argument('--auto-offset-step', type=int, default=1, help='Offset search step')
    args = ap.parse_args()

    gt, images_by_fname, cats_by_name, images_order, images_time = load_gt(args.gt)
    class_map = build_class_map(cats_by_name, args.class_map_json)

    # Optional: auto-estimate first_frame_offset when mapping by order
    if args.map_by_order and int(args.auto_offset_range) > 0:
        best_off = _auto_search_offset(
            gt,
            args.csv,
            max_frames=int(args.auto_offset_frames),
            search_range=int(args.auto_offset_range),
            iou_thr=float(args.auto_offset_iou),
            step=int(args.auto_offset_step),
        )
        if best_off is not None:
            print(f'[auto-offset] Selected first_frame_offset = {best_off}')
            args.first_frame_offset = int(best_off)
        else:
            print('[auto-offset] Unable to estimate offset; proceeding with provided --first-frame-offset')
    # If map-by-time, rebuild a quick lookup of GT timestamps
    if args.map_by_time:
        # Build list of (ts_ns, image_id)
        gt_ts = [(ts if ts is not None else -1, img_id) for (img_id, ts) in images_time]
        gt_ts = [x for x in gt_ts if x[0] >= 0]
        gt_ts.sort()
        # Build mapping from CSV row to image_id by nearest ts within tol
        tol_ns = int(args.time_tol_sec * 1e9)
        # Load detections with stamps directly
        dets = []
        with open(args.csv, 'r', newline='') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                try:
                    sec = int(float(row.get('stamp_sec', row.get('sec', 0))))
                    nsec = int(float(row.get('stamp_nanosec', row.get('nsec', 0))))
                    ts = sec * 1_000_000_000 + nsec
                    frame_idx = int(float(row.get('frame_idx', row.get('frame', '0'))))
                    x1 = int(float(row.get('xmin', 0))); y1 = int(float(row.get('ymin', 0)))
                    x2 = int(float(row.get('xmax', 0))); y2 = int(float(row.get('ymax', 0)))
                    w = max(0, x2 - x1); h = max(0, y2 - y1)
                    score = float(row.get('prob', row.get('score', 0.0)))
                    cls = row.get('class', row.get('class_id', ''))
                except Exception:
                    continue
                dets.append({'ts': ts, 'bbox':[x1,y1,w,h], 'score': score, 'class': cls})
        # helper: binary search nearest ts
        import bisect
        gtts = [ts for ts, _ in gt_ts]
        results = []
        for d in dets:
            # find nearest index
            i = bisect.bisect_left(gtts, d['ts'])
            candidates = []
            if i < len(gtts): candidates.append(i)
            if i > 0: candidates.append(i-1)
            best = None; best_dt = None
            for j in candidates:
                dt = abs(gtts[j] - d['ts'])
                if best is None or dt < best_dt:
                    best = j; best_dt = dt
            if best is None or best_dt is None or best_dt > tol_ns:
                continue
            image_id = gt_ts[best][1]
            cname = normalize_class(d['class'])
            cat_id = class_map.get(cname)
            if cat_id is None:
                continue
            results.append({
                'image_id': int(image_id),
                'category_id': int(cat_id),
                'bbox': [float(d['bbox'][0]), float(d['bbox'][1]), float(d['bbox'][2]), float(d['bbox'][3])],
                'score': float(d['score'])
            })
    else:
        results = csv_to_coco_results(
            args.csv,
            images_by_fname,
            class_map,
            filename_template=args.filename_template,
            id_offset=args.image_id_offset,
            images_order=images_order,
            map_by_order=args.map_by_order,
            first_frame_offset=args.first_frame_offset,
        )
    if not results:
        raise SystemExit('No detections converted. Check class map and CSV contents.')
    if args.results_out:
        os.makedirs(os.path.dirname(args.results_out) or '.', exist_ok=True)
        with open(args.results_out, 'w') as f:
            json.dump(results, f)
    run_eval(args.gt, results, iou_type='bbox', pr_out=args.pr_out)

if __name__ == '__main__':
    main()
