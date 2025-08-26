#!/usr/bin/env python3
import os, json, csv, glob, math
from collections import Counter, defaultdict

# Default locations (same convention as pick_best_frame.py)
A_JSON = os.environ.get("A_JSON", "/workspace/out_A.json")
A_CSV  = os.environ.get("A_CSV",  "/workspace/out_A.csv")
B_JSON = os.environ.get("B_JSON", "/workspace/out_B.json")
B_CSV  = os.environ.get("B_CSV",  "/workspace/out_B.csv")

PAIR_CSV = os.environ.get("PAIR_CSV", "/workspace/ab_pairs.csv")
SUMMARY_CSV = os.environ.get("SUMMARY_CSV", "/workspace/ab_summary.csv")

TIMESTAMP_TOL = float(os.environ.get("TIMESTAMP_TOL", "0.050"))  # 50 ms pairing tolerance

def load_ndjson(path):
    rows=[]
    with open(path,'r') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj=json.loads(line)
                rows.append(obj)
            except Exception:
                pass
    return rows

def load_json_or_csv(json_path, csv_path):
    """
    Returns list of dict per frame:
      {idx, sec, nsec, num, sum_conf, classes: Counter}
    """
    data=[]
    if os.path.exists(json_path):
        rows=load_ndjson(json_path)
        for i,obj in enumerate(rows):
            dets=obj.get("detections", obj.get("boxes", []))
            # sum of score/probability
            ssum=0.0
            classes=Counter()
            for d in dets:
                score = d.get("score", d.get("probability", 0.0))
                try:
                    ssum += float(score)
                except Exception:
                    pass
                cls = d.get("cls", d.get("class_id", ""))
                if cls != "":
                    classes[cls]+=1
            data.append({
                "idx": i,
                "sec": obj.get("sec", obj.get("header",{}).get("stamp",{}).get("sec",0)),
                "nsec": obj.get("nsec", obj.get("header",{}).get("stamp",{}).get("nanosec",0)),
                "num": len(dets),
                "sum_conf": ssum,
                "classes": classes
            })
        if data:
            return data

    if os.path.exists(csv_path):
        with open(csv_path,'r', newline='') as f:
            rdr=csv.DictReader(f)
            for i,row in enumerate(rdr):
                def getf(k, default=0.0):
                    try: return float(row.get(k, default))
                    except: return default
                def geti(k, default=0):
                    try: return int(float(row.get(k, default)))
                    except: return default
                idx = geti("frame_idx", i)
                sec = geti("sec", geti("stamp_sec", 0))
                nsec= geti("nsec", geti("stamp_nanosec", 0))
                num = geti("num_boxes", geti("detections", 0))
                ssum= getf("sum_conf", 0.0)
                # class columns optional: class:car, class:person ...
                classes=Counter()
                for k,v in row.items():
                    if k.startswith("class:"):
                        cls=k.split(":",1)[1]
                        try:
                            cnt=int(float(v))
                            if cnt>0: classes[cls]+=cnt
                        except Exception:
                            pass
                data.append({"idx": idx, "sec": sec, "nsec": nsec, "num": num, "sum_conf": ssum, "classes": classes})
        if data:
            return data

    raise SystemExit(f"데이터 파일을 찾지 못했습니다: {json_path} 또는 {csv_path}")

def time_of(r):
    return float(r["sec"]) + float(r["nsec"])*1e-9

def pair_by_time_or_index(A, B, tol=0.05):
    """Returns list of pairs (a_idx, b_idx, dt_seconds)."""
    # If all timestamps are zero, fall back to index pairing
    ts_all_zero_A = all(r["sec"]==0 and r["nsec"]==0 for r in A)
    ts_all_zero_B = all(r["sec"]==0 and r["nsec"]==0 for r in B)
    pairs=[]

    if ts_all_zero_A or ts_all_zero_B:
        n=min(len(A), len(B))
        for i in range(n):
            pairs.append((i, i, 0.0))
        return pairs

    # Timestamp-based nearest neighbor with tolerance
    B_times=[time_of(r) for r in B]
    for i,a in enumerate(A):
        ta=time_of(a)
        # linear scan (short lists) — good enough; could be optimized if needed
        best_j=-1; best_dt=1e9
        for j,tb in enumerate(B_times):
            dt=abs(tb-ta)
            if dt<best_dt:
                best_dt=dt; best_j=j
        if best_dt<=tol:
            pairs.append((i,best_j,best_dt))
    return pairs

def summarize_set(data):
    nums=[r["num"] for r in data]
    sums=[r["sum_conf"] for r in data]
    def mean(xs): return sum(xs)/len(xs) if xs else 0.0
    def p50(xs):
        if not xs: return 0.0
        xs=sorted(xs); n=len(xs); mid=n//2
        return (xs[mid] if n%2 else 0.5*(xs[mid-1]+xs[mid]))
    classes=Counter()
    for r in data:
        classes.update(r.get("classes", {}))
    return {
        "frames": len(data),
        "num_mean": mean(nums),
        "num_median": p50(nums),
        "num_max": max(nums) if nums else 0,
        "sum_mean": mean(sums),
        "sum_median": p50(sums),
        "sum_max": max(sums) if sums else 0.0,
        "top_classes": dict(classes.most_common(10))
    }

def main():
    A = load_json_or_csv(A_JSON, A_CSV)
    B = load_json_or_csv(B_JSON, B_CSV)

    # Per-run summary
    sumA = summarize_set(A)
    sumB = summarize_set(B)

    # Pair frames and compute deltas
    pairs = pair_by_time_or_index(A, B, tol=TIMESTAMP_TOL)
    rows=[]
    for ai, bj, dt in pairs:
        a=A[ai]; b=B[bj]
        rows.append({
            "idxA": a["idx"], "secA": a["sec"], "nsecA": a["nsec"],
            "idxB": b["idx"], "secB": b["sec"], "nsecB": b["nsec"],
            "dt": round(dt,6),
            "numA": a["num"], "numB": b["num"], "d_num": (b["num"]-a["num"]),
            "sumA": round(a["sum_conf"],6), "sumB": round(b["sum_conf"],6), "d_sum": round(b["sum_conf"]-a["sum_conf"],6),
        })

    # Write pairwise CSV
    os.makedirs(os.path.dirname(PAIR_CSV), exist_ok=True)
    with open(PAIR_CSV, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["idxA","idxB"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write summary CSV
    with open(SUMMARY_CSV, "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["run","frames","num_mean","num_median","num_max","sum_mean","sum_median","sum_max","top_classes(json)"])
        w.writerow(["A", sumA["frames"], f"{sumA['num_mean']:.3f}", f"{sumA['num_median']:.3f}", sumA["num_max"],
                    f"{sumA['sum_mean']:.3f}", f"{sumA['sum_median']:.3f}", f"{sumA['sum_max']:.3f}", json.dumps(sumA["top_classes"], ensure_ascii=False)])
        w.writerow(["B", sumB["frames"], f"{sumB['num_mean']:.3f}", f"{sumB['num_median']:.3f}", sumB["num_max"],
                    f"{sumB['sum_mean']:.3f}", f"{sumB['sum_median']:.3f}", f"{sumB['sum_max']:.3f}", json.dumps(sumB["top_classes"], ensure_ascii=False)])

    print("[A] frames={frames}  num_mean={num_mean:.2f}  sum_mean={sum_mean:.2f}  top={top}".format(**sumA, top=list(sumA["top_classes"].items())[:3]))
    print("[B] frames={frames}  num_mean={num_mean:.2f}  sum_mean={sum_mean:.2f}  top={top}".format(**sumB, top=list(sumB["top_classes"].items())[:3]))
    print(f"[PAIR] wrote {len(rows)} rows → {PAIR_CSV}")
    print(f"[SUMMARY] wrote 2 rows → {SUMMARY_CSV}")

if __name__ == "__main__":
    main()
