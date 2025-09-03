#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO


DET10_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Generate pseudo-labels (BDD10) from a COCO-pretrained YOLO model")
    ap.add_argument("images_root", type=Path, nargs="?", default=Path("/workspace/dataset/bdd100k/bdd100k/images/100k"))
    ap.add_argument("labels_root", type=Path, nargs="?", default=Path("/workspace/dataset/bdd100k/bdd100k/labels/100k"))
    ap.add_argument("--model", type=str, default="yolo11n.pt", help="Ultralytics model weights (COCO-pretrained)")
    ap.add_argument("--splits", type=str, default="train,val", help="Comma-separated splits to process")
    ap.add_argument("--max_per_split", type=int, default=0, help="Limit number of images per split (0 = all)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    args = ap.parse_args()

    model = YOLO(args.model)

    # COCO id -> BDD10 id mapping (subset). 'rider' not available in COCO, 'traffic sign' approximated by 'stop sign'.
    coco_to_bdd: Dict[int, int] = {
        0: 0,   # person -> person
        2: 2,   # car -> car
        7: 3,   # truck -> truck
        5: 4,   # bus -> bus
        6: 5,   # train -> train
        3: 6,   # motorcycle -> motorcycle
        1: 7,   # bicycle -> bicycle
        9: 8,   # traffic light -> traffic light
        11: 9,  # stop sign -> traffic sign (approx)
    }

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split_dir = args.images_root / split
        out_dir = args.labels_root / split
        ensure_dir(out_dir)
        imgs: List[Path] = [p for p in split_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if args.max_per_split > 0:
            imgs = imgs[: args.max_per_split]
        print(f"Split {split}: {len(imgs)} images")

        for i, img_path in enumerate(imgs, 1):
            try:
                results = model.predict(source=str(img_path), conf=args.conf, iou=args.iou, verbose=False)
            except Exception as e:
                print(f"Predict failed {img_path}: {e}")
                continue
            if not results:
                continue
            res = results[0]
            boxes = getattr(res, 'boxes', None)
            if boxes is None:
                continue
            xywhn = boxes.xywhn.cpu().numpy() if hasattr(boxes, 'xywhn') else None
            clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else None
            if xywhn is None or clss is None:
                continue
            lines = []
            for (cx, cy, w, h), coco_id in zip(xywhn, clss):
                if coco_id not in coco_to_bdd:
                    continue
                bdd_id = coco_to_bdd[coco_id]
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"{bdd_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            out_txt = out_dir / (img_path.stem + ".txt")
            with out_txt.open("w") as f:
                f.write("\n".join(lines))
            if i % 100 == 0:
                print(f"  {split}: processed {i}/{len(imgs)}")

    print("Done. Pseudo-labels written under:", args.labels_root)


if __name__ == "__main__":
    main()

