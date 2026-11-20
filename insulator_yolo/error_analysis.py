# -*- coding: utf-8 -*-
"""T10：误差分析——把验证集上的漏检 / 误检 / 定位偏差抽成可复核的清单。

分工
----
脚本负责"算"：跑推理、按 IoU 匹配 GT、分桶、导出 CSV 与样例图；
人负责"判"：看导出的样例图，给每一类写归因（小目标 / 遮挡 / 合成痕迹 / 框偏移 ...）。

匹配规则（只看 defect 类，和业务口径一致）
------------------------------------------
- 漏检 FN：GT 有 defect 框，但没有任何 defect 预测与之 IoU >= 0.5；
- 定位偏差 LOC：有 IoU 在 [0.1, 0.5) 的 defect 预测（框偏了，但没完全跑掉）；
- 误检 FP：有 defect 预测，但没有任何 GT defect 与之 IoU >= 0.5。

用法
----
    python error_analysis.py --weights runs/detect/ning_baseline-2/weights/best.pt
    python error_analysis.py --weights <best.pt> --conf 0.05 --imgsz 320
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import HERE, imread_u, imwrite_u

DATA = HERE / "datasets" / "cplid"
OUT = HERE / "runs" / "error_analysis"
DEFECT_ID = 1


def load_gt(img_path: Path) -> np.ndarray:
    """读 YOLO 标签 -> (n, 5) 的 [cls, x1, y1, x2, y2]（像素）。"""
    lab = DATA / "labels" / "val" / (img_path.stem + ".txt")
    if not lab.exists():
        return np.zeros((0, 5), dtype=np.float32)
    import cv2

    img = imread_u(img_path)
    h, w = img.shape[:2]
    rows = []
    for line in lab.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c, cx, cy, bw, bh = [float(x) for x in line.split()]
        rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,4) xyxy, b: (m,4) xyxy -> (n, m) IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def draw(img, gt, preds, path: Path):
    import cv2

    vis = img.copy()
    for x1, y1, x2, y2 in gt:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (80, 200, 80), 2)
    for x1, y1, x2, y2, conf in preds:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (60, 60, 230), 2)
        cv2.putText(vis, f"{conf:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 230), 1, cv2.LINE_AA)
    imwrite_u(path, vis)


def main() -> int:
    ap = argparse.ArgumentParser(description="验证集误差分析（漏检/误检/定位偏差）")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.05, help="筛查工作点（默认 0.05，与报告一致）")
    ap.add_argument("--iou", type=float, default=0.5, help="匹配 IoU 阈值")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--cases", type=int, default=20, help="每种错误导出的样例图数量")
    args = ap.parse_args()

    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases").mkdir(exist_ok=True)

    imgs = [Path(p) for p in (DATA / "val.txt").read_text(encoding="utf-8").split()]
    model = YOLO(str(args.weights))
    results = model.predict([str(p) for p in imgs], imgsz=args.imgsz, conf=args.conf,
                            verbose=False, max_det=300)

    fn_rows, fp_rows, loc_rows = [], [], []
    per_img = []
    for img_path, res in zip(imgs, results):
        gt = load_gt(img_path)
        gt_def = gt[gt[:, 0] == DEFECT_ID][:, 1:]
        boxes = res.boxes
        pred = np.zeros((0, 5), dtype=np.float32)
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            keep = clss == DEFECT_ID
            if keep.any():
                pred = np.concatenate([xyxy[keep], confs[keep][:, None]], axis=1)

        m = iou_matrix(gt_def, pred[:, :4] if len(pred) else np.zeros((0, 4)))
        best_gt = m.max(axis=1) if m.size else np.zeros(len(gt_def))
        best_pred = m.max(axis=0) if m.size else np.zeros(len(pred))

        for i, (box, sc) in enumerate(zip(gt_def, best_gt)):
            w, h = box[2] - box[0], box[3] - box[1]
            row = [img_path.name, "defect", round(float(w * h), 1), round(float(w), 1),
                   round(float(h), 1), round(float(sc), 3)]
            if sc >= args.iou:
                pass
            elif sc >= 0.1:
                loc_rows.append(row + ["定位偏差"])
            else:
                fn_rows.append(row + ["漏检"])
        for j, (box, sc) in enumerate(zip(pred, best_pred)):
            if sc < args.iou:
                w, h = box[2] - box[0], box[3] - box[1]
                fp_rows.append([img_path.name, "defect", round(float(w * h), 1), round(float(w), 1),
                                round(float(h), 1), round(float(box[4]), 3), round(float(sc), 3),
                                "误检"])

        n_gt_def, n_pred_def = len(gt_def), len(pred)
        per_img.append({
            "image": img_path.name,
            "gt_defect": n_gt_def,
            "pred_defect": n_pred_def,
            "hit": int((best_gt >= args.iou).sum()) if len(gt_def) else 0,
            "best_iou": round(float(best_gt.max()), 3) if len(gt_def) else None,
        })

        if len(gt_def) and (best_gt.max() < args.iou if len(gt_def) else False):
            draw(imread_u(img_path), gt_def, pred, OUT / "cases" / f"fn_{img_path.stem}.jpg")
        if len(pred) and len(gt_def) == 0:
            draw(imread_u(img_path), gt_def, pred, OUT / "cases" / f"fp_{img_path.stem}.jpg")

    with open(OUT / "fn.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "area_px", "w_px", "h_px", "best_iou", "type"])
        w.writerows(fn_rows)
    with open(OUT / "fp.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "area_px", "w_px", "h_px", "conf", "best_iou", "type"])
        w.writerows(fp_rows)
    with open(OUT / "loc.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "area_px", "w_px", "h_px", "best_iou", "type"])
        w.writerows(loc_rows)

    n_def_imgs = sum(1 for x in per_img if x["gt_defect"] > 0)
    hit_imgs = sum(1 for x in per_img if x["gt_defect"] > 0 and x["hit"] > 0)
    summary = {
        "weights": str(args.weights),
        "conf": args.conf,
        "iou_threshold": args.iou,
        "val_images": len(imgs),
        "val_defect_images": n_def_imgs,
        "images_with_hit": hit_imgs,
        "image_level_recall": round(hit_imgs / max(n_def_imgs, 1), 4),
        "gt_defect_boxes": sum(x["gt_defect"] for x in per_img),
        "pred_defect_boxes": sum(x["pred_defect"] for x in per_img),
        "fn": len(fn_rows),
        "fp": len(fp_rows),
        "loc_error": len(loc_rows),
        "fn_area_median": round(float(np.median([r[2] for r in fn_rows])), 1) if fn_rows else None,
        "fp_conf_median": round(float(np.median([r[5] for r in fp_rows])), 3) if fp_rows else None,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "per_image.json").write_text(json.dumps(per_img, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())