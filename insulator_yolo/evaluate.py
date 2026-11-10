# -*- coding: utf-8 -*-
"""
在验证集上评估训练好的模型，输出整体指标与逐类指标，并保存为 JSON/CSV。

用法：
    python evaluate.py --weights runs/detect/cplid_yolov8n_cpu/weights/best.pt
    python evaluate.py --weights best.pt --split val --imgsz 640
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from common import HERE, init_env

init_env()

DEFAULT_DATA = HERE / "datasets" / "cplid" / "data.yaml"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="评估 CPLID 检测模型")
    ap.add_argument("--weights", type=Path, required=True, help="模型权重，如 best.pt")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.001, help="评估时的置信度阈值（mAP 计算用低阈值）")
    ap.add_argument("--plots", action="store_true", help="输出 P-R 曲线、混淆矩阵等图")
    ap.add_argument("--project", type=Path, default=HERE / "runs" / "detect",
                    help="结果输出根目录")
    ap.add_argument("--name", default=None, help="输出子目录名，默认 eval_<权重所在目录>")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not args.weights.exists():
        sys.exit(f"[错误] 找不到权重：{args.weights}")
    if not args.data.exists():
        sys.exit(f"[错误] 找不到数据集配置：{args.data}（先运行 prepare_dataset.py）")

    import torch
    from ultralytics import YOLO

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    print(f"[评估] weights={args.weights}")
    print(f"[评估] data={args.data} split={args.split} imgsz={args.imgsz} device={device}")

    model = YOLO(str(args.weights))
    name = args.name or f"eval_{args.weights.parent.parent.name}"
    m = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        conf=args.conf,
        workers=0,
        plots=args.plots,
        project=str(args.project),
        name=name,
        exist_ok=True,
        verbose=True,
    )

    names = getattr(m, "names", {}) or {}
    raw_idx = getattr(m.box, "ap_class_index", None)
    idx = [int(i) for i in raw_idx] if raw_idx is not None else []

    ap50 = getattr(m.box, "ap50", None)
    ap = getattr(m.box, "ap", None)

    per_class = []
    for j, ci in enumerate(idx):
        per_class.append({
            "class_id": ci,
            "class_name": names.get(ci, str(ci)) if isinstance(names, dict) else str(ci),
            "AP50": round(float(ap50[j]), 4) if ap50 is not None and j < len(ap50) else None,
            "AP50-95": round(float(ap[j]), 4) if ap is not None and j < len(ap) else None,
        })

    nt = getattr(m, "nt_per_class", None)
    summary = {
        "weights": str(args.weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "images": int(getattr(m, "seen", 0)) or None,
        "instances": int(sum(nt)) if nt is not None else None,
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "per_class": per_class,
    }

    out_dir = Path(m.save_dir) if getattr(m, "save_dir", None) else args.weights.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "metrics_per_class.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "class_name", "AP50", "AP50-95"])
        w.writeheader()
        w.writerows(per_class)

    print("\n" + "=" * 60)
    print(f"整体: mAP50={summary['mAP50']:.4f}  mAP50-95={summary['mAP50-95']:.4f}  "
          f"P={summary['precision']:.4f}  R={summary['recall']:.4f}")
    print("逐类:")
    for row in per_class:
        print(f"  {row['class_name']:<10} AP50={row['AP50']}  AP50-95={row['AP50-95']}")
    print("=" * 60)
    print(f"指标已保存: {out_dir / 'metrics.json'}")
    print(f"逐类表格  : {csv_path}")

    return summary


if __name__ == "__main__":
    main()
