# -*- coding: utf-8 -*-
"""
用训练好的模型做推理：输出带框的可视化图片、逐框检测明细、按图片汇总的缺陷统计表。

用法：
    python predict.py --weights runs/detect/cplid_yolov8n_cpu/weights/best.pt
    python predict.py --weights best.pt --source D:/巡检图 --conf 0.3
    python predict.py --weights best.pt --source datasets/cplid/images/val --imgsz 640

产物（默认 runs/predict/<name>/）：
    *.jpg                 画好检测框的图片
    detections.csv        每个检测框一行：图片, 类别, 置信度, x1,y1,x2,y2
    summary.csv           每张图一行：绝缘子数, 缺陷数, 最高缺陷置信度, 结论
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from common import HERE, init_env

init_env()

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="CPLID 绝缘子缺陷检测推理")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=HERE / "datasets" / "cplid" / "images" / "val",
                    help="图片文件或目录")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="cplid_predict")
    ap.add_argument("--project", type=Path, default=HERE / "runs" / "predict")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not args.weights.exists():
        sys.exit(f"[错误] 找不到权重：{args.weights}")
    if not args.source.exists():
        sys.exit(f"[错误] 找不到输入：{args.source}")

    import torch
    from ultralytics import YOLO

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    model = YOLO(str(args.weights))
    names = model.names  # {id: name}

    t0 = time.time()
    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        save=True,
        save_txt=False,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        verbose=False,
    )
    dt = time.time() - t0

    save_dir = Path(results[0].save_dir) if results else args.project / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    det_rows, sum_rows = [], []
    n_defect_img = 0
    for r in results:
        img_name = Path(r.path).name
        boxes = r.boxes
        cnt = {name: 0 for name in names.values()}
        best_defect_conf = 0.0

        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cf, ci in zip(xyxy, confs, clss):
                cname = names.get(ci, str(ci))
                cnt[cname] = cnt.get(cname, 0) + 1
                if cname == "defect":
                    best_defect_conf = max(best_defect_conf, float(cf))
                det_rows.append({
                    "image": img_name, "class": cname, "conf": round(float(cf), 4),
                    "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                    "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                })

        has_defect = cnt.get("defect", 0) > 0
        n_defect_img += int(has_defect)
        sum_rows.append({
            "image": img_name,
            "insulator": cnt.get("insulator", 0),
            "defect": cnt.get("defect", 0),
            "max_defect_conf": round(best_defect_conf, 4),
            "result": "有缺陷" if has_defect else "正常",
        })

    # UTF-8 BOM，Excel 直接双击不乱码
    det_csv = save_dir / "detections.csv"
    with open(det_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "class", "conf", "x1", "y1", "x2", "y2"])
        w.writeheader()
        w.writerows(det_rows)

    sum_csv = save_dir / "summary.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "insulator", "defect", "max_defect_conf", "result"])
        w.writeheader()
        w.writerows(sum_rows)

    n = len(results)
    print("\n" + "=" * 60)
    print(f"推理完成：{n} 张图，用时 {dt:.1f}s（{dt / max(n, 1) * 1000:.0f} ms/张）")
    print(f"检出缺陷的图片：{n_defect_img} / {n}")
    print(f"可视化图片目录：{save_dir}")
    print(f"检测明细：{det_csv}")
    print(f"汇总统计：{sum_csv}")
    print("=" * 60)

    return sum_rows


if __name__ == "__main__":
    main()
