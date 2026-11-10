# -*- coding: utf-8 -*-
"""置信度阈值扫描：用一张"检出率–误报率"曲线选工作点。

思路：只做一次推理（conf 放到很低），把预测结果留在内存里，再离线遍历阈值。
这样 19 个阈值点的代价 = 1 次前向，而不是 19 次。

输出：
    runs/eval/conf_sweep/sweep.csv    每个阈值一行
    runs/eval/conf_sweep/sweep.png    召回 / 误报 / 每图框数 曲线

用法：
    python conf_sweep.py
    python conf_sweep.py --weights runs/detect/cplid_yolov8n_cpu/weights/best.pt --imgsz 320
    python conf_sweep.py --conf-min 0.05 --conf-max 0.9 --conf-step 0.05
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import HERE, init_env

init_env()

import cv2  # noqa: E402  (必须在 init_env 之后)

DATA_DIR = HERE / "datasets" / "cplid"
DEFAULT_WEIGHTS = HERE / "runs" / "detect" / "cplid_yolov8n_cpu" / "weights" / "best.pt"
CONF_FLOOR = 0.01  # 推理时保留的最低置信度，阈值扫描的下界


# ---------------------------------------------------------------- 数据
def load_ground_truth(labels_dir: Path, images_dir: Path) -> dict:
    """返回 {stem: [(cls, x1, y1, x2, y2), ...]}，坐标已换算到像素。"""
    files = {p.stem: p for p in sorted(images_dir.glob("*.jpg"))}
    gt = {}
    for stem, img_path in files.items():
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        boxes = []
        lbl = labels_dir / f"{stem}.txt"
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                boxes.append((
                    c,
                    (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h,
                ))
        gt[stem] = boxes
    return gt


# ---------------------------------------------------------------- 指标
def iou_one_to_many(box, boxes):
    if not boxes:
        return np.zeros(0, dtype=np.float32)
    b = np.asarray(boxes, dtype=np.float32)
    x1 = np.maximum(box[0], b[:, 0])
    y1 = np.maximum(box[1], b[:, 1])
    x2 = np.minimum(box[2], b[:, 2])
    y2 = np.minimum(box[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area = (box[2] - box[0]) * (box[3] - box[1])
    return inter / (area + area_b - inter + 1e-9)


def match_image(preds, gts, iou_thr=0.5):
    """单图内按置信度降序贪心匹配，返回每个预测是否 TP。"""
    used = [False] * len(gts)
    flags = []
    for p in preds:  # preds 已按 conf 降序
        candidates = [(i, g) for i, g in enumerate(gts) if not used[i] and g[0] == p[1]]
        if not candidates:
            flags.append(False)
            continue
        ious = iou_one_to_many(p[2:6], [g[1:5] for _, g in candidates])
        k = int(np.argmax(ious))
        if ious[k] >= iou_thr:
            used[candidates[k][0]] = True
            flags.append(True)
        else:
            flags.append(False)
    return flags


def sweep(preds_by_stem, gt, thresholds):
    rows = []
    stems = list(gt.keys())
    defect_imgs = [s for s in stems if s.startswith("d_")]
    normal_imgs = [s for s in stems if s.startswith("n_")]
    for t in thresholds:
        tp_def = fp_def = fn_def = 0
        tp_all = fp_all = fn_all = 0
        hit_any = hit_iou = fp_img = 0
        n_boxes = 0
        for s in stems:
            gt_boxes = gt[s]
            all_preds = [p for p in preds_by_stem.get(s, []) if p[0] >= t]
            n_boxes += len(all_preds)
            flags = match_image(all_preds, gt_boxes)
            # 框级统计
            for p, ok in zip(all_preds, flags):
                if p[1] == 1:
                    tp_def += ok
                    fp_def += not ok
                tp_all += ok
                fp_all += not ok
            n_gt_def = sum(1 for g in gt_boxes if g[0] == 1)
            fn_def += n_gt_def - sum(1 for p, ok in zip(all_preds, flags) if p[1] == 1 and ok)
            fn_all += len(gt_boxes) - sum(flags)
            # 图级统计（只看缺陷类）
            if s.startswith("d_"):
                has_def = any(p[1] == 1 for p in all_preds)
                has_def_matched = any(p[1] == 1 and ok for p, ok in zip(all_preds, flags))
                hit_any += has_def
                hit_iou += has_def_matched
            else:
                fp_img += any(p[1] == 1 for p in all_preds)
        p_def = tp_def / max(tp_def + fp_def, 1)
        r_def = tp_def / max(tp_def + fn_def, 1)
        p_all = tp_all / max(tp_all + fp_all, 1)
        r_all = tp_all / max(tp_all + fn_all, 1)
        rows.append(dict(
            conf=round(t, 3),
            preds_per_img=round(n_boxes / len(stems), 2),
            img_recall_any=round(hit_any / len(defect_imgs), 4),
            img_recall_iou50=round(hit_iou / len(defect_imgs), 4),
            img_fpr=round(fp_img / len(normal_imgs), 4),
            defect_P=round(p_def, 4),
            defect_R=round(r_def, 4),
            defect_F1=round(2 * p_def * r_def / max(p_def + r_def, 1e-9), 4),
            all_P=round(p_all, 4),
            all_R=round(r_all, 4),
        ))
    return rows, len(defect_imgs), len(normal_imgs)


# ---------------------------------------------------------------- 绘图
def pick_points(rows):
    """三档工作点：免复核（F1 最高）、筛查（误报≤3% 下召回最高）、保守（误报≤1%）。"""
    best_f1 = max(rows, key=lambda r: r["defect_F1"])
    screen = max([r for r in rows if r["img_fpr"] <= 0.03],
                 key=lambda r: r["img_recall_any"], default=None)
    conservative = max([r for r in rows if r["img_fpr"] <= 0.01],
                       key=lambda r: r["img_recall_any"], default=None)
    return best_f1, screen, conservative


def draw(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    conf = [r["conf"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(conf, [r["img_recall_any"] for r in rows], "o-", lw=2, label="图像级检出率（有缺陷框即算）")
    ax1.plot(conf, [r["img_recall_iou50"] for r in rows], "s--", lw=2, label="图像级检出率（IoU≥0.5 命中）")
    ax1.plot(conf, [r["defect_R"] for r in rows], "^-.", lw=1.5, label="框级缺陷 Recall")
    ax1.set_xlabel("置信度阈值 conf")
    ax1.set_ylabel("召回 / 检出率")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.3)
    ax1.set_title("阈值越高，召回越低")
    ax1.legend(fontsize=8, loc="lower left")

    color = "tab:red"
    fpr_pct = [r["img_fpr"] * 100 for r in rows]
    ax2.plot(conf, fpr_pct, "o-", color=color, lw=2, label="正常图误报率")
    ax2.axhline(10, color=color, ls="--", lw=1, alpha=0.6)
    ax2.text(conf[-1], 10.6, "可接受上限 10%", fontsize=8, color=color, ha="right")
    ax2.set_xlabel("置信度阈值 conf")
    ax2.set_ylabel("误报率（%）", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, max(12, max(fpr_pct) * 1.35))
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(conf, [r["preds_per_img"] for r in rows], "s--", color="tab:blue", lw=1.5,
              label="每图平均缺陷框数")
    ax2b.set_ylabel("每图平均框数", color="tab:blue")
    ax2b.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("阈值越低，误报与复核工作量越大")

    for a in (ax1, ax2):
        a.axvline(0.25, color="gray", ls=":", lw=1)
        y_text = 0.96 * a.get_ylim()[1]
        a.text(0.252, y_text, "当前默认 0.25", fontsize=7, color="gray",
               rotation=90, va="top")

    _, screen, _ = pick_points(rows)
    if screen:
        y = screen["img_recall_any"]
        ax1.plot([screen["conf"]], [y], "*", ms=15, color="tab:green", zorder=6)
        ax1.annotate(
            f"★ 推荐筛查工作点 {screen['conf']:.2f}：检出 {screen['img_recall_any']:.0%}"
            f" / 误报 {screen['img_fpr']:.1%}",
            xy=(screen["conf"], y), xytext=(0.30, 0.93),
            fontsize=8.5, color="tab:green", va="center",
        )

    fig.suptitle("CPLID 验证集（170 张）置信度阈值扫描 — YOLOv8n @320", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main(argv=None):
    ap = argparse.ArgumentParser(description="置信度阈值扫描（图级 + 框级）")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--confs", default=None,
                    help="自定义阈值列表（逗号分隔，优先于 min/max/step），如 0.02,0.05,0.1")
    ap.add_argument("--conf-min", type=float, default=0.05)
    ap.add_argument("--conf-max", type=float, default=0.90)
    ap.add_argument("--conf-step", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "eval" / "conf_sweep")
    args = ap.parse_args(argv)

    if not args.weights.exists():
        raise SystemExit(f"权重不存在：{args.weights}")

    images_dir = args.data / "images" / args.split
    labels_dir = args.data / "labels" / args.split
    gt = load_ground_truth(labels_dir, images_dir)
    print(f"[数据] {len(gt)} 张，缺陷图 {sum(s.startswith('d_') for s in gt)}，"
          f"正常图 {sum(s.startswith('n_') for s in gt)}，框 {sum(len(v) for v in gt.values())}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    paths = [str(images_dir / f"{s}.jpg") for s in gt]
    print(f"[推理] {len(paths)} 张 @ imgsz={args.imgsz}, conf>={CONF_FLOOR}")
    results = model.predict(source=paths, imgsz=args.imgsz, conf=CONF_FLOOR, iou=0.7,
                            max_det=300, device="cpu", verbose=False)

    preds = {}
    for r in results:
        stem = Path(r.path).stem
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            preds[stem] = []
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        order = np.argsort(-confs)
        preds[stem] = [(float(confs[i]), int(clss[i]), *[float(v) for v in xyxy[i]]) for i in order]

    if args.confs:
        thresholds = np.array([float(v) for v in args.confs.split(",") if v.strip()])
    else:
        thresholds = np.arange(args.conf_min, args.conf_max + 1e-9, args.conf_step)
    rows, n_def, n_norm = sweep(preds, gt, thresholds)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    draw(rows, args.out / "sweep.png")

    hdr = f"{'conf':>5} {'框/图':>5} {'检出率':>7} {'IoU命中':>7} {'误报率':>7} " \
          f"{'缺陷P':>6} {'缺陷R':>6} {'缺陷F1':>7} {'全体P':>6} {'全体R':>6}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        mark = "  <= 当前默认" if abs(r["conf"] - 0.25) < 1e-9 else ""
        print(f"{r['conf']:>5.2f} {r['preds_per_img']:>5.2f} {r['img_recall_any']:>7.3f} "
              f"{r['img_recall_iou50']:>7.3f} {r['img_fpr']:>7.3f} {r['defect_P']:>6.3f} "
              f"{r['defect_R']:>6.3f} {r['defect_F1']:>7.3f} {r['all_P']:>6.3f} {r['all_R']:>6.3f}{mark}")

    best_f1, screen, conservative = pick_points(rows)
    print("\n[推荐工作点]")
    print(f"  免复核/自动告警（缺陷 F1 最高）: conf={best_f1['conf']:.2f}  "
          f"检出率={best_f1['img_recall_any']:.1%} 误报率={best_f1['img_fpr']:.1%} "
          f"缺陷F1={best_f1['defect_F1']:.3f}")
    if screen:
        print(f"  筛查（误报≤3% 下召回最高）    : conf={screen['conf']:.2f}  "
              f"检出率={screen['img_recall_any']:.1%} 误报率={screen['img_fpr']:.1%} "
              f"每图 {screen['preds_per_img']:.2f} 框")
    if conservative:
        print(f"  保守（误报≤1%）               : conf={conservative['conf']:.2f}  "
              f"检出率={conservative['img_recall_any']:.1%} 误报率={conservative['img_fpr']:.1%} "
              f"每图 {conservative['preds_per_img']:.2f} 框")

    summary = dict(weights=str(args.weights), imgsz=args.imgsz, split=args.split,
                   images=len(gt), defect_images=n_def, normal_images=n_norm,
                   thresholds=[float(t) for t in thresholds], rows=rows,
                   recommend_alarm=best_f1["conf"],
                   recommend_screen=screen["conf"] if screen else None,
                   recommend_conservative=conservative["conf"] if conservative else None)
    (args.out / "sweep.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"\n产物：{args.out / 'sweep.csv'}\n      {args.out / 'sweep.png'}")


if __name__ == "__main__":
    main()
