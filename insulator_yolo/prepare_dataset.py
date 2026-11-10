# -*- coding: utf-8 -*-
"""
CPLID 数据集转换：VOC(XML) -> YOLO(txt)，并按 正常/缺陷 分层划分训练集与验证集。

原始结构（InsulatorDataSet-master）：
    Normal_Insulators/
        images/                 600 张正常绝缘子航拍图
        labels/*.xml            只标注 insulator
    Defective_Insulators/
        images/                 248 张（含合成）缺陷绝缘子图
        labels/insulator/*.xml  标注 insulator
        labels/defect/*.xml     标注 defect

输出结构（标准 YOLO 布局）：
    <out>/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt
        train.txt / val.txt     图片路径清单（data.yaml 引用）
        data.yaml               训练配置
        stats.json              数据统计

用法：
    python prepare_dataset.py                        # 默认复制图片，8:2 划分
    python prepare_dataset.py --mode hardlink        # 同盘硬链接，不额外占用空间
    python prepare_dataset.py --val-ratio 0.2 --seed 42 --check 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- 常量
CLASS_TO_ID = {"insulator": 0, "defect": 1}
NAMES = ["insulator", "defect"]

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent / "InsulatorDataSet-master"
DEFAULT_OUT = HERE / "datasets" / "cplid"


# ---------------------------------------------------------------- VOC 解析
def parse_voc(xml_path: Path) -> list[tuple[int, float, float, float, float]]:
    """解析单个 VOC 标注文件，返回 [(class_id, cx, cy, w, h), ...]（均已归一化）。"""
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path} 缺少 <size> 节点")
    W = float(size.findtext("width"))
    H = float(size.findtext("height"))
    if W <= 0 or H <= 0:
        raise ValueError(f"{xml_path} 的图像尺寸非法: {W}x{H}")

    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASS_TO_ID:
            continue  # 忽略未知类别

        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        x1 = float(bnd.findtext("xmin"))
        y1 = float(bnd.findtext("ymin"))
        x2 = float(bnd.findtext("xmax"))
        y2 = float(bnd.findtext("ymax"))

        # 裁剪到图像范围内，并剔除退化的框
        x1, x2 = sorted((max(0.0, min(x1, W)), max(0.0, min(x2, W))))
        y1, y2 = sorted((max(0.0, min(y1, H)), max(0.0, min(y2, H))))
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            continue

        boxes.append((
            CLASS_TO_ID[name],
            (x1 + x2) / 2.0 / W,
            (y1 + y2) / 2.0 / H,
            bw / W,
            bh / H,
        ))
    return boxes


def collect(src: Path) -> list[dict]:
    """遍历两个子集，得到 [{img, stem_out, kind, boxes}, ...]。"""
    normal = src / "Normal_Insulators"
    defect = src / "Defective_Insulators"

    jobs = [
        ("normal", normal / "images", [normal / "labels"], "n"),
        ("defective", defect / "images", [defect / "labels" / "insulator",
                                          defect / "labels" / "defect"], "d"),
    ]

    items: list[dict] = []
    for kind, img_dir, xml_dirs, prefix in jobs:
        if not img_dir.is_dir():
            raise FileNotFoundError(f"找不到图片目录: {img_dir}")

        for img in sorted(img_dir.glob("*.jpg")):
            stem = img.stem
            boxes, xml_used = [], []
            for xd in xml_dirs:
                xf = xd / f"{stem}.xml"
                if xf.exists():
                    boxes.extend(parse_voc(xf))
                    xml_used.append(xf.name)
            if not xml_used:
                raise FileNotFoundError(f"{img} 没有对应的 XML 标注")
            items.append({
                "img": img,
                "stem_out": f"{prefix}_{stem}",
                "kind": kind,
                "boxes": boxes,
            })
    return items


# ---------------------------------------------------------------- 划分
def stratified_split(items: list[dict], val_ratio: float, seed: int):
    rng = random.Random(seed)
    train, val = [], []
    for kind in ("normal", "defective"):
        group = [it for it in items if it["kind"] == kind]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------- 落盘
def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # 跨盘等情况退回复制
    shutil.copy2(src, dst)


def write_split(items: list[dict], out: Path, split: str, mode: str) -> list[str]:
    paths = []
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for it in items:
        img_dst = out / "images" / split / f"{it['stem_out']}.jpg"
        link_or_copy(it["img"], img_dst, mode)

        lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in it["boxes"]]
        (out / "labels" / split / f"{it['stem_out']}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        paths.append(str(img_dst.resolve()).replace("\\", "/"))
    return paths


def write_data_yaml(out: Path):
    yaml_text = (
        f"# CPLID 绝缘子缺陷检测数据集（由 prepare_dataset.py 自动生成）\n"
        f"path: {str(out.resolve()).replace(chr(92), '/')}\n"
        f"train: train.txt\n"
        f"val: val.txt\n"
        f"names:\n"
        f"  0: insulator\n"
        f"  1: defect\n"
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")


# ---------------------------------------------------------------- 自检可视化
def draw_check(out: Path, n: int, seed: int = 0):
    """在验证集上抽 n 张，把 YOLO 标签画回图片，肉眼确认转换正确。"""
    try:
        import cv2
    except ImportError:
        print("[跳过] 未安装 opencv，无法生成标注自检图")
        return None

    from common import imread_u, imwrite_u

    val_imgs = sorted((out / "images" / "val").glob("*.jpg"))
    if not val_imgs:
        return None
    rng = random.Random(seed)
    picked = rng.sample(val_imgs, min(n, len(val_imgs)))

    check_dir = out / "_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    colors = {0: (80, 200, 80), 1: (60, 60, 230)}  # BGR: insulator 绿, defect 红

    for img_path in picked:
        img = imread_u(img_path)
        if img is None:
            print(f"[自检] 读取失败，跳过: {img_path}")
            continue
        H, W = img.shape[:2]
        lbl = out / "labels" / "val" / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                c, cx, cy, w, h = line.split()
                c = int(c)
                cx, cy, w, h = map(float, (cx, cy, w, h))
                x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
                x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), colors.get(c, (255, 0, 0)), 2)
                cv2.putText(img, NAMES[c], (x1 + 2, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.get(c, (255, 0, 0)), 2)
        imwrite_u(check_dir / img_path.name, img)

    print(f"[自检] 已生成 {len(picked)} 张标注可视化 -> {check_dir}")
    return check_dir


# ---------------------------------------------------------------- 主流程
def main(argv=None):
    ap = argparse.ArgumentParser(description="CPLID: VOC -> YOLO 数据集转换")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="InsulatorDataSet-master 目录")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出目录")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（默认 0.2）")
    ap.add_argument("--seed", type=int, default=42, help="划分随机种子")
    ap.add_argument("--mode", choices=["copy", "hardlink"], default="copy",
                    help="copy=复制图片；hardlink=同盘硬链接（省空间）")
    ap.add_argument("--defect-repeat", type=int, default=1,
                    help="训练集中缺陷图重复次数，用于缓解类别不均衡（默认 1=不重复）")
    ap.add_argument("--check", type=int, default=0, help="生成 N 张标注自检图")
    args = ap.parse_args(argv)

    src, out = args.src.resolve(), args.out.resolve()
    print(f"源目录: {src}")
    print(f"输出目录: {out}")
    if not src.is_dir():
        sys.exit(f"[错误] 源目录不存在: {src}")

    print("\n[1/4] 解析 VOC 标注 ...")
    items = collect(src)
    n_normal = sum(1 for i in items if i["kind"] == "normal")
    n_defect = len(items) - n_normal
    print(f"      共 {len(items)} 张（正常 {n_normal}，缺陷 {n_defect}）")

    print(f"\n[2/4] 分层划分（val_ratio={args.val_ratio}, seed={args.seed}）...")
    train, val = stratified_split(items, args.val_ratio, args.seed)
    print(f"      train={len(train)}  val={len(val)}")
    for split, subset in (("train", train), ("val", val)):
        img_cls = Counter()
        box_cls = Counter()
        for it in subset:
            img_cls[it["kind"]] += 1
            for c, *_ in it["boxes"]:
                box_cls[NAMES[c]] += 1
        print(f"      {split}: 正常{img_cls['normal']} 缺陷{img_cls['defective']} | "
              f"框 insulator={box_cls['insulator']} defect={box_cls['defect']}")

    print(f"\n[3/4] 写出图片与标签（mode={args.mode}）...")
    if out.exists():
        for sub in ("images", "labels"):
            shutil.rmtree(out / sub, ignore_errors=True)
    train_paths = write_split(train, out, "train", args.mode)
    val_paths = write_split(val, out, "val", args.mode)

    if args.defect_repeat > 1:
        defect_set = {it["stem_out"] for it in train if it["kind"] == "defective"}
        extra = [p for p in train_paths
                 if Path(p).stem in defect_set for _ in range(args.defect_repeat - 1)]
        train_paths += extra
        print(f"      缺陷图过采样 x{args.defect_repeat}：train 条目 {len(train_paths)}")

    (out / "train.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val_paths) + "\n", encoding="utf-8")
    write_data_yaml(out)

    stats = {
        "images_total": len(items),
        "normal_images": n_normal,
        "defective_images": n_defect,
        "train_images": len(train),
        "val_images": len(val),
        "train_entries": len(train_paths),
        "classes": NAMES,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "defect_repeat": args.defect_repeat,
        "boxes": {"insulator": sum(1 for it in items for c, *_ in it["boxes"] if c == 0),
                  "defect": sum(1 for it in items for c, *_ in it["boxes"] if c == 1)},
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[4/4] 完成")
    print(f"      data.yaml : {out / 'data.yaml'}")
    print(f"      统计      : {out / 'stats.json'}")

    if args.check > 0:
        draw_check(out, args.check, args.seed)


if __name__ == "__main__":
    main()
