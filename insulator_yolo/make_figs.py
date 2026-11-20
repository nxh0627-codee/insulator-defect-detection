# -*- coding: utf-8 -*-
"""T14：把训练/评估/核查产物整理成带编号的图表包 figs/。

原则
----
1. 图号连续、文件名可读，报告里直接引用 "见图 fig02"；
2. 能拷贝的不重画（保持与训练产物一致），只有需要新信息的才现场生成；
3. 自动写 figs/INDEX.md：每张图的编号、内容、来源路径、建议用在报告哪一章。

用法
----
    python make_figs.py                 # 输出到仓库根目录 figs/
    python make_figs.py --out D:\figs   # 自定义输出目录
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from common import HERE

Runs = HERE / "runs"
DATASET = HERE / "datasets" / "cplid"

# (编号, 输出名, 内容说明, 来源, 报告用途)
COPY_ITEMS = [
    ("fig02", "pr_curve.png",
     "各类别 P-R 曲线（基线，8 轮 / 320）",
     Runs / "detect" / "cplid_yolov8n_cpu" / "BoxPR_curve.png",
     "报告「实验结果」：整体检测性能"),
    ("fig03", "f1_curve.png",
     "置信度-F1 曲线，用于说明工作点与 F1 的关系",
     Runs / "detect" / "cplid_yolov8n_cpu" / "BoxF1_curve.png",
     "报告「实验结果」：阈值选择依据"),
    ("fig04", "confusion_matrix.png",
     "混淆矩阵（绝对计数）",
     Runs / "detect" / "eval_cplid_yolov8n_cpu" / "confusion_matrix.png",
     "报告「实验结果」：类别间混淆"),
    ("fig05", "confusion_matrix_normalized.png",
     "归一化混淆矩阵",
     Runs / "detect" / "eval_cplid_yolov8n_cpu" / "confusion_matrix_normalized.png",
     "报告「实验结果」：漏检 / 误检比例"),
    ("fig06", "conf_sweep.png",
     "置信度阈值扫描：图级检出率 / 误报率 / 每图框数",
     Runs / "eval" / "conf_sweep" / "sweep.png",
     "报告「业务口径」：三档工作点"),
    ("fig08", "sample_defect.jpg",
     "缺陷样本标注示例（绿框绝缘子、红框缺陷）",
     DATASET / "_check" / "d_157.jpg",
     "报告「数据」：标注质量与合成缺陷说明"),
]


def fig_target_size(out: Path) -> Path:
    """图 fig01：目标尺寸分布（insulator vs defect 的框边长）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sides = {0: [], 1: []}
    for split in ("train", "val"):
        for lab in (DATASET / "labels" / split).glob("*.txt"):
            for line in lab.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                c, _, _, w, h = line.split()
                # 归一化边长 -> 像素（图像 1152x864）
                sides[int(c)].append((float(w) * 1152 + float(h) * 864) / 2)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    bins = np.logspace(np.log10(10), np.log10(1000), 40)
    ax.hist(sides[0], bins=bins, alpha=0.65, label=f"insulator (n={len(sides[0])})", color="#4c9f70")
    ax.hist(sides[1], bins=bins, alpha=0.75, label=f"defect (n={len(sides[1])})", color="#c0504d")
    ax.set_xscale("log")
    ax.set_xlabel("box side length (px, log scale)")
    ax.set_ylabel("count")
    ax.set_title("Target size distribution (CPLID, 1152x864)")
    ax.axvline(np.median(sides[0]), ls="--", lw=1, color="#2e6b4a")
    ax.axvline(np.median(sides[1]), ls="--", lw=1, color="#8b2e2b")
    ax.legend()
    ax.grid(alpha=0.25)
    p = out / "fig01_target_size.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_leakage(out: Path) -> Path:
    """图 fig07：最近邻汉明距离分布（泄漏核查的阈值选择依据）。"""
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = Runs / "leakage" / "nearest_dist.json"
    hist = {int(k): int(v) for k, v in json.loads(src.read_text(encoding="utf-8")).items()}
    xs = sorted(hist)
    ys = [hist[x] for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.bar(xs, ys, width=1.4, color="#4472c4")
    ax.axvline(12, ls="--", color="#c00000", lw=1.2)
    ax.annotate("threshold = 12", xy=(12, max(ys) * 0.85), xytext=(13.2, max(ys) * 0.85),
                color="#c00000", fontsize=9)
    ax.set_xlabel("hamming distance to nearest neighbour")
    ax.set_ylabel("number of images")
    ax.set_title("pHash nearest-neighbour distance (848 images)")
    ax.grid(alpha=0.25, axis="y")
    p = out / "fig07_leakage_nearest_dist.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="生成报告用图表包 figs/")
    ap.add_argument("--out", type=Path, default=HERE.parent / "figs")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.append(("fig01", "target_size.png", "目标尺寸分布：insulator 与 defect 的框边长（对数轴）",
                 "make_figs.py 现场绘制（源：datasets/cplid/labels）", "报告「数据」：小目标依据"))
    fig_target_size(out)

    for code, name, desc, src, use in COPY_ITEMS:
        if not src.exists():
            print(f"  [跳过] {code} {name}：源文件不存在 {src}")
            continue
        shutil.copy2(src, out / f"{code}_{name}")
        rows.append((code, name, desc, str(src.relative_to(HERE.parent)), use))
        print(f"  [拷贝] {code}_{name}")

    if (Runs / "leakage" / "nearest_dist.json").exists():
        fig_leakage(out)
        rows.append(("fig07", "leakage_nearest_dist.png", "pHash 最近邻汉明距离分布（阈值 12 的依据）",
                     "make_figs.py 现场绘制（源：runs/leakage/nearest_dist.json）", "报告「数据」：泄漏核查"))

    rows.sort(key=lambda r: r[0])
    lines = ["# 图表包索引（figs/）", "",
             "> 由 `python make_figs.py` 生成。图号在报告正文里直接引用（如「见图 fig02」）。", "",
             "| 编号 | 文件 | 内容 | 来源 | 建议用在 |", "|---|---|---|---|---|"]
    for code, name, desc, src, use in rows:
        lines.append(f"| {code} | `{code}_{name}` | {desc} | `{src}` | {use} |")
    lines += ["", "## 重新生成", "", "```powershell", "cd insulator_yolo", "python make_figs.py", "```", ""]
    (out / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n图表包完成：{out}（{len(rows)} 张图 + INDEX.md）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())