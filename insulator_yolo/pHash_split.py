# -*- coding: utf-8 -*-
"""T2：近重复图像分组（pHash），核查训练/验证集同源泄漏。

做法
----
1. 对每张图算 64 位 pHash（32x32 灰度 -> DCT -> 左上 8x8 -> 中位数阈值）；
2. 848 张两两比汉明距离，距离 <= 阈值的判为"近重复"；
3. 用并查集把近重复连成"同源组"；
4. 统计跨 train/val 的泄漏：有多少验证图在训练集里有近重复。

输出（默认写 runs/leakage/）
----
hashes.csv          每张图的 pHash（hex）与所属划分
pairs.csv           所有距离 <= 阈值的图对，标注是否跨划分
summary.json        汇总数字，供报告引用

用法
----
    python pHash_split.py                      # 默认阈值 10
    python pHash_split.py --threshold 8        # 更严格
    python pHash_split.py --dist-hist          # 额外输出最小距离分布，用于选阈值
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import HERE, imread_u

DATA = HERE / "datasets" / "cplid"
OUT = HERE / "runs" / "leakage"


def phash(img, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """经典 pHash：DCT 低频 8x8，去直流后按中位数取 0/1。"""
    import cv2

    img_size = hash_size * highfreq_factor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:hash_size, :hash_size].flatten()
    dc, rest = low[0], low[1:]
    bits = rest > np.median(rest)
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def load_split(split: str, data_dir: Path) -> list[Path]:
    txt = data_dir / f"{split}.txt"
    return [Path(p) for p in txt.read_text(encoding="utf-8").split()]


def hamming_matrix(codes: np.ndarray) -> np.ndarray:
    """codes: (N, 64) uint8 的 0/1 位矩阵 -> (N, N) 汉明距离。"""
    bits = codes.astype(np.float32)
    # 距离 = popcount(a xor b) = 64 - 相同位数
    same = bits @ bits.T + (1 - bits) @ (1 - bits).T
    return (64 - same).astype(np.int16)


def group_of(n: int, parent: list[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(parent, i), []).append(i)
    return groups


def find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent: list[int], a: int, b: int) -> None:
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[rb] = ra


def main() -> int:
    ap = argparse.ArgumentParser(description="pHash 近重复分组与训练/验证泄漏核查")
    ap.add_argument("--data", type=Path, default=DATA, help="YOLO 数据集目录")
    ap.add_argument("--out", type=Path, default=OUT, help="输出目录")
    ap.add_argument("--threshold", type=int, default=10, help="汉明距离阈值，<= 判为近重复")
    ap.add_argument("--dist-hist", action="store_true", help="额外输出最近邻距离分布")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    paths, splits = [], []
    for split in ("train", "val"):
        for p in load_split(split, args.data):
            paths.append(p)
            splits.append(split)
    print(f"图片 {len(paths)} 张（train {splits.count('train')} / val {splits.count('val')}）")

    codes = []
    for i, p in enumerate(paths):
        img = imread_u(p)
        if img is None:
            raise SystemExit(f"读不到图片：{p}")
        codes.append([int(b) for b in format(phash(img), "064b")])
        if (i + 1) % 200 == 0:
            print(f"  已计算 {i + 1}/{len(paths)}")
    codes_arr = np.array(codes, dtype=np.uint8)
    values = [int("".join(map(str, c)), 2) for c in codes_arr]

    with open(args.out / "hashes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "phash_hex"])
        for p, s, v in zip(paths, splits, values):
            w.writerow([str(p), s, f"{v:016x}"])

    print("计算两两距离 ...")
    dist = hamming_matrix(codes_arr)
    n = len(paths)
    split_arr = np.array(splits)

    pairs = []
    ii, jj = np.triu_indices(n, k=1)
    mask = dist[ii, jj] <= args.threshold
    for a, b in zip(ii[mask], jj[mask]):
        pairs.append((int(a), int(b), int(dist[a, b])))
    print(f"  近重复图对：{len(pairs)}（阈值 {args.threshold}）")

    parent = list(range(n))
    for a, b, _ in pairs:
        union(parent, a, b)
    groups = group_of(n, parent)

    cross = [(a, b, d) for a, b, d in pairs if split_arr[a] != split_arr[b]]
    leaked_val = sorted({b if split_arr[b] == "val" else a for a, b, _ in cross})
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}

    with open(args.out / "pairs.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["img_a", "split_a", "img_b", "split_b", "hamming", "cross_split"])
        for a, b, d in sorted(pairs, key=lambda x: x[2]):
            w.writerow([paths[a].name, split_arr[a], paths[b].name, split_arr[b], d,
                        "YES" if split_arr[a] != split_arr[b] else ""])

    summary = {
        "images": n,
        "train": int((split_arr == "train").sum()),
        "val": int((split_arr == "val").sum()),
        "threshold": args.threshold,
        "duplicate_pairs": len(pairs),
        "cross_split_pairs": len(cross),
        "leaked_val_images": len(leaked_val),
        "leaked_val_ratio": round(len(leaked_val) / max(int((split_arr == "val").sum()), 1), 4),
        "groups_total": len(groups),
        "groups_with_multiple": len(dup_groups),
        "largest_group": max((len(v) for v in groups.values()), default=1),
        "group_size_hist": dict(Counter(len(v) for v in groups.values()).most_common()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dist_hist:
        off = dist[~np.eye(n, dtype=bool)]
        nn = off.reshape(n, n - 1).min(axis=1)
        hist = {int(k): int(v) for k, v in zip(*np.unique(nn, return_counts=True))}
        (args.out / "nearest_dist.json").write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
        print("最近邻距离分布：", hist)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())