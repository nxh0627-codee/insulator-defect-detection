# -*- coding: utf-8 -*-
"""
YOLOv8 训练入口（CPLID 绝缘子缺陷检测）。

默认从 COCO 预训练权重 yolov8n.pt 做迁移学习；若本机没有该权重且无法联网下载，
会自动退化为「从随机初始化训练」（用包内的 yolov8n.yaml 结构），不会中途崩掉。

用法：
    python train.py                          # 默认 CPU 小规模预设（基线：原 BCE 损失）
    python train.py --profile full           # 开题报告里的正式配置（需 GPU）
    python train.py --epochs 50 --imgsz 480  # 手动覆盖单项
    python train.py --resume                 # 断点续训

损失层开关（消融实验用，见 improve.py）：
    --focal-gamma 1.5      启用 Focal Loss（0 = 关闭）
    --focal-alpha 0.25     Focal 的正样本权重（仅 gamma>0 生效）
    --defect-weight 2.0    缺陷类 cls 损失权重（1.0 = 不加权）
    --cache [ram|disk]     图片缓存加速；不带值时默认 ram

例：
    python train.py --focal-gamma 1.5 --focal-alpha 0.25 --defect-weight 2.0 \
                    --cache ram --name D_focal_w2

训练产物：
    runs/detect/<name>/weights/best.pt, last.pt
    runs/detect/<name>/results.csv, results.png, confusion_matrix.png, args.yaml
    runs/detect/<name>/improve.json   # 损失层配置（args.yaml 里没有这几个自定义参数）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from common import HERE, PROFILES, init_env, print_env, set_cpu_threads

init_env()  # 必须在 import ultralytics 之前

# 允许用 --offline 关闭 ultralytics 的联网检查（无网络环境可避免等待超时）
if "--offline" in sys.argv:
    __import__("os").environ["YOLO_OFFLINE"] = "1"

DEFAULT_DATA = HERE / "datasets" / "cplid" / "data.yaml"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="训练 CPLID 绝缘子缺陷检测模型（YOLOv8n）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="数据集配置 data.yaml")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="权重文件；给 .yaml 则从零训练（如 yolov8n.yaml）")
    ap.add_argument("--profile", choices=list(PROFILES), default="cpu",
                    help="cpu=小规模快速跑通；full=正式配置（需 GPU）")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None, help="数据加载进程数，Windows 建议 0")
    ap.add_argument("--device", default=None, help="cpu / 0 / 0,1 ...")
    ap.add_argument("--patience", type=int, default=30, help="早停：连续 N 轮无提升则停止")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="本次运行目录名")
    ap.add_argument("--project", type=Path, default=HERE / "runs" / "detect")
    ap.add_argument("--cache", nargs="?", const="ram", default=None, choices=["ram", "disk"],
                    help="图片缓存加速：--cache / --cache ram 缓存到内存；--cache disk 缓存到磁盘")
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help="Focal Loss 的 gamma（>0 启用；0 = 关闭，沿用原 BCE）")
    ap.add_argument("--focal-alpha", type=float, default=0.25,
                    help="Focal Loss 的 alpha（仅 gamma>0 时生效）")
    ap.add_argument("--defect-weight", type=float, default=1.0,
                    help="缺陷类（最后一个类别）的 cls 损失权重；1.0 = 不加权")
    ap.add_argument("--plots", action="store_true", help="额外保存训练曲线/预测样例图")
    ap.add_argument("--resume", action="store_true", help="从上次中断处继续训练")
    ap.add_argument("--offline", action="store_true",
                    help="禁用联网检查（无网络环境用，避免等待超时）")
    ap.add_argument("--threads", type=float, default=0.75,
                    help="CPU 训练的 torch 线程占核比例")
    args = ap.parse_args(argv)

    # 未显式指定的参数用预设补齐
    for key, val in PROFILES[args.profile].items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    if args.device is None:
        import torch
        args.device = "0" if torch.cuda.is_available() else "cpu"

    if args.name is None:
        args.name = f"cplid_yolov8n_{args.profile}"

    return args


def load_model(weights: str):
    """加载模型：预训练权重优先，拿不到就退化为从零训练。"""
    from ultralytics import YOLO

    try:
        model = YOLO(weights)
        print(f"[模型] 已加载 {weights}")
        return model, weights
    except Exception as e:  # 离线、下载失败、权重不存在等
        if not weights.lower().endswith(".pt"):
            raise
        fallback = weights.replace(".pt", ".yaml")
        print(f"[警告] 无法加载 {weights}：{type(e).__name__}: {e}")
        print(f"[警告] 退化为从零训练：{fallback}（没有 COCO 预训练，收敛更慢、精度更低）")
        return YOLO(fallback), fallback


def _apply_improve(model, args):
    """按命令行开关给损失打补丁（Focal + 类别权重），返回落盘用的配置。"""
    cfg = {
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "defect_weight": args.defect_weight,
        "cache": args.cache,
        "enabled": False,
    }
    if args.focal_gamma <= 0 and args.defect_weight == 1.0:
        print("[改进] 未启用（focal_gamma=0 且 defect_weight=1.0），沿用原 BCE 损失")
        return cfg

    from improve import patch_loss

    # 约定：类别 0 = insulator（正常），最后一个类别 = defect；
    # 真实 nc 由 trainer 用数据集决定，所以这里只传"缺陷类倍数"，由损失类按 nc 展开
    cfg.update(patch_loss(model.model, focal_gamma=args.focal_gamma,
                          focal_alpha=args.focal_alpha, defect_weight=args.defect_weight))
    return cfg


def main(argv=None):
    args = parse_args(argv)

    if not args.data.exists():
        sys.exit(f"[错误] 找不到数据集配置：{args.data}\n"
                 f"       请先运行：python prepare_dataset.py")

    print("=" * 70)
    print("CPLID 绝缘子缺陷检测 - 训练开始")
    print("=" * 70)
    env = print_env()
    if env["device"] == "cpu":
        n = set_cpu_threads(args.threads)
        print(f"[CPU] torch 线程数设为 {n}；无 GPU 时完整 150 epoch 会很慢，"
              f"建议先用 --profile cpu 跑通流程")
    print(f"[配置] profile={args.profile} model={args.model} "
          f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} workers={args.workers}")
    print(f"[数据] {args.data}")

    model, used_weights = load_model(args.model)
    improve_cfg = _apply_improve(model, args)
    print(f"[改进] config={improve_cfg}")

    t0 = time.time()
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        cache=False if args.cache is None else args.cache,
        plots=args.plots,
        resume=args.resume,
        verbose=True,
    )
    dt = time.time() - t0

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"

    # 把损失层配置落到运行目录，args.yaml 里没有这几个自定义参数，复现时看 improve.json
    import json
    improve_cfg["weights"] = str(used_weights)
    improve_cfg["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (save_dir / "improve.json").write_text(
        json.dumps(improve_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"训练结束，用时 {dt / 60:.1f} 分钟")
    print(f"损失配置：{improve_cfg}")
    print(f"权重：{best if best.exists() else save_dir / 'weights' / 'last.pt'}")
    print(f"结果：{save_dir / 'results.csv'}")
    print("=" * 70)

    # 训练结束后直接在验证集上评估一次，给出关键指标
    try:
        metrics = model.val(data=str(args.data), split="val", imgsz=args.imgsz,
                            batch=args.batch, device=args.device, workers=0, plots=False)
        print(f"[指标] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}  "
              f"precision={metrics.box.mp:.4f}  recall={metrics.box.mr:.4f}")
    except Exception as e:
        print(f"[提示] 训练后自动评估失败（不影响权重）：{type(e).__name__}: {e}")

    return results


if __name__ == "__main__":
    main()
