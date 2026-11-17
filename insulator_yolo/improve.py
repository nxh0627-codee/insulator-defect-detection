# -*- coding: utf-8 -*-
"""损失层改进：Focal Loss + 类别权重（运行时打补丁，不改第三方库源码）。

为什么要改
----------
CPLID 里缺陷图 248 张、正常图 600 张，一张图里背景锚点又远多于目标框，BCE 的梯度
会被大量"容易的负样本"淹没。基线正是这个症状：defect Recall 0.680、框级 defect P
0.500，但 IoU>=0.5 命中率有 84%（conf_sweep 观察 3）——模型"看得见"，只是置信度被压低。

两处改动，都只作用在**分类分支**，不碰 box / dfl，因此与基线严格可比：

1. Focal Loss：loss = alpha_t * (1 - p_t)^gamma * BCE，压低易样本、聚焦难样本；
2. 类别权重：cls 分支按类别乘权重（缺陷类 x2.0），直接抬高缺陷类正样本的梯度。

与 ultralytics 8.4 的关系
-------------------------
8.4 的 `v8DetectionLoss` 已经原生支持类别权重：它读 `model.class_weights`，
在 cls 分支 `bce_loss *= self.class_weights`（loss.py 第 378、466 行）。
但它**没有**把 Focal 接进检测损失（自带的 `FocalLoss` 返回 `.mean(1).sum()`，
和这里 `(bs, anchors, nc)` 的逐元素再求和口径不一致，不能直接用）。

所以本模块：
- 子类化 `v8DetectionLoss`，只把 `self.bce` 换成逐元素的 FocalBCE；
- 类别权重仍走父类原生的乘法，不在 FocalBCE 里重复加权（否则权重被平方）；
- 通过全局配置注入参数，而不是挂在模型对象上——因为 `YOLO.train()` 内部会用
  `trainer.get_model()` **重建一个新模型**（trainer.py 第 1412 行），模型上的
  属性会丢，全局配置不会。

用法
----
train.py 会自动调用 `patch_loss()`；也可以单独自检（不需要数据集和权重）：

    python improve.py
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FocalBCE", "patch_loss", "current_config", "self_test"]

# 全局配置：patch_loss 写入，ImprovedDetectionLoss 读取（不依赖模型对象存活）
_CONFIG: dict = {}
_PATCHED = False


class FocalBCE(nn.Module):
    """逐元素的 BCEWithLogits + Focal（可选独立类别权重）。

    参数
    ----
    gamma : 聚焦指数；<=0 时不启用 Focal，退化为纯 BCE
    alpha : 正样本权重；仅在 gamma>0 时生效，保证 gamma=0 时与 BCE 数值完全一致
    class_weights : 可选，长度 = 类别数。仅用于独立调用/自检；
                    接入 v8DetectionLoss 时**不要**传，否则会和父类的原生加权叠加
    reduction : "none"（默认，与 ultralytics 口径一致）/ "mean" / "sum"

    形状约定：pred / target 都是 [..., C]，C = num_classes。
    """

    def __init__(self, gamma: float = 0.0, alpha: float = 0.25,
                 class_weights=None, reduction: str = "none"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction
        w = None
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32).flatten()
        self.register_buffer("class_weights", w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        if self.gamma > 0:
            p = pred.sigmoid()
            p_t = target * p + (1.0 - target) * (1.0 - p)
            loss = loss * (1.0 - p_t).pow(self.gamma)
            if self.alpha > 0:
                alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
                loss = loss * alpha_t

        if self.class_weights is not None:
            w = self.class_weights.to(device=loss.device, dtype=loss.dtype)
            if w.numel() != loss.shape[-1]:
                raise ValueError(
                    f"class_weights 长度 {w.numel()} 与类别数 {loss.shape[-1]} 不匹配")
            loss = loss * w.view(*([1] * (loss.dim() - 1)), -1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        w = None if self.class_weights is None else [round(v, 4) for v in self.class_weights.tolist()]
        return f"gamma={self.gamma}, alpha={self.alpha}, class_weights={w}"


def _build_loss_class():
    """构造 ImprovedDetectionLoss 类（延迟导入 ultralytics，避免 import 期依赖）。"""
    from ultralytics.utils.loss import v8DetectionLoss

    class ImprovedDetectionLoss(v8DetectionLoss):
        """只替换分类分支的 BCE；box / dfl 完全沿用父类。"""

        def __init__(self, model):
            super().__init__(model)
            head = model.model[-1]
            if getattr(head, "one2one_cv2", None) is not None:
                print("[改进][警告] 检测到 one2one 分支（E2E/v10 头），本次补丁只作用于一对一主损失")

            cfg = _CONFIG or {}
            gamma = float(cfg.get("focal_gamma") or 0.0)
            alpha = float(cfg.get("focal_alpha") or 0.25)
            weights = cfg.get("class_weights")
            if weights is None and cfg.get("defect_weight") is not None:
                # 在 criterion 创建时才知道真实 nc（YOLO(yaml) 的 nc 可能还是 80，
                # trainer 会用数据集重建模型），所以这里按 nc 现场展开
                weights = [1.0] * (self.nc - 1) + [float(cfg["defect_weight"])]

            # 类别权重：父类已经按 (1,1,nc) 的形状在 cls 分支上乘，这里只需把向量准备好
            if weights is not None:
                w = torch.as_tensor(weights, dtype=torch.float32).flatten()
                if w.numel() != self.nc:
                    raise ValueError(
                        f"class_weights 有 {w.numel()} 个，但模型 nc={self.nc}；"
                        f"请按数据集类别数给权重（顺序与 data.yaml 的 names 一致）")
                self.class_weights = w.to(self.device).view(1, 1, -1)

            # Focal：只替换 bce 本身，不加权（加权交给父类，避免平方）
            if gamma > 0:
                self.bce = FocalBCE(gamma=gamma, alpha=alpha)

    return ImprovedDetectionLoss


def patch_loss(model=None, focal_gamma: float = 0.0, focal_alpha: float = 0.25,
               class_weights=None, defect_weight=None, verbose: bool = True) -> dict:
    """给检测模型打损失补丁，返回实际生效的配置（供 train.py 落盘）。

    参数
    ----
    model : YOLO(...).model，仅用于校验类别数；训练时模型会被 trainer 重建，
            所以配置存在模块全局里
    focal_gamma : Focal 的 gamma，0 表示关闭
    focal_alpha : Focal 的 alpha，仅 gamma>0 时生效
    class_weights : 类别权重列表，如 [1.0, 2.0]；None 表示不加权
    """
    global _PATCHED, _CONFIG

    cfg = {
        "focal_gamma": float(focal_gamma or 0.0),
        "focal_alpha": float(focal_alpha if focal_alpha is not None else 0.25),
        "class_weights": None if class_weights is None else [float(x) for x in class_weights],
        "defect_weight": None if defect_weight is None else float(defect_weight),
    }
    cfg["enabled"] = bool(cfg["focal_gamma"] > 0
                         or cfg["class_weights"] is not None
                         or (cfg["defect_weight"] is not None and cfg["defect_weight"] != 1.0))

    if not cfg["enabled"]:
        _CONFIG = cfg
        if verbose:
            print("[改进] 未启用（focal_gamma=0 且 class_weights=None），沿用原 BCE 损失")
        return cfg

    # 维度校验：显式权重在 patch 时就能查；defect_weight 要等 criterion 拿到真实 nc
    if model is not None and cfg["class_weights"] is not None:
        nc = int(getattr(model, "nc", 0) or 0)
        w = cfg["class_weights"]
        if nc and len(w) != nc:
            raise ValueError(f"class_weights 长度 {len(w)} 与模型 nc={nc} 不一致")

    _CONFIG = cfg
    if not _PATCHED:
        from ultralytics.nn.tasks import DetectionModel

        loss_cls = _build_loss_class()
        DetectionModel.init_criterion = lambda self: loss_cls(self)  # noqa: E731
        _PATCHED = True

    if verbose:
        w = cfg["class_weights"] if cfg["class_weights"] is not None else f"defect x{cfg['defect_weight']}"
        print(f"[改进] 已启用损失补丁：focal_gamma={cfg['focal_gamma']} "
              f"focal_alpha={cfg['focal_alpha']} class_weights={w}")
    return cfg


def current_config() -> dict:
    """返回当前生效的配置（没打过补丁则为空 dict）。"""
    return dict(_CONFIG)


# ------------------------------------------------------------------ 自检
def self_test() -> int:
    """张量级自检：四条不变式，不需要数据集、不需要权重。"""
    torch.manual_seed(0)
    ok = True

    def check(name: str, cond: bool, detail: str = ""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
        ok = ok and cond

    print("1) gamma=0 时必须与纯 BCE 数值一致（保证基线可比）")
    pred = torch.randn(2, 5, 2, requires_grad=True)
    tgt = (torch.rand(2, 5, 2) > 0.7).float()
    ref = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
    got = FocalBCE(gamma=0.0, alpha=0.25)(pred, tgt)
    diff = (ref - got).abs().max().item()
    check("gamma=0 == BCE", diff < 1e-6, f"max|diff|={diff:.3e}")
    check("输出形状与 BCE 一致", tuple(got.shape) == tuple(ref.shape), f"{tuple(got.shape)}")

    print("2) Focal 必须压低易样本、保留难样本的损失")
    easy_pred = torch.tensor([[6.0, -6.0]])
    hard_pred = torch.tensor([[0.2, -0.2]])
    t = torch.tensor([[1.0, 0.0]])
    bce_easy = F.binary_cross_entropy_with_logits(easy_pred, t, reduction="none").sum().item()
    bce_hard = F.binary_cross_entropy_with_logits(hard_pred, t, reduction="none").sum().item()
    f_easy = FocalBCE(gamma=1.5, alpha=0.0)(easy_pred, t).sum().item()
    f_hard = FocalBCE(gamma=1.5, alpha=0.0)(hard_pred, t).sum().item()
    r_easy = f_easy / max(bce_easy, 1e-12)
    r_hard = f_hard / max(bce_hard, 1e-12)
    check("易样本被压低更多", r_easy < r_hard * 0.2, f"easy→{r_easy:.4f}x, hard→{r_hard:.4f}x")

    print("3) 独立调用时，类别权重应按倍数放大对应类别的损失")
    w1 = FocalBCE(gamma=0.0, class_weights=[1.0, 1.0])(pred, tgt)
    w2 = FocalBCE(gamma=0.0, class_weights=[1.0, 2.0])(pred, tgt)
    mask = tgt[..., 1] > 0
    ratio = (w2[..., 1][mask] / w1[..., 1][mask].clamp_min(1e-12)).mean().item()
    check("缺陷类权重 x2.0 生效", abs(ratio - 2.0) < 1e-5, f"实测倍数={ratio:.4f}")
    untouched = (w2[..., 0] - w1[..., 0]).abs().max().item()
    check("正常类权重不变", untouched < 1e-6, f"max|diff|={untouched:.3e}")

    print("4) 长度不匹配必须报错，而不是静默算错")
    try:
        FocalBCE(gamma=1.0, class_weights=[1.0, 2.0, 3.0])(pred, tgt)
        check("nc 不匹配时报错", False)
    except ValueError as e:
        check("nc 不匹配时报错", True, str(e)[:48])

    print("\n结果：" + ("全部通过" if ok else "有失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())