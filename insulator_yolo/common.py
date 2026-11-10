# -*- coding: utf-8 -*-
"""公共工具：环境准备、中文路径安全的图像读写、训练预设。

所有脚本都从这里导入，保证行为一致：
    - 把 ultralytics 的配置目录固定到工程内（避免写到系统盘/临时目录）
    - 提供 imread_u / imwrite_u，绕开 OpenCV 在 Windows 上不支持中文路径的问题
    - 提供两套训练预设 cpu / full，供 train.py 直接取用
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 环境
def init_env():
    """固定 ultralytics 的配置目录到工程内，并顺手关掉不必要的联网检查。"""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / ".ultralytics"))
    (HERE / ".ultralytics").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 中文路径 IO
def imread_u(path, flags=None):
    """读取图片，支持中文/空格路径；失败返回 None。"""
    import cv2
    import numpy as np

    flags = cv2.IMREAD_COLOR if flags is None else flags
    p = str(path)
    try:
        data = np.fromfile(p, dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img


def imwrite_u(path, img) -> bool:
    """写出图片，支持中文/空格路径。"""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def imshow_safe_name(path) -> str:
    """把路径转成只含 ASCII 的文件名，用于给不支持中文的输出工具兜底。"""
    return Path(path).name


# ---------------------------------------------------------------- 训练预设
# cpu : 这台机器（无 GPU）上几分钟能跑完的小规模配置
# full: 开题报告里的正式配置（需要 GPU）
PROFILES = {
    "cpu": dict(epochs=30, imgsz=320, batch=8, workers=0),
    "full": dict(epochs=150, imgsz=640, batch=16, workers=4),
}


def print_env(silent: bool = False):
    """打印关键环境信息，方便复现问题。"""
    import torch
    import torchvision

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_count": os.cpu_count(),
        "threads": torch.get_num_threads(),
    }
    if not silent:
        print("环境: " + "  ".join(f"{k}={v}" for k, v in info.items()))
    return info


def set_cpu_threads(ratio: float = 0.75):
    """CPU 训练时把 torch 线程数调满一点，默认用 75% 的核。"""
    import torch

    n = max(1, int((os.cpu_count() or 4) * ratio))
    torch.set_num_threads(n)
    return n
