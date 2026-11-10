# 基于深度学习的输电线路绝缘子缺陷检测系统

CPLID 数据集 + YOLOv8 的绝缘子缺陷检测：从数据转换、训练、评估到推理与阈值工作点选择的一条完整链路。

- **任务**：目标检测 —— 先定位绝缘子，再判定有无缺陷（`insulator` / `defect` 两类）
- **模型**：YOLOv8n，COCO 预训练权重迁移学习
- **数据**：CPLID（Chinese Power Line Insulator Dataset），848 张无人机航拍图（正常 600 / 缺陷 248），全部 1152×864
- **指标**：mAP@0.5、mAP@0.5:0.95、**图级检出率 / 误报率**（贴近业务口径）

## 当前结果（基线：8 轮 / 320×320 / CPU）

| 类别 | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| 全部 | 0.806 | 0.816 | **0.871** | 0.547 |
| insulator | 0.902 | 0.952 | 0.963 | 0.772 |
| defect | 0.711 | 0.680 | 0.779 | 0.322 |

在验证集 170 张（缺陷 50 / 正常 120）上的业务口径表现，随置信度阈值变化很大：

| 用途 | conf | 图级检出率 | 误报率 | 每图框数 |
|---|---|---|---|---|
| 筛查（推荐） | 0.05 | **96.0%** | 2.5% | 2.58 |
| 保守（误报 ≤1%） | 0.07 | 88.0% | 0.8% | 2.30 |
| 免复核自动告警 | 0.35 | 54.0% | 0.0% | 1.72 |

> 详见 [`insulator_yolo/runs/eval/conf_sweep/REPORT.md`](insulator_yolo/runs/eval/conf_sweep/REPORT.md)。
> 结论：默认 0.25 对"漏检代价高"的巡检场景偏保守，图级检出率只有 54%；降到 0.05 可到 96%。

## 目录结构

```
.
├── insulator_yolo/            # 全部代码与训练产物
│   ├── common.py              # 环境预设 + 中文路径安全的图像读写
│   ├── prepare_dataset.py     # VOC(XML) → YOLO(txt)，分层划分 + 标注自检
│   ├── train.py               # 训练入口（cpu / full 两套预设）
│   ├── evaluate.py            # 验证集评估，整体 + 逐类指标
│   ├── predict.py             # 推理，输出标注图 + 逐框明细 + 逐图汇总
│   ├── conf_sweep.py          # 置信度阈值扫描（图级检出率 / 误报率 / 工作点）
│   ├── README.md              # 工程细节、参数说明、常见问题
│   └── runs/                  # 训练与评估产物（权重不入库）
├── 开题报告-*.pptx
├── 输电线路绝缘子缺陷检测-项目文档.docx
└── 项目下一步计划.md          # 阶段路线与实验排期
```

## 快速开始

```powershell
cd insulator_yolo
pip install -r requirements.txt

# 1) 数据转换（把 CPLID 解压到 ../InsulatorDataSet-master 后执行）
python prepare_dataset.py --check 8

# 2) 训练（CPU 小规模预设：30 轮 / 320）
python train.py

# 3) 评估
python evaluate.py --weights runs/detect/cplid_yolov8n_cpu/weights/best.pt

# 4) 推理（建议用 0.05 的筛查工作点）
python predict.py --weights runs/detect/cplid_yolov8n_cpu/weights/best.pt --conf 0.05

# 5) 阈值工作点扫描
python conf_sweep.py --confs "0.02,0.05,0.07,0.1,0.15,0.25,0.35"
```

## 未入库的内容

| 内容 | 原因 | 如何获取 |
|---|---|---|
| `InsulatorDataSet-master/`、`insulator_yolo/datasets/` | 约 400–500 MB 图像 | 下载 CPLID 后用 `prepare_dataset.py` 生成 |
| `insulator_yolo/runs/**/weights/*.pt` | 每个 6 MB | 由 `train.py` 产生 |
| `insulator_yolo/runs/predict/` | 推理标注图 | 由 `predict.py` 产生 |

评估汇总（`metrics.json`、`results.csv`、混淆矩阵、P-R 曲线、阈值扫描结果）都保留在库里。

## 说明

- 数据集 CPLID 仅提供"正常 / 缺陷"二级标注，因此本工程按**二分类检测**实现；"破损 / 自爆 / 闪络"的细分类需自有标注数据，列为后续工作。
- 当前机器无 NVIDIA GPU，训练在 CPU 上完成；`train.py --profile full`（150 轮 / 640）需 GPU 资源。
