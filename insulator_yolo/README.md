# 输电线路绝缘子缺陷检测（YOLOv8 / CPLID）

基于 CPLID（Chinese Power Line Insulator Dataset）训练一个绝缘子缺陷检测模型，
对应开题报告中的「基线模型 YOLOv8 + 迁移学习」方案。

- 数据集：848 张无人机航拍图（正常 600 张 / 缺陷 248 张），全部 1152×864
- 类别：`insulator`（绝缘子本体，1321 个框）、`defect`（缺陷，248 个框）
- 模型：YOLOv8n（最轻量的一档，CPU 也能跑）
- 指标：mAP@0.5、mAP@0.5:0.95、缺陷类 Recall

---

## 一、目录结构

```
insulator_yolo/
├── prepare_dataset.py    # VOC(XML) -> YOLO(txt)，分层划分 + 标注自检图
├── train.py              # 训练入口（支持 cpu / full 两套预设）
├── evaluate.py           # 验证集评估，输出整体与逐类指标
├── predict.py            # 推理，输出可视化图 + 检测明细 + 汇总统计
├── common.py             # 公共工具（中文路径读写、环境预设）
├── requirements.txt
├── datasets/cplid/       # 由 prepare_dataset.py 生成
│   ├── images/{train,val}/*.jpg
│   ├── labels/{train,val}/*.txt
│   ├── data.yaml
│   └── stats.json
└── runs/
    ├── detect/<name>/    # 训练产物：weights/best.pt、results.csv、混淆矩阵…
    └── predict/<name>/   # 推理产物：标注图、detections.csv、summary.csv
```

## 二、环境

本机已装好（Python 3.13 + torch 2.14 CPU 版 + ultralytics 8.4）。换机器时：

```powershell
pip install -r requirements.txt
```

> 无 GPU 也能跑，只是慢。检查是否识别到显卡：
> `python -c "import torch; print(torch.cuda.is_available())"`

## 三、四步跑起来

在 `insulator_yolo` 目录下执行：

```powershell
# 1) 数据转换（只需执行一次）
python prepare_dataset.py

# 2) 训练（CPU 小规模预设：30 epoch / 320 尺寸）
python train.py

# 3) 评估
python evaluate.py --weights runs/detect/cplid_yolov8n_cpu/weights/best.pt

# 4) 推理（默认跑验证集，也可以指到自己的巡检图目录）
python predict.py --weights runs/detect/cplid_yolov8n_cpu/weights/best.pt
```

有 GPU 时直接换正式配置：

```powershell
python train.py --profile full            # 150 epoch / 640 尺寸 / batch 16
```

## 四、数据转换

```powershell
python prepare_dataset.py                      # 默认 8:2 分层划分，seed=42
python prepare_dataset.py --check 8            # 另外生成 8 张标注可视化到 _check/
python prepare_dataset.py --defect-repeat 2    # 训练集缺陷图重复 2 倍，缓解类别不均衡
python prepare_dataset.py --mode hardlink      # 同盘硬链接，不额外占空间
```

原始标注是 VOC 格式，且缺陷图被拆在 `labels/insulator/` 与 `labels/defect/`
两个目录里；脚本会把它们合并成一份 YOLO 标签，并把两类框一起写进同一个 txt。

划分是**按 正常/缺陷 分层**的，避免验证集里缺陷样本过少导致指标抖动。
文件名加了前缀（`n_` 正常 / `d_` 缺陷）防止两个子集重名。

## 五、训练

```powershell
python train.py                              # cpu 预设
python train.py --profile full               # 正式预设
python train.py --epochs 50 --imgsz 480      # 单项覆盖
python train.py --cache                      # 图片缓存进内存，加速后续 epoch
python train.py --resume                     # 断点续训
python train.py --offline                    # 无网络环境，跳过联网检查
```

| 参数 | 说明 | cpu 预设 | full 预设 |
|---|---|---|---|
| `--epochs` | 训练轮数 | 30 | 150 |
| `--imgsz` | 输入尺寸 | 320 | 640 |
| `--batch` | 批大小 | 8 | 16 |
| `--workers` | 数据加载进程 | 0（Windows 必须） | 4 |
| `--patience` | 早停耐心值 | 30 | 30 |

说明：

- **迁移学习**：默认加载 `yolov8n.pt`（COCO 预训练）。若本机没有该权重且无法联网，
  脚本会打印警告并自动退化为从零训练（`yolov8n.yaml`），不会中断。
- **类别不均衡**：缺陷图只占 29%。基线先原样训练；要按开题报告做「缺陷类过采样」，
  先 `python prepare_dataset.py --defect-repeat 2`，再训练。两组结果正好做消融对比。
- **CPU 线程**：脚本默认把 torch 线程设为核数的 75%，可用 `--threads 1.0` 调满。

## 六、评估

```powershell
python evaluate.py --weights runs/detect/cplid_yolov8n_cpu/weights/best.pt
python evaluate.py --weights best.pt --imgsz 640 --plots     # 额外输出 P-R 曲线、混淆矩阵
```

输出 `metrics.json` 与 `metrics_per_class.csv`：

```
整体: mAP50=0.xxxx  mAP50-95=0.xxxx  P=0.xxxx  R=0.xxxx
逐类:
  insulator  AP50=0.xxxx  AP50-95=0.xxxx
  defect     AP50=0.xxxx  AP50-95=0.xxxx
```

答辩时重点看 **defect 的 AP50 和 Recall** —— 漏检代价远高于误检。

### 本机实测结果（CPU / 8 轮 / 320 尺寸）

一次真实跑通的结果，可作为后续调参的基线：

| 类别 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 全部 | 0.806 | 0.816 | **0.871** | 0.547 |
| insulator | 0.902 | 0.952 | 0.963 | 0.772 |
| defect | 0.711 | 0.680 | 0.779 | 0.322 |

配置：`yolov8n.pt` 迁移学习，8 epoch，imgsz=320，batch=16，验证集 170 张 / 319 个框，
CPU（i5-13500H）约 35 分钟。缺陷类 mAP50-95 偏低是训练轮数不足所致，
按 `--profile cpu`（30 轮）或换 GPU 跑 `--profile full`（150 轮）还会继续涨。

产物位置：

```
runs/detect/cplid_yolov8n_cpu/
├── weights/best.pt          最佳权重
├── results.png              损失与指标曲线（可直接插进 PPT）
├── BoxPR_curve.png          P-R 曲线
├── confusion_matrix.png     混淆矩阵
└── val_batch*_pred.jpg      验证集预测样例
runs/detect/eval_cplid_yolov8n_cpu/
├── metrics.json             整体指标
└── metrics_per_class.csv    逐类指标
```

## 七、推理

```powershell
python predict.py --weights best.pt                                  # 跑验证集
python predict.py --weights best.pt --source D:/巡检图 --conf 0.3     # 跑自己的图片
python predict.py --weights best.pt --source some.jpg --imgsz 640
```

产物在 `runs/predict/cplid_predict/`：

- `*.jpg`：画好检测框的图片，框上标注「类别 + 置信度」（颜色由 ultralytics 自动分配）
- `detections.csv`：每个检测框一行（图片、类别、置信度、像素坐标）
- `summary.csv`：每张图一行（绝缘子数、缺陷数、最高缺陷置信度、结论）
  —— 对应开题报告里「按杆塔汇总的缺陷统计表」，每张巡检图视作一基杆塔

两个 CSV 都带 UTF-8 BOM，Excel 双击直接打开不乱码。

> **阈值怎么选**：见 `runs/eval/conf_sweep/REPORT.md`。结论是默认的 0.25 对"漏检代价高"的
> 巡检场景偏保守——降到 **0.05** 时图级检出率从 54% 升到 96%，正常图误报率仅 2.5%。
> 演示与报告统一用 `--conf 0.05`，并在图注里写明工作点；要"免人工复核直接告警"则用 `--conf 0.35`（零误报）。
> 重新扫描：`python conf_sweep.py --confs "0.02,0.05,0.1,0.15,0.25,0.35,0.5"`

## 八、常见问题

**1. 训练报 `PermissionError: [WinError 5]` 或卡在 "Scanning ..."**
ultralytics 建标签缓存时要用多线程/命名管道。在受限容器里会失败；
普通 Windows 下正常。另外把 `--workers` 保持为 0。

**2. 中文路径读图失败**
OpenCV 在 Windows 上不支持中文路径。工程里 `common.imread_u()` 用
`np.fromfile + cv2.imdecode` 绕开了这个问题，自己的新代码也请用它。

**3. 内存不够 / 越跑越慢**
去掉 `--cache`，并降低 `--batch`（如 4）。`imgsz` 从 320 降到 256 也能明显提速。

**4. 结果不理想**
- 缺陷类指标低：先用 `--defect-repeat 2` 做缺陷过采样，再开 `--cache` 多训几十轮；
- 小目标漏检：`--imgsz 640`，或换 `yolov8s.pt`；
- 类别判错：检查标注质量（`python prepare_dataset.py --check 12` 看框贴不贴）。

**5. 想换成自己的数据**
按 `datasets/cplid/` 的结构放 `images/{train,val}` 与 `labels/{train,val}`
（YOLO txt：`class cx cy w h`，归一化），再改 `data.yaml` 里的 `names` 即可。

## 九、与开题报告的对应

| 报告里的内容 | 代码里的落点 |
|---|---|
| 基线 YOLOv8 + COCO 迁移学习 | `train.py --model yolov8n.pt` |
| 多尺度特征融合（FPN） | YOLOv8 自带 PAN-FPN，无需改代码 |
| 类别不均衡：缺陷过采样 | `prepare_dataset.py --defect-repeat 2` |
| 评价：mAP@0.5、mAP@0.5:0.95、缺陷 Recall | `evaluate.py` |
| 按杆塔汇总缺陷统计 | `predict.py` 的 `summary.csv` |
| 可运行的演示 | `predict.py` 输出标注图 + CSV |
