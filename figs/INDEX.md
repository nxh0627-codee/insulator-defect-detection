# 图表包索引（figs/）

> 由 `python make_figs.py` 生成。图号在报告正文里直接引用（如「见图 fig02」）。

| 编号 | 文件 | 内容 | 来源 | 建议用在 |
|---|---|---|---|---|
| fig01 | `fig01_target_size.png` | 目标尺寸分布：insulator 与 defect 的框边长（对数轴） | `make_figs.py 现场绘制（源：datasets/cplid/labels）` | 报告「数据」：小目标依据 |
| fig02 | `fig02_pr_curve.png` | 各类别 P-R 曲线（基线，8 轮 / 320） | `insulator_yolo\runs\detect\cplid_yolov8n_cpu\BoxPR_curve.png` | 报告「实验结果」：整体检测性能 |
| fig03 | `fig03_f1_curve.png` | 置信度-F1 曲线，用于说明工作点与 F1 的关系 | `insulator_yolo\runs\detect\cplid_yolov8n_cpu\BoxF1_curve.png` | 报告「实验结果」：阈值选择依据 |
| fig04 | `fig04_confusion_matrix.png` | 混淆矩阵（绝对计数） | `insulator_yolo\runs\detect\eval_cplid_yolov8n_cpu\confusion_matrix.png` | 报告「实验结果」：类别间混淆 |
| fig05 | `fig05_confusion_matrix_normalized.png` | 归一化混淆矩阵 | `insulator_yolo\runs\detect\eval_cplid_yolov8n_cpu\confusion_matrix_normalized.png` | 报告「实验结果」：漏检 / 误检比例 |
| fig06 | `fig06_conf_sweep.png` | 置信度阈值扫描：图级检出率 / 误报率 / 每图框数 | `insulator_yolo\runs\eval\conf_sweep\sweep.png` | 报告「业务口径」：三档工作点 |
| fig07 | `fig07_leakage_nearest_dist.png` | pHash 最近邻汉明距离分布（阈值 12 的依据） | `make_figs.py 现场绘制（源：runs/leakage/nearest_dist.json）` | 报告「数据」：泄漏核查 |
| fig08 | `fig08_sample_defect.jpg` | 缺陷样本标注示例（绿框绝缘子、红框缺陷） | `insulator_yolo\datasets\cplid\_check\d_157.jpg` | 报告「数据」：标注质量与合成缺陷说明 |

## 重新生成

```powershell
cd insulator_yolo
python make_figs.py
```
