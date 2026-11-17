# -*- coding: utf-8 -*-
"""生成《项目计划书》.docx —— 三人分工版。

排版沿用《项目文档》的字体（Arial + Microsoft YaHei）与表格样式（D9E2F3 表头、BFBFBF 细边框）。
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Pt, RGBColor

OUT = Path(r"E:\专业综合能力提升\项目计划书.docx")

ASCII_FONT = "Arial"
EA_FONT = "Microsoft YaHei"
HEADER_FILL = "D9E2F3"
BORDER_COLOR = "BFBFBF"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)


# ---------------------------------------------------------------- 基础工具
def style_font(style, size=None, bold=None, color=None):
    style.font.name = ASCII_FONT
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(attr), ASCII_FONT)
    rfonts.set(qn("w:eastAsia"), EA_FONT)


def mk(tag, **attrs):
    el = OxmlElement(tag)
    for k, v in attrs.items():
        el.set(qn(k), v)
    return el


def insert_ordered(parent, el, successors):
    for tag in successors:
        found = parent.find(qn(tag))
        if found is not None:
            found.addprevious(el)
            return
    parent.append(el)


def body(doc, text, *, size=12, bold=False, indent=0, space_after=6, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.5
    if indent:
        p.paragraph_format.first_line_indent = Pt(indent)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = ASCII_FONT
    r._element.rPr.rFonts.set(qn("w:eastAsia"), EA_FONT)
    return p


def bullet(doc, text, *, size=12):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.4
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.name = ASCII_FONT
    r._element.rPr.rFonts.set(qn("w:eastAsia"), EA_FONT)
    return p


def note(doc, text, *, size=10):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    r.font.name = ASCII_FONT
    r._element.rPr.rFonts.set(qn("w:eastAsia"), EA_FONT)
    return p


def heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.name = ASCII_FONT
        r._element.rPr.rFonts.set(qn("w:eastAsia"), EA_FONT)
        r.font.color.rgb = ACCENT
    h.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    h.paragraph_format.space_after = Pt(6)
    return h


def add_table(doc, rows, widths_twips, *, font_size=10.5, header=True,
              center_cols=(), first_col_bold=False):
    n_rows, n_cols = len(rows), len(rows[0])
    tbl = doc.add_table(rows=n_rows, cols=n_cols)
    tbl.autofit = False
    tblPr = tbl._tbl.tblPr

    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = mk("w:tblW")
        insert_ordered(tblPr, tblW, ["w:jc", "w:tblBorders", "w:tblLayout", "w:tblLook"])
    tblW.set(qn("w:type"), "dxa")
    tblW.set(qn("w:w"), str(sum(widths_twips)))

    jc = mk("w:jc", **{"w:val": "center"})
    insert_ordered(tblPr, jc, ["w:tblBorders", "w:tblLayout", "w:tblLook"])

    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(mk(f"w:{edge}", **{"w:val": "single", "w:color": BORDER_COLOR,
                                          "w:sz": "4", "w:space": "0"}))
    insert_ordered(tblPr, borders, ["w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook"])

    for i, w in enumerate(widths_twips):
        tbl.columns[i].width = Emu(w * 635)

    for ri, row in enumerate(rows):
        trPr = tbl.rows[ri]._tr.get_or_add_trPr()
        trPr.append(mk("w:cantSplit"))
        if ri == 0 and header:
            trPr.append(mk("w:tblHeader"))
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.width = Emu(widths_twips[ci] * 635)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tcPr = cell._tc.get_or_add_tcPr()
            mar = OxmlElement("w:tcMar")
            for edge, w in (("top", "70"), ("left", "110"), ("bottom", "70"), ("right", "110")):
                mar.append(mk(f"w:{edge}", **{"w:type": "dxa", "w:w": w}))
            tcPr.append(mar)
            if ri == 0 and header:
                tcPr.append(mk("w:shd", **{"w:fill": HEADER_FILL, "w:val": "clear"}))
            elif ri % 2 == 0:
                tcPr.append(mk("w:shd", **{"w:fill": "F7F9FC", "w:val": "clear"}))

            par = cell.paragraphs[0]
            par.paragraph_format.space_after = Pt(0)
            par.paragraph_format.space_before = Pt(0)
            par.paragraph_format.line_spacing = 1.15
            if ci in center_cols:
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            lines = str(val).split("\n")
            for li, line in enumerate(lines):
                if li:
                    par = cell.add_paragraph()
                    par.paragraph_format.space_after = Pt(0)
                    par.paragraph_format.space_before = Pt(0)
                    par.paragraph_format.line_spacing = 1.15
                    if ci in center_cols:
                        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = par.add_run(line)
                run.font.size = Pt(font_size)
                run.font.name = ASCII_FONT
                run._element.rPr.rFonts.set(qn("w:eastAsia"), EA_FONT)
                if (ri == 0 and header) or (ci == 0 and first_col_bold):
                    run.font.bold = True
    return tbl


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    run._element.addnext(fld)
    for node in (fld,):
        for rr in node.iter(qn("w:r")):
            rpr = OxmlElement("w:rPr")
            rpr.append(mk("w:rFonts", **{"w:ascii": ASCII_FONT, "w:hAnsi": ASCII_FONT,
                                          "w:eastAsia": EA_FONT}))
            rpr.append(mk("w:sz", **{"w:val": "20"}))
            rr.insert(0, rpr)


def normalize_tcpr(document):
    """把每个 tcPr 的子元素按 OOXML schema 顺序重排，避免 Word 报错。"""
    order = ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd",
             "noWrap", "tcMar", "textDirection", "tcFitText", "vAlign", "hideMark",
             "headers", "cellIns", "cellDel", "cellMerge"]

    def key(el):
        tag = el.tag.split("}")[-1]
        return order.index(tag) if tag in order else len(order)

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                tcPr = cell._tc.tcPr
                if tcPr is None:
                    continue
                children = list(tcPr)
                for child in children:
                    tcPr.remove(child)
                for child in sorted(children, key=key):
                    tcPr.append(child)


# ---------------------------------------------------------------- 文档
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.54)
sec.top_margin = sec.bottom_margin = Cm(2.54)

style_font(doc.styles["Normal"], size=12)
for lvl, sz in ((1, 16), (2, 13.5), (3, 12)):
    style_font(doc.styles[f"Heading {lvl}"], size=sz, bold=True, color=ACCENT)
add_page_number_footer(sec)

# ---------------- 封面
for _ in range(4):
    doc.add_paragraph()
body(doc, "基于深度学习的输电线路绝缘子缺陷检测系统", size=22, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
body(doc, "项 目 计 划 书", size=26, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)
body(doc, "（第三阶段：第 11–14 教学周）", size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=60)

info = [
    ("项目组", "崔光鹏、戴海青、宁湘涵"),
    ("课程", "专业综合能力提升"),
    ("代码仓库", "github.com/jinqu-bot/insulator-defect-detection（私有）"),
    ("制定日期", "2026 年 11 月 17 日"),
    ("交付截止", "第 14 教学周（约 2026 年 12 月 13 日）"),
]
t = doc.add_table(rows=len(info), cols=2)
t.autofit = False
t.columns[0].width = Emu(1700 * 635)
t.columns[1].width = Emu(4300 * 635)
for i, (k, v) in enumerate(info):
    for ci, val in ((0, k), (1, v)):
        cell = t.cell(i, ci)
        cell.width = Emu((1600 if ci == 0 else 3500) * 635)
        pr = cell.paragraphs[0]
        pr.paragraph_format.space_after = Pt(2)
        run = pr.add_run(val)
        run.font.size = Pt(11)
        run.font.name = ASCII_FONT
        run._element.rPr.rFonts.set(qn("w:eastAsia"), EA_FONT)
        if ci == 0:
            run.font.bold = True
            pr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
# 表中线去掉
for i in range(len(info)):
    for ci in (0, 1):
        tcPr = t.cell(i, ci)._tc.get_or_add_tcPr()
        tcPr.append(mk("w:tcBorders", ))

doc.add_page_break()

# ---------------- 一、项目概况
heading(doc, "一、项目概况", 1)

heading(doc, "1.1 基本情况", 2)
add_table(doc, [
    ["项目", "内容"],
    ["任务类型", "目标检测：先定位绝缘子，再判定有无缺陷（insulator / defect 两类）"],
    ["数据集", "CPLID 公开数据集 848 张（正常 600 / 缺陷 248），分辨率 1152×864"],
    ["模型", "YOLOv8n，COCO 预训练权重迁移学习"],
    ["运行环境", "Python 3.13 + ultralytics 8.4，Intel i5-13500H + 16 线程，仅 CPU"],
    ["评价指标", "mAP@0.5、mAP@0.5:0.95、缺陷类 Recall，以及图级检出率 / 误报率"],
], [1400, 7626], font_size=10.5)

heading(doc, "1.2 当前进展（截至 11 月 17 日）", 2)
body(doc, "基线流程已全部打通，但开题承诺的改进实验尚未开展。具体状态如下：", indent=24)
add_table(doc, [
    ["已完成", "状态与关键数字"],
    ["数据流水线", "VOC→YOLO 转换、本体/缺陷标注合并、按类别分层划分 678 / 170（seed=42）"],
    ["基线训练", "YOLOv8n 迁移学习，8 轮 / 320 尺寸 / CSV 记录完整，耗时约 28 分钟"],
    ["基线指标", "mAP@0.5 = 0.871；绝缘子 AP50 0.963；缺陷 AP50 0.779、AP50-95 0.322"],
    ["阈值工作点", "conf 0.25→0.05：图级检出率 54% → 96%，正常图误报率仅 2.5%"],
    ["工程与存档", "训练/评估/推理/阈值扫描四个脚本；代码与评估产物已推入私有仓库"],
], [2200, 6826], font_size=10.5)

body(doc, "尚未开展：缺陷类过采样、Focal Loss、640 分辨率对照三组实验；误差分析；演示系统；"
          "实验报告与 PPT。这也是本期计划的核心内容。", indent=24, space_after=10)

heading(doc, "1.3 本期目标", 2)
bullet(doc, "补齐 4 组消融实验（基线 / 过采样 / Focal + 类权重 / 640 分辨率），形成可对比的实验结论；")
bullet(doc, "在验证集上确定最终模型与工作点，缺陷 AP50 ≥ 0.80、图级检出率 ≥ 80%（误报 ≤ 10%）；")
bullet(doc, "产出可现场演示的系统（离线单页报告）与完整的实验报告、答辩 PPT。")

# ---------------- 二、交付物
heading(doc, "二、交付物与验收标准", 1)
body(doc, "所有交付物以“能否被他人独立验证”为准，不接受口头完成。", indent=24)
add_table(doc, [
    ["交付物", "形式", "验收标准", "截止"],
    ["消融实验对比表", "runs/ablation/summary.csv", "4 组齐全，同一划分、同一随机种子", "第 12 周"],
    ["最终模型", "best.pt + metrics.json", "缺陷 AP50 ≥ 0.80；图级检出率 ≥ 80%", "第 13 周"],
    ["误差分析", "归类表 + 可视化图", "漏检、误检各不少于 15 例并给出归因", "第 12 周"],
    ["演示系统", "report.html（离线）", "拖入图片即出检测框、统计表与结论", "第 13 周"],
    ["实验报告", "docx / pdf", "含方法、实验、结果、局限四部分，图表编号规范", "第 14 周中"],
    ["答辩 PPT", "pptx + 讲稿", "10 分钟内讲完，含一次现场演示", "第 14 周末"],
], [1900, 2700, 3400, 1026], font_size=10.5, center_cols=(3,))

# ---------------- 三、WBS
heading(doc, "三、工作分解（WBS）", 1)
body(doc, "任务按“一个人能在半天到一天内做完”拆分，工时为人时估算，不含机器训练时间。", indent=24)
add_table(doc, [
    ["编号", "任务内容", "产出物", "工时", "主责"],
    ["T1", "数据核对：标注自检图复核、类别与框数统计", "数据卡片更新", "1.5 h", "宁湘涵"],
    ["T2", "近重复图像分组（pHash），核查训练/验证同源泄漏", "泄漏核查报告", "3 h", "宁湘涵"],
    ["T3", "实现 Focal Loss 与缺陷类权重（子类化 v8DetectionLoss）", "improve.py", "3 h", "戴海青"],
    ["T4", "写消融调度脚本：顺序跑 4 组、自动汇总", "run_ablations.py", "2 h", "戴海青"],
    ["T5", "A 组：320 基线 30 轮", "runs/ablation/A", "2 h 机时", "戴海青"],
    ["T6", "B 组：320 + 缺陷过采样 ×2", "runs/ablation/B", "2.5 h 机时", "戴海青"],
    ["T7", "D 组：320 + 过采样 + Focal + 类权重", "runs/ablation/D", "2.5 h 机时", "戴海青"],
    ["T8", "C 组：640 + 过采样（需过夜）", "runs/ablation/C", "6 h 机时", "戴海青"],
    ["T9", "四组结果汇总、对比表与趋势图", "summary.csv + fig", "2 h", "戴海青"],
    ["T10", "误差分析：漏检 / 误检样例归类与归因", "误差分析表 + 图", "4 h", "宁湘涵"],
    ["T11", "最终模型选定，重跑阈值扫描", "conf_sweep 更新", "1.5 h", "戴海青"],
    ["T12", "单页 HTML 演示系统（离线、无依赖）", "report.html", "6 h", "崔光鹏"],
    ["T13", "按目录 / 杆塔分组汇总", "predict.py 增强", "1.5 h", "崔光鹏"],
    ["T14", "图表包：PR 曲线、混淆矩阵、样例图筛选", "figs/", "2 h", "宁湘涵"],
    ["T15", "报告撰写：各自章节 + 统稿排版", "报告初稿", "2 h×3 + 4 h", "三人 / 崔统稿"],
    ["T16", "PPT 与讲稿", "pptx + 讲稿", "5 h", "崔光鹏"],
    ["T17", "预演 2 次并修订", "终稿", "2 h", "三人"],
], [700, 4200, 2100, 1100, 926], font_size=10, center_cols=(0, 3, 4))

# ---------------- 四、分工
heading(doc, "四、三人分工", 1)

heading(doc, "4.1 分工原则", 2)
bullet(doc, "主责唯一：每个任务只有一个负责人，避免“都以为对方在做”；")
bullet(doc, "互为备份：每人熟悉另一人的主要产出，答辩时都能解释；")
bullet(doc, "可验收：每个交付物都有明确的完成判据，不以“差不多了”结束；")
bullet(doc, "负载均衡：人均约 17 小时、每周约 4 小时；机时集中在夜间无人值守。")

heading(doc, "4.2 角色与职责", 2)
add_table(doc, [
    ["成员", "角色", "主责任务", "核心交付物", "工时"],
    ["戴海青\n(jinqu-bot)", "模型与训练", "T3 T4 T5 T6 T7 T8 T9 T11",
     "improve.py、run_ablations.py、四组实验结果、最终权重", "21 h"],
    ["宁湘涵\n(nxh0627-codee)", "数据与评估", "T1 T2 T10 T14 + 报告数据章",
     "泄漏核查、误差分析、图表包、数据章节", "14.5 h"],
    ["崔光鹏\n(tomfjkl-a11y)", "系统与文档", "T12 T13 T15统稿 T16",
     "report.html、报告全篇、PPT 与讲稿", "16.5 h"],
], [1700, 1400, 2500, 2400, 1026], font_size=10, center_cols=(4,), first_col_bold=True)
note(doc, "说明：GitHub 账号对应按现有信息推定（jinqu-bot=戴海青、nxh0627-codee=宁湘涵、"
          "tomfjkl-a11y=崔光鹏）。如与实际不符，只改此表姓名，不影响分工内容。")

heading(doc, "4.3 为什么这样分", 2)
body(doc, "戴海青已完整跑通过训练与评估链路，Focal Loss 改造和实验排期需要改代码、盯结果，"
          "交给最熟悉现有脚本的人最省时间；宁湘涵负责数据侧与评估侧，工作与训练解耦，"
          "可以在别人跑实验时并行推进；崔光鹏负责演示系统与文档，产出集中但可提前开工，"
          "不依赖实验全部完成。", indent=24, space_after=8)
body(doc, "如果实际情况（比如某人不熟悉 Python）需要调整，建议的换法是：把 T3、T4 从戴海青转给"
          "熟悉 Python 的人，戴海青只负责跑实验与记录；宁湘涵接手 T12 的 HTML 部分。"
          "无论怎么调，保持“主责唯一”这一条即可。", indent=24)

# ---------------- 五、进度
heading(doc, "五、进度计划", 1)
body(doc, "以周为单位排期，● 表示主责，○ 表示参与。第 11 周因前期进度滞后，定为追赶周。", indent=24)
add_table(doc, [
    ["时间段", "第 11 周\n11/17–11/22", "第 12 周\n11/23–11/29", "第 13 周\n11/30–12/6", "第 14 周\n12/7–12/13"],
    ["Focal 改造与调度脚本", "●", "", "", ""],
    ["A / B 组训练", "●", "", "", ""],
    ["D / C 组训练", "", "●", "", ""],
    ["消融汇总与选型", "", "●", "●", ""],
    ["数据核查与误差分析", "○", "●", "", ""],
    ["演示系统开发", "●", "○", "●", ""],
    ["报告撰写", "", "○", "●", ""],
    ["PPT 与讲稿", "", "", "●", "●"],
    ["预演与定稿", "", "", "", "●"],
], [2700, 1580, 1580, 1580, 1586], font_size=10, center_cols=(1, 2, 3, 4))

note(doc, "关键约束：训练必须串行。CPU 只有 16 线程，两个训练进程同时跑会互相抢核，总耗时反而更长，"
          "请统一由戴海青排队执行，其他人不要在本机开训练。")

# ---------------- 六、里程碑
heading(doc, "六、里程碑与检查点", 1)
add_table(doc, [
    ["里程碑", "时间", "检查内容", "未达成的处置"],
    ["M1 代码就绪", "11/20", "improve.py + 调度脚本跑通 1 轮 smoke test", "退回用基线配置，D 组降级为仅过采样"],
    ["M2 实验齐全", "11/29", "A/B/D/C 四组 results.csv 齐全", "砍 C 组，改用 480 尺寸替代"],
    ["M3 系统与初稿", "12/6", "report.html 可用、报告初稿成篇", "演示降级为截图 + CSV，报告优先"],
    ["M4 交付", "12/13", "报告定稿、PPT、预演两次", "提前一周启动，不接受顺延"],
], [1700, 900, 3400, 3026], font_size=10, center_cols=(1,))

# ---------------- 七、协作机制
heading(doc, "七、协作机制", 1)

heading(doc, "7.1 代码协作", 2)
bullet(doc, "仓库：github.com/jinqu-bot/insulator-defect-detection（私有），主分支 main；")
bullet(doc, "分支：每人用 feat/姓名-任务 开分支（如 feat/ning-error-analysis），完成后提 PR 合入 main；")
bullet(doc, "提交：信息用“动词: 内容”，如 fix: 修正阈值扫描的 IoU 匹配；每晚 push 一次，避免堆积；")
bullet(doc, "评审：涉及 train.py / predict.py 的改动由另一人扫一眼再合，防止把别人的实验结果跑挂。")

heading(doc, "7.2 文件与数据约定", 2)
bullet(doc, "实验产物统一放 runs/ablation/<组号>/，组号用 A/B/C/D，不做私人命名；")
bullet(doc, "数据集（约 500 MB）与模型权重（每个 6 MB）不入库，已在 .gitignore 排除；")
bullet(doc, "每人本地保留自己的 runs 副本，只把 summary.csv、图表、报告入库，避免仓库膨胀。")

heading(doc, "7.3 沟通节奏", 2)
bullet(doc, "每周日 21:00 线上同步 30 分钟：各自说“完成了什么、卡在哪、下周做什么”；")
bullet(doc, "日常问题当天在群里提，超过半天没解决就升级给全组，不要独自硬扛；")
bullet(doc, "所有结论以仓库里的文件为准，群里说了不等于完成。")

heading(doc, "7.4 环境一致性", 2)
bullet(doc, "统一 Python 3.13 + ultralytics 8.4，依赖以 requirements.txt 为准；")
bullet(doc, "所有实验固定 seed=42、deterministic=True，否则指标不可比；")
bullet(doc, "网络限制：本机 github.com:443 不通，需用 SSH over 443，配置见附录。")

# ---------------- 八、风险
heading(doc, "八、风险与应对", 1)
add_table(doc, [
    ["风险", "影响", "应对"],
    ["无 GPU，640 训练约 11 分钟/轮", "C 组只能跑一次，且需过夜", "320 作主实验，640 仅作单次对照；不追 150 轮"],
    ["前期进度滞后一周", "实验窗口压缩到 3 周", "第 11 周定为追赶周，A/B 组必须 11/22 前排入队列"],
    ["三人环境不一致、结果对不上", "指标无法横向比较", "固定种子与版本，实验产物统一命名入库"],
    ["训练集/验证集同源图像泄漏", "指标偏高、答辩被质疑", "pHash 分组核查，并在报告“局限”中如实说明"],
    ["演示现场故障", "答辩失分", "准备离线 HTML + 录屏备份，不依赖网络"],
    ["分工不均、有人掉队", "整体延期", "主责唯一 + 每周同步会核对进度，落后立即重分配"],
], [3000, 2600, 3426], font_size=10)

# ---------------- 九、附录
heading(doc, "九、附录：环境与常用命令", 1)

heading(doc, "9.1 加入仓库（首次）", 2)
body(doc, "① 在 GitHub 接受邀请；② 把本机 SSH 公钥加到 GitHub（Settings → SSH and GPG keys）；"
          "③ 执行以下命令（本机 github.com:443 不通，必须走 443 端口的 SSH）：", indent=24, size=11)
code = doc.add_paragraph()
code.paragraph_format.left_indent = Pt(24)
code.paragraph_format.space_after = Pt(8)
for line in ["git clone git@ssh.github.com:jinqu-bot/insulator-defect-detection.git",
             "cd insulator-defect-detection",
             'git config core.sshCommand "ssh -p 443 -o StrictHostKeyChecking=accept-new"']:
    r = code.add_run(line + "\n")
    r.font.name = "Consolas"
    r.font.size = Pt(10)
    r._element.rPr.rFonts.set(qn("w:eastAsia"), EA_FONT)

heading(doc, "9.2 常用命令", 2)
add_table(doc, [
    ["目的", "命令"],
    ["生成数据集", "python prepare_dataset.py --check 8"],
    ["训练（CPU 预设）", "python train.py"],
    ["评估", "python evaluate.py --weights runs/detect/<name>/weights/best.pt"],
    ["推理（筛查工作点）", "python predict.py --weights <best.pt> --conf 0.05"],
    ["阈值扫描", 'python conf_sweep.py --confs "0.02,0.05,0.1,0.25,0.35"'],
], [2600, 6426], font_size=10)

doc.add_paragraph()
body(doc, "编制：项目组    日期：2026 年 11 月 17 日", size=10.5,
     align=WD_ALIGN_PARAGRAPH.RIGHT)

normalize_tcpr(doc)
doc.save(str(OUT))
print("saved:", OUT)