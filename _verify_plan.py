# -*- coding: utf-8 -*-
"""内容 + 溢出风险检查（当前模型不支持读图，只能做结构推断）。"""
import io
import zipfile

import docx
from docx.shared import Emu

P = r"E:\专业综合能力提升\项目计划书.docx"
d = docx.Document(P)

out = []
out.append("=== 标题结构 ===")
for p in d.paragraphs:
    if p.style.name.startswith("Heading"):
        out.append("  " * (int(p.style.name[-1]) - 1) + p.text)

out.append("\n=== 全文（非空段落）===")
for i, p in enumerate(d.paragraphs):
    t = p.text.strip()
    if t:
        out.append(f"{i:03d} [{p.style.name}] {t}")

out.append("\n=== 表格内容 ===")
for ti, t in enumerate(d.tables, 1):
    out.append(f"--- 表{ti} ({len(t.rows)}行 × {len(t.columns)}列) ---")
    for row in t.rows:
        out.append("   | " + " | ".join(c.text.replace("\n", " / ") for c in row.cells))

# ---- 单元格溢出风险：可用宽度 = 列宽 - 左右边距(110twips×2)
out.append("\n=== 溢出风险（估算字宽 > 可用宽度 5% 以上）===")
risks = 0
for ti, t in enumerate(d.tables, 1):
    for ri, row in enumerate(t.rows):
        for ci, cell in enumerate(row.cells):
            width_tw = t.columns[ci].width
            if width_tw is None:
                continue
            avail_pt = width_tw / 12700.0 - 11.0  # EMU -> pt，再减左右边距
            for par in cell.paragraphs:
                for line in par.text.split("\n"):
                    if not line:
                        continue
                    size = 10.5
                    for r in par.runs:
                        if r.font.size:
                            size = r.font.size.pt
                            break
                    est = 0.0
                    for ch in line:
                        est += size if ord(ch) > 0x2E7F else size * 0.55
                    if est > avail_pt * 1.05:
                        risks += 1
                        if risks <= 8:
                            out.append(f"  表{ti} 行{ri} 列{ci}: 需要 {est:.0f}pt / 可用 {avail_pt:.0f}pt -> {line[:40]}")
out.append(f"  合计风险单元格：{risks}")

# ---- 检查段落中是否误留空 run / 缺失东亚字体
missing_ea = 0
for p in d.paragraphs:
    for r in p.runs:
        rpr = r._element.rPr
        if rpr is None or rpr.rFonts is None:
            missing_ea += 1
out.append(f"\n缺少字体声明的 run：{missing_ea}")

io.open(r"E:\专业综合能力提升\_plan_dump.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out[:40]))
print("...")
print(f"溢出风险单元格: {risks}；缺字体 run: {missing_ea}")