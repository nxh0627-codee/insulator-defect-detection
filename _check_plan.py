# -*- coding: utf-8 -*-
import re
import zipfile
from lxml import etree

P = r"E:\专业综合能力提升\项目计划书.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

z = zipfile.ZipFile(P)
styles = z.read("word/styles.xml").decode("utf-8")
hits = set()
for m in re.finditer(r'<w:style [^>]*w:styleId="([^"]+)".*?</w:style>', styles, re.S):
    if "Courier" in m.group(0):
        hits.add(m.group(1))
print("styles referencing Courier:", sorted(hits))

doc = z.read("word/document.xml").decode("utf-8")
print("Courier in document.xml:", "Courier" in doc)
print("Consolas in document.xml:", "Consolas" in doc)

root = etree.fromstring(z.read("word/document.xml"))

TBL = ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize',
       'tblStyleColBandSize', 'tblW', 'jc', 'tblCellSpacing', 'tblInd', 'tblBorders',
       'shd', 'tblLayout', 'tblCellMar', 'tblLook', 'tblCaption', 'tblDescription']
TC = ['cnfStyle', 'tcW', 'gridSpan', 'hMerge', 'vMerge', 'tcBorders', 'shd', 'noWrap',
      'tcMar', 'textDirection', 'tcFitText', 'vAlign', 'hideMark', 'headers',
      'cellIns', 'cellDel', 'cellMerge']


def check(tag, order):
    bad = 0
    for el in root.iter("{%s}%s" % (W, tag)):
        tags = [etree.QName(c).localname for c in el]
        idx = [order.index(t) if t in order else -1 for t in tags]
        if idx != sorted(idx):
            bad += 1
            if bad <= 3:
                print("  %s 顺序异常: %s" % (tag, tags))
    return bad


print("tblPr 异常数:", check("tblPr", TBL))
print("tcPr  异常数:", check("tcPr", TC))

paras = root.findall(".//{%s}p" % W)
print("段落数:", len(paras))
tbls = root.findall(".//{%s}tbl" % W)
print("表格数:", len(tbls))
for i, t in enumerate(tbls, 1):
    grid = t.find("{%s}tblGrid" % W)
    widths = [int(g.get("{%s}w" % W)) for g in grid] if grid is not None else []
    print("  表%d: 列宽合计 %d twips (%.2f cm), 列数 %d" % (i, sum(widths), sum(widths) / 567.0, len(widths)))