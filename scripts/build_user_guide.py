"""Create the editable operations manual using the managed document runtime."""
import argparse
import re
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]


def style_font(style, size, name='Calibri'):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), 'Microsoft YaHei')


def add_table(doc, rows):
    table = doc.add_table(rows=0, cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = [1.25, 2.8, 2.65] if len(rows[0]) == 3 else [1.5, 5.2]
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        element = OxmlElement('w:' + edge)
        for key, value in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
            element.set(qn('w:' + key), value)
        borders.append(element)
    table._tbl.tblPr.append(borders)
    margins = OxmlElement('w:tblCellMar')
    for edge in ('top', 'bottom', 'left', 'right'):
        element = OxmlElement('w:' + edge)
        element.set(qn('w:w'), '90'); element.set(qn('w:type'), 'dxa')
        margins.append(element)
    table._tbl.tblPr.append(margins)
    for index, values in enumerate(rows):
        row = table.add_row()
        props = row._tr.get_or_add_trPr()
        props.append(OxmlElement('w:cantSplit'))
        if index == 0:
            props.append(OxmlElement('w:tblHeader'))
        for column, (cell, value, width) in enumerate(zip(row.cells, values, widths)):
            cell.width = Inches(width)
            table.columns[column].width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            shading = OxmlElement('w:shd')
            shading.set(qn('w:fill'), 'DAE7F1' if index == 0 else 'F5F7FA' if index % 2 == 0 else 'FFFFFF')
            cell._tc.get_or_add_tcPr().append(shading)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.10
            if column == 0:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(value)
            run.bold = index == 0
            run.font.size = Pt(10.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def build(source, output):
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5); section.page_height = Inches(11)
    section.top_margin = section.bottom_margin = Inches(.7)
    section.left_margin = section.right_margin = Inches(.9)
    section.footer_distance = Inches(.3)
    for border in list(doc.styles.element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
    for name, size in [('Normal', 11), ('Title', 24), ('Heading 1', 16), ('Heading 2', 12.5), ('Footer', 9)]:
        style_font(doc.styles[name], size)
    normal = doc.styles['Normal'].paragraph_format
    normal.line_spacing = 1.18; normal.space_after = Pt(7)
    for name in ('Heading 1', 'Heading 2'):
        fmt = doc.styles[name].paragraph_format
        fmt.space_before = Pt(12); fmt.space_after = Pt(7); fmt.keep_with_next = True
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('使用指南    ')
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
    lines = source.read_text(encoding='utf-8').splitlines()
    index = 0; code = False
    while index < len(lines):
        line = lines[index].strip(); index += 1
        if line.startswith('```'):
            code = not code
            continue
        if not line:
            continue
        if code:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.05
            run = p.add_run(line); run.font.name = 'Consolas'; run.font.size = Pt(9)
            continue
        if line.startswith('|'):
            rows = [[cell.strip() for cell in line.strip('|').split('|')]]
            while index < len(lines) and lines[index].strip().startswith('|'):
                values = [cell.strip() for cell in lines[index].strip().strip('|').split('|')]; index += 1
                if not all(re.fullmatch(r'[-: ]+', value) for value in values):
                    rows.append(values)
            add_table(doc, rows)
        elif line.startswith('# '):
            doc.add_paragraph(line[2:], 'Title')
        elif line.startswith('## '):
            doc.add_paragraph(line[3:], 'Heading 1')
        elif line.startswith('### '):
            doc.add_paragraph(line[4:], 'Heading 2')
        else:
            paragraph = doc.add_paragraph(line)
            if re.match(r'\d+\.', line):
                paragraph.paragraph_format.left_indent = Inches(.18)
                paragraph.paragraph_format.first_line_indent = Inches(-.18)
    doc.core_properties.title = '智能火灾报警系统完整使用指南'
    doc.core_properties.subject = '安装 检测 报警 数据 标注 训练 评测与故障排查'
    doc.core_properties.author = 'c1love9'
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'docs/user_guide.md')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/release/智能火灾报警系统完整使用指南.docx')
    arguments = parser.parse_args()
    build(arguments.source, arguments.output)
