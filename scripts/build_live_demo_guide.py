"""Build the printable live demonstration guide using the existing manual style."""
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from build_user_guide import build

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'artifacts/live_demo_20261108/智能火灾报警系统现场演示操作指南.docx'
    build(ROOT/'docs/live_demo_guide.md',output)
    doc=Document(output);section=doc.sections[0]
    section.page_width=Cm(21);section.page_height=Cm(29.7)
    section.left_margin=section.right_margin=Cm(2)
    section.top_margin=section.bottom_margin=Cm(1.8)
    normal=doc.styles['Normal'];normal.font.size=Pt(10.5)
    normal.paragraph_format.line_spacing=1.12;normal.paragraph_format.space_after=Pt(5)
    doc.styles['Title'].font.size=Pt(22)
    doc.styles['Heading 1'].font.size=Pt(15)
    doc.styles['Heading 2'].font.size=Pt(11.5)
    headings=[]
    for paragraph in doc.paragraphs:
        if paragraph.style.name=='Heading 1':
            headings.append(paragraph.text)
            paragraph.paragraph_format.page_break_before=len(headings)>1
    assert len(headings)==7
    for table in doc.tables:
        widths=[2.9,7.1,7.0] if len(table.columns)==3 else [4.2,12.8]
        assert len(widths)==len(table.columns)
        for index,width in enumerate(widths):
            table.columns[index].width=Cm(width)
        for row in table.rows:
            for cell,width in zip(row.cells,widths):cell.width=Cm(width)
    doc.core_properties.title='智能火灾报警系统现场演示操作指南'
    doc.core_properties.subject='现场启动 操作顺序 讲解提示 常见提问与故障处理'
    # Request Word to refresh the existing page number fields when opened.
    settings=doc.settings.element
    for child in list(settings.findall(qn('w:updateFields'))):settings.remove(child)
    update=OxmlElement('w:updateFields');update.set(qn('w:val'),'true');settings.append(update)
    doc.save(output);print(output)


if __name__=='__main__':main()
