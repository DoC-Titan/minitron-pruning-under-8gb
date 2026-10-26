"""Optional report rendering; uses existing ReportLab, independent of science env."""
from html import escape
from pathlib import Path
import re
import textwrap
import xml.etree.ElementTree as ET

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Preformatted,KeepTogether
from reportlab.graphics.shapes import Drawing,Rect,Circle,Line,PolyLine,String,Group

ROOT=Path(__file__).resolve().parent


def color(value):
    if not value or value=='none':return None
    if value.startswith('rgb('):
        return colors.Color(*[int(x)/255 for x in value[4:-1].split(',')])
    return colors.toColor(value)


def vector(path,width=475):
    root=ET.parse(path).getroot();w=float(root.attrib['width']);h=float(root.attrib['height']);d=Drawing(w,h)
    for e in root:
        tag=e.tag.split('}')[-1];a=e.attrib;fill=color(a.get('fill','black'));stroke=color(a.get('stroke'))
        if tag=='rect':d.add(Rect(float(a.get('x',0)),h-float(a.get('y',0))-float(a['height']),float(a['width']),float(a['height']),fillColor=fill,strokeColor=stroke))
        elif tag=='circle':d.add(Circle(float(a['cx']),h-float(a['cy']),float(a['r']),fillColor=fill,strokeColor=stroke))
        elif tag=='polyline':
            pairs=[tuple(map(float,p.split(','))) for p in a['points'].split()]
            d.add(PolyLine([v for x,y in pairs for v in (x,h-y)],strokeColor=stroke,strokeWidth=float(a.get('stroke-width',1)),fillColor=None))
        elif tag=='path':
            # Our own figures use only M x,y followed by H or V grid lines.
            m=re.fullmatch(r'M([\d.-]+),([\d.-]+)\s*([HV])([\d.-]+)',a.get('d',''))
            if m:
                x,y=float(m[1]),float(m[2]);dest=float(m[4]);xx,yy=(dest,y) if m[3]=='H' else (x,dest)
                d.add(Line(x,h-y,xx,h-yy,strokeColor=stroke or colors.lightgrey))
        elif tag=='text':
            x,y=float(a['x']),h-float(a['y']);text=e.text or '';size=float(a.get('font-size',12))
            item=String(0,0,text,fontName='Helvetica',fontSize=size,fillColor=fill,textAnchor=a.get('text-anchor','start'))
            g=Group();g.translate(x,y)
            if a.get('transform','').startswith('rotate(-90'):g.rotate(90)
            g.add(item);d.add(g)
    scale=width/w;d.scale(scale,scale);d.width=width;d.height=h*scale;return d


def inline(text):
    text=escape(text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1 (\2)',text)
    return text


def main():
    source=(ROOT/'reports/TECHNICAL_REPORT.md').read_text();styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyResearch',fontName='Helvetica',fontSize=9,leading=13,spaceAfter=7))
    styles.add(ParagraphStyle(name='CellResearch',fontName='Helvetica',fontSize=7,leading=9))
    normal=styles['BodyResearch'];story=[];lines=source.splitlines();i=0
    while i<len(lines):
        line=lines[i];i+=1
        if not line.strip():continue
        if line.startswith('```'):
            code=[]
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            wrapped=[part for row in code for part in (textwrap.wrap(row,width=112,replace_whitespace=False,drop_whitespace=False) or [''])]
            i+=1;story.append(Preformatted('\n'.join(wrapped),ParagraphStyle(name='CodeResearch',fontName='Courier',fontSize=6.5,leading=8)));story.append(Spacer(1,8));continue
        if line.startswith('|'):
            tablelines=[line]
            while i<len(lines) and lines[i].startswith('|'):tablelines.append(lines[i]);i+=1
            cells=[[x.strip() for x in row.strip('|').split('|')] for row in tablelines if not re.fullmatch(r'[| :\-]+',row)]
            data=[[Paragraph(inline(x),styles['CellResearch']) for x in row] for row in cells]
            table=Table(data,colWidths=[475/len(cells[0])]*len(cells[0]),repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dceeee')),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#c7d2d9')),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5)]))
            story+=[table,Spacer(1,10)];continue
        image=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if image:
            path=(ROOT/'reports'/image[2]).resolve()
            story+=[vector(path),Spacer(1,10)];continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));text=line[level:].strip()
            story.append(Paragraph(inline(text),styles['Title'] if level==1 else styles['Heading1' if level==2 else 'Heading2']));continue
        if line.startswith('- '):line='• '+line[2:]
        story.append(Paragraph(inline(line),normal))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',7);canvas.setFillColor(colors.HexColor('#627580'))
        canvas.drawString(42,26,'Built with Qwen | Minitron-inspired local research | Measured evidence; limitations disclosed')
        canvas.drawRightString(553,26,str(doc.page))
    doc=SimpleDocTemplate(str(ROOT/'reports/TECHNICAL_REPORT.pdf'),pagesize=(595.28,841.89),rightMargin=60,leftMargin=60,topMargin=48,bottomMargin=45,
                          title='Structured Pruning and Distillation under an 8 GiB GPU Budget',author='Local research portfolio')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('Technical report PDF rendered from measured Markdown and vector figures.')


if __name__=='__main__':main()
