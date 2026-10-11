import io
import os
import re

import fitz
import matplotlib
import mistune
from PIL import Image

REPORTS = r"c:\Users\hoang\project\stumpy\Whales_Song\reports"
NAME = "Motif_Hierarchy_Report"
FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
CSS = """
@font-face { font-family: dv; src: url(DejaVuSans.ttf); }
@font-face { font-family: dv; src: url(DejaVuSans-Bold.ttf); font-weight: bold; }
@font-face { font-family: dv; src: url(DejaVuSans-Oblique.ttf); font-style: italic; }
@font-face { font-family: dv; src: url(DejaVuSans-BoldOblique.ttf); font-weight: bold; font-style: italic; }
@font-face { font-family: dvm; src: url(DejaVuSansMono.ttf); }
body { font-family: dv; font-size: 9pt; line-height: 1.35; color: #222; }
h1 { font-size: 16pt; margin: 0; }
h2 { font-size: 13pt; margin: 0; }
h3 { font-size: 11pt; margin: 0; }
p { margin: 0; }
ul { margin: 0; }
table { border-collapse: collapse; margin: 0; font-size: 7pt; }
th, td { border: 1px solid #999; padding: 2px 3px; }
th { background-color: #e8ecf2; }
code { font-family: dvm; font-size: 8pt; }
li { margin-bottom: 2px; }
"""
PAGE = fitz.paper_rect("a4")
LEFT, TOP, RIGHT, BOTTOM = 42, 42, PAGE.width - 42, PAGE.height - 42
WIDTH = RIGHT - LEFT
GAP = {"h": 4, "p": 7, "img": 4, "table": 9}

md = mistune.create_markdown(plugins=["table", "strikethrough"], escape=False)
archive = fitz.Archive(REPORTS)
archive.add(FONT_DIR)

text = open(os.path.join(REPORTS, NAME + ".md"), encoding="utf-8").read()
body = text.split("<!-- TIMELINE_START -->")[0]
blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", body) if b.strip()]


def kind(block):
    if block.startswith("!["):
        return "img"
    if block.startswith("#"):
        return "h"
    if block.startswith("|"):
        return "table"
    return "p"


buf = io.BytesIO()
writer = fitz.DocumentWriter(buf)
dev, y = writer.begin_page(PAGE), TOP


def new_page():
    global dev, y
    writer.end_page()
    dev, y = writer.begin_page(PAGE), TOP


for i, block in enumerate(blocks):
    k = kind(block)
    if k == "img":
        src = re.search(r"\((.+?)\)", block).group(1)
        w_px, h_px = Image.open(os.path.join(REPORTS, src)).size
        width = WIDTH
        height = width * h_px / w_px
        if height > BOTTOM - TOP:
            height = BOTTOM - TOP
            width = height * w_px / h_px
        if y + height > BOTTOM:
            new_page()
        page_doc = fitz.open()
        rect = fitz.Rect(LEFT + (WIDTH - width) / 2, y, LEFT + (WIDTH + width) / 2, y + height)
        html = f'<img src="{src}" width="{width:.1f}pt" height="{height:.1f}pt"/>'
        story = fitz.Story(html=html, user_css="img { margin: 0; }", archive=archive)
        more, filled = story.place(fitz.Rect(rect.x0, rect.y0, rect.x1 + 2, BOTTOM))
        story.draw(dev)
        y = fitz.Rect(filled).y1 + GAP[k]
        continue
    html = md(block)
    story = fitz.Story(html=html, user_css=CSS, archive=archive)
    need = 90 if k == "h" else 0
    if y + need > BOTTOM:
        new_page()
        story = fitz.Story(html=html, user_css=CSS, archive=archive)
    more, filled = story.place(fitz.Rect(LEFT, y, RIGHT, BOTTOM))
    if more and (k == "table" or (k == "p" and fitz.Rect(filled).height < 40)) and y > TOP:
        new_page()
        story = fitz.Story(html=html, user_css=CSS, archive=archive)
        more, filled = story.place(fitz.Rect(LEFT, y, RIGHT, BOTTOM))
    story.draw(dev)
    while more:
        new_page()
        more, filled = story.place(fitz.Rect(LEFT, y, RIGHT, BOTTOM))
        story.draw(dev)
    y = fitz.Rect(filled).y1 + GAP[k]
writer.end_page()
writer.close()

doc = fitz.open("pdf", buf.getvalue())
n_body = doc.page_count
doc.insert_pdf(fitz.open(os.path.join(REPORTS, "Motif_Hierarchy_Timeline.pdf")))
for i, page in enumerate(doc):
    x = page.rect.width / 2 if page.rect.width < page.rect.height else page.rect.width - 60
    page.insert_text((x, page.rect.height - 14), f"{i + 1} / {doc.page_count}", fontsize=7, color=(0.4, 0.4, 0.4))
out = os.path.join(REPORTS, NAME + ".pdf")
doc.save(out, garbage=4, deflate=True, deflate_images=True)
doc.close()
print(f"{out}: {n_body} text pages + timeline, {fitz.open(out).page_count} pages, "
      f"{os.path.getsize(out) / 1e6:.1f} MB")
