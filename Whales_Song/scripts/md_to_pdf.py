import os
import fitz
import mistune

REPORTS = r"c:\Users\hoang\project\stumpy\Whales_Song\reports"
CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.35; color: #222; }
h1 { font-size: 18pt; margin-bottom: 6px; }
h2 { font-size: 14pt; margin-top: 16px; margin-bottom: 4px; }
h3 { font-size: 12pt; margin-top: 12px; margin-bottom: 4px; }
p { margin-top: 3px; margin-bottom: 5px; }
table { border-collapse: collapse; margin-top: 4px; margin-bottom: 8px; font-size: 8pt; }
th, td { border: 1px solid #999; padding: 2px 4px; }
th { background-color: #e8ecf2; }
code { font-family: monospace; font-size: 8.5pt; }
a { color: #1a5fb4; text-decoration: none; }
li { margin-bottom: 2px; }
"""
PAGE = fitz.paper_rect("a4")
WHERE = PAGE + (45, 45, -45, -45)

md = mistune.create_markdown(plugins=["table", "strikethrough"], escape=False)
archive = fitz.Archive(REPORTS)
for name in ("Phase1_Length_Range_Report", "Phase2_Forward_Motif_Report"):
    text = open(os.path.join(REPORTS, name + ".md"), encoding="utf-8").read()
    html = md(text).replace("<img ", '<img width="100%" ')
    story = fitz.Story(html=html, user_css=CSS, archive=archive)
    pdf_path = os.path.join(REPORTS, name + ".pdf")
    writer = fitz.DocumentWriter(pdf_path)
    more = True
    while more:
        dev = writer.begin_page(PAGE)
        more, _ = story.place(WHERE)
        story.draw(dev)
        writer.end_page()
    writer.close()
    doc = fitz.open(pdf_path)
    print(name, doc.page_count, "pages", os.path.getsize(pdf_path) // 1024, "KB")
