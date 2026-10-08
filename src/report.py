"""Build a local reading report from the maintained project documentation."""

from pathlib import Path
import re
import markdown

ROOT = Path(__file__).resolve().parents[1]


def main():
    chapters = [
        ("analysis", "分析结果"),
        ("data", "数据与 SQL"),
        ("experiment", "实验设计"),
        ("causal", "因果方法"),
    ]
    sections = []
    for name, title in chapters:
        text = (ROOT / "docs" / f"{name}.md").read_text()
        # Preserve TeX blocks through Markdown conversion.
        blocks = []

        def hold(m):
            blocks.append(m.group())
            return f"FORMULA{len(blocks)-1}END"

        text = re.sub(r"\$\$[\s\S]*?\$\$", hold, text)
        html = markdown.markdown(text, extensions=["tables", "fenced_code"])
        for i, block in enumerate(blocks):
            html = html.replace(f"FORMULA{i}END", block)
        html = html.replace("../results/", "").replace("../src/", "../src/")
        for anchor, _ in chapters:
            html = html.replace(f'href="{anchor}.md', f'href="#{anchor}')
        sections.append(f'<section id="{name}">{html}</section>')
    nav = "".join(f'<a href="#{name}">{title}</a>' for name, title in chapters)
    css = """body{margin:0;background:#f5f6f8;color:#182432;font:16px/1.8 system-ui,-apple-system,"PingFang SC",sans-serif}nav{position:sticky;top:0;background:#133744;padding:14px 24px;display:flex;gap:24px;z-index:1}nav a{color:#fff;text-decoration:none}main{max-width:1020px;margin:auto;padding:20px 32px 80px;background:#fff}section{padding:24px 0;border-bottom:1px solid #ddd;scroll-margin-top:70px}h1{font-size:30px}h2{font-size:23px;margin-top:36px;color:#155a6f}h3{margin-top:26px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px 12px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf4f5}pre{background:#f2f5f7;padding:16px;overflow:auto;line-height:1.55}code{font-size:14px}a{color:#156a8a} @media print{nav{display:none}main{padding:0}section{break-before:page}}"""
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GA4 电商增长诊断与实验评估</title><style>{css}</style><nav>{nav}</nav><main>{''.join(sections)}</main><script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['$$','$$']]}}}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script></html>"""
    (ROOT / "results/report.html").write_text(page)
    print("Created results/report.html")


if __name__ == "__main__":
    main()
