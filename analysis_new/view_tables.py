# Renders LaTeX tabular fragments as HTML and opens them in a browser.

import re
import webbrowser
import tempfile
from pathlib import Path

LATEX_DIR = Path(__file__).parent / "output_artifacts" / "latex"

def texify(s: str) -> str:
    result = []
    i = 0
    while i < len(s):
        if s[i:i+8] == r"\textbf{":
            content, end = _extract_braced(s, i + 7)
            result.append(f"<strong>{texify(content)}</strong>")
            i = end
        elif s[i:i+6] == r"\emph{":
            content, end = _extract_braced(s, i + 5)
            result.append(f"<em>{texify(content)}</em>")
            i = end
        else:
            result.append(s[i])
            i += 1

    s = "".join(result)

    def _math(m):
        t = m.group(1)
        t = re.sub(r"\^\\dagger", "<sup>†</sup>", t)
        t = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", t)
        t = re.sub(r"\^(\w)", r"<sup>\1</sup>", t)
        t = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", t)
        t = re.sub(r"_(\w)", r"<sub>\1</sub>", t)
        t = re.sub(r"\\[a-zA-Z]+", "", t)
        return t

    s = re.sub(r"\$([^$]+)\$", _math, s)

    s = re.sub(r"\\(?:footnotesize|small|normalsize|large|centering)\s*", "", s)

    s = s.replace(r"\%", "%")
    s = s.replace(r"\&", "&amp;")
    s = s.replace(r"\#", "#")
    s = s.replace("--", "–")

    return s.strip()

def _extract_braced(s: str, open_pos: int):
    assert s[open_pos] == "{"
    depth = 0
    k = open_pos
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[open_pos + 1 : k], k + 1
        k += 1
    return s[open_pos + 1 :], len(s)

def _split_cells(row: str):
    cells, current, depth = [], [], 0
    for ch in row:
        if ch == "{":
            depth += 1
            current.append(ch)
        elif ch == "}":
            depth -= 1
            current.append(ch)
        elif ch == "&" and depth == 0:
            cells.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        cells.append("".join(current))
    return cells

def _render_cell(cell: str) -> str:
    cell = cell.strip()
    mc = re.match(
        r"^\\multicolumn\{(\d+)\}\{([^}]*)\}\{(.*)\}$", cell, re.DOTALL
    )
    if mc:
        colspan = mc.group(1)
        raw_align = re.sub(r"[|@{}p]", "", mc.group(2)).strip()
        align_css = {"l": "left", "r": "right"}.get(raw_align[:1], "center")
        text = texify(mc.group(3))
        return f'<td colspan="{colspan}" style="text-align:{align_css};font-size:11px">{text}</td>'
    return f"<td>{texify(cell)}</td>"

def parse_tabular(tex: str) -> str:
    tex = re.sub(r"\\begin\{tabular\}\*?\{[^}]+\}", "", tex)
    tex = re.sub(r"\\end\{tabular\}", "", tex)

    items = []

    for raw in re.split(r"\\\\", tex):
        raw = raw.strip()
        if not raw:
            continue

        top_border = bot_border = None

        if re.search(r"\\toprule", raw):
            raw = re.sub(r"\\toprule", "", raw).strip()
            top_border = "2px solid #222"

        if re.search(r"\\midrule", raw):
            raw = re.sub(r"\\midrule", "", raw).strip()
            top_border = "1px solid #777"

        if re.search(r"\\bottomrule", raw):
            raw = re.sub(r"\\bottomrule", "", raw).strip()
            bot_border = "2px solid #222"

        raw = re.sub(r"\\cmidrule(?:\([a-z]+\))?\{[^}]+\}", "", raw).strip()
        raw = re.sub(r"\\addlinespace\b", "", raw).strip()
        raw = re.sub(r"\\hline\b", "", raw).strip()

        if raw:
            if bot_border and items:
                items[-1]["bot"] = bot_border
                bot_border = None
            items.append({"raw": raw, "top": top_border, "bot": bot_border})
        elif bot_border and items:
            items[-1]["bot"] = bot_border

    rows_html = []
    for item in items:
        parts = []
        if item["top"]:
            parts.append(f"border-top:{item['top']}")
        if item["bot"]:
            parts.append(f"border-bottom:{item['bot']}")
        tr_style = f' style="{";".join(parts)}"' if parts else ""
        cells = "".join(_render_cell(c) for c in _split_cells(item["raw"]))
        rows_html.append(f"  <tr{tr_style}>{cells}</tr>")

    return '<table class="lt">\n' + "\n".join(rows_html) + "\n</table>"

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif; font-size: 13px;
  display: flex; height: 100vh; overflow: hidden; background: #f0f0f0;
}
#sidebar {
  width: 210px; background: #1a1a2e; color: #c9d1d9; overflow-y: auto;
  flex-shrink: 0; padding: 10px 0;
}
#sidebar h2 {
  font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
  color: #58a6ff; padding: 10px 14px 6px;
}
#sidebar a {
  display: block; padding: 5px 14px; color: #c9d1d9;
  text-decoration: none; font-size: 11.5px;
  border-left: 3px solid transparent; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis;
}
#sidebar a:hover { background: #21262d; }
#sidebar a.active { background: #161b22; border-left-color: #58a6ff; color: #58a6ff; }
#main { flex: 1; overflow-y: auto; padding: 20px 28px; }
.section {
  background: #fff; border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0,0,0,.12); padding: 16px 20px; margin-bottom: 20px;
}
.section h3 {
  font-size: 12px; color: #555; margin-bottom: 12px; font-weight: 600;
  padding-bottom: 6px; border-bottom: 1px solid #e8e8e8; letter-spacing: .3px;
}
table.lt { border-collapse: collapse; font-size: 12px; width: auto; }
table.lt td {
  padding: 4px 12px; vertical-align: middle; white-space: nowrap;
  color: #1c1c1c;
}
table.lt tr:first-child td { background: #f7f9fc; font-weight: 600; }
"""

_JS = """
const links = document.querySelectorAll('#sidebar a');
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      links.forEach(l => l.classList.remove('active'));
      const a = document.querySelector('#sidebar a[href="#' + e.target.id + '"]');
      if (a) a.classList.add('active');
    }
  });
}, { threshold: 0.2 });
document.querySelectorAll('.section').forEach(s => observer.observe(s));
"""

def _build_page(tables: dict) -> str:
    nav = "\n  ".join(
        f'<a href="#{_id(k)}">{k}</a>' for k in tables
    )
    body = "\n".join(
        f'<div class="section" id="{_id(k)}"><h3>{k}</h3>{v}</div>'
        for k, v in tables.items()
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LaTeX Table Viewer</title>
  <style>{_CSS}</style>
</head>
<body>
  <nav id="sidebar">
    <h2>Tables</h2>
    {nav}
  </nav>
  <div id="main">{body}</div>
  <script>{_JS}</script>
</body>
</html>"""

def _id(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", key)

if __name__ == "__main__":
    if not LATEX_DIR.exists():
        print(f"Not found: {LATEX_DIR}")
        raise SystemExit(1)

    tables = {}
    for f in sorted(LATEX_DIR.rglob("*.tex")):
        label = f"{f.parent.name}/{f.stem}"
        try:
            tables[label] = parse_tabular(f.read_text(encoding="utf-8"))
        except Exception as exc:
            tables[label] = f'<p style="color:red">Parse error: {exc}</p>'

    html = _build_page(tables)
    out = Path(tempfile.mkdtemp()) / "latex_tables.html"
    out.write_text(html, encoding="utf-8")
    webbrowser.open(out.as_uri())
    print(f"Opened in browser: {out}")
