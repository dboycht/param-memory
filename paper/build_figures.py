"""Generate the paper's data figure as TikZ, from the same bundle the tables use.

Two reasons it is generated rather than drawn. The numbers must not drift from the tables:
a figure that disagrees with the table beside it is worse than no figure, and this project
has already been bitten by hand-carried numbers more than once. And TikZ inherits the
document's fonts, so the figure matches the body text without any font wrangling.

Coordinates are computed here rather than by a plotting package, which keeps the tick
placement and the label offsets under explicit control; overlapping text inside a shipped
figure is the failure mode this avoids.

Usage::

    python paper/build_figures.py                     # from runs/summary.json
    python paper/build_figures.py --bundle path.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(stream := _stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# Geometry in centimetres. The plot is wide and short, which leaves room for a legend row
# below the axes instead of on top of the data.
PLOT_W, PLOT_H = 12.0, 5.2
X_MIN, X_MAX = 1.0, 60.0
Y_MIN, Y_MAX = 0.0, 1.0


def x_pos(value: float) -> float:
    """Log scale: the doses span 1 to 60, so a linear axis would crowd the small end."""
    lo, hi = math.log10(X_MIN), math.log10(X_MAX)
    return (math.log10(value) - lo) / (hi - lo) * PLOT_W


def y_pos(value: float) -> float:
    return (value - Y_MIN) / (Y_MAX - Y_MIN) * PLOT_H


def marker(kind: str, x: float, y: float) -> str:
    """A marker drawn by hand: filled circle, open square, open triangle.

    Shape and fill, not colour, carry the distinction, so the figure survives grayscale
    printing.
    """
    if kind == "filled-circle":
        return f"\\fill[black] ({x:.3f},{y:.3f}) circle (2.1pt);"
    if kind == "open-square":
        return (f"\\draw[black,line width=0.7pt,fill=white] "
                f"({x:.3f},{y:.3f}) rectangle +(4.2pt,4.2pt);")
    if kind == "open-triangle":
        return (f"\\draw[black,line width=0.7pt,fill=white] ({x:.3f},{y:.3f}) "
                f"-- ++(3.4pt,0) -- ++(-1.7pt,5.0pt) -- cycle;")
    raise ValueError(kind)


def series(points: list[tuple[float, float]], style: str, kind: str) -> str:
    coords = " -- ".join(f"({x_pos(x):.3f},{y_pos(y):.3f})" for x, y in points)
    out = [f"\\draw[black,{style}] {coords};" if len(points) > 1
           else f"\\draw[black,{style}] ({x_pos(points[0][0]):.3f},"
                f"{y_pos(points[0][1]):.3f}) -- ++(0.001,0);"]
    for x, y in points:
        out.append("    " + marker(kind, x_pos(x), y_pos(y)))
    return "\n    ".join(out)


def build(bundle: dict, lang: str = "en") -> str:
    from parammem.report import headline_values

    v = headline_values(bundle)
    # The Chinese review copy gets its own labels rather than the English figure: a
    # reader checking the translation should not have to read the axis in another language.
    words = {
        "en": {
            "xlabel": "memory slots held open at read time",
            "ylabel": "containment",
            "real": "real content, 60 questions",
            "synthetic": "synthetic 0.6B, last-token key",
            "oracle": "oracle, one slot",
            "caption": (
                r"Containment against the number of memory slots held open at read time, on "
                r"a logarithmic axis. One slot is as good as the oracle; two already cost, "
                r"and the loss grows monotonically until the whole bank is open, where "
                r"$\tSum$ of the synthetic questions are answered. The real-content curve "
                r"falls more gently than the synthetic one but never turns upward, so the "
                r"generator exaggerates the size of the effect without creating it "
                r"(Table~\ref{tab:t7collision})."),
            "ref": r"Figure~\ref{fig:collapse}",
        },
        "zh": {
            "xlabel": "读取时同时打开的槽数",
            "ylabel": "包含率",
            "real": "真实内容（60 题）",
            "synthetic": "合成数据 0.6B（原始键）",
            "oracle": "oracle（只开一个槽）",
            "caption": (
                r"读取时同时打开的槽数与包含率的关系（横轴为对数轴）。只开一个槽时与 oracle 相当；"
                r"开到两个就已经付出代价，此后单调下降，直到整库全开时合成题全部答不出"
                r"（$\tSum$）。真实内容的曲线比合成数据更平缓，但从不回升 —— "
                r"也就是说生成器放大了效应的大小、并没有凭空制造它（见 "
                r"Table~\ref{tab:t7collision}）。"),
            "ref": r"图~\ref{fig:collapse}",
        },
    }[lang]

    def num(key: str) -> float | None:
        raw = v.get(key)
        if raw in (None, "n/a"):
            return None
        try:
            return float(str(raw).split("/")[0]) / float(str(raw).split("/")[1]) \
                if "/" in str(raw) else float(raw)
        except (ValueError, IndexError):
            return None

    # Real content: one to sixty slots open, sixty questions.
    real = [(1, num("tRealCollisionOne")), (2, num("tRealCollisionTwo")),
            (3, num("tRealCollisionThree")), (5, num("tRealCollisionFive")),
            (60, num("tRealCollisionAll"))]
    # Synthetic, last-token key: the same read condition on generated episodes, whose
    # totals are the composition arms (one slot, two slots, the whole bank). The 1.7B
    # series is deliberately absent: at two slots it sits 0.025 from the 0.6B one, so its
    # markers touched, and the scale control is in the table anyway.
    small = [(1, num("tTopOne")), (2, num("tTopTwo")), (8, num("tSum"))]
    oracle = num("tOracle")

    if any(value is None for _x, value in real):
        return "% figure: the collision runs are not in this bundle\n"

    parts = [r"\begin{figure}[t]", r"\centering",
             r"\begin{tikzpicture}[x=1cm,y=1cm]",
             f"  \\draw[black,line width=0.6pt] (0,0) -- ({PLOT_W:.2f},0);",
             f"  \\draw[black,line width=0.6pt] (0,0) -- (0,{PLOT_H:.2f});"]
    # Horizontal grid at quarter marks, light enough not to compete with the data.
    for value in (0.25, 0.5, 0.75, 1.0):
        parts.append(f"  \\draw[black!18,line width=0.3pt] (0,{y_pos(value):.3f}) -- "
                     f"({PLOT_W:.2f},{y_pos(value):.3f});")
    # Ticks and labels: x below the axis, y left of it, with a fixed gap so nothing touches.
    for value in (1, 2, 3, 5, 8, 60):
        parts.append(f"  \\draw[black,line width=0.6pt] ({x_pos(value):.3f},0) -- "
                     f"++(0,-0.10);")
        parts.append(f"  \\node[below=0.14cm,font=\\small] at ({x_pos(value):.3f},0) "
                     f"{{{value}}};")
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f"  \\draw[black,line width=0.6pt] (0,{y_pos(value):.3f}) -- ++(-0.10,0);")
        parts.append(f"  \\node[left=0.16cm,font=\\small] at (0,{y_pos(value):.3f}) "
                     f"{{{value:g}}};")
    # Axis titles, offset far enough that neither can reach the tick labels.
    parts.append(f"  \\node[below=0.62cm,font=\\small] at ({PLOT_W / 2:.2f},0) "
                 f"{{{words['xlabel']}}};")
    parts.append(f"  \\node[rotate=90,above=0.72cm,font=\\small] at (0,{PLOT_H / 2:.2f}) "
                 f"{{{words['ylabel']}}};")
    # The oracle ceiling as a reference line, labelled in the legend rather than inline so
    # it cannot collide with the real-content series that starts at the same height.
    if oracle is not None:
        parts.append(f"  \\draw[black!55,line width=0.5pt,dash pattern=on 1pt off 1.6pt] "
                     f"(0,{y_pos(oracle):.3f}) -- ({PLOT_W:.2f},{y_pos(oracle):.3f});")
    # Data series. Offsets are in TikZ units; the two series are separated by style and
    # marker shape so grayscale printing keeps them apart.
    parts.append("  % synthetic, 0.6B, last-token key")
    parts.append("    " + series([(x, y) for x, y in small if y is not None],
                                 "dash pattern=on 2.4pt off 1.6pt,line width=0.8pt",
                                 "open-square"))
    parts.append("  % real benchmark content, 60 questions")
    parts.append("    " + series([(x, y) for x, y in real if y is not None],
                                 "line width=1.0pt", "filled-circle"))
    # Legend below the axes: a swatch and a label per row, so no text sits over the plot.
    legend = [(words["real"], "line width=1.0pt", "filled-circle", 1.0),
              (words["synthetic"],
               "dash pattern=on 2.4pt off 1.6pt,line width=0.8pt", "open-square", 2.0),
              (f"{words['oracle']} ({oracle:.2f})" if oracle is not None else words["oracle"],
               "dash pattern=on 1pt off 1.6pt,line width=0.5pt", "none", 8.0)]
    base = -1.32
    parts.append("  % legend, outside the axes")
    for index, (label, style, kind, sample_x) in enumerate(legend):
        row = base - 0.52 * (index // 2)
        col = (index % 2) * (PLOT_W / 2)
        sx = col + 0.35
        if kind != "none":
            parts.append(f"  \\draw[black,{style}] ({sx:.2f},{row:.2f}) -- "
                         f"({sx + 0.95:.2f},{row:.2f});")
            if kind != "none":
                parts.append("    " + marker(kind, sx + 0.475, row)
                             .replace("circle", "circle").replace("rectangle", "rectangle"))
        else:
            parts.append(f"  \\draw[black,{style}] ({sx:.2f},{row:.2f}) -- "
                         f"({sx + 0.95:.2f},{row:.2f});")
        safe = label.replace("_", r"\_")
        parts.append(f"  \\node[right=0.16cm,font=\\small] at ({sx + 1.02:.2f},{row:.2f}) "
                     f"{{{safe}}};")
    parts += ["\\end{tikzpicture}",
              "\\caption{" + words["caption"] + "}",
              r"\label{fig:collapse}", r"\end{figure}", ""]
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default=str(ROOT / "runs" / "summary.json"))
    ap.add_argument("--lang", choices=("en", "zh"), default="en")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_path = pathlib.Path(args.out) if args.out else (
        ROOT / "paper" / ("generated_figure_collapse_zh.tex" if args.lang == "zh"
                          else "generated_figure_collapse.tex"))
    bundle = json.loads(pathlib.Path(args.bundle).read_text(encoding="utf-8"))
    text = build(bundle, lang=args.lang)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path} : {len(text)} bytes, "
          f"{text.count(chr(92) + 'draw')} draw commands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
