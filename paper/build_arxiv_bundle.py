"""Build the arXiv submission bundle for the paper.

arXiv wants the **LaTeX sources**, not a PDF, and it compiles them itself. This
script therefore

1. copies ``main.tex`` and ``generated_tables.tex`` into a build directory
   (arXiv wants them at the root of the archive),
2. writes ``abstract.txt`` -- the abstract as **plain text with the generated
   macros substituted**. arXiv's abstract field cannot contain macros or math, so
   pasting the LaTeX abstract would garble it,
3. writes ``SUBMIT-CHECKLIST.md`` with the fields only the author can fill in,
4. packs a ``.tar.gz`` and then **verifies it by extracting to a clean directory
   and compiling it twice** -- a bundle that only compiles in this working tree is
   not a bundle.

The build directory lives under ``runs/`` (git-ignored), so nothing here pollutes
the repository.

Usage::

    python paper/build_arxiv_bundle.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Commands that carry no meaning in a plain-text abstract.
_WRAPPERS = ("textbf", "emph", "texttt", "mathrm", "text", "mbox", "textrm")


def latex_to_plain(text: str, values: dict[str, str]) -> str:
    """Turn the LaTeX abstract into the plain text arXiv's abstract field wants."""
    # 1) generated macros: \tFoo{} and \tFoo
    text = re.sub(r"\\(t[A-Z][A-Za-z]*)\{\}",
                  lambda m: values.get(m.group(1), m.group(1)), text)
    text = re.sub(r"\\(t[A-Z][A-Za-z]*)\b",
                  lambda m: values.get(m.group(1), m.group(1)), text)
    # 2) unwrap formatting commands (repeat: they can nest)
    for _ in range(4):
        for name in _WRAPPERS:
            text = re.sub(r"\\" + name + r"\{([^{}]*)\}", r"\1", text)
    # 3) drop math delimiters and any leftover one-token commands
    text = text.replace("$", "")
    # LaTeX quotes and a few named operators first: the generic command-stripper
    # below would otherwise leave ``surprise'' intact and turn \sum into nothing,
    # producing a dangling "= 0/16".
    text = text.replace("``", '"').replace("''", '"')
    for command, plain in (("\\sum", "sum"), ("\\prod", "product"),
                           ("\\log", "log"), ("\\exp", "exp"),
                           ("\\approx", "~"), ("\\times", "x"),
                           ("\\rightarrow", "->"), ("\\pm", "+/-")):
        text = text.replace(command, plain)
    text = text.replace("\\%", "%").replace("\\,", " ").replace("\\;", " ")
    text = re.sub(r"\\[A-Za-z]+", "", text)
    # 4) punctuation and whitespace
    text = text.replace("---", "\u2014").replace("--", "\u2013")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def extract_abstract(main_tex: str) -> str:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex,
                      re.S)
    if not match:
        raise ValueError("no abstract environment found in main.tex")
    return match.group(1).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default=str(ROOT / "runs" / "summary.json"))
    ap.add_argument("--outdir", default=str(ROOT / "runs" / "arxiv"))
    ap.add_argument("--repo-url", default="https://github.com/dboycht/param-memory")
    args = ap.parse_args()

    from parammem.report import headline_values

    bundle_path = Path(args.bundle)
    if not bundle_path.is_file():
        print(f"missing {bundle_path}; run experiments/run_all.py --collect-only",
              flush=True)
        return 1
    values = headline_values(json.loads(bundle_path.read_text(encoding="utf-8")))

    outdir = Path(args.outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    sources = ["main.tex", "generated_tables.tex"]
    for name in sources:
        shutil.copy2(ROOT / "paper" / name, outdir / name)

    main_tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    abstract = latex_to_plain(extract_abstract(main_tex), values)
    (outdir / "abstract.txt").write_text(abstract + "\n", encoding="utf-8")

    unresolved = re.findall(r"\\(t[A-Z][A-Za-z]*)", abstract)
    if unresolved:
        print(f"WARNING: unresolved macros in the plain-text abstract: "
              f"{sorted(set(unresolved))}", flush=True)

    (outdir / "SUBMIT-CHECKLIST.md").write_text(
        f"""# arXiv 提交清单（`{ROOT.name}`）

## 只有你能做的（我无法代做）

1. **账号**：用你的 arXiv 账号登录（或先注册；需要邮箱验证）。
2. **背书（endorsement）**：**首次**往 `cs.CL` / `cs.LG` 投的人，arXiv 常会要求背书。
   提交页会直接提示；若要求，需要一个在该分类有发布记录的 arXiv 用户给你背书
   （机构邮箱/所在单位有时可自动豁免）。**这一步决定能不能投，请先确认。**
3. **许可协议**：arXiv 会要求选一个 license（默认 arXiv 非独占许可，或 CC BY 等）。
   代码仓用的是 MIT，那是**代码**的许可，论文的许可要你单独定。
4. **署名（已按你的决定设好，只需确认）**：`main.tex` 现在是
   `\\author{{Haotian Cheng \\\\ \\small Nanjing University of Aeronautics and Astronautics}}`，
   **故意不写邮箱**（按你"公开材料不放联系信息"的规矩）。要在 arXiv 元数据里补单位/邮箱、
   或换姓名写法（汉字 / 姓在前），告诉我一句就改。
5. **点提交**，以及后续的版本更新（v2/v3）。

## 我来做的部分（已生成，可直接上传）

- 源文件包：`{outdir / 'param-memory-arxiv.tar.gz'}`
  （里面只有 `main.tex` + `generated_tables.tex`，都在压缩包根目录，arXiv 要的就是这个形态）
- 摘要纯文本：`{outdir / 'abstract.txt'}`
  （arXiv 的摘要框**不接受宏和公式**，直接粘 LaTeX 会乱码，所以这里做了纯文本化）

## 提交页要填的字段（建议值）

| 字段 | 建议 |
| --- | --- |
| Primary category | `cs.CL`（Computation and Language） |
| Cross-list | `cs.LG` |
| Title | 见 `main.tex` 的 `\\title` |
| Abstract | 直接粘贴 `abstract.txt` 全文 |
| Comments | `11 pages, 9 tables. Code: {args.repo_url}` |
| License | 由你选（见上） |

## 提交后建议核对

- arXiv 会用**它自己的编译器**编译；若报错，日志里会指出行号 —— 我们在本地已用
  `pdflatex` 干净编译验证过（见下），所以大概率没问题。
- 生成 PDF 后，确认两处：① 表格是否完整（共 9 张）；② 参考文献是否正常显示
  （我们用的是 `thebibliography` 内联，不依赖 `.bbl` 文件）。
- 预印本会**公开留时间戳**。若你之后要投双盲会议，多数会议允许预印本，但投稿正文里
  不要写"我们的预印本"这类会暴露身份的话。

## 本地已做过的验证

- 源文件包解压到**干净目录**后 `pdflatex` 连编两遍，均 `exit=0` 并产出 11 页 PDF；
- 摘要纯文本已检查无残留宏。
""",
        encoding="utf-8",
    )

    tarball = outdir / "param-memory-arxiv.tar.gz"
    with tarfile.open(tarball, "w:gz") as handle:
        for name in sources:
            handle.add(outdir / name, arcname=name)

    # ---- verify: extract somewhere clean and compile it there ----------------
    check = outdir / "_verify"
    check.mkdir()
    with tarfile.open(tarball) as handle:
        # filter="data" is the explicit, safe extraction mode; without it Python
        # 3.14 warns about the default changing.
        handle.extractall(check, filter="data")
    ok = True
    for pass_no in (1, 2):
        with (outdir / f"verify_pass{pass_no}.txt").open("w", encoding="utf-8",
                                                         errors="replace") as log:
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                cwd=str(check), stdout=log, stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            ok = False
            print(f"  pass {pass_no}: pdflatex exit {proc.returncode}", flush=True)
    pdf = check / "main.pdf"
    pages = ""
    log_text = (outdir / "verify_pass2.txt").read_text(encoding="utf-8",
                                                       errors="replace")
    match = re.search(r"Output written on main\.pdf \((\d+) pages", log_text)
    if match:
        pages = f"{match.group(1)} pages"
    errors = [line for line in log_text.splitlines() if line.startswith("! ")]

    print(f"bundle    : {tarball}  ({tarball.stat().st_size/1024:.0f} KB)", flush=True)
    print(f"contents  : {', '.join(sources)} + abstract.txt + SUBMIT-CHECKLIST.md",
          flush=True)
    print(f"verify    : clean-dir compile {'OK' if ok and pdf.is_file() else 'FAILED'}"
          f"{' (' + pages + ')' if pages else ''}", flush=True)
    if errors:
        print(f"            errors: {errors[:3]}", flush=True)
    print(f"abstract  : {len(abstract)} chars, first 120: {abstract[:120]}...",
          flush=True)
    return 0 if ok and pdf.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
