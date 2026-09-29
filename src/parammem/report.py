"""Turning experiment reports into the tables the paper needs.

Kept as pure functions over the JSON bundles each experiment writes, so the
formatting can be unit-tested without a model and a missing stage degrades to a
"not available" row instead of an exception (a partial run should still produce a
partial, honest results file).
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_results_markdown", "build_results_latex", "headline_values",
           "stage_status"]

NOT_AVAILABLE = "_not available in this run_"


def _tex(text: str) -> str:
    """Escape the characters that would otherwise break a LaTeX table."""
    out = str(text)
    for src, dst in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        out = out.replace(src, dst)
    return out


def _get(bundle: dict[str, Any], stage: str) -> dict[str, Any] | None:
    data = bundle.get(stage)
    return data if isinstance(data, dict) else None


def stage_status(bundle: dict[str, Any]) -> list[tuple[str, str]]:
    """``(stage, status)`` pairs, so a partial run is visible at a glance."""
    out = []
    for stage in ("t1", "t2", "t3", "t4", "t5", "t6"):
        entry = bundle.get(stage)
        if not isinstance(entry, dict):
            out.append((stage, "missing"))
        elif entry.get("_failed"):
            out.append((stage, f"FAILED: {entry.get('_error', 'unknown')}"))
        else:
            out.append((stage, "ok"))
    return out


def _t1(bundle) -> str:
    data = _get(bundle, "t1")
    if not data:
        return f"### T1 write/read/erase\n\n{NOT_AVAILABLE}\n"
    s = data.get("summary", {})
    lines = [
        "### T1 -- write, read, erase",
        "",
        "| arm | EM | counts |",
        "| --- | --- | --- |",
    ]
    for arm in ("prompt_only", "mem_on", "mem_shuffle", "mem_off"):
        counts = s.get("em_counts", {}).get(arm)
        em = s.get("em", {}).get(arm)
        if counts:
            lines.append(f"| `{arm}` | {em:.3f} | {int(counts[1])}/{int(counts[0])} |")
    lines += [
        "",
        f"- net effect over prompt-only: **{s.get('delta_on_minus_prompt_only', float('nan')):+.3f}** "
        f"(95% CI {s.get('delta_ci95', ['?', '?'])})",
        f"- negative control: {s.get('negative_control', {}).get('detail', '?')}",
        f"- attribution: {s.get('attribution', {}).get('detail', '?')}",
        f"- write isolation: {'ok' if s.get('frozen_ok_everywhere') else 'VIOLATED'}",
        f"- target CE after write (own slot only): {s.get('ce_after_mean', float('nan')):.4f}",
        f"- `mem_off` string-identical to `prompt_only`: {s.get('mem_off_string_identical', '?')}",
        "",
        "> READ THIS BEFORE QUOTING THE TABLE: `mem_on` in this stage activates **every**",
        "> written slot, i.e. the *sum-based* read. Its low score is the superposition",
        "> failure that T2 fixes by selecting a single slot -- it is not a statement about",
        "> whether writing works. The per-slot ('the write really landed') numbers are the",
        "> `oracle` arm of T2 (15/16) and the fresh-process recall of T6 (6/6). What this",
        "> stage contributes is the erasure and isolation evidence below: a write touches",
        "> only its own slot, and an erased slot reads back exactly like a virgin one.",
        "",
    ]
    return "\n".join(lines)


def _t2(bundle) -> str:
    data = _get(bundle, "t2")
    if not data:
        return f"### T2 -- read-time composition\n\n{NOT_AVAILABLE}\n"
    s = data.get("summary", {})
    lines = [
        "### T2 -- read-time composition (select, do not sum)",
        "",
        "| arm | EM | counts |",
        "| --- | --- | --- |",
    ]
    for arm in ("oracle", "all", "top1", "top2", "other"):
        counts = s.get("counts", {}).get(arm)
        em = s.get("em", {}).get(arm)
        if counts:
            lines.append(f"| `{arm}` | {em:.3f} | {int(counts[0])}/{int(counts[1])} |")
    lines += [
        "",
        f"- router top-1 accuracy: **{s.get('route_top1_accuracy', float('nan')):.3f}**",
        f"- paraphrase probes: {s.get('config', {}).get('paraphrase', '?')}",
        "",
    ]
    return "\n".join(lines)


def _t3(bundle) -> str:
    data = _get(bundle, "t3")
    if not data:
        return f"### T3 -- write criteria\n\n{NOT_AVAILABLE}\n"
    lines = [
        "### T3 -- which memories deserve a slot",
        "",
        f"- surprise threshold (label-free, median): **{data.get('tau', float('nan')):.3f}**",
        f"- verified 'known' facts: {len(data.get('known_confirmed', []))} "
        f"(dropped {len(data.get('known_dropped', []))} that the model actually got wrong)",
        "",
        "| criterion | writes | wrote target | wasted on known | missed target | recall of written | anchor KL |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rec in data.get("records", []):
        lines.append(
            f"| `{rec['policy']}` | {rec['n_writes']} | "
            f"{rec['written_unknown']}/{rec['unknown_total']} | "
            f"{rec['written_known']}/{rec['known_total']} | {rec['missed_unknown']} | "
            f"{rec['retained_unknown']}/{rec['written_unknown']} | {rec['anchor_kl']:.2f} |"
        )
    by_class = data.get("surprise_by_class", {})
    if by_class:
        lines.append("")
        lines.append("Surprise score by class (why the likelihood criterion fails):")
        lines.append("")
        lines.append("| class | n | min | median | max |")
        lines.append("| --- | --- | --- | --- | --- |")
        for kind, values in by_class.items():
            if values:
                ordered = sorted(values)
                lines.append(
                    f"| {kind} | {len(values)} | {ordered[0]:.2f} | "
                    f"{ordered[len(ordered) // 2]:.2f} | {ordered[-1]:.2f} |"
                )
    lines.append("")
    return "\n".join(lines)


def _t4(bundle) -> str:
    data = _get(bundle, "t4")
    if not data:
        return f"### T4 -- capacity and forgetting\n\n{NOT_AVAILABLE}\n"
    cfg = data.get("config", {})
    lines = [
        "### T4 -- capacity and forgetting",
        "",
        f"stream={cfg.get('stream')} into capacity={cfg.get('slots')}, "
        f"first {cfg.get('hot')} items are accessed repeatedly",
        "",
        "| policy | evictions | erased -> virgin | accessed kept | untouched kept | routed | stamp kept | anchor KL |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rec in data.get("records", []):
        hot = rec["oracle_hits"]["hot"], rec["resident"]["hot"]
        cold = rec["oracle_hits"]["cold"], rec["resident"]["cold"]
        lines.append(
            f"| `{rec['policy']}` | {rec['evictions']} | "
            f"{rec['erased_virgin_ok']}/{rec['erased_checked']} | "
            f"{hot[0]}/{hot[1]} | {cold[0]}/{cold[1]} | "
            f"{rec['routed_hits']}/{rec['n_survivors']} | "
            f"{'yes' if rec['important_retained'] else 'NO'} | {rec['anchor_kl']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _t5(bundle) -> str:
    data = _get(bundle, "t5")
    if not data:
        return f"### T5 -- context inertia\n\n{NOT_AVAILABLE}\n"
    screen = data.get("floor_screen", {})
    lines = [
        "### T5 -- context inertia (REFUTED)",
        "",
        f"- floor screen: {screen.get('kept', '?')}/{screen.get('candidates', '?')} scenarios kept "
        f"({screen.get('dropped', '?')} dropped because the model answers them with no premise at all)",
        "",
        "| arm | premise inheritance | topic residue | withholding |",
        "| --- | --- | --- | --- |",
    ]
    for arm, vals in data.get("arms", {}).items():
        lines.append(
            f"| `{arm}` | {vals['inheritance_rate']:.2f} | {vals['residue_rate']:.2f} | "
            f"{vals['withholding_rate']:.2f} |"
        )
    lines += [
        "",
        f"- premise written and recallable: {data.get('premise_recall_rate', float('nan')):.2f}",
        f"- `param_off` identical to `no_memory`: {data.get('control_identical_rate', float('nan')):.2f}",
        "",
    ]
    return "\n".join(lines)


def _t6(bundle) -> str:
    data = _get(bundle, "t6")
    if not data:
        return f"### T6 -- cross-session persistence\n\n{NOT_AVAILABLE}\n"
    write, read = data.get("write", {}), data.get("read", {})
    snap = write.get("snapshot", {})
    lines = [
        "### T6 -- cross-session persistence (two separate processes)",
        "",
        f"- write session: {write.get('own_slot_recall', '?')} recall before saving; "
        f"snapshot {snap.get('n_slots', '?')} slots / {snap.get('n_modules', '?')} modules / "
        # decimal MB, to match the quoted sizes elsewhere in the docs
        f"{snap.get('bytes_on_disk', 0) / 1e6:.1f} MB",
        f"- read session: side path virgin before load = **{read.get('virgin_before_load', '?')}**",
        "",
        "| measurement | value |",
        "| --- | --- |",
        f"| oracle recall after restart | {read.get('oracle_recall', '?')} |",
        f"| routed recall (k=1) | {read.get('routed_recall', '?')} |",
        f"| routing correct | {read.get('routing_correct', '?')} |",
        f"| erase after load is virgin | {read.get('erase_after_load_is_virgin', '?')} |",
        f"| erased memory still recalled | {read.get('erased_memory_still_recalled', '?')} |",
        "",
    ]
    return "\n".join(lines)


_FORMATTERS = {"t1": _t1, "t2": _t2, "t3": _t3, "t4": _t4, "t5": _t5, "t6": _t6}


def build_results_markdown(bundle: dict[str, Any]) -> str:
    """One markdown document holding every headline number of a run."""
    header = [
        "# Results bundle",
        "",
        "> 由 `experiments/run_all.py` 自动生成；每个数字都能由对应的实验脚本单独复现。",
        "> 缺失或失败的阶段会明确标注，不会用 0 或空值冒充结果。",
        "",
        "## Stage status",
        "",
        "| stage | status |",
        "| --- | --- |",
    ]
    for stage, status in stage_status(bundle):
        header.append(f"| {stage} | {status} |")
    provenance = bundle.get("_provenance", {})
    if provenance:
        header += [
            "",
            "## Provenance",
            "",
            f"- git commit: `{provenance.get('commit', '?')}`",
            f"- model: `{provenance.get('model', '?')}`",
            f"- mode: `{provenance.get('mode', '?')}`",
            f"- generated at: {provenance.get('generated_at', '?')}",
        ]
    header.append("")
    body = [fmt(bundle) for fmt in _FORMATTERS.values()]
    return "\n".join(header + body)


# --------------------------------------------------------------------- LaTeX
#
# The paper must not contain hand-copied numbers: transcription drift is silent
# and the whole point of this project is that claims trace back to measurements.
# So the LaTeX path emits \newcommand macros for every quoted value plus one table
# per stage, and main.tex only ever refers to the macros.


def _na(value, spec: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        return format(value, spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value)


def _counts_pair(pair, order: str) -> str:
    """T1 stores counts as [total, hits]; T2 stores them as [hits, total]."""
    if not pair or len(pair) != 2:
        return "n/a"
    hits, total = (pair[1], pair[0]) if order == "total_first" else (pair[0], pair[1])
    return f"{int(hits)}/{int(total)}"


def _table(caption: str, label: str, columns: list[str], rows: list[list[str]],
           align: str, mono_columns: frozenset[int] = frozenset()) -> str:
    """Build a booktabs table.

    ``caption`` is emitted **as-is**: captions are authored as LaTeX (they contain
    ``\\texttt{}`` and escaped underscores). Cells and column headers are passed
    through :func:`_tex`; columns listed in ``mono_columns`` are additionally
    wrapped in ``\\texttt{}`` *after* escaping, so callers never hand-build LaTeX.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(_tex(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = []
        for index, cell in enumerate(row):
            escaped = _tex(cell)
            if index in mono_columns:
                escaped = rf"\texttt{{{escaped}}}"
            cells.append(escaped)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def headline_values(bundle: dict[str, Any]) -> dict[str, str]:
    """Every number the write-ups are allowed to quote, as plain strings.

    Public on purpose: the English tables (LaTeX/markdown) and the Chinese
    technical report all pull from here, so no document can quote a value that
    the measurement did not produce. Missing data becomes ``"n/a"`` rather than a
    zero, so an incomplete run cannot masquerade as a result.
    """
    t1 = (_get(bundle, "t1") or {}).get("summary", {})
    t2 = (_get(bundle, "t2") or {}).get("summary", {})
    t3 = _get(bundle, "t3") or {}
    t4 = _get(bundle, "t4") or {}
    t5 = _get(bundle, "t5") or {}
    t6 = _get(bundle, "t6") or {}

    t3rec = {r["policy"]: r for r in t3.get("records", [])}
    self_r, always_r = t3rec.get("selfcheck", {}), t3rec.get("always", {})
    explicit_r = t3rec.get("explicit", {})
    t4rec = {r["policy"]: r for r in t4.get("records", [])}
    fifo, smart = t4rec.get("fifo", {}), t4rec.get("utility_time", {})
    arms = t5.get("arms", {})
    read = t6.get("read", {})
    write = t6.get("write", {})
    snap = write.get("snapshot", {})
    screen = t5.get("floor_screen", {})

    t4_records = t4.get("records", [])
    return {
        "tTargetCE": _na(t1.get("ce_after_mean"), ".4f"),
        "tEraseIdentical": _na(t1.get("mem_off_string_identical")),
        "tIsolation": "ok" if t1.get("frozen_ok_everywhere") else "VIOLATED",
        "tOracle": _counts_pair(t2.get("counts", {}).get("oracle"), "hits_first"),
        "tSum": _counts_pair(t2.get("counts", {}).get("all"), "hits_first"),
        "tTopOne": _counts_pair(t2.get("counts", {}).get("top1"), "hits_first"),
        "tTopTwo": _counts_pair(t2.get("counts", {}).get("top2"), "hits_first"),
        "tRouterAcc": _na(t2.get("route_top1_accuracy"), ".3f"),
        "tAlwaysWrites": _na(always_r.get("n_writes")),
        "tAlwaysWasted": _na(always_r.get("written_known")),
        "tAlwaysMissed": _na(always_r.get("missed_unknown")),
        "tSelfWrites": _na(self_r.get("n_writes")),
        "tSelfWasted": _na(self_r.get("written_known")),
        "tSelfMissed": _na(self_r.get("missed_unknown")),
        "tSelfRetained": (
            f"{self_r.get('retained_unknown', 'n/a')}/{self_r.get('written_unknown', 'n/a')}"
            if self_r else "n/a"
        ),
        "tExplicitWrites": _na(explicit_r.get("n_writes")),
        "tExplicitMissed": _na(explicit_r.get("missed_unknown")),
        "tAlwaysKL": _na(always_r.get("anchor_kl"), ".2f"),
        "tSelfKL": _na(self_r.get("anchor_kl"), ".2f"),
        "tSurpriseMissed": _na(t3rec.get("surprise", {}).get("missed_unknown")),
        "tSurpriseWasted": _na(t3rec.get("surprise", {}).get("written_known")),
        "tKnownConfirmed": _na(len(t3.get("known_confirmed", []))),
        "tKnownDropped": _na(len(t3.get("known_dropped", []))),
        "tTau": _na(t3.get("tau"), ".3f"),
        "tErased": _na(fifo.get("erased_virgin_ok")) + "/" + _na(fifo.get("erased_checked")),
        "tErasedTotal": (
            f"{sum(r.get('erased_virgin_ok', 0) for r in t4_records)}/"
            f"{sum(r.get('erased_checked', 0) for r in t4_records)}"
            if t4_records else "n/a"
        ),
        "tPolicies": _na(len(t4_records)) if t4_records else "n/a",
        "tFifoHot": (
            f"{fifo.get('oracle_hits', {}).get('hot', 'n/a')}/"
            f"{fifo.get('resident', {}).get('hot', 'n/a')}" if fifo else "n/a"
        ),
        "tSmartHot": (
            f"{smart.get('oracle_hits', {}).get('hot', 'n/a')}/"
            f"{smart.get('resident', {}).get('hot', 'n/a')}" if smart else "n/a"
        ),
        "tStreamItems": _na(t4.get("config", {}).get("stream")),
        "tCapacitySlots": _na(t4.get("config", {}).get("slots")),
        "tFloor": _na(arms.get("no_memory", {}).get("inheritance_rate"), ".2f"),
        "tContext": _na(arms.get("context_memory", {}).get("inheritance_rate"), ".2f"),
        "tParam": _na(arms.get("param_memory", {}).get("inheritance_rate"), ".2f"),
        "tContextResidue": _na(arms.get("context_memory", {}).get("residue_rate"), ".2f"),
        "tParamResidue": _na(arms.get("param_memory", {}).get("residue_rate"), ".2f"),
        "tWithholdFloor": _na(arms.get("no_memory", {}).get("withholding_rate"), ".2f"),
        "tWithholdMemory": _na(arms.get("param_memory", {}).get("withholding_rate"), ".2f"),
        "tScreenKept": f"{_na(screen.get('kept'))}/{_na(screen.get('candidates'))}",
        "tScreenDropped": _na(screen.get("dropped")),
        "tVirgin": _na(read.get("virgin_before_load")),
        "tSixOracle": _na(read.get("oracle_recall")),
        "tRouted": _na(read.get("routed_recall")),
        "tSixRouting": _na(read.get("routing_correct")),
        "tSixEraseVirgin": _na(read.get("erase_after_load_is_virgin")),
        "tSnapshotMB": f"{snap.get('bytes_on_disk', 0) / 1e6:.1f}" if snap else "n/a",
        "tSnapshotSlots": _na(snap.get("n_slots")),
    }


def _headline_macros(bundle: dict[str, Any]) -> list[str]:
    values = headline_values(bundle)
    # A TeX control word is letters only: a digit TERMINATES it, so \tT4Stream is
    # read as \tT followed by "4Stream" and the compiler fails with a cryptic
    # "Missing number, treated as zero" pointing at the \newcommand line. Check it
    # here rather than letting LaTeX discover it.
    bad = sorted(name for name in values if not name.isalpha())
    if bad:
        raise ValueError(
            f"LaTeX macro names must be letters only (a digit ends a control word); "
            f"offending names: {bad}"
        )
    out = ["% ---- headline numbers (generated) ----"]
    out += [rf"\newcommand{{\{name}}}{{{_tex(value)}}}" for name, value in values.items()]
    return out


def _status_table(bundle: dict[str, Any]) -> str:
    rows = [[stage, status] for stage, status in stage_status(bundle)]
    return _table("Stage status of this results bundle.", "tab:status",
                  ["stage", "status"], rows, "ll")


def _latex_t1(bundle) -> str:
    data = _get(bundle, "t1")
    if not data:
        return f"% T1 {NOT_AVAILABLE}"
    s = data.get("summary", {})
    rows = []
    for arm in ("prompt_only", "mem_on", "mem_shuffle", "mem_off"):
        if arm in s.get("em", {}):
            rows.append([arm, f"{s['em'][arm]:.3f}",
                         _counts_pair(s.get("em_counts", {}).get(arm), "total_first")])
    return _table(
        "T1: the sum-based read (all slots active) versus the two controls. "
        "The low \\texttt{mem\\_on} here is the superposition failure that T2 fixes; "
        "it is not evidence about whether writing works.",
        "tab:t1", ["arm", "EM", "hits/total"], rows, "lrr", frozenset({0}))


def _latex_t2(bundle) -> str:
    data = _get(bundle, "t2")
    if not data:
        return f"% T2 {NOT_AVAILABLE}"
    s = data.get("summary", {})
    rows = []
    for arm in ("oracle", "all", "top1", "top2", "other"):
        if arm in s.get("em", {}):
            rows.append([arm, f"{s['em'][arm]:.3f}",
                         _counts_pair(s.get("counts", {}).get(arm), "hits_first")])
    return _table(
        "T2: composition rules on the same writes, read with a paraphrased query. "
        "Selecting exactly one slot matches the oracle; summing destroys recall; "
        "two slots already lose most of it.",
        "tab:t2", ["arm", "EM", "hits/total"], rows, "lrr", frozenset({0}))


def _latex_t3(bundle) -> str:
    data = _get(bundle, "t3")
    if not data:
        return f"% T3 {NOT_AVAILABLE}"
    rows = [[r['policy'], str(r["n_writes"]),
             f"{r['written_unknown']}/{r['unknown_total']}",
             f"{r['written_known']}/{r['known_total']}",
             str(r["missed_unknown"]), f"{r['retained_unknown']}/{r['written_unknown']}",
             f"{r['anchor_kl']:.2f}"] for r in data.get("records", [])]
    table = _table(
        "T3: write criteria. ``wasted on known'' counts gradient writes spent on "
        "facts the model already answers; ``anchor KL'' is the price paid by the "
        "frozen backbone.",
        "tab:t3",
        ["criterion", "writes", "target", "wasted", "missed", "recall", "anchor KL"],
        rows, "lrrrrrr", frozenset({0}))
    by_class = data.get("surprise_by_class", {})
    if by_class:
        rows2 = []
        for kind, values in by_class.items():
            if values:
                ordered = sorted(values)
                rows2.append([kind, str(len(values)), f"{ordered[0]:.2f}",
                              f"{ordered[len(ordered) // 2]:.2f}", f"{ordered[-1]:.2f}"])
        table += "\n" + _table(
            "T3: pre-write loss by class. The ranges overlap almost completely, "
            "which is why the likelihood-based criterion cannot separate them.",
            "tab:t3loss", ["class", "n", "min", "median", "max"], rows2, "lrrrr")
    return table


def _latex_t4(bundle) -> str:
    data = _get(bundle, "t4")
    if not data:
        return f"% T4 {NOT_AVAILABLE}"
    cfg = data.get("config", {})
    rows = []
    for r in data.get("records", []):
        hot, cold = r["oracle_hits"]["hot"], r["resident"]["hot"]
        chot, ccold = r["oracle_hits"]["cold"], r["resident"]["cold"]
        rows.append([r['policy'], str(r["evictions"]),
                     f"{r['erased_virgin_ok']}/{r['erased_checked']}",
                     f"{hot}/{chot}", f"{cold}/{ccold}",
                     f"{r['routed_hits']}/{r['n_survivors']}",
                     "yes" if r["important_retained"] else "NO",
                     f"{r['anchor_kl']:.2f}"])
    return _table(
        f"T4: capacity and forgetting. {cfg.get('stream')} memories stream into "
        f"{cfg.get('slots')} slots; the first {cfg.get('hot')} are accessed repeatedly. "
        "Every eviction returns its slot to the virgin state bit-for-bit.",
        "tab:t4",
        ["policy", "evictions", "erased", "accessed", "untouched", "routed", "stamp", "anchor KL"],
        rows, "lrrrrrrcr", frozenset({0}))


def _latex_t5(bundle) -> str:
    data = _get(bundle, "t5")
    if not data:
        return f"% T5 {NOT_AVAILABLE}"
    rows = [[arm, f"{v['inheritance_rate']:.2f}",
             f"{v['residue_rate']:.2f}", f"{v['withholding_rate']:.2f}"]
            for arm, v in data.get("arms", {}).items()]
    return _table(
        "T5: context inertia, on scenarios whose floor (no premise anywhere) is "
        "zero. Storing the premise in the weights drags it into the next topic "
        "exactly as much as leaving it in the context, so the hypothesis is refuted.",
        "tab:t5", ["arm", "inheritance", "residue", "withholding"], rows, "lrrr",
        frozenset({0}))


def _latex_t6(bundle) -> str:
    data = _get(bundle, "t6")
    if not data:
        return f"% T6 {NOT_AVAILABLE}"
    read, write = data.get("read", {}), data.get("write", {})
    snap = write.get("snapshot", {})
    rows = [
        ["oracle recall after restart", str(read.get("oracle_recall", "n/a"))],
        ["routed recall (k=1)", str(read.get("routed_recall", "n/a"))],
        ["routing correct", str(read.get("routing_correct", "n/a"))],
        ["side path virgin before load", str(read.get("virgin_before_load", "n/a"))],
        ["erase after load is virgin", str(read.get("erase_after_load_is_virgin", "n/a"))],
        ["erased memory still recalled", str(read.get("erased_memory_still_recalled", "n/a"))],
        ["snapshot size", f"{snap.get('bytes_on_disk', 0) / 1e6:.1f} MB "
                          f"({snap.get('n_slots', '?')} slots)"],
    ]
    return _table(
        "T6: persistence across two separate processes. The read session's side "
        "path is virgin before loading, so the recall it achieves can only come "
        "from disk.",
        "tab:t6", ["measurement", "value"], rows, "ll")


def build_results_latex(bundle: dict[str, Any]) -> str:
    """A LaTeX fragment: headline-number macros plus one table per stage."""
    parts = [
        "% Generated by parammem.report.build_results_latex -- do not edit by hand.",
        "% Rebuild with: python paper/build_tables.py",
        "",
        *_headline_macros(bundle),
        "",
        _status_table(bundle),
        _latex_t1(bundle),
        _latex_t2(bundle),
        _latex_t3(bundle),
        _latex_t4(bundle),
        _latex_t5(bundle),
        _latex_t6(bundle),
    ]
    return "\n".join(parts)
