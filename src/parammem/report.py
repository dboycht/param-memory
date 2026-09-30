"""Turning experiment reports into the tables the paper needs.

Kept as pure functions over the JSON bundles each experiment writes, so the
formatting can be unit-tested without a model and a missing stage degrades to a
"not available" row instead of an exception (a partial run should still produce a
partial, honest results file).
"""

from __future__ import annotations

import re
from typing import Any

from .eval.paired import paired_counts, sign_test_p

__all__ = ["build_results_markdown", "build_results_latex", "headline_values",
           "replication_verdict", "stage_status"]

NOT_AVAILABLE = "_not available in this run_"

#: Row sentinel for :func:`_table`; a row of exactly ``[MIDRULE]`` becomes a rule.
MIDRULE = "<midrule>"


#: Single-pass escaping map. A sequential ``str.replace`` chain is wrong here: the
#: replacement for a backslash introduces braces, and a later step would escape those
#: too, turning ``\quad`` into ``\textbackslash\{\}quad``.
_TEX_MAP = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "$": r"\$"}
_TEX_CHARS = re.compile(r"[\\&%#_{}$]")


def _tex(text: str) -> str:
    """Escape the characters that would otherwise break a LaTeX table.

    Callers pass **plain text**. If a cell contains LaTeX, this function turns it into
    its own source, which is how an entire table reached a compiled PDF reading
    ``\\{}texttt{frozen} containment``.
    """
    return _TEX_CHARS.sub(lambda match: _TEX_MAP[match.group(0)], str(text))


def _get(bundle: dict[str, Any], stage: str) -> dict[str, Any] | None:
    data = bundle.get(stage)
    return data if isinstance(data, dict) else None


def stage_status(bundle: dict[str, Any]) -> list[tuple[str, str]]:
    """``(stage, status)`` pairs, so a partial run is visible at a glance."""
    out = []
    for stage in ("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"):
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
    ]

    # Per-episode detail: a single aggregate hides how much of the result is seed
    # variance, which is exactly what a multi-seed re-run exists to expose.
    episodes = data.get("episodes") or []
    if len(episodes) > 1:
        arms = ("oracle", "all", "top1", "top2", "other")
        lines += ["", "| episode | seed | " + " | ".join(f"`{a}`" for a in arms) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in arms) + " |"]
        spread: dict[str, list[float]] = {arm: [] for arm in arms}
        for episode in episodes:
            row = []
            for arm in arms:
                values = (episode.get("arms") or {}).get(arm) or []
                total = len(values)
                hits = sum(values)
                rate = hits / total if total else float("nan")
                spread[arm].append(rate)
                row.append(f"{hits}/{total}")
            lines.append(f"| {episode.get('episode_id', '?')} | {episode.get('seed', '?')} "
                         f"| " + " | ".join(row) + " |")
        lines += ["", "| arm | min | max | spread |", "| --- | --- | --- | --- |"]
        for arm in arms:
            rates = spread[arm]
            lines.append(f"| `{arm}` | {min(rates):.3f} | {max(rates):.3f} | "
                         f"{max(rates) - min(rates):.3f} |")

    scale = (data.get("scale_17b") or {}).get("summary") or {}
    if scale.get("em"):
        em = scale["em"]
        lines += [
            "",
            f"**Scale control (1.7B backbone, same protocol):** "
            f"`oracle` {em.get('oracle', float('nan')):.3f}, "
            f"`top1` {em.get('top1', float('nan')):.3f}, "
            f"`top2` {em.get('top2', float('nan')):.3f}, "
            f"`all` {em.get('all', float('nan')):.3f}, "
            f"router {scale.get('route_top1_accuracy', float('nan')):.3f}",
        ]
    lines.append("")
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


def _t7(bundle) -> str:
    data = _get(bundle, "t7")
    if not data:
        return ("### T7-d -- public benchmark (controlled diagnostic)\n\n"
                f"{NOT_AVAILABLE}\n")
    c = data.get("containment") or {}
    write = data.get("write") or {}
    ids = data.get("subset_ids") or []
    leaking = data.get("p1_leaking_ids") or []
    lines = [
        "### T7-d -- LongMemEval controlled diagnostic (single-session subset)",
        "",
        f"- subset: {data.get('n_subset', '?')} questions + "
        f"{len(data.get('negative_ids') or [])} negative controls; "
        f"data sha256 `{str(data.get('data_sha256', ''))[:16]}...`",
        f"- P1 eviction check: {len(ids) - len(leaking)}/{len(ids)} clean",
        "",
        "| arm | containment |",
        "| --- | --- |",
        f"| `frozen` | {c.get('frozen', float('nan')):.3f} |",
        f"| `mem_on` | {c.get('mem_on', float('nan')):.3f} |",
        f"| `mem_off` | {c.get('mem_off', float('nan')):.3f} |",
        "",
        f"- verbatim containment under `mem_on`: **{c.get('mem_on_full_match', float('nan')):.3f}**",
        f"- routing accuracy: {data.get('routing_accuracy', float('nan')):.3f}",
        f"- `mem_off` bit-identical to `frozen`: "
        f"{data.get('mem_off_equals_frozen_rate', float('nan')):.3f}",
        f"- negative control: {(data.get('negative_control') or {}).get('frozen', float('nan')):.3f}"
        f" -> {(data.get('negative_control') or {}).get('mem_on', float('nan')):.3f}",
        f"- write: {write.get('count', '?')} items in "
        f"{write.get('total_seconds', 0) / 60:.1f} min, mean CE after write "
        f"{write.get('mean_ce_after', float('nan')):.4f}",
        "",
        "Caveats that must travel with these numbers:",
    ]
    lines += [f"- {cav}" for cav in data.get("caveats", [])]

    holdout = data.get("holdout") or {}
    if holdout:
        hc = holdout.get("containment") or {}
        lines += [
            "",
            "**Held-out replication** (pre-registered rule, docs/06 section 9; "
            f"{holdout.get('n_subset', '?')} *different* questions):",
            "",
            "| arm | containment |",
            "| --- | --- |",
            f"| `frozen` | {hc.get('frozen', float('nan')):.3f} |",
            f"| `mem_on` | {hc.get('mem_on', float('nan')):.3f} |",
            f"| `mem_off` | {hc.get('mem_off', float('nan')):.3f} |",
            "",
            f"- routing accuracy: {holdout.get('routing_accuracy', float('nan')):.3f}",
            f"- `mem_off` bit-identical to `frozen`: "
            f"{holdout.get('mem_off_equals_frozen_rate', float('nan')):.3f}",
            f"- **pre-registered verdict: {replication_verdict(bundle)}**",
        ]

    judge = _judge_block(data)
    if judge:
        arms = judge.get("arms") or {}
        lines += [
            "",
            "**Fair comparison (LLM judge, calibrated against human marks; docs/06"
            " section 13):**",
            "",
            "| arm | judged correct | 95% CI | extra prompt tokens |",
            "| --- | --- | --- | --- |",
        ]
        for arm, label in (("weights", "`weights` (parametric)"),
                           ("context_all", "`context_all` (all 30 pairs)"),
                           ("rag_top3", "`rag_top3`"), ("rag_top1", "`rag_top1`"),
                           ("context_target", "`context_target`"),
                           ("frozen", "`frozen` (floor)")):
            stats = arms.get(arm) or {}
            ci = list(stats.get("ci95") or [float("nan"), float("nan")])
            tokens = "0" if arm == "weights" else _judge_tokens(data, arm)
            lines.append(
                f"| {label} | {stats.get('correct_rate', float('nan')):.3f} | "
                f"[{ci[0]:.2f}, {ci[1]:.2f}] | {tokens} |"
            )
        wins, losses = _judge_paired(data)
        lines.append(f"| — paired vs `weights` | {wins} won / {losses} lost | | |")
        if _scale_judge_arms(data):
            lines += [
                "",
                "**Scale control (1.7B backbone, same protocol):**",
                "",
                "| arm | judged correct | extra prompt tokens |",
                "| --- | --- | --- |",
                f"| `weights` | {_scale_rate(data, 'weights')} | 0 |",
                f"| `context_all` | {_scale_rate(data, 'context_all')} | "
                f"{_scale_tokens(data, 'context_all')} |",
                f"| `context_target` | {_scale_rate(data, 'context_target')} | "
                f"{_scale_tokens(data, 'context_target')} |",
                f"| `frozen` | {_scale_rate(data, 'frozen')} | — |",
                "",
                f"- containment on the weights arm at 1.7B: {_scale_containment(data, 'mem_on')}",
            ]
        lines += ["",
                  "The containment column above is **not** the comparison to quote: it "
                  "rewards reproducing the reference string verbatim, which is what a "
                  "trained write is *for* (docs/06 section 10.2)."]
    lines.append("")
    return "\n".join(lines)


def _t8(bundle) -> str:
    data = _get(bundle, "t8")
    if not data:
        return ("### T8-a -- retention term in the write objective\n\n"
                f"{NOT_AVAILABLE}\n")
    settings = _retention_settings(data)
    arms = ("oracle", "top1", "top2", "all")
    lines = [
        "### T8-a -- B1: a retention term in the write objective",
        "",
        "Each write is also required to leave the *earlier* memories' queries "
        "unchanged under the read condition where interference appears (all written "
        "slots active). `lambda_ret = 0` reproduces T2.",
        "",
        "| `lambda_ret` | " + " | ".join(f"`{arm}`" for arm in arms)
        + " | base KL (on) |",
        "| --- | " + " | ".join("---" for _ in arms) + " | --- |",
    ]
    for setting in settings:
        lines.append(
            f"| {setting} | "
            + " | ".join(_retention_em(data, setting, arm) for arm in arms)
            + f" | {_retention_kl(data, setting)} |"
        )
    helped, hurt, tied = _retention_pair(data)
    p_value = _retention_p(data)
    lines += [
        "",
        f"- paired on `top2` against `lambda_ret = 0`: **{helped} probes helped, "
        f"{hurt} hurt, {tied} tied**"
        + (f" (exact sign test p = {p_value:.3f})" if p_value is not None else ""),
        "- routing is unchanged by construction (keys are captured with memory off); "
        "so a `top2` failure is interference, not retrieval",
    ]
    lines.append("")
    return "\n".join(lines)


_FORMATTERS = {"t1": _t1, "t2": _t2, "t3": _t3, "t4": _t4, "t5": _t5, "t6": _t6,
               "t7": _t7, "t8": _t8}


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

    A row equal to ``[MIDRULE]`` becomes a horizontal rule. That exists because
    hand-writing ``["\\\\midrule", ""]`` into a row does not work: the label goes
    through :func:`_tex` like any other cell and reaches the PDF as the literal text
    ``\\textbackslash{}midrule``. The same trap swallowed ``\\texttt{}``, ``\\emph{}``
    and ``\\quad`` in this table's labels for a whole round, because every check we had
    looked at the numbers and none looked at the labels.
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
        if len(row) == 1 and row[0] == MIDRULE:
            lines.append(r"\midrule")
            continue
        cells = []
        for index, cell in enumerate(row):
            escaped = _tex(cell)
            if index in mono_columns:
                escaped = rf"\texttt{{{escaped}}}"
            cells.append(escaped)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _forced_span(bundle: dict[str, Any], field: str) -> float | None:
    t3 = _get(bundle, "t3") or {}
    return (t3.get("forced_span") or {}).get(field)


def _surprise_median(bundle: dict[str, Any], klass: str) -> float | None:
    """Median pre-write loss for the unknown vs already-known classes.

    The closeness of these two numbers is the surprise criterion's failure: if the
    distributions nearly coincide, no threshold on them can separate knowledge.
    """
    t3 = _get(bundle, "t3") or {}
    scores = ((t3.get("summary") or t3).get("surprise_by_class") or {}).get(klass)
    if not scores:
        return None
    import statistics

    return statistics.median(scores)


def _write_reduction(bundle: dict[str, Any]) -> float | None:
    """How many fewer writes the self-check criterion needs than writing everything.

    Computed rather than written into the prose: the paper used to carry ``72\\%`` by
    hand, which is exactly the kind of number that drifts once a criterion changes.
    """
    t3 = _get(bundle, "t3") or {}
    records = (t3.get("summary") or t3).get("records") or []
    by_policy = {r.get("policy"): r for r in records}
    always, selfcheck = by_policy.get("always"), by_policy.get("selfcheck")
    if not always or not selfcheck:
        return None
    total = always.get("n_writes") or 0
    if not total:
        return None
    return 1 - (selfcheck.get("n_writes") or 0) / total


def _t3_seed_reductions(bundle: dict[str, Any]) -> list[float]:
    """Backbone-drift reduction (selfcheck vs always) for each T3 seed.

    The write counts are deterministic given the criteria, but which facts the model
    already answers --- and therefore how far the backbone moves --- depends on the
    seed, so the paper quotes a range instead of one number.
    """
    t3 = _get(bundle, "t3") or {}
    payloads = [t3] + list(t3.get("extra_seeds") or [])
    reductions = []
    for payload in payloads:
        records = (payload.get("summary") or payload).get("records") or []
        by_policy = {r.get("policy"): r for r in records}
        base, best = by_policy.get("always"), by_policy.get("selfcheck")
        if not base or not best:
            continue
        baseline_kl = base.get("anchor_kl") or 0.0
        if baseline_kl:
            reductions.append(1 - (best.get("anchor_kl") or 0.0) / baseline_kl)
    return reductions


def _t2_scale_em(bundle: dict[str, Any], arm: str) -> str:
    """The composition result on the 1.7B backbone, when the control has been run."""
    t2 = _get(bundle, "t2") or {}
    payload = t2.get("scale_17b") or {}
    return _na(((payload.get("summary") or {}).get("em") or {}).get(arm), ".3f")


def _t2_scale_router(bundle: dict[str, Any]) -> str:
    t2 = _get(bundle, "t2") or {}
    payload = t2.get("scale_17b") or {}
    return _na((payload.get("summary") or {}).get("route_top1_accuracy"), ".3f")


def _retention_small(t8: dict) -> dict:
    """The earlier, smaller retention run, when the bundle carries both.

    The paper reports both because the small one's apparent repair did not
    replicate; hiding the smaller run would make the larger one look like the only
    measurement that ever happened.
    """
    return t8.get("small_run") or {}


def _retention_block(t8: dict) -> dict:
    """The per-weight summary, wherever the experiment put it.

    The experiment nests it under ``summary``; accepting the flat shape too keeps the
    renderer from silently reporting "n/a" for every number if that ever changes
    (which is exactly what happened once already).
    """
    return (t8.get("by_lambda")
            or (t8.get("summary") or {}).get("by_lambda")
            or {})


def _retention_settings(t8: dict) -> list[str]:
    return sorted(_retention_block(t8), key=lambda key: float(key))


def _retention_best(t8: dict, arm: str = "top2") -> str:
    """The retention weight with the best recall on ``arm``.

    Ties break towards the *larger* weight only because the sweep's heaviest setting
    is also its most expensive; picking it is the conservative choice for reporting.
    """
    settings = _retention_settings(t8)
    if not settings:
        return "0.0"

    def score(setting: str) -> tuple[float, float]:
        em = (_retention_block(t8).get(setting) or {}).get("em") or {}
        value = em.get(arm)
        return (value if value is not None else -1.0, float(setting))

    return max(settings, key=score)


def _retention_em(t8: dict, setting: str, arm: str) -> str:
    em = ((_retention_block(t8).get(setting) or {}).get("em")) or {}
    return _na(em.get(arm), ".3f")


def _retention_kl(t8: dict, setting: str) -> str:
    block = _retention_block(t8).get(setting) or {}
    return _na(block.get("base_kl_slots_on"), ".3f")


def _retention_series(t8: dict, setting: str, arm: str) -> list:
    """Per-probe outcomes for one setting, in episode order.

    Episode order matters: the pairing is only meaningful if position i is the same
    probe in both settings, so the sort is explicit rather than trusting file order.
    """
    rows = []
    for episode in t8.get("episodes") or []:
        if f"{float(episode.get('lambda_ret', -1))}" != setting:
            continue
        rows.append((episode.get("episode_id", 0),
                     (episode.get("arms") or {}).get(arm) or []))
    rows.sort(key=lambda pair: pair[0])
    return [value for _, values in rows for value in values]


def _retention_pair(t8: dict, arm: str = "top2") -> tuple[int, int, int]:
    base = _retention_series(t8, "0.0", arm)
    best = _retention_series(t8, _retention_best(t8), arm)
    if not base or len(base) != len(best):
        return (0, 0, 0)
    return paired_counts(base, best)


def _retention_p(t8: dict, arm: str = "top2") -> float | None:
    base = _retention_series(t8, "0.0", arm)
    if not base:
        return None
    helped, hurt, _ = _retention_pair(t8, arm)
    return sign_test_p(helped, hurt)


def _retention_probes(t8: dict, arm: str = "top2") -> int | None:
    base = _retention_series(t8, "0.0", arm)
    return len(base) if base else None


def _router_rates(bundle: dict, scale: str = "") -> dict:
    """Router-key accuracies from the T2-c payload.

    Takes the whole bundle rather than the ``t2.summary`` sub-dict: the experiment
    payloads are merged at the *stage* level, so looking inside ``summary`` silently
    returns nothing (which is exactly how these macros first rendered as n/a).
    """
    t2 = _get(bundle, "t2") or {}
    payload = t2.get("router_keys_17b" if scale else "router_keys") or {}
    return payload.get("rates") or {}


def _router_best(bundle: dict, scale: str = "") -> str:
    rates = _router_rates(bundle, scale)
    if not rates:
        return "n/a"
    return max(rates, key=lambda name: (rates[name].get("top1", -1),
                                        rates[name].get("em", -1)))


def _router_stat(bundle: dict, arm: str, field: str, scale: str = "") -> str:
    rates = _router_rates(bundle, scale)
    return _na((rates.get(arm) or {}).get(field), ".3f")


def _router_pair(bundle: dict, arm: str, field: str, scale: str = "") -> str:
    """Paired outcome for one key against the incumbent, on the same probes."""
    t2 = _get(bundle, "t2") or {}
    payload = t2.get("router_keys_17b" if scale else "router_keys") or {}
    entry = (payload.get("paired_vs_last") or {}).get(arm) or {}
    return _na(entry.get(field), ".4f" if field == "sign_p" else ".0f")


def _training_baseline(t7: dict, arm: str, field: str) -> str:
    """One cell of the T7-g comparison, or n/a while the arm is still running."""
    payload = t7.get("training_baselines") or {}
    row = (payload.get("results") or {}).get(arm) or {}
    value = row.get(field)
    if isinstance(value, bool):
        return str(value)
    return _na(value, ".3f") if field == "containment" else _na(value, ".4f")


def _erasure_field(t7: dict, field: str) -> str | None:
    """One measured field of the T7-g erasure check on the slot arm."""
    payload = t7.get("training_baselines") or {}
    check = ((payload.get("results") or {}).get("slots") or {}).get("erasure_check") or {}
    value = check.get(field)
    return None if value is None else ("yes" if value else "no")


def _t4_seeds(bundle) -> list[list[dict]]:
    """Per-seed T4 record lists: the primary run first, then the extra seeds.

    The eviction assertions are deterministic, but which memories survive under each
    policy is not, so the retention fraction is only honest as a range over seeds.
    """
    t4 = _get(bundle, "t4") or {}
    lists = [t4.get("records") or []]
    for payload in t4.get("extra_seeds") or []:
        lists.append(payload.get("records") or [])
    return [records for records in lists if records]


def _hot_fraction(records: list[dict], policy: str) -> float | None:
    for row in records:
        if row.get("policy") != policy:
            continue
        resident = (row.get("resident") or {}).get("hot")
        hits = (row.get("oracle_hits") or {}).get("hot")
        if resident:
            return hits / resident
    return None


def _hot_range(bundle, policy: str) -> str:
    fractions = [value for records in _t4_seeds(bundle)
                 if (value := _hot_fraction(records, policy)) is not None]
    if not fractions:
        return "n/a"
    if len(fractions) == 1:
        return f"{fractions[0]:.0%}"
    return f"{min(fractions):.0%}--{max(fractions):.0%}"


def _multi_session(t7: dict, field: str, sub: str | None = None) -> str:
    """One field of the multi-session stage, or n/a while it is blocked or unrun."""
    payload = t7.get("multi_session") or {}
    value = payload.get(field)
    if sub is not None and isinstance(value, dict):
        value = value.get(sub)
    return _na(value, ".3f") if isinstance(value, float) else _na(value)


def _t5_seed_arms(bundle) -> list[dict]:
    """The inertia arms for every seed the bundle carries (primary first)."""
    t5 = _get(bundle, "t5") or {}
    arms = [t5.get("arms") or {}]
    for payload in t5.get("extra_seeds") or []:
        arms.append(payload.get("arms") or {})
    return [entry for entry in arms if entry]


def _t5_range(bundle, arm: str, field: str) -> str:
    values = [entry.get(arm, {}).get(field) for entry in _t5_seed_arms(bundle)]
    values = [value for value in values if isinstance(value, (int, float))]
    if not values:
        return "n/a"
    if len(values) == 1 or max(values) - min(values) < 1e-9:
        return f"{values[0]:.3f}".rstrip("0").rstrip(".") or "0"
    return f"{min(values):.3f}--{max(values):.3f}"


def _learned_key(bundle: dict, arm: str, field: str, scale: str = "") -> str:
    """One cell of the trained-key comparison (T2-e), optionally on the larger model."""
    t2 = _get(bundle, "t2") or {}
    payload = t2.get("learned_key_17b" if scale else "learned_key") or {}
    if field == "paired":
        entry = (payload.get("paired_vs_last") or {}).get(arm) or {}
        return _na(entry.get("sign_p"), ".4f")
    return _na((payload.get("rates") or {}).get(arm, {}).get(field), ".3f")


def _lex_composition(bundle: dict, field: str) -> str:
    """The 1.7B composition arms re-run with the model-free key (T2, --key-mode lexical).

    This is the causal version of the claim that the oracle-to-top-1 gap is routing: the
    gap is closed by fixing the router rather than inferred from a correlation.
    """
    t2 = _get(bundle, "t2") or {}
    payload = t2.get("composition_lexical_key") or {}
    summary = payload.get("summary") or {}
    if field == "routing":
        return _na(summary.get("route_top1_accuracy"), ".3f")
    if field == "gap":
        em = summary.get("em") or {}
        if em.get("oracle") is None or em.get("top1") is None:
            return "n/a"
        return f"{em['oracle'] - em['top1']:.3f}"
    return _na((summary.get("em") or {}).get(field), ".3f")


def _merged_calibration(t7: dict) -> dict:
    """Human-vs-judge agreement as merged from the marks and the current payload."""
    return t7.get("calibration_merged") or {}


def _scale_judge_arms(t7: dict) -> dict:
    payload = ((t7.get("scale_17b") or {}).get("judge")) or {}
    return (((payload.get("standards") or {}).get("lenient") or {}).get("arms")) or {}


def _scale_baseline_arms(t7: dict) -> dict:
    return (((t7.get("scale_17b") or {}).get("baselines") or {}).get("arms")) or {}


def _scale_rate(t7: dict, arm: str) -> str:
    return _na((_scale_judge_arms(t7).get(arm) or {}).get("correct_rate"), ".3f")


def _scale_tokens(t7: dict, arm: str) -> str:
    return _na((_scale_baseline_arms(t7).get(arm) or {}).get("mean_prompt_tokens"),
               ".0f")


def _scale_containment(t7: dict, key: str) -> str:
    payload = ((t7.get("scale_17b") or {}).get("weights")) or {}
    return _na((payload.get("containment") or {}).get(key), ".3f")


def _judge_block(t7: dict, standard: str = "lenient") -> dict:
    """The judge's per-arm summary for one grading standard."""
    return (((t7.get("judge") or {}).get("standards") or {}).get(standard) or {})


def _judge_rate(t7: dict, arm: str, standard: str = "lenient") -> str:
    arms = _judge_block(t7, standard).get("arms") or {}
    return _na((arms.get(arm) or {}).get("correct_rate"), ".3f")


def _judge_tokens(t7: dict, arm: str) -> str:
    """Extra prompt tokens an arm pays per question, from the baseline run."""
    arms = ((t7.get("baselines") or {}).get("arms")) or {}
    return _na((arms.get(arm) or {}).get("mean_prompt_tokens"), ".0f")


CONTEXT_ARMS = ("context_all", "context_target", "rag_top1", "rag_top3")


def _judge_paired(t7: dict, standard: str = "lenient") -> tuple[int, int]:
    """(questions the weights arm wins outright, questions it loses outright).

    A win is a question where the weights arm is correct and at least one context
    arm is not; a loss is the mirror image. Computed here so the paper cannot quote
    a paired count that disagrees with the per-question verdicts.
    """
    rows = _judge_block(t7, standard).get("rows") or {}
    wins = losses = 0
    for verdicts in rows.values():
        mine = (verdicts.get("weights") or {}).get("correct")
        others = [(verdicts.get(a) or {}).get("correct") for a in CONTEXT_ARMS
                  if a in verdicts]
        if mine is True and any(v is False for v in others):
            wins += 1
        if mine is False and any(v is True for v in others):
            losses += 1
    return wins, losses


def replication_verdict(bundle: dict[str, Any], tolerance: float = 0.15) -> str:
    """Apply the pre-registered replication rule (docs/06 section 9) mechanically.

    Doing this in code rather than by eye is the point: the criterion was written
    down before the run, so the verdict must not depend on who reads the number.
    """
    t7 = _get(bundle, "t7") or {}
    first = (t7.get("containment") or {}).get("mem_on")
    holdout = ((t7.get("holdout") or {}).get("containment") or {}).get("mem_on")
    if first is None or holdout is None:
        return "n/a"
    return ("within the pre-registered band"
            if abs(holdout - first) <= tolerance
            else "outside the pre-registered band")


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
    t7 = _get(bundle, "t7") or {}
    t8 = _get(bundle, "t8") or {}

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
        # ---- retention across seeds, because the surviving set is seed-dependent ----
        "tSeedsCapacity": _na(len(_t4_seeds(bundle)) or None),
        "tFifoHotRange": _hot_range(bundle, fifo.get("policy") if fifo else "fifo"),
        "tSmartHotRange": _hot_range(bundle, smart.get("policy") if smart else "lru"),
        # ---- inertia across seeds: the refutation is identical, the rates are not ----
        "tSeedsInertia": _na(len(_t5_seed_arms(bundle)) or None),
        "tContextRange": _t5_range(bundle, "context_memory", "inheritance_rate"),
        "tParamRange": _t5_range(bundle, "param_memory", "inheritance_rate"),
        "tWithholdFloorRange": _t5_range(bundle, "no_memory", "withholding_rate"),
        "tWithholdMemoryRange": _t5_range(bundle, "param_memory", "withholding_rate"),
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
        # ---- T7-d: public-benchmark controlled diagnostic ----
        "tLongMemFrozen": _na((t7.get("containment") or {}).get("frozen"), ".3f"),
        "tLongMemOn": _na((t7.get("containment") or {}).get("mem_on"), ".3f"),
        "tLongMemOff": _na((t7.get("containment") or {}).get("mem_off"), ".3f"),
        "tLongMemFull": _na((t7.get("containment") or {}).get("mem_on_full_match"), ".3f"),
        "tLongMemRouting": _na(t7.get("routing_accuracy"), ".3f"),
        "tLongMemSubset": _na(t7.get("n_subset")),
        "tLongMemWriteMin": (
            f"{(t7.get('write') or {}).get('total_seconds', 0) / 60:.1f}"
            if t7.get("write") else "n/a"
        ),
        "tLongMemCe": _na((t7.get("write") or {}).get("mean_ce_after"), ".4f"),
        # ---- T7-d held-out replication (pre-registered rule, docs/06 section 9) ----
        "tHoldoutOn": _na(
            ((t7.get("holdout") or {}).get("containment") or {}).get("mem_on"), ".3f"),
        "tHoldoutOff": _na(
            ((t7.get("holdout") or {}).get("containment") or {}).get("mem_off"), ".3f"),
        "tHoldoutRouting": _na((t7.get("holdout") or {}).get("routing_accuracy"), ".3f"),
        "tHoldoutVerdict": replication_verdict(bundle),
        # ---- T7 judge: the fair comparison (docs/06 section 13) ----
        # These are the numbers the paper's main claim rests on, so the paired win
        # and loss counts are computed from the per-question verdicts rather than
        # being transcribed from a table someone printed once.
        "tJudgeWeights": _judge_rate(t7, "weights"),
        "tJudgeContextAll": _judge_rate(t7, "context_all"),
        "tJudgeContextTarget": _judge_rate(t7, "context_target"),
        "tJudgeRagOne": _judge_rate(t7, "rag_top1"),
        "tJudgeRagThree": _judge_rate(t7, "rag_top3"),
        "tJudgeFrozen": _judge_rate(t7, "frozen"),
        "tJudgeWins": str(_judge_paired(t7)[0]),
        "tJudgeLosses": str(_judge_paired(t7)[1]),
        "tTokWeights": "0",
        "tTokContextAll": _judge_tokens(t7, "context_all"),
        "tTokContextTarget": _judge_tokens(t7, "context_target"),
        "tTokRagOne": _judge_tokens(t7, "rag_top1"),
        # The superseded 32-token run, quoted so the paper can show that an earlier
        # "the arms are indistinguishable" reading was an artefact of our own
        # generation budget rather than a property of the mechanisms.
        "tLongMemOnArchived": _na(
            ((t7.get("archived_gen32") or {}).get("containment") or {}).get("mem_on"),
            ".3f"),
        "tJudgeCalibMarks": _na((t7.get("calibration") or {}).get("agreement")),
        "tJudgeCalibRate": _na((t7.get("calibration") or {}).get("rate"), ".3f"),
        # The authoritative calibration numbers come from the merge tool, which reads
        # the human marks and the *current* judge payload. The older result file only
        # recorded the first pass, where two rows were unusable because the judge's own
        # reply was truncated; after the prompt revision all rows are readable and the
        # agreement is higher. Quoting the stale file understated our own evidence.
        "tJudgeMergedMarks": _na(
            f"{_merged_calibration(t7).get('agreement')}/"
            f"{_merged_calibration(t7).get('n_scored')}"
            if _merged_calibration(t7).get("n_scored") else None),
        "tJudgeMergedRate": _na(_merged_calibration(t7).get("agreement_rate"), ".3f"),
        "tJudgeMergedLow": _na((_merged_calibration(t7).get("wilson_95") or [None])[0],
                               ".3f"),
        "tJudgeMergedHigh": _na((_merged_calibration(t7).get("wilson_95") or [None, None])[1],
                                ".3f"),
        "tJudgeMergedFalsePositives": _na(
            _merged_calibration(t7).get("judge_false_positive")),
        # ---- training baselines on the same memories (T7-g) ----
        "tBaseFull": _training_baseline(t7, "full_ft", "containment"),
        "tBaseFullKl": _training_baseline(t7, "full_ft", "backbone_drift"),
        "tBaseAdapter": _training_baseline(t7, "one_adapter", "containment"),
        "tBaseAdapterKl": _training_baseline(t7, "one_adapter", "backbone_drift"),
        "tBaseSlots": _training_baseline(t7, "slots", "containment"),
        "tBaseSlotsKl": _training_baseline(t7, "slots", "backbone_drift"),
        # The erasure claim: only the slot bank can do this, so it is reported as the
        # measured checks rather than as a boolean the script asserts about itself.
        "tBaseErasedVirgin": _na(_erasure_field(t7, "slot_virgin")),
        "tBaseNeighbourKept": _na(_erasure_field(t7, "neighbour_kept")),
        "tBaseErasedGone": _na(_erasure_field(t7, "erased_gone")),
        # ---- multi-session, memories extracted from the conversation (T7-i) ----
        "tMultiItems": _multi_session(t7, "n_items"),
        "tMultiMemories": _multi_session(t7, "n_memories"),
        "tMultiKey": _multi_session(t7, "containment", "key"),
        "tMultiLexical": _multi_session(t7, "containment", "lexical"),
        "tMultiOracle": _multi_session(t7, "containment", "oracle"),
        "tMultiFrozen": _multi_session(t7, "containment", "frozen"),
        "tMultiRouteKey": _multi_session(t7, "routed_to_own", "key"),
        "tMultiRouteLexical": _multi_session(t7, "routed_to_own", "lexical"),
        # ---- T8-a (B1): the retention term in the write objective ----
        "tRetentionTopTwoBase": _retention_em(t8, "0.0", "top2"),
        "tRetentionTopTwoBest": _retention_em(t8, _retention_best(t8), "top2"),
        "tRetentionOwnBase": _retention_em(t8, "0.0", "oracle"),
        "tRetentionOwnBest": _retention_em(t8, _retention_best(t8), "oracle"),
        "tRetentionKlBase": _retention_kl(t8, "0.0"),
        "tRetentionKlBest": _retention_kl(t8, _retention_best(t8)),
        "tRetentionHelped": _na(_retention_pair(t8)[0]),
        "tRetentionHurt": _na(_retention_pair(t8)[1]),
        "tRetentionP": _na(_retention_p(t8), ".3f"),
        "tRetentionProbes": _na(_retention_probes(t8)),
        "tRetentionItems": _na(
            ((t8.get("summary") or {}).get("config") or {}).get("items")),
        "tRetentionBest": _retention_best(t8),
        "tRetentionAllBest": _retention_em(t8, _retention_best(t8), "all"),
        "tRetentionAllBase": _retention_em(t8, "0.0", "all"),
        "tRetentionRouteTwo": _na(
            (_retention_block(t8).get("0.0") or {}).get("route_top2_accuracy"), ".3f"),
        # The arm the retention term is actually evaluated on improves in a
        # one-directional way; the two-slot arm, where we hoped for a repair, does not.
        "tRetentionAllHelped": _na(_retention_pair(t8, "all")[0]),
        "tRetentionAllHurt": _na(_retention_pair(t8, "all")[1]),
        "tRetentionAllP": _na(_retention_p(t8, "all"), ".3f"),
        # The earlier, smaller run: quoted because its apparent repair did not
        # replicate, and a result that shrinks with n is itself the finding.
        "tRetentionSmallHelped": _na(_retention_pair(_retention_small(t8))[0]),
        "tRetentionSmallHurt": _na(_retention_pair(_retention_small(t8))[1]),
        "tRetentionSmallP": _na(_retention_p(_retention_small(t8)), ".3f"),
        "tRetentionSmallProbes": _na(_retention_probes(_retention_small(t8))),
        "tRetentionSmallBest": _retention_em(_retention_small(t8),
                                            _retention_best(_retention_small(t8)),
                                            "top2"),
        # ---- T3 robustness across seeds ----
        "tSelfSeeds": _na(len(_t3_seed_reductions(bundle)) or None),
        "tSelfKlReductionMin": _na(
            min(_t3_seed_reductions(bundle), default=None), ".0%"),
        "tSelfKlReductionMax": _na(
            max(_t3_seed_reductions(bundle), default=None), ".0%"),
        "tSelfWriteReduction": _na(_write_reduction(bundle), ".0%"),
        # The wording/likelihood gap behind the second negative result. The paper
        # used to carry 9.76 nats by hand from a machine-local probe; the value here
        # is the one an experiment in this repository produces.
        "tForcedParisKL": _na(_forced_span(bundle, "canonical_loss"), ".2f"),
        "tVerboseParisKL": _na(_forced_span(bundle, "verbose_loss"), ".2f"),
        "tSurpriseMedianUnknown": _na(_surprise_median(bundle, "unknown"), ".2f"),
        "tSurpriseMedianKnown": _na(_surprise_median(bundle, "known"), ".2f"),
        # ---- the 1.7B scale control for the real-content diagnostic ----
        "tScaleWeights": _scale_rate(t7, "weights"),
        "tScaleContextAll": _scale_rate(t7, "context_all"),
        "tScaleContextTarget": _scale_rate(t7, "context_target"),
        "tScaleFrozen": _scale_rate(t7, "frozen"),
        "tScaleContainment": _scale_containment(t7, "mem_on"),
        "tScaleTokens": _scale_tokens(t7, "context_all"),
        # ---- the composition result on the larger backbone ----
        "tScaleTwoOracle": _t2_scale_em(bundle, "oracle"),
        "tScaleTwoTopOne": _t2_scale_em(bundle, "top1"),
        "tScaleTwoTopTwo": _t2_scale_em(bundle, "top2"),
        "tScaleTwoSum": _t2_scale_em(bundle, "all"),
        "tScaleTwoRouter": _t2_scale_router(bundle),
        # ---- which key the router should use (T2-c) ----
        # LaTeX control words end at the first non-letter, so these names are letters
        # only; the guard in _headline_macros rejects digits and underscores.
        "tKeyBest": _router_best(bundle),
        "tKeyBestTop": _router_stat(bundle, _router_best(bundle), "top1"),
        "tKeyBestEm": _router_stat(bundle, _router_best(bundle), "em"),
        "tKeyLastTop": _router_stat(bundle, "last", "top1"),
        "tKeyLastEm": _router_stat(bundle, "last", "em"),
        "tKeyBestLarge": _router_best(bundle, "17b"),
        "tKeyBestTopLarge": _router_stat(bundle, _router_best(bundle, "17b"), "top1",
                                         "17b"),
        "tKeyBestEmLarge": _router_stat(bundle, _router_best(bundle, "17b"), "em",
                                        "17b"),
        "tKeyLastTopLarge": _router_stat(bundle, "last", "top1", "17b"),
        "tKeyPairHelpedLarge": _router_pair(bundle, _router_best(bundle, "17b"),
                                            "helped", "17b"),
        "tKeyPairHurtLarge": _router_pair(bundle, _router_best(bundle, "17b"), "hurt",
                                          "17b"),
        "tKeyPairPLarge": _router_pair(bundle, _router_best(bundle, "17b"), "sign_p",
                                       "17b"),
        "tKeyPairHelpedSmall": _router_pair(bundle, _router_best(bundle), "helped"),
        "tKeyPairHurtSmall": _router_pair(bundle, _router_best(bundle), "hurt"),
        "tKeyPairPSmall": _router_pair(bundle, _router_best(bundle), "sign_p"),
        "tKeyEntropyTop": _router_stat(bundle, "entropy", "top1"),
        "tKeyShallowTop": _router_stat(bundle, "shallow", "top1"),
        # ---- the trained key and multi-key voting (T2-e) ----
        "tKeyLearnedTop": _learned_key(bundle, "learned", "top1"),
        "tKeyLearnedEm": _learned_key(bundle, "learned", "em"),
        "tKeyVoteTop": _learned_key(bundle, "voting", "top1"),
        "tKeyVoteEm": _learned_key(bundle, "voting", "em"),
        "tKeyPairLearnedP": _learned_key(bundle, "learned", "paired"),
        "tKeyPairVoteP": _learned_key(bundle, "voting", "paired"),
        # the same two candidates on the larger backbone
        "tKeyLearnedTopLarge": _learned_key(bundle, "learned", "top1", "17b"),
        "tKeyLearnedEmLarge": _learned_key(bundle, "learned", "em", "17b"),
        "tKeyVoteTopLarge": _learned_key(bundle, "voting", "top1", "17b"),
        "tKeyVoteEmLarge": _learned_key(bundle, "voting", "em", "17b"),
        "tKeyBestEmLarge": _learned_key(bundle, "hybrid", "em", "17b"),
        # ---- the same composition arms with the good key (causal check) ----
        "tLexOracle": _lex_composition(bundle, "oracle"),
        "tLexTopOne": _lex_composition(bundle, "top1"),
        "tLexTopTwo": _lex_composition(bundle, "top2"),
        "tLexSum": _lex_composition(bundle, "all"),
        "tLexRouting": _lex_composition(bundle, "routing"),
        "tLexGap": _lex_composition(bundle, "gap"),
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
        _latex_t7(bundle),
        _latex_t8(bundle),
    ]
    return "\n".join(parts)


def _latex_t8(bundle) -> str:
    data = _get(bundle, "t8")
    if not data:
        return f"% T8-a {NOT_AVAILABLE}"
    settings = _retention_settings(data)
    # Plain text only: everything here goes through _tex, so a LaTeX command in a
    # label reaches the PDF as its own escaped source. Monospace is requested through
    # mono_columns instead of by writing \texttt{} by hand.
    rows = [[f"{s}", _retention_em(data, s, "top2"), _retention_em(data, s, "oracle"),
             _retention_kl(data, s)] for s in settings]
    return _table(
        "T8-a: the write objective also protects earlier memories (B1). "
        "$\\lambda_{\\mathrm{ret}} = 0$ reproduces the T2 composition result. "
        "\\texttt{top2} is recall with the two best-routed slots active; base KL is the "
        "damage the bank does to generic prompts while in use.",
        "tab:t8", ["lambda_ret", "top2", "oracle", "base KL"],
        rows, "lccc", mono_columns=frozenset({1, 2}))


def _latex_t7(bundle) -> str:
    data = _get(bundle, "t7")
    if not data:
        return f"% T7-d {NOT_AVAILABLE}"
    c = data.get("containment") or {}
    write = data.get("write") or {}
    neg = data.get("negative_control") or {}
    rows = [
        ["questions (single-session subset)", str(data.get("n_subset", "n/a"))],
        ["frozen containment", f"{c.get('frozen', float('nan')):.3f}"],
        ["mem_on containment", f"{c.get('mem_on', float('nan')):.3f}"],
        ["mem_off containment", f"{c.get('mem_off', float('nan')):.3f}"],
        ["verbatim containment (mem_on)", f"{c.get('mem_on_full_match', float('nan')):.3f}"],
        ["routing accuracy", f"{data.get('routing_accuracy', float('nan')):.3f}"],
        ["mem_off bit-identical to frozen",
         f"{data.get('mem_off_equals_frozen_rate', float('nan')):.3f}"],
        ["negative control: mem_on after writing to frozen",
         f"{neg.get('frozen', float('nan')):.3f} to {neg.get('mem_on', float('nan')):.3f}"],
        ["write cost", f"{write.get('count', 'n/a')} items / "
                       f"{write.get('total_seconds', 0) / 60:.1f} min"],
    ]
    holdout = data.get("holdout") or {}
    if holdout:
        hc = holdout.get("containment") or {}
        rows += [
            [MIDRULE],
            ["held-out replication, mem_on", f"{hc.get('mem_on', float('nan')):.3f}"],
            ["held-out replication, mem_off", f"{hc.get('mem_off', float('nan')):.3f}"],
            ["pre-registered verdict (docs/06 section 9)", replication_verdict(bundle)],
        ]
    judge = _judge_block(data)
    if judge:
        arms = judge.get("arms") or {}
        wins, losses = _judge_paired(data)
        rows += [
            [MIDRULE],
            ["LLM judge, weights",
             f"{arms.get('weights', {}).get('correct_rate', float('nan')):.3f}"],
            ["LLM judge, context_all",
             f"{arms.get('context_all', {}).get('correct_rate', float('nan')):.3f}"],
            ["LLM judge, rag_top1",
             f"{arms.get('rag_top1', {}).get('correct_rate', float('nan')):.3f}"],
            ["LLM judge, context_target",
             f"{arms.get('context_target', {}).get('correct_rate', float('nan')):.3f}"],
            ["LLM judge, paired wins / losses", f"{wins} / {losses}"],
            # The weights arm reads the memory through the forward pass, so it pays
            # no prompt tokens by construction; asking the baseline report for it
            # returns n/a and understates the result we are reporting.
            ["extra prompt tokens, weights", "0"],
            ["extra prompt tokens, context_all", _judge_tokens(data, "context_all")],
        ]
    if _scale_judge_arms(data):
        rows += [
            [MIDRULE],
            ["scale control 1.7B, weights", _scale_rate(data, "weights")],
            ["scale control 1.7B, context_all", _scale_rate(data, "context_all")],
            ["scale control 1.7B, context_target", _scale_rate(data, "context_target")],
            ["scale control 1.7B, frozen", _scale_rate(data, "frozen")],
            ["scale control 1.7B, containment (mem_on)",
             _scale_containment(data, "mem_on")],
        ]
    return _table(
        "T7-d: a controlled diagnostic on real benchmark content. The history is "
        "absent throughout; the answer is written into a slot and read back, so this "
        "measures whether the parameters can hold real content, \\emph{not} whether "
        "retrieval generalises. See the caveats in the text.",
        "tab:t7", ["measurement", "value"], rows, "ll")
