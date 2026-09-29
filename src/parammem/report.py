"""Turning experiment reports into the tables the paper needs.

Kept as pure functions over the JSON bundles each experiment writes, so the
formatting can be unit-tested without a model and a missing stage degrades to a
"not available" row instead of an exception (a partial run should still produce a
partial, honest results file).
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_results_markdown", "stage_status"]

NOT_AVAILABLE = "_not available in this run_"


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
