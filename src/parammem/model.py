"""The frozen backbone plus its rank-blocked memory side path.

Only three things live here beyond model loading:

* **write** -- teach one ``query -> value`` pair with cross-entropy on the answer
  span only, through the side path;
* **read** -- answer a query with a prompt identical in shape to the write prompt
  (that identity is what makes the association retrievable at all);
* **arms** -- set which slots are active, which is how protocol P4's
  MEM_ON / MEM_OFF / MEM_SHUFFLE are produced on a byte-identical model.

Qwen3 is a hybrid reasoning model, so the chat template is applied with
``enable_thinking=False``: otherwise the answers would be wrapped in reasoning
blocks and exact-match scoring would measure the wrapper, not the memory.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .memory.slots import RankBlockedLoRA, attach_slot_lora, count_memory_parameters

__all__ = ["BackboneConfig", "Backbone", "PROMPT_TEMPLATE", "resolve_model_path"]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question as briefly as possible, "
    "with just the value asked for."
)
PROMPT_TEMPLATE = "{system}\nQuestion: {query}\nAnswer:"

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

# Prompts used as the "do not damage the base model" anchor during a write.
# They are deliberately generic and answerable by the frozen model: the side path
# must not move the distribution on them. Without such an anchor the write term
# alone collapses the model into a constant continuation (observed 2026-09-29:
# every query returned fragments of the most recently written value, see
# DEVELOPMENT.md T1-a).
GENERIC_ANCHORS = (
    "What is the capital of France?",
    "Name a primary colour.",
    "What is 2 + 2?",
    "What is the opposite of hot?",
    "Name a day of the week.",
    "Say hello.",
)


def _normalized_name(text: str) -> str:
    """Compare cache directory names loosely: ModelScope rewrites '.' to '___'
    (``Qwen3-1.7B`` is stored as ``Qwen3-1___7B``), so exact string matching
    silently misses a perfectly good local copy."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _has_config(path: Path) -> bool:
    return (path / "config.json").is_file()


def _hf_snapshot(path: Path) -> str:
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        for snap in sorted(snapshots.iterdir()):
            if snap.is_dir() and _has_config(snap):
                return str(snap)
    return str(path)


def resolve_model_path(model_id: str) -> str:
    """Resolve a model id to a local directory when a usable cached copy exists.

    HuggingFace downloads are unreliable on the development machine (TLS
    interception, see ERROR.md E2/E3), so the backbone is fetched from ModelScope
    instead (E5). Rather than hardcoding a machine path we probe, in order:

    1. an existing local directory (a path was passed directly);
    2. ``$PARAMMEM_MODEL_DIR`` (explicit override);
    3. the ModelScope cache, ``~/.cache/modelscope/models/<org>/<name>``
       (directory names are compared *normalised*, see :func:`_normalized_name`);
    4. the HuggingFace cache, but **only if it is complete**.

    A half-finished HuggingFace cache is refused with a warning rather than used:
    on this machine one exists from a stalled download, and silently loading it
    is exactly the kind of false result this project must not produce.
    """
    candidate = Path(model_id).expanduser()
    if candidate.is_dir():
        if not _has_config(candidate):
            warnings.warn(f"{candidate} has no config.json; using it anyway")
        return str(candidate)

    override = os.environ.get("PARAMMEM_MODEL_DIR")
    if override:
        path = Path(override).expanduser()
        if path.is_dir():
            return str(path)

    org, _, name = model_id.partition("/")
    if name:
        ms_base = Path.home() / ".cache" / "modelscope" / "models" / org
        if ms_base.is_dir():
            want = _normalized_name(name)
            for entry in sorted(ms_base.iterdir()):
                if entry.is_dir() and _normalized_name(entry.name) == want and _has_config(entry):
                    return str(entry)

    hf = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + model_id.replace("/", "--")
    )
    if hf.is_dir():
        blobs = hf / "blobs"
        incomplete = list(blobs.glob("*.incomplete")) if blobs.is_dir() else []
        resolved_hf = _hf_snapshot(hf)
        if not incomplete and _has_config(Path(resolved_hf)):
            return resolved_hf
        warnings.warn(
            f"ignoring incomplete HuggingFace cache at {hf} "
            f"({len(incomplete)} leftover *.incomplete blob(s), snapshot usable: "
            f"{_has_config(Path(resolved_hf))}); fetch the model from ModelScope "
            "instead - see ERROR.md E3/E5"
        )
    return model_id


@dataclass
class BackboneConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    device: str = "cuda"
    dtype: str = "bfloat16"
    target_suffixes: Sequence[str] = DEFAULT_TARGETS
    n_slots: int = 32
    rank: int = 4
    alpha: float = 16.0
    seed: int = 0
    max_new_tokens: int = 16
    gradient_checkpointing: bool = False
    attn_implementation: str | None = None

    def torch_dtype(self):
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[self.dtype]


@dataclass
class Backbone:
    """Frozen weights + a memory side path. All memory state is in ``wrappers``."""

    cfg: BackboneConfig
    tokenizer: object
    model: object
    wrappers: dict[str, RankBlockedLoRA] = field(default_factory=dict)
    resolved_path: str = ""

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, cfg: BackboneConfig | None = None) -> "Backbone":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = cfg or BackboneConfig()
        resolved = resolve_model_path(cfg.model_id)
        tokenizer = AutoTokenizer.from_pretrained(resolved)
        kwargs = dict(torch_dtype=cfg.torch_dtype())
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(resolved, **kwargs)
        model.to(cfg.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        wrappers = attach_slot_lora(
            model,
            cfg.target_suffixes,
            n_slots=cfg.n_slots,
            rank=cfg.rank,
            alpha=cfg.alpha,
            seed=cfg.seed,
        )
        if not wrappers:
            raise RuntimeError(
                f"no linear layer matched {list(cfg.target_suffixes)}; check the "
                "model's module names before trusting any later number"
            )
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return cls(
            cfg=cfg, tokenizer=tokenizer, model=model,
            wrappers=wrappers, resolved_path=resolved,
        )

    # -------------------------------------------------------------- helpers
    @property
    def device(self) -> torch.device:
        return torch.device(self.cfg.device)

    @property
    def n_memory_parameters(self) -> int:
        return count_memory_parameters(self.wrappers)

    def _chat(self, query: str, *, answer: str | None = None) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}"},
        ]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            # Older/newer templates may not accept the flag; degrade loudly.
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        if answer is not None:
            text = text + " " + answer
        return text

    def _ids(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    def _answer_ids(self, answer: str) -> torch.Tensor:
        eos = self.tokenizer.eos_token or ""
        return self.tokenizer(
            " " + answer + eos, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.device)

    # ----------------------------------------------------------------- write
    def write_loss(self, query: str, value: str) -> torch.Tensor:
        """Cross-entropy on the answer span only, at the current slot state."""
        prompt = self._chat(query)
        prompt_ids = self._ids(prompt)
        answer_ids = self._answer_ids(value)
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        labels = torch.cat(
            [torch.full_like(prompt_ids, -100), answer_ids], dim=1
        )
        out = self.model(input_ids=input_ids, labels=labels)
        return out.loss

    # ------------------------------------------------------------------ read
    @torch.no_grad()
    def answer(self, query: str, *, max_new_tokens: int | None = None) -> str:
        prompt_ids = self._ids(self._chat(query))
        generated = self.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        new_tokens = generated[0][prompt_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def answer_many(self, queries: Iterable[str], **kwargs) -> list[str]:
        return [self.answer(q, **kwargs) for q in queries]

    # ------------------------------------------------------- write anchoring
    def next_token_logits(self, query: str) -> torch.Tensor:
        """Final-position logits for ``query`` at the current slot state (float32)."""
        prompt_ids = self._ids(self._chat(query))
        logits = self.model(input_ids=prompt_ids).logits[0, -1]
        return logits.float()

    def all_position_logits(self, query: str) -> torch.Tensor:
        """Logits at **every** position of ``query`` (float32, shape (seq, vocab))."""
        prompt_ids = self._ids(self._chat(query))
        return self.model(input_ids=prompt_ids).logits[0].float()

    @torch.no_grad()
    def anchor_logits(self, queries: Iterable[str]) -> dict[str, torch.Tensor]:
        """Reference distributions for the anchor prompts, at **every position**.

        Call with **all slots switched off** so the reference is the frozen model;
        that is what "do not damage the base model" is measured against.

        Every position, not just the last one: a final-position-only guard was
        measured to hold single-step KL at ~0.08 while generation still collapsed
        into repetition ("2222222222222222", "Monday, Monday, ..."). Generation is
        autoregressive, so a tiny per-step shift compounds; a one-position guard
        cannot protect a 16-step trajectory.
        """
        return {q: self.all_position_logits(q).clone() for q in queries}

    def kl_to_anchors(self, anchors: dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean KL(current || reference) over anchor prompts **and positions**."""
        if not anchors:
            return torch.zeros((), device=self.device)
        total = None
        for query, ref in anchors.items():
            cur = self.all_position_logits(query)
            if cur.shape != ref.shape:
                raise ValueError(
                    f"anchor shape drift for {query!r}: {tuple(cur.shape)} vs "
                    f"{tuple(ref.shape)}; the reference must be captured with the "
                    "same prompt template"
                )
            log_cur = F.log_softmax(cur, dim=-1)
            log_ref = F.log_softmax(ref, dim=-1)
            term = (log_ref.exp() * (log_ref - log_cur)).sum(dim=-1).mean()
            total = term if total is None else total + term
        return total / len(anchors)

    # ------------------------------------------------------------------ arms
    def set_read_slots(self, slots: Iterable[int] | None) -> None:
        for wrapper in self.wrappers.values():
            wrapper.set_read_slots(slots)

    def erase_slots(self, slots: Iterable[int]) -> None:
        for slot in slots:
            for wrapper in self.wrappers.values():
                wrapper.erase(slot)

    def erase_all(self) -> None:
        for wrapper in self.wrappers.values():
            wrapper.erase_all()

    def slot_norm(self, slot: int) -> float:
        return sum(w.slot_norm(slot) for w in self.wrappers.values())

    # -------------------------------------------------------------- reporting
    def memory_footprint(self) -> dict[str, float]:
        return {
            "memory_params": float(self.n_memory_parameters),
            "peak_vram_mb": (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0.0
            ),
        }

    def timed(self, fn, *args, **kwargs):
        """Run ``fn`` and return ``(result, seconds)`` (cost reporting, RQ4)."""
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        return result, time.perf_counter() - t0
