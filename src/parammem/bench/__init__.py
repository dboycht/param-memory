"""Controlled synthetic benchmark: fictional worlds, episodes, probe contexts.

Why fictional entities: if a fact is about a real person/place the model may
answer correctly from pre-training, and we could not attribute the answer to the
memory we wrote (see ``docs/00-problem-statement.md`` section 6.1). Every entity
and value here is generated, so a correct answer can only come from the write.
"""

from .entities import FictionalWorld
from .synthetic import Episode, InertiaBlock, MemoryItem, Probe, make_episode

__all__ = [
    "FictionalWorld",
    "Episode",
    "InertiaBlock",
    "MemoryItem",
    "Probe",
    "make_episode",
]
