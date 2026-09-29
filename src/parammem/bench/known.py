"""Facts the frozen model probably already knows, used as the "do not waste a slot
on this" class in the T3 write-criterion experiment.

Two rules for this list:

1. **Nothing here may be assumed known.** The experiment verifies every pair by
   asking the frozen model and keeps only the ones it actually answers correctly;
   pairs a 0.6B model gets wrong are dropped, not excused.
2. The ``statement`` wording deliberately contains **no** explicit-request marker
   (see ``criteria.EXPLICIT_MARKERS``), so the ``explicit`` criterion should skip
   these items while the ``surprise`` criterion has to detect them on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["KnownPair", "KNOWN_PAIRS"]


@dataclass(frozen=True)
class KnownPair:
    query: str
    answer: str
    statement: str

    @property
    def expected_fragment(self) -> str:
        """What the frozen answer must contain for the pair to count as known."""
        return self.answer.lower()


def _pair(query: str, answer: str, statement: str) -> KnownPair:
    return KnownPair(query=query, answer=answer, statement=statement)


KNOWN_PAIRS: tuple[KnownPair, ...] = (
    _pair("What is the capital of France?", "Paris",
          "The capital of France is Paris, as everyone knows."),
    _pair("What is 2 + 2?", "4",
          "Two plus two equals four; that is basic arithmetic."),
    _pair("What colour is the sky on a clear day?", "blue",
          "On a clear day the sky looks blue."),
    _pair("How many days are in a week?", "7",
          "A week has seven days."),
    _pair("What is the largest planet in the solar system?", "Jupiter",
          "Jupiter is the largest planet in the solar system."),
    _pair("What language is spoken in Brazil?", "Portuguese",
          "People in Brazil speak Portuguese."),
    _pair("What is the chemical symbol for water?", "H2O",
          "Water's chemical symbol is H2O."),
    _pair("How many continents are there on Earth?", "7",
          "There are seven continents on Earth."),
    _pair("What is the capital of Japan?", "Tokyo",
          "Tokyo is the capital of Japan."),
    _pair("What is the capital of Italy?", "Rome",
          "Rome is the capital of Italy."),
    _pair("What is the capital of the United Kingdom?", "London",
          "London is the capital of the United Kingdom."),
    _pair("What is the largest ocean on Earth?", "Pacific",
          "The Pacific is the largest ocean on Earth."),
    _pair("What do bees make?", "honey",
          "Bees make honey."),
    _pair("What is frozen water called?", "ice",
          "Frozen water is called ice."),
    _pair("What do you call a baby dog?", "puppy",
          "A baby dog is called a puppy."),
    _pair("Which planet do humans live on?", "Earth",
          "Humans live on Earth."),
    _pair("What is the opposite of hot?", "cold",
          "The opposite of hot is cold."),
    _pair("What is the first month of the year?", "January",
          "January is the first month of the year."),
    _pair("What is the largest mammal on Earth?", "whale",
          "The blue whale is the largest mammal on Earth."),
)
