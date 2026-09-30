"""Guards for the model-free lexical router.

T2-c found the lexical scorer beats the model's hidden-state key on the synthetic
benchmark, where the queries are template-generated and their content words are
distinctive. That is exactly why this scorer stays opt-in and why its limits are
tested: the same scorer must behave sensibly (deterministically, and without pretending
to a signal it does not have) when the items share their surface form.
"""

from __future__ import annotations

from parammem.memory.router import lexical_scores, route_lexical


def test_identical_text_selects_its_own_slot():
    slots = {0: "the capital of France is Paris", 1: "the largest planet is Jupiter"}
    assert route_lexical("the capital of France is Paris", slots).top == 0
    assert route_lexical("the largest planet is Jupiter", slots).top == 1


def test_paraphrase_with_shared_content_words_still_routes():
    slots = {0: "How long is my daily commute to work?",
             1: "How long have I been collecting vintage cameras?"}
    # different function words, same content word
    assert route_lexical("my commute to work, how long is it?", slots).top == 0
    assert route_lexical("how long have I collected vintage cameras?", slots).top == 1


def test_items_sharing_surface_form_are_not_separated():
    """The honest limit: when the items differ only by a number, lexical scoring has
    nothing to go on, and the result falls back to slot order rather than inventing a
    preference."""
    slots = {0: "what was the count in the Lost Temple of the Djinn?",
             1: "what was the count in the Lost Temple of the Djinn?"}
    scores = lexical_scores("what was the count in the Lost Temple of the Djinn?", slots)
    assert scores[0][1] == scores[1][1]
    assert route_lexical("what was the count in the Lost Temple of the Djinn?", slots).top == 0


def test_scoring_is_deterministic_and_ordered():
    """Ties are broken by slot index, and a slot sharing no term scores zero.

    Note the two slots that both contain "beta" do *not* tie: each document vector is
    normalised by its own length, so the slot with fewer competing terms gives its
    shared term more relative weight.
    """
    slots = {3: "alpha beta", 1: "beta gamma", 2: "gamma delta"}
    first = lexical_scores("beta", slots)
    assert lexical_scores("beta", slots) == first
    scores = dict(first)
    assert scores[2] == 0.0
    assert scores[1] > scores[3] > scores[2]
    assert first[0][0] == 1


def test_empty_bank_selects_nothing():
    assert route_lexical("anything", {}).slots == []
    assert route_lexical("anything", {0: "a"}, k=0).slots == []
