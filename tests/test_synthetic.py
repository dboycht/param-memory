"""Tests for the episode generator.

The structural half of protocol P1 lives here: an episode's read-time context
must not contain the answer to any probe *by construction*, so that a correct
answer can only come from the memory under test.
"""

from __future__ import annotations

from parammem.bench.protocol import assert_evicted
from parammem.bench.synthetic import make_episode, make_episodes


def test_episode_is_reproducible():
    a, b = make_episode(7), make_episode(7)
    assert [w.statement for w in a.writes] == [w.statement for w in b.writes]
    assert a.probe_context() == b.probe_context()
    assert [p.query for p in a.probes] == [p.query for p in b.probes]


def test_probe_context_contains_no_answer_for_any_seed():
    for seed in range(60):
        episode = make_episode(seed)
        context = episode.probe_context()
        for probe in list(episode.probes) + list(episode.negatives):
            report = assert_evicted(context, probe.value, probe.aliases)
            assert report.evicted, (seed, probe.query, report.reason())


def test_write_context_does_contain_the_answers():
    """Sanity check: the P1 test above is only meaningful if the write section
    really does carry the values."""
    episode = make_episode(3)
    context = episode.write_context()
    hits = [
        p for p in episode.probes
        if assert_evicted(context, p.value, p.aliases).checks["exact"]
    ]
    assert hits, "no probe value found in its own write section"


def test_structure_counts():
    episode = make_episode(11, n_facts=4, n_prefs=2, n_lessons=1, n_noise=3, n_negatives=2)
    assert len(episode.writes) == 10
    assert len(episode.probes) == 7  # 4 facts + 2 prefs + 1 lesson
    assert len(episode.negatives) == 2
    assert all(p.is_negative for p in episode.negatives)
    assert episode.inertia is not None


def test_negatives_are_about_entities_never_written():
    episode = make_episode(5)
    written = " ".join(w.statement for w in episode.writes)
    for negative in episode.negatives:
        assert negative.value not in written
        assert negative.item_id is None


def test_probes_point_at_written_items():
    episode = make_episode(17)
    by_id = {w.item_id: w for w in episode.writes}
    for probe in episode.probes:
        assert probe.item_id in by_id
        assert by_id[probe.item_id].value_text == probe.value


def test_importance_marking_matches_fraction():
    episode = make_episode(13, n_facts=4, n_prefs=2, n_lessons=2, important_fraction=0.5)
    probeable = episode.probeable_items
    expected = max(1, round(len(probeable) * 0.5))
    assert sum(i.is_important for i in probeable) == expected
    assert not any(i.is_important for i in episode.writes if i.is_noise)


def test_noise_is_not_probeable():
    episode = make_episode(21, n_noise=4)
    noise = [w for w in episode.writes if w.is_noise]
    assert len(noise) == 4
    assert all(not w.probeable and w.value is None for w in noise)


def test_episodes_from_make_episodes_have_disjoint_worlds():
    episodes = make_episodes(3)
    vocab = [
        {w.value_text for w in e.writes if w.value_text} for e in episodes
    ]
    assert not (vocab[0] & vocab[1])
    assert not (vocab[1] & vocab[2])
    assert not (vocab[0] & vocab[2])


def test_inertia_block_inherited_keywords_appear_in_the_premise():
    for seed in range(30):
        inertia = make_episode(seed).inertia
        assert inertia is not None
        low = inertia.premise.lower()
        assert any(k.lower() in low for k in inertia.inherited_keywords), (seed, inertia)


def test_inertia_block_can_be_disabled():
    assert make_episode(2, with_inertia=False).inertia is None


def test_inertia_premise_is_writable():
    """The premise must exist as a query->answer pair so the three-arm comparison
    (absent / in context / in the weights) can actually write it."""
    for seed in range(30):
        inertia = make_episode(seed).inertia
        assert inertia is not None
        assert inertia.premise_query, seed
        assert inertia.premise_answer, seed
        assert inertia.old_topic in inertia.premise_query, seed
        assert inertia.premise_query not in inertia.new_task, seed
        # The written answer must actually be part of the premise, otherwise the
        # parametric arm would be storing something the context arm never saw.
        assert inertia.premise_answer.lower() in inertia.premise.lower(), seed


def test_capacity_k_is_recorded():
    episode = make_episode(4, capacity_k=8, n_facts=3)
    assert episode.capacity_k == 8


# --------------------------------------------------------------- paraphrases

def test_default_probes_use_the_write_query_verbatim():
    episode = make_episode(3)
    by_id = {w.item_id: w for w in episode.writes}
    for probe in episode.probes:
        assert probe.query == by_id[probe.item_id].query
        assert by_id[probe.item_id].probe_query == ""


def test_paraphrase_probes_differ_from_the_write_query():
    """The router only has a real job when reading does not reuse the write string."""
    for seed in range(40):
        episode = make_episode(seed, n_facts=3, paraphrase_probes=True)
        by_id = {w.item_id: w for w in episode.writes}
        for probe in episode.probes:
            item = by_id[probe.item_id]
            assert item.probe_query, (seed, probe.item_id)
            assert probe.query == item.probe_query
            assert probe.query != item.query, (seed, probe.query)


def test_paraphrased_probes_still_pass_the_eviction_check():
    for seed in range(30):
        episode = make_episode(seed, paraphrase_probes=True)
        context = episode.probe_context()
        for probe in list(episode.probes) + list(episode.negatives):
            report = assert_evicted(context, probe.value, probe.aliases)
            assert report.evicted, (seed, probe.query, report.reason())


def test_paraphrases_are_reproducible():
    a = make_episode(11, paraphrase_probes=True)
    b = make_episode(11, paraphrase_probes=True)
    assert [p.query for p in a.probes] == [p.query for p in b.probes]
    assert [w.probe_query for w in a.writes] == [w.probe_query for w in b.writes]


def test_negative_controls_are_paraphrased_too():
    episode = make_episode(7, n_negatives=3, paraphrase_probes=True)
    assert episode.negatives
    for negative in episode.negatives:
        assert "Remind me" in negative.query or "again" in negative.query
