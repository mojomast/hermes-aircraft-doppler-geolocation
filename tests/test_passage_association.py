"""Hermetic regressions for distinct encounters, joint assignment and CPA bias."""

import itertools

import numpy as np
import pytest

from dopplergeo import score
from dopplergeo.detect import Event

C = 343.0


def repeated_track():
    t = np.arange(-100.0, 801.0)
    pos = np.c_[
        3000 * np.sin(2 * np.pi * t / 300), np.full_like(t, 600), np.full_like(t, 800)
    ]
    return score.Plane("same-flight", "piston", t, pos, "repeated flight")


def pass_near(pl, when):
    passes = score.find_passes([pl], np.zeros((1, 3)), score.Options(), C)[0]
    return min(passes, key=lambda p: abs(pl.tau[p[1]] - when))


def observation(pl, passage, start, bias=0):
    _, k, distance, received = passage
    times = np.arange(received - start - 30, received - start + 30.01, 0.5)
    ratio = score.predicted_ratio(pl, np.zeros(3), start + times, C, k, 120)
    return Event(
        times[0],
        times[-1],
        received - start + bias,
        80,
        60,
        distance,
        0.1,
        15,
        [1, 2, 3],
        1000,
        1e-12,
        True,
        times=times.tolist(),
        ratio=ratio.tolist(),
        sigma_ratio=0.003,
    )


def score_at_clock(events, planes, start, **options):
    return score.score_points(
        events,
        planes,
        np.zeros((1, 3)),
        C,
        score.Options(**options),
        (start - 0.1, start + 0.1),
        500,
    )


def test_two_distinct_passes_of_one_continuous_flight_can_both_match():
    pl = repeated_track()
    a, b = pass_near(pl, 150), pass_near(pl, 450)
    start = a[3] - 100
    events = [observation(pl, a, start), observation(pl, b, start)]
    result = score_at_clock(events, [pl], start)
    assert score.n_consistent_matches(result.matches[0]) == 2
    assert len({m["cpa_index"] for m in result.matches[0]}) == 2
    assert {m["hex"] for m in result.matches[0]} == {pl.hex}


def test_joint_assignment_is_event_order_invariant_and_does_not_reuse_a_pass():
    a = repeated_track()
    passage = pass_near(a, 150)
    start = passage[3] - 100
    event = observation(a, passage, start)
    # Both aircraft are plausible. Greedy conditioning of the marginalized
    # likelihood must not depend on which observed family is visited first.
    b = score.Plane("alternative", "piston", a.tau, a.pos + [0, 0, 10], "alternative")
    second = observation(a, passage, start)
    # One observed curve is exact on B, the other exact on A.
    first_ratio = score.predicted_ratio(
        b, np.zeros(3), start + np.asarray(event.times), C, passage[1], 120
    )
    event.ratio = first_ratio.tolist()
    first = score_at_clock([event, second], [a, b], start)
    reverse = score_at_clock([second, event], [a, b], start)
    assert score.n_consistent_matches(first.matches[0]) == 2
    np.testing.assert_allclose(first.logl, reverse.logl, rtol=0, atol=1e-12)
    np.testing.assert_allclose(first.t0, reverse.t0, rtol=0, atol=1e-9)
    keys = [(m["hex"], m["cpa_index"]) for m in first.matches[0]]
    assert len(set(keys)) == 2
    one = score_at_clock([event, second], [a], start)
    assert len(one.matches[0]) <= 1


def test_biased_fitted_cpa_does_not_discard_the_true_known_clock():
    pl = repeated_track()
    passage = pass_near(pl, 150)
    start = passage[3] - 100
    event = observation(pl, passage, start, bias=24)
    result = score_at_clock([event], [pl], start)
    assert score.n_consistent_matches(result.matches[0]) == 1
    assert abs(result.t0[0] - start) <= 0.3
    # The broad phase-search budget is explicit and does not relax agreement
    # between events on their recording start time.
    narrow = score_at_clock([event], [pl], start, cpa_search_s=6)
    assert score.n_consistent_matches(narrow.matches[0] or []) == 0


def test_inbound_track_endpoint_is_not_an_observed_cpa():
    t = np.arange(50.0)
    pos = np.c_[-6000 + 60 * t, np.zeros_like(t), np.full_like(t, 1000)]
    pl = score.Plane("inbound", "piston", t, pos)
    assert score.find_passes([pl], np.zeros((1, 3)), score.Options(), C) == [[]]


def test_joint_map_and_marginal_match_exhaustive_assignments():
    def candidate(value, aircraft):
        return score._Candidate(value, 0, aircraft, 10, 1000, 1, 0)

    events = [[candidate(10, 0), candidate(9, 1)], [candidate(9, 0), candidate(0, 1)]]
    miss = [5, 5]
    # The unexplained second occurrence of aircraft0 must retain its penalty.
    quiet = {(0, 10): 0.2, (1, 10): 0.4, (0, 20): 0.3}
    values = []
    for assignment in itertools.product(*[[None, *cs] for cs in events]):
        used = [c.occurrence for c in assignment if c is not None]
        if len(used) != len(set(used)):
            continue
        value = sum(-miss[i] if c is None else c.logl for i, c in enumerate(assignment))
        values.append(
            value - sum(penalty for key, penalty in quiet.items() if key not in used)
        )
    for marginalize in (False, True):
        actual, matches = score._joint_association(events, miss, quiet, marginalize)
        reverse, _ = score._joint_association(
            list(reversed(events)), miss, quiet, marginalize
        )
        expected = float(np.logaddexp.reduce(values)) if marginalize else max(values)
        assert actual == pytest.approx(expected, abs=1e-12)
        assert reverse == pytest.approx(actual, abs=1e-12)
        assert [(i, c.plane_idx) for i, c in matches] == [(0, 1), (1, 0)]


def test_phase_budget_does_not_relax_shared_clock_agreement():
    pl = repeated_track()
    a, b = pass_near(pl, 150), pass_near(pl, 450)
    start = a[3] - 100
    events = [observation(pl, a, start), observation(pl, b, start + 20)]
    result = score_at_clock(events, [pl], start, cpa_search_s=90)
    assert score.n_consistent_matches(result.matches[0]) <= 1


def test_coarse_phase_refinement_retains_dense_search_precision():
    pl = repeated_track()
    passage = pass_near(pl, 150)
    start = passage[3] - 100
    event = observation(pl, passage, start, bias=24)
    coarse = score_at_clock([event], [pl], start)
    dense = score_at_clock([event], [pl], start, cpa_step_s=0.5)
    np.testing.assert_allclose(coarse.logl, dense.logl, atol=1e-12, rtol=0)
    np.testing.assert_allclose(coarse.t0, dense.t0, atol=1e-9, rtol=0)


def test_disjoint_conflict_components_factorize():
    pair = [
        [
            score._Candidate(value, 0, aircraft, 10, 1000, 1, 0)
            for value, aircraft in row
        ]
        for row in [[(10, 0), (9, 1)], [(9, 0), (0, 1)]]
    ]
    events = []
    for group in range(20):
        events.extend(
            [
                [
                    score._Candidate(
                        c.logl,
                        c.t0,
                        c.plane_idx + 2 * group,
                        c.k_cpa,
                        c.r_min,
                        c.scale,
                        c.chi2nu,
                    )
                    for c in row
                ]
                for row in pair
            ]
        )
    for marginalize in (False, True):
        single, _ = score._joint_association(pair, [5, 5], {}, marginalize)
        combined, matches = score._joint_association(events, [5] * 40, {}, marginalize)
        assert combined == pytest.approx(20 * single, abs=1e-10)
        assert len(matches) == 40


@pytest.mark.parametrize("budget", [-1, np.nan, np.inf])
def test_phase_search_budget_is_finite_and_nonnegative(budget):
    with pytest.raises(ValueError, match="cpa_search_s"):
        score.Options(cpa_search_s=budget)
