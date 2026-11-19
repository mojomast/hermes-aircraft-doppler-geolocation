"""Hermetic regressions for distinct encounters, joint assignment and CPA bias."""

import copy
import itertools

import numpy as np
import pytest

from dopplergeo import detect, score
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


def test_stride_prefilter_covers_a_tangent_pass_between_anchors():
    t = np.arange(21.0)
    radius, altitude = 11999.9, 120.0
    y = np.sqrt(radius**2 - altitude**2)
    pos = np.c_[(t - 2) * 100, np.full_like(t, y), np.full_like(t, altitude)]
    plane = score.Plane("tangent", "piston", t, pos)
    assert np.linalg.norm(pos[::5, :2], axis=1).min() > 12000
    passes = score.find_passes(
        [plane], np.zeros((1, 3)), score.Options(r_max_m=12000), C
    )[0]
    assert len(passes) == 1
    assert passes[0][1] == 2
    assert passes[0][2] == pytest.approx(radius)


def test_broad_phase_neighbor_alias_cannot_duplicate_one_accepted_pass():
    t = np.arange(-100.0, 401.0)
    angle = 2 * np.pi * t / 50
    plane = score.Plane(
        "circling",
        "piston",
        t,
        np.c_[1000 + 400 * np.cos(angle), 400 * np.sin(angle), np.full_like(t, 800)],
    )
    passage = pass_near(plane, 150)
    start = passage[3] - 100
    times = np.arange(90, 110.01, 0.5)
    ratio = score.predicted_ratio(plane, np.zeros(3), start + times, C, passage[1], 120)
    event, reason = detect._accept(
        {
            "lines": {1: (times, 80 * ratio, 10 * np.ones_like(times))},
            "t_start": 90,
            "t_end": 110,
            "snr": 15,
        },
        C,
        0.5,
        2,
        1e-4,
    )
    assert event is not None, reason
    assert event.straight_fit_ok and not event.likely_ground_vehicle
    result = score_at_clock(
        [event, copy.deepcopy(event)], [plane], start, silence_penalty=0
    )
    assert score.n_consistent_matches(result.matches[0]) == 1
    assert len(result.matches[0]) == 1
    assert start + 90 <= result.matches[0][0]["cpa_receive_unix"] <= start + 110


def test_support_spanning_multiple_minima_is_explicitly_confidence_ambiguous():
    t = np.arange(-100.0, 401.0)
    angle = 2 * np.pi * t / 50
    plane = score.Plane(
        "circling",
        "piston",
        t,
        np.c_[1000 + 400 * np.cos(angle), 400 * np.sin(angle), np.full_like(t, 800)],
    )
    passage = pass_near(plane, 150)
    start = passage[3] - 100
    event = observation(plane, passage, start)
    event.times = np.arange(90, 170.01, 0.5).tolist()
    event.t_start, event.t_end = 90, 170
    event.ratio = score.predicted_ratio(
        plane, np.zeros(3), start + np.asarray(event.times), C, passage[1], 180
    ).tolist()
    result = score_at_clock([event], [plane], start, silence_penalty=0)
    assert len(result.matches[0]) == 1
    assert result.matches[0][0]["multiple_cpas_in_support"] is True
    assert score.n_consistent_matches(result.matches[0]) == 0


def test_shared_clock_interval_is_found_without_an_optional_seed():
    plane = repeated_track()
    a, b, c = [pass_near(plane, when) for when in (150, 450, 750)]
    start = a[3] - 100
    ordinary = [observation(plane, a, start), observation(plane, b, start + 10)]
    optional = observation(plane, c, start + 5)
    optional.likely_ground_vehicle = True
    options = score.Options(silence_penalty=0)

    def evaluate(events):
        return score.score_points(
            events,
            [plane],
            np.zeros((1, 3)),
            C,
            options,
            (start - 20, start + 20),
            1000,
        )

    baseline, supported = evaluate(ordinary), evaluate([*ordinary, optional])
    assert score.n_consistent_matches(baseline.matches[0]) == 2
    assert supported.logl[0] - baseline.logl[0] <= np.logaddexp(0, 1) + 1e-10


def test_clock_boundary_fit_is_finite_without_rounding_outside_window():
    plane = repeated_track()
    passage = pass_near(plane, 150)
    start = passage[3] - 100
    result = score_at_clock(
        [observation(plane, passage, start - 8)], [plane], start, silence_penalty=0
    )
    assert np.isfinite(result.logl[0]) and np.isfinite(result.t0[0])
    assert start - 0.1 <= result.t0[0] <= start + 0.1


@pytest.mark.parametrize("offset", [-5, 5])
@pytest.mark.parametrize("half_width", [0, 0.1])
def test_shared_clock_respects_requested_bounds_with_individual_fit_slack(
    offset, half_width
):
    plane = repeated_track()
    passage = pass_near(plane, 150)
    start = passage[3] - 100
    bounds = (start - half_width, start + half_width)
    result = score.score_points(
        [observation(plane, passage, start + offset)],
        [plane],
        np.zeros((1, 3)),
        C,
        score.Options(silence_penalty=0),
        bounds,
        500,
    )
    assert bounds[0] <= result.t0[0] <= bounds[1]
    assert score.n_consistent_matches(result.matches[0]) == 1
    fitted = result.matches[0][0]["recording_start_unix"]
    # Preserve individual-event clock jitter at a hard shared-clock boundary.
    assert fitted - start == pytest.approx(offset, abs=0.3)
    assert abs(fitted - result.t0[0]) <= 6


def test_dense_marginal_component_has_an_explicit_state_budget():
    events = [
        [score._Candidate(1, 0, aircraft, 10, 1000, 1, 0) for aircraft in range(6)]
        for _ in range(6)
    ]
    with pytest.raises(score.AssociationBudgetError, match="state budget"):
        score._joint_association(events, [5] * 6, {}, True, state_budget=16)


def test_map_assignment_uses_polynomial_path_for_a_large_dense_component():
    events = [
        [score._Candidate(1, 0, aircraft, 10, 1000, 1, 0) for aircraft in range(40)]
        for _ in range(40)
    ]
    value, matches = score._joint_association(
        events, [5] * 40, {}, False, state_budget=1
    )
    assert value == 40 and len(matches) == 40
    assert len({c.occurrence for _, c in matches}) == 40


@pytest.mark.parametrize("budget", [0, -1, np.inf, True])
def test_association_state_budget_is_a_positive_integer(budget):
    with pytest.raises(ValueError, match="association_state_budget"):
        score.Options(association_state_budget=budget)


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
