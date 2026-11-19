"""A low straight-fit speed is uncertain evidence, not an aircraft veto."""

import copy
import datetime as dt

import numpy as np
import pytest
from test_passage_association import (
    observation,
    pass_near,
    repeated_track,
    score_at_clock,
)

from dopplergeo import locate, score


def flagged_observation():
    plane = repeated_track()
    passage = pass_near(plane, 150)
    start = passage[3] - 100
    event = observation(plane, passage, start)
    # Real-trajectory replay also reproduces this failure: turning C172 traces
    # at51m/s can yield a clean19.7m/s straight fit and this same flag.
    event.speed_mps = 20
    event.likely_ground_vehicle = True
    return plane, start, event


def test_ground_like_candidate_multiplicity_is_bounded_and_still_returns_support():
    plane, start, event = flagged_observation()
    one = score_at_clock([event], [plane], start, silence_penalty=0)
    many = score_at_clock(
        [event], [copy.deepcopy(plane) for _ in range(20)], start, silence_penalty=0
    )
    np.testing.assert_allclose(one.logl, many.logl, atol=1e-12, rtol=0)
    assert len(one.matches[0]) == len(many.matches[0]) == 1
    assert one.matches[0][0]["event_ground_like"] is True
    assert one.logl[0] <= np.logaddexp(-1, 0) + 1e-12
    points = np.array([[0.0, 0.0, 0.0], [3000.0, 2000.0, 0.0]])
    opts = score.Options(silence_penalty=0)
    a = score.score_points(
        [event], [plane], points, 343, opts, (start - 0.1, start + 0.1), 200
    )
    b = score.score_points(
        [event],
        [copy.deepcopy(plane) for _ in range(20)],
        points,
        343,
        opts,
        (start - 0.1, start + 0.1),
        200,
    )
    np.testing.assert_allclose(
        score.posterior(a.logl, 1)[0], score.posterior(b.logl, 1)[0], atol=1e-12, rtol=0
    )


def test_optional_source_is_not_rejected_by_its_biased_straight_fit_slant():
    plane, start, event = flagged_observation()
    event.cpa_slant_m = 20000  # actual slant is1000; a straight-fit artifact
    assert score.cpa_consistency(event, 1000, score.Options()) == 0
    result = score_at_clock([event], [plane], start, silence_penalty=0)
    assert score.n_consistent_matches(result.matches[0]) == 1
    assert result.matches[0][0]["geometry_prior_ignored"] is True
    event.likely_ground_vehicle = False
    assert score.cpa_consistency(event, 1000, score.Options()) < -2


def test_unmatched_ground_like_clutter_has_a_strictly_bounded_miss_cost():
    plane = repeated_track()
    a, b = pass_near(plane, 150), pass_near(plane, 450)
    start = a[3] - 100
    clean = [observation(plane, p, start) for p in (a, b)]
    clutter = []
    for i in range(20):
        event = copy.deepcopy(clean[0])
        event.likely_ground_vehicle = True
        event.t_start += 5000 + i * 100
        event.t_end += 5000 + i * 100
        event.t_cpa += 5000 + i * 100
        event.times = (np.asarray(event.times) + 5000 + i * 100).tolist()
        clutter.append(event)
    baseline = score_at_clock(clean, [plane], start, silence_penalty=0)
    result = score_at_clock(clean + clutter, [plane], start, silence_penalty=0)
    assert result.logl[0] == pytest.approx(baseline.logl[0] - 20)
    assert result.t0[0] == baseline.t0[0]
    assert score.n_consistent_matches(result.matches[0]) == 2


@pytest.mark.parametrize("budget_error", [False, True])
def test_locate_retains_ground_like_track_support_and_preserves_low_confidence(
    tmp_path, monkeypatch, budget_error
):
    plane, start, event = flagged_observation()
    monkeypatch.setattr(locate.metadata, "hints", lambda *_: {})
    monkeypatch.setattr(
        locate,
        "detect_cached",
        lambda *a, **k: ([event], [], 200, None, {"hit": False}),
    )
    monkeypatch.setattr(locate.env, "sound_speed", lambda *a, **k: (343, "fixture"))
    monkeypatch.setattr(score, "prepare_planes", lambda *a, **k: [plane])

    def known_point(events, planes, *args, **kwargs):
        assert events == [event]
        if budget_error:
            raise score.AssociationBudgetError(
                "exact association state budget exceeded"
            )
        result = score_at_clock(events, planes, start, silence_penalty=0)
        best = {
            "lat": 43.8,
            "lon": -79.3,
            "logl": result.logl[0],
            "t0_unix": result.t0[0],
            "matches": result.matches[0],
        }
        level = {
            "step": 50,
            "x": np.array([0]),
            "lat": np.array([43.8]),
            "lon": np.array([-79.3]),
            "logl": result.logl,
            "t0": result.t0,
            "matches": result.matches,
            "a68_m2": 2500,
            "a95_m2": 2500,
        }
        return {"best": best, "levels": [level]}

    monkeypatch.setattr(score, "search", known_point)
    report = locate.locate(
        "fixture.wav",
        dt.datetime.fromtimestamp(start, dt.UTC),
        (43.8, -79.3, 1),
        tracks=[{}],
        plots=False,
        out_dir=tmp_path,
        offline=True,
    )
    if budget_error:
        assert report["ok"] is False
        assert "state budget" in report["error"]
        assert report["association_state_budget"] == 4096
        return
    assert len(report["best"]["matches"]) == 1
    assert report["events"][0]["likely_ground_vehicle"] is True
    assert report["best"]["confidence"] == "low"
    assert report["event_policy"]["ground_like_miss_cost"] == 1


@pytest.mark.parametrize("step", [0, -1, np.nan, np.inf])
def test_coarse_phase_step_must_be_finite_positive(step):
    with pytest.raises(ValueError, match="cpa_step_s"):
        score.Options(cpa_step_s=step)


@pytest.mark.parametrize("budget", [-1, np.nan, np.inf])
def test_optional_miss_budget_must_be_finite_nonnegative(budget):
    with pytest.raises(ValueError, match="ground_like_miss_cost"):
        score.Options(ground_like_miss_cost=budget)
