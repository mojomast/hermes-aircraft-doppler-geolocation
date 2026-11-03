"""Vectorised Viterbi recursions == the v0.3.0 per-jump np.roll loops (frozen below)."""

import numpy as np

from dopplergeo import broadband, detect


# --- frozen v0.3.0 reference implementations -------------------------------
def ref_viterbi_segments(score, max_jump, birth=6.0, death=6.0, jump_pen=0.1):
    """Viterbi over states {off, f0 bins}. score>0 rewards being on a cell.

    Returns a list of (frame_start, frame_end, bins[]) segments.
    """
    nb, nt = score.shape
    NEG = -1e18
    on = np.full(nb, NEG)
    off = 0.0
    back_on = np.zeros((nb, nt), dtype=np.int32)  # -1 means came from off
    back_off = np.zeros(nt, dtype=np.int32)  # -1 stay off, else came from bin
    for i in range(nt):
        best = np.full(nb, NEG)
        arg = np.full(nb, -1, dtype=np.int32)
        for j in range(-max_jump, max_jump + 1):
            sh = np.roll(on, j) - jump_pen * abs(j)
            if j > 0:
                sh[:j] = NEG
            elif j < 0:
                sh[j:] = NEG
            m = sh > best
            best[m] = sh[m]
            arg[m] = (np.arange(nb) - j)[m]
        from_off = off - birth
        m = from_off > best
        best[m] = from_off
        arg[m] = -1
        k = int(np.argmax(on))
        if on[k] - death > off:
            new_off, back_off[i] = on[k] - death, k
        else:
            new_off, back_off[i] = off, -1
        on = best + score[:, i]
        back_on[:, i] = arg
        off = new_off
    # backtrack
    segs, cur = [], []
    k = int(np.argmax(on))
    state = k if on[k] > off else -1
    for i in range(nt - 1, -1, -1):
        if state >= 0:
            cur.append((i, state))
            prev = back_on[state, i]
            if prev < 0:
                segs.append(cur[::-1])
                cur = []
            state = prev
        else:
            prev = back_off[i]
            state = prev  # -1 stays off, else we were on that bin at frame i-1
    if cur:
        segs.append(cur[::-1])
    out = []
    for s in segs[::-1]:
        fr = np.array([a for a, _ in s])
        bins = np.array([b for _, b in s])
        out.append((int(fr[0]), int(fr[-1]), bins))
    return out


def ref_track_segments(Z, max_jump=20, z_thr=4.0, birth=8.0, min_frames=16):
    """Viterbi track of the cepstral peak with an 'off' state.

    Emission score z - z_thr for on-states; bounded jump per hop. Returns a list of
    (frame_indices, q_indices) runs where the path is 'on'.
    """
    nt, nq = Z.shape
    if nt == 0:
        return []
    S = np.clip(Z, -2, 12) - z_thr
    v_on = S[0].copy() - birth
    v_off = 0.0
    bp = np.zeros((nt, nq), np.int32)  # predecessor q (>=0) or -1 = from off
    off_from_on = np.full(nt, -1, np.int32)
    for t in range(1, nt):
        best = np.full(nq, -np.inf)
        arg = np.zeros(nq, np.int32)
        for d in range(-max_jump, max_jump + 1):
            src = np.roll(v_on, d)
            if d > 0:
                src[:d] = -np.inf
            elif d < 0:
                src[d:] = -np.inf
            upd = src > best
            best[upd] = src[upd]
            arg[upd] = (np.arange(nq) - d)[upd]
        from_off = v_off - birth
        use_off = from_off > best
        best[use_off] = from_off
        arg[use_off] = -1
        k = int(np.argmax(v_on))
        if v_on[k] > v_off:
            new_off, off_from_on[t] = float(v_on[k]), k
        else:
            new_off, off_from_on[t] = v_off, -1
        v_on = best + S[t]
        bp[t] = arg
        v_off = new_off
    # backtrack
    state = int(np.argmax(v_on)) if v_on.max() > v_off else -1
    path = np.full(nt, -1, np.int32)
    for t in range(nt - 1, -1, -1):
        path[t] = state
        if t == 0:
            break
        if state >= 0:
            state = int(bp[t, state])
        else:
            state = int(off_from_on[t])
    runs, cur = [], []
    for t in range(nt):
        if path[t] >= 0:
            cur.append(t)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [(np.array(r), path[np.array(r)]) for r in runs if len(r) >= min_frames]


def test_tonal_viterbi_identical():
    rng = np.random.default_rng(0)
    for trial in range(6):
        nb, nt = 300, 400
        s = np.clip(rng.normal(-0.2, 1.0, (nb, nt)), -0.35, 6.0)
        # a few ridges plus exact ties (quantised scores) to exercise tie-breaks
        for _ in range(3):
            b0 = rng.integers(20, nb - 20)
            for i in range(100, 250):
                s[int(b0 - (i - 100) * 0.05 * trial) % nb, i] = 4.0
        if trial % 2:
            s = np.round(s * 2) / 2
        mask = np.ones_like(s)
        mask[rng.random(s.shape) < 0.05] = 0
        sc = s * mask + (mask - 1) * 5
        for mj in (2, 5):
            a = ref_viterbi_segments(sc, mj)
            b = detect.viterbi_segments(sc, mj)
            assert len(a) == len(b) and len(a) > 0
            for x, y in zip(a, b):
                assert x[0] == y[0] and x[1] == y[1]
                np.testing.assert_array_equal(x[2], y[2])


def test_broadband_track_identical():
    rng = np.random.default_rng(1)
    for trial in range(4):
        Z = rng.normal(0, 1, (300, 500)).astype(np.float32)
        for i in range(50, 200):
            Z[i, 200 + int(40 * np.sin(i / 30))] = 9.0
        if trial % 2:
            Z = np.round(Z)
        for mj in (5, 30):
            a = ref_track_segments(Z, max_jump=mj)
            b = broadband.track_segments(Z, max_jump=mj)
            assert len(a) == len(b) and len(a) > 0
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x[0], y[0])
                np.testing.assert_array_equal(x[1], y[1])


def _same_tracks(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x[0], y[0])
        np.testing.assert_array_equal(x[1], y[1])


def test_broadband_float32_scores_accumulate_in_float64():
    # review repro: a 1e-5 lead at frame 0 survives only in float64 accumulation
    Z = np.zeros((2000, 80), dtype=np.float32)
    Z[:, 5] = 4.1
    Z[:, 70] = 4.1
    Z[0, 70] = 4.10001
    a = ref_track_segments(Z, max_jump=30)
    b = broadband.track_segments(Z, max_jump=30)
    assert a and int(a[0][1][0]) == 70
    _same_tracks(a, b)


def test_broadband_float32_randomised_long_inputs():
    rng = np.random.default_rng(7)
    for trial in range(6):
        nt, nq = (3000, 200) if trial < 3 else (600, 1170)
        Z = rng.normal(0, 1, (nt, nq)).astype(np.float32)
        for _ in range(3):
            q0 = int(rng.integers(40, nq - 40))
            t0 = int(rng.integers(0, nt - 300))
            for i in range(t0, t0 + 250):
                Z[i, q0 + int(20 * np.sin(i / 40))] = np.float32(4.0 + rng.random())
        if trial % 2:
            Z = (np.round(Z * 4) / 4).astype(np.float32)  # many exact ties
        _same_tracks(
            ref_track_segments(Z, max_jump=30), broadband.track_segments(Z, max_jump=30)
        )


def test_tonal_viterbi_float32_input_matches():
    rng = np.random.default_rng(9)
    s = np.clip(rng.normal(-0.2, 1, (400, 1500)), -0.35, 6).astype(np.float32)
    s[100, :] = 4.0
    s[300, :] = 4.0
    s[300, 0] = np.float32(4.00001)
    a = ref_viterbi_segments(s, 5)
    b = detect.viterbi_segments(s, 5)
    assert len(a) == len(b) and a
    for x, y in zip(a, b):
        assert x[:2] == y[:2]
        np.testing.assert_array_equal(x[2], y[2])
