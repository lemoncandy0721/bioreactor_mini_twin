import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twin.generate import default_design, run_design  # noqa: E402
from twin.plant import FeedPlan, PlantParams, _apply_bolus, simulate  # noqa: E402

X0 = dict(X=0.4, G=35.0, L=0.0, P=0.0, V=1.0)


def test_batch_volume_constant_and_states_nonnegative():
    r = simulate(X0, FeedPlan(rate_L_per_h=0.0), t_end_h=200)
    assert np.allclose(r["V"], 1.0)
    for k in ["X", "G", "L", "P"]:
        assert (r[k] >= 0).all()


def test_continuous_feed_volume_matches_rate():
    feed = FeedPlan(rate_L_per_h=0.001, start_h=72)
    r = simulate(X0, feed, t_end_h=172)
    assert r["V"][-1] == pytest.approx(1.0 + 0.001 * 100, rel=1e-4)


def test_max_volume_stops_feed():
    feed = FeedPlan(rate_L_per_h=0.01, start_h=0, max_volume_L=1.3)
    r = simulate(X0, feed, t_end_h=200)
    assert r["V"][-1] == pytest.approx(1.3, abs=0.02)


def test_bolus_mixing_conserves_mass():
    y = np.array([10.0, 2.0, 20.0, 1000.0, 1.0])
    y2 = _apply_bolus(y, 0.25, G_feed=400.0)
    V2 = y2[4]
    assert V2 == pytest.approx(1.25)
    assert y2[0] * V2 == pytest.approx(10.0 * 1.0)            # 세포 수 보존
    assert y2[1] * V2 == pytest.approx(2.0 + 400.0 * 0.25)    # 포도당 몰수
    assert y2[3] * V2 == pytest.approx(1000.0 * 1.0)          # 항체 질량 보존


def test_sample_at_bolus_time_is_pre_feed():
    feed = FeedPlan(bolus=[(96.0, 0.05)])
    r = simulate(X0, feed, t_end_h=120)
    i = int(np.where(r["t"] == 96.0)[0][0])
    assert r["V"][i] == pytest.approx(1.0)          # 공급 직전
    assert r["V"][i + 1] == pytest.approx(1.05)     # 공급 직후


def test_generation_is_reproducible():
    specs = default_design(seed=3)
    m1, _, _ = run_design(specs, PlantParams(), seed=3)
    m2, _, _ = run_design(default_design(seed=3), PlantParams(), seed=3)
    assert m1.equals(m2)
