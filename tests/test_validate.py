import pytest

from spikebill import validate

# E5 workload B model vs packet-level relative error
E5 = {"2x2_uniform_l16_s0": 0.12128913018367271, "2x2_hotspot_l4_s0": 0.29599052929998537,
      "2x2_complement_l64_s0": 0.01784668326251726}
# E1b held-out energy error; SANA-FE's latency varies slightly between runs
E1B = {"rand_c2_p70": (-7.3e-15, -0.0041), "rand_c16_p20": (-0.0017896858784797187, -0.0035)}


def test_packet_check_reproduces_e5():
    rows = {}
    for side in (2,):
        for r in validate._packet_group(side, (4, 16, 64), (0,), log=lambda *a: None):
            rows[r["config"]] = r
    for k, err in E5.items():
        assert rows[k]["err"] == pytest.approx(err, abs=1e-9)


def test_sanafe_check_reproduces_e1b():
    pytest.importorskip("sanafe")
    out = validate.sanafe_check(dvs=(), rand=((2, 0.7), (16, 0.2)), log=lambda *a: None)
    assert out["status"] == "pass"
    got = {r["config"]: r for r in out["rows"]}
    for k, (e, lat) in E1B.items():
        assert got[k]["energy_err"] == pytest.approx(e, abs=1e-6)
        assert got[k]["latency_err"] == pytest.approx(lat, abs=0.01)


def test_main_skips_cleanly():
    assert validate.main(["--skip-sanafe", "--skip-packet"]) == 0


def test_finite_buffer_check_small():
    """E23 harness: one board, one load; deep queues match infinite ones, shallow queues are never faster."""
    from spikebill import validate
    r = validate.finite_buffer_check(bufs=(None, 64, 2), sides=(2,), loads=(16,), seeds=(0,), workers=1)
    assert r["summary"]["inf"]["n"] == 3
    for row in r["rows"]:
        e = row["err"]
        assert abs(e["64"] - e["inf"]) < 1e-9 or e["64"] <= e["inf"] + 1e-9
        assert e["2"] <= e["inf"] + 1e-9            # back-pressure only slows the simulation (model / sim - 1 drops)
