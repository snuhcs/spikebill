import json

import numpy as np
import pytest

from spikebill import NetBuilder, Simulator


def small_net():
    b = NetBuilder("cnn", steps=4)
    x = b.input("img", (2, 32, 32), rate=0.1)
    c1 = b.conv2d(x, "c1", out_ch=16, k=3, rate=0.1)
    p1 = b.pool(c1, "p1", k=2, stride=2, rate=0.2)
    t = b.tokens(p1)
    q = b.linear(t, "q", d_out=64, rate=0.05)
    o = b.dense(q, "out", 10, rate=0.1)
    return b


def test_builder_sizes():
    g = small_net().graph()
    assert g.populations["c1"].size == 16 * 32 * 32 and g.populations["p1"].size == 16 * 16 * 16
    lin = [p for p in g.projections if p.dst == "q"][0]
    assert lin.shared_weights and lin.share_group == 256 and g.n_synapses(lin) == 256 * 16 * 64


def test_simulator_report(tmp_path):
    sim = Simulator(small_net(), chip="loihi2", multicast="tree")
    rep = sim.run(ranges=True)
    v = rep.results["value"]
    assert rep.chip_steps == 4 + 4 and v["energy_J"] > 0 and v["latency_s"] > 0
    assert rep.results["low"]["energy_J"] <= v["energy_J"] <= rep.results["high"]["energy_J"]
    assert "energy / inference" in rep.summary()
    json.load(open(rep.save(str(tmp_path / "r.json"))))
    assert "<svg" in open(rep.view(str(tmp_path / "r.html"))).read()


def test_simulator_trace_matches_mean():
    sim = Simulator(small_net(), chip="loihi1")
    g = sim.graph
    sp = [{k: p.size * p.rate for k, p in g.populations.items()}] * 3
    ac = [{k: 0.0 for k in g.populations}] * 3
    ev, m = sim.run_trace(sp, ac)
    assert np.allclose(ev["step"], ev["step"][0]) and ev["synops"][0] > 0


def test_activity_from_nir(tmp_path):
    nir = pytest.importorskip("nir")
    from spikebill import activity
    rng = np.random.default_rng(0)
    lif = lambda n: nir.LIF(tau=np.full(n, 10.0), r=np.ones(n), v_leak=np.zeros(n), v_threshold=np.full(n, 0.5),
                            v_reset=np.zeros(n))
    g = nir.NIRGraph(nodes={"in": nir.Input(input_type=np.array([40])),
                            "fc": nir.Affine(weight=np.abs(rng.normal(size=(20, 40))), bias=np.zeros(20)),
                            "l": lif(20), "out": nir.Output(output_type=np.array([20]))},
                     edges=[("in", "fc"), ("fc", "l"), ("l", "out")], type_check=False)
    path = str(tmp_path / "m.nir"); nir.write(path, g)
    spikes = (rng.random((30, 40)) < 0.2).astype(float)
    act = activity.from_nir(path, spikes, device="cpu")
    assert abs(act.rates["in"] - spikes.mean()) < 1e-9 and act.rates["l"] > 0 and act.steps == 30
    rep = Simulator(path, chip="loihi1").run(rates=act.rates, steps=act.steps, balance=False)
    assert rep.results["value"]["energy_J"] > 0


def test_breakdown_sums_to_total():
    from spikebill import NetBuilder, Simulator
    b = NetBuilder("bd", steps=4)
    x = b.input("img", (3, 32, 32), rate=1.0)
    c = b.conv2d(x, "conv1", out_ch=32, k=3, rate=0.1)
    t = b.tokens(b.pool(c, "pool1", rate=0.2))
    b.linear(t, "fc1", d_out=128, rate=0.05)
    for mc in ("none", "tree"):
        rep = Simulator(b, chip="loihi2", multicast=mc).run(balance=False)
        v = rep.results["value"]
        tot = sum(p["total"] for p in rep.by_population.values())
        assert abs(tot - v["energy_J"]) <= 1e-9 * v["energy_J"]
        for comp in ("synaptic", "neuron", "network", "static"):
            s = sum(p[comp] for p in rep.by_population.values())
            assert abs(s - v["split"][comp] * v["energy_J"]) <= 1e-6 * v["energy_J"]
        assert any(p["critical"] for p in rep.by_population.values())
        assert "population" in rep.table()


def test_paper_block_and_chip_version():
    from spikebill import NetBuilder, Simulator
    b = NetBuilder("pb", steps=4)
    x = b.input("img", (3, 16, 16), rate=1.0)
    b.conv2d(x, "c1", out_ch=8, k=3, rate=0.1)
    rep = Simulator(b, chip="loihi2", multicast="tree").run(ranges=True)
    assert rep.provenance["chip_version"] and len(rep.provenance["chip_digest"]) == 12
    tex, md = rep.paper_block(), rep.paper_block("markdown")
    for t in (tex, md):
        assert "multicast tree" in t and rep.provenance["chip_digest"] in t and "45 nm" in t
    assert rep.provenance["opcount_J"] > 0


def test_sensitivity_tornado():
    from spikebill import NetBuilder, Simulator
    b = NetBuilder("tor", steps=4)
    x = b.input("img", (3, 16, 16), rate=1.0)
    b.conv2d(x, "c1", out_ch=8, k=3, rate=0.1)
    sim = Simulator(b, chip="loihi2")
    s = sim.sensitivity(balance=False)
    base = sim.run(balance=False).results["value"]
    assert abs(s["base_energy"] - base["energy_J"]) <= 1e-12 + 1e-9 * base["energy_J"]
    names = [r["param"] for r in s["rows"]]
    assert "e_synop" in names and "max_neurons" not in names
    r = [r for r in s["rows"] if r["param"] == "e_synop"][0]
    assert r["energy_low"] < s["base_energy"] < r["energy_high"]


def test_neuron_update_active_lowers_energy_and_breakdown_sums():
    from spikebill import NetBuilder, Simulator
    b = NetBuilder("nu", steps=4)
    x = b.input("img", (3, 16, 16), rate=0.05)
    b.conv2d(x, "c1", out_ch=8, k=3, rate=0.01)
    s_all = Simulator(b, chip="loihi2").run(balance=False)
    sim = Simulator(b, chip="loihi2"); sim.chip = sim.chip.with_neuron_update("active")
    s_act = sim.run(balance=False)
    assert s_act.results["value"]["energy_J"] < s_all.results["value"]["energy_J"]
    tot = sum(p["total"] for p in s_act.by_population.values())
    assert abs(tot - s_act.results["value"]["energy_J"]) <= 1e-9 * tot
    assert "neuron update active" in s_act.paper_block()


def test_uncertainty_groups_move_together():
    import numpy as np
    from spikebill import NetBuilder, Simulator
    b = NetBuilder("unc", steps=2)
    x = b.input("img", (2, 8, 8), rate=0.5)
    b.conv2d(x, "c1", out_ch=4, k=3, rate=0.1)
    sim = Simulator(b, chip="loihi2")
    u = sim.uncertainty(n=16, balance=False, groups=[["e_synop", "e_neuron_update"]])
    assert u["energy"]["p5"] <= u["energy"]["p50"] <= u["energy"]["p95"] and len(u["energy_samples"]) == 16
    for pr in ("loguniform", "uniform", "triangular"):
        assert np.isfinite(sim.uncertainty(n=4, prior=pr, balance=False)["energy"]["p50"])


def test_fast_path_matches_greedy_run_and_is_cached():
    import time
    from spikebill import NetBuilder, Simulator
    b = NetBuilder("fast", steps=4)
    x = b.input("img", (3, 32, 32), rate=1.0)
    c = b.conv2d(x, "c1", out_ch=16, k=3, rate=0.1)
    b.dense(b.pool(c, "p1", rate=0.2), "fc", 64, rate=0.05)
    sim = Simulator(b, chip="loihi2")
    a = sim.fast(rates={"c1": 0.2})
    ref = sim.run(rates={"c1": 0.2}, balance=False).results["value"]
    assert abs(a["energy_J"] - ref["energy_J"]) <= 1e-9 * ref["energy_J"]
    t0 = time.time()
    for r in (0.05, 0.1, 0.3):
        out = sim.fast(rates={"c1": r})
    assert time.time() - t0 < 1.0 and out["energy_J"] > a["energy_J"] * 0.5


def test_power_trace(tmp_path):
    sim = Simulator(small_net(), chip="loihi1")
    g = sim.graph
    mean = {k: p.size * p.rate for k, p in g.populations.items()}
    rep = sim.run(steps=3, execution="lockstep", balance=False)
    tr = sim.power_trace([mean] * 3)                          # active estimated from received events, as in run
    v = rep.results["value"]
    assert tr.steps == 3 and abs(tr.energy_J - v["energy_J"]) < 1e-9 * v["energy_J"]
    assert abs(tr.latency_s - v["latency_s"]) < 1e-9 * v["latency_s"]
    assert abs(tr.mean_W - v["energy_J"] / v["latency_s"]) < 1e-6 * tr.mean_W
    # a burst: the busy step costs more energy and power than the idle ones; step time = max(terms) + barrier
    idle = {k: 0.0 for k in g.populations}
    tr2 = sim.power_trace([idle, mean, {k: 4 * x for k, x in mean.items()}, idle])
    e = tr2.energy_step_J
    assert e[2] > e[1] > e[0] and np.isclose(e[0], e[3]) and tr2.peak_W == tr2.power_W.max()
    assert tr2.step_s[2] > tr2.step_s[1] > tr2.step_s[0]       # (an idle step is short: its power can be the peak)
    t = tr2.time
    assert np.allclose(tr2.step_s, np.maximum(t["compute"], t["network"]) + t["barrier"])
    assert set(tr2.bound_by()) <= {"compute", "network"}
    assert "peak" in tr2.summary()
    import csv
    rows = list(csv.DictReader(open(tr2.to_csv(str(tmp_path / "p.csv")))))
    assert len(rows) == 4 and np.isclose(sum(float(r["energy_J"]) for r in rows), tr2.energy_J)
    assert json.load(open(tr2.save(str(tmp_path / "p.json"))))["peak_W"] == tr2.peak_W
    page = open(tr2.view(str(tmp_path / "p.html"))).read()
    assert "<svg" in page and "Power per chip step" in page
    long = tr2.view(str(tmp_path / "q.html"), max_points=2)      # bucketed
    assert "mean of 2 chip steps" in open(long).read()


def test_power_trace_coproc():
    sim = Simulator("zoo:spikingformer_8_512", chip="loihi2")
    g = sim.graph
    sp = {k: p.size * p.rate for k, p in g.populations.items()}
    tr = sim.power_trace([sp, sp])
    assert "coproc" in tr.energy and "coproc" in tr.time and tr.energy["coproc"][0] > 0


def test_graph_save_roundtrip(tmp_path):
    from spikebill import graph as graph_mod
    g = Simulator("zoo:sdt_v1_8_512").graph
    g2 = graph_mod.load(g.save(str(tmp_path / "g.json")))
    assert g2.summary() == g.summary() and g2.meta == g.meta and g2.steps_per_inference == g.steps_per_inference
    a = Simulator(g, chip="loihi2").run(balance=False).results["value"]["energy_J"]
    b = Simulator(str(tmp_path / "g.json"), chip="loihi2").run(balance=False).results["value"]["energy_J"]
    assert a == b


def test_spread_fixed_mapping_per_input():
    import numpy as np
    import pytest
    from spikebill import NetBuilder, Simulator
    b = NetBuilder("spread", steps=4)
    x = b.input("img", (3, 32, 32), rate=1.0)
    c = b.conv2d(x, "c1", out_ch=16, k=3, rate=0.1)
    b.dense(b.pool(c, "p1", rate=0.2), "fc", 64, rate=0.05)
    sim = Simulator(b, chip="loihi2")
    same = sim.spread([{"c1": 0.2}] * 3, balance=False)            # identical inputs: no spread, equals run()
    ref = sim.run(rates={"c1": 0.2}, balance=False).results["value"]
    assert same["energy"]["max"] == pytest.approx(same["energy"]["min"], rel=1e-12)
    assert same["energy"]["p50"] == pytest.approx(ref["energy_J"], rel=1e-9)
    assert same["latency"]["p50"] == pytest.approx(ref["latency_s"], rel=1e-9)
    assert same["at_mean"]["energy_J"] == pytest.approx(ref["energy_J"], rel=1e-9)
    s = sim.spread([{"c1": r} for r in (0.05, 0.1, 0.2, 0.4)], balance=False)
    assert np.all(np.diff(s["energy_samples"]) > 0) and s["n"] == 4   # more spikes, more energy, same mapping
    assert s["energy"]["p5"] < s["energy"]["p50"] < s["energy"]["p95"]
    with pytest.raises(ValueError):
        sim.spread([{"nope": 0.1}])
    with pytest.raises(ValueError):
        sim.spread([])
    # tail-aware mapping (E28): the mapping rates are the per-population percentile, or an explicit dict
    rl = [{"c1": r} for r in (0.05, 0.1, 0.2, 0.4)]
    hi = sim.spread(rl, map_rates=100)
    assert hi["map_rates"]["c1"] == pytest.approx(0.4)
    assert sim.spread(rl, map_rates=0)["map_rates"]["c1"] == pytest.approx(0.05)
    d = sim.spread(rl, map_rates={"c1": 0.1875})                  # the mean, given explicitly
    m = sim.spread(rl)
    assert d["energy"]["mean"] == pytest.approx(m["energy"]["mean"], rel=1e-12)
    assert hi["at_mean"]["energy_J"] > 0 and hi["n"] == 4
    for bad in ("p95", 101, {"nope": 0.1}):
        with pytest.raises(ValueError):
            sim.spread(rl, map_rates=bad)


def test_balance_margin_headroom():
    import pytest
    from spikebill import NetBuilder, Simulator, compiled, mapping
    from spikebill import balance as B
    b = NetBuilder("margin", steps=4)
    x = b.input("img", (3, 32, 32), rate=1.0)
    c = b.conv2d(x, "c1", out_ch=32, k=3, rate=0.3)
    c2 = b.conv2d(c, "c2", out_ch=32, k=3, rate=0.05)
    b.dense(b.pool(c2, "p1", rate=0.1), "fc", 64, rate=0.05)
    sim = Simulator(b, chip="loihi2")
    sp, ac, _, _ = sim._activity(None, None, None, "pipelined")
    m0, r0, _, h0 = B.balance(sim.graph, sim.chip, sp, ac)
    m1, r1, _, h1 = B.balance(sim.graph, sim.chip, sp, ac, margin=0.5)
    # the reported result is at the nominal spikes, whatever the margin
    rn = compiled.step_result(compiled.compile_model(sim.graph, sim.chip, m1), sp, ac)
    assert r1["time"]["step"] == pytest.approx(rn["time"]["step"]) and "score_margin" in h1[0]
    assert m1.n_cores >= m0.n_cores                                # headroom never costs fewer cores here
    assert B.band(h1)["n"] >= 1
    with pytest.raises(ValueError):
        B.balance(sim.graph, sim.chip, sp, ac, margin=-0.1)
    rep = sim.run(balance_margin=0.5)
    assert rep.provenance["balance_margin"] == 0.5
    assert rep.results["value"]["latency_s"] > 0
