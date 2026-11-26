import numpy as np
import pytest

from spikebill import chip as chip_mod, compiled, cost, graph as graph_mod, mapping


def net():
    return graph_mod.from_dict(dict(name="n", populations=[dict(name="a", size=1000), dict(name="b", size=900),
                                                           dict(name="c", size=500)],
                                    projections=[dict(src="a", dst="b", kind="dense"),
                                                 dict(src="b", dst="c", kind="sparse", fanout=20),
                                                 dict(src="c", dst="c", kind="conv", fanout=9, weight_bits=1)]))


@pytest.mark.parametrize("upd", ["all", "active"])
@pytest.mark.parametrize("mc", ["none", "tree", "chip_broadcast"])
@pytest.mark.parametrize("chipname", ["toy", "loihi2", "toy3d"])
def test_compiled_equals_step_cost(mc, chipname, upd):
    g = net()
    c = chip_mod.load(chipname).with_multicast(mc).with_neuron_update(upd)
    m = mapping.map_graph(g, c)
    cm = compiled.compile_model(g, c, m)
    rng = np.random.default_rng(0)
    T = 5
    S = rng.uniform(0, 50, size=(T, 3)); S[2, 1] = 0
    A = rng.uniform(0, 300, size=(T, 3))
    ev = compiled.evaluate(cm, S, A)
    for t in range(T):
        sp = dict(zip(cm.pops, S[t])); ac = dict(zip(cm.pops, A[t]))
        r = cost.step_cost(g, c, m, sp, ac)
        for k in ("synaptic", "neuron", "network", "static"):
            assert ev[k][t] == pytest.approx(r["energy"][k], rel=1e-9, abs=1e-18), k
        assert ev["compute"][t] == pytest.approx(r["time"]["compute"], rel=1e-9)
        assert ev["network_t"][t] == pytest.approx(r["time"]["network"], rel=1e-9)
        assert ev["step"][t] == pytest.approx(r["time"]["step"], rel=1e-9)


def test_step_result_matches_step_cost():
    g = net()
    c = chip_mod.load("loihi2").with_multicast("tree")
    m = mapping.map_graph(g, c)
    sp, ac = {"a": 30.0, "b": 12.0, "c": 5.0}, {"a": 100.0, "b": 50.0, "c": 20.0}
    a = compiled.step_result(compiled.compile_model(g, c, m), sp, ac)
    b = cost.step_cost(g, c, m, sp, ac)
    assert np.allclose(a["core_time"], b["core_time"]) and a["time"]["step"] == pytest.approx(b["time"]["step"])


def test_rechip_matches_recompile():
    import numpy as np
    from spikebill import NetBuilder, chip as chip_mod, compiled, mapping as mapping_mod
    b = NetBuilder("rc", steps=4)
    x = b.input("img", (3, 16, 16), rate=1.0)
    c1 = b.conv2d(x, "c1", out_ch=8, k=3, rate=0.1)
    b.dense(c1, "fc", 32, rate=0.05)
    g = b.graph()
    c0 = chip_mod.load("loihi2")
    m = mapping_mod.map_graph(g, c0)
    cm0 = compiled.compile_model(g, c0, m)
    c = c0.with_overrides(e_synop=1.7e-11, t_synop=2.0e-9, t_hop=6.0e-9, p_static_chip=0.4)
    sp = {k: p.size * p.rate for k, p in g.populations.items()}
    ac = {k: float(p.size) for k, p in g.populations.items()}
    a = compiled.step_result(compiled.rechip(cm0, c), sp, ac)
    b_ = compiled.step_result(compiled.compile_model(g, c, m), sp, ac)
    for k in a["energy"]:
        assert np.isclose(a["energy"][k], b_["energy"][k], rtol=1e-9)
    assert np.isclose(a["time"]["step"], b_["time"]["step"], rtol=1e-9)


def test_coproc_compiled_equals_step_cost_and_first_order():
    g = net()
    g.populations["b"].coproc_ops = 40.0; g.populations["a"].coproc_msgs = 1.0
    c = chip_mod.load("toy").with_coproc_defaults()
    m = mapping.map_graph(g, c)
    cm = compiled.compile_model(g, c, m)
    sp, ac = {"a": 30.0, "b": 12.0, "c": 5.0}, {"a": 100.0, "b": 50.0, "c": 20.0}
    a, b = compiled.step_result(cm, sp, ac), cost.step_cost(g, c, m, sp, ac)
    for k in ("synaptic", "neuron", "network", "static", "coproc"):
        assert a["energy"][k] == pytest.approx(b["energy"][k], rel=1e-9)
    assert a["energy"]["coproc"] == pytest.approx(12 * 40 * 0.03e-12 + 30 * c["e_spike_out"])
    assert a["time"]["coproc"] == pytest.approx(12 * 40 * 0.125e-9) and a["time"]["step"] == pytest.approx(b["time"]["step"])
    bd = compiled.breakdown(cm, sp, ac)
    assert sum(v["total"] for v in bd.values()) == pytest.approx(sum(a["energy"].values()), rel=1e-9)
    # slow co-processor: it sets the step time
    c2 = c.with_overrides(t_coproc_op=1e-5)
    r2 = compiled.step_result(compiled.compile_model(g, c2, m), sp, ac)
    assert r2["time"]["step"] == pytest.approx(12 * 40 * 1e-5 + r2["time"]["barrier"])
    assert "coproc" in cost.first_order(g, c, rates={"a": 0.1, "b": 0.1, "c": 0.1})["split"]
    assert chip_mod.load("toy").c("e_coproc_op") == 0.03e-12
