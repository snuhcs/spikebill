import pytest

from spikebill import chip as chip_mod, zoo

# synaptic events per inference (E10, "synops"): the zoo graphs are the E10 graphs
E10_SYNOPS = {"sdt_v1_8_512": 8026656144.318144, "spikingformer_8_512": 6854515859.537921}


def test_names_and_meta():
    assert zoo.names() == ["sdt_v1_8_512", "sdt_v2_31m", "spikegpt_216m", "spikingformer_8_512"]
    for n in zoo.names():
        m = zoo.info(n)
        assert (m["claim_mJ"] is None or m["claim_mJ"] > 0) and m["paper"] and m["rates"]
    assert zoo.info("spikegpt_216m")["claim_mJ"] is None and "1024 tokens" in zoo.info("spikegpt_216m")["unit"]
    with pytest.raises(KeyError):
        zoo.load("nope")


@pytest.mark.parametrize("name", sorted(E10_SYNOPS))
def test_graph_matches_e10(name):
    g = zoo.load(name)
    assert g.steps_per_inference == 4
    syn = sum(g.syn_events_per_step().values()) * g.steps_per_inference
    assert syn == pytest.approx(E10_SYNOPS[name], rel=1e-9)


def test_attention_variants():
    for n in zoo.names():
        cop = zoo.load(n)
        exc = zoo.load(n, attention="excluded")
        ops = sum(p.coproc_ops * p.rate * p.size for p in cop.populations.values())
        assert ops > 0 and cop.meta["attention"] == "coproc"
        assert all(p.coproc_ops == 0 and p.coproc_msgs == 0 for p in exc.populations.values())
        assert exc.meta["attention"] == "excluded"
    with pytest.raises(KeyError):
        zoo.load("spikegpt_216m", attention="andgate")          # no AND-gate variant shipped


def test_shipped_variants():
    """AND-gate and RepConv-literal graphs (white paper Sec. 5 baselines, E10 v5 baselines)."""
    assert zoo.variants("sdt_v2_31m") == ["55m", "andgate", "repconv_literal"]
    m55 = zoo.load("sdt_v2_31m", variant="55m")
    assert m55.meta["claim_mJ"] is None and "assumption" in m55.meta["rates"] and m55.name == "sdt_v2_31m.55m"
    assert sum(p.size for p in m55.populations.values()) == 32791656          # E19(b) graph
    assert zoo.variants("sdt_v1_8_512") == zoo.variants("spikingformer_8_512") == ["andgate"]
    assert zoo.variants("spikegpt_216m") == []
    for n in ("sdt_v1_8_512", "spikingformer_8_512", "sdt_v2_31m"):
        a = zoo.load(n, attention="andgate")
        assert a.name == f"{n}.andgate" and a.meta["attention"] == "andgate" and a.meta["variant"] == "andgate"
        assert all(p.coproc_ops == 0 for p in a.populations.values())
        assert len(a.populations) > len(zoo.load(n).populations)
        assert zoo.from_ref(f"{n}.andgate").to_dict() == a.to_dict()
    lit = zoo.load("sdt_v2_31m", variant="repconv_literal")
    assert lit.meta["attention"] == "coproc" and len(lit.populations) == 160
    with pytest.raises(KeyError):
        zoo.load("sdt_v1_8_512", variant="repconv_literal")
    with pytest.raises(ValueError):
        zoo.load("sdt_v2_31m", attention="andgate", variant="repconv_literal")
    with pytest.raises(KeyError):
        zoo.load("sdt_v1_8_512.andgate")


def test_andgate_greedy_matches_e10():
    """Greedy cost of the shipped AND-gate SDT V1 equals the E10 v5 baseline 445.3 mJ)."""
    from spikebill import Simulator
    s = zoo.REFERENCE_SETTINGS
    r = Simulator("zoo:sdt_v1_8_512.andgate", chip="loihi2", multicast=s["multicast"], static=s["static"]).run(
        execution="pipelined", balance=False).results["value"]
    assert r["energy_J"] * 1e3 == pytest.approx(445.29, abs=0.05) and r["chips"] == 161


def test_reference_current():
    """The shipped reference numbers must be recomputed when the loihi2 chip file changes."""
    refs = zoo.reference()
    assert sorted(refs) == zoo.names()
    v = chip_mod.load("loihi2").version
    for n, r in refs.items():
        assert r["chip_version"] == v, f"{n}: reference computed on {r['chip_version']}, chip file is {v}"
        assert r["low"]["energy_J"] <= r["value"]["energy_J"] <= r["high"]["energy_J"]
        claim = zoo.info(n)["claim_mJ"]
        if claim is None:
            assert r["value"]["ratio_to_claim"] is None
        else:
            assert r["value"]["ratio_to_claim"] == pytest.approx(r["value"]["energy_J"] * 1e3 / claim)


def test_simulator_accepts_zoo_name():
    from spikebill import Simulator
    sim = Simulator("zoo:sdt_v1_8_512")
    assert sim.graph.name == "sdt_v1_8_512"
    assert "e_coproc_op" in sim.chip.params                 # co-processor defaults added


def test_spikegpt_matches_e19():
    """zoo:spikegpt_216m is the E19(a) graph: dense synapses = linear weights of the release (176,631,552)."""
    g = zoo.load("spikegpt_216m")
    assert g.steps_per_inference == 1024
    assert sum(g.n_synapses(q) for q in g.projections if q.kind == "dense") == 176631552
    assert zoo.reference("spikegpt_216m")["value"]["chips"] == 16
