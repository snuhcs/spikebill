from spikebill import chip as chip_mod, cost, graph as graph_mod, sweep


def _one(cfg):
    g = graph_mod.from_dict(dict(name="s", populations=[dict(name="a", size=500, rate=0.1), dict(name="b", size=300)],
                                 projections=[dict(src="a", dst="b", kind="dense")]))
    return cost.estimate(g, chip_mod.load("toy").at(cfg["which"]))["energy_per_inference"]


def test_sweep_matches_serial():
    cfgs = [dict(which=w) for w in ("low", "value", "high")]
    assert sweep.run(_one, cfgs, workers=3) == [_one(c) for c in cfgs]
