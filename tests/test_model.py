import os
import numpy as np
import pytest

from spikebill import chip as chip_mod, cost, graph as graph_mod, mapping, traffic


def two_layer(n_in=200, n_out=150, rate=0.1, kind="dense", fanout=None):
    return graph_mod.from_dict(dict(
        name="t", steps_per_inference=10,
        populations=[dict(name="a", size=n_in, rate=rate), dict(name="b", size=n_out, rate=0.0)],
        projections=[dict(src="a", dst="b", kind=kind, fanout=fanout, weight_bits=8)]))


@pytest.fixture
def toy():
    return chip_mod.load("toy")


def spikes_of(g):
    return cost.mean_activity(g)[0]


def test_chip_missing_param_raises():
    with pytest.raises(ValueError):
        chip_mod.from_dict(dict(name="x", topology=dict(cores_x=1, cores_y=1), params=dict(e_synop=1)))


def test_mapping_respects_limits(toy):
    m = mapping.map_graph(two_layer(), toy)
    a, b = m.pops["a"], m.pops["b"]
    assert a.per_core == 100 and a.n_cores == 2          # no inputs: neuron limit
    # b: fanin 200 * 8 bits = 1600 bits/neuron -> 50 neurons by memory
    assert b.per_core == 50 and b.limited_by == "syn_mem_bits" and b.n_cores == 3
    assert m.n_cores == 5 and m.n_chips == 1


def test_dense_traffic_counts(toy):
    g = two_layer()
    m = mapping.map_graph(g, toy)
    tr = traffic.step_traffic(g, m, toy, spikes_of(g))
    spikes = 200 * 0.1
    assert tr.messages == pytest.approx(spikes * 3)          # every spike reaches all 3 dst cores
    assert tr.core_synops.sum() == pytest.approx(spikes * 150)
    assert tr.chip_hops == 0
    exp = sum(np.abs(m.xy[s] - m.xy[d]).sum() * spikes / 2 for s in m.pops["a"].cores for d in m.pops["b"].cores)
    assert tr.hops == pytest.approx(exp)


def test_link_loads_match_bruteforce(toy):
    g = two_layer(n_in=1000, n_out=900, rate=0.05)
    m = mapping.map_graph(g, toy)
    tr = traffic.step_traffic(g, m, toy, spikes_of(g))
    loads = {}
    w = 1000 * 0.05 / m.pops["a"].n_cores
    for s in m.pops["a"].cores:
        for d in m.pops["b"].cores:
            (x, y), (x2, y2) = m.xy[s], m.xy[d]
            while x != x2:
                nx = x + (1 if x2 > x else -1); loads[(x, y, nx, y)] = loads.get((x, y, nx, y), 0) + w; x = nx
            while y != y2:
                ny = y + (1 if y2 > y else -1); loads[(x, y, x, ny)] = loads.get((x, y, x, ny), 0) + w; y = ny

    def inter(k):
        x, y, x2, y2 = k
        return x // toy.tiles_x != x2 // toy.tiles_x or y // toy.tiles_y != y2 // toy.tiles_y
    assert tr.max_link_intra == pytest.approx(max([v for k, v in loads.items() if not inter(k)] or [0]))
    assert tr.max_link_inter == pytest.approx(max([v for k, v in loads.items() if inter(k)] or [0]))
    assert tr.hops == pytest.approx(sum(loads.values()))
    assert m.n_chips > 1 and tr.chip_hops > 0


def test_random_sparse_reach(toy):
    g = two_layer(n_out=1000, kind="sparse", fanout=5)
    m = mapping.map_graph(g, toy)
    k = m.pops["b"].n_cores
    assert traffic.cores_reached(g, m, g.projections[0]) == pytest.approx(k * (1 - (1 - 1 / k) ** 5))


def test_cores_in_one_tile_cost_no_hops():
    c = chip_mod.load("loihi1")
    g = two_layer(n_in=2000, n_out=1000)     # explicit placement: 3 cores of tile 0 (limits not checked)
    p = {"a": [(0, 1000), (1, 1000)], "b": [(2, 1000)]}
    m = mapping.from_placement(g, c, p)
    assert m.n_tiles == 1
    tr = traffic.step_traffic(g, m, c, {"a": 10.0})
    assert tr.hops == 0 and tr.messages == pytest.approx(10.0)


def test_flows_from_edges_exact():
    # 4 src neurons on 2 cores, 4 dst neurons on 2 cores; src i -> dst i and dst (i+1)%4
    src_core_of = np.array([0, 0, 1, 1]); dst_core_of = np.array([0, 0, 1, 1])
    pre = np.array([0, 0, 1, 1, 2, 2, 3, 3]); post = np.array([0, 1, 1, 2, 2, 3, 3, 0])
    s, d, mw, sw = traffic.flows_from_edges(src_core_of, dst_core_of, pre, post, [2, 2])
    got = {(a, b): (x, y) for a, b, x, y in zip(s, d, mw, sw)}
    # src core 0: neuron 0 -> core 0 only (2 syn); neuron 1 -> cores 0 and 1 (1 syn each)
    assert got[(0, 0)] == pytest.approx((1.0, 1.5)) and got[(0, 1)] == pytest.approx((0.5, 0.5))
    assert got[(1, 1)] == pytest.approx((1.0, 1.5)) and got[(1, 0)] == pytest.approx((0.5, 0.5))


def test_barrier_table():
    c = chip_mod.load("loihi1")
    assert c.barrier(1, 1) == 0.6e-6 and c.barrier(3, 1) == 1.0e-6 and c.barrier(30, 1) == 1.8e-6
    assert c.barrier(3, 2) == c["t_barrier_multichip"]


def test_estimate_scales_with_activity(toy):
    lo = cost.estimate(two_layer(rate=0.05), toy, active_frac={"a": 0.0, "b": 0.0})
    hi = cost.estimate(two_layer(rate=0.10), toy, active_frac={"a": 0.0, "b": 0.0})
    assert hi["energy_per_step"]["synaptic"] == pytest.approx(2 * lo["energy_per_step"]["synaptic"])
    assert lo["latency_per_inference"] == pytest.approx(10 * lo["time_per_step"]["step"])
    assert sum(lo["energy_split"].values()) == pytest.approx(1)


def test_range_ends(toy):
    g = two_layer()
    assert cost.estimate(g, toy.at("high"))["energy_per_step"]["synaptic"] == \
        pytest.approx(2 * cost.estimate(g, toy)["energy_per_step"]["synaptic"])


def test_rate_weighted_flows():
    # core 0 holds neurons 0 (fires, 2 edges to core 1) and 1 (silent, no edges)
    s, d, mw, sw = traffic.flows_from_edges(np.array([0, 0]), np.array([0, 1]), [0, 0], [1, 1], [2],
                                            rates=[1.0, 0.0])
    assert list(zip(s, d)) == [(0, 1)] and mw[0] == pytest.approx(1.0) and sw[0] == pytest.approx(2.0)
    s, d, mw, sw = traffic.flows_from_edges(np.array([0, 0]), np.array([0, 1]), [0, 0], [1, 1], [2])
    assert mw[0] == pytest.approx(0.5) and sw[0] == pytest.approx(1.0)


def test_core_timing_max_vs_sum(toy):
    g = two_layer()
    a = cost.estimate(g, toy.with_core_timing("sum"))["time_per_step"]["compute"]
    b = cost.estimate(g, toy)["time_per_step"]["compute"]
    assert b < a


def test_local_projection_axons_shared(toy):
    # ring-like local projection with fanin 15 must not be capped at max_fanin_axons / 15 neurons per core
    g = graph_mod.from_dict(dict(name="ring", populations=[dict(name="m", size=5000)],
                                 projections=[dict(src="m", dst="m", kind="conv", fanout=15, weight_bits=1)]))
    n, which, _ = mapping.neurons_per_core(g, toy, "m")
    assert n == 100 and which == "max_neurons"


def test_tree_multicast_bruteforce(toy):
    # dense fan-out from one core to many: tree links = union of XY paths
    g = two_layer(n_in=50, n_out=900, rate=0.2)
    m = mapping.map_graph(g, toy)
    uni = traffic.step_traffic(g, m, toy, spikes_of(g))
    tr = traffic.step_traffic(g, m, toy.with_multicast("tree"), spikes_of(g))
    spikes = 50 * 0.2
    links = set()
    (sx, sy) = m.xy[m.pops["a"].cores[0]]
    for d in m.pops["b"].cores:
        x, y = sx, sy; x2, y2 = m.xy[d]
        while x != x2:
            nx = x + (1 if x2 > x else -1); links.add((x, y, nx, y)); x = nx
        while y != y2:
            ny = y + (1 if y2 > y else -1); links.add((x, y, x, ny)); y = ny
    assert tr.messages == pytest.approx(spikes)
    assert tr.hops == pytest.approx(spikes * len(links))
    assert tr.hops < uni.hops and tr.core_synops.sum() == pytest.approx(uni.core_synops.sum())
    assert tr.max_link_intra + tr.max_link_inter <= spikes + 1e-9


def test_funcsim_semantics():
    pytest.importorskip("torch")
    from spikebill import funcsim
    pops = [funcsim.FPop("a", 2, 0.5, bias=np.array([1.0, 0.0])), funcsim.FPop("b", 1, 0.5)]
    projs = [funcsim.FProj("a", "b", np.array([0]), np.array([0]), np.array([1.0]))]
    lo = funcsim.simulate(pops, projs, 3, "loihi", device="cpu")
    fw = funcsim.simulate(pops, projs, 3, "framework", device="cpu")
    # a0 fires every step; b fires one step later under Loihi semantics, the same step under framework semantics
    assert [s["a"] for s in lo.spikes] == [1, 1, 1] and [s["b"] for s in lo.spikes] == [0, 1, 1]
    assert [s["b"] for s in fw.spikes] == [1, 1, 1]
    assert lo.active[0]["a"] == 1 and lo.active[1]["b"] == 1


def test_static_accounting(toy):
    g = two_layer()
    e = {m: cost.estimate(g, toy.with_static_accounting(m))["energy_per_step"]["static"] for m in ("chip", "cores", "none")}
    assert e["none"] == 0 and 0 < e["cores"] < e["chip"]


def test_shared_weight_groups(toy):
    # 1x1 conv over 50 tokens, 4 -> 40 channels, 8-bit: private would need 4*8 = 32 bits per neuron (2500 per core
    # capped at 100 by max_neurons); shared by token: a core holding n neurons holds ceil(n/50)+1 channels
    g = graph_mod.from_dict(dict(name="s", populations=[dict(name="x", size=200), dict(name="y", size=2000)],
                                 projections=[dict(src="x", dst="y", kind="conv", fanout=400, weight_bits=8,
                                                   shared_weights=True, share_group=50)]))
    c = toy.with_overrides(syn_mem_bits=40 * 8 * 2)        # room for 2 channels of fan-in 40
    n, which, _ = mapping.neurons_per_core(g, c, "y")
    assert which == "syn_mem_bits" and n == 50             # ceil(50/50)+1 = 2 channels fit, 51 would need 3


def test_token_shared_layout_and_reach():
    # token-wise linear 64 tokens x 512 -> 512 channels, 8-bit, on Loihi 2 limits (1 Mbit, 4096 axons, 8192 neurons)
    c = chip_mod.load("loihi2")
    g = graph_mod.from_dict(dict(name="t", populations=[dict(name="x", size=64 * 512), dict(name="q", size=64 * 512)],
                                 projections=[dict(src="x", dst="q", kind="conv", fanout=512, weight_bits=8,
                                                   shared_weights=True, share_group=64)]))
    t, ch, n, lim = mapping.shared_layout(g, c, "q")
    assert t == 8 and ch == 256 and n == 2048            # axons: 8 x 512 = 4096; memory: 256 x 512 x 8 = 1 Mbit
    m = mapping.map_graph(g, c)
    assert m.pops["q"].per_core == 2048 and traffic.cores_reached(g, m, g.projections[0]) == 2


def test_3d_link_loads_match_bruteforce():
    c = chip_mod.load("toy3d")
    g = two_layer(n_in=1000, n_out=900, rate=0.05)
    m = mapping.map_graph(g, c)
    assert m.layers == 2 and set(np.unique(m.z)) == {0, 1}
    tr = traffic.step_traffic(g, m, c, spikes_of(g))
    loads, w = {}, 1000 * 0.05 / m.pops["a"].n_cores
    for s in m.pops["a"].cores:
        for d in m.pops["b"].cores:
            x, y, z = int(m.xy[s, 0]), int(m.xy[s, 1]), int(m.z[s])
            x2, y2, z2 = int(m.xy[d, 0]), int(m.xy[d, 1]), int(m.z[d])
            while x != x2:
                nx = x + (1 if x2 > x else -1); loads[(x, y, z, nx, y, z)] = loads.get((x, y, z, nx, y, z), 0) + w; x = nx
            while y != y2:
                ny = y + (1 if y2 > y else -1); loads[(x, y, z, x, ny, z)] = loads.get((x, y, z, x, ny, z), 0) + w; y = ny
            while z != z2:
                nz = z + (1 if z2 > z else -1); loads[(x, y, z, x, y, nz)] = loads.get((x, y, z, x, y, nz), 0) + w; z = nz
    vert = [v for k, v in loads.items() if k[2] != k[5]]
    plan = {k: v for k, v in loads.items() if k[2] == k[5]}
    inter = lambda k: k[0] // c.tiles_x != k[3] // c.tiles_x or k[1] // c.tiles_y != k[4] // c.tiles_y  # noqa: E731
    assert tr.max_link_vert == pytest.approx(max(vert))
    assert tr.max_link_intra == pytest.approx(max([v for k, v in plan.items() if not inter(k)] or [0]))
    assert tr.max_link_inter == pytest.approx(max([v for k, v in plan.items() if inter(k)] or [0]))
    assert tr.hops + tr.vhops == pytest.approx(sum(loads.values()))


def test_balance_reduces_hot_core_time():
    from spikebill import balance
    c = chip_mod.load("loihi1")
    # a hot population (dense, high rate) next to a cold one; balancing must not raise the step time
    g = graph_mod.from_dict(dict(name="b", populations=[dict(name="a", size=4000, rate=0.3),
                                                        dict(name="h", size=4000), dict(name="c", size=4000)],
                                 projections=[dict(src="a", dst="h", kind="dense", weight_bits=1),
                                              dict(src="h", dst="c", kind="sparse", fanout=4)]))
    sp, ac = cost.mean_activity(g)
    m0 = mapping.map_graph(g, c)
    r0 = cost.step_cost(g, c, m0, sp, ac)
    m, r, caps, hist = balance.balance(g, c, sp, ac, objective="latency")
    assert r["time"]["step"] <= r0["time"]["step"] and "h" in caps and m.pops["h"].per_core < m0.pops["h"].per_core


def test_chip_validation_errors():
    import copy
    import pytest
    import yaml
    from spikebill import chip as chip_mod
    d = yaml.safe_load(open(os.path.join(chip_mod.CHIP_DIR, "toy.yaml")))
    chip_mod.from_dict(copy.deepcopy(d))
    bad = copy.deepcopy(d); bad["params"]["e_synopp"] = bad["params"].pop("e_synop")
    with pytest.raises(ValueError, match="missing"):
        chip_mod.from_dict(bad)
    bad = copy.deepcopy(d); bad["params"]["e_synp"] = {"value": 1e-12}
    with pytest.raises(ValueError, match="did you mean 'e_synop'"):
        chip_mod.from_dict(bad)
    bad = copy.deepcopy(d); bad["params"]["e_hop"] = {"value": 1e-12, "low": 2e-12, "high": 3e-12}
    with pytest.raises(ValueError, match="low <= value <= high"):
        chip_mod.from_dict(bad)
    bad = copy.deepcopy(d); bad["params"]["t_hop"] = {"value": 1.0, "unit": "ns"}
    with pytest.raises(ValueError, match="SI units"):
        chip_mod.from_dict(bad)
    bad = copy.deepcopy(d); bad["multicast"] = "broadcast"
    with pytest.raises(ValueError, match="multicast"):
        chip_mod.from_dict(bad)
    assert "e_synop" in chip_mod.describe("toy")


def test_first_order_spreadsheet():
    from spikebill import NetBuilder, chip as chip_mod, cost
    b = NetBuilder("fo", steps=4)
    x = b.input("img", (3, 16, 16), rate=1.0)
    b.conv2d(x, "c1", out_ch=8, k=3, rate=0.1)
    g = b.graph()
    c = chip_mod.load("loihi2")
    r = cost.first_order(g, c, execution="lockstep")
    ev = 3 * 16 * 16 * 1.0 * (8 * 16 * 16 * 3 * 9) / (3 * 16 * 16)
    assert r["chips"] == 1 and r["chip_steps"] == 4
    assert abs(r["events_per_step"] - ev) < 1e-6 * ev
    assert r["energy_J"] > ev * c["e_synop"] * 4


def test_chip_broadcast_bruteforce(toy):
    # dense fan-out from one core: one packet per remote chip to its nearest destination tile, unicast on the own
    # chip; synaptic work and messages received unchanged
    g = two_layer(n_in=50, n_out=4000, rate=0.2)
    m = mapping.map_graph(g, toy)
    uni = traffic.step_traffic(g, m, toy, spikes_of(g))
    bc = traffic.step_traffic(g, m, toy.with_multicast("chip_broadcast"), spikes_of(g))
    spikes = 50 * 0.2
    src = m.pops["a"].cores
    assert len(src) == 1
    sx, sy = m.xy[src[0]]
    chip_of = lambda x, y: (x // toy.tiles_x, y // toy.tiles_y)
    dst = [tuple(m.xy[d]) for d in m.pops["b"].cores]
    local = [d for d in dst if chip_of(*d) == chip_of(sx, sy)]
    remote = {chip_of(*d) for d in dst} - {chip_of(sx, sy)}
    assert len(remote) >= 1
    assert bc.messages == pytest.approx(spikes * (len(local) + len(remote)))
    assert bc.core_out_msgs.sum() == pytest.approx(bc.messages)
    assert np.allclose(bc.core_in_msgs, uni.core_in_msgs) and np.allclose(bc.core_synops, uni.core_synops)
    # chip hops: one packet per remote chip along its XY route to the nearest destination tile
    ch = 0
    for k in remote:
        near = min((abs(x - sx) + abs(y - sy), x, y) for x, y in dst if chip_of(x, y) == k)
        ch += abs(near[1] // toy.tiles_x - sx // toy.tiles_x) + abs(near[2] // toy.tiles_y - sy // toy.tiles_y)
    assert bc.chip_hops == pytest.approx(spikes * ch)
    assert bc.chip_hops < uni.chip_hops


def test_balance_targets_ignore_placement_and_band():
    # the target range comes from population times only (gap 15): a chip with slower links must search the
    # same targets; band() covers the chosen mapping
    from spikebill import balance
    c = chip_mod.load("loihi1")
    g = graph_mod.from_dict(dict(name="b", populations=[dict(name="a", size=4000, rate=0.3),
                                                        dict(name="h", size=4000), dict(name="c", size=4000),
                                                        dict(name="d", size=2000)],
                                 projections=[dict(src="a", dst="h", kind="dense", weight_bits=1),
                                              dict(src="h", dst="c", kind="sparse", fanout=4),
                                              dict(src="c", dst="d", kind="sparse", fanout=2)]))
    sp, ac = cost.mean_activity(g)
    slow = c.with_overrides(t_hop=c.params["t_hop"]["value"] * 50) if "t_hop" in c.params else c
    _, r, _, h1 = balance.balance(g, c, sp, ac, n_targets=6)
    _, _, _, h2 = balance.balance(g, slow, sp, ac, n_targets=6)
    assert [x["target"] for x in h1] == [x["target"] for x in h2] and len(h1) == 7
    b = balance.band(h1, tol=0.05)
    e = sum(r["energy"].values())
    assert b["n"] >= 1 and b["energy"][0] <= e <= b["energy"][1] and b["chips"][0] <= b["chips"][1]


def test_packetsim_step_chip_broadcast():
    # gap 7: the packet-level driver in every multicast mode; chip_broadcast sends fewer packets than unicast and
    # its time agrees with the aggregated network term (E5b tolerance 20%)
    from spikebill import packetsim, traffic
    g = graph_mod.from_dict(dict(name="s", populations=[dict(name="a", size=16384), dict(name="b", size=16384)],
                                 projections=[dict(src="a", dst="b", kind="sparse", fanout=256, weight_bits=1)]))
    sp = {"a": 16384 * 0.01}
    res = {}
    for mc in ("none", "chip_broadcast"):
        c = chip_mod.load("loihi2").with_multicast(mc)
        m = mapping.map_graph(g, c)
        assert m.n_chips > 1
        r = packetsim.simulate_step(g, m, c, sp, seed=0)
        tr = traffic.step_traffic(g, m, c, sp)
        t_model = max(tr.core_out_msgs.max() * c["t_spike_out"], tr.max_link_intra * c["t_link"],
                      tr.max_link_inter * c["t_chip_hop"]) + tr.max_path_time
        assert abs(t_model / r["t_sim"] - 1) < 0.2
        res[mc] = r
    assert res["chip_broadcast"]["packets"] < res["none"]["packets"] and res["chip_broadcast"]["copies"] > 0
    assert res["none"]["copies"] == 0


def test_broadcast_links_route():
    from spikebill import packetsim
    net = packetsim.Net(8, 4, 4, 4, 1.0, 2.0)
    links = packetsim.broadcast_links(net, (0, 0), (5, 1), [(5, 1), (7, 1), (5, 3)])
    # leg (0,0)->(5,1): 5 + 1 hops; copies (5,1)->(7,1): 2 hops, (5,1)->(5,3): 2 hops; the entry itself: none
    assert len(links) == 6 + 2 + 2
    leg_end = 5
    assert sum(1 for _, p in links if p == leg_end) == 2 and sum(1 for _, p in links if p == -1) == 1


def test_with_overrides_point_and_range():
    """Gap 9: a numeric override is a point (range runs keep it fixed); a tuple / dict sets a range."""
    import pytest
    from spikebill import chip as chip_mod
    c = chip_mod.load("loihi2")
    k = "e_synop"
    assert c.params[k]["low"] < c.params[k]["high"]
    big = c.params[k]["high"] * 10
    o = c.with_overrides(**{k: big})
    assert o.at("low")[k] == o.at("high")[k] == o[k] == big        # was: low/high from the file, low > value
    r = c.with_overrides(**{k: (1e-12, 2e-12, 4e-12)})
    assert (r.at("low")[k], r[k], r.at("high")[k]) == (1e-12, 2e-12, 4e-12)
    d = c.with_overrides(**{k: dict(value=2e-12, high=3e-12)})
    assert (d.at("low")[k], d[k], d.at("high")[k]) == (2e-12, 2e-12, 3e-12)
    with pytest.raises(ValueError):
        c.with_overrides(**{k: (3e-12, 2e-12, 4e-12)})
    assert c.params[k]["value"] != big                               # original untouched


def test_to_table_latex_escapes_arrows():
    """Paper request 2026-11-24: "->" in a source rendered as "-¿"; every special character is escaped."""
    from spikebill import chip as chip_mod
    assert chip_mod._latex_escape("a -> b <= c, 5% & x_y #1 ~ ^ {} $ \\") == \
        "a $\\to$ b $\\le$ c, 5\\% \\& x\\_y \\#1 \\textasciitilde{} \\textasciicircum{} \\{\\} \\$ \\textbackslash{}"
    tex = chip_mod.to_table("loihi1", fmt="latex")
    body = "\n".join(l for l in tex.splitlines() if not l.startswith("%"))
    assert "->" not in body and "$\\to$" in body


def test_coproc_units_per_unit():
    """Gap 10: per_unit co-processor time = busiest unit; pooled = all ops; both cost paths agree; energy unchanged."""
    import numpy as np
    from spikebill import Simulator, zoo, compiled, mapping as mapping_mod
    from spikebill.cost import coproc_unit_ops, step_cost
    g = zoo.load("spikingformer_8_512")
    pooled = Simulator(g, multicast="tree")
    per = Simulator(g, multicast="tree", coproc_units="per_unit")
    assert per.chip.coproc_units == "per_unit" and pooled.chip.coproc_units == "pooled"
    spikes = {k: p.size * p.rate for k, p in g.populations.items()}
    units = coproc_unit_ops(g, spikes)
    assert len(units) == 8                                            # one co-processor population per block
    m = mapping_mod.map_graph(g, pooled.chip)
    act = {k: float(p.size) for k, p in g.populations.items()}
    for sim, want in ((pooled, sum(units.values())), (per, max(units.values()))):
        r = compiled.step_result(compiled.compile_model(g, sim.chip, m), spikes, act)
        assert np.isclose(r["time"]["coproc"], want * sim.chip.c("t_coproc_op"), rtol=1e-9)
        r2 = step_cost(g, sim.chip, m, spikes, act)
        assert np.isclose(r2["time"]["coproc"], r["time"]["coproc"], rtol=1e-6)
    e_pool = compiled.step_result(compiled.compile_model(g, pooled.chip, m), spikes, act)["energy"]["coproc"]
    e_per = compiled.step_result(compiled.compile_model(g, per.chip, m), spikes, act)["energy"]["coproc"]
    assert np.isclose(e_pool, e_per)
    rep = per.run(balance=False)
    assert "one unit per block" in rep.paper_block() and rep.provenance["coproc_units"] == "per_unit"


def test_to_table_paper_format():
    """fmt="paper": 5 columns, range in one cell ("--" when none), escaped sources, barrier row spans 4 columns."""
    from spikebill import chip as chip_mod
    t = chip_mod.to_table("loihi1", fmt="paper")
    assert "p{38mm}p{72mm}" in t and "Symbol & Value & Range & Meaning & Source" in t
    assert "2.9e-10--5.3e-09" in t and "\\multicolumn{4}{l}" in t and "->" not in t.split("\n", 2)[2]


def test_plots_save_all_and_compare(tmp_path):
    """spikebill.plots: the four single-report figures and a comparison render without error."""
    import pytest
    pytest.importorskip("matplotlib")
    from spikebill import Simulator, NetBuilder, plots
    b = NetBuilder("toy_plots", steps=4)
    x = b.input("inp", (512,), rate=0.2)
    h = b.dense(x, "h", 2048, rate=0.05)
    b.dense(h, "out", 256, rate=0.1)
    sim = Simulator(b, chip="loihi2")
    r1, r2 = sim.run(balance=False), sim.run()
    paths = plots.save_all(r1, str(tmp_path))
    assert len(paths) == 8 and all((tmp_path / p.split("/")[-1]).stat().st_size > 0 for p in paths)
    plots.compare([r1, r2], ["greedy", "balanced"]).savefig(str(tmp_path / "cmp.png"))


def test_chip_broadcast_dense_source_no_nan():
    """mw = 1 (dense source reaching every destination core) across several chips: chip_broadcast stays finite and
    matches the step_cost path (bug found in E19: 0 * log(0) = NaN for chips without destination cores)."""
    import numpy as np
    from spikebill import NetBuilder, Simulator, compiled, mapping as mapping_mod, cost
    b = NetBuilder("dense_mc", steps=2)
    x = b.input("x", (768,), rate=1.0)
    h = b.dense(x, "h", 60000, rate=0.1)
    b.dense(h, "o", 768, rate=0.1)
    sim = Simulator(b, chip="loihi2", multicast="chip_broadcast")
    g = sim.graph
    m = mapping_mod.map_graph(g, sim.chip)
    assert m.n_chips > 1
    sp = {k: p.size * p.rate for k, p in g.populations.items()}
    ac = {k: float(p.size) for k, p in g.populations.items()}
    r = compiled.step_result(compiled.compile_model(g, sim.chip, m), sp, ac)
    assert all(np.isfinite(v) for v in r["energy"].values()) and np.isfinite(r["time"]["step"])
    r2 = cost.step_cost(g, sim.chip, m, sp, ac)
    assert np.isclose(r["time"]["network"], r2["time"]["network"], rtol=0.02)


def _chain_sim(n=8, width=4096):
    from spikebill import NetBuilder, Simulator
    b = NetBuilder("chain_place", steps=2)
    x = b.input("p0", (width,), rate=0.2)
    for i in range(1, n):
        x = b.dense(x, f"p{i}", width, rate=0.1)
    return Simulator(b, chip="loihi2", multicast="none")


def test_map_graph_fill_order():
    """map_graph(order=...) permutes where populations sit, not how many cores each gets."""
    import pytest
    from spikebill import mapping as mapping_mod
    sim = _chain_sim()
    g, ch = sim.graph, sim.chip
    m0 = mapping_mod.map_graph(g, ch)
    m1 = mapping_mod.map_graph(g, ch, order=list(g.populations)[::-1])
    assert list(m1.pops) == list(g.populations) and m1.n_cores == m0.n_cores
    for k in g.populations:
        assert (m0.pops[k].counts == m1.pops[k].counts).all()
    assert m1.pops["p0"].cores[0] == m0.n_cores - m0.pops["p0"].n_cores      # filled last
    with pytest.raises(ValueError):
        mapping_mod.map_graph(g, ch, order=["p0"])


def test_place_proxy_matches_unicast_hops():
    """For dense unicast projections the placement proxy equals the full model's hop + chip-hop energy."""
    import numpy as np
    from spikebill import compiled, mapping as mapping_mod, place
    sim = _chain_sim()
    sp, ac, _, _ = sim._activity(None, None, None, "lockstep")
    for order in (None, list(sim.graph.populations)[::-1]):
        m = mapping_mod.map_graph(sim.graph, sim.chip, order=order)
        assert m.n_chips > 1
        cm = compiled.compile_model(sim.graph, sim.chip, m)
        S = np.array([sp[k] for k in cm.pops])
        full = ((cm.tot[1] - cm.tot[2]) @ S) * sim.chip["e_hop"] + (cm.tot[2] @ S) * sim.chip["e_chip_hop"]
        px = place.Proxy(sim.graph, sim.chip, mapping_mod.map_graph(sim.graph, sim.chip), sp)
        o = np.arange(len(px.names)) if order is None else np.array([px.idx[k] for k in order])
        assert abs(px.cost(o) / full - 1) < 1e-6


def test_place_anneal_recovers_chain_order():
    """From a shuffled order, annealing gets back to (or below) the chain's natural order; search never returns a
    worse full-model score than the graph order."""
    import numpy as np
    from spikebill import mapping as mapping_mod, place
    sim = _chain_sim()
    sp, ac, _, _ = sim._activity(None, None, None, "lockstep")
    px = place.Proxy(sim.graph, sim.chip, mapping_mod.map_graph(sim.graph, sim.chip), sp)
    natural = px.cost(np.arange(len(px.names)))
    shuffled = np.random.default_rng(0).permutation(len(px.names))
    best, c_best, c0 = place.anneal(px, init=shuffled, iters=3000, seed=0)
    assert c0 > natural and c_best <= natural * 1.001
    res = place.search(sim.graph, sim.chip, sp, ac, iters=500)
    rows = {r["candidate"]: r for r in res["rows"]}
    assert set(rows) == {"graph_order", "spectral", "anneal", "anneal_calibrated"} | {f"bisection_{b}" for b in range(5)}
    assert min(r["score"] for r in res["rows"]) <= rows["graph_order"]["score"]


def test_report_load_and_compare(tmp_path):
    """A saved report reads back with the same numbers; compare / diff of greedy vs balanced; CLI --compare."""
    import pytest
    from spikebill import NetBuilder, Simulator
    from spikebill.api import Report
    from spikebill import cli
    b = NetBuilder("toy_cmp", steps=4)
    x = b.input("inp", (512,), rate=0.2)
    h = b.dense(x, "h", 2048, rate=0.05)
    b.dense(h, "out", 256, rate=0.1)
    sim = Simulator(b, chip="loihi2")
    r1, r2 = sim.run(balance=False, ranges=True), sim.run(ranges=True)
    p1, p2 = r1.save(str(tmp_path / "greedy.json")), r2.save(str(tmp_path / "bal.json"))
    l1 = Report.load(p1)
    assert l1.results["value"]["energy_J"] == pytest.approx(r1.results["value"]["energy_J"])
    assert l1.by_population.keys() == r1.by_population.keys() and l1.summary() == r1.summary()
    d = l1.diff(Report.load(p2))
    assert d["results"]["value"]["energy_J"]["ratio"] == pytest.approx(
        r2.results["value"]["energy_J"] / r1.results["value"]["energy_J"])
    assert set(d["results"]) == {"value", "low", "high"}
    assert sum(v["b"] for v in d["split"].values()) == pytest.approx(r2.results["value"]["energy_J"])
    txt = l1.compare(Report.load(p2), labels=("greedy", "balanced"))
    assert "greedy" in txt and "step terms" in txt
    assert cli.main(["--compare", p1, p2])["results"]["value"]["chips"]["a"] == r1.results["value"]["chips"]
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(ValueError):
        Report.load(str(tmp_path / "bad.json"))


def test_cli_trace_power(tmp_path):
    """spikebill MODEL --trace (.npz and .json) gives the same PowerTrace as the API; outputs are written."""
    import json
    import numpy as np
    import pytest
    from spikebill import NetBuilder, Simulator, cli
    b = NetBuilder("toy_trace", steps=6)
    x = b.input("inp", (256,), rate=0.2)
    h = b.dense(x, "h", 1024, rate=0.05)
    b.dense(h, "out", 128, rate=0.1)
    gpath = b.graph().save(str(tmp_path / "toy.json")) or str(tmp_path / "toy.json")
    rng = np.random.default_rng(0)
    arr = {"inp": rng.integers(0, 80, 6), "h": rng.integers(0, 120, 6), "out": rng.integers(0, 20, 6)}
    np.savez(tmp_path / "t.npz", **arr)
    tr = cli.main([gpath, "--trace", str(tmp_path / "t.npz"), "--no-balance",
                   "--power-html", str(tmp_path / "p.html"), "--power-csv", str(tmp_path / "p.csv")])
    spikes_t = [{k: float(v[t]) for k, v in arr.items()} for t in range(6)]
    ref = Simulator(b, chip="loihi2").power_trace(spikes_t)
    assert tr.energy_J == pytest.approx(ref.energy_J) and tr.steps == 6
    assert (tmp_path / "p.html").stat().st_size > 0 and (tmp_path / "p.csv").stat().st_size > 0
    (tmp_path / "t.json").write_text(json.dumps(spikes_t))
    assert cli.main([gpath, "--trace", str(tmp_path / "t.json"), "--no-balance"]).energy_J == pytest.approx(ref.energy_J)
    (tmp_path / "bad.json").write_text(json.dumps([{"nope": 1.0}]))
    with pytest.raises(ValueError):
        cli.main([gpath, "--trace", str(tmp_path / "bad.json")])


def test_balance_budgets():
    """max_chips / max_energy_increase restrict the balancer's choice; no budget = unchanged choice."""
    from spikebill import NetBuilder, Simulator, balance as balance_mod
    b = NetBuilder("toy_budget", steps=4)
    x = b.input("inp", (1024,), rate=0.3)
    h = b.dense(x, "h", 4096, rate=0.2)
    h2 = b.dense(h, "h2", 4096, rate=0.2)
    b.dense(h2, "out", 512, rate=0.1)
    sim = Simulator(b, chip="loihi2", multicast="tree")
    sp, ac, _, _ = sim._activity(None, None, None, "pipelined")
    m, r, caps, hist = balance_mod.balance(sim.graph, sim.chip, sp, ac, objective="latency")
    assert all(h["eligible"] for h in hist)
    assert r["time"]["step"] == min(h["step"] for h in hist)
    e_min = min(h["energy"] for h in hist)
    m2, r2, _, hist2 = balance_mod.balance(sim.graph, sim.chip, sp, ac, objective="latency", max_energy_increase=0.0)
    assert sum(r2["energy"].values()) == e_min
    c_min = min(h["chips"] for h in hist)
    m3, _, _, _ = balance_mod.balance(sim.graph, sim.chip, sp, ac, objective="latency", max_chips=c_min)
    assert m3.n_chips == c_min
    rep = sim.run(max_chips=c_min)
    assert rep.results["value"]["chips"] <= c_min and "Balancing budget" in rep.paper_block("markdown")
    assert sim.run().provenance["balance_budget"] is None


def test_run_place_option():
    """run(place=True) never scores worse than the graph-order fill and records the placement in provenance."""
    sim = _chain_sim()
    a, b = sim.run(execution="lockstep", balance=False), sim.run(execution="lockstep", balance=False, place=True)
    sa = a.results["value"]["energy_J"] * a.results["value"]["step_s"]
    sb = b.results["value"]["energy_J"] * b.results["value"]["step_s"]
    assert sb <= sa * (1 + 1e-9)
    assert a.provenance["placement"] == "graph order" and "placement" in b.paper_block("markdown")


def test_place_calibrated_score_matches_reference():
    """After calibrate, the proxy score of the reference order equals the full model's objective there."""
    import numpy as np
    import pytest
    from spikebill import compiled, mapping as mapping_mod, place, balance
    sim = _chain_sim()
    sp, ac, _, _ = sim._activity(None, None, None, "lockstep")
    m = mapping_mod.map_graph(sim.graph, sim.chip)
    r = compiled.step_result(compiled.compile_model(sim.graph, sim.chip, m), sp, ac)
    px = place.Proxy(sim.graph, sim.chip, m, sp)
    o = np.arange(len(px.names))
    for obj in ("edp", "energy", "latency"):
        px.calibrate(o, r, obj)
        assert px.score(o) == pytest.approx(balance._score(r, obj), rel=1e-9)
    assert px.evaluate(o[::-1])[1] > 0


def test_balance_joint_placement():
    """balance(place=True) searches every candidate's placement: never worse than the graph-order balancer."""
    from spikebill import balance as balance_mod
    sim = _chain_sim()
    sp, ac, _, _ = sim._activity(None, None, None, "lockstep")
    _, r0, _, h0 = balance_mod.balance(sim.graph, sim.chip, sp, ac, n_targets=4)
    _, r1, _, h1 = balance_mod.balance(sim.graph, sim.chip, sp, ac, n_targets=4, place=True)
    assert balance_mod._score(r1, "edp") <= balance_mod._score(r0, "edp") * (1 + 1e-9)
    assert len(h1) == len(h0)
    rep = sim.run(execution="lockstep", place=True)
    assert "joint" in rep.provenance["placement"]


def test_variance_check():
    """Binomial per-core spikes: with many neurons per core the expected step is close to the mean-count step; a
    hot sender split into one-neuron cores has an expected compute term well above its mean-count value."""
    from spikebill import NetBuilder, Simulator
    b = NetBuilder("toy_var", steps=4)
    x = b.input("inp", (1024,), rate=0.2)
    h = b.dense(x, "h", 4096, rate=0.05)
    b.dense(h, "out", 512, rate=0.1)
    rep = Simulator(b, chip="loihi2").run(balance=False)
    v = rep.variance_check(steps=100)
    assert 0.98 < v["R"] < 1.2 and v["step_p95_s"] >= v["step_mean_s"] * 0.99
    from spikebill import mapping, compiled, variance
    b2 = NetBuilder("toy_hot", steps=4)
    src = b2.input("src", (64,), rate=0.05)
    dst = b2.pop("dst", (8192,), rate=0.0)
    b2.connect(src, dst, kind="sparse", fanout=64)
    sim = Simulator(b2, chip="loihi2")
    sp, ac, _, _ = sim._activity(None, None, None, "lockstep")
    m = mapping.map_graph(sim.graph, sim.chip, caps={"src": 1, "dst": 64})
    v2 = variance.busiest_core(compiled.compile_model(sim.graph, sim.chip.at("value"), m), sp, ac, steps=200)
    assert v2["R_compute"] > 1.2 and v2["compute_mean_s"] > v2["compute_ref_s"]
    cm = compiled.compile_model(sim.graph, sim.chip.at("value"), m)
    e, ref = variance.expected_compute(cm, sp, ac)                  # analytic E[max] (exact Binomial, gap 25)
    assert ref == pytest.approx(v2["compute_ref_s"]) and e == pytest.approx(v2["compute_mean_s"], rel=0.03)


def test_run_spike_variance():
    """spike_variance="binomial" raises only the compute term (never below the mean-count step) and is recorded."""
    from spikebill import NetBuilder, Simulator
    b = NetBuilder("toy_hot2", steps=4)
    src = b.input("src", (64,), rate=0.05)
    dst = b.pop("dst", (8192,), rate=0.0)
    b.connect(src, dst, kind="sparse", fanout=64)
    sim = Simulator(b, chip="loihi2")
    r0, r1 = sim.run(balance=False), sim.run(balance=False, spike_variance="binomial")
    v0, v1 = r0.results["value"], r1.results["value"]
    assert v1["step_s"] >= v0["step_s"] and v1["time"]["compute"] >= v0["time"]["compute"]
    assert v1["time"]["network"] == v0["time"]["network"]
    assert r1.provenance["spike_variance"] == "binomial" and "Binomial" in r1.paper_block("markdown")
    with pytest.raises(ValueError):
        sim.run(balance=False, spike_variance="gauss")


def test_packetsim_finite_buffers():
    """Infinite buffers reproduce simulate_mixed exactly; buffers larger than any queue change nothing; a 1-flit
    buffer delays this hotspot step."""
    import numpy as np
    from spikebill import packetsim, chip as chip_mod, mapping as mm
    chip = chip_mod.load("loihi2")
    ids = np.arange(4 * chip.cores_per_chip)
    xy, _, _, board, _ = mm._geometry(chip, ids, 4)
    net = packetsim.Net(board[0] * chip.tiles_x, board[1] * chip.tiles_y, chip.tiles_x, chip.tiles_y,
                        chip["t_link"], chip["t_chip_hop"])
    rng = np.random.default_rng(3)
    n = 4 * len(ids)
    src, dst = rng.integers(0, len(ids), n), rng.integers(0, len(ids), n)
    dst[rng.random(n) < 0.3] = len(ids) // 2                     # hotspot
    tin = rng.random(n) * 2e-8
    uni = (tin, xy[src, 0], xy[src, 1], xy[dst, 0], xy[dst, 1])
    ref, hops = packetsim.simulate_mixed(net, uni, [], chip["t_hop"])
    t_inf, h_inf, occ = packetsim.simulate_finite(net, uni, chip["t_hop"], None)
    assert t_inf == ref and h_inf == hops
    assert packetsim.simulate_finite(net, uni, chip["t_hop"], occ + 1)[0] == ref
    t1, h1, o1 = packetsim.simulate_finite(net, uni, chip["t_hop"], 1)
    assert h1 == hops and o1 == 1 and t1 > ref


def test_validate_dvs_packet_check_one_mapping():
    """Check 5 (E5 workload A) on one mapping: model within 20% of the packet simulation on every sampled step."""
    import pytest
    from spikebill import validate
    if validate._sanafe() is None:
        pytest.skip("sanafe not installed")
    r = validate.dvs_packet_check(maps=("M2",), log=lambda *a: None)
    assert r["status"] == "pass" and len(r["rows"]) == 10
    assert max(abs(x["err"]) for x in r["rows"]) < 0.2


def test_validate_funcsim_check_short():
    """Check 6 (E3(a)) on one mapping, 200 steps: funcsim spikes per layer equal SANA-FE's; cost within criteria."""
    import pytest
    from spikebill import validate
    if validate._sanafe() is None:
        pytest.skip("sanafe not installed")
    r = validate.funcsim_check(maps=("M4",), steps=200, log=lambda *a: None)
    assert r["status"] == "pass" and r["rows"][0]["spikes_exact"]


def _toy_cut():
    from spikebill import NetBuilder, Simulator
    b = NetBuilder("toy_cut", steps=4)
    x = b.input("inp", (2048,), rate=0.1)
    s = b.dense(x, "S", 8192, rate=0.3)
    t = b.dense(s, "T", 8192, rate=0.05)
    b.dense(t, "out", 256, rate=0.1)
    return Simulator(b, chip="loihi2")


def test_split_fill_order():
    """(population, cores) segments: a population may occupy non-adjacent blocks; segments must cover its cores."""
    import numpy as np
    from spikebill import mapping, place
    sim = _toy_cut()
    m0 = mapping.map_graph(sim.graph, sim.chip)
    k = {n: p.n_cores for n, p in m0.pops.items()}
    order = [("T", 100), "inp", "S", ("T", k["T"] - 100), "out"]
    m = mapping.map_graph(sim.graph, sim.chip, order=order)
    assert m.n_cores == m0.n_cores and list(m.pops) == list(m0.pops)
    assert list(m.pops["T"].cores[:100]) == list(range(100)) and m.pops["T"].cores[100] == 100 + 1 + k["S"]
    assert (m.pops["T"].counts == m0.pops["T"].counts).all()
    with pytest.raises(ValueError):
        mapping.map_graph(sim.graph, sim.chip, order=[("T", 100), "inp", "S", "out"])
    sp, _, _, _ = sim._activity(None, None, None, "lockstep")
    px = place.Proxy(sim.graph, sim.chip, m0, sp)
    o = np.arange(len(px.names))
    assert px.evaluate(o) == px.evaluate(px.segments(o))
    assert px.segments([(0, 1), (0, 2), (1, 3)]) == [(0, 3), (1, 3)]


def test_place_split_search():
    """split=True adds anneal_split and split_refine; the result is never worse than the contiguous candidates
    (toy network-bound case: split_refine finds a better non-contiguous order)."""
    from spikebill import place
    sim = _toy_cut()
    sp, ac, _, _ = sim._activity(None, None, None, "lockstep")
    ps = place.search(sim.graph, sim.chip, sp, ac, split=True)
    rows = {r["candidate"]: r for r in ps["rows"]}
    assert {"anneal_split", "split_refine"} <= set(rows)
    contiguous = min(r["score"] for n, r in rows.items() if n not in ("anneal_split", "split_refine"))
    assert rows["split_refine"]["score"] < contiguous * 0.95
    assert any(not isinstance(e, str) for e in ps["order"])
    pw = place.search(sim.graph, sim.chip, sp, ac, split=True, workers=2, split_rounds=8)
    p1 = place.search(sim.graph, sim.chip, sp, ac, split=True, workers=1, split_rounds=8)
    assert pw["order"] == p1["order"] and pw["rows"] == p1["rows"]      # parallel rounds: same result
    rep = sim.run(balance=False, execution="lockstep", place="split")
    assert rep.results["value"]["step_s"] == pytest.approx(rows["split_refine"]["step"])
    with pytest.raises(ValueError):
        sim.run(balance=False, place="blocks")


def test_proxy_vectorized_matches_reference():
    """Proxy.evaluate (vectorized) equals the per-edge reference (expected hops over independent cores for dense
    projections, aligned quantiles for local ones; cut loads per mesh boundary), contiguous and split orders."""
    import numpy as np
    from spikebill import NetBuilder, Simulator, mapping, place
    b = NetBuilder("toy_px", steps=4)
    x = b.input("inp", (2, 32, 32), rate=0.1)
    c = b.conv2d(x, "c1", 16, 3, rate=0.2)
    c2 = b.conv2d(c, "c2", 16, 3, rate=0.1)
    b.dense(c2, "fc", 512, rate=0.1)
    sim = Simulator(b, chip="loihi2")
    sp, _, _, _ = sim._activity(None, None, None, "lockstep")
    px = place.Proxy(sim.graph, sim.chip, mapping.map_graph(sim.graph, sim.chip), sp)
    assert len(px._local[2]) and len(px._dense[2])

    def ref(order):
        segs, at, blocks = px.segments(order), 0, {}
        for i, n in segs:
            blocks.setdefault(i, []).append(np.arange(at, at + n)); at += n
        sl = {i: np.concatenate(v) for i, v in blocks.items()}
        W, H = px.W, px.H
        cx, cy, tot = np.zeros((2, W - 1)), np.zeros((2, H - 1)), 0.0
        kx, ky = np.arange(W - 1), np.arange(H - 1)
        for a, bb, w, local in px.edges:
            if local:
                qa = sl[a][np.minimum((np.arange(64) + 0.5) * px.k[a] / 64, px.k[a] - 1).astype(int)]
                qb = sl[bb][np.minimum((np.arange(64) + 0.5) * px.k[bb] / 64, px.k[bb] - 1).astype(int)]
                xa, ya, xb, yb = px.x[qa], px.y[qa], px.x[qb], px.y[qb]
                hops = np.mean(abs(xa - xb) + abs(ya - yb))
                ch = np.mean(abs(xa // px.tx - xb // px.tx) + abs(ya // px.ty - yb // px.ty))
                for u, v, cc, k in ((xa, xb, cx, kx), (ya, yb, cy, ky)):
                    cc[0] += w * np.mean((u[:, None] <= k) & (v[:, None] > k), 0)
                    cc[1] += w * np.mean((v[:, None] <= k) & (u[:, None] > k), 0)
            else:
                pa, pb = sl[a], sl[bb]
                dxm = abs(px.x[pa][:, None] - px.x[pb][None, :]); dym = abs(px.y[pa][:, None] - px.y[pb][None, :])
                hops = (dxm + dym).mean()
                ch = (abs(px.x[pa][:, None] // px.tx - px.x[pb][None, :] // px.tx)
                      + abs(px.y[pa][:, None] // px.ty - px.y[pb][None, :] // px.ty)).mean()
                for u, v, cc, k in ((px.x[pa], px.x[pb], cx, kx), (px.y[pa], px.y[pb], cy, ky)):
                    Fa, Fb = (u[:, None] <= k).mean(0), (v[:, None] <= k).mean(0)
                    cc[0] += w * Fa * (1 - Fb); cc[1] += w * Fb * (1 - Fa)
            tot += w * (hops * px.e_hop + ch * px.e_chip)
        tx_ = np.where((kx + 1) % px.tx == 0, px.t_chip, px.t_link) / H
        ty_ = np.where((ky + 1) % px.ty == 0, px.t_chip, px.t_link) / W
        return tot, max((cx * tx_).max() if cx.size else 0.0, (cy * ty_).max() if cy.size else 0.0)

    rng = np.random.default_rng(0)
    P = len(px.names)
    orders = [np.arange(P), rng.permutation(P),
              [(0, 1), (2, int(px.k[2])), (0, int(px.k[0]) - 1), (1, int(px.k[1])), (3, int(px.k[3]))]]
    for o in orders:
        got, want = px.evaluate(o), ref(o)
        assert got[0] == pytest.approx(want[0], rel=1e-9) and got[1] == pytest.approx(want[1], rel=1e-9)


def test_route_tree_matches_reference():
    """traffic._route_tree (sorted int loop) adds the same loads and counts as the original loop, 2D and 3D."""
    import numpy as np
    from spikebill import chip as chip_mod, traffic
    for name in ("loihi2", "loihi2_3d"):
        chip = chip_mod.load(name)
        Z = max(1, getattr(chip, "tiles_z", 1))
        W, H = 3 * chip.tiles_x, 2 * chip.tiles_y
        rng = np.random.default_rng(0)
        for _ in range(50):
            n = int(rng.integers(1, 40))
            dst = np.column_stack([rng.integers(0, W, n), rng.integers(0, H, n)] + ([rng.integers(0, Z, n)] if Z > 1 else []))
            src = [int(rng.integers(0, W)), int(rng.integers(0, H))] + ([int(rng.integers(0, Z))] if Z > 1 else [])
            mk = lambda: {**{k: np.zeros((Z, H + 1, W + 1)) for k in "EWNS"}, **{k: np.zeros((Z + 1, H + 1, W + 1)) for k in "UD"}}
            d1, d2 = mk(), mk()
            r1 = traffic._route_tree(d1, np.array(src), dst, 0.5, chip)
            r2 = traffic._route_tree_ref(d2, np.array(src), dst, 0.5, chip)
            assert r1 == r2
            assert all(np.array_equal(d1[k], d2[k]) for k in d1)


def test_sparse_sum_matches_bincount():
    """compiled._sparse_sum equals bincount + nonzero, bit for bit (same per-key summation order)."""
    import numpy as np
    from spikebill import compiled
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 10**6, 5000)
    idx[:50] = 7
    v = rng.random(5000) * (rng.random(5000) > 0.1)
    v[50:52] = [1.0, -1.0]
    idx[50:52] = 3                                          # sums to zero: dropped
    acc = np.bincount(idx, weights=v)
    nz = np.flatnonzero(acc)
    k, s = compiled._sparse_sum(idx, v)
    assert np.array_equal(k, nz) and np.array_equal(s, acc[nz])


@pytest.mark.parametrize("Z", [1, 3])
def test_tree_entries_match_route_tree(Z):
    """compiled._tree_entries (all trees of a projection at once) equals traffic._route_tree per source: summed
    difference-array entries, planar / chip-crossing / vertical links."""
    import numpy as np
    from spikebill import chip as chip_mod, compiled, traffic
    ch = chip_mod.load("loihi2").with_multicast("tree")
    W, H = 16, 12
    _, _, nd = compiled._diff_shape(W, H, Z)
    rng = np.random.default_rng(Z)
    n, m = 7, 200
    src = np.column_stack([rng.integers(0, W, n), rng.integers(0, H, n), rng.integers(0, Z, n)])
    ps = np.sort(rng.integers(0, n - 1, m))                     # source n-1 has no destinations
    dst = np.column_stack([rng.integers(0, W, m), rng.integers(0, H, m), rng.integers(0, Z, m)])
    dst[:20] = dst[20:40]                                       # duplicate destination rows
    ws = rng.random(n)
    i, v, h, c, vl = compiled._tree_entries(src, ws, ps, dst, W, H, Z, ch.tiles_x, ch.tiles_y)
    ref = np.zeros(nd)
    for k in range(n - 1):
        rec = compiled._Recorder(W, H, Z)
        out = traffic._route_tree(rec, src[k] if Z > 1 else src[k, :2], dst[ps == k] if Z > 1 else dst[ps == k, :2],
                                  ws[k], ch)
        ref += np.bincount(np.array(rec.idx, dtype=np.int64), np.array(rec.val), minlength=nd)
        assert (h[k], c[k], vl[k]) == out
    assert h[n - 1] == c[n - 1] == vl[n - 1] == 0
    assert np.allclose(np.bincount(i, v, minlength=nd), ref, rtol=0, atol=1e-12)


def test_bisection_order():
    """place.bisection_order: a permutation; on two disconnected chains it never interleaves them."""
    import numpy as np
    from spikebill import place

    class P:
        names = [f"p{i}" for i in range(8)]
        edges = [(0, 2, 5.0, False), (2, 4, 5.0, False), (4, 6, 5.0, False),
                 (1, 3, 5.0, False), (3, 5, 5.0, False), (5, 7, 5.0, False)]
    o = place.bisection_order(P(), seed=0)
    assert sorted(o.tolist()) == list(range(8))
    first = set(o[:4].tolist())
    assert first in ({0, 2, 4, 6}, {1, 3, 5, 7})
    assert np.array_equal(o, place.bisection_order(P(), seed=0))


def test_simulator_honours_chip_at_corner():
    """Simulator(chip=chip.at("high")) runs at the high end (it used to be reset to value by run()); Chip.pinned."""
    from spikebill import Simulator, chip as chip_mod, zoo
    g = zoo.load("sdt_v1_8_512")
    c = chip_mod.load("loihi2")
    assert c.pinned() is c
    p = c.at("high").pinned()
    assert p.which == "value" and p["e_synop"] == c.at("high")["e_synop"] == p.at("low")["e_synop"]
    kw = dict(multicast="tree", static="chip")
    hi = Simulator(g, chip=c.at("high"), **kw).run(balance=False).results["value"]["energy_J"]
    rng = Simulator(g, chip=c, **kw).run(balance=False, ranges=True).results
    assert abs(hi / rng["high"]["energy_J"] - 1) < 1e-9 and hi > 2 * rng["value"]["energy_J"]


def test_roofline_baseline():
    """cost.roofline (release gap 40): coproc=False is E17's definition; equals the full model when a core binds and
    never exceeds it in latency; Report.roofline uses the report's mapping."""
    import importlib.util
    from spikebill import Simulator, cost, zoo, mapping as mapping_mod
    g = zoo.load("sdt_v1_8_512")
    sim = Simulator(g, chip="loihi2", multicast="tree")
    sp, ac, _, cs = sim._activity(None, None, None, "pipelined")
    m = mapping_mod.map_graph(g, sim.chip)
    r = cost.roofline(g, sim.chip, m, sp, ac, cs, coproc=False)
    spec = importlib.util.spec_from_file_location("e17", os.path.join(os.path.dirname(__file__), "..", "experiments",
                                                                      "e17_review3", "run.py"))
    if os.path.exists(spec.origin):
        e17 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(e17)
        ref = e17.roofline(sim, m, sp, ac, cs)
        assert abs(ref["energy_J"] - r["energy_J"]) <= 1e-12 * ref["energy_J"]
        assert abs(ref["latency_s"] - r["latency_s"]) <= 1e-12 * ref["latency_s"]
    rc = cost.roofline(g, sim.chip, m, sp, ac, cs)
    assert rc["latency_s"] <= rc["full_latency_s"] * (1 + 1e-12)
    if rc["full_bound"] == "compute":
        assert abs(rc["latency_s"] / rc["full_latency_s"] - 1) < 1e-9
    rep = sim.run(balance=False)
    rr = rep.roofline()
    assert abs(rr["full_energy_J"] / rep.results["value"]["energy_J"] - 1) < 1e-9


def test_cli_spread(tmp_path):
    """spikebill MODEL --spread (.npz and .csv) equals Simulator.spread; HTML / CSV written; bad inputs are errors."""
    import numpy as np
    import pytest
    from spikebill import NetBuilder, Simulator, cli
    b = NetBuilder("toy_spread", steps=4)
    x = b.input("inp", (256,), rate=0.2)
    h = b.dense(x, "h", 1024, rate=0.05)
    b.dense(h, "out", 128, rate=0.1)
    gpath = str(tmp_path / "toy.json"); b.graph().save(gpath)
    R = np.array([[0.02, 0.1], [0.05, 0.2], [0.1, 0.05], [0.2, 0.3]])
    np.savez(tmp_path / "r.npz", rates=R, pops=np.array(["h", "out"]))
    sp = cli.main([gpath, "--spread", str(tmp_path / "r.npz"), "--no-balance",
                   "--spread-html", str(tmp_path / "s.html"), "--spread-csv", str(tmp_path / "s.csv")])
    ref = Simulator(b, chip="loihi2").spread([{"h": a, "out": c} for a, c in R], balance=False)
    assert sp["energy"]["mean"] == pytest.approx(ref["energy"]["mean"]) and sp["n"] == 4
    assert "<svg" in (tmp_path / "s.html").read_text() and len((tmp_path / "s.csv").read_text().splitlines()) == 5
    (tmp_path / "r.csv").write_text("h,out\n" + "\n".join(f"{a},{c}" for a, c in R) + "\n")
    sc = cli.main([gpath, "--spread", str(tmp_path / "r.csv"), "--no-balance"])
    assert sc["latency"]["p95"] == pytest.approx(ref["latency"]["p95"])
    assert cli.main([gpath, "--spread", str(tmp_path / "r.csv"), "--no-balance", "--map-rates", "95"])["n"] == 4
    (tmp_path / "bad.csv").write_text("nope\n0.1\n")
    with pytest.raises(ValueError):
        cli.main([gpath, "--spread", str(tmp_path / "bad.csv")])
    with pytest.raises(ValueError):
        cli.main([gpath, "--spread", str(tmp_path / "r.csv"), "--map-rates", "p95"])
    with pytest.raises(SystemExit):
        cli.main([gpath, "--spread-html", str(tmp_path / "x.html")])


def test_spread_paper_block(capsys, tmp_path):
    """spread_block: the spread's numbers (median, p5-p95, mean, at-mean gap) and the settings that change them;
    --paper-block with --spread prints it."""
    import numpy as np
    import pytest
    from spikebill import NetBuilder, Simulator, cli, spread_block
    b = NetBuilder("toy_spread_block", steps=4)
    x = b.input("inp", (256,), rate=0.2)
    h = b.dense(x, "h", 1024, rate=0.05)
    b.dense(h, "out", 128, rate=0.1)
    sim = Simulator(b, chip="loihi2", overrides={"e_syn": 1e-12})
    R = [{"h": a, "out": c} for a, c in ((0.02, 0.1), (0.05, 0.2), (0.1, 0.05), (0.2, 0.3))]
    sp = sim.spread(R, balance=False, map_rates=95)
    md = spread_block(sp, "markdown")
    e = sp["energy"]
    assert f"{e['p50'] * 1e3:.3g} mJ median" in md and f"{e['p5'] * 1e3:.3g}-{e['p95'] * 1e3:.3g} mJ (p5-p95)" in md
    assert "4 inputs" in md and "p95 rates" in md and "greedy" in md and "e_syn=1e-12" in md
    assert f"v{sim.chip.version}" in md and sim.chip.digest in md
    tex = spread_block(sp)
    assert tex.startswith("% generated by spikebill spread_block()") and "\\%" in tex
    assert spread_block(sim.spread(R, balance=False, percentiles=(10, 90)), "markdown").count("(p10-p90)") == 2
    with pytest.raises(ValueError):
        spread_block(sp, "html")
    gpath = str(tmp_path / "toy.json"); b.graph().save(gpath)
    np.savez(tmp_path / "r.npz", rates=np.array([[r["h"], r["out"]] for r in R]), pops=np.array(["h", "out"]))
    capsys.readouterr()
    cli.main([gpath, "--spread", str(tmp_path / "r.npz"), "--no-balance", "--paper-block", "markdown"])
    out = capsys.readouterr().out
    assert "- **Energy / inference:**" in out and "the mean rates" in out
