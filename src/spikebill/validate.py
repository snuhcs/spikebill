"""Reproducible validation: `python -m spikebill.validate` (or `spikebill-validate`).

Reruns the checks behind the validation claims in docs/cost_model.md section 10 and prints pass / fail against the
criteria that were fixed before the original experiments ran:

1. SANA-FE (E1b, Loihi 1, held-out configurations): this model vs the SANA-FE event-level simulator (v2.2.9,
   `pip install sanafe`), same chip parameters, mapping and activity. DVS gesture net with mappings M4 / M5 (1000 steps)
   and random networks on {2, 16, 32} cores x spike fraction {0.2, 0.7} (100 steps, seed 1). Criteria: dynamic energy
   per run within +-10% and total latency within +-20% for every configuration; per-step latency within +-20% on
   >= 75% of steps, pooled per workload. Skipped (reported as such) when `sanafe` is not installed.
2. Packet level (E5 workload B, Loihi 2, 2x2 and 4x4 chips): the model's network time vs the event-driven packet simulator
   `spikebill.packetsim` on synthetic traffic (uniform, bit-complement, hotspot; 4 / 16 / 64 messages per core; seeds 0-2).
   Criterion (E5): |error| > 20% on at most 50% of the steps of the group. E5 found the low-load steps (4 messages
   per core) over-estimated by 16-31% (path latency added in series); they are reported, not hidden.
3. chip_broadcast (E5b, Loihi 2, multi-chip): the model's network time vs `packetsim.simulate_step` on two sparse
   networks spanning 33 chips (fan-out 256 / 64, rates 0.01 / 0.05, seeds 1-2; the E5b W1 graphs). Criterion (E5b):
   |error| > 20% on at most 25% of the steps.
5. DVS at packet level (E5 workload A, Loihi 1, mappings M1-M5): spikes from `spikebill.funcsim` (1000 steps, 10 sampled
   steps per mapping), exact per-neuron destinations as packets vs the model's network time from the step's counts
   and rate-weighted flows. Criterion (E5): |error| > 20% on at most 25% of the steps. Needs `sanafe` (network and
   input data); skipped otherwise.
6. funcsim vs SANA-FE (E3(a), DVS M4 / M5, 1000 steps): spikes per layer identical, and the cost computed from
   funcsim's activity alone within the check-1 criteria (energy 10%, latency 20%). Needs `sanafe`.
4. A real public model: zoo:spikegpt_216m (13 chips, greedy
   mapping, one mean pipelined step), unicast and tree multicast, seeds 1-3, vs `packetsim.simulate_step`. Criterion
   as 3 (E5b's), chosen after a first look at seed 1 (errors -5% / -7%): post-hoc, stated as such.

The SANA-FE harness is a copy of the white paper's E1 / E1b harness (same construction, seeds and
activity extraction), so a pass here reproduces E1b's numbers.
"""
import argparse
import json
import random
import sys
import time

import numpy as np

from . import chip as chip_mod, cost, graph as graph_mod, mapping as mapping_mod, packetsim, traffic

DVS = {"M4": [1, 4, 12, 12, 2, 1], "M5": [1, 6, 24, 20, 6, 1]}
RAND = [(c, f) for c in (2, 16, 32) for f in (0.2, 0.7)]
TOL = dict(energy=0.10, latency=0.20, step=0.20, step_share=0.75, packet=0.20, packet_outside=0.50)


# ------------------------------------------------------------------------------------------------ SANA-FE side
def _sanafe():
    try:
        import sanafe
        return sanafe
    except ImportError:
        return None


def _arch(sanafe):
    return sanafe.load_loihi()


def _run_sanafe(sanafe, arch, snn, steps):
    chip = sanafe.SpikingChip(arch)
    chip.load(snn)
    t0 = time.time()
    r = chip.sim(steps, spike_trace=True, perf_trace=True, timing_model="detailed")
    return r, time.time() - t0


def _build_dvs(sanafe, arch, counts):
    import sanafe.layers
    import sanafe.tutorial
    snn = sanafe.Network()
    with sanafe.tutorial.get_dvs_data() as f:
        d = np.load(f)
        a = {k: d[k] for k in d.files}
    th = a["thresholds"]
    L = [sanafe.layers.Input2D(snn, 32, 32, threshold=th[0])]
    L.append(sanafe.layers.Conv2D(snn, L[-1], a["conv1"], stride_width=2, stride_height=2, threshold=th[1]))
    for i, k in enumerate(["conv2", "conv3", "conv4"]):
        L.append(sanafe.layers.Conv2D(snn, L[-1], a[k], threshold=th[i + 2]))
    L.append(sanafe.layers.Dense(snn, L[-1], 11, a["dense1"], threshold=th[5]))
    for n, b in zip(L[0], a["inputs"]):
        n.set_attributes(model_attributes={"bias": b})
    cores = arch.cores()
    core_of, used = [], 0
    for layer, k in zip(L, counts):                       # even split, as SANA-FE tutorial 5
        n = len(layer)
        pos = np.minimum(np.arange(n) // (n // k), k - 1)
        for nrn, p in zip(layer, pos):
            nrn.map_to_core(cores[used + p])
            nrn.set_attributes(log_spikes=True)
        core_of.append((used + pos).astype(np.int64))
        used += k
    return snn, L, core_of, int(np.count_nonzero(a["inputs"]))


def _build_random(sanafe, n_cores, per_core, msgs, syn_per_msg, frac, seed, arch):
    """SANA-FE's random benchmark (scripts/tcad2025/random_network.py), re-implemented as in E1."""
    rng = random.Random(seed)
    snn = sanafe.Network()
    N = n_cores * per_core
    pop = snn.create_neuron_group("pop", N, model_attributes={"force_update": True, "threshold": 64, "reset": 0},
                                  log_spikes=True)
    pre, post = [], []
    for n in range(N):
        if rng.random() >= frac:
            continue
        src = pop.neurons[n]
        src.set_attributes(model_attributes={"bias": 128})
        for c in rng.sample(range(n_cores), msgs):
            for d in rng.sample(range(per_core), syn_per_msg):
                dst = c * per_core + d
                src.connect_to_neuron(pop.neurons[dst], {"weight": 128})
                pre.append(n); post.append(dst)
    cores = arch.cores()
    core_of = np.arange(N) // per_core
    for n in range(N):
        pop.neurons[n].map_to_core(cores[core_of[n]])
    return snn, core_of, np.array(pre), np.array(post)


def _neuron_rates(r, group_sizes):
    cnt = {g: np.zeros(n) for g, n in group_sizes.items()}
    for fired in r["spike_trace"]:
        for a in fired:
            cnt[a.group_name][a.neuron_offset] += 1
    T = max(1, len(r["spike_trace"]))
    return {g: c / T for g, c in cnt.items()}


def _activity_from_trace(r, names, group_of, syn_into, n_bias):
    """Per-step spikes per population from the spike trace; 'updated' neurons split as in E1 (input layer: its biased
    neurons; the rest shared by synaptic events received)."""
    spikes_t, active_t = [], []
    upd = r["perf_trace"]["updated"]
    prev = {n: 0.0 for n in names}
    for t, fired in enumerate(r["spike_trace"]):
        s = {n: 0.0 for n in names}
        for a in fired:
            s[group_of[a.group_name]] += 1
        spikes_t.append(s)
        recv = {n: sum(prev[src] * f for src, f in syn_into.get(n, [])) for n in names}
        act = {n: 0.0 for n in names}
        base = min(upd[t], n_bias["count"])
        act[n_bias["pop"]] = base
        rest = max(0.0, upd[t] - base)
        tot = sum(recv.values())
        for n in names:
            if tot > 0:
                act[n] += rest * recv[n] / tot
        active_t.append(act)
        prev = s
    return spikes_t, active_t


def _compare(name, r, ours, wall):
    pt = r["perf_trace"]
    sf_e, sf_t = np.array(pt["total_energy"]), np.array(pt["sim_time"])
    our_e, our_t = sum(ours["energy"].values()), ours["time"]["step"]
    rel_t = (our_t - sf_t) / sf_t
    return dict(config=name, steps=len(sf_t), sanafe_wall_s=wall,
                energy_err=float(our_e.sum() / sf_e.sum() - 1), latency_err=float(our_t.sum() / sf_t.sum() - 1),
                step_within_20pct=float(np.mean(np.abs(rel_t) <= TOL["step"]))), rel_t


def _dvs_case(sanafe, chip, mname, counts, steps):
    arch = _arch(sanafe)
    snn, L, core_of, n_bias = _build_dvs(sanafe, arch, counts)
    r, wall = _run_sanafe(sanafe, arch, snn, steps)
    gname = [l.group.get_name() for l in L]
    rates = _neuron_rates(r, {gn: len(l) for gn, l in zip(gname, L)})
    rates_by_layer = [rates[gn] for gn in gname]
    sizes, names = [len(l) for l in L], [f"L{i}" for i in range(len(L))]
    projs, edges = [], []
    for i in range(len(L) - 1):
        pre, post = [], []
        for j, nrn in enumerate(L[i]):
            for e in nrn.edges_out:
                pre.append(j); post.append(e.post_neuron.neuron_offset)
        edges.append((np.array(pre), np.array(post)))
        projs.append(dict(src=names[i], dst=names[i + 1], kind="dense" if i == len(L) - 2 else "conv",
                          fanout=len(pre) / sizes[i]))
    g = graph_mod.from_dict(dict(name="dvs", populations=[dict(name=n, size=s) for n, s in zip(names, sizes)],
                                 projections=projs))
    for i, (proj, (pre, post)) in enumerate(zip(g.projections, edges)):
        cs, cd = core_of[i] - core_of[i].min(), core_of[i + 1] - core_of[i + 1].min()
        proj.flows = traffic.flows_from_edges(cs, cd, pre, post, np.bincount(cs), rates=rates_by_layer[i])
    placement = {}
    for n, co in zip(names, core_of):
        u, c = np.unique(co, return_counts=True)
        placement[n] = list(zip(u.tolist(), c.tolist()))
    m = mapping_mod.from_placement(g, chip, placement)
    for i, n in enumerate(names):
        m.pops[n].set_rates(core_of[i] - core_of[i].min(), rates_by_layer[i])
    syn_into = {}
    for proj in g.projections:
        syn_into.setdefault(proj.dst, []).append((proj.src, g.fanout(proj)))
    spikes_t, active_t = _activity_from_trace(r, names, dict(zip(gname, names)), syn_into,
                                              {"pop": names[0], "count": n_bias})
    ours = cost.estimate_trace(g, chip, m, spikes_t, active_t, static=False)
    return _compare(f"dvs_{mname}", r, ours, wall)


def _rand_case(sanafe, chip, n_cores, frac, steps, seed=1):
    arch = _arch(sanafe)
    snn, core_of, pre, post = _build_random(sanafe, n_cores, 256, min(4, n_cores), 4, frac,
                                            seed * 1000 + n_cores * 10 + int(frac * 10), arch)
    r, wall = _run_sanafe(sanafe, arch, snn, steps)
    N = n_cores * 256
    rates = _neuron_rates(r, {"pop": N})["pop"]
    spikes_t = [{"pop": float(len(f))} for f in r["spike_trace"]]
    active_t = [{"pop": float(u)} for u in r["perf_trace"]["updated"]]
    g = graph_mod.from_dict(dict(name="rand", populations=[dict(name="pop", size=N)],
                                 projections=[dict(src="pop", dst="pop", kind="sparse", fanout=max(len(pre), 1) / N)]))
    if len(pre):
        g.projections[0].flows = traffic.flows_from_edges(core_of, core_of, pre, post,
                                                          np.bincount(core_of, minlength=n_cores), rates=rates)
    m = mapping_mod.from_placement(g, chip, {"pop": [(c, 256) for c in range(n_cores)]})
    m.pops["pop"].set_rates(core_of, rates)
    ours = cost.estimate_trace(g, chip, m, spikes_t, active_t, static=False)
    return _compare(f"rand_c{n_cores}_p{int(frac * 100)}", r, ours, wall)


def sanafe_check(dvs=tuple(DVS), rand=tuple(RAND), dvs_steps=1000, rand_steps=100, log=print):
    """E1b held-out check. Returns dict(status="pass"|"fail"|"skipped", rows, criteria)."""
    sanafe = _sanafe()
    if sanafe is None:
        return dict(status="skipped", reason="sanafe not installed (pip install sanafe==2.2.9)")
    chip = chip_mod.load("loihi1")                       # SANA-FE's calibrated Loihi 1 costs are its value entries
    rows, per = [], {"A": [], "B": []}
    for mname in dvs:
        res, rel = _dvs_case(sanafe, chip, mname, DVS[mname], dvs_steps)
        rows.append(res); per["A"].extend(rel); log(_fmt(res))
    for n_cores, frac in rand:
        res, rel = _rand_case(sanafe, chip, n_cores, frac, rand_steps)
        rows.append(res); per["B"].extend(rel); log(_fmt(res))
    within = {w: float(np.mean(np.abs(v) <= TOL["step"])) for w, v in per.items() if len(v)}
    crit = dict(energy_all_within_10pct=all(abs(r["energy_err"]) <= TOL["energy"] for r in rows),
                latency_all_within_20pct=all(abs(r["latency_err"]) <= TOL["latency"] for r in rows),
                steps_within_20pct=within, steps_pass=all(v >= TOL["step_share"] for v in within.values()))
    ok = crit["energy_all_within_10pct"] and crit["latency_all_within_20pct"] and crit["steps_pass"]
    return dict(status="pass" if ok else "fail", rows=rows, criteria=crit, sanafe_version=getattr(sanafe, "__version__", "?"))


def _fmt(r):
    return (f"  {r['config']:<16} energy {r['energy_err']:+.2%}  latency {r['latency_err']:+.2%}  "
            f"steps within 20%: {r['step_within_20pct']:.2f}")


# -------------------------------------------------------------------------------------------- packet level (E5 B)
def _inject_times(src, t_out, rng):
    order = rng.permutation(len(src))
    s = src[order]
    idx = np.argsort(s, kind="stable")
    ss = s[idx]
    start = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    rank = np.arange(len(ss)) - np.repeat(start, np.diff(np.r_[start, len(ss)]))
    t = np.empty(len(src))
    t[order[idx]] = rank * t_out
    return t


def _t_agg_list(chip, mesh, src, sxy, dxy):
    """The model's network time for an explicit unicast message list."""
    W, H = mesh
    diff = traffic.new_diff(W, H)
    traffic._route_unicast(diff, sxy[:, 0], sxy[:, 1], dxy[:, 0], dxy[:, 1], np.ones(len(src)))
    li, le, _ = traffic.max_links(diff, W, H, chip)
    out = np.bincount(src).max() if len(src) else 0
    path = int(np.abs(dxy - sxy).sum(1).max()) if len(src) else 0
    terms = dict(inject=out * chip["t_spike_out"], link=li * chip["t_link"], chip_link=le * chip["t_chip_hop"])
    return max(terms.values()) + path * chip["t_hop"], terms


def _finite_one(args):
    """One E23 configuration: E5 workload B traffic (same RNG seeds as packet_check), simulated with link queues of
    each buffer depth in `bufs` (None = infinite) against the model's infinite-buffer network time."""
    side, pattern, load, seed, bufs = args
    chip = chip_mod.load("loihi2")
    n_chips = side * side
    n_cores = n_chips * chip.cores_per_chip
    ids = np.arange(n_cores)
    xy, _, _, board, _ = mapping_mod._geometry(chip, ids, n_chips)
    mesh = (board[0] * chip.tiles_x, board[1] * chip.tiles_y)
    net = packetsim.Net(mesh[0], mesh[1], chip.tiles_x, chip.tiles_y, chip["t_link"], chip["t_chip_hop"])
    tile_core = {}
    for c in ids:
        tile_core.setdefault((int(xy[c, 0]), int(xy[c, 1])), []).append(c)
    rng = np.random.default_rng(1000 * side + 100 * load + seed)
    n = load * n_cores
    src = rng.integers(0, n_cores, n)
    if pattern == "uniform":
        dst = rng.integers(0, n_cores, n)
    elif pattern == "complement":
        tx, ty = mesh[0] - 1 - xy[src, 0], mesh[1] - 1 - xy[src, 1]
        dst = np.array([tile_core[(int(a), int(b))][rng.integers(0, chip.cores_per_tile)] for a, b in zip(tx, ty)])
    else:
        dst = rng.integers(0, n_cores, n)
        dst[rng.random(n) < 0.25] = n_cores // 2
    sxy, dxy = xy[src], xy[dst]
    tin = _inject_times(src, chip["t_spike_out"], rng)
    t_agg, _ = _t_agg_list(chip, mesh, src, sxy, dxy)
    errs = {}
    for b in bufs:
        t, _, _ = packetsim.simulate_finite(net, (tin, sxy[:, 0], sxy[:, 1], dxy[:, 0], dxy[:, 1]), chip["t_hop"], b)
        errs["inf" if b is None else str(b)] = t_agg / float(t) - 1
    return dict(config=f"{side}x{side}_{pattern}_l{load}_s{seed}", err=errs)


def finite_buffer_check(bufs=(None, 16, 4), sides=(2, 4), loads=(4, 16, 64), seeds=(0, 1, 2), workers=8):
    """E23: the infinite-buffer network term vs packetsim.simulate_finite with link queues of B packets and credit
    back-pressure, on E5 workload B (uniform / complement / hotspot, 54 configurations). Returns the share of
    configurations within 20% and the worst error per B. No pass criterion here: E23's pre-registered criterion
    (B = 4 within 20% on >= 75%) was refuted; the model holds down to B = 16 (docs/cost_model.md)."""
    from . import sweep
    jobs = [(s, p, l, sd, tuple(bufs)) for s in sides for p in ("uniform", "complement", "hotspot")
            for l in loads for sd in seeds]
    rows = sweep.run(_finite_one, jobs, workers=workers)
    out = {}
    for k in rows[0]["err"]:
        e = np.array([r["err"][k] for r in rows])
        out[k] = dict(share_within_20pct=float(np.mean(np.abs(e) <= 0.20)), worst=float(e[np.argmax(np.abs(e))]),
                      n=len(rows))
    return dict(status="measured", summary=out, rows=rows)


def packet_check(chips_sides=(2, 4), loads=(4, 16, 64), seeds=(0, 1, 2), log=print):
    """E5 workload B (same seeds). Returns dict(status, rows, criteria)."""
    rows = []
    for side in chips_sides:
        rows += _packet_group(side, loads, seeds, log)
    outside = float(np.mean([abs(r["err"]) > TOL["packet"] for r in rows]))
    crit = dict(share_outside_20pct=outside, pass_=outside <= TOL["packet_outside"])
    return dict(status="pass" if crit["pass_"] else "fail", rows=rows, criteria=crit)


def _packet_group(chips_side, loads, seeds, log):
    chip = chip_mod.load("loihi2")
    n_chips = chips_side * chips_side
    n_cores = n_chips * chip.cores_per_chip
    ids = np.arange(n_cores)
    xy, _, _, board, _ = mapping_mod._geometry(chip, ids, n_chips)
    mesh = (board[0] * chip.tiles_x, board[1] * chip.tiles_y)
    net = packetsim.Net(mesh[0], mesh[1], chip.tiles_x, chip.tiles_y, chip["t_link"], chip["t_chip_hop"])
    tile_core = {}
    for c in ids:
        tile_core.setdefault((int(xy[c, 0]), int(xy[c, 1])), []).append(c)
    rows = []
    for pattern in ("uniform", "complement", "hotspot"):
        for load in loads:
            for seed in seeds:
                rng = np.random.default_rng(1000 * chips_side + 100 * load + seed)
                n = load * n_cores
                src = rng.integers(0, n_cores, n)
                if pattern == "uniform":
                    dst = rng.integers(0, n_cores, n)
                elif pattern == "complement":
                    tx, ty = mesh[0] - 1 - xy[src, 0], mesh[1] - 1 - xy[src, 1]
                    dst = np.array([tile_core[(int(a), int(b))][rng.integers(0, chip.cores_per_tile)]
                                    for a, b in zip(tx, ty)])
                else:
                    dst = rng.integers(0, n_cores, n)
                    dst[rng.random(n) < 0.25] = n_cores // 2
                sxy, dxy = xy[src], xy[dst]
                tin = _inject_times(src, chip["t_spike_out"], rng)
                t_sim, _ = packetsim.simulate_mixed(net, (tin, sxy[:, 0], sxy[:, 1], dxy[:, 0], dxy[:, 1]), [],
                                                    chip["t_hop"])
                t_agg, terms = _t_agg_list(chip, mesh, src, sxy, dxy)
                r = dict(config=f"{chips_side}x{chips_side}_{pattern}_l{load}_s{seed}", t_sim=t_sim, t_agg=t_agg,
                         err=t_agg / t_sim - 1, binding=max(terms, key=terms.get))
                rows.append(r)
                log(f"  {r['config']:<26} packet {t_sim * 1e6:8.3f} us  model {t_agg * 1e6:8.3f} us  "
                    f"err {r['err']:+.1%} ({r['binding']})")
    return rows


def broadcast_check(log=print):
    """E5b W1 multi-chip graphs in chip_broadcast mode. Returns dict(status, rows, criteria)."""
    chip = chip_mod.load("loihi2").with_multicast("chip_broadcast")
    rows = []
    for N, fan in ((65536, 256), (262144, 64)):
        g = graph_mod.from_dict(dict(name=f"sparse{N}", populations=[dict(name="a", size=N), dict(name="b", size=N)],
                                     projections=[dict(src="a", dst="b", kind="sparse", fanout=fan, weight_bits=1)]))
        m = mapping_mod.map_graph(g, chip)
        for rate in (0.01, 0.05):
            sp = {"a": N * rate}
            tr = traffic.step_traffic(g, m, chip, sp)
            t_agg = max(float(tr.core_out_msgs.max()) * chip["t_spike_out"], tr.max_link_intra * chip["t_link"],
                        tr.max_link_inter * chip["t_chip_hop"]) + tr.max_path_time
            for seed in (1, 2):
                t_sim = packetsim.simulate_step(g, m, chip, sp, seed=seed)["t_sim"]
                r = dict(config=f"{g.name}_r{rate}_s{seed}", chips=m.n_chips, t_sim=t_sim, t_agg=t_agg,
                         err=t_agg / t_sim - 1)
                rows.append(r)
                log(f"  {r['config']:<26} packet {t_sim * 1e6:8.3f} us  model {t_agg * 1e6:8.3f} us  err {r['err']:+.1%}")
    outside = float(np.mean([abs(r["err"]) > TOL["packet"] for r in rows]))
    crit = dict(share_outside_20pct=outside, pass_=outside <= 0.25)
    return dict(status="pass" if crit["pass_"] else "fail", rows=rows, criteria=crit)


DVS_ALL = {"M1": [1, 4, 16, 16, 4, 1], "M2": [1, 4, 8, 8, 1, 1], "M3": [1, 8, 32, 32, 8, 1], **DVS}
DVS_PACKET_STEPS, DVS_SAMPLE = 1000, list(range(99, 1000, 100))


def _dvs_edges(L):
    edges = []
    for i in range(len(L) - 1):
        pre, post, w = [], [], []
        for j, nrn in enumerate(L[i]):
            for e in nrn.edges_out:
                pre.append(j); post.append(e.post_neuron.neuron_offset); w.append(e.synapse_attributes["w"])
        edges.append((np.array(pre), np.array(post), np.array(w, float)))
    return edges


def _dvs_funcsim(a, L, edges, steps, record=None):
    from . import funcsim
    pops = [funcsim.FPop(f"L{i}", len(L[i]), float(a["thresholds"][i]),
                         bias=(a["inputs"].astype(float) if i == 0 else None)) for i in range(len(L))]
    projs = [funcsim.FProj(f"L{i}", f"L{i + 1}", *edges[i]) for i in range(len(L) - 1)]
    return funcsim.simulate(pops, projs, steps, "loihi", **({"record": record} if record else {}))


def _dvs_graph(L, core_of, edges, rates, chip):
    """Graph (populations L0.., exact rate-weighted flows) and the explicit placement of a DVS mapping."""
    names, sizes = [f"L{i}" for i in range(len(L))], [len(l) for l in L]
    g = graph_mod.from_dict(dict(name="dvs", populations=[dict(name=n, size=sz) for n, sz in zip(names, sizes)],
                                 projections=[dict(src=names[i], dst=names[i + 1],
                                                   kind="dense" if i == len(L) - 2 else "conv",
                                                   fanout=len(edges[i][0]) / sizes[i]) for i in range(len(L) - 1)]))
    for i, proj in enumerate(g.projections):
        cs, cd = core_of[i] - core_of[i].min(), core_of[i + 1] - core_of[i + 1].min()
        proj.flows = traffic.flows_from_edges(cs, cd, edges[i][0], edges[i][1], np.bincount(cs), rates=rates[i])
    placement = {}
    for n, co in zip(names, core_of):
        u, c = np.unique(co, return_counts=True)
        placement[n] = list(zip(u.tolist(), c.tolist()))
    m = mapping_mod.from_placement(g, chip, placement)
    for i, n in enumerate(names):
        m.pops[n].set_rates(core_of[i] - core_of[i].min(), rates[i])
    return g, m


def funcsim_check(maps=tuple(DVS), steps=1000, log=print):
    """Check 6 (E3(a)): funcsim in Loihi semantics vs SANA-FE on the DVS net: spikes per layer identical, and the cost
    from funcsim's activity (no SANA-FE trace) within the E1b criteria of SANA-FE (energy 10%, latency 20%)."""
    sanafe = _sanafe()
    if sanafe is None:
        return dict(status="skipped", reason="sanafe not installed", rows=[])
    import sanafe.tutorial
    with sanafe.tutorial.get_dvs_data() as f:
        d = np.load(f)
        a = {k: d[k] for k in d.files}
    chip = chip_mod.load("loihi1")
    rows, fr = [], None
    for mname in maps:
        arch = _arch(sanafe)
        snn, L, core_of, _ = _build_dvs(sanafe, arch, DVS_ALL[mname])
        r, wall = _run_sanafe(sanafe, arch, snn, steps)
        edges = _dvs_edges(L)
        if fr is None:
            fr = _dvs_funcsim(a, L, edges, steps)
        gname = [l.group.get_name() for l in L]
        sf = _neuron_rates(r, {gn: len(l) for gn, l in zip(gname, L)})
        sf_layer = [float(sf[gn].sum() * steps) for gn in gname]
        fs_layer = [float(fr.neuron_spikes[f"L{i}"].sum()) for i in range(len(L))]
        rates = [fr.neuron_spikes[f"L{i}"] / steps for i in range(len(L))]
        g, m = _dvs_graph(L, core_of, edges, rates, chip)
        ours = cost.estimate_trace(g, chip, m, fr.spikes, fr.active, static=False)
        res, _ = _compare(f"dvs_{mname}_funcsim", r, ours, wall)
        res.update(spikes_sanafe=sf_layer, spikes_funcsim=fs_layer, spikes_exact=bool(np.allclose(sf_layer, fs_layer)))
        rows.append(res)
        log(f"  {res['config']:<22} spikes {'identical' if res['spikes_exact'] else 'DIFFER'} ({sum(fs_layer):.0f}), "
            f"energy {res['energy_err']:+.1%}, latency {res['latency_err']:+.1%}")
    ok = all(x["spikes_exact"] and abs(x["energy_err"]) <= TOL["energy"] and abs(x["latency_err"]) <= TOL["latency"]
             for x in rows)
    return dict(status="pass" if ok else "fail", rows=rows, criteria=dict(pass_=ok))


def dvs_packet_check(maps=tuple(DVS_ALL), log=print):
    """Check 5 (E5 workload A). Returns dict(status, rows, criteria); status "skipped" without sanafe."""
    sanafe = _sanafe()
    if sanafe is None:
        return dict(status="skipped", reason="sanafe not installed", rows=[])
    import sanafe.tutorial
    from . import funcsim
    with sanafe.tutorial.get_dvs_data() as f:
        d = np.load(f)
        a = {k: d[k] for k in d.files}
    chip = chip_mod.load("loihi1")
    rows, fr = [], None
    for mname in maps:
        snn, L, core_of, _ = _build_dvs(sanafe, _arch(sanafe), DVS_ALL[mname])
        edges = _dvs_edges(L)
        if fr is None:                                    # same network and input for every mapping
            fr = _dvs_funcsim(a, L, edges, DVS_PACKET_STEPS, DVS_SAMPLE)
        names = [f"L{i}" for i in range(len(L))]
        rates = [fr.neuron_spikes[f"L{i}"] / DVS_PACKET_STEPS for i in range(len(L))]
        g, m = _dvs_graph(L, core_of, edges, rates, chip)
        W, H = m.mesh
        net = packetsim.Net(W, H, chip.tiles_x, chip.tiles_y, chip["t_link"], chip["t_chip_hop"])
        dst_sets = []
        for i, (pre, post, _) in enumerate(edges):
            cd = core_of[i + 1] - core_of[i + 1].min()
            k = int(cd.max()) + 1
            pairs = np.unique(pre.astype(np.int64) * k + cd[post])
            dst_sets.append((pairs // k, pairs % k))
        rng = np.random.default_rng(0)
        for t in DVS_SAMPLE:
            spiking = fr.meta["spiking"][t]
            src, dst = [], []
            for i in range(len(L) - 1):
                cs = core_of[i] - core_of[i].min()
                pn, pc = dst_sets[i]
                sel = np.isin(pn, spiking[f"L{i}"])
                src.append(m.pops[names[i]].cores[cs[pn[sel]]]); dst.append(m.pops[names[i + 1]].cores[pc[sel]])
            src, dst = np.concatenate(src), np.concatenate(dst)
            sxy, dxy = m.xy[src], m.xy[dst]
            tin = _inject_times(src, chip["t_spike_out"], rng)
            t_sim, _ = packetsim.simulate_mixed(net, (tin, sxy[:, 0], sxy[:, 1], dxy[:, 0], dxy[:, 1]), [], chip["t_hop"])
            tr = traffic.step_traffic(g, m, chip, {names[i]: float(fr.spikes[t][f"L{i}"]) for i in range(len(L))})
            t_agg = max(float(tr.core_out_msgs.max()) * chip["t_spike_out"], tr.max_link_intra * chip["t_link"],
                        tr.max_link_inter * chip["t_chip_hop"]) + tr.max_path_hops * chip["t_hop"]
            r = dict(config=f"dvs_{mname}_t{t + 1}", t_sim=t_sim, t_agg=t_agg, err=t_agg / t_sim - 1 if t_sim else 0.0)
            rows.append(r)
            log(f"  {r['config']:<26} packet {t_sim * 1e6:8.3f} us  model {t_agg * 1e6:8.3f} us  err {r['err']:+.1%}")
    outside = float(np.mean([abs(r["err"]) > TOL["packet"] for r in rows]))
    crit = dict(share_outside_20pct=outside, pass_=outside <= 0.25)
    return dict(status="pass" if crit["pass_"] else "fail", rows=rows, criteria=crit)


def model_check(name="spikegpt_216m", modes=("none", "tree"), seeds=(1, 2, 3), log=print):
    """Check 4: a zoo model's mean step, network term vs packet-level simulation. Returns dict(status, rows, criteria)."""
    from . import zoo
    from .api import Simulator
    rows = []
    for mc in modes:
        sim = Simulator(zoo.load(name), chip="loihi2", multicast=mc)
        chip = sim.chip.at("value")
        sp, _, _, _ = sim._activity(None, None, None, "pipelined")
        m = mapping_mod.map_graph(sim.graph, chip)
        tr = traffic.step_traffic(sim.graph, m, chip, sp)
        t_agg = max(float(tr.core_out_msgs.max()) * chip["t_spike_out"], tr.max_link_intra * chip["t_link"],
                    tr.max_link_inter * chip["t_chip_hop"]) + tr.max_path_time
        for seed in seeds:
            t_sim = packetsim.simulate_step(sim.graph, m, chip, sp, seed=seed)["t_sim"]
            r = dict(config=f"{name}_{mc}_s{seed}", chips=m.n_chips, t_sim=t_sim, t_agg=t_agg, err=t_agg / t_sim - 1)
            rows.append(r)
            log(f"  {r['config']:<26} packet {t_sim * 1e6:9.3f} us  model {t_agg * 1e6:9.3f} us  err {r['err']:+.1%}")
    outside = float(np.mean([abs(r["err"]) > TOL["packet"] for r in rows]))
    crit = dict(share_outside_20pct=outside, pass_=outside <= 0.25)
    return dict(status="pass" if crit["pass_"] else "fail", rows=rows, criteria=crit)


def main(argv=None):
    """Command line: run the checks, print pass/fail per criterion; exit status 1 if any check fails.
    Options: --quick, --skip-sanafe, --skip-packet, --json PATH."""
    p = argparse.ArgumentParser(prog="python -m spikebill.validate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="random networks only (no DVS net): seconds instead of ~1 min")
    p.add_argument("--skip-sanafe", action="store_true")
    p.add_argument("--skip-packet", action="store_true")
    p.add_argument("--json", help="write the results to this file")
    a = p.parse_args(argv)
    out = {}
    if not a.skip_sanafe:
        print("1. vs SANA-FE event-level simulator (E1b held-out, Loihi 1)")
        out["sanafe"] = sanafe_check(dvs=() if a.quick else tuple(DVS))
        print(f"   -> {out['sanafe']['status'].upper()}" + (f" ({out['sanafe'].get('reason')})"
                                                           if out["sanafe"]["status"] == "skipped" else ""))
    if not a.skip_packet:
        print("2. network term vs packet-level simulation (E5 workload B, Loihi 2, 2x2 and 4x4 chips)")
        out["packet"] = packet_check()
        print(f"   -> {out['packet']['status'].upper()} "
              f"({out['packet']['criteria']['share_outside_20pct']:.0%} of steps outside +-20%; limit 50%)")
        if not a.quick:
            print("3. chip_broadcast network term vs packet-level simulation (E5b, Loihi 2, 33 chips)")
            out["broadcast"] = broadcast_check()
            print(f"   -> {out['broadcast']['status'].upper()} "
                  f"({out['broadcast']['criteria']['share_outside_20pct']:.0%} of steps outside +-20%; limit 25%)")
            print("5. DVS packet level (E5 workload A, Loihi 1, mappings M1-M5)")
            out["dvs_packet"] = dvs_packet_check()
            print(f"   -> {out['dvs_packet']['status'].upper()}" + (
                f" ({out['dvs_packet']['criteria']['share_outside_20pct']:.0%} of steps outside +-20%; limit 25%)"
                if out["dvs_packet"]["status"] != "skipped" else f" ({out['dvs_packet'].get('reason')})"))
            print("6. funcsim vs SANA-FE (E3(a): spikes identical, cost from funcsim activity; DVS M4 / M5)")
            out["funcsim"] = funcsim_check()
            print(f"   -> {out['funcsim']['status'].upper()}")
            print("4. a real model: zoo:spikegpt_216m mean step vs packet-level simulation (Loihi 2, 13 chips)")
            out["model"] = model_check()
            print(f"   -> {out['model']['status'].upper()} "
                  f"({out['model']['criteria']['share_outside_20pct']:.0%} of steps outside +-20%; limit 25%)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=float)
    failed = [k for k, v in out.items() if v["status"] == "fail"]
    print("FAILED: " + ", ".join(failed) if failed else "all run checks passed")
    return 1 if failed else 0


def entry():
    """Console-script entry point (`spikebill-validate`)."""
    sys.exit(main())


if __name__ == "__main__":
    entry()
