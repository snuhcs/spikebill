"""Reproduce the white paper's cost numbers that this package can compute, in one command.

    python examples/reproduce_paper.py              # about 3 min with 8 worker processes
    python examples/reproduce_paper.py --validate   # also the validation numbers (Sec. 4; needs sanafe, ~7 min)
    python examples/reproduce_paper.py --only table # claims whose id starts with "table"

Each number is compared with `examples/expected/paper_numbers.json` (the value as printed in the paper and the paper's
experiment ID); it matches when it rounds to the printed digits (half a unit of the last printed digit). Prints one line per number, writes
`examples/out/reproduce.json`, exits with status 1 if any number is off.

Covered: the three spiking transformers (Table "transformers": balanced mapping per parameter end, chips, energy,
rows (c) 45 nm physical events, (c') + SRAM read, (d) synaptic and (e) dynamic energy,
latency, ratio to the papers' own 45 nm estimates; E10), the first-order baseline (E15), the low Loihi 2 calibration
scale s = 0.33 (E13), SpikeGPT-216M (E19(a)), greedy vs balanced mapping and the energy split (Sec. 5 text), the
joint-uncertainty percentiles (E17), the favourable floor (E16(b)) and the layer variants (attention as AND-gate
neurons, RepConv unfused; zoo variants, E10 baselines), Meta-SpikeFormer 55M (E19(b); zoo variant), the placement
search on SDT V2 and the 55M model (E20(e), E20 calibrated-proxy addendum); with --validate, the validation numbers against SANA-FE and the
packet-level simulator (E1b, E3(a), E5b; checked as bounds, see `validation`) and the finite-buffer
network check (E23). Not reproducible from this repository: the case studies on models that are not released here
(also in the pooled E5 packet-level share), wall-clock timings (machine-dependent,
E9), the literature counts (E0) and row (b) of Table "transformers" (each paper's own formula applied to its own
rate table, a reconstruction of the claim rather than a simulator output).
"""
import argparse
import json
import os
import sys
import time

from spikebill import Simulator, balance as balance_mod, chip as chip_mod, cost, place, zoo
from spikebill.api import depth

HERE = os.path.dirname(os.path.abspath(__file__))

TRANSFORMERS = ("sdt_v1_8_512", "spikingformer_8_512", "sdt_v2_31m")
SETTINGS = zoo.REFERENCE_SETTINGS                     # loihi2, tree multicast, chip static, pipelined, balanced (edp)
CORE_E = ("e_synop", "e_neuron_update", "e_neuron_active", "e_spike_gen", "e_spike_out", "e_hop")  # E6-scaled
EXPECTED = os.path.join(HERE, "expected", "paper_numbers.json")


def chip(which="value"):
    return chip_mod.load(SETTINGS["chip"]).at(which)


def run(g, which="value", overrides=None, ranges=False, workers=8):
    sim = Simulator(g, chip=chip(which), multicast=SETTINGS["multicast"], static=SETTINGS["static"],
                    overrides=overrides)
    return sim.run(execution=SETTINGS["execution"], balance=SETTINGS["balance"], objective=SETTINGS["objective"],
                   ranges=ranges, workers=workers)


E_AC, E_MAC, SRAM_32KB = 0.9e-12, 4.6e-12, 20e-12   # 45 nm add / MAC; 64-bit read from a 32 KB SRAM (Horowitz 2014)
ATTN_INTERNAL = ("_A", "_S", "_O", "_qk", "_g")      # zoo population names of attention internals (score, AND, ...)


def physical_45nm(g, name):
    """Row (c): physical synaptic events x E_AC + the first layer's dense MACs x E_MAC, per inference (E4, E10).
    SDT V1 / Spikingformer: the layers the papers count (attention internals, max-pool OR neurons and residual
    copies excluded, as in their formula); SDT V2: every synaptic event of the graph."""
    T, P = g.steps_per_inference, g.populations
    first = [p for p in g.projections if p.src == "img"][0]
    mac = P["img"].size * g.fanout(first) * T
    if name == "sdt_v2_31m":
        syn = sum(v for (src, _), v in g.syn_events_per_step(None).items() if src != "img") * T
    else:
        syn = sum(P[p.src].rate * P[p.src].size * g.fanout(p) * T for p in g.projections
                  if p.src != "img" and p.kind != "one_to_one" and "_or" not in p.dst
                  and not any(s in p.src or s in p.dst for s in ATTN_INTERNAL))
    return syn * E_AC + mac * E_MAC


def transformers(workers, models=TRANSFORMERS):
    """Table "transformers" (E10) and the first-order baseline (E15): the mapping is balanced again at each
    parameter end (the zoo's shipped reference keeps the value mapping for low / high, so its range differs)."""
    out, lat = {}, []
    for name in (n for n in TRANSFORMERS if n in models):
        g = zoo.load(name)
        claim = g.meta["claim_mJ"]
        full = {}
        for which in ("value", "low", "high"):
            v = run(g, which, workers=workers).results["value"]
            full[which] = v
            out[f"table.{name}.{which}.energy_mJ"] = v["energy_J"] * 1e3
            out[f"table.{name}.{which}.ratio"] = v["energy_J"] * 1e3 / claim
        e, sp = full["value"]["energy_J"] * 1e3, full["value"]["split"]
        c = physical_45nm(g, name) * 1e3
        events = e * sp["synaptic"] * 1e-3 / chip_mod.load(SETTINGS["chip"])["e_synop"]   # chip synaptic events
        out[f"table.{name}.row_c_physical_45nm_mJ"] = c
        out[f"table.{name}.row_c2_plus_sram_mJ"] = c + events * SRAM_32KB * 1e3       # (c') + one 32 KB read each
        out[f"table.{name}.row_d_synaptic_mJ"] = e * sp["synaptic"]                 # (d) events x chip synaptic
        out[f"table.{name}.row_e_dynamic_mJ"] = e * (1 - sp["static"])             # (e) + neuron, network, coproc
        out[f"table.{name}.chips"] = full["value"]["chips"]
        out[f"table.{name}.latency_ms"] = full["value"]["latency_s"] * 1e3
        c = chip_mod.load(SETTINGS["chip"]).with_multicast(SETTINGS["multicast"]) \
            .with_static_accounting(SETTINGS["static"]).with_coproc_defaults()
        fo = cost.first_order(g, c, execution=SETTINGS["execution"])
        out[f"first_order.{name}.chips"] = fo["chips"]
        out[f"first_order.{name}.ratio"] = fo["energy_J"] * 1e3 / claim
        out[f"first_order.{name}.full_over_first"] = full["value"]["energy_J"] / fo["energy_J"]
        lat.append(fo["latency_s"] / full["value"]["latency_s"])
    if len(lat) == len(TRANSFORMERS):                    # first-order latency / mapped latency, over the three
        out["first_order.all.latency_x_min"], out["first_order.all.latency_x_max"] = min(lat), max(lat)
    return out


def calibration_low(workers, models=TRANSFORMERS):
    """E13: per-event energies at the low end of the Loihi 2 calibration scale (s = 0.33), everything else at value."""
    lo = chip_mod.load(SETTINGS["chip"]).params
    out = {}
    for name in (n for n in TRANSFORMERS if n in models):
        g = zoo.load(name)
        v = run(g, overrides={k: lo[k]["low"] for k in CORE_E}, workers=workers).results["value"]
        out[f"s033.{name}.energy_mJ"] = v["energy_J"] * 1e3
        out[f"s033.{name}.ratio"] = v["energy_J"] * 1e3 / g.meta["claim_mJ"]
    return out


def spikegpt(workers, models=None):
    """E19(a): SpikeGPT-216M per token, tree multicast, value mapping with parameter ranges."""
    g = zoo.load("spikegpt_216m")
    tokens = g.steps_per_inference
    r = run(g, ranges=True, workers=workers).results
    out = {"spikegpt.chips": r["value"]["chips"], "spikegpt.latency_ms_per_token": r["value"]["latency_s"] * 1e3 / tokens}
    for which in ("value", "low", "high"):
        out[f"spikegpt.{which}.energy_mJ_per_token"] = r[which]["energy_J"] * 1e3 / tokens
    return out


def validation(workers=None, models=None):
    """Sec. 4 (E1b, E3(a), E5b) through spikebill.validate (needs `pip install sanafe`; about 7 min) and E23 (2 min).
    The paper's ranges are maxima over runs and mappings (SANA-FE's latency varies by about 0.6 points between runs; E3(a) used five
    DVS mappings, this check two), so these numbers are checked as bounds: |error| at most the printed value."""
    import tempfile
    from spikebill import validate
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "validate.json")
        validate.main(["--json", path])
        with open(path) as f:
            v = json.load(f)
    out = {}
    if v.get("sanafe", {}).get("status") in ("pass", "fail"):
        r = v["sanafe"]["rows"]
        out["validation.e1b.configs"] = len(r)
        out["validation.e1b.energy_max_abs_pct"] = max(abs(x["energy_err"]) for x in r) * 100
        out["validation.e1b.latency_max_abs_pct"] = max(abs(x["latency_err"]) for x in r) * 100
    if v.get("funcsim", {}).get("status") in ("pass", "fail"):
        r = v["funcsim"]["rows"]
        out["validation.e3a.spikes"] = min(sum(x["spikes_funcsim"]) for x in r) if all(
            x["spikes_exact"] for x in r) else -1
        out["validation.e3a.energy_max_abs_pct"] = max(abs(x["energy_err"]) for x in r) * 100
        out["validation.e3a.latency_max_abs_pct"] = max(abs(x["latency_err"]) for x in r) * 100
    if "broadcast" in v:
        out["validation.e5b.outside_20pct_pct"] = v["broadcast"]["criteria"]["share_outside_20pct"] * 100
    fb = validate.finite_buffer_check(workers=workers or 8)["summary"]        # E23, about 2 min
    for b, r in fb.items():
        out[f"validation.e23.b{b}.within_20pct_pct"] = r["share_within_20pct"] * 100
        out[f"validation.e23.b{b}.worst_pct"] = r["worst"] * 100
    return out


def mapping_and_split(workers, models=TRANSFORMERS):
    """Sec. 5 text: greedy (unbalanced) vs balanced mapping (E10 baselines (a)) and the energy split at value (E15)."""
    out, step_x, energy_x, syn, static = {}, [], [], [], []
    for name in (n for n in TRANSFORMERS if n in models):
        g = zoo.load(name)
        sim = Simulator(g, chip=chip(), multicast=SETTINGS["multicast"], static=SETTINGS["static"])
        kw = dict(execution=SETTINGS["execution"], objective=SETTINGS["objective"], workers=workers)
        gr, bal = sim.run(balance=False, **kw).results["value"], sim.run(balance=True, **kw).results["value"]
        out[f"mapping.{name}.greedy_ratio"] = gr["energy_J"] * 1e3 / g.meta["claim_mJ"]
        step_x.append(gr["latency_s"] / bal["latency_s"])             # same chip steps: latency ratio = step ratio
        energy_x.append(gr["energy_J"] / bal["energy_J"])
        syn.append(bal["split"]["synaptic"] * 100)
        static.append(bal["split"]["static"] * 100)
    if len(step_x) == len(TRANSFORMERS):
        out.update({"mapping.all.greedy_step_x_max": max(step_x), "mapping.all.greedy_energy_x_max": max(energy_x),
                    "mapping.all.synaptic_pct_min": min(syn), "mapping.all.synaptic_pct_max": max(syn),
                    "mapping.all.static_pct_min": min(static), "mapping.all.static_pct_max": max(static)})
    return out


def monte_carlo(workers, models=TRANSFORMERS):
    """E17: joint uncertainty, 512 log-uniform draws (seed 0) on the value mapping; the six E6-scaled per-event
    energies share one quantile (one calibration scale), every other ranged parameter drawn independently."""
    out = {}
    for name in (n for n in TRANSFORMERS if n in models):
        g = zoo.load(name)
        u = Simulator(g, chip="loihi2", multicast="tree").uncertainty(n=512, prior="loguniform", seed=0,
                                                                       workers=workers, groups=[list(CORE_E)])
        for q, v in u["energy"].items():
            out[f"mc.{name}.{q}_ratio"] = v * 1e3 / g.meta["claim_mJ"]
    return out


def floor(workers=None, models=TRANSFORMERS):
    """E16(b): a floor no real mapping reaches: the first-order chip count (every weight stored once), every
    parameter at its low end, static energy over the T steps of back-to-back pipelined inferences only."""
    out = {}
    for name in (n for n in TRANSFORMERS if n in models):
        g = zoo.load(name)
        fo = cost.first_order(g, chip("low"), rates={k: p.rate for k, p in g.populations.items()})
        T, D = g.steps_per_inference, depth(g)
        fav = fo["energy_J"] - fo["energy_J"] * fo["split"]["static"] * (1 - T / (T + D))
        out[f"floor.{name}.ratio"] = fav * 1e3 / g.meta["claim_mJ"]
    return out


def layer_variants(workers, models=TRANSFORMERS):
    """Sec. 5 / App. transformers baselines (E10 v5 baselines (b), (c)): attention as AND-gate neurons instead of a
    co-processor (zoo variant "andgate"; balanced, value) and SDT V2's RepConv as its trained unfused sequence
    (variant "repconv_literal"). Spikingformer's AND-gate graph (3.35e8 neurons) takes about 30 s."""
    out = {}

    def value(g):
        return run(g, workers=workers).results["value"]
    for name in (n for n in TRANSFORMERS if n in models):
        a = zoo.load(name, attention="andgate")
        ra = value(a)
        out[f"andgate.{name}.ratio"] = ra["energy_J"] * 1e3 / a.meta["claim_mJ"]
        if name == "spikingformer_8_512":
            rc = value(zoo.load(name))
            out[f"andgate.{name}.chips"] = ra["chips"]
            out[f"andgate.{name}.neurons_1e8"] = sum(p.size for p in a.populations.values()) / 1e8
            out[f"andgate.{name}.energy_x"] = ra["energy_J"] / rc["energy_J"]
            out[f"andgate.{name}.chips_x"] = ra["chips"] / rc["chips"]
    if "sdt_v2_31m" in models:
        lit = zoo.load("sdt_v2_31m", variant="repconv_literal")
        r = value(lit)
        out["repconv.sdt_v2_31m.literal_mJ"] = r["energy_J"] * 1e3
        out["repconv.sdt_v2_31m.literal_ratio"] = r["energy_J"] * 1e3 / lit.meta["claim_mJ"]
    return out


def large_55m(workers, models=None):
    """E19(b): Meta-SpikeFormer 55M (zoo variant "sdt_v2_31m.55m": SDT V2 widths x 4/3, the 31M rates as an
    assumption), reference settings, balanced; ranges on the value mapping as in E19; greedy mapping, the first-order
    estimate, the roofline baseline and every sub-unit firing rate x 0.5 / x 2 (E19 scaled the per-layer rates before building the graph;
    scaling the population rates agrees within 0.02%)."""
    g = zoo.load("sdt_v2_31m", variant="55m")
    sim = Simulator(g, chip=chip(), multicast=SETTINGS["multicast"], static=SETTINGS["static"])
    kw = dict(execution=SETTINGS["execution"], objective=SETTINGS["objective"], workers=workers)
    rep = sim.run(balance=True, ranges=True, **kw)
    v = rep.results["value"]
    out = {"large.sdt_v2_55m.chips": v["chips"], "large.sdt_v2_55m.energy_J": v["energy_J"],
           "large.sdt_v2_55m.energy_low_J": rep.results["low"]["energy_J"],
           "large.sdt_v2_55m.energy_high_J": rep.results["high"]["energy_J"],
           "large.sdt_v2_55m.latency_ms": v["latency_s"] * 1e3,
           "large.sdt_v2_55m.network_us": v["time"]["network"] * 1e6,
           "large.sdt_v2_55m.compute_us": v["time"]["compute"] * 1e6}
    fo = cost.first_order(g, chip())
    out.update({"large.sdt_v2_55m.first_order_chips": fo["chips"], "large.sdt_v2_55m.first_order_J": fo["energy_J"],
                "large.sdt_v2_55m.full_over_first": v["energy_J"] / fo["energy_J"]})
    out["large.sdt_v2_55m.chips_over_first"] = v["chips"] / fo["chips"]
    rf = rep.roofline()             # per-core / per-link roofline on the same mapping (E19(b) addendum)
    out.update({"large.sdt_v2_55m.roofline_J": rf["energy_J"], "large.sdt_v2_55m.roofline_ms": rf["latency_s"] * 1e3,
                "large.sdt_v2_55m.roofline_latency_x": rf["full_latency_s"] / rf["latency_s"],
                "large.sdt_v2_55m.roofline_energy_low_pct": 100 * (1 - rf["energy_J"] / rf["full_energy_J"])})
    gr = sim.run(balance=False, **kw).results["value"]
    out.update({"large.sdt_v2_55m.greedy_J": gr["energy_J"], "large.sdt_v2_55m.greedy_ms": gr["latency_s"] * 1e3})
    for k, tag in ((0.5, "half"), (2.0, "double")):
        rates = {n: min(1.0, p.rate * k) if p.rate < 1.0 else p.rate for n, p in g.populations.items()}
        out[f"large.sdt_v2_55m.rates_{tag}_J"] = sim.run(rates=rates, balance=True, **kw).results["value"]["energy_J"]
    return out


def placement(workers, models=None):
    """E20(e) and the E20 calibrated-proxy addendum: a searched placement (`place.search`, seed 0, 40 iterations per
    population) against the graph-order fill on the same partition. SDT V2 31M: balanced partition, non-contiguous
    blocks allowed (split=True), as for the zoo references; Meta-SpikeFormer 55M: contiguous blocks, greedy and
    balanced partitions; its energy / latency per image after the search, and the factor by which the E19(b) numbers
    overstate them."""
    out = {}

    def search(g, balanced, **kw):
        sim = Simulator(g, chip=SETTINGS["chip"], multicast=SETTINGS["multicast"], static=SETTINGS["static"])
        sp, ac, _, steps = sim._activity(None, None, None, SETTINGS["execution"])
        caps = (balance_mod.balance(g, sim.chip, sp, ac, objective=SETTINGS["objective"], workers=workers)[2]
                if balanced else {})
        rows = {r["candidate"]: r for r in place.search(g, sim.chip, sp, ac, caps=caps,
                                                         iters=40 * len(g.populations), **kw)["rows"]}
        g0, best = rows["graph_order"], min(rows.values(), key=lambda r: r["score"])
        return g0, best, steps

    if "sdt_v2_31m" in models:
        g0, b, _ = search(zoo.load("sdt_v2_31m"), True, split=True, objective=SETTINGS["objective"],
                          workers=min(4, workers))
        out.update({"placement.sdt_v2_31m.edp_pct": (b["score"] / g0["score"] - 1) * 100,
                    "placement.sdt_v2_31m.step_pct": (b["step"] / g0["step"] - 1) * 100,
                    "placement.sdt_v2_31m.energy_pct": (b["energy"] / g0["energy"] - 1) * 100})
    if "sdt_v2_55m" in models:
        g = zoo.load("sdt_v2_31m", variant="55m")
        g0, b, _ = search(g, False)
        out["placement.sdt_v2_55m.greedy_edp_pct"] = (b["score"] / g0["score"] - 1) * 100
        g0, b, steps = search(g, True)
        out.update({"placement.sdt_v2_55m.edp_pct": (b["score"] / g0["score"] - 1) * 100,
                    "placement.sdt_v2_55m.energy_J": b["energy"] * steps,
                    "placement.sdt_v2_55m.latency_ms": b["step"] * steps * 1e3,
                    "placement.sdt_v2_55m.energy_x": g0["energy"] / b["energy"],
                    "placement.sdt_v2_55m.latency_x": g0["step"] / b["step"]})
    return out


GROUPS = dict(placement=placement, large=large_55m, table=transformers, first_order=transformers, s033=calibration_low, spikegpt=spikegpt,
              validation=validation, mapping=mapping_and_split, mc=monte_carlo, floor=floor,
              andgate=layer_variants, repconv=layer_variants)


def tolerance(printed):
    """Half a unit of the last printed digit: "352.7" -> 0.05, "164" -> 0.5."""
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    return 0.5 * 10 ** -decimals * (1 + 1e-9)


def compare(got, expected):
    rows, bad = [], 0
    for e in expected:
        v = got.get(e["id"])
        if v is None:
            continue
        if e.get("kind") == "max":                      # a bound: |error| no larger than the printed maximum
            ok = v <= float(e["paper"]) + tolerance(e["paper"])
        else:
            ok = abs(v - float(e["paper"])) <= tolerance(e["paper"])
        bad += not ok
        rows.append(dict(e, got=v, ok=ok))
    return rows, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--only", default="", help="run only the claims whose id starts with this prefix")
    ap.add_argument("--validate", action="store_true",
                    help="also the validation numbers (Sec. 4; runs spikebill-validate, needs sanafe, ~7 min)")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "reproduce.json"))
    a = ap.parse_args(argv)
    with open(EXPECTED) as f:
        expected = [e for e in json.load(f)["numbers"] if e["id"].startswith(a.only)]
    if not (a.validate or a.only.startswith("validation")):
        expected = [e for e in expected if not e["id"].startswith("validation")]
    need = {e["id"].split(".")[0] for e in expected}
    models = {e["id"].split(".")[1] for e in expected}
    got, t0 = {}, time.time()
    for fn in dict.fromkeys(GROUPS[g] for g in need if g in GROUPS):
        got.update(fn(a.workers, models))
    rows, bad = compare(got, expected)
    for r in rows:
        print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['id']:<44} {'max' if r.get('kind') == 'max' else 'paper'} {r['paper']:>7}  got {r['got']:>10.4f}  "
              f"({r['eid']}; {r['where']})")
    print(f"{len(rows) - bad} / {len(rows)} numbers match the paper ({time.time() - t0:.0f} s)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(chip_version=chip_mod.load(SETTINGS["chip"]).version, rows=rows), f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
