"""Command line: chip-level cost of an SNN (same model as `spikebill.Simulator`).

    spikebill MODEL --chip loihi2 [--rates RATES.json | --rate 0.05] [--steps T] [--multicast tree]
                 [--execution pipelined|lockstep] [--no-balance] [--range] [--by-pop 10]
                 [--json out.json] [--csv pops.csv] [--viz out.html]
    spikebill MODEL --trace spikes.npz [--power-html p.html] [--power-csv p.csv] [--power-json p.json]
    spikebill --list-chips | spikebill --chip loihi2 --describe | spikebill --compare a.json b.json

MODEL is a NIR file (.nir / .h5) or a spikebill JSON graph (.json).
Activity: --rates is a JSON {population: spikes per neuron per SNN step} (optionally {"rates": ..., "active": ...});
--rate sets one rate for every population without one. With neither, each population's own `rate` is used.
--range also reports the estimate with every chip parameter at its low and high end.
"""
import argparse
import json
import os
import sys

from . import chip as chip_mod
from . import graph as graph_mod
from . import __version__
from .api import Simulator


def load_model(path, weight_bits, steps):
    if path.startswith("zoo:"):
        from . import zoo
        g = zoo.from_ref(path[4:])
    elif path.endswith(".json"):
        g = graph_mod.load(path)
    else:
        from . import nir_frontend
        g, _ = nir_frontend.from_nir(path, weight_bits=weight_bits, steps_per_inference=steps or 1)
    if steps:
        g.steps_per_inference = steps
    return g


def load_trace(path, g):
    """Recorded activity for `Simulator.power_trace`: (spikes_t, active_t or None), lists of {pop: count} per chip
    step. .npz: one array per population (spike counts per step), 'active:<pop>' arrays optional; .json: a list of
    {pop: count} or {"spikes_t": [...], "active_t": [...]}. Unknown population names are an error."""
    if path.endswith(".npz"):
        import numpy as np
        z = np.load(path)
        sp = {k: z[k] for k in z.files if not k.startswith("active:")}
        ac = {k[7:]: z[k] for k in z.files if k.startswith("active:")}
        lens = {len(v) for v in list(sp.values()) + list(ac.values())}
        if len(lens) != 1:
            raise ValueError(f"{path}: arrays have different lengths {sorted(lens)}")
        T = lens.pop()
        spikes_t = [{k: float(v[t]) for k, v in sp.items()} for t in range(T)]
        active_t = [{k: float(v[t]) for k, v in ac.items()} for t in range(T)] if ac else None
    else:
        d = json.load(open(path))
        spikes_t, active_t = (d["spikes_t"], d.get("active_t")) if isinstance(d, dict) else (d, None)
    unknown = {k for s in spikes_t for k in s} - set(g.populations)
    if unknown:
        raise ValueError(f"{path}: populations not in the model: {', '.join(sorted(unknown)[:5])}")
    if active_t is not None:            # active given for some populations only: the rest fully active
        active_t = [{k: a.get(k, float(g.populations[k].size)) for k in g.populations} for a in active_t]
    return spikes_t, active_t


def list_chips():
    d = os.path.join(os.path.dirname(__file__), "chips")
    names = sorted(f[:-5] for f in os.listdir(d) if f.endswith(".yaml"))
    for n in names:
        c = chip_mod.load(n)
        print(f"{n:<12} {c.tiles_x}x{c.tiles_y}x{c.tiles_z} tiles x {c.cores_per_tile} cores")
    return names


def main(argv=None):
    p = argparse.ArgumentParser(prog="spikebill", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--chip", default="loihi2", help="chip name (spikebill/chips/*.yaml) or a YAML path")
    p.add_argument("--list-chips", action="store_true")
    p.add_argument("--list-zoo", action="store_true", help="list the model zoo (use as MODEL: zoo:<name>)")
    p.add_argument("--describe", action="store_true", help="print the chip's parameters (value, range, source)")
    p.add_argument("--rates")
    p.add_argument("--rate", type=float)
    p.add_argument("--steps", type=int, help="SNN time steps per inference")
    p.add_argument("--multicast", choices=["none", "tree", "chip_broadcast"])
    p.add_argument("--plots", metavar="DIR", help="write paper-ready figures (energy by population, step terms, "
                   "placement, core load) as PDF + PNG into DIR")
    p.add_argument("--coproc-units", choices=["pooled", "per_unit"],
                   help="co-processor: one pooled unit (default) or one unit per block (Population.coproc_unit)")
    p.add_argument("--execution", choices=["pipelined", "lockstep"], default="pipelined")
    p.add_argument("--no-balance", action="store_true", help="greedy placement only (no load balancing)")
    p.add_argument("--objective", choices=["edp", "energy", "latency"], default="edp")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--place", nargs="?", const="contiguous", choices=["contiguous", "split"],
                   help="also search the placement (fill order; spikebill.place); 'split': non-contiguous blocks too")
    p.add_argument("--max-chips", type=int, help="balancer budget: at most this many chips")
    p.add_argument("--max-energy-increase", type=float, metavar="FRAC",
                   help="balancer budget: energy within FRAC (e.g. 0.05) of the lowest-energy candidate")
    p.add_argument("--spike-variance", choices=["mean", "binomial"], default="mean",
                   help="compute term from mean spike counts (default) or the expected busiest core (Binomial)")
    p.add_argument("--weight-bits", type=int, default=8)
    p.add_argument("--static", choices=["chip", "cores", "none"], help="static power accounting")
    p.add_argument("--range", action="store_true")
    p.add_argument("--by-pop", type=int, default=0, metavar="N", help="print the N most expensive populations")
    p.add_argument("--paper-block", choices=["latex", "markdown"], help="print a citable description of the estimate")
    p.add_argument("--json")
    p.add_argument("--csv", help="per-population costs as CSV")
    p.add_argument("--viz", help="write an HTML view of one mean-activity step (value parameters)")
    p.add_argument("--trace", help="recorded activity: .npz {population: spike counts per chip step} (optional "
                   "'active:<population>' arrays), or .json [{pop: count}, ...] / {spikes_t: [...], active_t: [...]}; "
                   "prints the per-step power summary")
    p.add_argument("--power-html", help="with --trace: HTML power plot (stacked components, step terms)")
    p.add_argument("--power-csv", help="with --trace: per-step columns as CSV")
    p.add_argument("--power-json", help="with --trace: the power trace as JSON")
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                   help="compare two saved reports (--json output): totals, components, step terms, populations")
    a = p.parse_args(argv)
    if a.compare:
        from .api import Report
        ra, rb = (Report.load(x) for x in a.compare)
        print(ra.compare(rb, labels=tuple(os.path.splitext(os.path.basename(x))[0][:12] for x in a.compare)))
        return ra.diff(rb)
    if a.list_chips:
        return list_chips()
    if a.list_zoo:
        from . import zoo
        for n in zoo.names():
            m, r = zoo.info(n), zoo.reference(n)["value"]
            claim = f"paper {m['claim_mJ']} mJ" if m.get("claim_mJ") else "no paper estimate shipped"
            print(f"zoo:{n:<22} {m['model']} ({m['paper']}); {claim}, "
                  f"reference {r['energy_J'] * 1e3:.0f} mJ on {r['chips']} chips" + (f" ({m['unit']})" if m.get("unit") else "")
                  + "".join(f"; variant zoo:{n}.{v}" for v in zoo.variants(n)))
        return zoo.names()
    if a.describe:
        print(chip_mod.describe(a.chip))
        return chip_mod.load(a.chip)
    if not a.model:
        p.error("MODEL is required")
    g = load_model(a.model, a.weight_bits, a.steps)
    rates, active = None, None
    if a.rates:
        d = json.load(open(a.rates))
        rates, active = (d.get("rates"), d.get("active")) if "rates" in d else (d, None)
    if a.rate is not None:
        rates = dict({k: a.rate for k in g.populations}, **(rates or {}))
    sim = Simulator(g, chip=a.chip, multicast=a.multicast, static=a.static, weight_bits=a.weight_bits,
                    coproc_units=a.coproc_units)
    if a.trace:
        spikes_t, active_t = load_trace(a.trace, g)
        tr = sim.power_trace(spikes_t, active_t, balance=not a.no_balance)
        print(tr.summary())
        for path, fn in ((a.power_csv, tr.to_csv), (a.power_json, tr.save), (a.power_html, tr.view)):
            if path:
                print("wrote", fn(path))
        return tr
    if a.power_html or a.power_csv or a.power_json:
        p.error("--power-html / --power-csv / --power-json need --trace")
    rep = sim.run(rates=rates, active=active, steps=a.steps, execution=a.execution, balance=not a.no_balance,
                  ranges=a.range, objective=a.objective, workers=a.workers,
                  max_chips=a.max_chips, max_energy_increase=a.max_energy_increase, place={None: False, "contiguous": True}.get(a.place, a.place),
                  spike_variance=a.spike_variance)
    print(rep.summary())
    if a.by_pop:
        print(rep.table(top=a.by_pop))
    if a.paper_block:
        print(rep.paper_block(a.paper_block))
    if a.json:
        rep.save(a.json)
    if a.csv:
        rep.to_csv(a.csv)
    if a.viz:
        print("view:", rep.view(a.viz))
    if a.plots:
        from . import plots
        print("plots:", ", ".join(plots.save_all(rep, a.plots)))
    return rep


def entry(argv=None):
    """Console script: exit status 0 on success; a user error (unknown model / chip / population, missing file,
    bad value) prints one line and exits with status 2 instead of a traceback (release gap 38)."""
    try:
        main(argv)
    except (ValueError, KeyError, FileNotFoundError, IsADirectoryError, json.JSONDecodeError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"spikebill: error: {msg}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(entry())
