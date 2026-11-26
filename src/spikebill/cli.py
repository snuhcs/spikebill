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


def load_rates_table(path, g):
    """Per-input rates for `Simulator.spread`: a list of {pop: rate}, one per input. .npz: `rates` (N x P) and `pops`
    (P names); .csv: a header row of population names, one row per input. Unknown
    population names are an error."""
    import numpy as np
    if path.endswith(".npz"):
        z = np.load(path)
        if "rates" not in z.files or "pops" not in z.files:
            raise ValueError(f"{path}: needs arrays 'rates' (inputs x populations) and 'pops' (names)")
        R, pops = np.asarray(z["rates"], dtype=float), [str(x) for x in z["pops"]]
    else:
        import csv
        rows = list(csv.reader(open(path)))
        if len(rows) < 2:
            raise ValueError(f"{path}: needs a header row of population names and at least one row of rates")
        pops, R = [x.strip() for x in rows[0]], np.array([[float(v) for v in r] for r in rows[1:] if r], dtype=float)
    if R.ndim != 2 or R.shape[1] != len(pops):
        raise ValueError(f"{path}: rates have shape {R.shape}, expected (inputs, {len(pops)})")
    unknown = set(pops) - set(g.populations)
    if unknown:
        raise ValueError(f"{path}: populations not in the model: {', '.join(sorted(unknown)[:5])}")
    return [dict(zip(pops, row)) for row in R]


def spread_summary(sp):
    e, l, m = sp["energy"], sp["latency"], sp["at_mean"]
    pct = [k for k in e if k.startswith("p")]
    lines = [f"Per-input spread over {sp['n']} inputs ({sp['chips']} chips, one mapping fixed at "
             f"{sp.get('map_label', 'the mean rates')})"]
    lines.append("  energy  (mJ): " + "  ".join(f"{k} {e[k] * 1e3:.4g}" for k in pct)
                 + f"  mean {e['mean'] * 1e3:.4g}  min {e['min'] * 1e3:.4g}  max {e['max'] * 1e3:.4g}")
    lines.append("  latency (ms): " + "  ".join(f"{k} {l[k] * 1e3:.4g}" for k in pct)
                 + f"  mean {l['mean'] * 1e3:.4g}  min {l['min'] * 1e3:.4g}  max {l['max'] * 1e3:.4g}")
    lines.append(f"  at the mean rates: {m['energy_J'] * 1e3:.4g} mJ, {m['latency_s'] * 1e3:.4g} ms "
                 f"({m['energy_J'] / e['mean'] - 1:+.1%} / {m['latency_s'] / l['mean'] - 1:+.1%} vs the mean per-input "
                 "cost; a single estimate from mean rates is this number)")
    return "\n".join(lines)


def spread_html(sp, path, bins=40):
    """Self-contained HTML: histograms of per-input energy and latency (inline SVG), with p5 / p50 / p95 and the cost
    at the mean rates marked."""
    import html
    import numpy as np

    def hist(x, at_mean, scale, unit, title):
        x = np.asarray(x) * scale
        lo, hi = float(min(x.min(), at_mean * scale)), float(max(x.max(), at_mean * scale))
        hi = hi if hi > lo else lo + 1e-12
        c, edges = np.histogram(x, bins=bins, range=(lo, hi))
        W, H, L, B = 560, 200, 40, 30
        sx = lambda v: L + (v - lo) / (hi - lo) * (W - L - 10)
        sy = lambda n: H - B - n / max(c.max(), 1) * (H - B - 20)
        bars = "".join(f'<rect x="{sx(edges[i]) + 1:.1f}" y="{sy(n):.1f}" width="{max(sx(edges[i + 1]) - sx(edges[i]) - 2, 1):.1f}" '
                       f'height="{H - B - sy(n):.1f}" class="bar"><title>{edges[i]:.4g}-{edges[i + 1]:.4g} {unit}: {n}</title></rect>'
                       for i, n in enumerate(c))
        marks = ""
        for lab, v, cls in (("p5", np.percentile(x, 5), "q"), ("p50", np.percentile(x, 50), "q"),
                            ("p95", np.percentile(x, 95), "q"), ("at mean rates", at_mean * scale, "m")):
            marks += (f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="14" y2="{H - B}" class="{cls}"/>'
                      f'<text x="{sx(v) + 3:.1f}" y="{12 if cls == "m" else 24}" class="lab">{lab} {v:.4g}</text>')
        axis = (f'<line x1="{L}" x2="{W - 10}" y1="{H - B}" y2="{H - B}" class="ax"/>'
                f'<text x="{L}" y="{H - 8}" class="lab">{lo:.4g}</text>'
                f'<text x="{W - 10}" y="{H - 8}" class="lab" text-anchor="end">{hi:.4g} {unit}</text>')
        return (f"<h2>{html.escape(title)}</h2><svg viewBox='0 0 {W} {H}' role='img' "
                f"aria-label='{html.escape(title)} histogram'>{bars}{marks}{axis}</svg>")

    body = (hist(sp["energy_samples"], sp["at_mean"]["energy_J"], 1e3, "mJ", "Energy per inference")
            + hist(sp["latency_samples"], sp["at_mean"]["latency_s"], 1e3, "ms", "Latency per inference"))
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Per-input cost spread</title><style>
:root{{--bg:#fff;--fg:#1f2328;--muted:#656d76;--bar:#8aa9d6;--q:#656d76;--m:#c2410c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--muted:#8b949e;--bar:#3b6db3;--q:#8b949e;--m:#fb923c}}}}
body{{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;max-width:640px;margin:24px auto;padding:0 16px}}
svg{{width:100%;height:auto}} .bar{{fill:var(--bar)}} .q{{stroke:var(--q);stroke-dasharray:3 3}} .m{{stroke:var(--m);stroke-width:2}}
.ax{{stroke:var(--muted)}} .lab{{fill:var(--muted);font-size:11px}} pre{{white-space:pre-wrap;color:var(--muted)}}
</style></head><body><h1>Per-input cost spread</h1><pre>{html.escape(spread_summary(sp))}</pre>{body}</body></html>"""
    with open(path, "w") as f:
        f.write(doc)
    return path


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
    p.add_argument("--spread", metavar="RATES", help="per-input rates (.npz with 'rates' N x P and 'pops', or .csv with a "
                   "header of population names, one row per input): one fixed mapping, every input evaluated on it; "
                   "prints p5 / p50 / p95 / mean energy and latency and the cost at the mean rates")
    p.add_argument("--map-rates", default="mean", metavar="mean|Q",
                   help="with --spread: map at the mean rates (default) or at the per-population Q-th percentile "
                        "(e.g. 95: tail-aware, E28)")
    p.add_argument("--spread-html", help="with --spread: HTML histograms of per-input energy and latency")
    p.add_argument("--spread-csv", help="with --spread: per-input energy and latency as CSV")
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
    if a.spread:
        try:
            mr = a.map_rates if a.map_rates == "mean" else float(a.map_rates)
        except ValueError:
            raise ValueError(f"--map-rates must be 'mean' or a percentile, not {a.map_rates!r}")
        sp = sim.spread(load_rates_table(a.spread, g), steps=a.steps, execution=a.execution,
                        balance=not a.no_balance, objective=a.objective, workers=a.workers, map_rates=mr)
        sp["map_label"] = "the mean rates" if mr == "mean" else f"the per-population p{mr:g} rates"
        print(spread_summary(sp))
        if a.spread_html:
            print("wrote", spread_html(sp, a.spread_html))
        if a.spread_csv:
            with open(a.spread_csv, "w") as f:
                f.write("input,energy_J,latency_s\n")
                f.writelines(f"{i},{e:.6g},{l:.6g}\n" for i, (e, l) in
                             enumerate(zip(sp["energy_samples"], sp["latency_samples"])))
            print("wrote", a.spread_csv)
        return sp
    if a.spread_html or a.spread_csv or a.map_rates != "mean":
        p.error("--spread-html / --spread-csv / --map-rates need --spread")
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
