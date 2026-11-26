"""Paper-ready static figures (matplotlib) of a `Report`: where the energy goes, what sets the step time, how the
model is laid out over the chips, and how several settings compare. Every function returns the matplotlib Figure;
`save_all(report, outdir)` writes the four single-report figures as PDF + PNG. matplotlib is an optional dependency.

    rep = Simulator("zoo:spikingformer_8_512", multicast="tree").run()
    plots.save_all(rep, "figs/")                  # energy_by_population, step_terms, placement, chip_load
    plots.compare([rep_a, rep_b], ["greedy", "balanced"])
"""
import os

import numpy as np

COMPONENTS = ("synaptic", "neuron", "network", "coproc", "static")
# categorical palette in fixed order (validated light-mode palette, as in the paper figures)
COLORS = dict(synaptic="#2a78d6", neuron="#eb6834", network="#1baf7a", coproc="#6250d6", static="#eda100")
TIMES = ("compute", "network", "coproc", "barrier")
TCOLORS = dict(compute="#2a78d6", network="#1baf7a", coproc="#6250d6", barrier="#9a9893")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _plt():
    import matplotlib
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg") and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    return plt


def _fmt_J(x):
    return f"{x * 1e3:.3g} mJ" if x < 1 else f"{x:.3g} J"


def energy_by_population(report, top=15, ax=None):
    """Horizontal stacked bars: energy per inference of the `top` costliest populations by component (the rest folded
    into one "other" row); the critical population (sets the busiest core) is marked."""
    plt = _plt()
    bp = report.by_population
    rows = sorted(bp.items(), key=lambda kv: -kv[1]["total"])
    head, tail = rows[:top], rows[top:]
    if tail:
        other = {c: sum(v.get(c, 0.0) for _, v in tail) for c in COMPONENTS}
        other["total"] = sum(v["total"] for _, v in tail)
        head = head + [(f"other ({len(tail)})", other)]
    fig = ax.figure if ax is not None else plt.figure(figsize=(6, 0.28 * len(head) + 1.0))
    ax = ax or fig.add_subplot(111)
    y = np.arange(len(head))[::-1]
    comps = [c for c in COMPONENTS if any(v.get(c, 0.0) for _, v in head)]
    for yi, (name, v) in zip(y, head):
        left = 0.0
        for c in comps:
            w = v.get(c, 0.0) * 1e3
            ax.barh(yi, w, left=left, color=COLORS[c], height=0.7, edgecolor="white", lw=0.5,
                    label=c if yi == y[0] else None)
            left += w
        ax.text(left, yi, f" {v['total'] * 1e3:.3g}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([n + (" *" if bp.get(n, {}).get("critical") else "") for n, _ in head], fontsize=7)
    ax.set_xlabel("energy per inference (mJ)")
    tot = report.results["value"]["energy_J"]
    ax.set_title(f"{report.model} on {report.chip}: {_fmt_J(tot)} per inference (* = sets the busiest core)",
                 fontsize=8, loc="left", color=INK, pad=18)
    ax.legend(ncol=len(comps), loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=7,
              handlelength=1.0, columnspacing=0.8)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    return fig


def step_terms(report, ax=None):
    """What sets the step time: compute / network / co-processor (overlapping, the max binds) plus the barrier, at
    value with low-high whiskers when the report has ranges."""
    plt = _plt()
    res = report.results
    t = res["value"]["time"]
    keys = [k for k in TIMES if k in t]
    fig = ax.figure if ax is not None else plt.figure(figsize=(4.5, 2.4))
    ax = ax or fig.add_subplot(111)
    bind = max((k for k in keys if k != "barrier"), key=lambda k: t[k])
    for i, k in enumerate(keys):
        v = t[k] * 1e6
        ax.bar(i, v, color=TCOLORS[k], alpha=1.0 if k in (bind, "barrier") else 0.35, width=0.6)
        if "low" in res and "high" in res:
            lo, hi = res["low"]["time"].get(k, 0.0) * 1e6, res["high"]["time"].get(k, 0.0) * 1e6
            ax.errorbar(i, v, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], color=INK, capsize=3, lw=0.8)
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(keys)), [k + (" (binds)" if k == bind else "") for k in keys], fontsize=7)
    ax.set_ylabel("time per chip step (µs)")
    ax.set_title(f"step {t['step'] * 1e6:.3g} µs = max(compute, network, co-proc) + barrier; "
                 f"{report.chip_steps} steps -> {res['value']['latency_s'] * 1e3:.3g} ms", fontsize=8, loc="left")
    return fig


def _tile_grid(report):
    from . import viz
    from .api import levels
    g, chip, m, sp, ac = report._snapshot_args
    snap = viz.snapshot(g, chip, m, sp, ac)
    lv = levels(g)
    Z, H, W = snap["tile_time"].shape
    lev = np.full((Z, H, W), np.nan)
    for z in range(Z):
        for yy in range(H):
            for xx in range(W):
                n = snap["tile_pop"][z, yy, xx]
                if n:
                    lev[z, yy, xx] = lv.get(n, 0)
    return snap, lev, chip, m


def _chip_lines(ax, chip, W, H):
    for x in range(chip.tiles_x, W, chip.tiles_x):
        ax.axvline(x - 0.5, color="white", lw=1.2)
    for y in range(chip.tiles_y, H, chip.tiles_y):
        ax.axhline(y - 0.5, color="white", lw=1.2)


def placement(report, ax=None):
    """The tile mesh of all chips, each tile coloured by the pipeline level of the population on it (early layers
    light, late layers dark; white lines = chip boundaries): how the model is laid out over the chips."""
    plt = _plt()
    snap, lev, chip, m = _tile_grid(report)
    fig = ax.figure if ax is not None else plt.figure(figsize=(6, 6 * snap["H"] / max(snap["W"], 1) + 0.8))
    ax = ax or fig.add_subplot(111)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            top = np.nanmax(lev, axis=0)                         # empty tiles stay NaN (blank)
    im = ax.imshow(top, cmap="viridis_r", origin="lower", interpolation="nearest",
                   aspect="equal")
    _chip_lines(ax, chip, snap["W"], snap["H"])
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="pipeline level of the population")
    ax.set_title(f"placement: {m.n_chips} chips, {m.n_cores:,} cores (white lines = chip boundaries)",
                 fontsize=8, loc="left")
    return fig


def chip_load(report, ax=None):
    """Busiest-core time per tile (log colour, one mean step): hot spots that set the compute term."""
    plt = _plt()
    from matplotlib.colors import LogNorm
    snap, _, chip, m = _tile_grid(report)
    tt = snap["tile_time"].max(axis=0) * 1e6
    fig = ax.figure if ax is not None else plt.figure(figsize=(6, 6 * snap["H"] / max(snap["W"], 1) + 0.8))
    ax = ax or fig.add_subplot(111)
    pos = tt[tt > 0]
    im = ax.imshow(np.where(tt > 0, tt, np.nan), cmap="magma_r", origin="lower", interpolation="nearest",
                   norm=LogNorm(vmin=max(pos.min(), pos.max() / 1e3), vmax=pos.max()) if pos.size else None)
    _chip_lines(ax, chip, snap["W"], snap["H"])
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="busiest-core time per step (µs)")
    ax.set_title(f"core load: max {pos.max():.3g} µs (compute term), median {np.median(pos):.3g} µs"
                 if pos.size else "core load", fontsize=8, loc="left")
    return fig


def compare(reports, labels, ax=None):
    """Several settings side by side: energy per inference split by component (bars) and latency (dots, right
    panel), e.g. greedy vs balanced, multicast modes, two chips."""
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 0.45 * len(reports) + 1.2), sharey=True,
                               gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.08))
    y = np.arange(len(reports))[::-1]
    comps = [c for c in COMPONENTS if any(r.results["value"]["split"].get(c) for r in reports)]
    for yi, r in zip(y, reports):
        v = r.results["value"]; left = 0.0
        for c in comps:
            w = v["split"].get(c, 0.0) * v["energy_J"] * 1e3
            a.barh(yi, w, left=left, color=COLORS[c], height=0.6, edgecolor="white", lw=0.5,
                   label=c if yi == y[0] else None)
            left += w
        a.text(left, yi, f" {_fmt_J(v['energy_J'])}, {v['chips']} chips", va="center", fontsize=7)
        b.plot(v["latency_s"] * 1e3, yi, "o", color=INK, ms=4)
        b.text(v["latency_s"] * 1e3, yi + 0.18, f"{v['latency_s'] * 1e3:.3g} ms", ha="center", fontsize=7)
    a.set_yticks(y, labels); a.set_xlabel("energy per inference (mJ)")
    a.legend(ncol=len(comps), loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=7)
    b.set_xlabel("latency per inference (ms)"); b.set_xlim(left=0)
    xmax = max(r.results["value"]["energy_J"] for r in reports) * 1e3
    a.set_xlim(0, xmax * 1.45)
    return fig


def save_all(report, outdir, formats=("pdf", "png")):
    """Write energy_by_population, step_terms, placement and chip_load of one report; returns the paths."""
    plt = _plt()
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for fn in (energy_by_population, step_terms, placement, chip_load):
        fig = fn(report)
        for ext in formats:
            p = os.path.join(outdir, f"{fn.__name__}.{ext}")
            fig.savefig(p, bbox_inches="tight", dpi=200)
            paths.append(p)
        plt.close(fig)
    return paths
