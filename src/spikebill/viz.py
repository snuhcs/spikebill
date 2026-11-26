"""Visualise one time step: per-tile compute load, link loads (planar and vertical), energy and time breakdown.

`snapshot(graph, chip, mapping, spikes, active)` collects the arrays; `to_html(snap, path)` writes a self-contained
HTML page (inline SVG, no external libraries). Cost: one extra step_cost call; nothing is added to the simulation loop.
"""
import html
import json

import numpy as np

from . import cost, traffic


def snapshot(graph, chip, mapping, spikes, active):
    """Arrays for one step: busiest-core time per tile, link loads per direction, energy / time / counts."""
    r = cost.step_cost(graph, chip, mapping, spikes, active)
    tr = r["traffic"]
    W, H = mapping.mesh
    Z = getattr(mapping, "layers", 1) or 1
    zc = mapping.z if getattr(mapping, "z", None) is not None else np.zeros(mapping.n_cores, dtype=np.int64)
    neurons = cost._per_core(mapping, {k: p.size for k, p in graph.populations.items()})
    act = cost._per_core(mapping, active)
    fired = cost._per_core(mapping, spikes, by_activity=True)
    t_neur = (neurons * chip["t_neuron_update"] + act * chip["t_neuron_active"] + fired * chip["t_spike_gen"]
              + tr.core_out_msgs * chip["t_spike_out"])
    t_msg = tr.core_in_msgs * chip["t_msg_in"] + tr.core_synop_time
    t_core = np.maximum(t_neur, t_msg) if chip.core_timing == "max" else t_neur + t_msg
    tile_t = np.zeros((Z, H, W))
    np.maximum.at(tile_t, (zc, mapping.xy[:, 1], mapping.xy[:, 0]), t_core)
    pop_of = np.full((Z, H, W), "", dtype=object)
    for name, p in mapping.pops.items():
        for c in p.cores:
            pop_of[zc[c], mapping.xy[c, 1], mapping.xy[c, 0]] = name
    loads = traffic.link_loads(tr.diff)
    horiz = (loads["E"] + loads["W"])[:, :H, :W]
    vertl = (loads["N"] + loads["S"])[:, :H, :W]
    up = (loads["U"] + loads["D"])[:Z, :H, :W]
    return dict(model=graph.name, chip=chip.name, W=W, H=H, Z=Z, tiles_x=chip.tiles_x, tiles_y=chip.tiles_y,
                tile_time=tile_t, tile_pop=pop_of, link_x=horiz, link_y=vertl, link_z=up,
                energy=r["energy"], time=r["time"], counts=r["counts"])


def _heat(arr, title, unit, labels=None, cell=14, chip_tx=None, chip_ty=None):
    H, W = arr.shape
    vmax = float(arr.max()) if arr.size and arr.max() > 0 else 1.0
    parts = [f'<figure><figcaption><span class="cap">{html.escape(title)}</span>'
             f'<span class="max">max {vmax:.3g} {unit}</span></figcaption><div class="scroll">'
             f'<svg width="{W * cell}" height="{H * cell}" viewBox="0 0 {W * cell} {H * cell}" role="img" '
             f'aria-label="{html.escape(title)}">']
    for y in range(H):
        for x in range(W):
            v = float(arr[y, x]) / vmax
            pct = 8 + 92 * v if arr[y, x] > 0 else 0
            tip = f"tile ({x},{y}): {arr[y, x]:.3g} {unit}" + (f", {labels[y, x]}" if labels is not None and labels[y, x] else "")
            parts.append(f'<rect x="{x * cell}" y="{(H - 1 - y) * cell}" width="{cell - 1}" height="{cell - 1}" '
                         f'style="fill:color-mix(in oklab, var(--heat) {pct:.0f}%, var(--cell))">'
                         f'<title>{html.escape(tip)}</title></rect>')
    if chip_tx:
        for x in range(0, W + 1, chip_tx):
            parts.append(f'<line x1="{x * cell}" y1="0" x2="{x * cell}" y2="{H * cell}" class="die"/>')
        for y in range(0, H + 1, chip_ty):
            parts.append(f'<line x1="0" y1="{H * cell - y * cell}" x2="{W * cell}" y2="{H * cell - y * cell}" class="die"/>')
    parts.append("</svg></div></figure>")
    return "".join(parts)


def to_html(snap, path, title=None, subtitle=None):
    """Write a snapshot as a self-contained HTML page (inline SVG heat maps per die layer); returns path."""
    cell = max(4, min(14, 760 // max(1, snap["W"])))
    e, t = snap["energy"], snap["time"]
    tot = sum(e.values())
    rows = "".join(f"<tr><th>{k}</th><td>{v * 1e6:.4g} µJ</td><td>{v / tot:.0%}</td></tr>" for k, v in e.items())
    trows = "".join(f"<tr><th>{k}</th><td>{v * 1e6:.4g} µs</td></tr>" for k, v in t.items())
    c = snap["counts"]
    crows = "".join(f"<tr><th>{k.replace('_', ' ')}</th><td>{v:.3g}</td></tr>" for k, v in c.items())
    layers = []
    for z in range(snap["Z"]):
        figs = [_heat(snap["tile_time"][z], "busiest-core time per tile", "s", snap["tile_pop"][z], cell,
                      snap["tiles_x"], snap["tiles_y"]),
                _heat(snap["link_x"][z], "X-link load", "msg", None, cell, snap["tiles_x"], snap["tiles_y"]),
                _heat(snap["link_y"][z], "Y-link load", "msg", None, cell, snap["tiles_x"], snap["tiles_y"])]
        if snap["Z"] > 1 and z < snap["Z"] - 1:
            figs.append(_heat(snap["link_z"][z], f"vertical load to layer {z + 1}", "msg", None, cell,
                              snap["tiles_x"], snap["tiles_y"]))
        head = f"<h2>Layer {z}</h2>" if snap["Z"] > 1 else ""
        layers.append(f'<section>{head}<div class="maps">{"".join(figs)}</div></section>')
    name = title or f"{snap['model']} on {snap['chip']}"
    doc = f"""<title>{html.escape(name)}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400&display=swap">
<style>
/* Layout: datasheet header, three spec tables, then one row of tile maps per die layer. */
:root {{
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1b2330; --muted: #5d6878; --rule: #d5dae2;
  --cell: #e9edf2; --heat: #0f7c86; --die: #8a96a8;
  --display: "Barlow Semi Condensed", "Arial Narrow", sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg: #10151c; --panel: #161d26; --ink: #e6ebf2; --muted: #9aa6b6;
    --rule: #2a3442; --cell: #1c2530; --heat: #3fc1c9; --die: #5c6a7c; color-scheme: dark; }}
}}
:root[data-theme="dark"] {{ --bg: #10151c; --panel: #161d26; --ink: #e6ebf2; --muted: #9aa6b6;
  --rule: #2a3442; --cell: #1c2530; --heat: #3fc1c9; --die: #5c6a7c; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.5 var(--body); margin: 0;
  padding-block: 24px 40px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
header {{ display: grid; gap: 4px; margin-bottom: 20px; }}
h1 {{ font: 600 1.9rem/1.15 var(--display); letter-spacing: .01em; margin: 0; text-wrap: balance; }}
h2 {{ font: 600 1.15rem var(--display); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); margin: 0 0 8px; }}
.sub {{ color: var(--muted); max-width: 70ch; margin: 0; }}
.specs {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }}
table {{ background: var(--panel); border: 1px solid var(--rule); border-collapse: collapse; min-width: 220px; }}
caption {{ text-align: left; font: 600 .8rem var(--display); letter-spacing: .08em; text-transform: uppercase;
  color: var(--muted); padding: 0 0 6px; }}
th, td {{ padding: 4px 12px; border-top: 1px solid var(--rule); text-align: left; font-weight: 400; }}
td {{ font: .85rem var(--mono); font-variant-numeric: tabular-nums; text-align: right; }}
section {{ margin-bottom: 20px; }}
.maps {{ display: flex; flex-wrap: wrap; gap: 20px; }}
figure {{ margin: 0; display: grid; gap: 6px; max-width: 100%; }}
figcaption {{ display: flex; justify-content: space-between; gap: 12px; font-size: .9rem; }}
.max {{ font: .8rem var(--mono); color: var(--muted); }}
.scroll {{ overflow-x: auto; max-width: 100%; }}
svg {{ display: block; background: var(--panel); }}
line.die {{ stroke: var(--die); stroke-width: 1.5; }}
.legend {{ color: var(--muted); font-size: .85rem; max-width: 70ch; }}
</style>
<header>
  <h1>{html.escape(name)}</h1>
  <p class="sub">{html.escape(subtitle or "One mean-activity time step from the spikebill cost model.")}</p>
</header>
<div class="specs">
  <table><caption>Energy per step</caption>{rows}</table>
  <table><caption>Time per step</caption>{trows}</table>
  <table><caption>Events per step</caption>{crows}</table>
</div>
<p class="legend">Each cell is one tile (router with its cores); x runs right, y up. Darker means more load. Grey lines
mark die boundaries. Hover a cell for its value and the population on that tile.</p>
{''.join(layers)}
"""
    with open(path, "w") as f:
        f.write(doc)
    return path
