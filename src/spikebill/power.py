"""Per-step energy, time and power for a recorded activity trace.

    from spikebill import Simulator, activity
    act = activity.from_nir("model.nir", inputs=spikes)          # or lists of {pop: count} per chip step
    tr = Simulator("model.nir", chip="loihi2").power_trace(act)
    print(tr.summary()); tr.to_csv("power.csv"); tr.view("power.html")

A rate-based `Simulator.run` gives one mean step; a trace shows how energy, step time and power vary over the run
(bursts, idle steps, which term sets the step). Power per step is that step's energy over its time,
P_t = E_t / t_step,t; the mean power is total energy over total time, not the mean of P_t.
"""
import csv
import html
import json
from dataclasses import dataclass, field

import numpy as np

ENERGY = ("synaptic", "neuron", "network", "coproc", "static")
TIMES = ("compute", "network", "coproc", "barrier")


@dataclass
class PowerTrace:
    model: str
    chip: str
    energy: dict              # component -> (T,) J per chip step
    time: dict                # compute / network / coproc / barrier -> (T,) s (the step is their max + barrier)
    step_s: np.ndarray        # (T,) s per chip step
    counts: dict              # spikes, active, synops, messages, hops, chip_hops -> (T,)
    chips: int
    cores: int
    provenance: dict = field(default_factory=dict)

    @property
    def steps(self):
        return len(self.step_s)

    @property
    def energy_step_J(self):
        return sum(self.energy.values())

    @property
    def power_W(self):
        t = self.step_s
        return np.divide(self.energy_step_J, t, out=np.zeros_like(t), where=t > 0)

    @property
    def energy_J(self):
        return float(self.energy_step_J.sum())

    @property
    def latency_s(self):
        return float(self.step_s.sum())

    @property
    def mean_W(self):
        return self.energy_J / self.latency_s if self.latency_s > 0 else 0.0

    @property
    def peak_W(self):
        return float(self.power_W.max()) if self.steps else 0.0

    def bound_by(self):
        """Per step, which term sets the step time: 'compute', 'network' or 'coproc'."""
        keys = [k for k in ("compute", "network", "coproc") if k in self.time]
        return np.array(keys)[np.argmax(np.stack([self.time[k] for k in keys]), axis=0)]

    def summary(self):
        """Text summary: energy and split, time and which term sets the step, mean / peak / p95 power."""
        b = self.bound_by()
        e = self.energy_J or 1.0
        bound = ", ".join(f"{k} {np.mean(b == k):.0%}" for k in ("compute", "network", "coproc") if np.any(b == k))
        return "\n".join([
            f"{self.model} on {self.chip}: {self.steps} chip steps, {self.chips} chip(s), {self.cores} cores",
            f"  energy {self.energy_J:.4g} J  ("
            + ", ".join(f"{k} {v.sum() / e:.0%}" for k, v in self.energy.items()) + ")",
            f"  time   {self.latency_s:.4g} s  (step {np.min(self.step_s) * 1e6:.3g} - {np.max(self.step_s) * 1e6:.3g} us;"
            f" step set by {bound})",
            f"  power  mean {self.mean_W:.4g} W, peak {self.peak_W:.4g} W (step {int(np.argmax(self.power_W))}),"
            f" p95 {np.percentile(self.power_W, 95):.4g} W",
        ])

    def columns(self):
        """Per-step columns (step, start time, step time, power, energy and its components, time terms, counts)."""
        cols = {"step": np.arange(self.steps), "t_start_s": np.concatenate([[0.0], np.cumsum(self.step_s)[:-1]]),
                "step_s": self.step_s, "power_W": self.power_W, "energy_J": self.energy_step_J}
        cols.update({f"E_{k}_J": v for k, v in self.energy.items()})
        cols.update({f"t_{k}_s": v for k, v in self.time.items()})
        cols.update(self.counts)
        return cols

    def to_csv(self, path):
        """`columns()` as CSV, one row per chip step; returns path."""
        cols = self.columns()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(cols))
            for row in zip(*cols.values()):
                w.writerow([f"{x:.9g}" for x in row])
        return path

    def save(self, path):
        """Totals, provenance and the per-step columns as JSON; returns path."""
        with open(path, "w") as f:
            json.dump(dict(model=self.model, chip=self.chip, chips=self.chips, cores=self.cores,
                           energy_J=self.energy_J, latency_s=self.latency_s, mean_W=self.mean_W, peak_W=self.peak_W,
                           provenance=self.provenance,
                           trace={k: np.asarray(v).tolist() for k, v in self.columns().items()}), f, default=float)
        return path

    def view(self, path, title=None, max_points=1500):
        """Self-contained HTML plot (stacked power by component, step time, hover); traces longer than max_points
        are bucketed (bucket power = bucket energy / bucket time). Returns path."""
        with open(path, "w") as f:
            f.write(_html(self, title, max_points))
        return path


def from_evaluate(ev, t_barrier, model, chip, chips, cores, provenance=None):
    """Build a PowerTrace from `compiled.evaluate` output."""
    energy = {k: np.asarray(ev[k], float) for k in ENERGY if k in ev}
    time = {"compute": ev["compute"], "network": ev["network_t"]}
    if "coproc_t" in ev:
        time["coproc"] = ev["coproc_t"]
    time["barrier"] = np.full_like(ev["step"], t_barrier)
    counts = {k: ev[k] for k in ("spikes", "active", "synops", "messages", "hops", "chip_hops")}
    return PowerTrace(model, chip, energy, time, np.asarray(ev["step"], float), counts, chips, cores,
                      dict(provenance or {}))


# ---- HTML view: stacked power per step (by energy component) and step time, with hover read-out -------------

_COLORS = {  # categorical slots 1-5 of the reference palette (validated, adjacent pairs), light / dark
    "synaptic": ("#2a78d6", "#3987e5"), "neuron": ("#eb6834", "#d95926"), "network": ("#1baf7a", "#199e70"),
    "coproc": ("#eda100", "#c98500"), "static": ("#e87ba4", "#d55181"),
}


def _bucket(x, n):
    """Mean over n roughly equal buckets (keeps the energy-weighted shape of a long trace)."""
    if len(x) <= n:
        return np.asarray(x, float)
    idx = np.linspace(0, len(x), n + 1).astype(int)
    return np.add.reduceat(np.asarray(x, float), idx[:-1]) / np.diff(idx)


def _html(tr, title, max_points):
    T = tr.steps
    n = min(T, max_points)
    step = _bucket(tr.step_s, n)
    comps = [k for k in ENERGY if k in tr.energy and tr.energy[k].sum() > 0]
    # power of a bucket = its energy over its time (consistent with the mean-power definition)
    P = {k: _bucket(tr.energy[k], n) / np.where(step > 0, step, 1) for k in comps}
    data = dict(n=n, T=T, step_us=(step * 1e6).round(6).tolist(),
                power={k: v.round(9).tolist() for k, v in P.items()}, comps=comps,
                bucket=T / n)
    name = title or f"{tr.model} on {tr.chip}: power trace"
    css_l = "".join(f"--c-{k}: {_COLORS[k][0]};" for k in comps)
    css_d = "".join(f"--c-{k}: {_COLORS[k][1]};" for k in comps)
    e = tr.energy_J or 1.0
    legend = "".join(f'<span class="key"><i style="background:var(--c-{k})"></i>{k} '
                     f'<b>{tr.energy[k].sum() / e:.0%}</b></span>' for k in reversed(comps))
    b = tr.bound_by()
    stats = [("steps", f"{T}"), ("energy", f"{tr.energy_J:.4g} J"), ("time", f"{tr.latency_s:.4g} s"),
             ("mean power", f"{tr.mean_W:.4g} W"), ("peak power", f"{tr.peak_W:.4g} W"),
             ("chips / cores", f"{tr.chips} / {tr.cores}")]
    tiles = "".join(f'<div class="stat"><span>{k}</span><b>{v}</b></div>' for k, v in stats)
    pct = lambda q: np.percentile(tr.power_W, q)
    rows = "".join(
        f"<tr><th>{k}</th><td>{v.sum():.4g}</td><td>{v.sum() / e:.1%}</td></tr>" for k, v in tr.energy.items())
    trows = "".join(f"<tr><th>{k}</th><td>{np.mean(b == k):.0%}</td></tr>"
                    for k in ("compute", "network", "coproc") if k in tr.time)
    note = (f"Each point is the mean of {T / n:.3g} chip steps (power = bucket energy / bucket time)."
            if n < T else "Each point is one chip step.")
    return f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(name)}</title>
<style>
:root {{ --bg:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --muted:#52514e; --rule:#e2e1dc; --grid:#ecebe7; {css_l}
  --body: system-ui, "Segoe UI", sans-serif; --mono: ui-monospace, "SFMono-Regular", monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --panel:#222220;
  --ink:#f0efec; --muted:#c3c2b7; --rule:#3a3a37; --grid:#2e2e2b; {css_d} color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --panel:#222220; --ink:#f0efec; --muted:#c3c2b7; --rule:#3a3a37;
  --grid:#2e2e2b; {css_d} color-scheme: dark; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 var(--body); margin:0; padding:24px 16px 40px; }}
main {{ max-width: 980px; margin: 0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; text-wrap:balance; }}
.sub, .note {{ color:var(--muted); margin:0 0 16px; max-width:75ch; }}
.stats {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:20px; }}
.stat {{ background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:8px 12px; display:grid; }}
.stat span {{ color:var(--muted); font-size:.8rem; }} .stat b {{ font:600 1.05rem var(--mono); }}
figure {{ margin:0 0 20px; background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:12px; }}
figcaption {{ font-weight:600; margin-bottom:6px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:.85rem; color:var(--muted); margin-bottom:6px; }}
.key i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }}
.key b {{ color:var(--ink); font-weight:600; }}
svg {{ display:block; width:100%; height:auto; overflow:visible; }}
.axis text {{ fill:var(--muted); font:11px var(--mono); }} .axis line {{ stroke:var(--grid); }}
.cross {{ stroke:var(--muted); stroke-width:1; }}
#tip {{ position:fixed; pointer-events:none; background:var(--panel); border:1px solid var(--rule); border-radius:6px;
  padding:6px 10px; font:12px var(--mono); color:var(--ink); display:none; box-shadow:0 2px 8px #0002; z-index:2; }}
table {{ border-collapse:collapse; background:var(--panel); border:1px solid var(--rule); margin:0 16px 12px 0; }}
th, td {{ padding:3px 12px; border-top:1px solid var(--rule); text-align:left; font-weight:400; }}
td {{ font:.85rem var(--mono); text-align:right; }}
.tables {{ display:flex; flex-wrap:wrap; }}
</style>
<main>
<h1>{html.escape(name)}</h1>
<p class="sub">Per-step cost of a recorded activity trace from the spikebill cost model
(chip parameters at their value; static power: {html.escape(str(tr.provenance.get("static_accounting", "?")))}).</p>
<div class="stats">{tiles}</div>
<figure><figcaption>Power per chip step (W), by energy component</figcaption>
<div class="legend">{legend}</div><svg id="p" viewBox="0 0 900 260" role="img" aria-label="stacked power per step"></svg></figure>
<figure><figcaption>Step time (µs)</figcaption><svg id="t" viewBox="0 0 900 160" role="img" aria-label="step time per step"></svg></figure>
<p class="note">{note} Hover for values.</p>
<div class="tables"><table><caption>Energy over the trace (J)</caption>{rows}</table>
<table><caption>Step set by</caption>{trows}</table>
<table><caption>Power percentiles (W)</caption><tr><th>p5</th><td>{pct(5):.4g}</td></tr>
<tr><th>p50</th><td>{pct(50):.4g}</td></tr><tr><th>p95</th><td>{pct(95):.4g}</td></tr></table></div>
</main><div id="tip"></div>
<script>
const D = {json.dumps(data)};
const NS = "http://www.w3.org/2000/svg", L = 56, R = 890, fmt = v => (+v).toPrecision(3);
function el(p, tag, a) {{ const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; }}
function axes(svg, top, bot, vmax, unit) {{
  const g = el(svg, "g", {{class: "axis"}});
  for (let i = 0; i <= 4; i++) {{ const y = bot - (bot - top) * i / 4;
    el(g, "line", {{x1: L, x2: R, y1: y, y2: y}}); el(g, "text", {{x: L - 6, y: y + 4, "text-anchor": "end"}}).textContent = fmt(vmax * i / 4); }}
  el(g, "text", {{x: R, y: bot + 16, "text-anchor": "end"}}).textContent = "chip step " + D.T;
  el(g, "text", {{x: L, y: bot + 16}}).textContent = "0";
}}
const X = i => L + (R - L) * (D.n > 1 ? i / (D.n - 1) : 0.5);
function chart(svg, H, layers, vmax) {{
  const top = 8, bot = H - 22, Y = v => bot - (bot - top) * v / (vmax || 1);
  axes(svg, top, bot, vmax, "");
  let base = new Array(D.n).fill(0);
  for (const [key, vals, fill] of layers) {{
    const up = base.map((b, i) => b + vals[i]);
    let d = "M" + up.map((v, i) => X(i) + "," + Y(v)).join("L");
    if (fill) {{ d += "L" + base.map((v, i) => X(i) + "," + Y(v)).reverse().join("L") + "Z";
      el(svg, "path", {{d, fill: "var(--c-" + key + ")", stroke: "var(--panel)", "stroke-width": 1}}); }}
    else el(svg, "path", {{d, fill: "none", stroke: "var(--ink)", "stroke-width": 2}});
    if (fill) base = up;
  }}
  const c = el(svg, "line", {{class: "cross", y1: top, y2: bot, visibility: "hidden"}});
  return c;
}}
const tot = D.step_us.map((_, i) => D.comps.reduce((s, k) => s + D.power[k][i], 0));
const svgP = document.getElementById("p"), svgT = document.getElementById("t");
const cP = chart(svgP, 260, D.comps.map(k => [k, D.power[k], true]), Math.max(...tot) * 1.05);
const cT = chart(svgT, 160, [["step", D.step_us, false]], Math.max(...D.step_us) * 1.05);
const tip = document.getElementById("tip");
function hover(ev, svg) {{
  const r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) * 900 / r.width;
  const i = Math.max(0, Math.min(D.n - 1, Math.round((x - L) / (R - L) * (D.n - 1))));
  for (const c of [cP, cT]) {{ c.setAttribute("x1", X(i)); c.setAttribute("x2", X(i)); c.setAttribute("visibility", "visible"); }}
  const s0 = Math.floor(i * D.bucket), s1 = Math.max(s0, Math.floor((i + 1) * D.bucket) - 1);
  tip.innerHTML = (s1 > s0 ? `steps ${{s0}}-${{s1}}` : `step ${{s0}}`) + `<br>power ${{fmt(tot[i])}} W<br>`
    + D.comps.slice().reverse().map(k => `&nbsp;${{k}} ${{fmt(D.power[k][i])}}`).join("<br>") + `<br>step ${{fmt(D.step_us[i])}} µs`;
  tip.style.display = "block";
  tip.style.left = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px"; tip.style.top = (ev.clientY + 14) + "px";
}}
for (const s of [svgP, svgT]) {{ s.addEventListener("pointermove", e => hover(e, s));
  s.addEventListener("pointerleave", () => {{ tip.style.display = "none"; for (const c of [cP, cT]) c.setAttribute("visibility", "hidden"); }}); }}
</script>
"""
