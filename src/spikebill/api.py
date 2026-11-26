"""High-level entry point: one call from a model to a chip-level cost report.

    from spikebill import Simulator
    sim = Simulator("model.nir", chip="loihi2", multicast="tree")      # or a Graph / NetBuilder / JSON path
    rep = sim.run(rates={"l1": 0.05}, steps=4, balance=True, ranges=True)
    print(rep.summary()); rep.save("out.json"); rep.view("out.html")

Activity: per-population firing rates (spikes per neuron per SNN time step; e.g. from a training framework, or
`spikebill.funcsim` in the chip's semantics, which E3 shows matters), or a per-step trace (`run_trace`; `power_trace` gives energy / time / power per step).
Chip steps per inference: "pipelined" (default; a spike crosses one layer per chip step, so an inference of T SNN
steps takes T + depth chip steps, the Loihi execution model) or "lockstep" (T chip steps, layer delays ignored).
"""
import json
from dataclasses import dataclass, field

import numpy as np

from . import balance as balance_mod
from . import chip as chip_mod
from . import compiled
from . import graph as graph_mod
from . import mapping as mapping_mod


def _load_model(model, weight_bits):
    if isinstance(model, graph_mod.Graph):
        return model
    if hasattr(model, "graph") and callable(getattr(model, "graph")) and not hasattr(model, "nodes"):   # NetBuilder
        return model.graph()
    if isinstance(model, str) and model.startswith("zoo:"):          # zoo:<name>, see spikebill.zoo
        from . import zoo
        return zoo.from_ref(model[4:])
    if isinstance(model, str) and model.endswith(".json"):
        return graph_mod.load(model)
    from . import nir_frontend
    g, _ = nir_frontend.from_nir(model, weight_bits=weight_bits)
    return g


def depth(graph):
    """Longest chain of projections (and co-processor dependencies, `Population.coproc_inputs`) from any population
    without inputs (cycles cut)."""
    lv = levels(graph)
    return max(lv.values()) if lv else 0


def levels(graph):
    """Pipeline level of every population: longest chain of projections / co-processor dependencies reaching it."""
    succ, indeg = {}, {k: 0 for k in graph.populations}
    links = [(p.src, p.dst) for p in graph.projections]
    links += [(s, k) for k, q in graph.populations.items() for s in (q.coproc_inputs or ()) if s in graph.populations]
    for src, dst in links:
        if src != dst:
            succ.setdefault(src, []).append(dst)
            indeg[dst] += 1
    order, frontier, seen = [], [k for k, d in indeg.items() if d == 0], set()
    level = {k: 0 for k in graph.populations}
    while frontier:
        n = frontier.pop()
        if n in seen:
            continue
        seen.add(n)
        order.append(n)
        for s in succ.get(n, []):
            level[s] = max(level[s], level[n] + 1)
            indeg[s] -= 1
            if indeg[s] == 0:
                frontier.append(s)
    return level


@dataclass
class Report:
    model: str
    chip: str
    chip_steps: int
    snn_steps: int
    results: dict                     # which (value / low / high) -> per-inference numbers
    mapping: dict
    balanced: bool
    caps: dict = field(default_factory=dict)
    _snapshot_args: tuple = None
    by_population: dict = field(default_factory=dict)   # per-inference cost per population (value parameters)
    provenance: dict = field(default_factory=dict)      # everything a reader needs to reproduce the number

    def paper_block(self, fmt="latex"):
        """A citable description of this estimate for a paper: numbers + every setting that changes them.
        fmt: "latex" (a footnote-sized paragraph) or "markdown"."""
        v, pv = self.results["value"], self.provenance
        rng = ""
        if "low" in self.results:
            rng = (f" [{self.results['low']['energy_J'] * 1e3:.3g}, {self.results['high']['energy_J'] * 1e3:.3g}]")
        lat_rng = ""
        if "low" in self.results:
            lat_rng = f" [{self.results['low']['latency_s'] * 1e3:.3g}, {self.results['high']['latency_s'] * 1e3:.3g}]"
        op = pv.get("opcount_J")
        items = [
            ("Energy / inference", f"{v['energy_J'] * 1e3:.3g} mJ{rng}" + (" (low, high chip parameters)" if rng else "")),
            ("Latency / inference", f"{v['latency_s'] * 1e3:.3g} ms{lat_rng}"),
            ("Op-count reference", f"{op * 1e3:.3g} mJ (synaptic events x 0.9 pJ, 45 nm)" if op is not None else "-"),
            ("Chip", f"{self.chip} parameters v{pv.get('chip_version') or '?'} (sha256 {pv.get('chip_digest') or '?'}), "
                     f"{v['chips']} chip(s), {v['cores']} cores"),
            ("Mapping", f"{'balanced (' + pv.get('objective', '') + ')' if self.balanced else 'greedy'}, "
                        f"placement {pv.get('placement', 'graph order')}, "
                        f"multicast {pv.get('multicast')}, static power {pv.get('static_accounting')}, "
                        f"core timing {pv.get('core_timing')}, neuron update {pv.get('neuron_update')}"
                        + (", compute term = expected busiest core under Binomial spikes"
                           if pv.get("spike_variance") == "binomial" else "")),
            *([("Balancing spread", f"{bb['candidates']} mapping(s) within {bb['tol']:.0%} of the best "
                                    f"{pv.get('objective', '')}: {bb['energy_J'][0] * 1e3:.3g}-{bb['energy_J'][1] * 1e3:.3g} mJ, "
                                    f"{bb['latency_s'][0] * 1e3:.3g}-{bb['latency_s'][1] * 1e3:.3g} ms, "
                                    f"{bb['chips'][0]}-{bb['chips'][1]} chips")]
              if (bb := pv.get("balance_band")) and self.balanced else []),
            *([("Balancing budget", ", ".join(f"{k.replace('_', ' ')} {x}" for k, x in bu.items() if x is not None))]
              if (bu := pv.get("balance_budget")) else []),
            ("Execution", f"{pv.get('execution')}: {self.snn_steps} SNN steps -> {self.chip_steps} chip steps"),
            ("Activity", pv.get("activity", "?")),
            ("Model", f"{self.model}: {pv.get('neurons', 0):,} neurons, {pv.get('synapses', 0):,.0f} synapses"),
            ("Tool", f"spikebill {pv.get('spikebill_version')}"),
        ]
        if pv.get("attention"):
            att = {"excluded": "spike x spike products (attention) not costed",
                   "andgate": "spike x spike products (attention) mapped as AND + sum neurons",
                   "coproc": f"spike x spike products (attention) on a co-processor "
                             f"({'one unit per block' if pv.get('coproc_units') == 'per_unit' else 'one pooled unit'}), "
                             f"{v['split'].get('coproc', 0.0):.0%} of the energy"}[pv["attention"]]
            items.insert(-1, ("Attention", att))
        if fmt == "markdown":
            return "\n".join(f"- **{k}:** {x}" for k, x in items)
        esc = lambda t: t.replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")
        return ("% generated by spikebill Report.paper_block()\n\\noindent\\textit{Hardware cost (spikebill).} "
                + "; ".join(f"{k}: {esc(x)}" for k, x in items) + ".")

    def table(self, top=10, sort="total"):
        """Text table of the most expensive populations (energy per inference, share, cores, core time)."""
        rows = sorted(self.by_population.items(), key=lambda kv: -kv[1][sort])
        tot = sum(v["total"] for v in self.by_population.values()) or 1.0
        cop = any("coproc" in v for v in self.by_population.values())
        comps = ("synaptic", "neuron", "network", "static") + (("coproc",) if cop else ())
        lines = [f"{'population':<24}{'energy J':>11}{'share':>7}{'syn':>6}{'neu':>6}{'net':>6}{'stat':>6}"
                 + (f"{'cop':>6}" if cop else "") + f"{'cores':>9}{'core us':>9}"]
        for k, v in rows[:top]:
            e = v["total"] or 1.0
            lines.append(f"{k[:23]:<24}{v['total']:>11.3g}{v['total'] / tot:>7.1%}"
                         + "".join(f"{v.get(c, 0.0) / e:>6.0%}" for c in comps)
                         + f"{v['cores']:>9.1f}{v['core_time'] * 1e6:>9.3g}" + (" *" if v["critical"] else ""))
        if len(rows) > top:
            lines.append(f"... {len(rows) - top} more; * = on the slowest core")
        return "\n".join(lines)

    def to_csv(self, path):
        """Per-population costs (J per inference, cores, core time, critical flag) as CSV; returns path."""
        import csv
        keys = ["synaptic", "neuron", "network", "static", "total", "cores", "core_time", "critical"]
        if any("coproc" in v for v in self.by_population.values()):
            keys.insert(4, "coproc")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["population"] + keys)
            for k, v in self.by_population.items():
                w.writerow([k] + [v[c] for c in keys])
        return path

    def summary(self):
        """Three-line text summary: chips/cores/steps, energy and its split, latency and its step terms (+ range)."""
        r = self.results["value"]
        lines = [f"{self.model} on {self.chip}: {r['chips']} chip(s), {r['cores']} cores, "
                 f"{self.chip_steps} chip steps per inference ({self.snn_steps} SNN steps)"
                 + (" [balanced]" if self.balanced else ""),
                 f"  energy / inference  {r['energy_J']:.4g} J  ("
                 + ", ".join(f"{k} {v:.0%}" for k, v in r["split"].items()) + ")",
                 f"  latency / inference {r['latency_s']:.4g} s  (step {r['step_s'] * 1e6:.3g} us: compute "
                 f"{r['time']['compute'] * 1e6:.3g}, network {r['time']['network'] * 1e6:.3g}, "
                 + (f"co-processor {r['time']['coproc'] * 1e6:.3g}, " if "coproc" in r["time"] else "")
                 + f"barrier {r['time']['barrier'] * 1e6:.3g})"]
        if "low" in self.results:
            lo, hi = self.results["low"], self.results["high"]
            lines.append(f"  parameter range: energy {lo['energy_J']:.3g} - {hi['energy_J']:.3g} J, "
                         f"latency {lo['latency_s']:.3g} - {hi['latency_s']:.3g} s")
        return "\n".join(lines)

    def save(self, path):
        """Everything in the report (results, mapping, by_population, provenance) as JSON; returns path."""
        with open(path, "w") as f:
            json.dump(dict(model=self.model, chip=self.chip, chip_steps=self.chip_steps, snn_steps=self.snn_steps,
                           results=self.results, mapping=self.mapping, balanced=self.balanced, caps=self.caps,
                           by_population=self.by_population, provenance=self.provenance),
                      f, indent=1, default=float)
        return path

    def variance_check(self, steps=200, seed=0):
        """Is the mean-count step reliable for this mapping? Draws Binomial per-core spikes (spikebill.variance) and
        returns R = E[step] / step(mean counts) and R_compute for the compute term alone (value parameters). R near 1:
        fine; a large R_compute: cores holding few neurons of a hot sender (E22). Needs the original run."""
        from . import variance
        g, chip, m, sp, ac = self._snapshot_args
        return variance.busiest_core(compiled.compile_model(g, chip.at("value"), m), sp, ac, steps=steps, seed=seed)

    @classmethod
    def load(cls, path):
        """Read a report written by `save` (JSON). The loaded report has no mapping objects, so `view` and
        `spikebill.plots.placement` / `chip_load` need the original run; everything else works."""
        with open(path) as f:
            d = json.load(f)
        keys = ("model", "chip", "chip_steps", "snn_steps", "results", "mapping", "balanced", "caps", "by_population",
                "provenance")
        missing = [k for k in keys[:7] if k not in d]
        if missing:
            raise ValueError(f"{path}: not a spikebill report (missing {', '.join(missing)})")
        return cls(**{k: d.get(k, {}) for k in keys})

    def diff(self, other):
        """Numbers of `other` relative to this report (ratio other / self): energy, latency, chips, cores per
        parameter end both have; energy split and step terms (value); per-population energy (value) with the
        absolute change in J. Returns a dict."""
        out = dict(results={}, split={}, time={}, by_population={})
        for w in ("value", "low", "high"):
            a, b = self.results.get(w), other.results.get(w)
            if a and b:
                out["results"][w] = {k: dict(a=a[k], b=b[k], ratio=b[k] / a[k] if a[k] else float("nan"))
                                     for k in ("energy_J", "latency_s", "chips", "cores")}
        a, b = self.results["value"], other.results["value"]
        ea, eb = a["energy_J"], b["energy_J"]
        for k in sorted(set(a["split"]) | set(b["split"])):
            out["split"][k] = dict(a=a["split"].get(k, 0.0) * ea, b=b["split"].get(k, 0.0) * eb)
        for k in sorted(set(a["time"]) | set(b["time"])):
            out["time"][k] = dict(a=a["time"].get(k, 0.0), b=b["time"].get(k, 0.0))
        for k in set(self.by_population) | set(other.by_population):
            x = self.by_population.get(k, {}).get("total", 0.0)
            y = other.by_population.get(k, {}).get("total", 0.0)
            out["by_population"][k] = dict(a=x, b=y, delta=y - x)
        return out

    def compare(self, other, labels=("A", "B"), top=8):
        """Text comparison of two reports (chip A vs B, setting X vs Y): totals and range, energy components, step
        terms, and the populations whose energy changed most. Works on loaded reports (`Report.load`)."""
        d = self.diff(other)
        la, lb = labels
        r = lambda x: f"{x:.3g}"
        lines = [f"{la}: {self.model} on {self.chip}; {lb}: {other.model} on {other.chip}",
                 f"{'':<22}{la[:11]:>12}{lb[:11]:>12}{'ratio':>10}"]
        for w, v in d["results"].items():
            for k, unit in (("energy_J", "J"), ("latency_s", "s"), ("chips", ""), ("cores", "")):
                if w != "value" and k in ("chips", "cores"):
                    continue
                lines.append(f"{(k.split('_')[0] + ' ' + unit).strip() + ('' if w == 'value' else ' (' + w + ')'):<22}"
                             f"{r(v[k]['a']):>12}{r(v[k]['b']):>12}{v[k]['ratio']:>9.3g}x")
        lines.append("energy components (J, value)")
        for k, v in d["split"].items():
            lines.append(f"  {k:<20}{r(v['a']):>12}{r(v['b']):>12}" + (f"{v['b'] / v['a']:>9.3g}x" if v["a"] else ""))
        lines.append("step terms (us, value)")
        for k, v in d["time"].items():
            lines.append(f"  {k:<20}{v['a'] * 1e6:>12.3g}{v['b'] * 1e6:>12.3g}"
                         + (f"{v['b'] / v['a']:>9.3g}x" if v["a"] else ""))
        pops = sorted(d["by_population"].items(), key=lambda kv: -abs(kv[1]["delta"]))
        if pops:
            lines.append(f"largest per-population energy changes (J, value; {len(pops)} populations)")
            for k, v in pops[:top]:
                lines.append(f"  {k[:20]:<20}{r(v['a']):>12}{r(v['b']):>12}{v['delta']:>+12.3g}")
        pa, pb = self.provenance, other.provenance
        changed = [k for k in sorted(set(pa) | set(pb)) if k not in ("balance_band", "opcount_J")
                   and pa.get(k) != pb.get(k)]
        if changed:
            lines.append("provenance differs in: " + ", ".join(changed))
        return "\n".join(lines)

    def roofline(self, coproc=True):
        """The per-core / per-link roofline baseline on this report's mapping and activity (value parameters;
        `spikebill.cost.roofline`): energy / latency per inference, its binding roof, and the full model's numbers on the
        same mapping. Shows what the routed network model adds: equal while a core binds, lower when links bind."""
        from . import cost
        g, chip, m, sp, ac = self._snapshot_args
        return cost.roofline(g, chip, m, sp, ac, self.chip_steps, coproc=coproc)

    def view(self, path, title=None):
        """Self-contained HTML view of one mean-activity step (tile loads, link loads, energy/time); returns path."""
        from . import viz
        g, chip, m, sp, ac = self._snapshot_args
        return viz.to_html(viz.snapshot(g, chip, m, sp, ac), path, title=title,
                           subtitle=f"{self.chip_steps} chip steps per inference; one mean step shown.")


def _binomial_step(cm, spikes, act, r):
    """Step result with the compute term replaced by its expectation under Binomial spikes (static scaled)."""
    from . import variance
    t_c, _ = variance.expected_compute(cm, spikes, act)
    tm = dict(r["time"])
    step = max(t_c, tm["network"], tm.get("coproc", 0.0)) + tm["barrier"]
    en = dict(r["energy"])
    if tm["step"] > 0:
        en["static"] = en.get("static", 0.0) * step / tm["step"]
    tm.update(compute=t_c, step=step)
    return dict(r, time=tm, energy=en)


def _check_pops(g, d, what, hi):
    """Activity dicts name populations of the graph: a typo would otherwise silently fall back to the graph's own
    rate (release gap 38). Values must be finite, >= 0 and (active) <= 1."""
    import difflib
    import math
    import numbers
    unknown = [k for k in d if k not in g.populations]
    if unknown:
        k = unknown[0]
        near = difflib.get_close_matches(k, list(g.populations), n=3)
        raise ValueError(f"{what}: {len(unknown)} population(s) not in model {g.name!r}, e.g. {k!r}"
                         + (f" (did you mean {', '.join(map(repr, near))}?)" if near else ""))
    for k, v in d.items():
        if not isinstance(v, numbers.Real) or not math.isfinite(v) or v < 0 or (hi is not None and v > hi):
            raise ValueError(f"{what}[{k!r}] = {v!r}: must be a number >= 0" + (f" and <= {hi}" if hi else ""))


class Simulator:
    """A model on a chip. `model`: Graph, NetBuilder, NIR / JSON path or "zoo:<name>"; `chip`: name or YAML path
    or Chip; `multicast` / `static` / `coproc_units` override the chip file's options; `overrides`: {param: value} (a point: low = value
    = high) or {param: (low, value, high)}. Graphs with co-processor work get the assumed co-processor parameters."""
    def __init__(self, model, chip="loihi2", multicast=None, static=None, weight_bits=8, overrides=None,
                 coproc_units=None):
        self.graph = _load_model(model, weight_bits)
        self.chip = chip_mod.load(chip) if isinstance(chip, str) else chip
        if multicast:
            self.chip = self.chip.with_multicast(multicast)
        if coproc_units:
            self.chip = self.chip.with_coproc_units(coproc_units)
        if static:
            self.chip = self.chip.with_static_accounting(static)
        if any(p.coproc_ops or p.coproc_msgs for p in self.graph.populations.values()):
            self.chip = self.chip.with_coproc_defaults()          # assumed co-processor parameters, with ranges
        self.overrides = overrides
        if overrides:
            self.chip = self.chip.with_overrides(**overrides)
        # chip.at("low" / "high"): run() picks the ends itself, so fix every parameter at the chosen end
        self.chip = self.chip.pinned()

    @classmethod
    def from_torch(cls, model, inputs, chip="loihi2", trace_kw=None, **kw):
        """Trace a spiking PyTorch module (snnTorch, SpikingJelly, ...) on input batches; measured rates are kept
        in the graph, so `run()` needs no rates. `trace_kw` goes to spikebill.torch_frontend.from_torch."""
        from .torch_frontend import from_torch
        g, info = from_torch(model, inputs, **(trace_kw or {}))
        sim = cls(g, chip=chip, **kw)
        sim.trace_info = info
        return sim

    def _activity(self, rates, active, steps, execution):
        g = self.graph
        T = steps or g.steps_per_inference or 1
        chip_steps = T + depth(g) if execution == "pipelined" else T
        scale = T / chip_steps                                  # spikes per neuron per chip step
        rates = dict(rates or {})
        _check_pops(g, rates, "rates", hi=None)
        if active is not None:
            _check_pops(g, active, "active", hi=1.0)
        spikes = {k: p.size * rates.get(k, p.rate) * scale for k, p in g.populations.items()}
        if active is None:                                      # neurons with input or state: received events
            recv = {k: 0.0 for k in g.populations}
            for p in g.projections:
                recv[p.dst] += spikes[p.src] * g.fanout(p)
            act = {k: min(float(g.populations[k].size), recv[k]) for k in g.populations}
        else:
            act = {k: g.populations[k].size * active.get(k, 1.0) for k in g.populations}
        return spikes, act, T, chip_steps

    def run(self, rates=None, active=None, steps=None, execution="pipelined", balance=True, ranges=False,
            objective="edp", workers=1, max_chips=None, max_energy_increase=None, place=False,
            spike_variance="mean"):
        """Estimate one inference. rates: {pop: spikes per neuron per SNN step} (default: the graph's); active:
        {pop: fraction of neurons updated} (default: estimated from received events); steps: SNN steps (default:
        the graph's); execution "pipelined" (T + depth chip steps) | "lockstep" (T); balance: load-balance the
        mapping (objective "edp" | "energy" | "latency"); ranges: also evaluate all parameters at low / high;
        max_chips / max_energy_increase: budgets for the balanced mapping (see spikebill.balance.balance); place: also
        search the placement (fill order, spikebill.place.search) and keep the best order under `objective`
        (place="split": populations may also occupy non-contiguous blocks, release gap 23); with
        balance, jointly for every balancer candidate (E20: -36 to -38% EDP on the network-bound Meta-SpikeFormer 55M, 0% on the compute-bound zoo transformers).
        spike_variance: "mean" (default; core loads from mean spike counts) | "binomial" (the compute term is the
        expected busiest core under independent Binomial spikes, spikebill.variance.expected_compute; static energy
        follows the step; the mapping is still chosen on means; E22). Returns a Report."""
        if spike_variance not in ("mean", "binomial"):
            raise ValueError(f"spike_variance must be 'mean' or 'binomial', not {spike_variance!r}")
        g = self.graph
        spikes, act, T, chip_steps = self._activity(rates, active, steps, execution)
        bband = None
        if balance:
            m, _, caps, hist = balance_mod.balance(g, self.chip, spikes, act, objective=objective, workers=workers,
                                                   max_chips=max_chips, max_energy_increase=max_energy_increase,
                                                   place=place)
            b = balance_mod.band(hist)
            bband = dict(tol=b["tol"], candidates=b["n"], chips=b["chips"],
                         energy_J=tuple(x * chip_steps for x in b["energy"]),
                         latency_s=tuple(x * chip_steps for x in b["step"]))
        else:
            m, caps = mapping_mod.map_graph(g, self.chip), {}
        placement = "graph order"
        if place not in (False, None, True, "split"):
            raise ValueError(f"place must be False, True or 'split', not {place!r}")
        if place and balance:
            placement = ("searched per balancer candidate (spikebill.place, joint"
                         + (", non-contiguous blocks)" if place == "split" else ")"))
        elif place:
            from . import place as place_mod
            ps = place_mod.search(g, self.chip, spikes, act, caps=caps, objective=objective, split=place == "split")
            m = ps["mapping"]
            best = min(ps["rows"], key=lambda r: r["score"])["candidate"]
            placement = "graph order (searched)" if best == "graph_order" else f"{best} fill order (spikebill.place)"
        results, by_pop = {}, {}
        for which in (("value", "low", "high") if ranges else ("value",)):
            c = self.chip.at(which)
            cm = compiled.compile_model(g, c, m)
            r = compiled.step_result(cm, spikes, act)
            if spike_variance == "binomial":
                r = _binomial_step(cm, spikes, act, r)
            if which == "value":
                by_pop = {k: {q: (x * chip_steps if q in ("synaptic", "neuron", "network", "static", "coproc", "total") else x)
                              for q, x in v.items()} for k, v in compiled.breakdown(cm, spikes, act).items()}
            e = sum(r["energy"].values())
            results[which] = dict(chips=m.n_chips, cores=m.n_cores, energy_J=e * chip_steps,
                                  split={k: v / e if e else 0.0 for k, v in r["energy"].items()},
                                  latency_s=r["time"]["step"] * chip_steps, step_s=r["time"]["step"],
                                  time=r["time"], counts_per_step=r["counts"])
        mp = {k: dict(cores=p.n_cores, per_core=p.per_core, limited_by=p.limited_by) for k, p in m.pops.items()}
        from . import __version__
        summ = g.summary()
        snn_rates = {k: (rates or {}).get(k, p.rate) for k, p in g.populations.items()}
        opcount = sum(g.syn_events_per_step(snn_rates).values()) * 0.9e-12 * T
        info = getattr(self, "trace_info", None)
        activity = ("measured by spikebill.from_torch on the given inputs" if info is not None and not rates
                    else "rates given by the user" if rates else "rates stored in the model graph")
        prov = dict(chip_version=self.chip.version, chip_digest=self.chip.digest, multicast=self.chip.multicast,
                    static_accounting=self.chip.static_accounting, core_timing=self.chip.core_timing,
                    neuron_update=self.chip.neuron_update, coproc_units=self.chip.coproc_units,
                    execution=execution, objective=objective, activity=activity, spikebill_version=__version__,
                    neurons=summ["neurons"], synapses=summ["synapses"], opcount_J=opcount,
                    overrides=getattr(self, "overrides", None),
                    attention=(info or {}).get("attention") or (
                        "coproc" if any(p.coproc_ops for p in g.populations.values()) else None), balance_band=bband,
                    placement=placement, spike_variance=spike_variance,
                    balance_budget=(dict(max_chips=max_chips, max_energy_increase=max_energy_increase)
                                    if balance and (max_chips is not None or max_energy_increase is not None) else None))
        return Report(g.name, self.chip.name, chip_steps, T, results, mp, balance, caps,
                      (g, self.chip, m, spikes, act), by_pop, prov)

    def sensitivity(self, rates=None, active=None, steps=None, execution="pipelined", balance=True, params=None,
                    objective="edp", workers=1):
        """One-at-a-time ("tornado") sensitivity: each chip parameter with a range set to its low and high end, all
        others at their value, mapping fixed at the value mapping (capacity parameters are skipped: they change the
        mapping). Returns rows sorted by energy swing: dict(param, energy_low, energy_high, latency_low,
        latency_high) per inference, plus the base energy / latency."""
        g = self.graph
        spikes, act, T, chip_steps = self._activity(rates, active, steps, execution)
        if balance:
            m = balance_mod.balance(g, self.chip, spikes, act, objective=objective, workers=workers)[0]
        else:
            m = mapping_mod.map_graph(g, self.chip)

        def ev(chip):
            r = compiled.step_result(compiled.compile_model(g, chip, m), spikes, act)
            return sum(r["energy"].values()) * chip_steps, r["time"]["step"] * chip_steps

        base_e, base_t = ev(self.chip)
        skip = {"max_neurons", "syn_mem_bits", "max_fanin_axons", "max_fanout_axons"}
        names = params or [k for k, v in self.chip.params.items()
                           if k not in skip and ("low" in v or "high" in v)
                           and (v.get("low", v["value"]) != v["value"] or v.get("high", v["value"]) != v["value"])]
        rows = []
        for k in names:
            v = self.chip.params[k]
            out = {}
            for end in ("low", "high"):
                c = self.chip.with_overrides(**{k: v.get(end, v["value"])})
                out[end] = ev(c)
            rows.append(dict(param=k, energy_low=out["low"][0], energy_high=out["high"][0],
                             latency_low=out["low"][1], latency_high=out["high"][1]))
        rows.sort(key=lambda r: -abs(r["energy_high"] - r["energy_low"]))
        return dict(base_energy=base_e, base_latency=base_t, rows=rows, chip_steps=chip_steps)

    def uncertainty(self, n=512, prior="loguniform", seed=0, rates=None, active=None, steps=None,
                    execution="pipelined", balance=True, objective="edp", workers=1, groups=()):
        """Joint (Monte Carlo) uncertainty over the chip-parameter box: every parameter with a range is drawn
        independently from `prior` on [low, high] ("loguniform", "uniform", or "triangular" with mode = value),
        the mapping fixed at the value mapping (capacity parameters stay at value). Returns energy / latency
        samples per inference and their 5 / 50 / 95 percentiles. `groups`: lists of parameters that move together
        (one shared quantile per group, e.g. per-event energies that share one calibration scale)."""
        g = self.graph
        spikes, act, T, chip_steps = self._activity(rates, active, steps, execution)
        m = (balance_mod.balance(g, self.chip, spikes, act, objective=objective, workers=workers)[0] if balance
             else mapping_mod.map_graph(g, self.chip))
        cm0 = compiled.compile_model(g, self.chip, m)
        skip = {"max_neurons", "syn_mem_bits", "max_fanin_axons", "max_fanout_axons"}
        box = {k: (v.get("low", v["value"]), v["value"], v.get("high", v["value"])) for k, v in self.chip.params.items()
               if k not in skip and (v.get("low", v["value"]) != v.get("high", v["value"]))}
        rng = np.random.default_rng(seed)
        E, L = np.zeros(n), np.zeros(n)
        group_of = {k: gi for gi, grp in enumerate(groups) for k in grp}

        def quantile(u, lo, mid, hi):
            if prior == "uniform" or lo <= 0:
                return lo + u * (hi - lo)
            if prior == "triangular":
                f = (mid - lo) / (hi - lo)
                return lo + np.sqrt(u * (hi - lo) * (mid - lo)) if u < f else hi - np.sqrt((1 - u) * (hi - lo) * (hi - mid))
            return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))

        for i in range(n):
            gu = rng.uniform(size=max(1, len(groups)))
            draw = {}
            for k, (lo, mid, hi) in box.items():
                u = gu[group_of[k]] if k in group_of else rng.uniform()
                draw[k] = quantile(u, lo, mid, hi)
            c = self.chip.with_overrides(**draw)
            cm = compiled.rechip(cm0, c) or compiled.compile_model(g, c, m)
            r = compiled.step_result(cm, spikes, act)
            E[i] = sum(r["energy"].values()) * chip_steps
            L[i] = r["time"]["step"] * chip_steps
        pct = lambda x: dict(zip(("p5", "p50", "p95"), np.percentile(x, [5, 50, 95]).tolist()))
        return dict(energy=pct(E), latency=pct(L), energy_samples=E, latency_samples=L, prior=prior, n=n,
                    params=sorted(box), chips=m.n_chips, groups=[list(g_) for g_ in groups])

    def fast(self, rates=None, active=None, steps=None, execution="pipelined"):
        """Cheap repeated estimates (e.g. inside a training loop, as a regulariser or a logged metric): the first call
        maps (greedy) and compiles the model and caches it; later calls only re-evaluate the matrices with new rates
        (milliseconds). The mapping is not re-balanced when rates change. Returns dict(energy_J, latency_s, split)."""
        g = self.graph
        spikes, act, T, chip_steps = self._activity(rates, active, steps, execution)
        if getattr(self, "_fast_cm", None) is None:
            self._fast_cm = compiled.compile_model(g, self.chip, mapping_mod.map_graph(g, self.chip))
        r = compiled.step_result(self._fast_cm, spikes, act)
        e = sum(r["energy"].values())
        return dict(energy_J=e * chip_steps, latency_s=r["time"]["step"] * chip_steps,
                    split={k: v / e if e else 0.0 for k, v in r["energy"].items()}, chips=self._fast_cm.n_chips)

    def run_trace(self, spikes_t, active_t, balance=False):
        """Per-step costs for a recorded activity trace (lists of {pop: count}); batched (spikebill.compiled)."""
        g = self.graph
        mean_s = {k: float(np.mean([s.get(k, 0.0) for s in spikes_t])) for k in g.populations}
        mean_a = {k: float(np.mean([a.get(k, 0.0) for a in active_t])) for k in g.populations}
        m = (balance_mod.balance(g, self.chip, mean_s, mean_a)[0] if balance else mapping_mod.map_graph(g, self.chip))
        cm = compiled.compile_model(g, self.chip, m)
        S = np.array([[s.get(k, 0.0) for k in cm.pops] for s in spikes_t])
        A = np.array([[a.get(k, 0.0) for k in cm.pops] for a in active_t])
        return compiled.evaluate(cm, S, A), m

    def measured_trace(self, execution="pipelined"):
        """Per-chip-step spike counts {pop: count} measured by `from_torch` (info["rates_t"]: rate per SNN step and
        population, batch mean); populations without a per-step measurement keep their mean rate. "lockstep": T
        chip steps; "pipelined": T + depth chip steps, population p at pipeline level l works on SNN step s - l at
        chip step s. Use as `power_trace(sim.measured_trace())`."""
        info = getattr(self, "trace_info", None) or {}
        rt = info.get("rates_t")
        if not rt:
            raise ValueError("no per-step measurement: build the Simulator with Simulator.from_torch (without a "
                             "custom run function)")
        g = self.graph
        T = g.steps_per_inference or 1
        lv = levels(g) if execution == "pipelined" else {k: 0 for k in g.populations}
        n_chip = T + (max(lv.values()) if execution == "pipelined" else 0)
        out = []
        for s in range(n_chip):
            row = {}
            for k, p in g.populations.items():
                t = s - lv[k]
                if 0 <= t < T:
                    row[k] = p.size * (rt[k][t] if k in rt else p.rate)
            out.append(row)
        return out

    def power_trace(self, spikes_t, active_t=None, balance=False):
        """Per-step energy, time and power (spikebill.power.PowerTrace) for a recorded trace: lists of {pop: count}
        per chip step, or an `activity.Activity` (from_nir), or `measured_trace()` of a from_torch Simulator.
        Without `active_t`, active neurons are estimated from received events as in `run`. Chip parameters at
        their value."""
        if hasattr(spikes_t, "spikes_t"):
            spikes_t, active_t = spikes_t.spikes_t, spikes_t.active_t if active_t is None else active_t
        g = self.graph
        if active_t is None:
            active_t = []
            for s in spikes_t:
                recv = {k: 0.0 for k in g.populations}
                for p in g.projections:
                    recv[p.dst] += s.get(p.src, 0.0) * g.fanout(p)
                active_t.append({k: min(float(g.populations[k].size), recv[k]) for k in g.populations})
        ev, m = self.run_trace(spikes_t, active_t, balance=balance)
        from . import __version__, power
        prov = dict(chip_version=self.chip.version, chip_digest=self.chip.digest, multicast=self.chip.multicast,
                    static_accounting=self.chip.static_accounting, core_timing=self.chip.core_timing,
                    neuron_update=self.chip.neuron_update, coproc_units=self.chip.coproc_units, balanced=balance, spikebill_version=__version__,
                    overrides=getattr(self, "overrides", None))
        return power.from_evaluate(ev, self.chip.barrier(m.n_tiles, m.n_chips), g.name, self.chip.name,
                                   m.n_chips, m.n_cores, prov)
