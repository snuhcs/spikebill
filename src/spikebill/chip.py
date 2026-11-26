"""Chip parameter files: every parameter carries a value, an optional [low, high] range and a source tag.

YAML layout (src/spikebill/chips/<name>.yaml):

    name: loihi1
    topology: {tiles_x: 8, tiles_y: 4, cores_per_tile: 4}   # or {cores_x, cores_y} = one core per tile
    barrier_table: {1: 0.6e-6, 2: 1.0e-6}                   # optional: sync time by number of used tiles
    core_timing: max                                         # optional: max (pipelined, default) | sum
    multicast: none                                          # optional: none (unicast, default) | tree | chip_broadcast
    static_accounting: chip                                  # optional: chip (default) | cores | none
    params:
      e_synop:      {value: 23.6e-12, low: ..., high: ..., unit: J, source: "..."}
      e_synop@conv: {...}     # optional per-synapse-type override (Projection.synapse = "conv")
      ...

Cores sharing a tile share one router: messages between them cost no hop. Tiles form a 2D mesh per chip layer;
`tiles_z` > 1 stacks layers (3D-stacked dies, vertical links with t_vhop latency, t_vlink per message, e_vhop energy);
chips form a 2D board. Within a layer, tile id t sits at (t // tiles_y, t % tiles_y) (SANA-FE convention);
layer = t // (tiles_x * tiles_y).

Required parameter names (all SI units: J, s, W, bits):
  core limits:  max_neurons, syn_mem_bits, max_fanin_axons, max_fanout_axons
  energy:       e_synop (per synaptic event), e_neuron_update (every neuron, every step),
                e_neuron_active (extra, per neuron that has input or state to update), e_spike_gen (per
                spike), e_spike_out (per message sent), e_hop (tile to tile), e_chip_hop
  time:         t_synop, t_neuron_update, t_neuron_active, t_spike_gen, t_spike_out, t_msg_in (per message
                received), t_hop (latency per tile hop), t_link (time per message on a tile link =
                1 / link bandwidth), t_chip_hop (time per message on a chip link), t_barrier (one chip; overridden
                by barrier_table),
                t_barrier_multichip
  static:       p_static_core, p_static_chip
Missing parameters raise at load time so that a silent zero never enters an estimate.
"""
import os
from dataclasses import dataclass, field, replace

import yaml

REQUIRED = [
    "max_neurons", "syn_mem_bits", "max_fanin_axons", "max_fanout_axons",
    "e_synop", "e_neuron_update", "e_neuron_active", "e_spike_gen", "e_spike_out", "e_hop", "e_chip_hop",
    "t_synop", "t_neuron_update", "t_neuron_active", "t_spike_gen", "t_spike_out", "t_msg_in",
    "t_hop", "t_link", "t_chip_hop", "t_barrier", "t_barrier_multichip",
    "p_static_core", "p_static_chip",
]
OPTIONAL = ["t_vhop", "t_vlink", "e_vhop"]
# Co-processor (e.g. spiking-attention products off the neuro-cores); used only by graphs with coproc_ops. Chips
# without one get these assumed defaults (E4: near-memory AND/add unit, 0.03 pJ/op, 8 units at 1 op/ns each).
COPROC = {
    "e_coproc_op": dict(value=0.03e-12, low=0.01e-12, high=0.9e-12, unit="J",
                        source="assumed (E4): near-memory AND/add unit; high end = 45 nm add (Horowitz 2014)"),
    "t_coproc_op": dict(value=0.125e-9, low=0.125e-9, high=1e-9, unit="s",
                        source="assumed (E4): 8 units at 1 op/ns each; high end = a single unit"),
}
UNITS = {"e": "J", "t": "s", "p": "W"}
DESCRIPTION = {
    "max_neurons": "neurons per core", "syn_mem_bits": "synapse memory per core (bits)",
    "max_fanin_axons": "input axons per core", "max_fanout_axons": "output axons per core",
    "e_synop": "energy per synaptic event", "e_neuron_update": "energy per neuron per step (all mapped neurons)",
    "e_neuron_active": "extra energy per neuron with input or state", "e_spike_gen": "energy per spike generated",
    "e_spike_out": "energy per message sent", "e_hop": "energy per tile-to-tile hop",
    "e_chip_hop": "energy per chip-to-chip hop", "e_vhop": "energy per vertical (die-to-die) hop",
    "t_synop": "time per synaptic event", "t_neuron_update": "time per neuron per step",
    "t_neuron_active": "extra time per active neuron", "t_spike_gen": "time per spike generated",
    "t_spike_out": "time per message sent (core side)", "t_msg_in": "time per message received (core side)",
    "t_hop": "latency per tile hop", "t_link": "time per message on a tile link (1 / bandwidth)",
    "t_chip_hop": "time per message on a chip-to-chip link", "t_vhop": "latency per vertical hop",
    "t_vlink": "time per message on a vertical link", "t_barrier": "barrier (one chip; barrier_table overrides)",
    "t_barrier_multichip": "barrier across chips", "p_static_core": "static power per used core",
    "p_static_chip": "static power per chip",
    "e_coproc_op": "energy per co-processor operation", "t_coproc_op": "time per co-processor operation (1 / throughput)",
}
CHIP_DIR = os.path.join(os.path.dirname(__file__), "chips")


@dataclass
class Chip:
    name: str
    tiles_x: int
    tiles_y: int
    cores_per_tile: int
    params: dict           # name -> dict(value, low, high, unit, source)
    which: str = "value"   # which point of the range to use: value | low | high
    barrier_table: dict = field(default_factory=dict)  # used tiles -> sync time (largest key <= n)
    core_timing: str = "max"   # "max": neuron/message pipelines overlap; "sum": serial
    multicast: str = "none"    # "none": unicast copy per destination core; "tree": XY-tree multicast;
                               # "chip_broadcast": one packet per remote destination chip, copied by routers there
    static_accounting: str = "chip"   # "chip" (every powered chip) | "cores" (pro-rated) | "none" (dynamic only)
    tiles_z: int = 1           # stacked tile layers per chip (3D-stacked dies); vertical links need t_vhop, t_vlink, e_vhop
    neuron_update: str = "all"   # "all": every mapped neuron is updated every step (Loihi 1, SANA-FE); "active": only
                                 # neurons with input or state (event-driven update; an option, not documented for Loihi 2)
    version: str = ""          # parameter-set version from the YAML (`version:`), cite it with results
    digest: str = ""           # sha256 prefix of the YAML file content (identifies the exact parameter set)
    coproc_units: str = "pooled"   # "pooled": one co-processor whose throughput is 1 / t_coproc_op serves every block;
                                   # "per_unit": one unit per Population.coproc_unit (default: per population), each at
                                   # 1 / t_coproc_op, the step waits for the busiest unit (set t_coproc_op per unit)

    @property
    def tiles_per_layer(self):
        return self.tiles_x * self.tiles_y

    @property
    def tiles_per_chip(self):
        return self.tiles_x * self.tiles_y * self.tiles_z

    @property
    def cores_per_chip(self):
        return self.tiles_per_chip * self.cores_per_tile

    def tile_xy(self, tile):
        """(x, y) of a tile index within its layer (SANA-FE convention within a layer)."""
        r = tile % self.tiles_per_layer
        return r // self.tiles_y, r % self.tiles_y

    def tile_z(self, tile):
        """Die layer of a tile index (0 for a planar chip)."""
        return tile // self.tiles_per_layer

    def v(self, k):
        """Vertical-link parameter (t_vhop, t_vlink, e_vhop); 0 for a planar chip."""
        return self[k] if self.tiles_z > 1 else 0.0

    def __getitem__(self, k):
        p = self.params[k]
        return float(p.get(self.which, p["value"]))

    def get(self, k, synapse=None):
        """Parameter k, or its per-synapse-type override k@synapse when present."""
        if synapse and f"{k}@{synapse}" in self.params:
            return self[f"{k}@{synapse}"]
        return self[k]

    def c(self, k):
        """Co-processor parameter: the chip file's, else the assumed default (chip.COPROC)."""
        if k in self.params:
            return self[k]
        p = COPROC[k]
        return float(p.get(self.which, p["value"]))

    def with_coproc_defaults(self):
        """This chip with the assumed co-processor parameters added where the file has none (so ranges,
        sensitivity and uncertainty include them)."""
        add = {k: dict(v) for k, v in COPROC.items() if k not in self.params}
        return replace(self, params={**self.params, **add}) if add else self

    def barrier(self, n_tiles, n_chips):
        """Barrier time per step: t_barrier_multichip for several chips, else the barrier table by tiles used."""
        if n_chips > 1:
            return self["t_barrier_multichip"]
        if self.barrier_table:
            keys = sorted(k for k in self.barrier_table if k <= n_tiles)
            return float(self.barrier_table[keys[-1] if keys else min(self.barrier_table)])
        return self["t_barrier"]

    def at(self, which):
        """Same chip with every parameter at its low/high end (falls back to value when no range)."""
        return replace(self, which=which)

    def pinned(self):
        """Same chip with every parameter fixed at the current end (value = low = high = the `which` point), so code
        that picks the ends itself (Simulator.run, ranges) keeps it: `Simulator(g, chip=chip.at("high"))`."""
        if self.which == "value":
            return self
        params = {}
        for k, p in self.params.items():
            v = float(p.get(self.which, p["value"]))
            params[k] = dict(p, value=v, low=v, high=v)
        return replace(self, params=params, which="value")

    def with_overrides(self, **kv):
        """Copy with parameters replaced. A number sets value = low = high (the override is a point: a range run
        keeps it fixed instead of mixing it with the file's old range); a (low, value, high) tuple or a dict with
        any of value / low / high sets a range (missing ends default to value)."""
        params = {k: dict(v) for k, v in self.params.items()}
        for k, v in kv.items():
            if isinstance(v, dict):
                val = float(v.get("value", params.get(k, {}).get("value", 0.0)))
                lo, hi = float(v.get("low", val)), float(v.get("high", val))
            elif isinstance(v, (tuple, list)):
                if len(v) != 3:
                    raise ValueError(f"override {k}: expected (low, value, high), got {v!r}")
                lo, val, hi = (float(x) for x in v)
            else:
                lo = val = hi = float(v)
            if not lo <= val <= hi:
                raise ValueError(f"override {k}: need low <= value <= high, got {lo}, {val}, {hi}")
            params[k] = dict(params.get(k, {}), value=val, low=lo, high=hi, source="override")
        return replace(self, params=params)

    def with_core_timing(self, mode):
        """Copy with core_timing "max" (neuron and message sides overlap) or "sum"."""
        return replace(self, core_timing=mode)

    def with_multicast(self, mode):
        """Copy with multicast "none" | "tree" | "chip_broadcast"."""
        return replace(self, multicast=mode)

    def with_neuron_update(self, mode):
        """Copy with neuron_update "all" (every mapped neuron each step) or "active"."""
        return replace(self, neuron_update=mode)

    def with_coproc_units(self, mode):
        """Copy with coproc_units "pooled" (one shared co-processor) or "per_unit" (one per Population.coproc_unit)."""
        if mode not in ("pooled", "per_unit"):
            raise ValueError(f"coproc_units = {mode!r}, expected 'pooled' or 'per_unit'")
        return replace(self, coproc_units=mode)

    def with_static_accounting(self, mode):
        """Copy with static power booked per powered chip ("chip"), per used core ("cores") or not at all ("none")."""
        return replace(self, static_accounting=mode)

    def sources(self):
        """{param: source string} (tag [P]/[C]/[E6]/[L1]/[D]/[A] + document)."""
        return {k: v.get("source", "") for k, v in self.params.items()}


def from_dict(d):
    """Chip from a parsed YAML dict (validated: keys, ranges, units, options)."""
    t = d["topology"]
    if "tiles_x" in t:
        tx, ty, cpt = t["tiles_x"], t["tiles_y"], t.get("cores_per_tile", 1)
    else:
        tx, ty, cpt = t["cores_x"], t["cores_y"], 1
    params = {k: (dict(v) if isinstance(v, dict) else {"value": v}) for k, v in d["params"].items()}
    tz = int(t.get("tiles_z", 1))
    name = d.get("name")
    missing = [k for k in REQUIRED + (OPTIONAL if tz > 1 else []) if k not in params]
    if missing:
        raise ValueError(f"chip {name}: missing parameters {missing}")
    known = set(REQUIRED + OPTIONAL + list(COPROC))
    for k, v in params.items():
        base = k.split("@")[0]
        if base not in known:
            import difflib
            close = difflib.get_close_matches(base, sorted(known), n=1)
            raise ValueError(f"chip {name}: unknown parameter {k!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        if "value" not in v:
            raise ValueError(f"chip {name}: parameter {k} has no value")
        for e in ("value", "low", "high"):
            if e in v:
                v[e] = float(v[e])
                if v[e] < 0:
                    raise ValueError(f"chip {name}: {k}.{e} = {v[e]} is negative")
        lo, hi = v.get("low", v["value"]), v.get("high", v["value"])
        if not lo <= v["value"] <= hi:
            raise ValueError(f"chip {name}: {k} needs low <= value <= high, got {lo} / {v['value']} / {hi}")
        unit = v.get("unit")
        want = UNITS.get(base[0]) if base[1:2] == "_" else None
        if unit and want and unit != want:
            raise ValueError(f"chip {name}: {k} has unit {unit!r}, expected {want!r} (SI units only)")
    for key, allowed in (("core_timing", ("max", "sum")), ("multicast", ("none", "tree", "chip_broadcast")),
                         ("static_accounting", ("chip", "cores", "none")), ("neuron_update", ("all", "active")),
                         ("coproc_units", ("pooled", "per_unit"))):
        if key in d and d[key] not in allowed:
            raise ValueError(f"chip {name}: {key} = {d[key]!r}, expected one of {allowed}")
    bt = {int(k): float(v) for k, v in (d.get("barrier_table") or {}).items()}
    return Chip(d["name"], tx, ty, cpt, params, "value", bt, d.get("core_timing", "max"), d.get("multicast", "none"),
                d.get("static_accounting", "chip"), tz, d.get("neuron_update", "all"), str(d.get("version", "")),
                coproc_units=d.get("coproc_units", "pooled"))


def describe(chip):
    """Text table of a chip's parameters: value, range, unit, meaning, source."""
    c = chip if isinstance(chip, Chip) else load(chip)
    lines = [f"{c.name} (version {c.version or '-'}, sha256 {c.digest or '-'}): {c.tiles_x}x{c.tiles_y}x{c.tiles_z} tiles x {c.cores_per_tile} cores per chip "
             f"({c.cores_per_chip} cores); core_timing {c.core_timing}, multicast {c.multicast}, "
             f"static {c.static_accounting}"]
    for k, v in c.params.items():
        rng = f"[{v.get('low', v['value']):.3g}, {v.get('high', v['value']):.3g}]" if ("low" in v or "high" in v) else ""
        lines.append(f"  {k:<22}{v['value']:>11.4g} {v.get('unit', ''):<2}{rng:<22}{DESCRIPTION.get(k.split('@')[0], '')}"
                     + (f"  | {v['source']}" if v.get("source") else ""))
    if c.barrier_table:
        lines.append("  barrier_table (tiles -> s): " + ", ".join(f"{k}: {v:.3g}" for k, v in sorted(c.barrier_table.items())))
    return "\n".join(lines)


SYMBOL = {
    "max_neurons": r"N_{core}", "syn_mem_bits": r"M_{syn}", "max_fanin_axons": r"A_{in}", "max_fanout_axons": r"A_{out}",
    "e_synop": r"e_{synop}", "e_neuron_update": r"e_{neuron\_update}", "e_neuron_active": r"e_{neuron\_active}",
    "e_spike_gen": r"e_{spike\_gen}", "e_spike_out": r"e_{spike\_out}", "e_hop": r"e_{hop}", "e_chip_hop": r"e_{chip\_hop}",
    "e_vhop": r"e_{vhop}", "t_synop": r"t_{synop}", "t_neuron_update": r"t_{neuron\_update}",
    "t_neuron_active": r"t_{neuron\_active}", "t_spike_gen": r"t_{spike\_gen}", "t_spike_out": r"t_{spike\_out}",
    "t_msg_in": r"t_{msg\_in}", "t_hop": r"t_{hop}", "t_vhop": r"t_{vhop}", "t_link": r"t_{link}",
    "t_chip_hop": r"t_{chip\_hop}", "t_vlink": r"t_{vlink}", "t_barrier": r"t_{barrier}",
    "t_barrier_multichip": r"t_{barrier}^{multi}", "p_static_core": r"p_{static\_core}", "p_static_chip": r"p_{static\_chip}",
    "e_coproc_op": r"e_{cop}", "t_coproc_op": r"t_{cop}",
}


_LATEX_SPECIAL = {"\\": "\\textbackslash{}", "_": "\\_", "%": "\\%", "&": "\\&", "#": "\\#", "$": "\\$",
                  "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}",
                  "<": "\\textless{}", ">": "\\textgreater{}"}
_LATEX_SEQ = {"->": "$\\to$", "<-": "$\\leftarrow$", "<=": "$\\le$", ">=": "$\\ge$", "+-": "$\\pm$"}


def _latex_escape(text):
    """Text for a LaTeX table cell: arrows / comparisons as math, every other special character escaped
    (a bare "->" renders as "-¿" in T1 text)."""
    t, out, i = str(text), [], 0
    while i < len(t):
        two = t[i:i + 2]
        if two in _LATEX_SEQ:
            out.append(_LATEX_SEQ[two]); i += 2
        else:
            out.append(_LATEX_SPECIAL.get(t[i], t[i])); i += 1
    return "".join(out)


def to_table(chip, fmt="markdown"):
    """Parameter table (symbol, value, low, high, unit, meaning, source) for an appendix; fmt markdown | latex | paper
    (latex with the range in one column, the layout of the white paper's appendix tables)."""
    c = chip if isinstance(chip, Chip) else load(chip)
    rows = []
    for k, v in c.params.items():
        base = k.split("@")[0]
        rows.append((k, SYMBOL.get(base, base), v["value"], v.get("low", v["value"]), v.get("high", v["value"]),
                     v.get("unit", ""), DESCRIPTION.get(base, ""), v.get("source", "")))
    head = f"{c.name} v{c.version} (sha256 {c.digest}); {c.tiles_x}x{c.tiles_y}x{c.tiles_z} tiles x {c.cores_per_tile} cores"
    if fmt == "paper":       # 5 columns (range in one cell, "--" when none), widths for a two-column IEEE page
        lines = ["% generated by spikebill.chip.to_table(fmt=\"paper\")", "% " + head,
                 "\\begin{tabular}{@{}lllp{38mm}p{72mm}@{}}", "\\toprule",
                 "Symbol & Value & Range & Meaning & Source \\\\", "\\midrule"]
        for k, sym, val, lo, hi, unit, desc, src in rows:
            rng = "--" if lo == hi else f"{lo:.3g}--{hi:.3g}"
            lines.append(f"${sym}$ & {val:.3g}{' ' + unit if unit else ''} & {rng} & {_latex_escape(desc)} & "
                         f"{_latex_escape(src)} \\\\")
        if c.barrier_table:
            lines.append("\\midrule")
            lines.append("barrier table & \\multicolumn{4}{l}{" + ", ".join(
                f"{n} tile{'s' if n > 1 else ''}: {t * 1e6:.2g} us" for n, t in sorted(c.barrier_table.items())) + "} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(lines)
    if fmt == "latex":
        esc = _latex_escape
        lines = ["% generated by spikebill.chip.to_table", "% " + head,
                 "\\begin{tabular}{@{}llllll@{}}", "\\toprule",
                 "Symbol & Value & Low & High & Meaning & Source \\\\", "\\midrule"]
        for k, sym, val, lo, hi, unit, desc, src in rows:
            lines.append(f"${sym}$ & {val:.3g} {unit} & {lo:.3g} & {hi:.3g} & {esc(desc)} & {esc(src)} \\\\")
        if c.barrier_table:
            lines.append("\\midrule")
            lines.append("barrier table & \\multicolumn{5}{l}{" + ", ".join(
                f"{n} tiles: {t * 1e6:.2g} us" for n, t in sorted(c.barrier_table.items())) + "} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(lines)
    lines = [f"**{head}**", "", "| key | value | low | high | unit | meaning | source |", "|---|---|---|---|---|---|---|"]
    for k, sym, val, lo, hi, unit, desc, src in rows:
        lines.append(f"| `{k}` | {val:.4g} | {lo:.4g} | {hi:.4g} | {unit} | {desc} | {src} |")
    if c.barrier_table:
        lines.append("")
        lines.append("Barrier table (used tiles -> s): " + ", ".join(f"{n}: {t:.3g}" for n, t in sorted(c.barrier_table.items())))
    lines.append(f"Options: core_timing {c.core_timing}, multicast {c.multicast}, static_accounting {c.static_accounting}.")
    return "\n".join(lines)


def load(name_or_path):
    """Chip by shipped name (spikebill/chips/<name>.yaml) or YAML path; carries version and sha256 digest."""
    path = name_or_path if os.path.exists(name_or_path) else os.path.join(CHIP_DIR, f"{name_or_path}.yaml")
    if not os.path.exists(path):
        shipped = sorted(f[:-5] for f in os.listdir(CHIP_DIR) if f.endswith(".yaml"))
        raise ValueError(f"unknown chip {name_or_path!r}: not a file and not a shipped chip ({', '.join(shipped)})")
    import hashlib
    with open(path, "rb") as f:
        raw = f.read()
    c = from_dict(yaml.safe_load(raw))
    return replace(c, digest=hashlib.sha256(raw).hexdigest()[:12])
