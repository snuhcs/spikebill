"""Model zoo: published spiking networks as spikebill graphs with their activity, and reference chip estimates.

    from spikebill import zoo, Simulator
    zoo.names()                              # ['sdt_v1_8_512', 'sdt_v2_31m', 'spikegpt_216m', 'spikingformer_8_512']
    g = zoo.load("sdt_v1_8_512")             # Graph; rates and steps_per_inference (T) stored in it
    rep = Simulator(g, multicast="tree").run(ranges=True)
    zoo.reference("sdt_v1_8_512")            # the shipped estimate (same settings), to compare against

Each graph carries the paper's own per-layer firing rates (no re-simulation) and `meta`: the paper, where the
rates come from, the paper's 45 nm op-count energy (`claim_mJ`) and the graph conventions. Attention products
run on a co-processor (`attention="coproc"`, the reference variant) or are dropped (`attention="excluded"`:
co-processor work removed, the attention output stays a spike source; a lower bound). How the graphs were built:
the white paper's experiment E4 (E4, E4b; findings E4, E10, E17). spikegpt_216m: a spiking language
model whose rates we measured on its public checkpoint (E19; the white paper's experiment E19), no paper claim.

Reference settings: chip `loihi2` (the shipped file's version is recorded), tree multicast, static accounting
"chip", pipelined execution (T + depth chip steps), balanced mapping (objective edp), value / low / high corners.
The low / high corners are evaluated on the value mapping; the white paper's table balances the mapping again at
each end (`Simulator(g, chip=chip.at("high"))`; examples/reproduce_paper.py), which gives the same value and a
slightly different range (SDT V1 high end 1198.7 vs 1292.9 mJ).
A paper comparing its own network against these should use the same settings and report the chip-file version.

Variants (the white paper's Sec. 5 baselines; no reference estimate shipped): `variants(name)` lists them,
`load(name, variant="andgate")` (or `attention="andgate"`) gives the graph with attention built from AND-gate neurons
for the three transformers, `load("sdt_v2_31m", variant="repconv_literal")` keeps the RepConv branches unfused, and
`load("sdt_v2_31m", variant="55m")` is Meta-SpikeFormer 55M (widths x 4/3, the 31M rates as an assumption, no paper
estimate; E19(b)).
"""
import json
import os

from .. import graph as graph_mod

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_SETTINGS = dict(chip="loihi2", multicast="tree", static="chip", execution="pipelined", balance=True,
                          objective="edp")


def names():
    """Names of the shipped models."""
    return sorted(f[:-5] for f in os.listdir(HERE) if f.endswith(".json") and f != "reference.json" and "." not in f[:-5])


def variants(name):
    """Names of the shipped variants of one model (e.g. ['andgate', 'repconv_literal'])."""
    _raw(name)
    return sorted(f[len(name) + 1:-5] for f in os.listdir(HERE) if f.startswith(name + ".") and f.endswith(".json")
                  and f != name + ".json")


def _raw(name, variant=None):
    if "." in name:
        raise KeyError(f"unknown zoo model {name!r}; pass the variant separately: load(name, variant=...)")
    path = os.path.join(HERE, f"{name}.json")
    if not os.path.exists(path):
        raise KeyError(f"unknown zoo model {name!r}; available: {', '.join(names())}")
    if variant is not None:
        path = os.path.join(HERE, f"{name}.{variant}.json")
        if not os.path.exists(path):
            raise KeyError(f"no variant {variant!r} of {name!r}; available: {', '.join(variants(name)) or 'none'}")
    with open(path) as f:
        return json.load(f)


def load(name, attention="coproc", variant=None):
    """The model as a Graph (rates in `Population.rate`, T in `steps_per_inference`, description in `meta`).
    attention: "coproc" (reference), "excluded" (co-processor work dropped; a lower bound) or "andgate" (the AND-gate
    variant); variant: a name from `variants(name)`."""
    if attention not in ("coproc", "excluded", "andgate"):
        raise ValueError("attention must be 'coproc', 'excluded' or 'andgate'")
    if attention == "andgate" or variant == "andgate":
        if variant not in (None, "andgate") or attention == "excluded":
            raise ValueError("attention='andgate' is the 'andgate' variant; it does not combine with another variant")
        attention, variant = "andgate", "andgate"
    d = _raw(name, variant)
    if attention == "excluded":
        for p in d["populations"]:
            p.pop("coproc_ops", None)
            p.pop("coproc_msgs", None)
    d["meta"]["attention"] = attention
    d["meta"]["variant"] = variant
    return graph_mod.from_dict(d)


def from_ref(ref):
    """Graph for "<name>" or "<name>.<variant>" (what follows "zoo:" in a model path)."""
    name, _, variant = ref.partition(".")
    return load(name, variant=variant or None)


def info(name, variant=None):
    """The model's description (`meta`: paper, source of the rates, the paper's own energy claim, conventions)."""
    return dict(_raw(name, variant)["meta"])


def reference(name=None):
    """Shipped reference estimates (dict per model: settings, chip version, value/low/high results)."""
    with open(os.path.join(HERE, "reference.json")) as f:
        refs = json.load(f)
    return refs if name is None else refs[name]


def compute_reference(name, workers=1):
    """Recompute the reference estimate of one model with the current code and chip files (minutes per model)."""
    from .. import __version__
    from ..api import Simulator
    s = REFERENCE_SETTINGS
    g = load(name)
    sim = Simulator(g, chip=s["chip"], multicast=s["multicast"], static=s["static"])
    rep = sim.run(execution=s["execution"], balance=s["balance"], objective=s["objective"], ranges=True,
                  workers=workers)
    claim = g.meta.get("claim_mJ")
    out = dict(settings=s, chip_version=sim.chip.version, spikebill_version=__version__, chip_steps=rep.chip_steps)
    for which, r in rep.results.items():
        out[which] = dict(chips=r["chips"], cores=r["cores"], energy_J=r["energy_J"], latency_s=r["latency_s"],
                          split=r["split"],
                          ratio_to_claim=(r["energy_J"] * 1e3 / claim) if claim else None)
    return out
