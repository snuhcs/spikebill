"""SNN graph: populations, projections and activity, loadable from a simple JSON format.

JSON format (one file per model):

    {"name": "...", "steps_per_inference": 16,
     "populations": [{"name": "l1", "size": 1024, "model": "LIF", "rate": 0.05}, ...],
     "projections": [{"src": "in", "dst": "l1", "kind": "dense", "weight_bits": 8}, ...]}

`rate` is the mean spike probability per neuron per step (used when no measured activity is given).
Projection kinds and how fan-out is derived:
  dense        every src neuron connects to every dst neuron (fanout = dst.size)
  one_to_one   fanout = 1 (src.size must equal dst.size)
  sparse       give `fanout` (synapses per src neuron) explicitly
  conv         give `fanout` (= out_channels * k * k / stride^2, roughly) explicitly
`fanout` may be given for any kind to override the default.
"""
import json
from dataclasses import dataclass, field


@dataclass
class Population:
    name: str
    size: int
    model: str = "LIF"
    rate: float = 0.0           # spikes per neuron per step (assumed activity)
    active: float = 1.0         # fraction of neurons with input/state to update per step (assumed)
    state_bits: int = 24        # neuron state memory, used for core capacity checks
    coproc_ops: float = 0.0     # co-processor operations per spike of this population (e.g. attention products)
    coproc_msgs: float = 0.0    # messages to the co-processor per spike of this population
    coproc_inputs: list = None  # populations whose spikes are the co-processor operands behind this population's
                                # output (a dependency, one level of pipelined depth; no synapses, no cost)
    coproc_unit: str = None     # co-processor unit label (chip option coproc_units = "per_unit"): populations with the
                                # same label share one unit; None = the population's own unit


@dataclass
class Projection:
    src: str
    dst: str
    kind: str = "dense"
    fanout: float = None        # synapses per src neuron
    weight_bits: int = 8
    shared_weights: bool = False  # conv-like weight sharing: synapse memory = n_weights, not n_synapses
    n_weights: int = None
    share_group: int = None      # with shared_weights: neurons that use the same weights (e.g. tokens / positions of
                                 # one output channel); weight memory of a core then scales with the channels it holds
    axon_fanin: int = None       # with share_group: distinct input axons per output position (default: fan-in)
    synapse: str = None         # synapse type: selects per-type chip costs (e.g. e_synop@conv)
    flows: tuple = None         # exact core-level flows (set in code, see traffic.flows_from_edges)


@dataclass
class Graph:
    name: str
    populations: dict
    projections: list
    steps_per_inference: int = 1
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        """JSON-ready dict (inverse of graph.from_dict). Exact core-level `flows` (set in code) are not written;
        a warning names the projections that had them (they fall back to the statistical traffic model)."""
        import warnings
        from dataclasses import fields
        conv = lambda v: v.tolist() if hasattr(v, "tolist") else v
        row = lambda o, skip=(): {f.name: conv(getattr(o, f.name)) for f in fields(o) if f.name not in skip}
        lost = [f"{q.src}->{q.dst}" for q in self.projections if q.flows is not None]
        if lost:
            warnings.warn(f"exact flows not saved for {', '.join(lost)}")
        return dict(name=self.name, steps_per_inference=self.steps_per_inference, meta=self.meta,
                    populations=[row(p) for p in self.populations.values()],
                    projections=[row(q, ("flows",)) for q in self.projections])

    def save(self, path):
        """Write the graph as JSON (reload with graph.load or Simulator(path)); returns path."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, default=float)
        return path

    def fanout(self, proj):
        """Mean synapses per source neuron of a projection."""
        if proj.fanout is not None:
            return float(proj.fanout)
        if proj.kind == "dense":
            return float(self.populations[proj.dst].size)
        if proj.kind == "one_to_one":
            return 1.0
        raise ValueError(f"projection {proj.src}->{proj.dst} of kind {proj.kind} needs an explicit fanout")

    def n_synapses(self, proj):
        """Synapses of a projection."""
        return self.fanout(proj) * self.populations[proj.src].size

    def fanin(self, proj):
        """Mean synapses per destination neuron from this projection."""
        return self.n_synapses(proj) / self.populations[proj.dst].size

    def syn_events_per_step(self, rates=None):
        """Synaptic events per step per projection, with rates {pop: spikes/neuron/step} (default: pop.rate)."""
        rates = rates or {}
        out = {}
        for p in self.projections:
            src = self.populations[p.src]
            out[(p.src, p.dst)] = src.size * rates.get(p.src, src.rate) * self.fanout(p)
        return out

    def summary(self):
        """Counts: populations, neurons, projections, synapses."""
        n = sum(p.size for p in self.populations.values())
        s = sum(self.n_synapses(p) for p in self.projections)
        return dict(neurons=n, synapses=s, populations=len(self.populations), projections=len(self.projections))


def from_dict(d):
    """Graph from its JSON dict (Graph.to_dict / Graph.save, the zoo files)."""
    pops = {}
    for p in d["populations"]:
        pops[p["name"]] = Population(**p)
    projs = [Projection(**q) for q in d["projections"]]
    for q in projs:
        if q.src not in pops or q.dst not in pops:
            raise ValueError(f"projection {q.src}->{q.dst} refers to an unknown population")
        if q.kind == "one_to_one" and pops[q.src].size != pops[q.dst].size:
            raise ValueError(f"one_to_one {q.src}->{q.dst} with different sizes")
    return Graph(name=d.get("name", "model"), populations=pops, projections=projs,
                 steps_per_inference=d.get("steps_per_inference", 1), meta=d.get("meta", {}))


def load(path):
    """Graph from a JSON file."""
    with open(path) as f:
        return from_dict(json.load(f))
