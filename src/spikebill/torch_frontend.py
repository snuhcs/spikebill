"""PyTorch front end: a spiking nn.Module + a few input batches -> spikebill Graph with measured firing rates.

    import spikebill
    g = spikebill.from_torch(net, x)                       # x: (T, B, ...) input spikes; net called once per step
    g = spikebill.from_torch(net, x, multi_step=True)      # SpikingJelly multi-step: net(x) once with x (T, B, ...)
    g = spikebill.from_torch(net, x, run=my_loop)          # any loop: my_loop(net, x) runs one full inference batch
    rep = spikebill.Simulator(g, chip="loihi2").run()      # rates are already in the graph

How it works. Forward hooks on the leaf modules record which module produced which tensor; a TorchFunctionMode
carries that provenance through functional ops between modules (additions of residual branches, reshapes, cat), so
the dataflow is recovered without tracing. Spiking-neuron modules (snnTorch, SpikingJelly, or classes given in
`neuron_types`) become populations with their measured firing rate (spikes per neuron per step). Conv / Linear layers
become projections from the populations feeding them to the neurons they feed:
  - Conv{1,2}d: shared weights over output positions (synapses = out positions x out channels x in/groups x kernel);
  - Linear on (..., N, D) tokens: token-wise shared weights; on vectors: dense;
  - pooling of spikes: OR-neuron populations (as NetBuilder.pool); pooling of currents between a conv and its
    neurons: folded into the conv (larger effective kernel at the pooled resolution, an approximation);
  - BatchNorm, Dropout, Flatten, Identity and functional reshapes are pass-through;
  - a weight layer feeding another weight layer, several populations, or the output gets a graded population
    (rate 1, or 0 at the end) that its consumers read one-to-one (its synapses are not duplicated per consumer);
  - a residual sum (`x + f(x)`) read by several nodes is a graded accumulator population (one-to-one copies in,
    weight layers write into it); read by one node, it is folded into that node's input.
Inputs: binary inputs give a measured rate; graded inputs are rate 1 (every pixel sent every step) with a warning.
Element-wise products of two traced tensors (`*`, `mul`: e.g. SDT's spike-driven attention Q * K and g * V) are
handled like matrix products with one AND operation per output element (names `hd<i>`; under `andgate` the product
population is the AND neurons, fed by both operands). Non-spiking functional ops on traced tensors (softmax,
layer_norm, sigmoid, tanh, exp, gelu, silu, erf; and the modules Softmax, LayerNorm, GroupNorm, Sigmoid, Tanh,
GELU, SiLU) are never costed: they are listed in `info["unmapped_ops"]` (kind "nonspiking", elements per inference)
with a warning. A direct link between populations of different sizes (a reduction such as a sum over tokens, or a
broadcast) is costed as a local projection with fan-out max(1, size ratio), with a warning.
Products of two traced tensors (attention: Q K^T, then scores x V; `matmul`, `@`, `bmm`) are measured while tracing:
nonzero products ("AND operations"), output elements and nonzero outputs, per inference. They are listed in
`info["unmapped_ops"]` with those counts. With `attention="excluded"` (default) the product is not mapped: its
output carries the operands' provenance (consumers get one-to-one edges from Q, K, V), and the counts are left for
the user to cost (e.g. as a co-processor). With `attention="andgate"` each product becomes neurons on the chip, as
E4's AND-gate construction: an AND population (one neuron per (i, l, j) term of A[i, l] B[l, j], rate = measured
nonzero products) fed by A (contiguous, fan-out n) and B (strided, fan-out m), and a graded sum population (one neuron
per output element, rate = measured nonzero outputs) fed one-to-one by the AND neurons; the output then comes from the
sum population.
With `attention="coproc"` the products run on a co-processor: each operand spike is one message to it
(`Population.coproc_msgs`), the measured nonzero products are its operations (`coproc_ops`, per output spike), and
the output is a graded population (measured nonzero rate) that the co-processor writes back, feeding the consumers
as any population does. A product read only by another product (Q K^T inside (Q K^T) V) stays in the co-processor:
no population; its operations are added to the product that reads it. Chip parameters e_coproc_op / t_coproc_op
(assumed defaults in spikebill.chip.COPROC). Populations are emitted in dataflow order (the mapper places them in graph
order, so consecutive layers become neighbours).
"""
import math
from collections import defaultdict

import numpy as np

from . import graph as graph_mod

POOL = ("MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d",
        "AdaptiveMaxPool2d")
PASS = ("BatchNorm1d", "BatchNorm2d", "LayerNorm", "Dropout", "Dropout2d", "Flatten", "Identity", "Unflatten")


def _is_neuron(mod, extra):
    if extra and isinstance(mod, tuple(extra)):
        return True
    m = type(mod).__module__ or ""
    if m.startswith("snntorch"):
        import snntorch
        return isinstance(mod, getattr(snntorch, "SpikingNeuron", ())) or type(mod).__name__ in (
            "Leaky", "Synaptic", "Lapicque", "Alpha", "RLeaky", "RSynaptic")
    if m.startswith("spikingjelly") and ".neuron" in m:
        return True
    return False


HADAMARD_OPS = ("mul", "__mul__", "__rmul__", "mul_", "__imul__", "multiply", "logical_and", "__and__")
NONSPIKING_FUNCS = ("softmax", "log_softmax", "layer_norm", "group_norm", "sigmoid", "tanh", "exp", "gelu", "silu",
                    "erf")
NONSPIKING_MODULES = ("Softmax", "LogSoftmax", "LayerNorm", "GroupNorm", "Sigmoid", "Tanh", "GELU", "SiLU")


def _kind(mod, extra):
    if _is_neuron(mod, extra):
        return "neuron"
    n = type(mod).__name__
    if n in ("Conv1d", "Conv2d", "Linear"):
        return "weight"
    if n in POOL:
        return "pool"
    return "pass"


def _first_tensor(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _tensors(x):
    import torch
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for v in x for t in _tensors(v)]
    if isinstance(x, dict):
        return [t for v in x.values() for t in _tensors(v)]
    return []


class _Tracker:
    def __init__(self, model, neuron_types):
        self.extra = tuple(neuron_types or ())
        self.names = {}
        self.kinds = {}
        self.prod = {}                 # id(tensor) -> frozenset of node names (kept alive in self.keep)
        self.keep = []
        self.edges = defaultdict(set)  # dst node -> set of src nodes
        self.stats = defaultdict(lambda: dict(spikes=0.0, elems=0.0, calls=0, numel=0, in_shape=None,
                                              out_shape=None, binary=True))
        self.input_stats = dict(spikes=0.0, elems=0.0, numel=0, binary=True)
        self.matmuls = []
        self.mm = {}                   # "mm<i>" -> measured product statistics (see matmul)
        self.mm_idx = 0
        self.hd_idx = 0
        self.add_idx = 0
        self.nonspiking = {}           # (op, sources) -> elements per forward pass of non-spiking functional ops
        self.order = {}                # node -> first time it produced an output (dataflow order for placement)
        self.attention = "excluded"
        self.depth = 0
        leaves = []
        for name, m in model.named_modules():
            k = _kind(m, self.extra)
            if k == "neuron" or (not list(m.children()) and name):
                if any(name.startswith(n + ".") for n in leaves):      # inside a neuron module
                    continue
                if k == "neuron":
                    leaves.append(name)
                self.names[m] = name
                self.kinds[name] = k
        self.handles = []
        for m, name in self.names.items():
            self.handles.append(m.register_forward_pre_hook(self._pre(name)))
            self.handles.append(m.register_forward_hook(self._post(name)))
        self.handles.append(model.register_forward_pre_hook(lambda mod, inp: self._reset()))

    def _reset(self):
        self.prod.clear(); self.keep.clear()
        self.mm_idx = 0
        self.hd_idx = 0
        self.add_idx = 0

    def add(self, out, srcs):
        """A sum of two traced tensors between modules (residual add): a node `add<i>` fed by the operands' sources.
        Becomes a graded accumulator population if several nodes consume it, else is folded (pass-through)."""
        name = f"add{self.add_idx}"
        self.add_idx += 1
        self.order.setdefault(name, len(self.order))
        self.kinds[name] = "sum"
        self.edges[name] |= set(srcs)
        st = self.stats[name]
        o = out.detach()
        if st["out_shape"] is None:
            st["out_shape"] = tuple(o.shape)
        st["calls"] += 1
        st["numel"] += o.numel()
        st["spikes"] += float((o != 0).sum())
        st["elems"] += float(o.numel())
        st["binary"] = False
        return name

    def matmul(self, op, a, b, out, sa, sb):
        """Record one product of two traced tensors (call order within a forward pass names it); returns its name."""
        import torch
        name = f"mm{self.mm_idx}"
        self.mm_idx += 1
        self.order.setdefault(name, len(self.order))
        m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
        A, B = (a.detach() != 0).float(), (b.detach() != 0).float()
        ands = float(torch.matmul(A, B).sum())
        o = out.detach()
        r = self.mm.setdefault(name, dict(name=name, op=op, a_srcs=set(), b_srcs=set(), mkn=(m, k, n), calls=0,
                                          and_ops=0.0, and_elems=0.0, out_nonzero=0.0, out_elems=0.0, out_numel=0))
        r["a_srcs"] |= set(sa); r["b_srcs"] |= set(sb)
        r["calls"] += 1
        r["and_ops"] += ands
        r["and_elems"] += float(o.numel() * k)
        r["out_nonzero"] += float((o != 0).sum())
        r["out_elems"] += float(o.numel())
        return name

    def hadamard(self, op, a, b, out, sa, sb):
        """Record one element-wise product of two traced tensors (e.g. SDSA's Q * K, g * V); returns its name. One
        AND operation per output element; nonzero products and outputs are measured like `matmul`."""
        name = f"hd{self.hd_idx}"
        self.hd_idx += 1
        self.order.setdefault(name, len(self.order))
        o = out.detach()
        ands = float(((a.detach() != 0) & (b.detach() != 0)).sum()) if a.shape == b.shape else float((o != 0).sum())
        r = self.mm.setdefault(name, dict(name=name, op=op, kind="hadamard", a_srcs=set(), b_srcs=set(),
                                          mkn=tuple(o.shape[-2:]) + (1,), calls=0, and_ops=0.0, and_elems=0.0,
                                          out_nonzero=0.0, out_elems=0.0, out_numel=0))
        r["a_srcs"] |= set(sa); r["b_srcs"] |= set(sb)
        r["calls"] += 1
        r["and_ops"] += ands
        r["and_elems"] += float(o.numel())
        r["out_nonzero"] += float((o != 0).sum())
        r["out_elems"] += float(o.numel())
        return name

    def nonspiking_op(self, op, out, srcs):
        """A non-spiking functional op on traced tensors (softmax, layer_norm, exp, ...): counted, not costed."""
        k = (op, frozenset(srcs))
        r = self.nonspiking.setdefault(k, dict(op=op, srcs=set(srcs), elems=0.0, calls=0))
        r["elems"] += float(out.numel())
        r["calls"] += 1

    def sources(self, t):
        return self.prod.get(id(t))

    def mark(self, t, srcs):
        self.prod[id(t)] = frozenset(srcs)
        self.keep.append(t)

    def _pre(self, name):
        def hook(mod, inp):
            self.depth += 1
            srcs = set()
            ts = _tensors(inp)
            if self.kinds[name] == "neuron":            # state tensors passed in (mem, syn) are not inputs
                ts = ts[:1]
            for t in ts:
                s = self.sources(t)
                if s is None:
                    s = {"input"}
                    self._input(t)
                srcs |= s
            self.edges[name] |= (srcs - {name})
            st = self.stats[name]
            x = _first_tensor(inp)
            if x is not None and st["in_shape"] is None:
                st["in_shape"] = tuple(x.shape)
        return hook

    def _post(self, name):
        def hook(mod, inp, out):
            self.depth -= 1
            y = _first_tensor(out)
            if y is None:
                return
            st = self.stats[name]
            if st["out_shape"] is None:
                st["out_shape"] = tuple(y.shape)
            st["calls"] += 1
            st["numel"] += y.numel()
            if self.kinds[name] in ("neuron", "pool"):
                yd = y.detach()
                st["spikes"] += float((yd != 0).sum())
                st["elems"] += float(yd.numel())
                lead = yd.reshape(yd.shape[0], -1) if yd.dim() > 1 else yd.reshape(1, -1)
                st.setdefault("per_call", []).append(((lead != 0).sum(1).double().cpu().numpy(), lead.shape[1]))
                if st["binary"] and not bool(((yd == 0) | (yd == 1)).all()):
                    st["binary"] = False
            self.order.setdefault(name, len(self.order))
            for t in _tensors(out):
                self.mark(t, {name})
        return hook

    def _input(self, t):
        if id(t) in self.prod:
            return
        td = t.detach()
        lead = td.reshape(td.shape[0], -1) if td.dim() > 1 else td.reshape(1, -1)
        self.input_stats.setdefault("per_call", []).append(((lead != 0).sum(1).double().cpu().numpy(), lead.shape[1]))
        self.input_stats["spikes"] += float((td != 0).sum())
        self.input_stats["elems"] += float(td.numel())
        if self.input_stats["binary"] and not bool(((td == 0) | (td == 1)).all()):
            self.input_stats["binary"] = False
        if not self.input_stats["numel"]:
            self.input_stats["numel"] = t.numel()
            self.input_stats["shape"] = tuple(t.shape)
        self.mark(t, {"input"})

    def close(self):
        for h in self.handles:
            h.remove()
        self._reset()


def _mode(tr):
    import torch
    from torch.overrides import TorchFunctionMode

    class Prov(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if tr.depth > 0:                               # inside a hooked module: its hook labels the output
                return out
            ins = _tensors(args) + _tensors(kwargs or {})
            srcs = set()
            known = [tr.sources(t) for t in ins]
            for s in known:
                if s:
                    srcs |= s
            outs = _tensors(out)
            if srcs and outs:
                name = getattr(func, "__name__", str(func))
                if (name in ("matmul", "__matmul__", "bmm") and len(ins) >= 2 and known[0] and known[1]
                        and ins[0].dim() >= 2 and ins[1].dim() >= 2):
                    mm = tr.matmul(name, ins[0], ins[1], outs[0], known[0], known[1])
                    if tr.attention in ("andgate", "coproc"):
                        srcs = {mm}
                elif (name in HADAMARD_OPS and len(ins) >= 2 and known[0] and known[1]
                      and ins[0].dim() >= 1 and ins[1].dim() >= 1 and len(outs) == 1):
                    hd = tr.hadamard(name, ins[0], ins[1], outs[0], known[0], known[1])
                    if tr.attention in ("andgate", "coproc"):
                        srcs = {hd}
                elif name in NONSPIKING_FUNCS and len(outs) >= 1:
                    tr.nonspiking_op(name, outs[0], srcs)
                elif (name in ("add", "__add__", "__radd__", "__iadd__", "add_") and sum(1 for s in known if s) >= 2
                      and len(outs) == 1):
                    srcs = {tr.add(outs[0], srcs)}
                elif name == "einsum" and sum(1 for s in known if s) >= 2:
                    tr.matmuls.append(dict(name=None, op=name, srcs=sorted(srcs), shape=tuple(outs[0].shape)))
                for t in outs:
                    if isinstance(t, torch.Tensor) and t.data_ptr() != 0:
                        tr.mark(t, srcs)
            return out
    return Prov()


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def from_torch(model, inputs, run=None, multi_step=False, steps=None, name=None, weight_bits=8,
               neuron_types=(), reset=None, attention="excluded"):
    """Trace `model` on `inputs` and return (Graph, info). See the module docstring.
    attention: "excluded" (products of two traced tensors not mapped; counts in info["unmapped_ops"]), "andgate"
    (mapped as AND + sum neuron populations) or "coproc" (run on a co-processor; output as a graded population)."""
    import torch
    if attention not in ("excluded", "andgate", "coproc"):
        raise ValueError(f"attention must be 'excluded', 'andgate' or 'coproc', not {attention!r}")
    tr = _Tracker(model, neuron_types)
    tr.attention = attention
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _mode(tr):
            if run is not None:
                run(model, inputs)
            elif multi_step:
                if reset:
                    reset(model)
                model(inputs)
            else:
                if reset:
                    reset(model)
                for t in range(inputs.shape[0]):
                    model(inputs[t])
    finally:
        tr.close()
        model.train(was_training)
    T = steps or int(inputs.shape[0])
    B = int(inputs.shape[1]) if inputs.dim() > 1 else 1
    calls_root = T if (run is None and not multi_step) else 1
    info = dict(warnings=[], unmapped_ops=[], steps=T, batch=B, rates={}, attention=attention)
    for r in tr.mm.values():
        info["unmapped_ops"].append(dict(
            name=r["name"], op=r["op"], a_srcs=sorted(r["a_srcs"]), b_srcs=sorted(r["b_srcs"]), mkn=r["mkn"],
            and_ops_per_inference=r["and_ops"] / B, and_density=r["and_ops"] / max(r["and_elems"], 1.0),
            out_nonzero_per_inference=r["out_nonzero"] / B, out_density=r["out_nonzero"] / max(r["out_elems"], 1.0),
            mapped=attention != "excluded", kind=r.get("kind", "matmul")))
    info["unmapped_ops"] += tr.matmuls
    # non-spiking operations (softmax, normalisation, smooth activations): counted, never costed
    nonsp = [dict(name=None, op=r["op"], kind="nonspiking", srcs=sorted(r["srcs"]),
                  elems_per_inference=r["elems"] / max(B, 1), mapped=False) for r in tr.nonspiking.values()]
    for m, mname in tr.names.items():
        st = tr.stats.get(mname)
        if type(m).__name__ in NONSPIKING_MODULES and st and st["calls"]:
            nonsp.append(dict(name=mname, op=type(m).__name__, kind="nonspiking", srcs=sorted(tr.edges.get(mname, ())),
                              elems_per_inference=st["numel"] / max(B, 1), mapped=False))
    info["unmapped_ops"] += nonsp
    if nonsp:
        info["warnings"].append(
            f"{len(nonsp)} non-spiking operation(s) ({', '.join(sorted({u['op'] for u in nonsp}))}) are not costed: "
            "they need a host, a co-processor or a neuron construction; info['unmapped_ops'] has their element counts")
    kinds, stats, edges = tr.kinds, tr.stats, tr.edges

    def per_sample(name):
        st = stats[name]
        n = st["numel"] / max(1, calls_root * B)                  # per sample per root call
        return n / (T if multi_step or run is not None else 1)

    # ---- resolve pass-through nodes -----------------------------------------------------------------------
    def up(node, seen=()):
        """Non-pass sources of a node (through pass-through nodes)."""
        out = set()
        for s in edges.get(node, ()):
            if s in seen:
                continue
            if s != "input" and kinds.get(s) == "pass":
                out |= up(s, seen + (s,))
            else:
                out.add(s)
        return out

    def build_down():
        d = defaultdict(set)
        for n in kinds:
            if kinds[n] == "pass":
                continue
            for s in up(n):
                d[s].add(n)
        return d

    down = build_down()
    # a residual sum read by one node is folded into that node's input; read by several, it is an accumulator
    for n in [n for n, k in kinds.items() if k == "sum"]:
        if len(down.get(n, ())) <= 1:
            kinds[n] = "pass"
    down = build_down()

    b_pops, b_projs = [], []
    sizes = {}

    def add_pop(pname, size, rate, graded=False):
        sizes[pname] = int(round(size))
        b_pops.append(dict(name=pname, size=int(round(size)), rate=float(rate)))
        info["rates"][pname] = float(rate)

    ins = tr.input_stats
    in_size = ins["numel"] / max(1, B) / (T if multi_step and ins["numel"] and len(ins.get("shape", ())) and
                                            ins["shape"][0] == T else 1)
    in_rate = ins["spikes"] / ins["elems"] if ins["elems"] else 0.0
    if not ins["binary"]:
        info["warnings"].append("input is graded (not 0/1): every input value is sent every step (rate 1)")
        in_rate = 1.0
    add_pop("input", in_size, in_rate)

    for n, k in kinds.items():
        if k == "neuron":
            st = stats[n]
            if not st["calls"]:
                continue
            add_pop(n, per_sample(n), st["spikes"] / st["elems"] if st["elems"] else 0.0)
            if not st["binary"]:
                info["warnings"].append(f"{n}: output is not binary; counted non-zeros as spikes")
    for n, k in kinds.items():
        if k == "sum":
            st = stats[n]
            add_pop(n, per_sample(n), st["spikes"] / st["elems"] if st["elems"] else 0.0)
    pop_kind = {p["name"]: True for p in b_pops}
    mm_operands = set()
    for r in tr.mm.values():
        mm_operands |= r["a_srcs"] | r["b_srcs"]
    per_root = max(1, calls_root * B) * (T if multi_step or run is not None else 1)
    # coproc: a product read only by other products (Q K^T feeding (Q K^T) V) stays inside the co-processor: no
    # population, its operations are counted with the product that reads it
    internal, extra_ops = set(), defaultdict(float)
    if attention == "coproc":
        for r in tr.mm.values():
            readers = [q for q in tr.mm.values() if r["name"] in (q["a_srcs"] | q["b_srcs"])]
            if readers and not down.get(r["name"]):
                internal.add(r["name"])
                for q in readers:
                    extra_ops[q["name"]] += r["and_ops"] / len(readers)
    if attention != "excluded":
        for r in tr.mm.values():
            if r["name"] in internal:
                continue
            n_and, n_out = r["and_elems"] / per_root, r["out_elems"] / per_root
            if attention == "andgate" and r.get("kind") != "hadamard":      # a Hadamard product is its AND neurons
                add_pop(r["name"] + ":and", n_and, r["and_ops"] / max(r["and_elems"], 1.0))
                pop_kind[r["name"] + ":and"] = True
            add_pop(r["name"], n_out, r["out_nonzero"] / max(r["out_elems"], 1.0))
            pop_kind[r["name"]] = True
            kinds[r["name"]] = "mm"
            if attention == "coproc":
                b_pops[-1]["coproc_ops"] = (r["and_ops"] + extra_ops[r["name"]]) / max(r["out_nonzero"], 1.0)

    mods_by_name = {name: m for m, name in tr.names.items()}

    def pool_geom(n):
        """(kernel h, kernel w, stride h, stride w): from the module when it has them (MaxPool / AvgPool), else
        from the shapes (adaptive pooling)."""
        st, m = stats[n], mods_by_name.get(n)
        ish, osh = st["in_shape"], st["out_shape"]
        Hi, Wi = ish[-2], ish[-1]
        Ho, Wo = osh[-2], osh[-1]
        if m is not None and hasattr(m, "kernel_size") and not type(m).__name__.startswith("Adaptive"):
            ks = _pair(m.kernel_size)
            sd = _pair(m.stride if m.stride is not None else m.kernel_size)
            return ks[0], ks[1], sd[0], sd[1]
        kh, kw = max(1, math.ceil(Hi / Ho)), max(1, math.ceil(Wi / Wo))
        return kh, kw, Hi // max(Ho, 1), Wi // max(Wo, 1)

    # spike pooling (input from a population) -> OR population; current pooling -> folded later
    for n, k in kinds.items():
        if k != "pool" or not stats[n]["calls"]:
            continue
        srcs = up(n)
        if all(s == "input" or kinds.get(s) in ("neuron", "pool") for s in srcs):
            st = stats[n]
            add_pop(n, per_sample(n), st["spikes"] / st["elems"] if st["elems"] else 0.0)
            pop_kind[n] = True
            kh, kw, _, _ = pool_geom(n)
            for s in srcs:
                b_projs.append(dict(src=s, dst=n, kind="conv", fanout=sizes[n] * kh * kw / max(sizes[s], 1),
                                    weight_bits=1))

    def targets(w):
        """Populations a weight layer drives, with the pools folded on the way: [(dst, [pool nodes])]."""
        res = []
        for d in down.get(w, ()):
            if d in pop_kind and kinds.get(d) in ("neuron", "sum"):
                res.append((d, []))
            elif kinds.get(d) == "pool" and d not in pop_kind:
                for d2, pools in targets(d):
                    res.append((d2, [d] + pools))
            elif kinds.get(d) == "weight":
                res.append((None, []))                              # weight -> weight: graded population
        return res

    graded = {}
    for n, k in kinds.items():
        if k != "weight" or not stats[n]["calls"]:
            continue
        tg = targets(n)
        # graded output population when the layer feeds a weight layer, nothing, or several populations (computed
        # once, then copied one-to-one, instead of duplicating its synapses per consumer)
        if not tg or any(d is None for d, _ in tg) or len({d for d, _ in tg}) > 1:
            gname = n + ":out"
            last = not tg and n not in mm_operands
            add_pop(gname, per_sample(n), 0.0 if last else 1.0)
            pop_kind[gname] = True
            graded[n] = gname

    def weight_srcs(w):
        out = set()
        for s in up(w):
            if s in pop_kind or s == "input":
                out.add(s)
            elif kinds.get(s) == "weight":
                out.add(graded.get(s, s + ":out"))
            elif kinds.get(s) == "pool":                            # pooled currents feeding a weight layer
                out |= weight_srcs(s)
        return out

    mods = {name: m for m, name in tr.names.items()}
    for n, k in kinds.items():
        if k != "weight" or not stats[n]["calls"]:
            continue
        m, st = mods[n], stats[n]
        ish, osh = st["in_shape"], st["out_shape"]
        tg = targets(n) if n not in graded else [(graded[n], [])]
        for dst, pools in tg:
            if dst is None:
                continue
            if type(m).__name__ in ("Conv1d", "Conv2d"):
                ks = _pair(m.kernel_size) if type(m).__name__ == "Conv2d" else (m.kernel_size[0], 1)
                st_ = _pair(m.stride) if type(m).__name__ == "Conv2d" else (m.stride[0], 1)
                cin = m.in_channels // m.groups
                kh, kw = ks
                for p in pools:                                       # pooled currents: larger effective kernel
                    ph, pw, sh, sw = pool_geom(p)
                    kh, kw = kh + (ph - 1) * st_[0], kw + (pw - 1) * st_[1]
                    st_ = (st_[0] * sh, st_[1] * sw)
                pos = sizes[dst] / m.out_channels
                syn = sizes[dst] * cin * kh * kw
                shared = dict(shared_weights=True, share_group=max(1, int(round(pos))), axon_fanin=cin)
                kind = "conv"
            else:
                din, dout = m.in_features, m.out_features
                N = max(1, int(round(sizes[dst] / dout)))
                syn = N * din * dout
                shared = dict(shared_weights=True, share_group=N, axon_fanin=din) if N > 1 else {}
                kind = "conv" if N > 1 else "dense"
            if pools:
                info["warnings"].append(f"{n}: pooling of currents {pools} folded into the weight layer (approximation)")
            srcs = weight_srcs(n)
            if kind == "dense":
                # a reduction between the sources and the layer (e.g. a mean over tokens before the classifier head):
                # the layer reads in_elems values, not every source neuron. Cost the reduction as a graded population
                # of in_elems neurons, each fed by its share of the source elements (fan-out 1), and the layer from it
                # (release gap 48, E30: dense from every token made the head bind the step).
                in_elems = sizes[dst] / dout * din
                tot = sum(sizes[x] for x in srcs)
                if tot > 1.5 * in_elems and in_elems >= 1:
                    rname = n + ":in"
                    if rname not in sizes:
                        add_pop(rname, in_elems, 1.0)
                        pop_kind[rname] = True
                        for s in srcs:
                            b_projs.append(dict(src=s, dst=rname, kind="conv", fanout=1.0, weight_bits=weight_bits))
                        info["warnings"].append(f"{n}: input is a reduction of {', '.join(sorted(srcs))} "
                                                f"({tot:.0f} -> {in_elems:.0f} elements); costed as a graded population "
                                                f"{rname} (fan-out 1 from each source element)")
                    srcs = {rname}
            for s in srcs:
                frac = sizes[s] / max(1, sum(sizes[x] for x in srcs))
                b_projs.append(dict(src=s, dst=dst, kind=kind, fanout=syn * frac / max(sizes[s], 1),
                                    weight_bits=weight_bits, **shared))
    def pops_of(nodes, seen=()):
        """Populations behind a set of traced nodes (through pass-through nodes and graded weight outputs)."""
        out = set()
        for s in nodes:
            if s in pop_kind or s == "input":
                out.add(s)
            elif kinds.get(s) == "weight":
                out.add(graded.get(s, s + ":out"))
            elif s not in seen and kinds.get(s) in ("pass", "pool"):
                out |= pops_of(up(s), seen + (s,))
        return out

    if attention == "andgate":
        for r in tr.mm.values():
            if r.get("kind") == "hadamard":            # each operand element drives the AND neuron(s) of its outputs
                h = r["name"]
                for srcs in (r["a_srcs"], r["b_srcs"]):
                    ps = pops_of(srcs)
                    tot = sum(sizes[x] for x in ps)
                    for s in ps:
                        b_projs.append(dict(src=s, dst=h, kind="conv", fanout=sizes[h] / max(tot, 1), weight_bits=1))
                continue
            a_n, s_n = r["name"] + ":and", r["name"]
            m, k, n = r["mkn"]
            for srcs, kind, per in ((r["a_srcs"], "conv", n), (r["b_srcs"], "sparse", m)):
                ps = pops_of(srcs)
                tot = sum(sizes[x] for x in ps)
                for s in ps:          # every operand element reaches `per` AND neurons
                    b_projs.append(dict(src=s, dst=a_n, kind=kind, fanout=sizes[a_n] * sizes[s] / max(tot, 1)
                                        / max(sizes[s], 1), weight_bits=1))
            b_projs.append(dict(src=a_n, dst=s_n, kind="conv", fanout=1.0, weight_bits=1))
    if attention == "coproc":
        byname = {p["name"]: p for p in b_pops}
        for r in tr.mm.values():
            if r["name"] in internal:
                continue
            ops_in = set(r["a_srcs"] | r["b_srcs"])
            while ops_in & internal:          # operands of the internal products it reads
                i = (ops_in & internal).pop()
                ops_in = (ops_in - {i}) | tr.mm[i]["a_srcs"] | tr.mm[i]["b_srcs"]
            operands = sorted(pops_of(ops_in))
            for s in operands:                # every operand spike is sent to the co-processor once
                byname[s]["coproc_msgs"] = byname[s].get("coproc_msgs", 0.0) + 1.0
            byname[r["name"]]["coproc_inputs"] = operands     # result depends on the operands (pipelined depth)
    # neurons / accumulators fed directly by populations (or by a graded weight output): one-to-one
    for n, k in kinds.items():
        if k in ("neuron", "sum") and n in pop_kind:
            for s in up(n):
                src = graded.get(s) if kinds.get(s) == "weight" else s
                if src in pop_kind or src == "input":
                    if sizes[src] == sizes[n]:
                        b_projs.append(dict(src=src, dst=n, kind="one_to_one", fanout=1.0, weight_bits=weight_bits))
                    else:                     # a reduction (e.g. sum over tokens) or a broadcast between the two
                        b_projs.append(dict(src=src, dst=n, kind="conv", fanout=max(1.0, sizes[n] / max(sizes[src], 1)),
                                            weight_bits=weight_bits))
                        info["warnings"].append(f"{src} -> {n}: sizes differ ({sizes[src]:.0f} vs {sizes[n]:.0f}); "
                                                "a reduction / broadcast between them is costed as local fan-out "
                                                f"{max(1.0, sizes[n] / max(sizes[src], 1)):.3g}")
    n_un = sum(1 for u in info["unmapped_ops"] if not u.get("mapped") and u.get("kind") != "nonspiking")
    if n_un:
        info["warnings"].append(f"{n_un} product(s) of two traced tensors (e.g. attention) are not mapped; cost them "
                                "explicitly (info['unmapped_ops'] has measured op counts) or use attention='andgate'")
    # populations in dataflow order (the mapper places them along the snake path in graph order)
    def rank(pname):
        if pname == "input":
            return -1
        base = pname.split(":")[0]
        return tr.order.get(base, len(tr.order)) + (0.5 if pname.endswith(":out") else -0.25 if pname.endswith(":in") else 0.0)
    b_pops.sort(key=lambda p: rank(p["name"]))
    used = {p["src"] for p in b_projs} | {p["dst"] for p in b_projs}
    b_pops = [p for p in b_pops if p["name"] in used or p["name"] == "input" or p.get("coproc_msgs") or p.get("coproc_ops")]
    g = graph_mod.from_dict(dict(name=name or type(model).__name__, steps_per_inference=T, populations=b_pops,
                                 projections=b_projs))
    info["rates_t"] = _rates_t(tr, stats, g, T, multi_step, run is not None)
    return g, info


def _rates_t(tr, stats, g, T, multi_step, custom_run):
    """Measured rate of each spiking population (neurons, pools of spikes, input) per SNN step t (mean over the
    batch): {pop: [rate_0, ..., rate_{T-1}]}. Single-step models: call c is step c mod T; multi-step: the leading
    dimension is time. Empty with a custom `run` (the step of a call is unknown)."""
    import numpy as np
    if custom_run:
        return {}
    out = {}
    for pname in g.populations:
        st = tr.input_stats if pname == "input" else stats.get(pname)
        calls = (st or {}).get("per_call")
        if not calls:
            continue
        cnt, el = np.zeros(T), np.zeros(T)
        for c, (counts, width) in enumerate(calls):
            if multi_step:
                if len(counts) != T:
                    break
                cnt += counts; el += width
            else:
                cnt[c % T] += counts.sum(); el[c % T] += width * len(counts)
        else:
            out[pname] = (cnt / np.maximum(el, 1)).tolist()
    return out
