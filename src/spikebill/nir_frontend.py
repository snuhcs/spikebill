"""NIR graph -> spikebill population graph (and funcsim populations/projections for activity).

Populations: neuron nodes (IF, LIF, CubaLIF, LI, CubaLI, Threshold) and the graph's Input node (spike sources).
Projections: the weight node(s) on the path between two populations, through shape-only nodes (Flatten, Scale,
Delay). Synapse counts are exact: dense layers count non-zero weights; conv / pooling layers count valid kernel taps
with non-zero weight (padding and stride included). A path through two weight nodes (e.g. Conv2d -> SumPool2d) is
composed exactly by multiplying the two (sparse) connectivity matrices.
Weight precision is not in NIR: `weight_bits` is a parameter.
"""
import numpy as np
import nir
import scipy.sparse as sp

from . import graph as graph_mod
from . import funcsim

NEURON_TYPES = (nir.IF, nir.LIF, nir.CubaLIF, nir.LI, nir.CubaLI, nir.Threshold)
PASS_TYPES = (nir.Flatten, nir.Scale, nir.Delay)
WEIGHT_TYPES = (nir.Affine, nir.Linear, nir.Conv2d, nir.Conv1d, nir.SumPool2d, nir.AvgPool2d)


def _size(node):
    for k in ("v_threshold", "threshold", "tau", "tau_mem", "r"):
        if hasattr(node, k) and getattr(node, k) is not None:
            return int(np.prod(np.shape(getattr(node, k))) or 1)
    t = node.output_type.get("output") if node.output_type else None
    return int(np.prod(t)) if t is not None else 0


def _shape_out(node):
    t = node.output_type.get("output") if node.output_type else None
    return tuple(int(x) for x in t) if t is not None else None


def _shape_in(node):
    t = node.input_type.get("input") if node.input_type else None
    return tuple(int(x) for x in t) if t is not None else None


def _pair(x):
    x = np.atleast_1d(np.asarray(x)).astype(int)
    return (int(x[0]), int(x[0])) if len(x) == 1 else (int(x[0]), int(x[1]))


def _window_matrix(in_shape, out_c, kernel, stride, padding, dilation, weight_of):
    """Exact connectivity of a 2D window operator on a (C, H, W) input (channel-first, row-major indices).

    weight_of(co, ci, ky, kx) -> weight array (broadcast) and the input channels each output channel reads."""
    C, H, W = in_shape
    (kh, kw), (sh, sw), (ph, pw), (dh, dw) = _pair(kernel), _pair(stride), _pair(padding), _pair(dilation)
    Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    rows, cols, vals = [], [], []
    yo, xo = np.meshgrid(np.arange(Ho), np.arange(Wo), indexing="ij")
    yo, xo = yo.ravel(), xo.ravel()
    for co in range(out_c):
        for ci, ky, kx, w in weight_of(co):
            if w == 0:
                continue
            yi = yo * sh - ph + ky * dh
            xi = xo * sw - pw + kx * dw
            ok = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
            rows.append(co * Ho * Wo + (yo * Wo + xo)[ok])
            cols.append(ci * H * W + (yi * W + xi)[ok])
            vals.append(np.full(int(ok.sum()), w, dtype=float))
    n_out, n_in = out_c * Ho * Wo, C * H * W
    if not rows:
        return sp.csr_matrix((n_out, n_in)), (out_c, Ho, Wo)
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_out, n_in))
    return M, (out_c, Ho, Wo)


def _conv_matrix(weight, in_shape, stride, padding, dilation=1, groups=1):
    w = np.asarray(weight, dtype=float)               # (Co, Ci/groups, kh, kw)
    Co, Cig, kh, kw = w.shape
    per_out = Co // groups

    def weight_of(co):
        g0 = (co // per_out) * Cig
        for ci in range(Cig):
            for ky in range(kh):
                for kx in range(kw):
                    yield g0 + ci, ky, kx, w[co, ci, ky, kx]
    return _window_matrix(in_shape, Co, (kh, kw), stride, padding, dilation, weight_of)


def _pool_matrix(in_shape, kernel, stride, padding, avg):
    kh, kw = _pair(kernel)
    val = 1.0 / (kh * kw) if avg else 1.0

    def weight_of(co):
        for ky in range(kh):
            for kx in range(kw):
                yield co, ky, kx, val
    return _window_matrix(in_shape, in_shape[0], (kh, kw), stride, padding, 1, weight_of)


def _weight_matrix(node, in_shape):
    """(sparse matrix n_out x n_in, out_shape, kind)."""
    if isinstance(node, (nir.Affine, nir.Linear)):
        Wm = np.asarray(node.weight, dtype=float)
        return sp.csr_matrix(Wm), (Wm.shape[0],), "dense"
    if isinstance(node, nir.Conv2d):
        w = np.asarray(node.weight)
        shp = in_shape if in_shape and len(in_shape) == 3 else (w.shape[1] * int(node.groups),) + tuple(node.input_shape)
        if isinstance(node.padding, str):
            raise NotImplementedError("string padding")
        M, out = _conv_matrix(w, shp, node.stride, node.padding, node.dilation, int(node.groups))
        return M, out, "conv"
    if isinstance(node, (nir.SumPool2d, nir.AvgPool2d)):
        M, out = _pool_matrix(in_shape, node.kernel_size, node.stride, node.padding, isinstance(node, nir.AvgPool2d))
        return M, out, "conv"
    raise NotImplementedError(type(node).__name__)


def _paths(g, start):
    """Paths from a population node to the next population nodes: list of (dst, [weight nodes])."""
    out, stack = [], [(start, [])]
    succ = {}
    for a, b in g.edges:
        succ.setdefault(a, []).append(b)
    while stack:
        node, ops = stack.pop()
        for nxt in succ.get(node, []):
            n = g.nodes[nxt]
            if isinstance(n, NEURON_TYPES):
                out.append((nxt, ops))
            elif isinstance(n, WEIGHT_TYPES):
                stack.append((nxt, ops + [nxt]))
            elif isinstance(n, PASS_TYPES):
                stack.append((nxt, ops))
            # Output nodes end the path (no synapses on chip)
    return out


def read(path):
    """nir.read, falling back to no type check for files written by older NIR versions."""
    try:
        return nir.read(path)
    except ValueError:
        return nir.read(path, type_check=False)


def _input_shape(n):
    """Shape of a NIR Input node without a leading batch dimension of 1 (exporters such as snnTorch keep it)."""
    shp = tuple(int(x) for x in n.input_type["input"])
    if len(shp) in (2, 4) and shp[0] == 1:
        shp = shp[1:]
    return shp


def infer_shapes(g):
    """Output shape of every node, propagated from the Input node(s) in topological order (neuron nodes keep their
    input shape; weight nodes from their connectivity; Flatten flattens). Needed when neuron parameters are scalars."""
    pred = {}
    for a, b in g.edges:
        pred.setdefault(b, []).append(a)
    shapes, mats = {}, {}
    todo = list(g.nodes)
    while todo:
        progressed = False
        for k in list(todo):
            n = g.nodes[k]
            if isinstance(n, nir.Input):
                shapes[k] = _input_shape(n)
            elif all(p in shapes for p in pred.get(k, [])) and pred.get(k):
                ins = shapes[pred[k][0]]
                if isinstance(n, WEIGHT_TYPES):
                    M, out, _ = _weight_matrix(n, ins)
                    mats[k] = M
                    shapes[k] = tuple(int(x) for x in out)
                elif isinstance(n, nir.Flatten):
                    shapes[k] = (int(np.prod(ins)),)
                else:
                    shapes[k] = ins
            else:
                continue
            todo.remove(k)
            progressed = True
        if not progressed:
            raise ValueError(f"cannot infer shapes of {todo} (cycle or missing input)")
    return shapes


def from_nir(g, weight_bits=8, steps_per_inference=1, name="nir_model"):
    """Returns (spikebill Graph, info) where info[(src, dst)] holds the exact connectivity matrix (n_dst x n_src)."""
    if isinstance(g, str):
        g = read(g)
    pops = {}
    try:
        inferred = infer_shapes(g)
    except ValueError:
        inferred = {}
    shapes = {}
    for k, n in g.nodes.items():
        if isinstance(n, nir.Input):
            shp = _input_shape(n)
            pops[k] = int(np.prod(shp)); shapes[k] = shp
        elif isinstance(n, NEURON_TYPES):
            own = _size(n)
            shp = inferred.get(k) or _shape_out(n) or (own,)
            if own > 1 and int(np.prod(shp)) != own:
                raise ValueError(f"{k}: parameter size {own} does not match inferred shape {shp}")
            pops[k] = int(np.prod(shp)); shapes[k] = shp
    projs, info = [], {}
    for src in pops:
        for dst, ops in _paths(g, src):
            if not ops:
                M = sp.identity(pops[src], format="csr")
                kind = "one_to_one"
            else:
                M, shp, kind, bvec = None, shapes[src], None, None
                for o in ops:
                    Mi, shp, ki = _weight_matrix(g.nodes[o], shp)
                    M = Mi if M is None else Mi @ M
                    kind = ki if kind in (None, ki) else "sparse"
                    bvec = Mi @ bvec if bvec is not None else None        # earlier biases pass through this layer
                    bias = getattr(g.nodes[o], "bias", None)
                    if bias is not None and np.any(np.asarray(bias) != 0):
                        bb = np.asarray(bias, dtype=float).reshape(-1)
                        n_out = Mi.shape[0]
                        bb = np.repeat(bb, n_out // len(bb)) if len(bb) != n_out else bb   # conv: per channel
                        bvec = bb if bvec is None else bvec + bb
                if bvec is not None:
                    info.setdefault("_bias", {})
                    info["_bias"][dst] = info["_bias"].get(dst, 0.0) + bvec
            if M.shape != (pops[dst], pops[src]):
                raise ValueError(f"{src}->{dst}: connectivity {M.shape} does not match sizes ({pops[dst]}, {pops[src]})")
            nnz = int((M != 0).sum())
            shared = kind == "conv"
            n_weights = int(sum(np.count_nonzero(np.asarray(g.nodes[o].weight)) for o in ops
                                if hasattr(g.nodes[o], "weight"))) if shared else None
            projs.append(dict(src=src, dst=dst, kind=kind if kind != "dense" or nnz >= 0.5 * M.size else "sparse",
                              fanout=nnz / pops[src], weight_bits=weight_bits, shared_weights=shared,
                              n_weights=n_weights))
            info[(src, dst)] = M
    G = graph_mod.from_dict(dict(name=name, steps_per_inference=steps_per_inference,
                                 populations=[dict(name=k, size=v) for k, v in pops.items()], projections=projs))
    return G, info


def to_funcsim(g, info, dt=None):
    """funcsim populations / projections from a NIR graph (IF/LIF/Threshold; LIF discretised with Euler step dt):
    leak = 1 - dt/tau, input gain = dt * r / tau. NIR carries no time step: with dt=None it is taken per node as
    tau / r (input gain 1), the convention of discrete-time exporters such as snnTorch (tau = dt/(1-beta), r = tau/dt)."""
    if isinstance(g, str):
        g = read(g)
    fpops, gain = [], {}
    try:
        shapes = infer_shapes(g)
    except ValueError:
        shapes = {}

    def size_of(k, n):
        own = _size(n)
        return int(np.prod(shapes[k])) if (own <= 1 and k in shapes) else own
    for k, n in g.nodes.items():
        if isinstance(n, nir.Input):
            fpops.append(funcsim.FPop(k, int(np.prod(n.input_type["input"])), threshold=0.5))
        elif isinstance(n, (nir.IF, nir.LIF, nir.Threshold)):
            thr = float(np.mean(getattr(n, "v_threshold", getattr(n, "threshold", 1.0))))
            leak, g_in = 1.0, 1.0
            if isinstance(n, nir.LIF):
                tau, r = float(np.mean(n.tau)), float(np.mean(n.r))
                d = dt if dt is not None else tau / r
                leak, g_in = 1.0 - d / tau, d * r / tau
            elif isinstance(n, nir.IF) and dt is not None:
                g_in = dt * float(np.mean(n.r))
            fpops.append(funcsim.FPop(k, size_of(k, n), thr, leak_decay=leak))
            gain[k] = g_in
    fprojs = []
    bias = info.get("_bias", {})
    for p in fpops:
        if p.name in bias:
            p.bias = gain.get(p.name, 1.0) * np.asarray(bias[p.name], dtype=float).reshape(-1)
    for key, M in info.items():
        if not isinstance(key, tuple):
            continue
        s, d = key
        C = M.tocoo()
        fprojs.append(funcsim.FProj(s, d, C.col, C.row, C.data * gain.get(d, 1.0)))
    return fpops, fprojs


def export_snntorch(net, sample):
    """snnTorch -> NIR graph. snnTorch's exporter fails NIR 1.0's type inference on the batch dimension; inference is
    switched off during export (this front end infers shapes itself). Use the returned graph object directly
    (nir.write cannot store the exporter's scalar LIF parameters)."""
    from snntorch.export_nir import export_to_nir
    _infer = nir.NIRGraph.infer_types
    nir.NIRGraph.infer_types = lambda self: None
    try:
        return export_to_nir(net, sample)
    finally:
        nir.NIRGraph.infer_types = _infer
