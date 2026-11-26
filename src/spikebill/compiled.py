"""Compiled cost model: build once, evaluate many steps fast (batched, optionally on GPU).

Everything the per-step cost needs is linear in the per-population spike counts (messages, hops, synaptic events,
per-core loads, link-load difference arrays); only maxima are not. `compile_model` turns (graph, chip, mapping) into
matrices with one column per population; `evaluate` multiplies them by a (steps x populations) spike matrix and
takes the maxima, for all steps at once. Same model as `cost.step_cost` (tested for equality).
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import traffic


@dataclass
class CompiledModel:
    pops: list               # population order (columns)
    sizes: np.ndarray        # (P,)
    core_out: sp.csr_matrix  # (C, P) messages sent per core per unit population spike
    core_in: sp.csr_matrix   # (C, P) messages received
    core_syn: sp.csr_matrix  # (C, P) synaptic events
    core_synt: sp.csr_matrix # (C, P) synaptic events x t_synop (per synapse type)
    fired: sp.csr_matrix     # (C, P) share of a population's spikes on each core
    cnt: sp.csr_matrix       # (C, P) share of a population's neurons on each core
    neurons: np.ndarray      # (C,) neurons per core
    tot: np.ndarray          # (5, P) messages, planar hops, chip hops, synaptic energy, vertical hops per unit spike
    diff: sp.csr_matrix      # (n_diff, P) link-load difference entries (E, W, N, S, U, D)
    path: np.ndarray         # (P,) longest path latency (s) of a population's messages
    mesh: tuple
    chip: object
    n_tiles: int
    n_chips: int
    n_cores: int
    layers: int = 1
    cop: np.ndarray = None   # (2, P) co-processor operations and messages per unit spike, or None
    cop_units: np.ndarray = None   # (U, P) co-processor operations per unit spike by unit (chip coproc_units "per_unit")


def _diff_shape(W, H, Z):
    """Flat layout of the difference arrays: E, W, N, S (Z, H+1, W+1) then U, D (Z+1, H+1, W+1)."""
    plane = Z * (H + 1) * (W + 1)
    vert = (Z + 1) * (H + 1) * (W + 1)
    return plane, vert, 4 * plane + 2 * vert


class _Recorder:
    """Stands in for the difference arrays in traffic._route_tree: records (flat index, value) of every update
    instead of filling mesh-sized arrays (one per tree source core would cost O(cores x mesh))."""

    class _Arr:
        def __init__(self, rec, off, shape):
            self.rec, self.off, self.shape = rec, off, shape

        def __getitem__(self, idx):
            return 0.0

        def __setitem__(self, idx, value):
            z, y, x = idx
            self.rec.idx.append(self.off + (z * self.shape[1] + y) * self.shape[2] + x)
            self.rec.val.append(float(value))

    def __init__(self, W, H, Z):
        plane, vert, _ = _diff_shape(W, H, Z)
        self.idx, self.val = [], []
        self.arrs = {k: self._Arr(self, i * plane, (Z, H + 1, W + 1)) for i, k in enumerate("EWNS")}
        self.arrs.update({k: self._Arr(self, 4 * plane + i * vert, (Z + 1, H + 1, W + 1)) for i, k in enumerate("UD")})

    def __getitem__(self, k):
        return self.arrs[k]


def _route_entries(x1, y1, z1, x2, y2, z2, w, W, H, Z):
    """Difference-array entries (flat index, value) of XYZ unicast routes."""
    plane, vert, _ = _diff_shape(W, H, Z)
    idx, val = [], []

    def add(d, zs, ys, xs, v):
        off = d * plane if d < 4 else 4 * plane + (d - 4) * vert
        idx.append(off + (zs * (H + 1) + ys) * (W + 1) + xs); val.append(v)
    e = x2 > x1
    add(0, z1[e], y1[e], x1[e], w[e]); add(0, z1[e], y1[e], x2[e], -w[e])
    e = x2 < x1
    add(1, z1[e], y1[e], x2[e], w[e]); add(1, z1[e], y1[e], x1[e], -w[e])
    e = y2 > y1
    add(2, z1[e], y1[e], x2[e], w[e]); add(2, z1[e], y2[e], x2[e], -w[e])
    e = y2 < y1
    add(3, z1[e], y2[e], x2[e], w[e]); add(3, z1[e], y1[e], x2[e], -w[e])
    e = z2 > z1
    add(4, z1[e], y2[e], x2[e], w[e]); add(4, z2[e], y2[e], x2[e], -w[e])
    e = z2 < z1
    add(5, z2[e], y2[e], x2[e], w[e]); add(5, z1[e], y2[e], x2[e], -w[e])
    return np.concatenate(idx), np.concatenate(val)


def _tiles(cores, weights, tile_of, xyz):
    """Group cores by tile: (tile coordinates (n, 3), summed weights, core count per tile)."""
    t = tile_of[cores]
    u, inv = np.unique(t, return_inverse=True)
    w = np.bincount(inv, weights=weights)
    cnt = np.bincount(inv)
    first = np.zeros(len(u), dtype=np.int64)
    first[inv[::-1]] = np.arange(len(cores))[::-1]
    return xyz[cores[first]], w, cnt


def _legs(chip, send, route, W, H, Z, nd, j, tot, d_idx, d_col, d_val):
    """Hops, chip hops, vertical hops and link-load entries of chip_broadcast legs (sent and router-copied)."""
    acc = np.zeros(nd)
    for A, B, w in (send, route):
        if not len(w):
            continue
        tot[1, j] += (w * (np.abs(B[:, 0] - A[:, 0]) + np.abs(B[:, 1] - A[:, 1]))).sum()
        tot[2, j] += (w * (np.abs(B[:, 0] // chip.tiles_x - A[:, 0] // chip.tiles_x)
                           + np.abs(B[:, 1] // chip.tiles_y - A[:, 1] // chip.tiles_y))).sum()
        tot[4, j] += (w * np.abs(B[:, 2] - A[:, 2])).sum()
        i_, v_ = _route_entries(A[:, 0], A[:, 1], A[:, 2], B[:, 0], B[:, 1], B[:, 2], w, W, H, Z)
        acc += np.bincount(i_, weights=v_, minlength=nd)
    nz = np.flatnonzero(acc)
    d_idx.append(nz); d_col.append(np.full(len(nz), j)); d_val.append(acc[nz])


def _nonlocal(graph, chip, mapping, proj, j, rows, d_idx, d_col, d_val, tot, path, xyz, tile_of, mesh, nd,
              t_hop, t_vhop, chunk=4_000_000):
    """Dense / random projection without explicit flows, exactly and without core pairs: every spike reaches
    `reach` of the k_d destination cores uniformly, so per-core loads are analytic, and routes depend only on
    (source tile, destination tile) pairs."""
    W, H, Z = mesh
    Pp, Qp = mapping.pops[proj.src], mapping.pops[proj.dst]
    C = mapping.n_cores
    k_d = Qp.n_cores
    reach = traffic.cores_reached(graph, mapping, proj)
    mw = reach / k_d                                              # messages to each dst core per spike
    f = graph.fanout(proj)
    share_s, share_d = Pp.share, Qp.counts / Qp.counts.sum()
    tree = chip.multicast == "tree" and mw >= 0.999

    def put(key, cores, v):
        rows[key][0].append(np.asarray(cores)); rows[key][1].append(np.full(len(cores), j))
        rows[key][2].append(np.asarray(v, dtype=float))
    put("in", Qp.cores, np.full(k_d, mw))
    put("syn", Qp.cores, f * share_d)
    put("synt", Qp.cores, f * share_d * chip.get("t_synop", proj.synapse))
    tot[3, j] += f * chip.get("e_synop", proj.synapse)
    st, sw_, _ = _tiles(Pp.cores, share_s, tile_of, xyz)
    dt, _, dcnt = _tiles(Qp.cores, np.ones(k_d), tile_of, xyz)
    if chip.multicast == "chip_broadcast":
        # per source core: mw per destination core on its own chip + one packet per remote chip k, needed with
        # probability p_k = 1 - (1 - mw)^(destination cores on k)
        ids = traffic.chip_ids(np.vstack([xyz[Pp.cores], xyz[Qp.cores]]), chip)
        cs, cd = ids[:len(Pp.cores)], ids[len(Pp.cores):]
        n_k = np.bincount(cd, minlength=ids.max() + 1)
        with np.errstate(invalid="ignore"):    # mw = 1 (every core reached): log_miss = -inf; chips without
            p_k = np.where(n_k > 0, -np.expm1(n_k * traffic.log_miss(mw)), 0.0)   # destinations stay 0, not NaN
        out = share_s * (mw * n_k[cs] + p_k.sum() - p_k[cs])
        put("out", Pp.cores, out)
        tot[0, j] += out.sum()
        lm = traffic.log_miss(mw)
        step = max(1, chunk // max(1, len(dt)))
        for a in range(0, len(st), step):
            ns = len(st[a:a + step])
            si = np.repeat(np.arange(ns), len(dt))
            A = np.repeat(st[a:a + step], len(dt), axis=0); B = np.tile(dt, (ns, 1))
            sw = np.repeat(sw_[a:a + step], len(dt)); dc = np.tile(dcnt, ns)
            (_, A1, B1, w1), route, (ph, pv) = traffic.broadcast_split(si, A, B, sw, sw * dc * mw, dc * lm, chip)
            _legs(chip, (A1, B1, w1), route, W, H, Z, nd, j, tot, d_idx, d_col, d_val)
            live = sw * dc * mw > 0
            if live.any():
                path[j] = max(path[j], float((ph[live] * t_hop + pv[live] * t_vhop).max()))
        return
    put("out", Pp.cores, share_s * (1.0 if tree else reach))
    if tree:
        for i in range(len(st)):
            if sw_[i] <= 0:
                continue
            rec = _Recorder(W, H, Z)
            h, ch, hv = traffic._route_tree(rec, st[i], dt, float(sw_[i]), chip)
            d_idx.append(np.array(rec.idx, dtype=np.int64)); d_col.append(np.full(len(rec.idx), j))
            d_val.append(np.array(rec.val))
            tot[0, j] += sw_[i]; tot[1, j] += sw_[i] * h; tot[2, j] += sw_[i] * ch; tot[4, j] += sw_[i] * hv
    else:
        tot[0, j] += reach
        acc = np.zeros(nd)
        step = max(1, chunk // max(1, len(dt)))
        for a in range(0, len(st), step):
            s_ = slice(a, a + step)
            ns = len(st[s_])
            x1 = np.repeat(st[s_, 0], len(dt)); y1 = np.repeat(st[s_, 1], len(dt)); z1 = np.repeat(st[s_, 2], len(dt))
            x2 = np.tile(dt[:, 0], ns); y2 = np.tile(dt[:, 1], ns); z2 = np.tile(dt[:, 2], ns)
            w = np.repeat(sw_[s_], len(dt)) * np.tile(dcnt, ns) * mw
            hops = np.abs(x2 - x1) + np.abs(y2 - y1)
            vh = np.abs(z2 - z1)
            chops = np.abs(x2 // chip.tiles_x - x1 // chip.tiles_x) + np.abs(y2 // chip.tiles_y - y1 // chip.tiles_y)
            tot[1, j] += (w * hops).sum(); tot[2, j] += (w * chops).sum(); tot[4, j] += (w * vh).sum()
            i_, v_ = _route_entries(x1, y1, z1, x2, y2, z2, w, W, H, Z)
            acc += np.bincount(i_, weights=v_, minlength=nd)
            live = w > 0
            if live.any():
                path[j] = max(path[j], float((hops[live] * t_hop + vh[live] * t_vhop).max()))
        nz = np.flatnonzero(acc)
        d_idx.append(nz); d_col.append(np.full(len(nz), j)); d_val.append(acc[nz])
    if tree and len(st) and len(dt):
        hp = np.abs(dt[None, :, 0] - st[:, None, 0]) + np.abs(dt[None, :, 1] - st[:, None, 1])
        vz = np.abs(dt[None, :, 2] - st[:, None, 2])
        path[j] = max(path[j], float((hp * t_hop + vz * t_vhop).max()))


def _sparse_sum(idx, v):
    """Sum v over equal idx: (keys, sums) with zero sums dropped. Same sums as np.bincount (elements added in the
    same order per key), without scanning a dense array of length max(idx) (compile_model's per-projection cost)."""
    keys, inv = np.unique(idx, return_inverse=True)
    acc = np.bincount(inv, weights=v, minlength=len(keys))
    nz = acc != 0
    return keys[nz], acc[nz]


def _tree_entries(src, ws, pair_src, dst, W, H, Z, tx, ty):
    """XYZ multicast trees of many sources at once (the trees of traffic._route_tree, vectorized over a projection).
    src: (n, 3) source core positions, ws: (n,) load per tree link, pair_src: source index of each destination row,
    dst: (m, 3) destination positions. Returns (flat difference-array indices, values, planar links, chip-crossing
    links, vertical links per source)."""
    plane, vert, _ = _diff_shape(W, H, Z)
    n = len(src)
    sx, sy, sz = src[:, 0], src[:, 1], src[:, 2]
    o = np.lexsort((dst[:, 2], dst[:, 1], dst[:, 0], pair_src))
    p, d = pair_src[o], dst[o]
    def starts(*keys):                       # first row of each run of equal keys (rows sorted)
        new = np.zeros(len(p), dtype=bool)
        new[0] = True
        for k in keys:
            new[1:] |= k[1:] != k[:-1]
        return np.flatnonzero(new)
    # per source: X segment in the source row and layer
    a = starts(p)
    ps = p[a]
    xmin = np.minimum(sx[ps], np.minimum.reduceat(d[:, 0], a))
    xmax = np.maximum(sx[ps], np.maximum.reduceat(d[:, 0], a))
    links = np.zeros(n); ch = np.zeros(n); vl = np.zeros(n)
    np.add.at(links, ps, xmax - xmin)
    np.add.at(ch, ps, np.abs(xmin // tx - sx[ps] // tx) + np.abs(xmax // tx - sx[ps] // tx))
    idx, val = [], []
    def put(arr, z, y, x, v, m):
        off, rows = (arr * plane, H + 1) if arr < 4 else (4 * plane + (arr - 4) * vert, H + 1)
        idx.append(off + (z[m] * rows + y[m]) * (W + 1) + x[m]); val.append(v[m])
    w, X, Y, Zs = ws[ps], sx[ps], sy[ps], sz[ps]
    put(0, Zs, Y, X, w, xmax > X); put(0, Zs, Y, xmax, -w, xmax > X)
    put(1, Zs, Y, xmin, w, xmin < X); put(1, Zs, Y, X, -w, xmin < X)
    # per (source, column): Y segment in the source layer
    b = starts(p, d[:, 0])
    pc, c = p[b], d[b, 0]
    ymin = np.minimum(sy[pc], np.minimum.reduceat(d[:, 1], b))
    ymax = np.maximum(sy[pc], np.maximum.reduceat(d[:, 1], b))
    np.add.at(links, pc, ymax - ymin)
    np.add.at(ch, pc, np.abs(ymin // ty - sy[pc] // ty) + np.abs(ymax // ty - sy[pc] // ty))
    w, Y, Zs = ws[pc], sy[pc], sz[pc]
    put(2, Zs, Y, c, w, ymax > Y); put(2, Zs, ymax, c, -w, ymax > Y)
    put(3, Zs, ymin, c, w, ymin < Y); put(3, Zs, Y, c, -w, ymin < Y)
    # per (source, column, row): Z segment
    e = starts(p, d[:, 0], d[:, 1])
    pz, cx, cy = p[e], d[e, 0], d[e, 1]
    zmin = np.minimum(sz[pz], np.minimum.reduceat(d[:, 2], e))
    zmax = np.maximum(sz[pz], np.maximum.reduceat(d[:, 2], e))
    np.add.at(vl, pz, zmax - zmin)
    w, Zs = ws[pz], sz[pz]
    put(4, Zs, cy, cx, w, zmax > Zs); put(4, zmax, cy, cx, -w, zmax > Zs)
    put(5, zmin, cy, cx, w, zmin < Zs); put(5, Zs, cy, cx, -w, zmin < Zs)
    return np.concatenate(idx), np.concatenate(val), links, ch, vl


def compile_model(graph, chip, mapping):
    """Compile graph + mapping into sparse matrices so a step (or a batch of steps) is a few mat-vecs; equal to
    cost.step_cost to floating-point precision (tests)."""
    pops = list(graph.populations)
    col = {k: i for i, k in enumerate(pops)}
    P, C = len(pops), mapping.n_cores
    W, H = mapping.mesh
    Z = getattr(mapping, "layers", 1) or 1
    zc = mapping.z if getattr(mapping, "z", None) is not None else np.zeros(C, dtype=np.int64)
    xyz = np.column_stack([mapping.xy, zc])
    _, _, nd = _diff_shape(W, H, Z)
    t_hop, t_vhop = chip["t_hop"], chip.v("t_vhop")
    rows = {k: ([], [], []) for k in ("out", "in", "syn", "synt")}
    d_idx, d_col, d_val = [], [], []
    tot = np.zeros((5, P))
    path = np.zeros(P)
    tile_of = mapping.tile_of
    for proj in graph.projections:
        j = col[proj.src]
        Pp, Qp = mapping.pops[proj.src], mapping.pops[proj.dst]
        if proj.flows is None and traffic.locality(proj) != "local":
            _nonlocal(graph, chip, mapping, proj, j, rows, d_idx, d_col, d_val, tot, path, xyz, tile_of,
                      (W, H, Z), nd, t_hop, t_vhop)
            continue
        s, d, mw, sw = traffic.flows(graph, mapping, proj)
        share = Pp.share
        w = mw * share[s]                 # messages per pair per unit population spike
        y = sw * share[s]                 # synaptic events per pair per unit population spike
        gs, gd = Pp.cores[s], Qp.cores[d]
        x1, y1, z1 = mapping.xy[gs, 0], mapping.xy[gs, 1], zc[gs]
        x2, y2, z2 = mapping.xy[gd, 0], mapping.xy[gd, 1], zc[gd]
        hops = np.abs(x2 - x1) + np.abs(y2 - y1)
        vh = np.abs(z2 - z1)
        chops = (np.abs(x2 // chip.tiles_x - x1 // chip.tiles_x) + np.abs(y2 // chip.tiles_y - y1 // chip.tiles_y))
        def put(key, cores, v):
            nz, acc = _sparse_sum(cores, v)                       # reduce duplicates before collecting
            rows[key][0].append(nz); rows[key][1].append(np.full(len(nz), j)); rows[key][2].append(acc)
        put("in", gd, w); put("syn", gd, y); put("synt", gd, y * chip.get("t_synop", proj.synapse))
        tot[3, j] += y.sum() * chip.get("e_synop", proj.synapse)
        if chip.multicast == "chip_broadcast":
            if len(s):
                (si, A, B, ws), route, (ph, pv) = traffic.broadcast_split(
                    s, xyz[gs], xyz[gd], share[s], w, traffic.log_miss(mw), chip)
                put("out", gs[si], ws)
                tot[0, j] += ws.sum()
                _legs(chip, (A, B, ws), route, W, H, Z, nd, j, tot, d_idx, d_col, d_val)
                live = w > 0
                if live.any():
                    path[j] = max(path[j], float((ph[live] * t_hop + pv[live] * t_vhop).max()))
            continue
        tree = np.zeros(len(s), dtype=bool)
        if chip.multicast == "tree" and len(s):
            full = np.ones(Pp.n_cores, dtype=bool)
            np.logical_and.at(full, s, mw >= 0.999)
            tree = full[s]
        u = ~tree
        put("out", gs[u], w[u])
        if u.any():
            i_, v_ = _route_entries(x1[u], y1[u], z1[u], x2[u], y2[u], z2[u], w[u], W, H, Z)
            nz, acc = _sparse_sum(i_, v_)
            d_idx.append(nz); d_col.append(np.full(len(nz), j)); d_val.append(acc)
        tot[0, j] += w[u].sum(); tot[1, j] += (w[u] * hops[u]).sum(); tot[2, j] += (w[u] * chops[u]).sum()
        tot[4, j] += (w[u] * vh[u]).sum()
        if tree.any():
            srcs, ps = np.unique(s[tree], return_inverse=True)
            ws = share[srcs].astype(float)
            ok = ws > 0
            if ok.any():
                keep = ok[ps]
                remap = np.cumsum(ok) - 1
                sc_cores = Pp.cores[srcs[ok]]
                i_, v_, h, ch, hv = _tree_entries(xyz[sc_cores], ws[ok], remap[ps[keep]], xyz[gd[tree][keep]],
                                                  W, H, Z, chip.tiles_x, chip.tiles_y)
                d_idx.append(i_); d_col.append(np.full(len(i_), j)); d_val.append(v_)
                rows["out"][0].append(sc_cores); rows["out"][1].append(np.full(len(sc_cores), j))
                rows["out"][2].append(ws[ok])
                wk = ws[ok]
                tot[0, j] += wk.sum(); tot[1, j] += (wk * h).sum(); tot[2, j] += (wk * ch).sum()
                tot[4, j] += (wk * hv).sum()
        live = (w > 0)
        if live.any():
            path[j] = max(path[j], float((hops[live] * t_hop + vh[live] * t_vhop).max()))

    def mat(key):
        r, c, v = rows[key]
        if not r:
            return sp.csr_matrix((C, P))
        return sp.coo_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(C, P)).tocsr()

    fr, cn, fv, cv, neurons = [], [], [], [], np.zeros(C)
    for k, p in mapping.pops.items():
        fr.append(p.cores); cn.append(np.full(p.n_cores, col[k])); fv.append(p.share)
        cv.append(p.counts / p.counts.sum()); neurons[p.cores] += p.counts
    fired = sp.coo_matrix((np.concatenate(fv), (np.concatenate(fr), np.concatenate(cn))), shape=(C, P)).tocsr()
    cnt = sp.coo_matrix((np.concatenate(cv), (np.concatenate(fr), np.concatenate(cn))), shape=(C, P)).tocsr()
    diff = (sp.coo_matrix((np.concatenate(d_val), (np.concatenate(d_idx), np.concatenate(d_col))), shape=(nd, P)).tocsr()
            if d_idx else sp.csr_matrix((nd, P)))
    cop = np.array([[graph.populations[k].coproc_ops for k in pops], [graph.populations[k].coproc_msgs for k in pops]])
    labels = sorted({graph.populations[k].coproc_unit or k for k in pops if graph.populations[k].coproc_ops})
    cop_units = np.zeros((len(labels), len(pops)))
    for j, k in enumerate(pops):
        q = graph.populations[k]
        if q.coproc_ops:
            cop_units[labels.index(q.coproc_unit or k), j] = q.coproc_ops
    return CompiledModel(pops, np.array([graph.populations[k].size for k in pops], float), mat("out"), mat("in"),
                         mat("syn"), mat("synt"), fired, cnt, neurons, tot, diff, path, (W, H), chip,
                         mapping.n_tiles, mapping.n_chips, mapping.n_cores, Z, cop if cop.any() else None,
                         cop_units if cop_units.size else None)


def _link_max(diff_flat, W, H, Z, chip):
    """diff_flat: (T, n_diff) -> busiest tile link, chip link and vertical link per step."""
    T = diff_flat.shape[0]
    plane, vert, _ = _diff_shape(W, H, Z)
    d = diff_flat[:, :4 * plane].reshape(T, 4, Z, H + 1, W + 1)
    v = diff_flat[:, 4 * plane:].reshape(T, 2, Z + 1, H + 1, W + 1)
    xs, ys = np.arange(W + 1), np.arange(H + 1)
    inter_x = ((xs + 1) % chip.tiles_x == 0)[None, None, None, :]
    inter_y = ((ys + 1) % chip.tiles_y == 0)[None, None, :, None]
    lx = np.cumsum(d[:, :2], axis=4)          # E, W along x
    ly = np.cumsum(d[:, 2:], axis=3)          # N, S along y
    ax = (1, 2, 3, 4)
    intra = np.maximum(np.where(inter_x, 0, lx).max(axis=ax), np.where(inter_y, 0, ly).max(axis=ax))
    inter = np.maximum(np.where(inter_x, lx, 0).max(axis=ax), np.where(inter_y, ly, 0).max(axis=ax))
    vmax = np.cumsum(v, axis=2).max(axis=ax)
    return intra, inter, vmax


def evaluate(cm, spikes, active, static=True, chunk=256):
    """spikes, active: (T, P) arrays (population order cm.pops). Returns per-step energy components and times."""
    chip = cm.chip
    S = np.atleast_2d(np.asarray(spikes, float))
    A = np.atleast_2d(np.asarray(active, float))
    out = {k: [] for k in ("synaptic", "neuron", "network", "static", "compute", "network_t", "step",
                           "spikes", "active", "synops", "messages", "hops", "chip_hops")}
    if cm.cop is not None:
        out["coproc"], out["coproc_t"] = [], []
    from .cost import static_power
    p_static = static_power(chip, type("M", (), dict(n_cores=cm.n_cores, n_chips=cm.n_chips))())
    t_bar = chip.barrier(cm.n_tiles, cm.n_chips)
    W, H = cm.mesh
    # cap the (steps x link-difference) block at ~16M values: past a few hundred MB per array the cumulative sums in
    # _link_max slow down ~9x (E9: 1406-chip mesh, 256-step chunks 82 ms/step vs <= 128 steps 9 ms/step)
    chunk = max(1, min(chunk, int(16e6 // max(cm.diff.shape[0], 1))))
    for a in range(0, S.shape[0], chunk):
        s, act = S[a:a + chunk], A[a:a + chunk]
        St, At = s.T, act.T
        upd = cm.neurons[:, None] if chip.neuron_update == "all" else (cm.cnt @ At)
        t_neur = (upd * chip["t_neuron_update"] + (cm.cnt @ At) * chip["t_neuron_active"]
                  + (cm.fired @ St) * chip["t_spike_gen"] + (cm.core_out @ St) * chip["t_spike_out"])
        t_msg = (cm.core_in @ St) * chip["t_msg_in"] + cm.core_synt @ St
        t_core = np.maximum(t_neur, t_msg) if chip.core_timing == "max" else t_neur + t_msg
        t_compute = t_core.max(axis=0)
        intra, inter, vmax = _link_max((cm.diff @ St).T, W, H, cm.layers, chip)
        path = np.where(s > 0, cm.path[None, :], 0).max(axis=1)
        t_network = (np.maximum(np.maximum(intra * chip["t_link"], inter * chip["t_chip_hop"]), vmax * chip.v("t_vlink"))
                     + path)
        if cm.cop is not None:
            ops, cmsg = cm.cop @ St
            if chip.coproc_units == "per_unit" and cm.cop_units is not None:
                t_cop = (cm.cop_units @ St).max(axis=0) * chip.c("t_coproc_op")
            else:
                t_cop = ops * chip.c("t_coproc_op")
            out["coproc"].append(ops * chip.c("e_coproc_op") + cmsg * chip["e_spike_out"]); out["coproc_t"].append(t_cop)
            t_step = np.maximum(np.maximum(t_compute, t_network), t_cop) + t_bar
        else:
            t_step = np.maximum(t_compute, t_network) + t_bar
        tot = cm.tot @ St
        out["synaptic"].append(tot[3])
        n_upd = cm.neurons.sum() if chip.neuron_update == "all" else act.sum(1)
        out["neuron"].append(n_upd * chip["e_neuron_update"] + act.sum(1) * chip["e_neuron_active"]
                             + s.sum(1) * chip["e_spike_gen"])
        out["network"].append(tot[0] * chip["e_spike_out"] + (tot[1] - tot[2]) * chip["e_hop"] + tot[2] * chip["e_chip_hop"]
                              + tot[4] * chip.v("e_vhop"))
        out["static"].append(p_static * t_step if static else np.zeros_like(t_step))
        out["compute"].append(t_compute); out["network_t"].append(t_network); out["step"].append(t_step)
        out["spikes"].append(s.sum(1)); out["active"].append(act.sum(1))
        out["synops"].append(np.asarray(cm.core_syn.sum(axis=0)).ravel() @ St)
        out["messages"].append(tot[0]); out["hops"].append(tot[1]); out["chip_hops"].append(tot[2])
    return {k: np.concatenate(v) for k, v in out.items()}


def step_result(cm, spikes, active):
    """One step from {pop: count} dicts, in the same shape as cost.step_cost (energy, time, counts, core_time)."""
    chip = cm.chip
    S = np.array([[float(spikes.get(k, 0.0)) for k in cm.pops]])
    A = np.array([[float(active.get(k, 0.0)) for k in cm.pops]])
    ev = evaluate(cm, S, A)
    St, At = S.T, A.T
    upd = cm.neurons[:, None] if chip.neuron_update == "all" else (cm.cnt @ At)
    t_neur = (upd * chip["t_neuron_update"] + (cm.cnt @ At) * chip["t_neuron_active"]
              + (cm.fired @ St) * chip["t_spike_gen"] + (cm.core_out @ St) * chip["t_spike_out"])
    t_msg = (cm.core_in @ St) * chip["t_msg_in"] + cm.core_synt @ St
    t_core = (np.maximum(t_neur, t_msg) if chip.core_timing == "max" else t_neur + t_msg)[:, 0]
    time = dict(compute=float(ev["compute"][0]), network=float(ev["network_t"][0]),
                barrier=chip.barrier(cm.n_tiles, cm.n_chips), step=float(ev["step"][0]))
    if cm.cop is not None:
        time["coproc"] = float(ev["coproc_t"][0])
    return dict(energy={k: float(ev[k][0]) for k in ("synaptic", "neuron", "network", "static", "coproc") if k in ev},
                time=time,
                counts={k: float(ev[k][0]) for k in ("spikes", "active", "synops", "messages", "hops", "chip_hops")},
                core_time=t_core)


def breakdown(cm, spikes, active):
    """Per-population attribution of one step's cost (sums to step_result's energy).

    synaptic / network: charged to the source population (the spikes that cause the events and messages);
    neuron: the population's own updates, active updates and spike generation; static: the step's static energy
    split by the population's share of the occupied cores. core_time: slowest core holding the population;
    critical: True for the population(s) on the slowest core."""
    chip = cm.chip
    S = np.array([float(spikes.get(k, 0.0)) for k in cm.pops])
    A = np.array([float(active.get(k, 0.0)) for k in cm.pops])
    r = step_result(cm, spikes, active)
    syn = cm.tot[3] * S
    net = (cm.tot[0] * chip["e_spike_out"] + (cm.tot[1] - cm.tot[2]) * chip["e_hop"] + cm.tot[2] * chip["e_chip_hop"]
           + cm.tot[4] * chip.v("e_vhop")) * S
    neu = (cm.sizes if chip.neuron_update == "all" else A) * chip["e_neuron_update"] + A * chip["e_neuron_active"] + S * chip["e_spike_gen"]
    occ = np.asarray((cm.cnt.multiply(cm.sizes[None, :])).tocsr().multiply(
        1.0 / np.maximum(cm.neurons, 1e-30)[:, None]).sum(axis=0)).ravel()       # fractional cores per population
    occupied = (cm.neurons > 0).sum()
    stat = r["energy"]["static"] * occ / max(occupied, 1)
    cop = ((cm.cop[0] * chip.c("e_coproc_op") + cm.cop[1] * chip["e_spike_out"]) * S if cm.cop is not None
           else np.zeros(len(S)))
    tc = r["core_time"]
    crit = int(np.argmax(tc))
    cnt = cm.cnt.tocsc()
    out = {}
    for j, k in enumerate(cm.pops):
        cores = cnt.indices[cnt.indptr[j]:cnt.indptr[j + 1]]
        out[k] = dict(synaptic=float(syn[j]), neuron=float(neu[j]), network=float(net[j]), static=float(stat[j]),
                      total=float(syn[j] + neu[j] + net[j] + stat[j] + cop[j]), cores=float(occ[j]),
                      core_time=float(tc[cores].max()) if len(cores) else 0.0, critical=bool(crit in set(cores)))
        if cm.cop is not None:
            out[k]["coproc"] = float(cop[j])
    return out


def rechip(cm, chip):
    """The compiled model under different chip *parameter values* (same topology, mapping, multicast): only e_synop,
    t_synop and t_hop / t_vhop are folded into the matrices, and they enter linearly, so they are rescaled instead of
    recompiled. Falls back to None when per-synapse-type overrides or a 3D vertical latency change make it inexact."""
    import dataclasses
    old = cm.chip
    if any("@" in k for k in chip.params) or (cm.layers > 1 and chip.v("t_vhop") != old.v("t_vhop")):
        return None
    tot = cm.tot.copy()
    tot[3] *= chip["e_synop"] / old["e_synop"]
    return dataclasses.replace(cm, chip=chip, tot=tot, core_synt=cm.core_synt * (chip["t_synop"] / old["t_synop"]),
                               path=cm.path * (chip["t_hop"] / old["t_hop"]))
