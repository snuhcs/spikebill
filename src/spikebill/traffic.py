"""Aggregated per-step spike traffic: messages per projection, hops, chip crossings and XYZ-routed link loads.

A spike becomes one message per destination core holding at least one of its targets (Loihi-style unicast
copies), or, with `chip.multicast == "tree"`, one packet routed along an XY tree when every spike of a source core
reaches all its destination cores (dense fan-out), or, with "chip_broadcast", one packet per remote destination chip
that routers on that chip copy to its destination cores (`broadcast_split`). Messages between cores of the same tile cost no hop.

Core-to-core flows of a projection are (src core, dst core, msg_w, syn_w): expected messages and synaptic
events at the dst core per spike of a neuron on the src core. They come either from the exact connectivity
(`Projection.flows`, see `flows_from_edges`; "R-exact") or from the formula below ("R-formula"):
  dense             every dst core; msg_w = 1, syn_w = neurons on dst core
  random (sparse)   every dst core; msg_w = reach / k_d with reach = k_d (1 - (1 - 1/k_d)^fanout)
  local (conv, one_to_one)  a window of dst cores aligned with the src core's index range;
                    reach = ceil(fanout / neurons per dst core) + (1 if fanout > 1 else 0), capped at k_d
Spikes are spread over a population's cores in proportion to their neuron counts. Routing is dimension-
ordered (X then Y) on the global tile mesh; link loads use difference arrays, so each flow costs O(1).
"""
import math
from dataclasses import dataclass

import numpy as np

MAX_PAIRS = 50_000_000


def locality(proj):
    if proj.kind in ("conv", "one_to_one"):
        return "local"
    if proj.kind == "dense":
        return "dense"
    return "random"


def cores_reached(graph, mapping, proj):
    """Mean destination cores per spike (from flows when given, else the formula)."""
    if proj.flows is not None:
        s, d, mw, sw = proj.flows
        cnt = mapping.pops[proj.src].counts
        per_src = np.bincount(s, weights=mw, minlength=len(cnt))
        return float((per_src * cnt).sum() / cnt.sum())
    k_d = mapping.pops[proj.dst].n_cores
    f = graph.fanout(proj)
    loc = locality(proj)
    if loc == "dense":
        return float(k_d)
    if loc == "random":
        return k_d * (1.0 - (1.0 - 1.0 / k_d) ** f) if k_d > 1 else 1.0
    Q = mapping.pops[proj.dst]
    if proj.shared_weights and proj.share_group and Q.layout is not None:
        # a spike at one position reaches every output channel at that position (and its receptive field):
        # the cores holding those positions = output channels / channels per core
        c_out = max(1, graph.populations[proj.dst].size // proj.share_group)
        return float(min(k_d, math.ceil(c_out / Q.layout[1])))
    n_d = Q.per_core
    return float(min(k_d, math.ceil(f / n_d) + (1 if f > 1 else 0)))


def flows(graph, mapping, proj):
    """(s, d, msg_w, syn_w) with s, d indices into the src / dst population's core lists."""
    if proj.flows is not None:
        return proj.flows
    P, Q = mapping.pops[proj.src], mapping.pops[proj.dst]
    k_s, k_d = P.n_cores, Q.n_cores
    f = graph.fanout(proj)
    reach = cores_reached(graph, mapping, proj)
    share_d = Q.counts / Q.counts.sum()
    if locality(proj) != "local":
        if k_s * k_d > MAX_PAIRS:
            raise MemoryError(f"{proj.src}->{proj.dst}: {k_s}x{k_d} core pairs; aggregate further")
        s = np.repeat(np.arange(k_s), k_d)
        d = np.tile(np.arange(k_d), k_s)
        msg = np.full(len(s), reach / k_d)
        syn = f * share_d[d]
        return s, d, msg, syn
    r = int(round(reach))
    i = np.arange(k_s)
    a = np.floor(i * k_d / k_s).astype(np.int64)
    b = np.maximum(a + 1, np.ceil((i + 1) * k_d / k_s).astype(np.int64))
    lo = np.maximum(0, a - (r - 1) // 2)
    hi = np.minimum(k_d, b + r // 2)
    n = hi - lo
    S = np.repeat(i, n)
    D = np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    nn = np.repeat(n, n).astype(float)
    return S, D, reach / nn, f / nn


def flows_from_edges(src_core_of, dst_core_of, pre, post, n_src_on_core, rates=None):
    """Exact flows from an edge list.

    src_core_of[i] / dst_core_of[j]: core position (within its population) of src neuron i / dst neuron j;
    pre, post: edge endpoints (neuron indices within the populations); n_src_on_core: neurons per src core.
    rates: optional mean firing rate per src neuron; flows are then averaged over a spike drawn in proportion
    to the rates (cores whose neurons never fire fall back to the uniform average).
    """
    pre, post = np.asarray(pre, dtype=np.int64), np.asarray(post, dtype=np.int64)
    src_core_of = np.asarray(src_core_of, dtype=np.int64)
    n_src = len(src_core_of)
    r = np.ones(n_src) if rates is None else np.asarray(rates, dtype=float)
    k_s = len(n_src_on_core)
    den = np.bincount(src_core_of, weights=r, minlength=k_s)
    uniform = den <= 0
    if uniform.any():                       # fall back to uniform weights on cores with no activity
        r = np.where(uniform[src_core_of], 1.0, r)
        den = np.bincount(src_core_of, weights=r, minlength=k_s)
    sc, dc = src_core_of[pre], np.asarray(dst_core_of, dtype=np.int64)[post]
    k_d = int(np.max(dst_core_of)) + 1
    syn_key = sc * k_d + dc
    syn_u, syn_inv = np.unique(syn_key, return_inverse=True)
    syn_w = np.bincount(syn_inv, weights=r[pre])
    pairs = np.unique(pre * k_d + dc)          # distinct (src neuron, dst core) = messages
    pn = pairs // k_d
    msg_u, msg_inv = np.unique(src_core_of[pn] * k_d + pairs % k_d, return_inverse=True)
    msg_w = np.bincount(msg_inv, weights=r[pn])
    keys = np.union1d(syn_u, msg_u)
    syn = np.zeros(len(keys)); msg = np.zeros(len(keys))
    syn[np.searchsorted(keys, syn_u)] = syn_w
    msg[np.searchsorted(keys, msg_u)] = msg_w
    s, d = keys // k_d, keys % k_d
    return s, d, msg / den[s], syn / den[s]


def _z(arr, n):
    return np.zeros(n, dtype=np.int64) if arr is None else arr


def _route_unicast(diff, x1, y1, x2, y2, w, z1=None, z2=None):
    """Add XYZ routes (X leg on (y1, z1), Y leg on (x2, z1), Z leg on (x2, y2)) to the difference arrays."""
    z1, z2 = _z(z1, len(x1)), _z(z2, len(x1))
    e = x2 > x1
    np.add.at(diff["E"], (z1[e], y1[e], x1[e]), w[e]); np.add.at(diff["E"], (z1[e], y1[e], x2[e]), -w[e])
    e = x2 < x1
    np.add.at(diff["W"], (z1[e], y1[e], x2[e]), w[e]); np.add.at(diff["W"], (z1[e], y1[e], x1[e]), -w[e])
    e = y2 > y1
    np.add.at(diff["N"], (z1[e], y1[e], x2[e]), w[e]); np.add.at(diff["N"], (z1[e], y2[e], x2[e]), -w[e])
    e = y2 < y1
    np.add.at(diff["S"], (z1[e], y2[e], x2[e]), w[e]); np.add.at(diff["S"], (z1[e], y1[e], x2[e]), -w[e])
    e = z2 > z1
    np.add.at(diff["U"], (z1[e], y2[e], x2[e]), w[e]); np.add.at(diff["U"], (z2[e], y2[e], x2[e]), -w[e])
    e = z2 < z1
    np.add.at(diff["D"], (z2[e], y2[e], x2[e]), w[e]); np.add.at(diff["D"], (z1[e], y2[e], x2[e]), -w[e])


def new_diff(W, H, Z=1):
    """Link-load difference arrays: E/W/N/S per layer (Z, H+1, W+1); U/D vertical (Z+1, H+1, W+1)."""
    d = {k: np.zeros((Z, H + 1, W + 1)) for k in "EWNS"}
    d.update({k: np.zeros((Z + 1, H + 1, W + 1)) for k in "UD"})
    return d


def max_links(diff, W, H, chip):
    """Busiest tile link, busiest chip link and busiest vertical link from the difference arrays."""
    max_intra = max_inter = 0.0
    xs, ys = np.arange(W + 1), np.arange(H + 1)
    inter_x = ((xs + 1) % chip.tiles_x == 0)[None, None, :]
    inter_y = ((ys + 1) % chip.tiles_y == 0)[None, :, None]
    for dname, axis, inter in (("E", 2, inter_x), ("W", 2, inter_x), ("N", 1, inter_y), ("S", 1, inter_y)):
        load = np.cumsum(diff[dname], axis=axis)
        max_intra = max(max_intra, float(np.where(inter, 0, load).max()))
        max_inter = max(max_inter, float(np.where(inter, load, 0).max()))
    vert = max(float(np.cumsum(diff[k], axis=0).max()) for k in "UD")
    return max_intra, max_inter, vert


def _crossings(a, b, period):
    return np.abs(np.floor_divide(b, period) - np.floor_divide(a, period))


def _route_tree(diff, src, dst, w, chip):
    """XYZ multicast tree from src (x, y[, z]) to dst rows (n, 2 or 3): an X segment in the source row and layer,
    a Y segment per destination column (source layer), a Z segment per destination (x, y).
    Adds load w per tree link; returns (planar links, chip-crossing links, vertical links). One sort, then a loop
    over Python ints (calls have few columns, so numpy per-column work costs more than it saves);
    `_route_tree_ref` is the original version, kept for tests."""
    sx, sy = int(src[0]), int(src[1])
    sz = int(src[2]) if len(src) > 2 else 0
    dst = np.asarray(dst, dtype=np.int64)
    three = dst.shape[1] > 2 and bool((dst[:, 2] != sz).any())
    tx, ty = chip.tiles_x, chip.tiles_y
    if three:
        o = np.lexsort((dst[:, 2], dst[:, 1], dst[:, 0]))
        rows = dst[o].tolist()
    else:
        key = np.unique((dst[:, 0] << 32) | dst[:, 1])                  # sorted by x, then y (1-D unique: fast)
        rows = np.column_stack([key >> 32, key & 0xFFFFFFFF]).tolist()
    xmin, xmax = min(sx, rows[0][0]), max(sx, rows[-1][0])
    links = xmax - xmin
    ch = abs(xmin // tx - sx // tx) + abs(xmax // tx - sx // tx)
    if xmax > sx:
        diff["E"][sz, sy, sx] += w; diff["E"][sz, sy, xmax] -= w
    if xmin < sx:
        diff["W"][sz, sy, xmin] += w; diff["W"][sz, sy, sx] -= w
    vlinks, n, i = 0, len(rows), 0
    Nn, S, U, D = diff["N"], diff["S"], diff["U"] if three else None, diff["D"] if three else None
    while i < n:
        c = rows[i][0]
        j = i
        while j < n and rows[j][0] == c:
            j += 1
        ymin, ymax = min(sy, rows[i][1]), max(sy, rows[j - 1][1])     # rows sorted by y within the column
        links += ymax - ymin
        ch += abs(ymin // ty - sy // ty) + abs(ymax // ty - sy // ty)
        if ymax > sy:
            Nn[sz, sy, c] += w; Nn[sz, ymax, c] -= w
        if ymin < sy:
            S[sz, ymin, c] += w; S[sz, sy, c] -= w
        if three:
            k = i
            while k < j:
                yy = rows[k][1]
                m = k
                while m < j and rows[m][1] == yy:
                    m += 1
                zmin, zmax = min(sz, rows[k][2]), max(sz, rows[m - 1][2])
                vlinks += zmax - zmin
                if zmax > sz:
                    U[sz, yy, c] += w; U[zmax, yy, c] -= w
                if zmin < sz:
                    D[zmin, yy, c] += w; D[sz, yy, c] -= w
                k = m
        i = j
    return float(links), float(ch), float(vlinks)


def _route_tree_ref(diff, src, dst, w, chip):
    """XYZ multicast tree from src (x, y[, z]) to dst rows (n, 2 or 3): an X segment in the source row and layer,
    a Y segment per destination column (source layer), a Z segment per destination (x, y).
    Adds load w per tree link; returns (planar links, chip-crossing links, vertical links). Reference loop version
    of `_route_tree` (kept for tests)."""
    sx, sy = int(src[0]), int(src[1])
    sz = int(src[2]) if len(src) > 2 else 0
    dz = dst[:, 2] if dst.shape[1] > 2 else np.zeros(len(dst), dtype=np.int64)
    cols = np.unique(dst[:, 0])
    xmin, xmax = min(sx, int(cols.min())), max(sx, int(cols.max()))
    links = (xmax - xmin)
    vlinks = 0
    ch = float(_crossings(np.array([xmin]), np.array([sx]), chip.tiles_x)[0]
               + _crossings(np.array([sx]), np.array([xmax]), chip.tiles_x)[0])
    if xmax > sx:
        diff["E"][sz, sy, sx] += w; diff["E"][sz, sy, xmax] -= w
    if xmin < sx:
        diff["W"][sz, sy, xmin] += w; diff["W"][sz, sy, sx] -= w
    for c in cols:
        inc = dst[:, 0] == c
        ys = dst[inc, 1]
        ymin, ymax = min(sy, int(ys.min())), max(sy, int(ys.max()))
        links += ymax - ymin
        ch += float(_crossings(np.array([ymin]), np.array([sy]), chip.tiles_y)[0]
                    + _crossings(np.array([sy]), np.array([ymax]), chip.tiles_y)[0])
        if ymax > sy:
            diff["N"][sz, sy, c] += w; diff["N"][sz, ymax, c] -= w
        if ymin < sy:
            diff["S"][sz, ymin, c] += w; diff["S"][sz, sy, c] -= w
        for yy in np.unique(ys):
            zs = dz[inc][ys == yy]
            zmin, zmax = min(sz, int(zs.min())), max(sz, int(zs.max()))
            vlinks += zmax - zmin
            if zmax > sz:
                diff["U"][sz, yy, c] += w; diff["U"][zmax, yy, c] -= w
            if zmin < sz:
                diff["D"][zmin, yy, c] += w; diff["D"][sz, yy, c] -= w
    return float(links), ch, float(vlinks)


def chip_ids(xyz, chip):
    """Integer id of the chip holding each tile (x, y, z); stacked layers (z) belong to the same chip."""
    xyz = np.asarray(xyz, dtype=np.int64)
    key = np.column_stack([xyz[:, 1] // chip.tiles_y, xyz[:, 0] // chip.tiles_x])
    return np.unique(key, axis=0, return_inverse=True)[1].ravel()


def broadcast_split(si, a, b, sw, w, logmiss, chip):
    """chip_broadcast routing of pair flows (source id si, source / destination tiles a, b as (n, 3) xyz).

    Destinations on the source's chip stay unicast. For every (source, remote destination chip) the source sends one
    packet, needed with probability 1 - prod(1 - p_msg) over that chip's destination cores (logmiss = log(1 - p_msg)
    per pair, times the cores a pair stands for), to the destination tile nearest the source (ties: lowest x, y, z);
    routers there copy it to every destination core on the chip (hops and link loads, no sender cost, as for "tree").
    sw: spikes of the pair's source; w: unicast messages of the pair.
    Returns send = (pair index of the leg's source, from xyz, to xyz, weight): legs a core sends;
    route = (from xyz, to xyz, weight): router copies; (ph, pv): planar / vertical hops of each pair's full path."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    ids = chip_ids(np.vstack([a, b]), chip)
    ka, kb = ids[:len(a)], ids[len(a):]
    hp = np.abs(b[:, 0] - a[:, 0]) + np.abs(b[:, 1] - a[:, 1])
    hv = np.abs(b[:, 2] - a[:, 2])
    ph, pv = hp.astype(float), hv.astype(float)
    loc = np.flatnonzero(ka == kb)
    rem = np.flatnonzero(ka != kb)
    send = [(loc, a[loc], b[loc], w[loc])]
    route = (np.zeros((0, 3), np.int64), np.zeros((0, 3), np.int64), np.zeros(0))
    if len(rem):
        g = np.unique(np.column_stack([si[rem], kb[rem]]), axis=0, return_inverse=True)[1].ravel()
        br = b[rem]
        order = np.lexsort((br[:, 2], br[:, 1], br[:, 0], hp[rem] + hv[rem], g))
        gs = g[order]
        first = order[np.r_[True, gs[1:] != gs[:-1]]]          # entry pair (within rem) of group 0, 1, ...
        ent = rem[first]
        p = -np.expm1(np.bincount(g, weights=logmiss[rem]))
        send.append((ent, a[ent], b[ent], sw[ent] * p))
        e = b[ent][g]
        route = (e, br, w[rem])
        ph[rem] = hp[ent][g] + np.abs(br[:, 0] - e[:, 0]) + np.abs(br[:, 1] - e[:, 1])
        pv[rem] = hv[ent][g] + np.abs(br[:, 2] - e[:, 2])
    idx = np.concatenate([x[0] for x in send])
    return (idx, np.vstack([x[1] for x in send]), np.vstack([x[2] for x in send]),
            np.concatenate([x[3] for x in send])), route, (ph, pv)


def log_miss(mw, mult=1.0):
    """log(1 - p) per pair for broadcast_split (p = message probability, clipped to [0, 1])."""
    with np.errstate(divide="ignore"):
        return mult * np.log1p(-np.clip(mw, 0.0, 1.0))


@dataclass
class Traffic:
    per_proj: dict          # (src, dst) -> dict(spikes, messages, hops, chip_hops, vhops, synops)
    messages: float
    hops: float             # planar tile hops (including chip crossings)
    chip_hops: float
    max_link_intra: float   # messages per step on the busiest on-chip planar link
    max_link_inter: float   # messages per step on the busiest chip-to-chip link
    max_path_hops: int      # longest planar path (hops)
    core_out_msgs: np.ndarray
    core_in_msgs: np.ndarray
    core_synops: np.ndarray
    core_synop_time: np.ndarray   # synops weighted by per-synapse-type t_synop
    synop_energy: float           # synops weighted by per-synapse-type e_synop
    vhops: float = 0.0            # vertical hops (3D-stacked chips)
    max_link_vert: float = 0.0
    max_path_time: float = 0.0    # longest path latency: planar hops x t_hop + vertical hops x t_vhop
    diff: dict = None             # link-load difference arrays (for link_loads / visualisation)


def step_traffic(graph, mapping, chip, spikes):
    """Traffic of one step with spikes {pop: spikes in this step} (counts, not rates)."""
    W, H = mapping.mesh
    Z = getattr(mapping, "layers", 1) or 1
    zc = mapping.z if getattr(mapping, "z", None) is not None else np.zeros(mapping.n_cores, dtype=np.int64)
    diff = new_diff(W, H, Z)
    nc = mapping.n_cores
    out_m, in_m, syn, syn_t = np.zeros(nc), np.zeros(nc), np.zeros(nc), np.zeros(nc)
    per, tot = {}, dict(messages=0.0, hops=0.0, chip_hops=0.0, vhops=0.0, e_syn=0.0)
    max_path, max_path_t = 0, 0.0
    t_hop, t_vhop = chip["t_hop"], chip.v("t_vhop")
    for proj in graph.projections:
        P, Q = mapping.pops[proj.src], mapping.pops[proj.dst]
        sp = float(spikes.get(proj.src, 0.0))
        s, d, mw, sw = flows(graph, mapping, proj)
        sp_core = sp * P.share                # spikes per src core (activity share)
        w = mw * sp_core[s]               # messages per pair
        y = sw * sp_core[s]               # synaptic events per pair
        gs, gd = P.cores[s], Q.cores[d]   # used-core indices
        x1, y1, z1 = mapping.xy[gs, 0], mapping.xy[gs, 1], zc[gs]
        x2, y2, z2 = mapping.xy[gd, 0], mapping.xy[gd, 1], zc[gd]
        hops = np.abs(x2 - x1) + np.abs(y2 - y1)
        vh = np.abs(z2 - z1)
        chops = np.abs(x2 // chip.tiles_x - x1 // chip.tiles_x) + np.abs(y2 // chip.tiles_y - y1 // chip.tiles_y)
        np.add.at(in_m, gd, w)
        np.add.at(syn, gd, y)
        np.add.at(syn_t, gd, y * chip.get("t_synop", proj.synapse))
        # multicast: a src core whose every spike reaches all its dst cores (msg_w ~ 1) sends one packet along an
        # XYZ tree; the rest stays unicast
        if chip.multicast == "chip_broadcast" and len(s):
            xyz1, xyz2 = np.column_stack([x1, y1, z1]), np.column_stack([x2, y2, z2])
            (si, A, B, ws), (RA, RB, rw), (ph, pv) = broadcast_split(s, xyz1, xyz2, sp_core[s], w, log_miss(mw), chip)
            np.add.at(out_m, gs[si], ws)
            n_msg = float(ws.sum())
            n_hop = n_chop = n_v = 0.0
            for (A_, B_, w_) in ((A, B, ws), (RA, RB, rw)):
                _route_unicast(diff, A_[:, 0], A_[:, 1], B_[:, 0], B_[:, 1], w_, A_[:, 2], B_[:, 2])
                n_hop += float((w_ * (np.abs(B_[:, 0] - A_[:, 0]) + np.abs(B_[:, 1] - A_[:, 1]))).sum())
                n_chop += float((w_ * (np.abs(B_[:, 0] // chip.tiles_x - A_[:, 0] // chip.tiles_x)
                                       + np.abs(B_[:, 1] // chip.tiles_y - A_[:, 1] // chip.tiles_y))).sum())
                n_v += float((w_ * np.abs(B_[:, 2] - A_[:, 2])).sum())
            hops, vh = ph, pv
            tree = np.ones(len(s), dtype=bool)     # nothing left for the unicast branch below
        else:
            tree = np.zeros(len(s), dtype=bool)
        if chip.multicast == "tree" and len(s):
            full = np.ones(P.n_cores, dtype=bool)
            np.logical_and.at(full, s, mw >= 0.999)
            tree = full[s]
        u = ~tree
        if chip.multicast != "chip_broadcast":
            np.add.at(out_m, gs[u], w[u])
            _route_unicast(diff, x1[u], y1[u], x2[u], y2[u], w[u], z1[u], z2[u])
            n_msg, n_hop = float(w[u].sum()), float((w[u] * hops[u]).sum())
            n_chop, n_v = float((w[u] * chops[u]).sum()), float((w[u] * vh[u]).sum())
        if tree.any() and chip.multicast == "tree":
            xyz = np.column_stack([mapping.xy, zc])
            for sc in np.unique(s[tree]):
                k = tree & (s == sc)
                ws = float(sp_core[sc])
                if ws <= 0:
                    continue
                h, ch, hv = _route_tree(diff, xyz[P.cores[sc]], xyz[gd[k]], ws, chip)
                out_m[P.cores[sc]] += ws
                n_msg += ws; n_hop += ws * h; n_chop += ws * ch; n_v += ws * hv
        rec = dict(spikes=sp, messages=n_msg, hops=n_hop, chip_hops=n_chop, vhops=n_v, synops=float(y.sum()))
        per[(proj.src, proj.dst)] = rec
        for k_ in ("messages", "hops", "chip_hops", "vhops"):
            tot[k_] += rec[k_]
        tot["e_syn"] += rec["synops"] * chip.get("e_synop", proj.synapse)
        live = w > 0
        if len(hops) and sp > 0 and live.any():
            max_path = max(max_path, int(hops[live].max()))
            max_path_t = max(max_path_t, float((hops[live] * t_hop + vh[live] * t_vhop).max()))
    max_intra, max_inter, max_vert = max_links(diff, W, H, chip)
    return Traffic(per, tot["messages"], tot["hops"], tot["chip_hops"], max_intra, max_inter, max_path,
                   out_m, in_m, syn, syn_t, tot["e_syn"], tot["vhops"], max_vert, max_path_t, diff)


def link_loads(diff):
    """Messages per step on every link: E/W/N/S arrays (Z, H+1, W+1), U/D arrays (Z+1, H+1, W+1)."""
    out = {k: np.cumsum(diff[k], axis=2) for k in "EW"}
    out.update({k: np.cumsum(diff[k], axis=1) for k in "NS"})
    out.update({k: np.cumsum(diff[k], axis=0) for k in "UD"})
    return out
