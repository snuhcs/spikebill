"""Event-driven packet-level NoC simulation of one time step (reference for the aggregated network term, H2).

Model: one packet = one flit; XYZ routing on the global tile mesh (Z stacked layers); FIFO links with infinite buffers (no back-
pressure, no virtual channels); a link is busy for t_link (tile link) or t_chip_hop (chip-to-chip link) per packet;
a packet reaches the next router t_hop after leaving the link. Each source core injects its packets one after
another every t_spike_out, in a given order, starting at t = 0. A multicast packet follows an XY tree and is
replicated at branch routers. Unicast and multicast packets are both represented as trees of links (a unicast
route is a chain). Returns the time the last packet copy arrives.
"""
import heapq
import warnings

import numpy as np

try:                                       # numba is optional (extra "packet"): without it the kernels run as plain
    import numba                           # Python, identical results, far slower on large steps
    _njit = numba.njit(cache=True)
    HAVE_NUMBA = True
except ImportError:                        # pragma: no cover (exercised in the minimal-install CI job)
    warnings.warn("numba not installed: spikebill.packetsim runs as plain Python (slow); pip install numba",
                  RuntimeWarning, stacklevel=2)
    HAVE_NUMBA = False

    def _njit(f):
        return f

E, W, N, S, U, D = 0, 1, 2, 3, 4, 5


class Net:
    """Link indexing, per-link serialisation time and per-link latency for a W x H x Z tile mesh made of chips of
    tx x ty tiles (Z stacked layers; vertical links t_vlink / t_vhop)."""

    def __init__(self, W_, H_, tx, ty, t_link, t_chip, Z=1, t_vlink=0.0, t_hop=None, t_vhop=0.0):
        self.W, self.H, self.Z, self.tx, self.ty = W_, H_, Z, tx, ty
        n = W_ * H_ * Z * 6
        self.ser = np.full(n, t_link)
        self.lat = np.full(n, np.nan if t_hop is None else t_hop)
        for z in range(Z):
            for y in range(H_):
                for x in range(W_):
                    base = ((z * H_ + y) * W_ + x) * 6
                    if (x + 1) % tx == 0:
                        self.ser[base + E] = t_chip
                    if x % tx == 0:
                        self.ser[base + W] = t_chip
                    if (y + 1) % ty == 0:
                        self.ser[base + N] = t_chip
                    if y % ty == 0:
                        self.ser[base + S] = t_chip
                    self.ser[base + U] = self.ser[base + D] = t_vlink
                    self.lat[base + U] = self.lat[base + D] = t_vhop

    def link(self, x, y, d, z=0):
        return ((z * self.H + y) * self.W + x) * 6 + d


def _tree_links(net, sx, sy, dsts, sz=0):
    """Links of the XYZ multicast tree from (sx, sy, sz) to the tile set dsts ((x, y) or (x, y, z)):
    list of (link id, parent index)."""
    out = []
    if not dsts:
        return out
    cols = {}
    for t in dsts:
        x, y, z = int(t[0]), int(t[1]), (int(t[2]) if len(t) > 2 else 0)
        cols.setdefault(x, {}).setdefault(y, []).append(z)

    def zleg(c, y, parent):
        zs = cols[c].get(y)
        if not zs:
            return
        for d, step in ((U, 1), (D, -1)):
            far = max(zs) if step > 0 else min(zs)
            if (far - sz) * step <= 0:
                continue
            p, z = parent, sz
            while z != far:
                out.append((net.link(c, y, d, z), p)); p = len(out) - 1; z += step

    def vertical(c, parent):
        ys = cols.get(c)
        if not ys:
            return
        zleg(c, sy, parent)
        for d, step in ((N, 1), (S, -1)):
            far = max(ys) if step > 0 else min(ys)
            if (far - sy) * step <= 0:
                continue
            p, y = parent, sy
            while y != far:
                out.append((net.link(c, y, d, sz), p)); p = len(out) - 1; y += step
                zleg(c, y, p)

    vertical(sx, -1)
    for d, step in ((E, 1), (W, -1)):
        far = max(cols) if step > 0 else min(cols)
        if (far - sx) * step <= 0:
            continue
        p, x = -1, sx
        while x != far:
            out.append((net.link(x, sy, d, sz), p)); p = len(out) - 1; x += step
            vertical(x, p)
    return out


def build(net, packets):
    """packets: list of (src core id, inject time, src (x, y), [dst (x, y), ...]) -> flat arrays for `run`."""
    items = []
    for _, t, src, dsts in packets:
        sx, sy = src[0], src[1]
        sz = src[2] if len(src) > 2 else 0
        items.append((t, _tree_links(net, sx, sy, [d for d in dsts if tuple(d) != tuple(src)], sz)))
    return _flatten(items)


def _attach(out, sub, parent):
    """Append link list `sub` (from _tree_links) to `out`, its roots continuing from out[parent] (-1: new root)."""
    off = len(out)
    out.extend((l, p + off if p >= 0 else parent) for l, p in sub)


def broadcast_links(net, src, entry, dsts):
    """chip_broadcast packet (spikebill.traffic.broadcast_split): one XY route from src to the entry tile on the
    destination chip, then one router copy per destination, each its own XY route from the entry tile (as the cost
    model counts them). Returns a link list (link id, parent index) for `_flatten`."""
    out = []
    _attach(out, _tree_links(net, src[0], src[1], [tuple(entry)]), -1)
    last = len(out) - 1
    for d in dsts:
        if tuple(d)[:2] != tuple(entry)[:2]:
            _attach(out, _tree_links(net, entry[0], entry[1], [tuple(d)]), last)
    return out


def _flatten(items):
    """items: list of (inject time, link list) -> flat arrays for `_run`."""
    link_id, first_child, n_child, roots, t0 = [], [], [], [], []
    children = []
    for t, tl in items:
        base = len(link_id)
        kids = [[] for _ in tl]
        for i, (_, p) in enumerate(tl):
            if p >= 0:
                kids[p].append(base + i)
            else:
                roots.append(base + i); t0.append(t)
        for i, (l, _) in enumerate(tl):
            link_id.append(l)
            first_child.append(len(children)); n_child.append(len(kids[i])); children.extend(kids[i])
    return (np.array(link_id, np.int64), np.array(first_child, np.int64), np.array(n_child, np.int64),
            np.array(children, np.int64), np.array(roots, np.int64), np.array(t0, np.float64))


@_njit
def _run(link_id, first_child, n_child, children, roots, t0, ser, lat, n_links):
    free = np.zeros(n_links)
    heap = [(t0[i], roots[i]) for i in range(len(roots))]
    heapq.heapify(heap)
    t_last = 0.0
    hops = 0
    while len(heap) > 0:
        t, fi = heapq.heappop(heap)
        l = link_id[fi]
        start = t if t > free[l] else free[l]
        dep = start + ser[l]
        free[l] = dep
        arr = dep + lat[l]
        hops += 1
        if arr > t_last:
            t_last = arr
        for k in range(first_child[fi], first_child[fi] + n_child[fi]):
            heapq.heappush(heap, (arr, children[k]))
    return t_last, hops


@_njit
def _chains(sx, sy, sz, dx, dy, dz, W_, H_):
    """Unicast XYZ routes as chains: flat link ids, child index (-1 at the end), root index per packet."""
    n = 0
    for i in range(len(sx)):
        n += abs(dx[i] - sx[i]) + abs(dy[i] - sy[i]) + abs(dz[i] - sz[i])
    link = np.empty(n, np.int64)
    child = np.full(n, -1, np.int64)
    root = np.full(len(sx), -1, np.int64)
    k = 0
    for i in range(len(sx)):
        x, y, z = sx[i], sy[i], sz[i]
        first = True
        while x != dx[i] or y != dy[i] or z != dz[i]:
            if x != dx[i]:
                d = 0 if dx[i] > x else 1
            elif y != dy[i]:
                d = 2 if dy[i] > y else 3
            else:
                d = 4 if dz[i] > z else 5
            link[k] = ((z * H_ + y) * W_ + x) * 6 + d
            if first:
                root[i] = k
                first = False
            else:
                child[k - 1] = k
            if d == 0:
                x += 1
            elif d == 1:
                x -= 1
            elif d == 2:
                y += 1
            elif d == 3:
                y -= 1
            elif d == 4:
                z += 1
            else:
                z -= 1
            k += 1
    return link, child, root


def build_mixed(net, uni, trees, custom=()):
    """uni: arrays (t, sx, sy, dx, dy[, sz, dz]) of unicast packets; trees: list of (t, (sx, sy[, sz]), dsts);
    custom: list of (t, link list) (e.g. `broadcast_links`). Flat arrays for _run."""
    uni = [np.asarray(a) for a in uni]
    t_u, sx, sy, dx, dy = uni[:5]
    sz, dz = (uni[5], uni[6]) if len(uni) > 5 else (np.zeros_like(sx), np.zeros_like(sx))
    i64 = lambda a: np.asarray(a).astype(np.int64)  # noqa: E731
    link, child, root = _chains(i64(sx), i64(sy), i64(sz), i64(dx), i64(dy), i64(dz), net.W, net.H)
    first_child = np.arange(len(link), dtype=np.int64)
    n_child = (child >= 0).astype(np.int64)
    children = np.where(child >= 0, child, 0)
    keep = root >= 0
    roots, t0 = list(root[keep]), list(t_u[keep])
    parts = ([build(net, [(0, t, src, d) for t, src, d in trees])] if trees else []) + (
        [_flatten(list(custom))] if len(custom) else [])
    for lt, fc, nc, ch, rt, tt in parts:
        off = len(link)
        link = np.concatenate([link, lt]); first_child = np.concatenate([first_child, fc + len(children)])
        n_child = np.concatenate([n_child, nc]); children = np.concatenate([children, ch + off])
        roots += list(rt + off); t0 += list(tt)
    return (link, first_child, n_child, children, np.array(roots, np.int64), np.array(t0, np.float64))


def _lat(net, t_hop):
    return np.where(np.isnan(net.lat), t_hop, net.lat)


def simulate_mixed(net, uni, trees, t_hop, custom=()):
    """Last arrival time and link traversals of unicast packets, trees and custom link lists (see build_mixed)."""
    arrs = build_mixed(net, uni, trees, custom)
    t_inj = max(float(np.max(uni[0])) if len(uni[0]) else 0.0, max((t for t, _, _ in trees), default=0.0),
                max((t for t, _ in custom), default=0.0))
    if len(arrs[4]) == 0:
        return t_inj, 0
    t, hops = _run(*arrs, net.ser, _lat(net, t_hop), len(net.ser))
    return max(t, t_inj), hops


def simulate(net, packets, t_hop):
    """Last arrival time of one step's packets (0 if no packet leaves its tile; local delivery is immediate)."""
    arrs = build(net, packets)
    t_inject_last = max((p[1] for p in packets), default=0.0)
    if len(arrs[4]) == 0:
        return t_inject_last, 0
    t, hops = _run(*arrs, net.ser, _lat(net, t_hop), len(net.ser))
    return max(t, t_inject_last), hops


def inject_schedule(src_cores, t_spike_out, rng):
    """Injection time of each packet: per source core, packets in a random order every t_spike_out."""
    src_cores = np.asarray(src_cores)
    order = rng.permutation(len(src_cores))
    t = np.zeros(len(src_cores))
    count = {}
    for i in order:
        c = int(src_cores[i])
        k = count.get(c, 0)
        t[i] = k * t_spike_out
        count[c] = k + 1
    return t


def inject_times(src, t_out, rng):
    """Per source core, its packets in a random order every t_out (vectorised inject_schedule)."""
    src = np.asarray(src)
    order = rng.permutation(len(src))
    s = src[order]
    idx = np.argsort(s, kind="stable")
    ss = s[idx]
    start = np.r_[0, np.flatnonzero(np.diff(ss)) + 1] if len(ss) else np.zeros(0, np.int64)
    rank = np.arange(len(ss)) - np.repeat(start, np.diff(np.r_[start, len(ss)]))
    t = np.empty(len(src))
    t[order[idx]] = rank * t_out
    return t


def step_packets(graph, mapping, chip, spikes, rng):
    """Sample one step's packets from the cost model's flows (`traffic.flows`), in the chip's multicast mode.

    Per source core, Poisson(spikes x activity share) spikes; each spike reaches each destination core of a pair
    with the pair's message probability. "none": one unicast packet per (spike, reached core); "tree": a source
    core whose every pair has probability ~1 sends one XY tree per spike (as the cost model); "chip_broadcast":
    unicast on the source's chip, and per remote chip with at least one reached core one packet to the entry tile
    (destination tile nearest the source; ties lowest x, y) plus router copies (`broadcast_links`).
    Returns (uni_src, uni_dst, trees, custom): used-core ids of unicast packets, trees as (src core, dst cores),
    custom as (src core, entry core, dst cores). 2D meshes only."""
    from . import traffic
    mode = chip.multicast
    xy = mapping.xy
    u_s, u_d, trees, custom = [], [], [], []
    for proj in graph.projections:
        P, Q = mapping.pops[proj.src], mapping.pops[proj.dst]
        sp_core = float(spikes.get(proj.src, 0.0)) * P.share
        s, d, mw, _ = traffic.flows(graph, mapping, proj)
        if not len(s):
            continue
        gs, gd = P.cores[s], Q.cores[d]
        kd = traffic.chip_ids(np.column_stack([xy[gd], np.zeros(len(gd), np.int64)]), chip) if mode == "chip_broadcast" else None
        ks = traffic.chip_ids(np.column_stack([xy[gs], np.zeros(len(gs), np.int64)]), chip) if mode == "chip_broadcast" else None
        order = np.argsort(s, kind="stable")
        bounds = np.r_[0, np.flatnonzero(np.diff(s[order])) + 1, len(order)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            J = order[a:b]
            sc = int(s[J[0]])
            k = rng.poisson(sp_core[sc])
            if k == 0:
                continue
            src = int(P.cores[sc])
            if mode == "tree" and np.all(mw[J] >= 0.999):
                trees += [(src, gd[J])] * int(k)
                continue
            hit = rng.random((k, len(J))) < np.clip(mw[J], 0.0, 1.0)
            if mode != "chip_broadcast":
                r, c = np.nonzero(hit)
                u_s.append(np.full(len(c), src)); u_d.append(gd[J][c])
                continue
            same = (kd[J] == ks[J])
            r, c = np.nonzero(hit[:, same])
            u_s.append(np.full(len(c), src)); u_d.append(gd[J][same][c])
            rem = J[~same]
            if not len(rem):
                continue
            hrem = hit[:, ~same]
            sx, sy = xy[src]
            for kc in np.unique(kd[rem]):
                m = kd[rem] == kc
                cores = gd[rem][m]
                dist = np.abs(xy[cores, 0] - sx) + np.abs(xy[cores, 1] - sy)
                ent = cores[np.lexsort((xy[cores, 1], xy[cores, 0], dist))[0]]
                for row in hrem[:, m]:
                    if row.any():
                        custom.append((src, int(ent), cores[row]))
    cat = lambda v: np.concatenate(v).astype(np.int64) if v else np.zeros(0, np.int64)  # noqa: E731
    return cat(u_s), cat(u_d), trees, custom


def simulate_step(graph, mapping, chip, spikes, seed=0):
    """Packet-level time of one step of a mapped graph (`step_packets`, injection every t_spike_out per source core
    in a random order). Returns dict(t_sim, link_traversals, packets (sent by cores), copies)."""
    rng = np.random.default_rng(seed)
    us, ud, trees, custom = step_packets(graph, mapping, chip, spikes, rng)
    src = np.concatenate([us, np.array([t[0] for t in trees], np.int64), np.array([c[0] for c in custom], np.int64)])
    tin = inject_times(src, chip["t_spike_out"], rng)
    W, H = mapping.mesh
    net = Net(W, H, chip.tiles_x, chip.tiles_y, chip["t_link"], chip["t_chip_hop"])
    xy = mapping.xy
    nu, nt = len(us), len(trees)
    tr = [(float(tin[nu + i]), tuple(xy[s_]), [tuple(v) for v in xy[d_]]) for i, (s_, d_) in enumerate(trees)]
    cu = [(float(tin[nu + nt + i]), broadcast_links(net, tuple(xy[s_]), tuple(xy[e_]), [tuple(v) for v in xy[d_]]))
          for i, (s_, e_, d_) in enumerate(custom)]
    t, hops = simulate_mixed(net, (tin[:nu], xy[us, 0], xy[us, 1], xy[ud, 0], xy[ud, 1]), tr, chip["t_hop"], cu)
    return dict(t_sim=float(t), link_traversals=int(hops), packets=int(len(src)),
                copies=int(sum(len(c[2]) for c in custom)))


def simulate_finite(net, uni, t_hop, buf):
    """Unicast packets (as `simulate_mixed`'s `uni`) with finite buffers: every link has a FIFO queue at its upstream
    router holding at most `buf` single-flit packets (waiting or reserved). A packet starts on a link only when a slot
    is free in the queue of its next link (credit-based back-pressure); a blocked head-of-line packet blocks its
    queue. Each source tile injects through its own unbounded network-interface queue. buf=None: infinite buffers
    (same model as `simulate_mixed`). Single flits, XYZ routing (deadlock-free on a mesh). Pure Python: for
    validation-size steps. Returns (last arrival time, link traversals, max queue occupancy seen)."""
    import collections
    uni = [np.asarray(a) for a in uni]
    t_u, sx, sy, dx, dy = uni[:5]
    sz, dz = (uni[5], uni[6]) if len(uni) > 5 else (np.zeros_like(sx), np.zeros_like(sx))
    i64 = lambda a: np.asarray(a).astype(np.int64)  # noqa: E731
    link, child, root = _chains(i64(sx), i64(sy), i64(sz), i64(dx), i64(dy), i64(dz), net.W, net.H)
    lat_l, ser_l = _lat(net, t_hop), net.ser
    n_real = len(ser_l)
    tiles = {}
    routes, t_inj = [], []
    for i in range(len(sx)):
        if root[i] < 0:
            continue
        ni = n_real + tiles.setdefault((int(sx[i]), int(sy[i]), int(sz[i])), len(tiles))
        r, k = [ni], root[i]
        while k >= 0:
            r.append(int(link[k])); k = child[k]
        routes.append(r); t_inj.append(float(t_u[i]))
    t_last = float(np.max(t_u)) if len(t_u) else 0.0
    if not routes:
        return t_last, 0, 0
    N = n_real + len(tiles)
    ser = np.concatenate([ser_l, np.zeros(len(tiles))])
    lat = np.concatenate([lat_l, np.zeros(len(tiles))])
    cap = np.full(N, np.inf if buf is None else float(buf)); cap[n_real:] = np.inf
    occ = np.zeros(N, np.int64)
    busy = np.zeros(N)
    queue = collections.defaultdict(collections.deque)
    waiters = collections.defaultdict(set)
    heap, seq = [], 0
    for p, t in enumerate(t_inj):
        heap.append((t, seq, 0, p, 0)); seq += 1
    heapq.heapify(heap)
    hops, occ_max = 0, 0
    while heap:
        t, _, kind, a, h = heapq.heappop(heap)
        if kind == 0:                                     # packet a arrives at the queue of its hop h
            l = routes[a][h]
            queue[l].append((a, h))
            if l >= n_real:
                occ[l] += 1
            heapq.heappush(heap, (t, seq, 1, l, 0)); seq += 1
            continue
        l = a                                             # try to start the head packet of link l's queue
        q = queue[l]
        if not q or busy[l] > t:
            continue
        p, hh = q[0]
        r = routes[p]
        nl = r[hh + 1] if hh + 1 < len(r) else -1
        if nl >= 0 and occ[nl] >= cap[nl]:
            waiters[nl].add(l)
            continue
        q.popleft()
        if nl >= 0:
            occ[nl] += 1
            occ_max = max(occ_max, int(occ[nl]) if nl < n_real else 0)
        occ[l] -= 1
        dep = t + ser[l]
        busy[l] = dep
        if l < n_real:
            hops += 1
        heapq.heappush(heap, (dep, seq, 1, l, 0)); seq += 1
        if nl >= 0:
            heapq.heappush(heap, (dep + lat[l], seq, 0, p, hh + 1)); seq += 1
        else:
            t_last = max(t_last, dep + lat[l])
        if waiters[l]:
            for w in waiters.pop(l):
                heapq.heappush(heap, (t, seq, 1, w, 0)); seq += 1
    return t_last, hops, occ_max
