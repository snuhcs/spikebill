"""Placement baseline: search the order in which populations fill the tile path (simulated annealing on traffic).

`mapping.map_graph` fills cores population by population along a snake path over tiles and chips, in graph order.
That is one placement among many. This module searches the fill order to minimise the network energy of a step
estimated from population-level traffic, the objective classic SNN mappers use (spike traffic x distance; e.g.
SpiNeMap, PACMAN-style partition-and-place), and returns the order; the full cost model then evaluates it.

Proxy (per step): for every projection, messages = spikes(src) x cores reached per spike (from the given mapping;
independent of placement), times the expected planar hops and chip crossings between a source core and a
destination core: for dense / random projections the expectation over independent cores of the two populations
(|dx| + |dy| are separable, so per-axis histograms give it exactly); for local projections (one-to-one, conv) the
aligned distance between the populations' quantiles. Energy = hops x e_hop + chip hops x (e_chip_hop - e_hop).
Multicast trees and link maxima are not in the proxy; the full model decides (`search` returns both evaluations).

Scope: what a population needs (cores, neurons per core) is the partitioner's job (greedy limits, spikebill.balance);
this baseline moves only where those cores sit. Orders are population indices (each population one contiguous block)
or segment lists [(population index, cores), ...] in which a population may occupy several non-adjacent blocks
(`anneal(..., split=True)`; release gap 23).
"""
import math

import numpy as np

from . import mapping as mapping_mod, traffic

_Q = 64     # quantiles for aligned (local) projections


def _slot_geometry(chip, total):
    slots = mapping_mod._tile_order(chip, math.ceil(total / chip.cores_per_chip))[:total]
    n_chips = int(slots.max() // chip.cores_per_chip) + 1
    xy, ch, _, _, _ = mapping_mod._geometry(chip, slots, n_chips)
    return xy[:, 0].astype(np.int64), xy[:, 1].astype(np.int64)


class Proxy:
    """Network energy per step of a fill order, from population-level traffic (see module docstring)."""

    def __init__(self, graph, chip, mapping, spikes):
        self.names = list(graph.populations)
        self.idx = {k: i for i, k in enumerate(self.names)}
        self.k = np.array([mapping.pops[k].n_cores for k in self.names])
        self.x, self.y = _slot_geometry(chip, int(self.k.sum()))
        self.tx, self.ty = chip.tiles_x, chip.tiles_y
        W, H = int(self.x.max()) + 1, int(self.y.max()) + 1
        self.dx = np.abs(np.arange(W)[:, None] - np.arange(W)[None, :]).astype(float)
        self.dy = np.abs(np.arange(H)[:, None] - np.arange(H)[None, :]).astype(float)
        cx, cy = np.arange(W) // self.tx, np.arange(H) // self.ty
        self.cdx = np.abs(cx[:, None] - cx[None, :]).astype(float)
        self.cdy = np.abs(cy[:, None] - cy[None, :]).astype(float)
        self.W, self.H = W, H
        e_hop, e_chip = chip["e_hop"], chip["e_chip_hop"]
        self.e_hop, self.e_chip = e_hop, e_chip - e_hop
        self.t_link, self.t_chip = chip["t_link"], chip["t_chip_hop"]
        self.edges = []
        for p in graph.projections:
            msgs = spikes.get(p.src, 0.0) * traffic.cores_reached(graph, mapping, p)
            if msgs > 0:
                self.edges.append((self.idx[p.src], self.idx[p.dst], float(msgs), traffic.locality(p) == "local"))
        e = np.array([(a, b) for a, b, _, _ in self.edges], dtype=np.int64).reshape(-1, 2)
        w = np.array([x[2] for x in self.edges])
        loc = np.array([x[3] for x in self.edges], dtype=bool)
        self._dense = (e[~loc, 0], e[~loc, 1], w[~loc])
        self._local = (e[loc, 0], e[loc, 1], w[loc])

    def cost(self, order_idx):
        """order_idx: population indices in fill order -> proxy network energy (J per step)."""
        return self.evaluate(order_idx)[0]

    def segments(self, order):
        """Segment list [(population index, cores)] of an order (index array or segment list), adjacent blocks of
        the same population merged."""
        segs = ([(int(i), int(self.k[i])) for i in order] if len(order) and np.ndim(order[0]) == 0
                else [(int(i), int(c)) for i, c in order])
        out = []
        for i, c in segs:
            if out and out[-1][0] == i:
                out[-1] = (i, out[-1][1] + c)
            elif c > 0:
                out.append((i, c))
        return out

    def evaluate(self, order_idx):
        """(network energy J, busiest-cut time s) of a fill order. The cut time is the largest message load across
        one column / row boundary of the mesh per direction, spread over the links of that cut (H or W), times
        t_link or t_chip_hop (chip boundaries): a bisection-style proxy of the link maxima, calibrated against the
        full model by `calibrate`."""
        segs = self.segments(order_idx)
        P, W, H = len(self.k), self.W, self.H
        owner = np.repeat(np.array([i for i, _ in segs], dtype=np.int64), [c for _, c in segs])   # pop of each slot
        slot = np.argsort(owner, kind="stable")              # slots grouped by population, in path (neuron) order
        off = np.concatenate([[0], np.cumsum(self.k)[:-1]])
        sx, sy = self.x[slot], self.y[slot]
        kk = self.k.astype(float)
        hx = np.bincount(owner * W + self.x[:len(owner)], minlength=P * W).reshape(P, W) / kk[:, None]
        hy = np.bincount(owner * H + self.y[:len(owner)], minlength=P * H).reshape(P, H) / kk[:, None]
        total = 0.0
        cx = np.zeros((2, W - 1)); cy = np.zeros((2, H - 1))      # (direction, cut) message loads
        a, b, w = self._dense
        if len(w):
            hops = ((hx[a] @ self.dx) * hx[b]).sum(1) + ((hy[a] @ self.dy) * hy[b]).sum(1)
            chops = ((hx[a] @ self.cdx) * hx[b]).sum(1) + ((hy[a] @ self.cdy) * hy[b]).sum(1)
            total += float(w @ (hops * self.e_hop + chops * self.e_chip))
            Fx, Fy = np.cumsum(hx, 1)[:, :-1], np.cumsum(hy, 1)[:, :-1]
            cx[0] += w @ (Fx[a] * (1 - Fx[b])); cx[1] += w @ (Fx[b] * (1 - Fx[a]))
            cy[0] += w @ (Fy[a] * (1 - Fy[b])); cy[1] += w @ (Fy[b] * (1 - Fy[a]))
        a, b, w = self._local
        if len(w):
            qi = np.minimum((np.arange(_Q) + 0.5)[None, :] * kk[:, None] / _Q, kk[:, None] - 1).astype(np.int64)
            pos = off[:, None] + qi                                   # (P, Q) indices into the grouped slots
            qx, qy = sx[pos], sy[pos]
            xa, xb, ya, yb = qx[a], qx[b], qy[a], qy[b]
            hops = np.mean(np.abs(xa - xb) + np.abs(ya - yb), 1)
            chops = np.mean(np.abs(xa // self.tx - xb // self.tx) + np.abs(ya // self.ty - yb // self.ty), 1)
            total += float(w @ (hops * self.e_hop + chops * self.e_chip))
            wq = np.repeat(w / _Q, _Q)
            for u, v, c, n in ((xa, xb, cx, W), (ya, yb, cy, H)):     # cuts k with lo <= k < hi, by direction
                u, v = u.ravel(), v.ravel()
                for d, fwd in ((0, u < v), (1, v < u)):
                    lo, hi = np.minimum(u, v), np.maximum(u, v)
                    diff = (np.bincount(lo, wq * fwd, minlength=n + 1) - np.bincount(hi, wq * fwd, minlength=n + 1))
                    c[d] += np.cumsum(diff)[:n - 1]
        kx, ky = np.arange(W - 1), np.arange(H - 1)
        tx_ = np.where((kx + 1) % self.tx == 0, self.t_chip, self.t_link) / max(H, 1)
        ty_ = np.where((ky + 1) % self.ty == 0, self.t_chip, self.t_link) / max(W, 1)
        t_cut = max(float((cx * tx_).max()) if cx.size else 0.0, float((cy * ty_).max()) if cy.size else 0.0)
        return total, t_cut

    def calibrate(self, order_idx, result, objective="edp"):
        """Turn the proxy into an estimate of the full objective around a reference order whose full-model step
        result is `result`: energy outside the network and static terms and the compute / co-processor times are
        held fixed, the network energy moves with the proxy energy, the network time with the cut time (scaled so
        both match the reference), static energy with the step. After this, `score(order)` is the objective."""
        e_net0, t_cut0 = self.evaluate(order_idx)
        en, tm = result["energy"], result["time"]
        self._ref = dict(e_fixed=sum(v for k, v in en.items() if k not in ("network", "static")),
                         e_net_scale=en.get("network", 0.0) / e_net0 if e_net0 else 1.0,
                         t_net_scale=tm["network"] / t_cut0 if t_cut0 else 0.0,
                         t_other=max(tm["compute"], tm.get("coproc", 0.0)), t_bar=tm["barrier"],
                         p_static=en.get("static", 0.0) / tm["step"] if tm["step"] else 0.0, objective=objective)

    def score(self, order_idx):
        """Objective estimate of a fill order (after `calibrate`; before it, the proxy network energy)."""
        ref = getattr(self, "_ref", None)
        if ref is None:
            return self.cost(order_idx)
        e_net, t_cut = self.evaluate(order_idx)
        t = max(ref["t_other"], t_cut * ref["t_net_scale"]) + ref["t_bar"]
        e = ref["e_fixed"] + e_net * ref["e_net_scale"] + ref["p_static"] * t
        return {"edp": e * t, "energy": e, "latency": t}[ref["objective"]]


def spectral_order(proxy):
    """Fiedler-vector order of the symmetric population traffic graph (a min-cut linear arrangement)."""
    P = len(proxy.names)
    A = np.zeros((P, P))
    for a, b, w, _ in proxy.edges:
        if a != b:
            A[a, b] += w; A[b, a] += w
    d = A.sum(1)
    if P < 3 or not (d > 0).any():
        return np.arange(P)
    L = np.diag(d) - A
    _, vec = np.linalg.eigh(L)
    return np.argsort(vec[:, 1], kind="stable")


def bisection_order(proxy, seed=0):
    """Generic graph-partitioner baseline (release gap 30): recursive Kernighan-Lin bisection (networkx) of the
    symmetric population traffic graph; the fill order lists the two halves one after the other at every level, so
    populations that exchange traffic end up in the same part of the path. Edge-free subsets keep graph order.
    Halves have equal population counts (KL balances nodes, not cores)."""
    import networkx as nx
    from networkx.algorithms.community import kernighan_lin_bisection
    G = nx.Graph()
    G.add_nodes_from(range(len(proxy.names)))
    for a, b, w, _ in proxy.edges:
        if a != b:
            G.add_edge(a, b, weight=G.get_edge_data(a, b, {"weight": 0.0})["weight"] + w)
    rng = np.random.default_rng(seed)

    def split(nodes):
        if len(nodes) < 2:
            return list(nodes)
        sub = G.subgraph(nodes)
        if sub.number_of_edges() == 0:
            return sorted(nodes)
        a, b = kernighan_lin_bisection(sub, weight="weight", seed=int(rng.integers(2**31)))
        a, b = sorted(a), sorted(b)
        if min(a) > min(b):                     # keep the half holding the earliest population first
            a, b = b, a
        return split(a) + split(b)
    return np.array(split(list(range(len(proxy.names)))), dtype=np.int64)


def anneal(proxy, init=None, iters=None, seed=0, t0=0.05, t1=1e-4, split=False, max_blocks=4):
    """Simulated annealing over fill orders (moves: swap two blocks, move one, reverse a run of blocks) on
    `proxy.score` (network energy, or the calibrated objective after `proxy.calibrate`). Temperatures are relative
    to the initial cost. split=True also moves part of a population's cores to another position (at most
    `max_blocks` blocks per population) and merges a block back into another block of its population; the order is
    then a segment list. Returns (best order, best score, initial score)."""
    rng = np.random.default_rng(seed)
    P = len(proxy.names)
    cur = np.arange(P) if init is None else init
    if split:
        cur = proxy.segments(cur)
    else:
        cur = np.asarray(cur).copy()
    c_cur = c0 = proxy.score(cur)
    best, c_best = list(cur) if split else cur.copy(), c_cur
    if P < 2 or c0 == 0:
        return best, c_best, c0
    iters = iters or 40 * P * (2 if split else 1)
    for it in range(iters):
        T = c0 * t0 * (t1 / t0) ** (it / max(1, iters - 1))
        if split:
            nxt = _segment_move(proxy, list(cur), rng, max_blocks)
            if nxt is None:
                continue
        else:
            nxt = cur.copy()
            i, j = sorted(rng.choice(P, 2, replace=False))
            mv = rng.integers(3)
            if mv == 0:
                nxt[i], nxt[j] = nxt[j], nxt[i]
            elif mv == 1:
                v = nxt[i]; nxt = np.insert(np.delete(nxt, i), j, v)
            else:
                nxt[i:j + 1] = nxt[i:j + 1][::-1]
        c = proxy.score(nxt)
        if c <= c_cur or rng.random() < math.exp(-(c - c_cur) / T):
            cur, c_cur = nxt, c
            if c < c_best:
                best, c_best = (list(cur) if split else cur.copy()), c
    return best, c_best, c0


def _segment_move(proxy, segs, rng, max_blocks):
    """One random move on a segment list (swap / move / reverse blocks, split a block, merge a block)."""
    n = len(segs)
    mv = rng.integers(5)
    if mv < 3:
        if n < 2:
            return None
        i, j = sorted(rng.choice(n, 2, replace=False))
        if mv == 0:
            segs[i], segs[j] = segs[j], segs[i]
        elif mv == 1:
            segs.insert(j, segs.pop(i))
        else:
            segs[i:j + 1] = segs[i:j + 1][::-1]
    elif mv == 3:                                       # split: move part of a block elsewhere
        cand = [b for b, (p, c) in enumerate(segs) if c >= 2 and sum(q == p for q, _ in segs) < max_blocks]
        if not cand:
            return None
        b = cand[rng.integers(len(cand))]
        p, c = segs[b]
        part = int(rng.integers(1, c))
        segs[b] = (p, c - part)
        segs.insert(int(rng.integers(n + 1)), (p, part))
    else:                                               # merge: give a block's cores to another block of its pop
        multi = [b for b, (p, _) in enumerate(segs) if sum(q == p for q, _ in segs) > 1]
        if not multi:
            return None
        b = multi[rng.integers(len(multi))]
        p, c = segs.pop(b)
        others = [o for o, (q, _) in enumerate(segs) if q == p]
        o = others[rng.integers(len(others))]
        segs[o] = (p, segs[o][1] + c)
    return proxy.segments(segs)


def _full(graph, chip, caps, spikes, active, objective, order):
    """Full-model mapping, step result and score of one fill order."""
    from . import balance, compiled
    m = mapping_mod.map_graph(graph, chip, caps, order=order)
    r = compiled.step_result(compiled.compile_model(graph, chip, m), spikes, active)
    return m, r, balance._score(r, objective)


_CTX = None


def _init_worker(ctx):
    global _CTX
    _CTX = ctx


def _full_worker(order):
    return _full(*_CTX, order)


class _Pool:
    """Worker processes holding the model once (initializer), evaluating fill orders with `_full`."""

    def __init__(self, workers, ctx):
        from concurrent.futures import ProcessPoolExecutor
        self.ex = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(ctx,))

    def map(self, orders):
        return list(self.ex.map(_full_worker, orders))

    def close(self):
        self.ex.shutdown()


def search(graph, chip, spikes, active, caps=None, iters=None, seed=0, objective="edp", calibrated=True,
           split=False, split_rounds=32, split_proposals=8, split_evals=2, workers=1, bisection_seeds=5):
    """Placement baseline for a partition (caps): candidate fill orders = graph order, spectral order, annealing on
    the network-energy proxy (from the better of the two; E20), and, with `calibrated`, annealing on the proxy
    calibrated to the full objective (network energy + busiest-cut time; release gap 24). Each candidate is evaluated
    with the full cost model. Returns dict(order, mapping, result, rows) with the best order under `objective` and one
    row per candidate (proxy and full-model numbers). split (needs calibrated): non-contiguous blocks, two candidates
    whose order is a segment list of (population, cores): `anneal_split` (annealing with split / merge moves on the
    calibrated proxy) and `split_refine` (hill climb from the best candidate so far: per round `split_proposals`
    random block moves, the `split_evals` best under the proxy evaluated with the full model, kept if better; the
    proxy's cut time misses XY-routing hot spots, so the full model decides each step; release gap 23).
    bisection_seeds: that many recursive Kernighan-Lin orders (`bisection_order`, seeds seed..) as further
    candidates, after the others (0 gives the search before release gap 30).
    workers > 1: the full-model evaluations of each split_refine round run in worker processes (same result as
    workers=1: the proposals are accepted in the same order; release gap 28)."""
    m0 = mapping_mod.map_graph(graph, chip, caps)
    px = Proxy(graph, chip, m0, spikes)
    rows, best, full = [], None, {}

    def full_eval(name, o):
        nonlocal best
        order = ([px.names[i] for i in o] if len(o) and np.ndim(o[0]) == 0
                 else [(px.names[i], c) for i, c in px.segments(o)])
        m, r, sc = _full(graph, chip, caps, spikes, active, objective, order)
        rows.append(dict(candidate=name, proxy_J=px.cost(o), chips=m.n_chips, cores=m.n_cores,
                         step=r["time"]["step"], energy=sum(r["energy"].values()),
                         network=r["energy"].get("network", 0.0), score=sc))
        full[name] = r
        if best is None or sc < best[3]:
            best = (order, m, r, sc)

    cands = {"graph_order": np.arange(len(px.names)), "spectral": spectral_order(px)}
    init = min(cands.values(), key=px.cost)
    cands["anneal"] = anneal(px, init=init, iters=iters, seed=seed)[0]
    for name, o in cands.items():
        full_eval(name, o)
    if calibrated:
        px.calibrate(cands["graph_order"], full["graph_order"], objective)
        start = min(cands.values(), key=px.score)
        cands["anneal_calibrated"] = anneal(px, init=start, iters=iters, seed=seed)[0]
        full_eval("anneal_calibrated", cands["anneal_calibrated"])
    for b in range(bisection_seeds):                  # generic-partitioner candidates (E20(f): KL beat the search
        o = bisection_order(px, seed=seed + b)        # where one busy population feeds every block; lost elsewhere)
        cands[f"bisection_{b}"] = o
        full_eval(f"bisection_{b}", o)
    if calibrated:
        if split:
            start = min(cands.values(), key=px.score)
            full_eval("anneal_split", anneal(px, init=start, iters=iters, seed=seed, split=True)[0])
            rng = np.random.default_rng(seed)
            cur = px.segments([(px.idx[e], int(px.k[px.idx[e]])) if isinstance(e, str) else (px.idx[e[0]], int(e[1]))
                               for e in best[0]])
            c_cur, m_cur, r_cur = best[3], best[1], best[2]
            pool = _Pool(workers, (graph, chip, caps, spikes, active, objective)) if workers > 1 else None
            try:
                for _ in range(split_rounds):
                    props = [o for o in (_segment_move(px, list(cur), rng, 4) for _ in range(split_proposals)) if o]
                    picks = sorted(props, key=px.score)[:split_evals]
                    orders = [[(px.names[i], c) for i, c in o] for o in picks]
                    outs = pool.map(orders) if pool else [_full(graph, chip, caps, spikes, active, objective, od)
                                                          for od in orders]
                    for o, (m, r, sc) in zip(picks, outs):
                        if sc < c_cur:
                            cur, c_cur, m_cur, r_cur = o, sc, m, r
            finally:
                if pool:
                    pool.close()
            rows.append(dict(candidate="split_refine", proxy_J=px.cost(cur), chips=m_cur.n_chips, cores=m_cur.n_cores,
                             step=r_cur["time"]["step"], energy=sum(r_cur["energy"].values()),
                             network=r_cur["energy"].get("network", 0.0), score=c_cur, blocks=len(cur)))
            if c_cur < best[3]:
                best = ([(px.names[i], c) for i, c in cur], m_cur, r_cur, c_cur)
    return dict(order=best[0], mapping=best[1], result=best[2], rows=rows)
