"""Functional simulation of a population graph on GPU, to produce the activity the cost model needs.

Populations: size, threshold, bias, leak_decay (potential), input_decay (current), reset.
Projections: explicit edge lists (pre, post, weight) -> sparse matrices.
Two semantics:
  "loihi"      spikes reach their targets one step later; per step: current *= input_decay, v *= leak_decay
               (not on the first step), v quantised to 1/64 (truncation), v += bias + current, spike if v > threshold,
               hard reset to `reset`. A neuron is 'active' (updated) if v != 0 before the update, it received a
               message this step, or bias != 0 (SANA-FE's rule for Loihi). Matches SANA-FE's LoihiLifModel.
  "framework"  typical training-framework (snnTorch-style) semantics: same-step propagation through the graph in
               population order (recurrent edges use the previous step), v = leak * v + bias + input, spike if
               v >= threshold, subtractive reset, no quantisation.
Returns per-step spike counts and active counts per population and per-neuron spike totals.
"""
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class FPop:
    name: str
    size: int
    threshold: float
    bias: np.ndarray = None
    leak_decay: float = 1.0
    input_decay: float = 0.0
    reset: float = 0.0


@dataclass
class FProj:
    src: str
    dst: str
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray


@dataclass
class FuncResult:
    spikes: list            # per step: {pop: count}
    active: list            # per step: {pop: count}
    neuron_spikes: dict     # pop -> np.ndarray of total spikes per neuron
    steps: int
    meta: dict = field(default_factory=dict)


def _sparse(proj, n_src, n_dst, device):
    idx = torch.tensor(np.stack([proj.post, proj.pre]), dtype=torch.long, device=device)
    val = torch.tensor(proj.weight, dtype=torch.float64, device=device)
    return torch.sparse_coo_tensor(idx, val, (n_dst, n_src)).coalesce()


def simulate(pops, projs, steps, semantics="loihi", device=None, record=(), inputs=None):
    """record: steps whose spiking-neuron indices are returned in result.meta["spiking"][t][pop].
    inputs: {pop: array (steps, size)} of externally given spikes (the population is driven, not simulated)."""
    inputs = inputs or {}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = [p.name for p in pops]
    P = {p.name: p for p in pops}
    W = {}
    for q in projs:
        W.setdefault(q.dst, []).append((q.src, _sparse(q, P[q.src].size, P[q.dst].size, device),
                                        _sparse(FProj(q.src, q.dst, q.pre, q.post, np.ones(len(q.pre))),
                                                P[q.src].size, P[q.dst].size, device)))
    order = {n: i for i, n in enumerate(names)}
    f64 = dict(dtype=torch.float64, device=device)
    v = {n: torch.zeros(P[n].size, **f64) for n in names}
    cur = {n: torch.zeros(P[n].size, **f64) for n in names}
    bias = {n: torch.as_tensor(P[n].bias if P[n].bias is not None else np.zeros(P[n].size), **f64) for n in names}
    prev = {n: torch.zeros(P[n].size, **f64) for n in names}      # spikes of the previous step
    tot = {n: torch.zeros(P[n].size, **f64) for n in names}
    spikes_t, active_t = [], []
    record, spiking = set(record), {}
    for t in range(steps):
        now = {}
        s_cnt, a_cnt = {}, {}
        for n in names:
            p = P[n]
            if n in inputs:                                  # externally driven population
                sp = torch.as_tensor(np.asarray(inputs[n][t]).reshape(-1), **f64)
                sp = (sp != 0).to(torch.float64)
                now[n] = sp; tot[n] += sp
                s_cnt[n] = float(sp.sum()); a_cnt[n] = 0.0
                continue
            inp = torch.zeros(p.size, **f64)
            got = torch.zeros(p.size, dtype=torch.bool, device=device)
            for src, w, ones in W.get(n, []):
                if semantics == "loihi" or order[src] >= order[n]:
                    s = prev[src]                 # delayed (Loihi) or recurrent edge
                else:
                    s = now[src]                  # same-step feed-forward (framework)
                if s.any():
                    inp += torch.sparse.mm(w, s[:, None])[:, 0]
                    got |= torch.sparse.mm(ones, s[:, None])[:, 0] > 0
            active = (v[n] != 0) | got | (bias[n] != 0)    # the chip's update rule, under either semantics
            if semantics == "loihi":
                if t > 0:
                    cur[n] = cur[n] * p.input_decay
                    v[n] = v[n] * p.leak_decay
                v[n] = torch.trunc(v[n] * 64.0) / 64.0
                cur[n] = cur[n] + inp
                v[n] = v[n] + bias[n] + cur[n]
                fire = v[n] > p.threshold
                v[n] = torch.where(fire, torch.full_like(v[n], p.reset), v[n])
            else:
                v[n] = v[n] * p.leak_decay + bias[n] + inp
                fire = v[n] >= p.threshold
                v[n] = torch.where(fire, v[n] - p.threshold, v[n])
            sp = fire.to(torch.float64)
            now[n] = sp
            tot[n] += sp
            s_cnt[n] = float(sp.sum())
            a_cnt[n] = float(active.sum())
        prev = now
        if t in record:
            spiking[t] = {n: torch.nonzero(now[n]).flatten().cpu().numpy() for n in names}
        spikes_t.append(s_cnt)
        active_t.append(a_cnt)
    return FuncResult(spikes_t, active_t, {n: tot[n].cpu().numpy() for n in names}, steps,
                      dict(semantics=semantics, device=str(device), spiking=spiking))
