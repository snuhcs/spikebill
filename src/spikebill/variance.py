"""Spike-count fluctuations: how far the mean-count step is from the expected step (E22).

The cost model takes core loads from mean spike counts, i.e. the busiest core of the expected loads (max of
expectations). `busiest_core` draws per-core spikes as Binomial(neurons of the population on the core, rate),
independently per neuron and step, recomputes the sender-side core terms (spike generation, message copies) from the
drawn per-core counts and the receive / network / co-processor terms from the drawn population totals, and reports
R = E[step] / step(mean counts) and the same ratio for the compute term. Independent spikes are the optimistic case;
correlated (bursty) activity raises the ratios. Neuron updates / active neurons stay at their means.
"""
import numpy as np

from . import compiled

_trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy >= 2.0 renamed it; np.trapz is deprecated there


def busiest_core(cm, spikes, active, steps=200, seed=0):
    """cm: compiled model; spikes, active: {pop: mean count per chip step}. Returns dict(R, R_compute, step_ref_s,
    step_mean_s, step_p95_s, compute_ref_s, compute_mean_s)."""
    chip = cm.chip
    S0 = np.array([float(spikes.get(k, 0.0)) for k in cm.pops])
    A0 = np.array([float(active.get(k, 0.0)) for k in cm.pops])
    cnt = cm.cnt.tocoo()
    rows, cols = cnt.row, cnt.col
    n = np.rint(cnt.data * cm.sizes[cols]).astype(np.int64)
    r = np.clip(S0 / np.maximum(cm.sizes, 1), 0, 1)
    fired = np.asarray(cm.fired.tocsr()[rows, cols]).ravel()
    out = np.asarray(cm.core_out.tocsr()[rows, cols]).ravel()
    ratio = np.where(fired > 0, out / np.where(fired > 0, fired, 1), 0.0)
    C = cm.n_cores
    act_c = cm.cnt @ A0
    upd_c = cm.neurons if chip.neuron_update == "all" else act_c
    base_c = upd_c * chip["t_neuron_update"] + act_c * chip["t_neuron_active"]
    rng = np.random.default_rng(seed)
    X = rng.binomial(n[None, :], r[cols][None, :], size=(steps, len(n))).astype(float)
    S = np.zeros((steps, len(cm.pops)))
    for j in np.unique(cols):
        S[:, j] = X[:, cols == j].sum(1)
    tin = (cm.core_in @ S.T) * chip["t_msg_in"] + cm.core_synt @ S.T
    t_comp = np.empty(steps)
    for t in range(steps):
        gen = np.bincount(rows, X[t], minlength=C)
        msgs = np.bincount(rows, X[t] * ratio, minlength=C)
        t_neur = base_c + gen * chip["t_spike_gen"] + msgs * chip["t_spike_out"]
        tc = np.maximum(t_neur, tin[:, t]) if chip.core_timing == "max" else t_neur + tin[:, t]
        t_comp[t] = tc.max()
    ev = compiled.evaluate(cm, S, np.repeat(A0[None, :], steps, 0))
    others = ev["network_t"] if "coproc_t" not in ev else np.maximum(ev["network_t"], ev["coproc_t"])
    step = np.maximum(t_comp, others) + chip.barrier(cm.n_tiles, cm.n_chips)
    ref = compiled.evaluate(cm, S0[None, :], A0[None, :])
    return dict(R=float(step.mean() / ref["step"][0]), R_compute=float(t_comp.mean() / ref["compute"][0]),
                step_ref_s=float(ref["step"][0]), step_mean_s=float(step.mean()),
                step_p95_s=float(np.percentile(step, 95)), compute_ref_s=float(ref["compute"][0]),
                compute_mean_s=float(t_comp.mean()))


def _sender_moments(cm, spikes, active):
    """Per core: deterministic floor, mean and variance of the neuron-side time (Binomial spikes, as busiest_core),
    and the receive-side time at mean counts."""
    chip = cm.chip
    S0 = np.array([float(spikes.get(k, 0.0)) for k in cm.pops])
    A0 = np.array([float(active.get(k, 0.0)) for k in cm.pops])
    cnt = cm.cnt.tocoo()
    rows, cols = cnt.row, cnt.col
    n = np.rint(cnt.data * cm.sizes[cols])
    r = np.clip(S0 / np.maximum(cm.sizes, 1), 0, 1)[cols]
    fired = np.asarray(cm.fired.tocsr()[rows, cols]).ravel()
    out = np.asarray(cm.core_out.tocsr()[rows, cols]).ravel()
    w = chip["t_spike_gen"] + np.where(fired > 0, out / np.where(fired > 0, fired, 1), 0.0) * chip["t_spike_out"]
    C = cm.n_cores
    act_c = cm.cnt @ A0
    upd_c = cm.neurons if chip.neuron_update == "all" else act_c
    base = upd_c * chip["t_neuron_update"] + act_c * chip["t_neuron_active"]
    mu = base + np.bincount(rows, n * r * w, minlength=C)
    var = np.bincount(rows, n * r * (1 - r) * w * w, minlength=C)
    tin = cm.core_in @ S0 * chip["t_msg_in"] + cm.core_synt @ S0
    single = np.bincount(rows, (n * r > 0).astype(float), minlength=C) == 1      # one spiking population: exact
    idx = np.full(C, -1)
    hot = n * r > 0
    idx[rows[hot]] = np.nonzero(hot)[0]
    binom = dict(single=single, base=base, n=n[np.maximum(idx, 0)], r=r[np.maximum(idx, 0)],
                 w=w[np.maximum(idx, 0)])
    return mu, np.sqrt(var), np.asarray(tin).ravel(), binom


def _emax(mu, sd, tin, bn, core_timing, z, grid):
    """E[max_c Y_c] over independent cores for a fixed receive-side time t_in per core (see expected_compute)."""
    from scipy.special import log_ndtr
    from scipy.stats import binom as binom_dist
    if core_timing != "max":
        mu, bn, tin = mu + tin, dict(bn, base=bn["base"] + tin), np.zeros_like(tin)
    lo, hi, ref = np.maximum(tin, mu - z * sd), np.maximum(tin, mu + z * sd), np.maximum(tin, mu)
    y0, t_ref = float(lo.max()), float(ref.max())
    keep = (hi > y0) & (sd > 0)
    if not keep.any():
        return t_ref
    cols = np.column_stack([mu[keep], sd[keep], tin[keep]] + [np.asarray(bn[k], float)[keep]
                                                              for k in ("base", "w", "n", "r", "single")])
    cols, cnt = np.unique(cols, axis=0, return_counts=True)      # identical cores (same population, same load) once
    mu, sd, tin = cols[:, 0], cols[:, 1], cols[:, 2]
    bn = dict(base=cols[:, 3], w=cols[:, 4], n=cols[:, 5], r=cols[:, 6], single=cols[:, 7] > 0)
    y = np.linspace(y0, float(hi.max()), grid)
    logF = np.zeros(grid)
    for a in range(0, len(mu), 512):                     # chunk over cores: (cores x grid) blocks
        sl = slice(a, a + 512)
        lf = log_ndtr((y[None, :] - mu[sl, None]) / sd[sl, None])
        k = np.floor((y[None, :] - bn["base"][sl, None]) / bn["w"][sl, None] * (1 + 1e-12))
        with np.errstate(divide="ignore"):
            lb = np.log(binom_dist.cdf(k, bn["n"][sl, None], bn["r"][sl, None]))
        lf = np.where(bn["single"][sl, None], lb, lf)
        logF += (np.where(y[None, :] >= tin[sl, None], lf, -np.inf) * cnt[sl, None]).sum(0)
    return max(t_ref, y0 + float(_trapz(1.0 - np.exp(logF), y)))


def expected_compute(cm, spikes, active, draws=32, seed=0, z=8.0, grid=4096):
    """Expected busiest-core compute time under independent Binomial spikes (the model of `busiest_core`), without
    per-core sampling (release gap 25). Sender side (spike generation, message copies): per core exact (scaled
    Binomial) when the core holds one spiking population, Gaussian with matched moments otherwise; cores independent;
    E[max_c Y_c] = y0 + int_{y0}^inf (1 - prod_c F_c(y)) dy, Y_c = max(X_c, t_in_c) (core_timing "max") or
    X_c + t_in_c ("sum"). Receive side (t_in, linear in the population totals, shared by all cores receiving from a
    population): if it can set the busiest core (within z standard deviations), `draws` Binomial draws of the
    population totals (seeded), each integrated as above, averaged; otherwise held at its mean (no sampling).
    The sender counts and the totals are drawn independently (ignores their dependence; E22(b)).
    Returns (expected compute time, compute time at mean counts)."""
    mu, sd, tin, bn = _sender_moments(cm, spikes, active)
    ct = cm.chip.core_timing
    t_ref = float((np.maximum(tin, mu) if ct == "max" else mu + tin).max())
    chip = cm.chip
    S0 = np.array([float(spikes.get(k, 0.0)) for k in cm.pops])
    N = np.asarray(cm.sizes, float)
    r = np.clip(S0 / np.maximum(N, 1), 0, 1)
    Ain = (cm.core_in * chip["t_msg_in"] + cm.core_synt).tocsr()
    sd_in = np.sqrt(np.asarray(Ain.multiply(Ain) @ (N * r * (1 - r))).ravel())
    y0 = float(np.maximum(tin, mu - z * sd).max())
    if not (sd_in > 0).any() or (ct == "max" and float((tin + z * sd_in).max()) <= y0):
        return _emax(mu, sd, tin, bn, ct, z, grid), t_ref     # receive side cannot set the busiest core
    rng = np.random.default_rng(seed)
    Nn = np.rint(N).astype(np.int64)
    vals = [_emax(mu, sd, np.asarray(Ain @ rng.binomial(Nn, r).astype(float)).ravel(), bn, ct, z, grid)
            for _ in range(draws)]
    return float(np.mean(vals)), t_ref
