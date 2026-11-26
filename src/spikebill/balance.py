"""Load balancing: give hot populations fewer neurons per core so that no core sets the step time alone.

Per-core time of a population is roughly proportional to its neurons per core (synaptic events, messages in and
neuron updates all scale with the neurons a core holds). For a target per-core time T, every population whose
busiest core exceeds T gets its neurons per core scaled by T / t_pop (rounded down). More cores shorten the step but
can add chips (static power, longer paths), so `balance` tries a range of targets and keeps the mapping that
minimises the objective: "edp" (energy x step time, default), "energy" or "latency".

Targets: `n_targets` geometric steps from the busiest population's time down to the median population time
(both from the greedy mapping; neither depends on placement). Until 2026-11-24 the range also stopped at the
greedy mapping's network time, which moves by +-15% with placement: two near-identical graphs then searched
different ranges and landed 13% apart in latency (E18). The objective is flat near its optimum (many mappings
within a few percent), so `band()` reports the energy / latency / chips spread of the near-optimal candidates.
"""
import numpy as np

from . import compiled, mapping as mapping_mod


def _evaluate(graph, chip, spikes, active, caps):
    m = mapping_mod.map_graph(graph, chip, caps)
    r = compiled.step_result(compiled.compile_model(graph, chip, m), spikes, active)
    return m, r


def _score(r, objective):
    e, t = sum(r["energy"].values()), r["time"]["step"]
    return {"edp": e * t, "energy": e, "latency": t}[objective]


def pop_times(m, r):
    """Busiest-core time of each population."""
    return {k: float(r["core_time"][p.cores].max()) for k, p in m.pops.items()}


def _placed(graph, chip, spikes, active, caps, objective, split=False):
    from . import place
    ps = place.search(graph, chip, spikes, active, caps=caps, objective=objective, split=split)
    return ps["mapping"], ps["result"]


def _try_target(args):
    graph, chip, spikes, active, T, iters, m0_caps = args[:7]
    place_obj = args[7] if len(args) > 7 else None
    caps = dict(m0_caps)
    m, r = _evaluate(graph, chip, spikes, active, caps)
    for _ in range(iters):                         # re-map, re-measure: neighbouring populations change too
        for k, t in pop_times(m, r).items():
            if t > T:
                caps[k] = max(1, int(m.pops[k].per_core * T / t))
        m, r = _evaluate(graph, chip, spikes, active, caps)
    if place_obj:                                  # joint: search the placement of this candidate's partition
        m, r = _placed(graph, chip, spikes, active, caps, place_obj, split=len(args) > 8 and args[8])
    return T, caps, m, r


def balance(graph, chip, spikes, active, objective="edp", targets=None, iters=2, workers=1, n_targets=16,
            max_chips=None, max_energy_increase=None, place=False, margin=0.0):
    """Returns (best mapping, its step result, caps, history). history: one dict per candidate (target, chips,
    cores, step, energy per step, score, eligible); `band(history)` gives the near-optimal spread. workers > 1 tries
    the targets in parallel. Budgets (the objective is flat near its optimum, so the reported point otherwise depends
    on the search range): max_chips keeps candidates with at most that many chips; max_energy_increase (e.g. 0.05)
    keeps candidates within that fraction of the lowest-energy candidate. The best eligible candidate under
    `objective` is returned; if none is eligible, the one with the fewest chips (then lowest energy).
    place: also search the placement (spikebill.place.search) of every candidate partition, the greedy one included
    (joint partition + placement; one placement search per target); place="split" also lets populations occupy
    non-contiguous blocks (spikebill.place.search(split=True)).
    margin: headroom (E29). Candidates are generated and chosen under spike counts x (1 + margin), so the chosen
    mapping keeps slack on its busiest cores when the real rates run higher than the given ones; the returned step
    result and the history's step / energy are at the nominal spikes (score and eligibility at the raised ones).
    Not recommended as a default: on SDT V1 a uniform margin of 0.2 recovers 31% of the tail-aware mapping's held-out
    latency gain and costs up to 12% nominal EDP on the zoo (E29); with per-input samples, use
    Simulator.spread(map_rates=95)."""
    if margin < 0:
        raise ValueError("margin must be >= 0")
    nominal = spikes
    if margin:
        spikes = {k: v * (1.0 + margin) for k, v in spikes.items()}
    m0, r0 = _evaluate(graph, chip, spikes, active, {})
    t_pop0 = pop_times(m0, r0)                     # target range from the graph-order greedy mapping
    if place:
        m0, r0 = _placed(graph, chip, spikes, active, {}, objective, split=place == "split")
    cands = [(m0, r0, {})]
    hist = [dict(target=None, chips=m0.n_chips, cores=m0.n_cores, step=r0["time"]["step"],
                 energy=sum(r0["energy"].values()), score=_score(r0, objective))]
    tmax = max(t_pop0.values())
    if targets is None:
        lo = min(float(np.median(list(t_pop0.values()))), tmax)
        targets = np.geomspace(tmax, lo, n_targets + 1)[1:] if lo < tmax else []
    jobs = [(graph, chip, spikes, active, float(T), iters, {}) + ((objective, place == "split") if place else ()) for T in targets]
    if workers > 1:
        from . import sweep
        outs = sweep.run(_try_target, jobs, workers=workers)
    else:
        outs = [_try_target(j) for j in jobs]
    for T, caps, m, r in outs:
        cands.append((m, r, dict(caps)))
        hist.append(dict(target=float(T), chips=m.n_chips, cores=m.n_cores, step=r["time"]["step"],
                         energy=sum(r["energy"].values()), score=_score(r, objective), n_capped=len(caps)))
    e_min = min(h["energy"] for h in hist)
    for h in hist:
        h["eligible"] = ((max_chips is None or h["chips"] <= max_chips)
                         and (max_energy_increase is None or h["energy"] <= e_min * (1 + max_energy_increase)))
    ok = [i for i, h in enumerate(hist) if h["eligible"]]
    i = (min(ok, key=lambda j: hist[j]["score"]) if ok
         else min(range(len(hist)), key=lambda j: (hist[j]["chips"], hist[j]["energy"])))
    if margin:                                     # report at the nominal spikes
        for h, (mc, _, _) in zip(hist, cands):
            rn = compiled.step_result(compiled.compile_model(graph, chip, mc), nominal, active)
            h.update(step=rn["time"]["step"], energy=sum(rn["energy"].values()), score_margin=h["score"],
                     score=_score(rn, objective))
            if mc is cands[i][0]:
                r_nom = rn
        m, _, caps = cands[i]
        return m, r_nom, caps, hist
    m, r, caps = cands[i]
    return m, r, caps, hist


def band(hist, tol=0.05):
    """Spread of the near-optimal candidates in a `balance` history: every mapping whose score is within `tol` of
    the best. Returns dict(n, tol, energy=(min, max), step=(min, max), chips=(min, max)) per chip step."""
    pool = [h for h in hist if h.get("eligible", True)] or hist
    key = "score_margin" if "score_margin" in pool[0] else "score"     # the score the candidates were chosen by
    best = min(h[key] for h in pool)
    near = [h for h in pool if h[key] <= best * (1 + tol)]
    rng = lambda k: (min(h[k] for h in near), max(h[k] for h in near))
    return dict(n=len(near), tol=tol, energy=rng("energy"), step=rng("step"), chips=rng("chips"))
