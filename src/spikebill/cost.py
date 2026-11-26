"""Energy and latency per step and per inference from a graph, a chip, a mapping and activity.

Activity per step: spikes {pop: spikes} and active {pop: neurons with input/state to update}.
Per step:
  energy  synaptic = sum over projections of synaptic events * e_synop(@synapse type)
          neuron   = all mapped neurons (or active only, chip.neuron_update) * e_neuron_update + active * e_neuron_active + spikes * e_spike_gen
          network  = messages * e_spike_out + tile hops * e_hop + chip hops * e_chip_hop + vertical hops * e_vhop
          static   = static_power(chip, mapping) * step time  (accounting: chip | cores | none)
  time    neuron side  = neurons * t_neuron_update + active * t_neuron_active + spikes * t_spike_gen
                         + messages out * t_spike_out
          message side = messages in * t_msg_in + synaptic events * t_synop
          core     = max(neuron side, message side) if chip.core_timing == "max" (pipelined; default), else sum
          compute  = max over cores of core time   (activity spread over a population's cores by neuron count)
          network  = max(busiest tile link * t_link, busiest chip link * t_chip_hop, busiest vertical link * t_vlink)
                     + longest path latency (planar hops * t_hop + vertical hops * t_vhop)
          step     = max(compute, network) + barrier(used tiles, chips)
The compute/network overlap and the max over cores make this a bound, not a schedule; E1/H2 test it.
"""
import math

import numpy as np

from . import mapping as mapping_mod
from . import traffic as traffic_mod


def _per_core(mapping, values, by_activity=False):
    """Spread {pop: amount} over each population's cores by neuron count (or by activity share)."""
    out = np.zeros(mapping.n_cores)
    for name, p in mapping.pops.items():
        v = float(values.get(name, 0.0))
        out[p.cores] += v * (p.share if by_activity else p.counts / p.counts.sum())
    return out


def static_power(chip, mapping):
    """Static power under the chip's accounting convention (chip.static_accounting):
    "chip"  every powered chip in full (system view);
    "cores" pro-rated by the fraction of cores used (NxSDK / TrueNorth practice);
    "none"  dynamic energy only (SANA-FE practice)."""
    mode = chip.static_accounting
    if mode == "none":
        return 0.0
    if mode == "cores":
        return (chip["p_static_core"] * mapping.n_cores
                + chip["p_static_chip"] * mapping.n_cores / chip.cores_per_chip)
    return chip["p_static_core"] * mapping.n_cores + chip["p_static_chip"] * mapping.n_chips


def coproc_work(graph, spikes):
    """Co-processor operations and messages to it in one step (spikes: {pop: count})."""
    ops = sum(float(spikes.get(k, 0.0)) * p.coproc_ops for k, p in graph.populations.items())
    msgs = sum(float(spikes.get(k, 0.0)) * p.coproc_msgs for k, p in graph.populations.items())
    return ops, msgs


def coproc_unit_ops(graph, spikes):
    """Co-processor operations per unit in one step: {unit label: ops} (label = Population.coproc_unit or the name)."""
    out = {}
    for k, p in graph.populations.items():
        if p.coproc_ops:
            u = p.coproc_unit or k
            out[u] = out.get(u, 0.0) + float(spikes.get(k, 0.0)) * p.coproc_ops
    return out


def coproc_time(chip, graph, spikes, ops):
    """Co-processor time per step: pooled = all ops on one unit; per_unit = the busiest unit."""
    if chip.coproc_units == "per_unit":
        ops = max(coproc_unit_ops(graph, spikes).values(), default=0.0)
    return ops * chip.c("t_coproc_op")


def core_times(chip, mapping, tr, spikes, active, graph):
    """Per-core time of one step (neuron side vs message side, combined per chip.core_timing)."""
    neurons = _per_core(mapping, {k: p.size for k, p in graph.populations.items()})
    act = _per_core(mapping, active)
    fired = _per_core(mapping, spikes, by_activity=True)
    upd = neurons if chip.neuron_update == "all" else act
    t_neur = (upd * chip["t_neuron_update"] + act * chip["t_neuron_active"] + fired * chip["t_spike_gen"]
              + tr.core_out_msgs * chip["t_spike_out"])
    t_msg = tr.core_in_msgs * chip["t_msg_in"] + tr.core_synop_time
    # "max": neuron-side and message-side pipelines overlap (SANA-FE's scheduler); "sum": serial (E1 model)
    t_core = np.maximum(t_neur, t_msg) if chip.core_timing == "max" else t_neur + t_msg
    return t_core, neurons, act, fired


def step_cost(graph, chip, mapping, spikes, active, static=True):
    """Reference implementation of one chip step (docs/cost_model.md Secs. 3-5): spikes / active {pop: count}.
    Returns energy components, time terms, counts, per-core times and the traffic object."""
    tr = traffic_mod.step_traffic(graph, mapping, chip, spikes)
    t_core, neurons, act, fired = core_times(chip, mapping, tr, spikes, active, graph)
    t_compute = float(t_core.max()) if len(t_core) else 0.0
    t_network = (max(tr.max_link_intra * chip["t_link"], tr.max_link_inter * chip["t_chip_hop"],
                     tr.max_link_vert * chip.v("t_vlink"))
                 + tr.max_path_time)
    t_bar = chip.barrier(mapping.n_tiles, mapping.n_chips)
    ops, cmsg = coproc_work(graph, spikes)
    t_cop = coproc_time(chip, graph, spikes, ops) if ops or cmsg else 0.0
    t_step = max(t_compute, t_network, t_cop) + t_bar
    p_static = static_power(chip, mapping)
    e = dict(
        synaptic=tr.synop_energy,
        neuron=float((neurons if chip.neuron_update == "all" else act).sum()) * chip["e_neuron_update"] + float(act.sum()) * chip["e_neuron_active"]
               + float(fired.sum()) * chip["e_spike_gen"],
        network=tr.messages * chip["e_spike_out"] + (tr.hops - tr.chip_hops) * chip["e_hop"]
                + tr.chip_hops * chip["e_chip_hop"] + tr.vhops * chip.v("e_vhop"),
        static=p_static * t_step if static else 0.0,
    )
    tm = dict(compute=t_compute, network=t_network, barrier=t_bar, step=t_step)
    if ops or cmsg:
        e["coproc"] = ops * chip.c("e_coproc_op") + cmsg * chip["e_spike_out"]
        tm["coproc"] = t_cop
    return dict(energy=e, time=tm,
                counts=dict(spikes=float(fired.sum()), active=float(act.sum()), synops=float(tr.core_synops.sum()),
                            messages=tr.messages, hops=tr.hops, chip_hops=tr.chip_hops, vhops=tr.vhops), traffic=tr,
                core_time=t_core)


def mean_activity(graph, rates=None, active_frac=None):
    rates, active_frac = rates or {}, active_frac or {}
    spikes = {k: p.size * rates.get(k, p.rate) for k, p in graph.populations.items()}
    active = {k: p.size * active_frac.get(k, p.active) for k, p in graph.populations.items()}
    return spikes, active


def estimate(graph, chip, rates=None, active_frac=None, mapping=None):
    """Cost per inference at mean activity (rates: spikes/neuron/step; active_frac: fraction per step)."""
    m = mapping or mapping_mod.map_graph(graph, chip)
    spikes, active = mean_activity(graph, rates, active_frac)
    r = step_cost(graph, chip, m, spikes, active)
    e, T = r["energy"], graph.steps_per_inference
    tot = sum(e.values())
    return dict(
        model=graph.name, chip=chip.name, which=chip.which, static_accounting=chip.static_accounting, cores=m.n_cores, tiles=m.n_tiles, chips=m.n_chips,
        board=m.board, steps=T,
        energy_per_step=e, energy_per_inference=tot * T,
        energy_split={k: v / tot if tot else 0.0 for k, v in e.items()},
        time_per_step=r["time"], latency_per_inference=r["time"]["step"] * T,
        counts_per_step=r["counts"],
        mapping={k: dict(n_cores=p.n_cores, per_core=p.per_core, limited_by=p.limited_by) for k, p in m.pops.items()},
        warnings=m.warnings,
    )


def estimate_trace(graph, chip, mapping, spikes_t, active_t, static=False):
    """Per-step cost for a list of per-step activity dicts {pop: count}. Uses the compiled model (fast, batched);
    identical to step_cost per step (tests/test_compiled.py). Returns arrays: energy by component, times, counts."""
    from . import compiled
    cm = compiled.compile_model(graph, chip, mapping)
    S = np.array([[float(s.get(k, 0.0)) for k in cm.pops] for s in spikes_t])
    A = np.array([[float(a.get(k, 0.0)) for k in cm.pops] for a in active_t])
    ev = compiled.evaluate(cm, S, A, static=static)
    t_bar = chip.barrier(mapping.n_tiles, mapping.n_chips)
    return dict(energy={k: ev[k] for k in ("synaptic", "neuron", "network", "static", "coproc") if k in ev},
                time=dict(compute=ev["compute"], network=ev["network_t"], barrier=np.full(len(S), t_bar), step=ev["step"]),
                counts={k: ev[k] for k in ("spikes", "active", "synops", "messages", "hops", "chip_hops")})


def op_count_estimate(graph, rates=None, e_ac=0.9e-12, e_mac=4.6e-12, ann_equiv=False):
    """The common '45 nm op count' estimate: synops * E_AC per inference (or MACs * E_MAC for the ANN)."""
    if ann_equiv:
        return sum(graph.n_synapses(p) for p in graph.projections) * e_mac
    return sum(graph.syn_events_per_step(rates).values()) * e_ac * graph.steps_per_inference


def first_order(graph, chip, rates=None, steps=None, execution="pipelined"):
    """The 'spreadsheet' estimate with the same chip parameters, no mapping or network model (a baseline that shows
    what the system model adds):
      chips  = max(neurons / (cores_per_chip * max_neurons), weight bits / (cores_per_chip * syn_mem_bits)),
               weights stored once per shared-weight projection, rounded up;
      step   = (synaptic events * t_synop + neurons * t_neuron_update + spikes * t_spike_gen) / cores
               + barrier, i.e. work spread perfectly over all cores of those chips, no network;
      energy = events * e_synop + neurons * e_neuron_update + spikes * e_spike_gen + chips * p_static_chip * step,
               per chip step, times the chip steps of an inference (T + depth pipelined, or T).
    rates: spikes per neuron per SNN step (default: each population's rate)."""
    from .api import depth
    rates = rates or {}
    T = steps or graph.steps_per_inference or 1
    chip_steps = T + depth(graph) if execution == "pipelined" else T
    scale = T / chip_steps
    N = sum(p.size for p in graph.populations.values())
    spikes = sum(p.size * rates.get(k, p.rate) for k, p in graph.populations.items()) * scale
    events = sum(graph.syn_events_per_step(rates).values()) * scale
    bits = 0.0
    for p in graph.projections:
        n = graph.n_synapses(p)
        if p.shared_weights:
            n = p.n_weights if p.n_weights else n / max(1, p.share_group or 1)
        bits += n * p.weight_bits
    cpc = chip.cores_per_chip
    chips = max(1, math.ceil(max(N / (cpc * chip["max_neurons"]), bits / (cpc * chip["syn_mem_bits"]))))
    cores = chips * cpc
    t_bar = chip.barrier(chips * chip.tiles_per_chip, chips)
    sp_ = {k: p.size * rates.get(k, p.rate) * scale for k, p in graph.populations.items()}
    ops, cmsg = coproc_work(graph, sp_)
    t_core = (events * chip["t_synop"] + N * chip["t_neuron_update"] + spikes * chip["t_spike_gen"]) / cores
    t_cop = coproc_time(chip, graph, sp_, ops) if ops else 0.0
    t_step = max(t_core, t_cop) + t_bar
    e = dict(synaptic=events * chip["e_synop"], neuron=N * chip["e_neuron_update"] + spikes * chip["e_spike_gen"],
             static=chips * chip["p_static_chip"] * t_step)
    if ops or cmsg:
        e["coproc"] = ops * chip.c("e_coproc_op") + cmsg * chip["e_spike_out"]
    return dict(chips=chips, chip_steps=chip_steps, step_s=t_step, latency_s=t_step * chip_steps,
                t_core_s=t_core, t_cop_s=t_cop, t_barrier_s=t_bar,
                energy_J=sum(e.values()) * chip_steps, split={k: v / sum(e.values()) for k, v in e.items()},
                weight_bits=bits, events_per_step=events)


def roofline(graph, chip, mapping, spikes, active, chip_steps=1, coproc=True):
    """Floorline-style per-core / per-link roofline with the same parameters and the same mapping (a stronger baseline
    than `first_order`; E17): step = max(slowest core, link roofs, co-processor) + barrier, where the link roofs spread
    the step's on-chip hops uniformly over all tile links and its chip-to-chip hops over all chip links (no routing,
    no hot spots, no path latency). Dynamic energy is the full model's; static energy follows the roofline step.
    spikes / active: {pop: count per chip step} (as `Simulator._activity` returns). coproc=False drops the co-processor
    roof (E17's definition). Returns per-inference energy / latency, the binding roof, and the full model's values on
    the same mapping for comparison: the roofline matches the full model while a core binds and misses link hot spots
    (Meta-SpikeFormer 55M on 506 chips: 3.2x low in latency, 34% in energy)."""
    from . import compiled
    cm = compiled.compile_model(graph, chip, mapping)
    r = compiled.step_result(cm, spikes, active)
    W, H = mapping.mesh
    links = 2 * ((W - 1) * H + W * (H - 1))
    cx, cy = math.ceil(W / chip.tiles_x), math.ceil(H / chip.tiles_y)
    chip_links = max(1, 2 * ((cx - 1) * H + W * (cy - 1)))
    hops, ch = r["counts"]["hops"], r["counts"]["chip_hops"]
    t_core = float(r["core_time"].max())
    t_link = max((hops - ch) * chip["t_link"] / max(links, 1), ch * chip["t_chip_hop"] / chip_links)
    t_cop = r["time"].get("coproc", 0.0) if coproc else 0.0
    roofs = dict(core=t_core, link=t_link, coproc=t_cop)
    t = max(roofs.values()) + r["time"]["barrier"]
    e_dyn = sum(v for k, v in r["energy"].items() if k != "static")
    e = (e_dyn + r["energy"]["static"] * t / r["time"]["step"]) * chip_steps
    tm = r["time"]
    full_bound = max(("compute", "network", "coproc"), key=lambda k: tm.get(k, 0.0))
    return dict(step_s=t, latency_s=t * chip_steps, energy_J=e, bound=max(roofs, key=roofs.get),
                full_step_s=tm["step"], full_latency_s=tm["step"] * chip_steps,
                full_energy_J=sum(r["energy"].values()) * chip_steps, full_bound=full_bound)
