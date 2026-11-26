"""Activity for the cost model from a NIR model and input spikes, simulated in the chip's semantics.

    from spikebill import activity
    act = activity.from_nir("model.nir", inputs=spikes, semantics="loihi")   # spikes: (T, n_input) array
    rep = Simulator("model.nir", chip="loihi2").run(rates=act.rates, steps=act.steps)
    ev, m = Simulator("model.nir", chip="loihi2").run_trace(act.spikes_t, act.active_t)

`semantics="loihi"` delays spikes by one step per layer and quantises like Loihi (E3: spike-exact vs SANA-FE on its
DVS network); "framework" mimics typical training-framework simulation (same-step propagation), which E3 found to
overstate chip cost ~1.8x.
"""
from dataclasses import dataclass

import numpy as np

from . import funcsim, nir_frontend


@dataclass
class Activity:
    rates: dict          # pop -> spikes per neuron per step (mean over the run)
    spikes_t: list       # per step {pop: spikes}
    active_t: list       # per step {pop: neurons updated}
    neuron_rates: dict   # pop -> per-neuron mean rate (for rate-weighted flows / core shares)
    steps: int


def from_nir(model, inputs, semantics="loihi", input_pop=None, device=None, dt=None):
    """Simulate a NIR model on input spikes (T, n_input) in the chip's semantics ("loihi") or the training
    framework's ("framework") with spikebill.funcsim; returns an Activity (mean rates, per-step spikes / active
    neurons, per-neuron rates) for Simulator.run / run_trace / power_trace."""
    g = nir_frontend.read(model) if isinstance(model, str) else model
    G, info = nir_frontend.from_nir(g)
    fpops, fprojs = nir_frontend.to_funcsim(g, info, dt=dt)
    inputs = np.asarray(inputs)
    steps = inputs.shape[0]
    if input_pop is None:
        input_pop = [k for k, n in g.nodes.items() if type(n).__name__ == "Input"][0]
    fr = funcsim.simulate(fpops, fprojs, steps, semantics, device=device, inputs={input_pop: inputs})
    sizes = {p.name: p.size for p in fpops}
    rates = {k: float(v.sum() / (steps * sizes[k])) for k, v in fr.neuron_spikes.items()}
    return Activity(rates, fr.spikes, fr.active, {k: v / steps for k, v in fr.neuron_spikes.items()}, steps)
