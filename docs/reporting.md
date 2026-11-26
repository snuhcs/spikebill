# Reporting hardware cost in SNN papers with spikebill

Most spiking-network papers report energy as synaptic operations x 0.9 pJ (and MACs x 4.6 pJ), the 45 nm
arithmetic energies of Horowitz (ISSCC 2014). That number leaves out reading the weight from memory, updating every
neuron every step, moving spikes between cores and chips, and the static power of every chip the model needs; on
Loihi-class hardware these make the deployed cost 17-178x larger for published spiking transformers
(white paper, experiment E10). This page proposes what to report instead, and how to get it in a few lines.

## Minimal recipe

```python
import spikebill
x = ...                                    # (T, B, ...) input spikes, a few batches of the test set
sim = spikebill.Simulator.from_torch(net, x, chip="loihi2")   # traces net, measures every layer's firing rate
rep = sim.run(ranges=True)                 # balanced mapping, pipelined execution, low/high chip parameters
print(rep.summary()); print(rep.table())   # where the energy goes
print(rep.paper_block())                   # paste into the paper (LaTeX); "markdown" also available
```

SpikingJelly multi-step models: `Simulator.from_torch(net, x, trace_kw=dict(multi_step=True, reset=functional.reset_net))`.
Custom loops (snnTorch with explicit membrane state): `trace_kw=dict(run=my_loop)`, where `my_loop(net, x)` runs one
inference batch. Your own neuron classes: `trace_kw=dict(neuron_types=(MyLIF,))`.

## What to report (and why each item matters)

| item | why | where it comes from |
|---|---|---|
| energy and latency per inference, with the low / high range | per-event energies of most chips are not public; the range is the honest error bar | `ranges=True` |
| chip parameter file, version and sha256 | parameters get recalibrated; the hash pins the exact numbers | `paper_block()` |
| number of chips and cores | static power and chip-to-chip traffic scale with it | report |
| mapping: balanced or greedy, multicast or unicast, static-power accounting | each changes the result by up to 2x (multicast: up to 100x at LLM scale) | report |
| execution model (pipelined T + depth chip steps, or lockstep) | sets latency and static energy | report |
| activity source: measured on the test set, in framework or chip semantics | framework semantics can overstate deep-network cost by up to 1.8x | `from_torch` (framework), `spikebill.funcsim` / NIR path (chip semantics) |
| the op-count number alongside | keeps comparability with earlier papers | `paper_block()` |

## What not to do

- Do not compare an SNN's op-count energy with an ANN's MAC-count energy and call the ratio the hardware saving; the
  terms that dominate on hardware (memory, static, network) are in neither number.
- Do not report a single number for a chip whose per-event energies are proxies; report the range.
- Do not drop attention products between two spike tensors silently: `from_torch` measures them while tracing
  (nonzero products and outputs per inference, in `info["unmapped_ops"]`). Either map them on the chip as AND-gate
  neurons (`trace_kw=dict(attention="andgate")`) or on a co-processor (`attention="coproc"`: its energy and time
  enter the report; the per-op parameters are assumed, with a wide range, unless your chip file sets them), and
  say which.

## Limits to state

spikebill models neuro-cores, the mesh network, chip-to-chip links, barrier synchronisation and static power. It does
not model host I/O, embedded processors or input encoding, which can dominate small, I/O-bound workloads on real
boards (`docs/cost_model.md`, section 9).

## Relation to NeuroBench

NeuroBench (Yik et al., *Nature Communications* 16:1545, 2025) has two tracks. Its algorithm track measures a model
without hardware: footprint, connection sparsity, activation sparsity and synaptic operations (dense, effective MACs,
effective ACs); the authors note that these metrics do not by themselves predict deployed latency or energy. Its
system track reports measurements (latency, throughput, power, energy per sample) taken on real neuromorphic systems.

spikebill sits between the two. It takes the same inputs as the algorithm track (the network and its activity, for
example the effective synaptic operations NeuroBench counts) and maps them onto a modelled chip system, adding what
the counts leave out: synaptic-memory reads per event, the number of chips the weights need and their static power,
spike traffic on the mesh and between chips, and the time per step. It does not replace the system track: its numbers
are model estimates with a parameter range, not measurements. Use both where you can: report the NeuroBench
algorithm-track metrics, the spikebill estimate (chip, version, mapping, activity source), and a system-track
measurement when hardware is available.

## Reproducing the white paper's numbers

`python examples/reproduce_paper.py` recomputes the numbers in the white paper that this package can compute (the
three spiking transformers at value / low / high parameters, greedy vs balanced mapping, the energy split, the
joint-uncertainty percentiles, the first-order baseline and the favourable floor, the AND-gate attention and unfused
RepConv variants and Meta-SpikeFormer 55M (`zoo.variants`), the placement search, the low calibration scale and
SpikeGPT-216M) and checks each against the printed value (`examples/expected/paper_numbers.json`, with the paper's
experiment ID). It takes about 3 min with 8 worker processes. With `--validate` it also reruns the validation
checks against SANA-FE and the packet-level simulator, with and without finite router buffers (needs `pip install
sanafe`, about 9 min), and checks their
errors against the paper's printed maxima.
