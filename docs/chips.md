# Chips: parameters, sources, adding a chip

A chip is one YAML file (`src/spikebill/chips/<name>.yaml`, or any path). `spikebill --list-chips` lists the shipped
ones; `spikebill --chip loihi2 --describe` prints every parameter with its range, meaning and source. How each
parameter enters the estimate: `docs/cost_model.md`.

## Shipped chips

| name | what | status of the numbers |
|---|---|---|
| `loihi1` | Intel Loihi (14 nm), 8x4 tiles x 4 cores | per-event energies and times calibrated against Nahuku boards by SANA-FE (Boyle et al., TCAD 2025); ranges span Davies et al. 2018 (pre-silicon). Our model reproduces SANA-FE within 0.2% energy / 6% latency on held-out configurations (E1b). |
| `loihi2` | Intel Loihi 2 (Intel 4), same mesh assumed | capacities public (tech brief); per-event energies NOT public: Loihi 1 values scaled by one factor fitted to one Loihi 2 workload (0.65, range 0.33-1.0; E6). Times from the brief's speed-ups. A held-out I/O-bound workload is under-predicted 11-22x (host I/O and embedded cores are not modelled). Treat as a proxy; always report the range. |
| `loihi2_3d` | hypothetical: 4 stacked Loihi 2 dies | vertical-link parameters are assumptions with wide ranges; for design-space exploration only (E7). |
| `toy`, `toy3d` | round numbers | unit tests only. |

## Source tags

Each parameter's `source` starts with a tag: `[P]` published by the vendor or a paper (pre-silicon or brief),
`[C]` calibrated against silicon by a third party, `[E6]` fitted here to one silicon workload, `[L1]` borrowed from
Loihi 1 as a proxy, `[D]` derived here from published totals, `[A]` assumption. The YAML header explains the tags
for that chip and names the documents.

## Parameters

All in SI units (J, s, W, bits). `low` / `high` are optional; a missing end equals `value`.

| key | meaning | enters |
|---|---|---|
| `max_neurons` | neurons per core | mapping |
| `syn_mem_bits` | synapse memory per core | mapping |
| `max_fanin_axons` | input axons per core | mapping |
| `max_fanout_axons` | output axons per core | mapping (checked) |
| `e_synop` | energy per synaptic event | $E_{syn}$ |
| `e_neuron_update` | energy per mapped neuron per step | $E_{neu}$ |
| `e_neuron_active` | extra energy per neuron with input or state | $E_{neu}$ |
| `e_spike_gen` | energy per spike generated | $E_{neu}$ |
| `e_spike_out` | energy per message sent | $E_{net}$ |
| `e_hop`, `e_chip_hop`, `e_vhop` | energy per tile hop / chip-to-chip hop / vertical hop | $E_{net}$ |
| `t_synop` | time per synaptic event | core message side |
| `t_msg_in` | time per message received | core message side |
| `t_neuron_update`, `t_neuron_active`, `t_spike_gen`, `t_spike_out` | per-neuron / per-spike / per-message times | core neuron side |
| `t_hop`, `t_vhop` | latency per tile / vertical hop | path term |
| `t_link`, `t_chip_hop`, `t_vlink` | time per message on a tile / chip / vertical link (1 / bandwidth) | busiest-link term |
| `t_barrier`, `t_barrier_multichip` | barrier per step (one chip / several) | step time |
| `p_static_core`, `p_static_chip` | static power per used core / per chip | $E_{static}$ |
| `e_coproc_op`, `t_coproc_op` (optional) | energy / time per co-processor operation; chips without them get the assumed defaults of `spikebill.chip.COPROC` (0.03 pJ, 0.01-0.9 pJ; 0.125 ns/op = 8 units at 1 ns, range 0.125-1 ns) when a graph uses a co-processor; option `coproc_units: pooled` (one shared unit) or `per_unit` (one per block, set `t_coproc_op` per unit) | $E_{cop}$, $t_{cop}$ |

Per-synapse-type overrides: `e_synop@conv`, `t_synop@sparse`, ... apply to projections whose `synapse` field
names that type. Options outside `params`: `barrier_table` (barrier by number of used tiles), `core_timing`
(`max` | `sum`), `multicast` (`none` | `tree` | `chip_broadcast`), `static_accounting` (`chip` | `cores` | `none`); the last three can
also be set per run (`Simulator(..., multicast="tree", static="none")`, `chip.with_core_timing("sum")`).

## Adding a chip

1. Copy `docs/chip_template.yaml`, set the topology, replace every value, and give each parameter a unit, a source
   tag and, where it is not measured, a `low` / `high` range. Write the documents behind the tags in the header.
2. `spikebill --chip path/to/my_chip.yaml --describe`. Loading checks the file: missing or misspelt parameters,
   `low <= value <= high`, non-negative values, SI units, and the option values; an error names the key.
3. Sanity-check against one number you trust (a published energy per synaptic op or per inference of a small
   network), then run the model of interest with `ranges=True`.
4. If the chip differs structurally (no shared routers: `cores_per_tile: 1`; tree multicast in hardware:
   `multicast: tree`; serial core pipeline: `core_timing: sum`), set the option rather than folding it into a
   parameter.

What the YAML cannot express (host I/O, embedded processors, learning engines, analog variability) is outside the
model; see `docs/cost_model.md` section 9.
