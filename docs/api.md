# spikebill API reference

The public interface, by task. Every function here has a docstring (`help(...)` gives the details); a test
(`code/tests/test_docs.py`) checks that each name below exists, is documented in code, and is listed here.
The model behind the numbers: `docs/cost_model.md`. Chip files: `docs/chips.md`. What to report: `docs/reporting.md`.

## 1. Get a model into spikebill

| call | what it gives |
|---|---|
| `spikebill.from_torch(model, inputs, multi_step=False, run=None, attention="excluded", ...)` | `(Graph, info)` from a spiking PyTorch module (snnTorch, SpikingJelly, own neuron classes) traced on input batches; firing rates measured. `attention`: `"excluded"` / `"andgate"` / `"coproc"` for spike x spike products (matrix and element-wise). `info`: rates, warnings, `unmapped_ops` (products with measured op counts; non-spiking ops such as softmax / LayerNorm with element counts, not costed). |
| `Simulator.from_torch(model, inputs, chip=..., trace_kw=None, **kw)` | the same, wrapped in a `Simulator` |
| `spikebill.NetBuilder(name, steps, weight_bits)` | build a graph in code: `input`, `pop`, `connect`, `dense`, `conv2d`, `pool`, `tokens`, `linear`, `lif`, `graph()` |
| `spikebill.nir_frontend.read(path)`, `nir_frontend.from_nir(g)` | a NIR graph -> `(Graph, info)` with exact connectivity |
| `spikebill.graph.load(path)`, `Graph.save(path)`, `Graph.to_dict()`, `graph.from_dict(d)` | graphs as JSON |
| `spikebill.zoo.names()`, `zoo.load(name, attention="coproc", variant=None)`, `zoo.info(name)` | shipped spiking transformers (SDT V1, Spikingformer, SDT V2) with the papers' rates; also `Simulator("zoo:<name>")` |
| `spikebill.zoo.variants(name)` | the shipped variants of a model: `andgate` (attention built from AND-gate neurons; also `attention="andgate"`) for the three transformers, `repconv_literal` (unfused RepConv branches) and `55m` (Meta-SpikeFormer 55M: widths x 4/3, the 31M rates as an assumption) for SDT V2; `"zoo:<name>.<variant>"` in model paths |
| `spikebill.activity.from_nir(model, inputs, semantics="loihi")` | `Activity` (mean rates, per-step spikes / active neurons) by simulating the model in the chip's semantics (`spikebill.funcsim`) |

`Graph` helpers: `summary()` (neurons, synapses, ...), `fanout(proj)`, `fanin(proj)`, `n_synapses(proj)`,
`syn_events_per_step(rates)`.

## 2. Estimate

`Simulator(model, chip="loihi2", multicast=None, static=None, weight_bits=8, overrides=None, coproc_units=None)`. `model`: Graph,
NetBuilder, NIR / JSON path or `"zoo:<name>"`; `chip`: shipped name, YAML path or `Chip`.

| method | returns |
|---|---|
| `run(rates=None, active=None, steps=None, execution="pipelined", balance=True, ranges=False, objective="edp", max_chips=None, max_energy_increase=None, place=False, spike_variance="mean")` | `Report` for one inference; the budgets restrict the balanced mapping (CLI `--max-chips`, `--max-energy-increase`); `place=True` also searches the fill order (CLI `--place`), `place="split"` also non-contiguous blocks per population (CLI `--place split`); `spike_variance="binomial"` sets the compute term to the expected busiest core under Binomial spikes (`spikebill.variance.expected_compute`, analytic; CLI `--spike-variance binomial`) |
| `sensitivity(...)` | one-at-a-time (tornado) rows per chip parameter, sorted by energy swing |
| `uncertainty(n=512, prior="loguniform", groups=(), ...)` | Monte Carlo over the parameter box: energy / latency p5 / p50 / p95 and samples |
| `fast(rates=None, ...)` | `dict(energy_J, latency_s, split, chips)` in milliseconds after the first call (cached greedy mapping; for training loops) |
| `run_trace(spikes_t, active_t, balance=False)` | `(per-step arrays, mapping)` for a recorded trace (lists of `{pop: count}` per chip step) |
| `measured_trace(execution="pipelined")` | per-chip-step `{pop: count}` measured by `from_torch` (`trace_info["rates_t"]`, skewed by pipeline level); `power_trace(sim.measured_trace())` |
| `power_trace(spikes_t or Activity, active_t=None, balance=False)` | `PowerTrace`; CLI `spikebill MODEL --trace spikes.npz --power-html p.html` (`spikebill.cli.load_trace`: .npz one array per population, or JSON) |

Rates are spikes per neuron per SNN step. `execution="pipelined"` runs T + depth chip steps per inference,
`"lockstep"` runs T.

### Report

Fields: `results` (`"value"` and, with `ranges=True`, `"low"` / `"high"`: `energy_J`, `latency_s`, `step_s`,
`split`, `time`, `chips`, `cores`, `counts_per_step`), `by_population`, `mapping`, `provenance` (chip version and
digest, options, activity source, spikebill version, `balance_band`: spread of the mappings within 5% of the best), `chip_steps`, `snn_steps`.

| method | output |
|---|---|
| `summary()` | three-line text summary |
| `table(top=10)` | the most expensive populations |
| `paper_block(fmt="latex" \| "markdown")` | a citable paragraph: numbers + every setting that changes them |
| `to_csv(path)`, `save(path)`, `view(path)` | per-population CSV, full JSON, HTML view of one mean step |
| `variance_check(steps=200, seed=0)` | Binomial per-core spikes vs the mean-count step: `R`, `R_compute` (`spikebill.variance.busiest_core`, Monte Carlo; E22); `variance.expected_compute(cm, spikes, active)` is the analytic compute term |
| `Report.load(path)`, `diff(other)`, `compare(other, labels)` | read a saved report back; compare two reports (ratios, components, step terms, largest per-population changes, differing provenance); CLI `spikebill --compare a.json b.json
spikebill MODEL --trace spikes.npz [--power-html p.html] [--power-csv p.csv] [--power-json p.json]` |

### PowerTrace (`spikebill.power`)

Per chip step: `energy` (components), `time` (compute / network / coproc / barrier), `step_s`, `power_W`, `counts`;
totals `energy_J`, `latency_s`, `mean_W` (total energy / total time), `peak_W`; `bound_by()` (which term sets each
step); `summary()`, `columns()`, `to_csv(path)`, `save(path)`, `view(path)` (HTML plot).

## 3. Chips

| call | what it does |
|---|---|
| `spikebill.chip.load(name_or_path)`, `chip.from_dict(d)` | load and validate a chip file |
| `chip.describe(chip)`, `chip.to_table(chip, fmt)` | parameter table with ranges and sources (text / markdown / LaTeX) |
| `Chip.at("low" \| "high")` | the chip with every parameter at one end of its range; `Simulator(g, chip=c.at("high"))` runs (and balances the mapping) at that end |
| `Chip.pinned()` | the chip fixed at its current end (value = low = high); the Simulator applies it to an `at(...)` chip |
| `Chip.with_coproc_units(mode)` | co-processor `"pooled"` (one shared unit) or `"per_unit"` (one per `Population.coproc_unit`, busiest unit sets $t_{cop}$) |
| `Chip.with_overrides(**kv)` | parameters replaced: a number is a point (low = value = high); `(low, value, high)` or a dict sets a range |
| `Chip.with_multicast(m)`, `with_static_accounting(s)`, `with_core_timing(t)`, `with_neuron_update(u)` | options |
| `Chip.sources()` | source of each parameter |

## 4. Check and compare

| call | what it does |
|---|---|
| `spikebill.validate.main()` / `spikebill-validate` / `python -m spikebill.validate` | rerun the E1b (SANA-FE), E5 and E5b (packet-level) checks, pass / fail |
| `validate.sanafe_check(...)`, `validate.packet_check(...)`, `validate.broadcast_check(...)`, `validate.model_check(...)`, `validate.dvs_packet_check(...)`, `validate.funcsim_check(...)` | the checks as functions (model_check: zoo:spikegpt_216m vs packet level; dvs_packet_check: E5 workload A; funcsim_check: E3(a)) |
| `spikebill.zoo.reference(name)`, `zoo.compute_reference(name)` | shipped reference estimates (`zoo.REFERENCE_SETTINGS`, graph-order placement: SDT V2's step is 13% above a searched placement, the others within 0.2%, E20(e)), and recomputing them |
| `spikebill.cost.first_order(graph, chip, rates)` | the "spreadsheet" estimate with the same parameters (no mapping, no network) |
| `Report.roofline()` / `spikebill.cost.roofline(graph, chip, mapping, spikes, active, chip_steps)` | per-core / per-link roofline on the same mapping (links loaded uniformly, no routing): equals the full model while a core binds; Meta-SpikeFormer 55M (`zoo:sdt_v2_31m.55m`): 1.65 J, 3.39 ms core-bound vs full 2.51 J, 10.7 ms network-bound |

## 5. Lower level

`spikebill.cost.step_cost(graph, chip, mapping, spikes, active)` is the reference implementation of one step;
`spikebill.compiled.compile_model(graph, chip, mapping)` + `compiled.evaluate(cm, S, A)` is the same model as
matrices (batched over steps); `spikebill.mapping.map_graph(graph, chip, caps=None, order=None)` (order: names or `(name, cores)` segments) and `spikebill.balance.balance(...)` place
the model (`balance.band(history)`: energy / latency / chips spread of the near-optimal mappings); `spikebill.place.search(graph, chip, spikes, active, caps, split=False, workers=1)` is a placement baseline (fill order from `spectral_order`, simulated `anneal`ing on a traffic `Proxy` and recursive Kernighan-Lin `bisection_order` (generic partitioner, E20(f)), each evaluated with the full model; `split=True` adds non-contiguous blocks: `anneal_split` and the full-model hill climb `split_refine`, whose rounds run in `workers` processes); `spikebill.viz.snapshot` / `viz.to_html` draw one step; `spikebill.plots` writes paper-ready figures of a Report (`energy_by_population`, `step_terms`, `placement`, `chip_load`, `compare(reports, labels)`, `save_all(report, dir)`; CLI `--plots DIR`); `spikebill.packetsim` is the packet-level NoC
simulator used by E5 (`packetsim.simulate_step(graph, mapping, chip, spikes, seed)`: one step of a mapped graph as
packets in its multicast mode, including chip_broadcast; `packetsim.simulate_finite(net, uni, t_hop, buf)`: unicast
packets with `buf`-flit link queues and credit back-pressure, E23).

## 6. Command line

```
spikebill MODEL [--chip loihi2] [--rate R | --rates rates.json] [--steps T] [--range] [--multicast tree]
       [--static chip|cores|none] [--execution pipelined|lockstep] [--no-balance] [--objective edp]
       [--by-pop N] [--paper-block latex] [--json out.json] [--csv pops.csv] [--viz out.html] [--plots DIR]
spikebill --compare a.json b.json
spikebill --list-chips | --list-zoo | --chip NAME --describe | --version
spikebill-validate [--quick] [--skip-sanafe] [--skip-packet] [--json out.json]
```

`MODEL` is a NIR or JSON path or `zoo:<name>`. Population names in `--rates` (and in `Simulator.run(rates=, active=)`)
must exist in the model (a typo is an error with close matches, not a silent fallback to the graph's rate); rates
must be >= 0 and `active` fractions in [0, 1]. User errors print one line and exit with status 2.
