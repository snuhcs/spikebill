# spikebill cost model

This is the model `spikebill` evaluates, written as equations. Every symbol is a chip-YAML key
(`src/spikebill/chips/*.yaml`, reference in `docs/chips.md`) or a quantity the mapping and traffic stages compute.
The code is `spikebill.cost.step_cost` (reference implementation) and `spikebill.compiled` (the same model compiled
into matrices; tests check that the two agree to floating-point precision).

## 1. Inputs

- **Graph.** Populations $p$ (size $N_p$, neuron model, state bits) and projections $q: p \to p'$ (kind `dense`,
  `sparse`/`random`, `conv`, `one_to_one`; fan-out; weight bits; whether weights are shared; optional exact edge
  list). Built with `NetBuilder`, read from NIR or JSON.
- **Chip.** A mesh of $W_c \times H_c$ tiles per chip (optionally $Z$ stacked dies), $k$ cores per tile sharing one
  router, per-core limits, per-event energies and times, barrier and static power (`docs/chips.md`).
- **Activity.** Per population and chip step: spikes $s_p$ and active neurons $a_p$ (neurons that received input or
  have non-resting state). Either mean rates (spikes per neuron per SNN step) or a recorded per-step trace. Rates can
  come from a training framework or from `spikebill.funcsim`, which runs the network in the chip's semantics
  (one-step delay per layer, fixed-point state, the chip's threshold / reset rules).

## 2. Mapping

Synapses are stored at the destination core; a core holds neurons of one population. The neurons per core $n_p$
are the largest $n$ that meets all limits:

$$ n \le \texttt{max\_neurons}, \qquad \mathrm{mem}_p(n) \le \texttt{syn\_mem\_bits}, \qquad
   \mathrm{axons}_p(n) \le \texttt{max\_fanin\_axons}. $$

- Private weights: $\mathrm{mem}_p(n) = n \sum_{q \to p} \mathrm{fanin}_q\, b_q$ ($b_q$ = weight bits).
- Shared weights (conv, token-wise linear): a core holds $t$ positions $\times$ $c$ channels and stores the
  channels' weights once, $\mathrm{mem} = c \sum_q \mathrm{fanin}_q b_q$; axons $= t \sum_q \mathrm{axon\_fanin}_q$.
  The layout $(t, c)$ maximising $n = tc$ under both limits is chosen (`mapping.shared_layout`).
- Cores are placed along a snake path over the tiles of a chip, then over chips (a near-square board), population
  by population, so consecutive layers are neighbours. An explicit placement can be given instead
  (`mapping.from_placement`).
- Placement search (`spikebill.place`, `Simulator.run(place=True)`, off by default): the population fill order is
  searched (spectral order, simulated annealing on a traffic x distance proxy with a busiest-cut time term, each
  candidate checked with the full model); with balancing, for every balancer candidate. It matters where the
  network is a large share of the step: on Meta-SpikeFormer 55M (506 chips, network-bound) it cuts EDP by 36-38%
  (E20), on the compute-bound zoo transformers by 0%.
  `place="split"` also lets a population occupy non-contiguous blocks (full-model hill climb): a further -16% EDP
  on SDT V2, -6% on Meta-SpikeFormer 55M, ~0% on compute-bound cases (E20(d)).
- Load balancing (`spikebill.balance`, on by default in `Simulator.run`) caps neurons per core for the populations on
  the critical path, trading more cores (static power, longer paths) against shorter per-core time; it searches
  caps per population for the best energy x delay (or energy, or latency). The per-core time targets run in 16
  geometric steps from the busiest population's time to the median population time (both from the greedy
  mapping, independent of placement). The objective is flat near its optimum, so reports also give the spread of
  the mappings within 5% of the best (`provenance["balance_band"]`, the "Balancing spread" line of
  `paper_block`); balanced numbers are best read with it.

A layer that does not fit even one neuron per core (memory) is reported in `Mapping.warnings`, not silently
dropped.

## 3. Traffic

A spike of a neuron on source core $i$ becomes one message to each destination core $j$ that holds at least one of
its targets (unicast copies, Loihi style); with `multicast: tree`, one packet routed along an XY(Z) tree when all
spikes of core $i$ reach all of its destination cores (dense fan-out); with `multicast: chip_broadcast`
(Loihi 2 style "broadcast at the destination chip"), destinations on the source's chip stay unicast, and for every
remote chip $k$ holding destination cores the source sends one packet, needed with probability
$1 - \prod_{j \in k} (1 - m_{ij})$, to the destination tile on $k$ nearest the source; routers on $k$ copy it to the
other destination cores there (their hops and link loads are counted; like the tree mode, the copies cost the sending
core no $e_{\text{spike\_out}}$ / $t_{\text{spike\_out}}$). Per projection and core pair the model needs
the expected messages $m_{ij}$ and synaptic events $y_{ij}$ per source spike:

| projection | $m_{ij}$ | $y_{ij}$ |
|---|---|---|
| dense | 1 for every destination core | neurons on $j$ |
| random (fan-out $f$, $k$ destination cores) | $1 - (1 - 1/k)^f$ | $f$ times the share of $j$ |
| local (conv, one-to-one) | 1 inside a window of $\lceil f / n_{p'} \rceil + 1$ cores aligned with $i$ | from the window |
| exact edges | from the edge list, weighted by per-neuron rates | from the edge list |

Shared-weight layouts change the window: a source spike reaches $\lceil C_{out}/c \rceil$ channel groups.
Routes are dimension ordered (X, then Y, then Z) on the global tile mesh; messages between cores of one tile cost
no hop. Link loads are accumulated with difference arrays (O(1) per flow), so a step costs O(flows), not
O(messages x hops).

Cost of the implementation (`spikebill.compiled`; release gap 31). Notation: $F$ flows (core pairs of local and
exact-edge projections), $D$ (source tile, destination tile) pairs of dense / random projections, $N_d$ entries of
the link difference arrays ($6 (W+1)(H+1)(Z+1)$ for a $W \times H \times Z$ tile mesh), $P$ populations, $C$ cores.
- `compile_model` (once per mapping): $O(F \log F + D)$; tree multicast adds one sort of the destination rows per
  projection ($O(F \log F)$, vectorised over source cores); the result is sparse matrices
  (cores x populations, link entries x populations) with at most $O(F + D)$ non-zeros.
- `evaluate` (per chip step, batched over steps): sparse mat-vecs, $O(\text{nnz})$, plus one cumulative sum over the
  link mesh, $O(N_d)$, for the busiest links. No term grows with messages x hops.
- Measured on one machine: compile 0.08-0.13 s for the zoo models (164-189 chips, unicast or tree); evaluate
  2.2-8.8 ms per step up to 1406 chips (E9 v5).

From the routes, per step: messages $M$, planar tile hops $h$, of which chip-boundary crossings $h_c$, vertical
hops $h_v$; per core $c$: messages sent $M^{out}_c$, received $M^{in}_c$, synaptic events $Y_c$; the busiest
intra-chip link load $L_{tile}$, chip-to-chip link load $L_{chip}$, vertical link load $L_{v}$; the longest
path latency $P = \max(h\, t_{hop} + h_v\, t_{vhop})$ over active flows.

## 4. Energy per chip step

$$ E_{syn} = \sum_q Y_q\, e_{synop}(\text{synapse type of } q) $$
$$ E_{neu} = N_{mapped}\, e_{neuron\_update} + a\, e_{neuron\_active} + s\, e_{spike\_gen} $$
$$ E_{net} = M\, e_{spike\_out} + (h - h_c)\, e_{hop} + h_c\, e_{chip\_hop} + h_v\, e_{vhop} $$
$$ E_{static} = P_{static}\, t_{step}, \qquad
   P_{static} = \begin{cases} p_{static\_core}\,C + p_{static\_chip}\,N_{chips} & \text{chip (default)}\\
   p_{static\_core}\,C + p_{static\_chip}\,C / C_{chip} & \text{cores}\\ 0 & \text{none} \end{cases} $$

$C$ = cores used, $C_{chip}$ = cores per chip. "chip" books every powered chip in full (system view); "cores"
pro-rates by used cores (the NxSDK / TrueNorth reporting practice); "none" is dynamic energy only (SANA-FE's
practice). Which convention a number uses changes it by up to the static share, so every report states it.

Co-processor (only for graphs whose populations carry `coproc_ops` / `coproc_msgs`, e.g. spiking attention traced
with `from_torch(..., attention="coproc")`): with $O = \sum_k s_k\, \text{coproc\_ops}_k$ operations and
$M_{cop} = \sum_k s_k\, \text{coproc\_msgs}_k$ messages to it in a step (operand spikes; the results come back
as the spikes of the product's output population, which the network model costs like any other messages),
$$ E_{cop} = O\, e_{coproc\_op} + M_{cop}\, e_{spike\_out}, \qquad t_{cop} = O\, t_{coproc\_op}. $$
Both are linear in the spikes of the populations they are attached to (operations per output spike as measured
when tracing), which is exact at the traced rates and an approximation when rates are changed.
With the chip option `coproc_units: per_unit` (default `pooled`), each co-processor unit (one per
`Population.coproc_unit` label, by default one per population, i.e. one per attention block in the zoo graphs) serves
only its own operations, and the step waits for the busiest unit: $t_{cop} = \max_u O_u\, t_{coproc\_op}$, with
$t_{coproc\_op}$ now the time of one unit. Energy is the same in both modes.

## 5. Time per chip step

Per core $c$ (activity spread over a population's cores by neuron count or by measured per-neuron rates):

$$ t^{neu}_c = n_c\, t_{neuron\_update} + a_c\, t_{neuron\_active} + s_c\, t_{spike\_gen} + M^{out}_c\, t_{spike\_out} $$
$$ t^{msg}_c = M^{in}_c\, t_{msg\_in} + \sum_q Y_{c,q}\, t_{synop}(q) $$
$$ t_c = \max(t^{neu}_c, t^{msg}_c) \ \ (\texttt{core\_timing: max}, \text{pipelined}) \quad\text{or}\quad
   t^{neu}_c + t^{msg}_c \ \ (\texttt{sum}) $$

$$ t_{compute} = \max_c t_c, \qquad
   t_{network} = \max(L_{tile}\, t_{link},\ L_{chip}\, t_{chip\_hop},\ L_v\, t_{vlink}) + P $$
$$ t_{step} = \max(t_{compute}, t_{network}, t_{cop}) + t_{barrier}(\text{tiles used}, N_{chips}) $$

$t_{barrier}$ comes from a table indexed by the number of tiles used (single chip) or `t_barrier_multichip`.
The overlap of compute and network and the max over cores make $t_{step}$ a bound on a step, not a schedule.
E1b (event-level, SANA-FE) and E5 (packet-level NoC simulation) measure how tight it is.

## 6. Per inference

An inference of $T$ SNN steps takes $T_{chip} = T + D$ chip steps with `execution="pipelined"` (a spike crosses one
layer per chip step; $D$ = network depth; the Loihi execution model) or $T_{chip} = T$ with `"lockstep"`. Spikes per
chip step are the per-SNN-step spikes scaled by $T / T_{chip}$.

$$ E_{inf} = T_{chip} \sum_{x} E_x, \qquad t_{inf} = T_{chip}\, t_{step}. $$

With a per-step trace, the sum runs over the recorded steps (`Simulator.run_trace`). `Simulator.power_trace`
returns per step the energy components, the time terms (which one sets the step), and the power
$P_t = E_t / t_{step,t}$; its mean power is $\sum_t E_t / \sum_t t_{step,t}$ (CSV, JSON, an HTML plot). An idle
step can have the highest power: it still pays the neuron updates and static power, over a short step.

## 7. Per-population attribution

`Report.by_population` splits $E_{inf}$ over populations: synaptic and network energy to the source population
(the spikes that cause them), neuron energy to the population itself, static energy by the population's share of
the occupied cores. The parts sum to the total. `core_time` is the slowest core holding the population; `critical`
marks the population(s) on the slowest core.

## 8. Parameter ranges

Every chip parameter has a value and, where it is not measured, a low / high end. `ranges=True` evaluates the
model with all parameters at their low end and at their high end (the same mapping); reports carry the range,
and published numbers should too.

## 9. What the model does not include

- Contention dynamics inside a step (queueing, back-pressure) beyond the busiest-link bound: E5 compares with a
  packet-level simulation (within 1% at multi-chip scale; at light load the model over-estimates by 16-31%
  because it adds the longest path latency in series with the busiest link). Infinite buffers in both. With
  finite link queues (E23, `packetsim.simulate_finite`) the model stays within 20% on >= 75% of the E5 steps down
  to 16-flit queues; at 4 flits it under-estimates bisection-heavy traffic by up to 59% (head-of-line blocking).
- Spike-count fluctuations: core loads use mean counts (max of expectations). Under independent Binomial spikes
  the expected step is within 1% on the reported cases (E22), but the compute term of cores holding a few neurons
  of a busy sender can be several times its mean-count value (on the zoo transformers it stays within 3%). `run(spike_variance="binomial")` replaces the compute term by its
  expectation (analytic, within 0.5% of the Monte Carlo, E22(b)); network fluctuation stays at mean counts.
- Host I/O, embedded processors (Loihi's x86 cores), input encoding and off-chip memory: E6 shows these can
  dominate a real Loihi 2 workload (11-22x under-prediction on an I/O-bound task).
- Learning rules, dendritic compartments, delays beyond one step per layer.
- Non-spiking operations (softmax, normalisation, smooth activations, exp). `from_torch` lists them with element
  counts in `info["unmapped_ops"]` and warns; they are not costed. Spike x spike products (matrix products and
  element-wise products) are costed as AND-gate neurons or on an assumed co-processor (`attention="andgate"` /
  `"coproc"`); with the default `"excluded"` they are listed and warned about, not costed.

## 10. Validation summary

| check | reference | result | E-ID |
|---|---|---|---|
| energy / latency, Loihi 1 | SANA-FE event-level simulator, held-out configurations | energy within 0.2%, latency within 6% | E1b |
| activity in chip semantics | SANA-FE spike trace | spike-exact | E3 |
| network term, `chip_broadcast` | packet-level NoC simulator (`packetsim.simulate_step`), 1-57 chips | 22 steps all within 20% (max 17%, sampling noise at light load); none / tree on the same steps within 17% | E5b |
| network term, real model | packet level, zoo:spikegpt_216m (13 chips, unicast / tree, 3 seeds) | -7.9% to +7.7% | E19 / validate check 4 |
| network term, finite buffers | `packetsim.simulate_finite` (credit back-pressure), E5 workload B | within 20% on >= 75% of steps for >= 16-flit queues; up to -59% at 4 flits | E23 |
| busiest core, stochastic spikes | Binomial per-core spikes vs mean counts, 5 models | expected step within 0.9% | E22 |
| Loihi 2 energies | one silicon workload (calibration), one held out | calibration s = 0.33-0.65; held-out I/O-bound task 11-22x under | E6 |

`python -m spikebill.validate` reruns six checks: E1b (SANA-FE), E5 workload B, and, without `--quick`, E5b
(chip_broadcast), the SpikeGPT packet check, E5 workload A (DVS packets) and E3(a) (funcsim vs SANA-FE), and prints
pass / fail against their criteria (the E1b check needs `pip install sanafe==2.2.9`). Numbers: the white paper.
