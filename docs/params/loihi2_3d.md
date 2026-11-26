**loihi2_3d v2026.11.3-e6 (sha256 7f69951fa76a); 8x4x4 tiles x 4 cores**

| key | value | low | high | unit | meaning | source |
|---|---|---|---|---|---|---|
| `max_neurons` | 8192 | 8192 | 8192 |  | neurons per core | [P] brief Fig. 1 (up to 8192; depends on state size) |
| `syn_mem_bits` | 1.049e+06 | 1.049e+06 | 1.049e+06 |  | synapse memory per core (bits) | [P] brief Fig. 1: up to 128 KB synaptic memory per core |
| `max_fanin_axons` | 4096 | 4096 | 4096 |  | input axons per core | [A] Loihi 1 value (Naxin); Loihi 2 not public |
| `max_fanout_axons` | 4096 | 4096 | 4096 |  | output axons per core | [A] Loihi 1 value (Naxout); Loihi 2 not public |
| `e_synop` | 2.31e-11 | 1.17e-11 | 3.55e-11 | J | energy per synaptic event | [E6] 0.65 x [L1] (range 0.33-1.0); [L1] SANA-FE dense synapse |
| `e_neuron_update` | 3.33e-11 | 1.69e-11 | 5.12e-11 | J | energy per neuron per step (all mapped neurons) | [E6] 0.65 x [L1] (range 0.33-1.0); [L1] SANA-FE soma access |
| `e_neuron_active` | 1.4e-11 | 7.13e-12 | 2.16e-11 | J | extra energy per neuron with input or state | [E6] 0.65 x [L1] (range 0.33-1.0); [L1] SANA-FE soma update |
| `e_spike_gen` | 4.5e-11 | 2.29e-11 | 6.93e-11 | J | energy per spike generated | [E6] 0.65 x [L1] (range 0.33-1.0); [L1] SANA-FE soma spike out |
| `e_spike_out` | 7.22e-11 | 3.66e-11 | 1.11e-10 | J | energy per message sent | [E6] 0.65 x [L1] (range 0.33-1.0); [L1] SANA-FE axon out per message |
| `e_hop` | 2.34e-12 | 1.19e-12 | 3.6e-12 | J | energy per tile-to-tile hop | [E6] 0.65 x [L1] (range 0.33-1.0); [L1] SANA-FE hop mean |
| `e_chip_hop` | 3.6e-11 | 3.6e-12 | 3.6e-10 | J | energy per chip-to-chip hop | [A] not public; 10x on-chip hop, range 1x-100x |
| `t_synop` | 7.6e-10 | 7.6e-10 | 3.8e-09 | s | time per synaptic event | [P] brief: 5x Loihi 1 synaptic ops; [L1] 3.8 ns / 5 |
| `t_neuron_update` | 3e-09 | 3e-09 | 6e-09 | s | time per neuron per step | [P] brief: 2x neuron update; [L1] 6.0 ns / 2 |
| `t_neuron_active` | 1.85e-09 | 1.85e-09 | 3.7e-09 | s | extra time per active neuron | [P] 2x; [L1] 3.7 ns / 2 |
| `t_spike_gen` | 3e-09 | 3e-09 | 3e-08 | s | time per spike generated | [P] brief: 10x spike generation; [L1] 30 ns / 10 |
| `t_spike_out` | 5.1e-09 | 5.1e-09 | 5.1e-09 | s | time per message sent (core side) | [L1] axon out (no Loihi 2 figure) |
| `t_msg_in` | 1.6e-08 | 3.2e-09 | 1.6e-08 | s | time per message received (core side) | [L1] axon in; low assumes 5x like synops [A] |
| `t_hop` | 5.3e-09 | 4.1e-09 | 6.5e-09 | s | latency per tile hop | [L1] hop latency |
| `t_link` | 2.9e-10 | 2.9e-10 | 5.3e-09 | s | time per message on a tile link (1 / bandwidth) | [L1] Loihi 1 tile bandwidth 3.44 Gspike/s; high = hop latency |
| `t_chip_hop` | 1.25e-09 | 1.25e-09 | 5e-09 | s | time per message on a chip-to-chip link | [P] brief fn 3: 800 Mtransfers/s (simulated); high = 4 transfers/packet [A] |
| `t_barrier` | 1.8e-06 | 2e-07 | 1.8e-06 | s | barrier (one chip; barrier_table overrides) | [P] brief: min timestep < 200 ns (low); [L1] 1.8 us (value) |
| `t_barrier_multichip` | 1.43e-05 | 1.8e-06 | 1.62e-05 | s | barrier across chips | [L1] Frady 2020 (Loihi 1 multichip) as proxy |
| `p_static_core` | 0 | 0 | 0 | W | static power per used core | static booked per chip |
| `p_static_chip` | 0.92 | 0.6 | 2.2 | W | static power per chip | 4 dies x loihi2 p_static_chip (v2026.11.2 range 0.15-0.55 W per die) |
| `t_vhop` | 1e-09 | 5e-10 | 5e-09 | s | latency per vertical hop | [A] vertical router hop latency |
| `t_vlink` | 2.9e-10 | 1e-10 | 1e-09 | s | time per message on a vertical link | [A] vertical link time per message (planar t_link as value) |
| `e_vhop` | 1e-12 | 3e-13 | 3e-11 | J | energy per vertical (die-to-die) hop | [A] vertical hop energy per spike packet (~32 bit); to verify against TSV / hybrid-bonding literature |
Options: core_timing max, multicast none, static_accounting chip.
