**loihi1 v2026.11.2 (sha256 1b4315144404); 8x4x1 tiles x 4 cores**

| key | value | low | high | unit | meaning | source |
|---|---|---|---|---|---|---|
| `max_neurons` | 1024 | 1024 | 1024 |  | neurons per core | [P] Davies18 constraint 1 (Ncx) |
| `syn_mem_bits` | 1.049e+06 | 1.049e+06 | 1.049e+06 |  | synapse memory per core (bits) | [P] Davies18 constraint 2 (128 KB; before compression) |
| `max_fanin_axons` | 4096 | 4096 | 4096 |  | input axons per core | [P] Davies18 constraint 4 (Naxin) |
| `max_fanout_axons` | 4096 | 4096 | 4096 |  | output axons per core | [P] Davies18 constraint 3 (Naxout) |
| `e_synop` | 3.55e-11 | 2.36e-11 | 3.55e-11 | J | energy per synaptic event | [C] SANA-FE dense synapse; [P] Davies18 T2 23.6 pJ (min) |
| `e_synop@conv` | 2.4e-11 | 2.4e-11 | 2.4e-11 | J | energy per synaptic event | [C] SANA-FE conv synapse |
| `e_synop@sparse` | 3.36e-11 | 3.36e-11 | 3.36e-11 | J | energy per synaptic event | [C] SANA-FE sparse synapse |
| `t_synop@conv` | 3.1e-09 | 3.1e-09 | 3.1e-09 | s | time per synaptic event | [C] SANA-FE conv synapse |
| `t_synop@sparse` | 4.7e-09 | 4.7e-09 | 4.7e-09 | s | time per synaptic event | [C] SANA-FE sparse synapse |
| `e_neuron_update` | 5.12e-11 | 5.12e-11 | 5.2e-11 | J | energy per neuron per step (all mapped neurons) | [C] SANA-FE soma access, every neuron; [P] Davies18 T2 52 inactive |
| `e_neuron_active` | 2.16e-11 | 2.16e-11 | 2.9e-11 | J | extra energy per neuron with input or state | [C] SANA-FE soma update; [P] Davies18 81 active - 52 inactive |
| `e_spike_gen` | 6.93e-11 | 0 | 6.93e-11 | J | energy per spike generated | [C] SANA-FE soma spike out; not separate in Davies18 |
| `e_spike_out` | 1.11e-10 | 1.7e-12 | 1.11e-10 | J | energy per message sent | [C] SANA-FE axon out per message; [P] Davies18 T2 within-tile spike 1.7 pJ |
| `e_hop` | 3.6e-12 | 3e-12 | 4.2e-12 | J | energy per tile-to-tile hop | [C] SANA-FE 3.0 E-W / 4.2 N-S, mean; [P] Davies18 3.0/4.0 |
| `e_chip_hop` | 3.6e-11 | 3.6e-12 | 3.6e-10 | J | energy per chip-to-chip hop | [A] not public; 10x on-chip hop, range 1x-100x |
| `t_synop` | 3.8e-09 | 3.5e-09 | 4.7e-09 | s | time per synaptic event | [C] SANA-FE dense 3.8 / sparse 4.7 ns; [P] Davies18 3.5 ns |
| `t_neuron_update` | 6e-09 | 5.3e-09 | 6e-09 | s | time per neuron per step | [C] SANA-FE soma access 6.0 ns; [P] Davies18 5.3 inactive |
| `t_neuron_active` | 3.7e-09 | 3.1e-09 | 3.7e-09 | s | extra time per active neuron | [C] SANA-FE soma update; [P] Davies18 8.4 - 5.3 |
| `t_spike_gen` | 3e-08 | 3e-08 | 3e-08 | s | time per spike generated | [C] SANA-FE soma spike out |
| `t_spike_out` | 5.1e-09 | 2.1e-09 | 5.1e-09 | s | time per message sent (core side) | [C] SANA-FE axon out per message; [P] Davies18 within-tile 2.1 ns |
| `t_msg_in` | 1.6e-08 | 1.6e-08 | 1.6e-08 | s | time per message received (core side) | [C] SANA-FE axon in per message |
| `t_hop` | 5.3e-09 | 4.1e-09 | 6.5e-09 | s | latency per tile hop | [P]/[C] 4.1 E-W / 6.5 N-S, mean |
| `t_link` | 2.9e-10 | 2.9e-10 | 5.3e-09 | s | time per message on a tile link (1 / bandwidth) | [P] Davies18 T2: 3.44 Gspike/s cross-sectional bandwidth per tile -> 1/3.44e9; high = hop latency (E1/E1b/E2 semantics) |
| `t_chip_hop` | 5.4e-09 | 5.4e-09 | 2.16e-08 | s | time per message on a chip-to-chip link | [D] 1/185 Mtransfers/s (Loihi 2 brief fn 3) per transfer; high = 4 transfers/packet [A] |
| `t_barrier` | 1.8e-06 | 4.65e-07 | 1.8e-06 | s | barrier (one chip; barrier_table overrides) | [C] SANA-FE latency_sync 29 cores 1.8 us; [P] Davies18 113-465 ns (1-32 tiles) |
| `t_barrier_multichip` | 1.43e-05 | 1.8e-06 | 1.62e-05 | s | barrier across chips | [P] Frady et al. 2020 Sec 5.1: 14.3 us (4x64 chips), 16.2 us (768 chips); low = single chip |
| `p_static_core` | 0 | 0 | 0 | W | static power per used core | static booked per chip; NxSDK pro-rates by used cores (alternative accounting) |
| `p_static_chip` | 0.0695 | 0 | 0.0695 | W | static power per chip | [D] Frady 2020 Table 1: 53.4 W / 768 chips; low = dynamic-only accounting (SANA-FE) |

Barrier table (used tiles -> s): 1: 6e-07, 2: 1e-06, 4: 1.4e-06, 29: 1.8e-06
Options: core_timing max, multicast none, static_accounting chip.
