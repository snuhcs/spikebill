# snnsim examples

Each script runs on a CPU in seconds to a few minutes and writes its outputs to `examples/out/` (or the
directory given as the first argument). `QUICK=1` makes them faster (no load balancing / fewer steps; used by the
tests).

| script | what it shows |
|---|---|
| `quickstart.py` | build a small SNN with `NetBuilder`, estimate energy / latency on Loihi 2 with ranges, print the summary, the per-population table and a citable `paper_block`; save JSON / HTML / figures |
| `from_torch_snntorch.py` | trace an snnTorch model on input spikes (measured rates), estimate it, and write a per-step power trace from the measured activity (`measured_trace`) |
| `zoo_compare.py` | a published spiking transformer from the zoo against the shipped reference and against the paper's own 45 nm op-count claim; greedy vs balanced mapping with `Report.compare` |
| `reproduce_paper.py` | recompute the white paper's numbers (112 values: spiking transformers at value / low / high with rows (c), (c'), (d), (e), greedy vs balanced mapping, energy split, Monte Carlo percentiles, first-order and roofline baselines, favourable floor, AND-gate attention and unfused RepConv variants, Meta-SpikeFormer 55M, placement search, calibration scale, SpikeGPT; about 3 min) and check each against `expected/paper_numbers.json`; `--validate` adds 11 validation numbers (SANA-FE, packet level, finite buffers; needs `sanafe`, ~9 min) |
| `chip_compare.py` | one model on Loihi 1, Loihi 2 and a 3D-stacked Loihi 2 variant; `plots.compare` |

Run from the repository root: `PYTHONPATH=code python examples/quickstart.py`.
