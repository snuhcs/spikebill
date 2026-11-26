"""Quickstart: a small convolutional SNN on Loihi 2."""
import os
import sys

from spikebill import NetBuilder, Simulator

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out, exist_ok=True)
quick = bool(os.environ.get("QUICK"))

b = NetBuilder("quickstart_cnn", steps=4)                 # 4 SNN time steps per inference
x = b.input("img", (3, 32, 32), rate=1.0)                  # direct (graded) input, sent every step
c1 = b.conv2d(x, "conv1", out_ch=32, k=3, rate=0.15)       # rates: spikes per neuron per step (measured or assumed)
c2 = b.conv2d(b.pool(c1, "pool1", rate=0.2), "conv2", out_ch=64, k=3, rate=0.1)
h = b.dense(b.pool(c2, "pool2", rate=0.15), "fc1", 256, rate=0.05)
b.dense(h, "out", 10, rate=0.1)

sim = Simulator(b, chip="loihi2", multicast="tree")
rep = sim.run(balance=not quick, ranges=True)              # value parameters + every parameter at its low / high end
print(rep.summary())
print(rep.table(top=6))
print(rep.paper_block("markdown"))                          # paste into a paper: numbers + every setting behind them
rep.save(os.path.join(out, "quickstart.json"))
rep.view(os.path.join(out, "quickstart.html"))
if not quick:
    from spikebill import plots
    plots.save_all(rep, out)
