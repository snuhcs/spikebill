"""One model on three chips: Loihi 1, Loihi 2, a 3D-stacked Loihi 2 variant (assumed parameters, see docs/chips.md)."""
import os
import sys

from spikebill import NetBuilder, Simulator, plots

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out, exist_ok=True)
quick = bool(os.environ.get("QUICK"))

b = NetBuilder("mlp", steps=8)
x = b.input("in", (2048,), rate=0.1)
h = b.dense(x, "h1", 4096, rate=0.05)
h = b.dense(h, "h2", 4096, rate=0.05)
b.dense(h, "out", 100, rate=0.1)

reps = []
for chip in ("loihi1", "loihi2", "loihi2_3d"):
    r = Simulator(b, chip=chip, multicast="tree").run(balance=not quick, ranges=True)
    print(r.summary())
    reps.append(r)
print(reps[0].compare(reps[1], labels=("loihi1", "loihi2")))
plots.compare(reps, ["Loihi 1", "Loihi 2", "Loihi 2 3D"]).savefig(os.path.join(out, "chip_compare.png"), dpi=120)
