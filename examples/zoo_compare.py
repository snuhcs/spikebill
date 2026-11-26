"""A published spiking transformer: chip-level estimate vs the shipped reference and vs the paper's 45 nm claim."""
import os
import sys

from spikebill import Simulator, zoo

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out, exist_ok=True)
quick = bool(os.environ.get("QUICK"))
name = "sdt_v1_8_512"

print("\n".join(zoo.names()))
meta, ref = zoo.info(name), zoo.reference(name)
sim = Simulator(f"zoo:{name}", **{k: v for k, v in zoo.REFERENCE_SETTINGS.items() if k in ("chip", "multicast", "static")})
greedy = sim.run(balance=False)
print(greedy.summary())
if not quick:
    bal = sim.run(balance=True, objective="edp", workers=4)  # the reference settings
    print(bal.summary())
    print(f"reference {ref['value']['energy_J'] * 1e3:.0f} mJ, this run {bal.results['value']['energy_J'] * 1e3:.0f} mJ")
    print(greedy.compare(bal, labels=("greedy", "balanced")))
    bal.save(os.path.join(out, f"{name}.json"))
print(f"paper's own estimate {meta['claim_mJ']} mJ ({meta['claim_convention']}); "
      f"chip-level estimate / claim = {greedy.results['value']['energy_J'] * 1e3 / meta['claim_mJ']:.0f}x (greedy)")
