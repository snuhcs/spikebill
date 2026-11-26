"""Trace an snnTorch model (activity measured on real inputs), estimate it, and write a per-step power trace."""
import os
import sys

import torch
import torch.nn as nn
import snntorch as snn

from spikebill import Simulator

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out, exist_ok=True)


class CSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.p1 = nn.Conv2d(1, 8, 5), nn.MaxPool2d(2)
        self.l1 = snn.Leaky(beta=0.9, init_hidden=True, threshold=0.3)
        self.c2, self.l2 = nn.Conv2d(8, 16, 5), snn.Leaky(beta=0.9, init_hidden=True, threshold=0.3)
        self.fc, self.l3 = nn.Linear(16 * 8 * 8, 10), snn.Leaky(beta=0.9, init_hidden=True, threshold=0.3)

    def forward(self, x):
        x = self.l1(self.p1(self.c1(x)))
        return self.l3(self.fc(self.l2(self.c2(x)).flatten(1)))


torch.manual_seed(0)
x = (torch.rand(8, 4, 1, 28, 28) < 0.3).float()            # (T, batch, ...) input spikes; use your data here
sim = Simulator.from_torch(CSNN(), x, chip="loihi2")        # traces the model once per step, measures every layer
print(sim.graph.summary())
rep = sim.run(balance=False)
print(rep.summary())
trace = sim.power_trace(sim.measured_trace("pipelined"))   # per chip step, from the measured per-step rates
print(trace.summary())
trace.view(os.path.join(out, "power_trace.html"))
trace.to_csv(os.path.join(out, "power_trace.csv"))
