"""Build large SNN graphs in code, layer by layer, with exact synapse counts and weight-sharing layouts.

    from spikebill.build import NetBuilder
    b = NetBuilder("my_net", steps=4)
    x = b.input("img", (3, 32, 32))
    c1 = b.conv2d(x, "c1", out_ch=64, k=3, stride=1, rate=0.1)        # LIF population after a conv
    p1 = b.pool(c1, "p1", k=2, stride=2, rate=0.2)                     # OR neurons (max-pooling of spikes)
    t = b.tokens(p1)                                                   # (C, H, W) -> N tokens x C channels
    q = b.linear(t, "q", d_out=128, rate=0.05)                         # token-wise linear (1x1 conv)
    g = b.graph()                                                      # spikebill.graph.Graph

Rates are spikes per neuron per SNN time step (e.g. a framework's firing rate); `steps` is the number of SNN time
steps per inference. Weight sharing is declared so the mapper can place shared weights per core
(`mapping.shared_layout`); fully private projections use `dense` / `sparse` / `one_to_one`.
"""
from dataclasses import dataclass

import numpy as np

from . import graph as graph_mod


@dataclass
class Tensor:
    """Handle to a population and its logical shape: (C, H, W) for maps, (N, C) for tokens, (n,) otherwise."""
    name: str
    shape: tuple

    @property
    def size(self):
        return int(np.prod(self.shape))


class NetBuilder:
    """Builds a Graph layer by layer (see the module docstring); `steps` = SNN time steps per inference."""
    def __init__(self, name="net", steps=1, weight_bits=8):
        self.name, self.steps, self.bits = name, steps, weight_bits
        self.pops, self.projs, self.meta = [], [], {}

    # -- populations ---------------------------------------------------------------------------------------------
    def pop(self, name, shape, rate=0.0, active=None):
        """A neuron population of `shape` (size = product) with firing rate `rate` (spikes/neuron/step)."""
        shape = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
        d = dict(name=name, size=int(np.prod(shape)), rate=float(rate))
        if active is not None:
            d["active"] = float(active)
        self.pops.append(d)
        return Tensor(name, shape)

    def input(self, name, shape, rate=1.0):
        """Input spikes (or graded inputs, rate 1) entering the chip."""
        return self.pop(name, shape, rate)

    # -- projections ---------------------------------------------------------------------------------------------
    def _proj(self, src, dst, synapses, kind, bits=None, group=None, axon_fanin=None):
        d = dict(src=src.name, dst=dst.name, kind=kind, fanout=float(synapses) / src.size,
                 weight_bits=int(bits or self.bits))
        if group:
            d.update(shared_weights=True, share_group=int(group))
            if axon_fanin:
                d["axon_fanin"] = int(axon_fanin)
        self.projs.append(d)

    def connect(self, src, dst, kind="dense", fanout=None, bits=None):
        """Private synapses: dense (all-to-all), sparse (random, `fanout` per source neuron) or one_to_one."""
        syn = {"dense": src.size * dst.size, "one_to_one": src.size}.get(kind)
        if syn is None:
            syn = src.size * float(fanout)
        self._proj(src, dst, syn, kind, bits)

    # -- layers (create the destination population and its input projection) -------------------------------------
    def dense(self, src, name, n_out, rate=0.0, bits=None):
        """A population of n_out neurons, fully connected from src (private weights)."""
        dst = self.pop(name, (n_out,), rate)
        self.connect(src, dst, "dense", bits=bits)
        return dst

    def conv2d(self, src, name, out_ch, k=3, stride=1, padding=None, rate=0.0, bits=None):
        """A conv layer + its neurons (out_ch, H', W') from a (C, H, W) source; weights shared over positions."""
        C, H, W = src.shape
        padding = k // 2 if padding is None else padding
        Ho, Wo = (H + 2 * padding - k) // stride + 1, (W + 2 * padding - k) // stride + 1
        dst = self.pop(name, (out_ch, Ho, Wo), rate)
        # synapses = output positions x out channels x (in channels x k x k) (borders ignored)
        self._proj(src, dst, out_ch * Ho * Wo * C * k * k, "conv", bits, group=Ho * Wo, axon_fanin=C)
        return dst

    def pool(self, src, name, k=2, stride=2, rate=0.0):
        """Spike max-pooling as OR neurons (fan-in k*k, 1-bit synapses)."""
        C, H, W = src.shape
        Ho, Wo = (H - k) // stride + 1, (W - k) // stride + 1
        dst = self.pop(name, (C, Ho, Wo), rate)
        self._proj(src, dst, C * Ho * Wo * k * k, "conv", bits=1)
        return dst

    def tokens(self, src):
        """View a (C, H, W) map as N = H*W tokens x C channels (no new neurons)."""
        if len(src.shape) == 3:
            C, H, W = src.shape
            return Tensor(src.name, (H * W, C))
        return src

    def linear(self, src, name, d_out, rate=0.0, bits=None):
        """Token-wise linear layer (1x1 conv): weights shared over the N tokens."""
        N, D = src.shape
        dst = self.pop(name, (N, d_out), rate)
        self._proj(src, dst, N * D * d_out, "conv", bits, group=N, axon_fanin=D)
        return dst

    def lif(self, name, like, rate=0.0):
        """A population shaped like another tensor (connect it yourself)."""
        return self.pop(name, like.shape, rate)

    # -- output ----------------------------------------------------------------------------------------------------
    def graph(self):
        """The built spikebill Graph."""
        return graph_mod.from_dict(dict(name=self.name, steps_per_inference=self.steps,
                                        populations=self.pops, projections=self.projs, meta=self.meta))
