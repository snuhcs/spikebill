"""spikebill: chip-level energy and latency of spiking neural networks on neuromorphic hardware."""

__version__ = "0.1.0.dev0"

from .api import Simulator  # noqa: E402,F401
from .build import NetBuilder  # noqa: E402,F401


def from_torch(model, inputs, **kw):
    """Trace a spiking PyTorch module on input batches -> (Graph, info); see spikebill.torch_frontend."""
    from .torch_frontend import from_torch as _f
    return _f(model, inputs, **kw)
