"""docs/api.md lists the public API; every listed name exists, has a docstring, and appears in the document."""
import importlib
import inspect
import os

import pytest

PUBLIC = [
    "spikebill.cost.roofline", "spikebill.api.Report.roofline",
    "spikebill.from_torch", "spikebill.Simulator", "spikebill.NetBuilder",
    "spikebill.api.Simulator.from_torch", "spikebill.api.Simulator.run", "spikebill.api.Simulator.sensitivity",
    "spikebill.api.Simulator.uncertainty", "spikebill.api.Simulator.fast", "spikebill.api.Simulator.run_trace",
    "spikebill.chip.Chip.with_coproc_units", "spikebill.plots.save_all", "spikebill.plots.compare",
    "spikebill.plots.energy_by_population", "spikebill.plots.step_terms", "spikebill.plots.placement", "spikebill.plots.chip_load",
    "spikebill.api.Simulator.power_trace", "spikebill.api.Simulator.measured_trace",
    "spikebill.api.Report.summary", "spikebill.api.Report.table", "spikebill.api.Report.paper_block",
    "spikebill.api.Report.to_csv", "spikebill.api.Report.save", "spikebill.api.Report.view",
    "spikebill.api.Report.load", "spikebill.api.Report.variance_check", "spikebill.variance.busiest_core", "spikebill.variance.expected_compute", "spikebill.cli.load_trace", "spikebill.api.Report.diff", "spikebill.api.Report.compare",
    "spikebill.power.PowerTrace.summary", "spikebill.power.PowerTrace.columns", "spikebill.power.PowerTrace.to_csv",
    "spikebill.power.PowerTrace.save", "spikebill.power.PowerTrace.view", "spikebill.power.PowerTrace.bound_by",
    "spikebill.build.NetBuilder.input", "spikebill.build.NetBuilder.pop", "spikebill.build.NetBuilder.connect",
    "spikebill.build.NetBuilder.dense", "spikebill.build.NetBuilder.conv2d", "spikebill.build.NetBuilder.pool",
    "spikebill.build.NetBuilder.tokens", "spikebill.build.NetBuilder.linear", "spikebill.build.NetBuilder.lif",
    "spikebill.build.NetBuilder.graph",
    "spikebill.nir_frontend.read", "spikebill.nir_frontend.from_nir",
    "spikebill.graph.load", "spikebill.graph.from_dict", "spikebill.graph.Graph.save", "spikebill.graph.Graph.to_dict",
    "spikebill.graph.Graph.summary", "spikebill.graph.Graph.fanout", "spikebill.graph.Graph.fanin",
    "spikebill.graph.Graph.n_synapses", "spikebill.graph.Graph.syn_events_per_step",
    "spikebill.zoo.names", "spikebill.zoo.load", "spikebill.zoo.variants", "spikebill.zoo.info", "spikebill.zoo.reference", "spikebill.zoo.compute_reference",
    "spikebill.activity.from_nir",
    "spikebill.chip.load", "spikebill.chip.from_dict", "spikebill.chip.describe", "spikebill.chip.to_table",
    "spikebill.chip.Chip.at", "spikebill.chip.Chip.with_overrides", "spikebill.chip.Chip.with_multicast",
    "spikebill.chip.Chip.with_static_accounting", "spikebill.chip.Chip.with_core_timing",
    "spikebill.chip.Chip.with_neuron_update", "spikebill.chip.Chip.sources",
    "spikebill.validate.main", "spikebill.validate.sanafe_check", "spikebill.validate.packet_check", "spikebill.validate.broadcast_check", "spikebill.validate.model_check", "spikebill.validate.dvs_packet_check", "spikebill.validate.funcsim_check", "spikebill.packetsim.simulate_step", "spikebill.packetsim.simulate_finite",
    "spikebill.cost.first_order", "spikebill.cost.step_cost", "spikebill.compiled.compile_model", "spikebill.compiled.evaluate",
    "spikebill.mapping.map_graph", "spikebill.balance.balance", "spikebill.place.search", "spikebill.place.anneal",
    "spikebill.place.spectral_order", "spikebill.place.Proxy", "spikebill.balance.band", "spikebill.viz.snapshot", "spikebill.viz.to_html",
]
DOC = os.path.join(os.path.dirname(__file__), "..", "docs", "api.md")


def _resolve(path):
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError as e:
            if e.name and not e.name.startswith(parts[0]):    # an optional dependency (nir, torch) is missing
                pytest.skip(f"{path}: needs {e.name}")
            continue
        except ImportError:
            continue
        for a in parts[i:]:
            obj = getattr(obj, a)
        return obj
    raise ImportError(path)


@pytest.mark.parametrize("name", PUBLIC)
def test_public_name_documented(name):
    obj = _resolve(name)
    assert inspect.getdoc(obj), f"{name} has no docstring"
    short = name.split(".")[-1]
    assert short in open(DOC).read(), f"{short} not in docs/api.md"
