import numpy as np
import pytest

nir = pytest.importorskip("nir")
torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from spikebill import nir_frontend


def _lif(n):
    return nir.LIF(tau=np.full(n, 10.0), r=np.ones(n), v_leak=np.zeros(n), v_threshold=np.ones(n), v_reset=np.zeros(n))


def test_conv_matrix_matches_torch():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 2, 3, 3)); w[0, 0, 1, 1] = 0
    x = rng.normal(size=(2, 7, 6))
    M, out = nir_frontend._conv_matrix(w, (2, 7, 6), stride=2, padding=1)
    ref = F.conv2d(torch.tensor(x)[None], torch.tensor(w), stride=2, padding=1)[0]
    assert out == tuple(ref.shape)
    assert np.allclose(M @ x.ravel(), ref.numpy().ravel())


def test_from_nir_counts():
    rng = np.random.default_rng(1)
    w1 = rng.normal(size=(3, 1, 3, 3))
    conv = nir.Conv2d(input_shape=(8, 8), weight=w1, stride=1, padding=0, dilation=1, groups=1, bias=np.zeros(3))
    lin = nir.Affine(weight=rng.normal(size=(5, 3 * 6 * 6)), bias=np.zeros(5))
    g = nir.NIRGraph(nodes={"in": nir.Input(input_type=np.array([1, 8, 8])), "c": conv, "l1": _lif(3 * 6 * 6),
                            "f": nir.Flatten(input_type={"input": np.array([3, 6, 6])}, start_dim=0, end_dim=-1),
                            "fc": lin, "l2": _lif(5), "out": nir.Output(output_type=np.array([5]))},
                     edges=[("in", "c"), ("c", "l1"), ("l1", "f"), ("f", "fc"), ("fc", "l2"), ("l2", "out")],
                     type_check=False)
    G, info = nir_frontend.from_nir(g)
    assert set(G.populations) == {"in", "l1", "l2"}
    assert G.n_synapses(G.projections[0]) == 3 * 6 * 6 * 9          # valid conv taps, no padding
    assert G.n_synapses(G.projections[1]) == 5 * 108
    assert G.projections[0].kind == "conv" and G.projections[1].kind == "dense"


def test_cli_end_to_end(tmp_path):
    from spikebill import cli
    rng = np.random.default_rng(2)
    g = nir.NIRGraph(nodes={"in": nir.Input(input_type=np.array([100])),
                            "fc": nir.Affine(weight=rng.normal(size=(50, 100)), bias=np.zeros(50)),
                            "l": _lif(50), "out": nir.Output(output_type=np.array([50]))},
                     edges=[("in", "fc"), ("fc", "l"), ("l", "out")], type_check=False)
    path = str(tmp_path / "m.nir")
    nir.write(path, g)
    rep = cli.main([path, "--chip", "loihi1", "--rate", "0.1", "--steps", "20", "--range", "--by-pop", "3",
                    "--csv", str(tmp_path / "p.csv"), "--json", str(tmp_path / "r.json")])
    out = rep.results
    assert set(out) == {"value", "low", "high"} and rep.snn_steps == 20
    assert out["value"]["energy_J"] > 0
    assert out["low"]["energy_J"] <= out["value"]["energy_J"] <= out["high"]["energy_J"]
    assert (tmp_path / "p.csv").exists() and (tmp_path / "r.json").exists()
    assert "loihi2" in cli.main(["--list-chips"])


def test_scalar_neuron_params_shape_inference():
    # neuron nodes with scalar parameters (as in older NIR files): sizes come from shape inference
    rng = np.random.default_rng(3)
    lif = lambda: nir.LIF(tau=np.array(10.0), r=np.array(1.0), v_leak=np.array(0.0), v_threshold=np.array(1.0),
                          v_reset=np.array(0.0))
    g = nir.NIRGraph(nodes={"in": nir.Input(input_type=np.array([1, 6, 6])),
                            "c": nir.Conv2d(input_shape=(6, 6), weight=rng.normal(size=(2, 1, 3, 3)), stride=1,
                                            padding=1, dilation=1, groups=1, bias=np.zeros(2)),
                            "l1": lif(), "p": nir.SumPool2d(kernel_size=np.array([2, 2]), stride=np.array([2, 2]),
                                                            padding=np.array([0, 0])),
                            "f": nir.Flatten(input_type={"input": np.array([2, 3, 3])}, start_dim=0, end_dim=-1),
                            "fc": nir.Affine(weight=rng.normal(size=(4, 18)), bias=np.zeros(4)), "l2": lif()},
                     edges=[("in", "c"), ("c", "l1"), ("l1", "p"), ("p", "f"), ("f", "fc"), ("fc", "l2")],
                     type_check=False)
    G, _ = nir_frontend.from_nir(g)
    assert G.populations["l1"].size == 72 and G.populations["l2"].size == 4
    # l1 -> l2 through SumPool + Affine: every l1 neuron reaches all 4 outputs (dense weights): 72 * 4 synapses
    assert G.n_synapses([p for p in G.projections if p.src == "l1"][0]) == 72 * 4


def test_viz_html(tmp_path):
    from spikebill import chip as chip_mod, cost, mapping, viz, graph as graph_mod
    g = graph_mod.from_dict(dict(name="v", populations=[dict(name="a", size=3000, rate=0.1), dict(name="b", size=2000)],
                                 projections=[dict(src="a", dst="b", kind="sparse", fanout=30)]))
    for name in ("toy", "toy3d"):
        c = chip_mod.load(name)
        sp, ac = cost.mean_activity(g)
        snap = viz.snapshot(g, c, mapping.map_graph(g, c), sp, ac)
        assert snap["tile_time"].max() > 0 and snap["link_x"].sum() > 0
        out = viz.to_html(snap, str(tmp_path / f"{name}.html"))
        assert "<svg" in open(out).read()


def test_affine_bias_reaches_funcsim():
    from spikebill import funcsim
    g = nir.NIRGraph(nodes={"in": nir.Input(input_type=np.array([3])),
                            "fc": nir.Affine(weight=np.zeros((2, 3)), bias=np.array([1.5, 0.0])),
                            "l": _lif(2), "out": nir.Output(output_type=np.array([2]))},
                     edges=[("in", "fc"), ("fc", "l"), ("l", "out")], type_check=False)
    G, info = nir_frontend.from_nir(g)
    pops, projs = nir_frontend.to_funcsim(g, info)
    r = funcsim.simulate(pops, projs, 5, "framework", device="cpu", inputs={"in": np.zeros((5, 3))})
    assert r.neuron_spikes["l"][0] > 0 and r.neuron_spikes["l"][1] == 0   # bias drives neuron 0 only


def test_lif_time_step_convention():
    # snnTorch-style export: tau = dt/(1-beta), r = tau/dt with dt = 1e-4, beta = 0.9 -> leak 0.9, gain 1
    lif = nir.LIF(tau=np.array([1e-3]), r=np.array([10.0]), v_leak=np.array([0.0]), v_threshold=np.array([1.0]),
                  v_reset=np.array([0.0]))
    g = nir.NIRGraph(nodes={"in": nir.Input(input_type=np.array([1])), "fc": nir.Linear(weight=np.ones((1, 1))),
                            "l": lif}, edges=[("in", "fc"), ("fc", "l")], type_check=False)
    G, info = nir_frontend.from_nir(g)
    pops, projs = nir_frontend.to_funcsim(g, info)
    p = [x for x in pops if x.name == "l"][0]
    assert p.leak_decay == pytest.approx(0.9) and projs[0].weight[0] == pytest.approx(1.0)
