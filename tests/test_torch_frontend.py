import numpy as np
import pytest

torch = pytest.importorskip("torch")
snn = pytest.importorskip("snntorch")
import torch.nn as nn  # noqa: E402

from spikebill import Simulator, from_torch  # noqa: E402


class CSNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv2d(1, 8, 5); s.p1 = nn.MaxPool2d(2); s.l1 = snn.Leaky(beta=0.9, init_hidden=True, threshold=0.3)
        s.c2 = nn.Conv2d(8, 16, 5); s.l2 = snn.Leaky(beta=0.9, init_hidden=True, threshold=0.3)
        s.fc = nn.Linear(16 * 8 * 8, 10); s.l3 = snn.Leaky(beta=0.9, init_hidden=True, threshold=0.3)

    def forward(s, x):
        x = s.l1(s.p1(s.c1(x)))
        y = s.l2(s.c2(x))
        return s.l3(s.fc(y.flatten(1)))


def test_snntorch_conv_trace_counts_and_rates():
    torch.manual_seed(0)
    net = CSNN()
    x = (torch.rand(6, 4, 1, 28, 28) < 0.3).float()
    g, info = from_torch(net, x)
    P = {(p.src, p.dst): p for p in g.projections}
    assert set(P) == {("input", "l1"), ("l1", "l2"), ("l2", "l3")}
    assert g.populations["l1"].size == 8 * 12 * 12 and g.populations["l2"].size == 16 * 8 * 8
    # conv 5x5 followed by 2x2 pooling of currents = 6x6 effective kernel at the pooled resolution
    assert g.n_synapses(P[("input", "l1")]) == pytest.approx(8 * 12 * 12 * 1 * 6 * 6)
    assert g.n_synapses(P[("l1", "l2")]) == pytest.approx(16 * 8 * 8 * 8 * 25)
    assert P[("l2", "l3")].kind == "dense" and g.n_synapses(P[("l2", "l3")]) == 1024 * 10
    # measured rate = spikes / (neurons x steps x batch), checked by hand for l1
    spk = []
    with torch.no_grad():
        net.l1.reset_hidden(); net.l2.reset_hidden(); net.l3.reset_hidden()
        for t in range(6):
            spk.append(net.l1(net.p1(net.c1(x[t]))))
    assert info["rates"]["l1"] == pytest.approx(float(torch.stack(spk).mean()), rel=1e-6)
    assert info["rates"]["input"] == pytest.approx(float(x.mean()))
    rep = Simulator(g, chip="loihi2").run()
    assert rep.results["value"]["energy_J"] > 0


def test_residual_add_becomes_one_to_one():
    class LIF(nn.Module):                     # a user-defined neuron type
        def forward(self, x):
            return (x > 0.1).float()

    class Res(nn.Module):
        def __init__(s):
            super().__init__()
            s.c0 = nn.Conv2d(2, 4, 3, padding=1); s.n0 = LIF()
            s.c1 = nn.Conv2d(4, 4, 3, padding=1); s.n1 = LIF()

        def forward(s, x):
            a = s.n0(s.c0(x))
            return s.n1(s.c1(a) + a)

    x = (torch.rand(3, 2, 2, 8, 8) < 0.5).float()
    g, info = from_torch(Res(), x, neuron_types=(LIF,))
    got = sorted((p.src, p.dst, p.kind) for p in g.projections)
    assert got == [("input", "n0", "conv"), ("n0", "n1", "conv"), ("n0", "n1", "one_to_one")]
    assert any(p.kind == "one_to_one" for p in g.projections) and any(
        p.kind == "conv" and p.src == "n0" for p in g.projections)


def test_simulator_from_torch():
    torch.manual_seed(1)
    x = (torch.rand(4, 2, 1, 28, 28) < 0.3).float()
    sim = Simulator.from_torch(CSNN(), x, chip="loihi1")
    assert sim.graph.steps_per_inference == 4 and "rates" in sim.trace_info
    assert sim.run(balance=False).results["value"]["energy_J"] > 0


class _Spk(nn.Module):
    def forward(self, x):
        return (x > 0.0).float()


class _Attn(nn.Module):
    """Spikingformer-style block on (B, N, D) tokens: out = LIF(proj(LIF(Q K^T V)))."""
    def __init__(s, D=8):
        super().__init__()
        s.q, s.k, s.v = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        s.qn, s.kn, s.vn, s.an, s.on = _Spk(), _Spk(), _Spk(), _Spk(), _Spk()
        s.proj = nn.Linear(D, D)

    def forward(s, x):
        q, k, v = s.qn(s.q(x)), s.kn(s.k(x)), s.vn(s.v(x))
        a = s.an((q @ k.transpose(-2, -1)) @ v * 0.25)
        return s.on(s.proj(a))


@pytest.mark.parametrize("mode", ["excluded", "andgate", "coproc"])
def test_attention_products_measured_and_mapped(mode):
    torch.manual_seed(0)
    T, B, N, D = 3, 2, 5, 8
    net = _Attn(D)
    x = (torch.rand(T, B, N, D) < 0.4).float()
    g, info = from_torch(net, x, neuron_types=(_Spk,), attention=mode)
    ops = {u["name"]: u for u in info["unmapped_ops"]}
    assert set(ops) == {"mm0", "mm1"} and ops["mm0"]["mkn"] == (N, D, N) and ops["mm1"]["mkn"] == (N, N, D)
    # hand count of nonzero products of Q K^T over all steps and batches
    ands = 0.0
    with torch.no_grad():
        for t in range(T):
            q, k = net.qn(net.q(x[t])), net.kn(net.k(x[t]))
            ands += float(((q != 0).float() @ (k != 0).float().transpose(-2, -1)).sum())
    assert ops["mm0"]["and_ops_per_inference"] == pytest.approx(ands / B)
    assert ops["mm0"]["a_srcs"] == ["qn"] and ops["mm0"]["b_srcs"] == ["kn"] and ops["mm1"]["b_srcs"] == ["vn"]
    P = {(p.src, p.dst): p for p in g.projections}
    if mode == "excluded":
        assert "mm0" not in g.populations and {("qn", "an"), ("kn", "an"), ("vn", "an")} <= set(P)
        assert any("not mapped" in w for w in info["warnings"])
    elif mode == "coproc":
        pops = g.populations
        # Q K^T is read only by (Q K^T) V: it stays inside the co-processor (no population); its operations are
        # counted per output spike of the final product
        assert "mm0" not in pops and "mm0:and" not in pops and pops["mm1"].size == N * D
        assert pops["mm1"].coproc_ops == pytest.approx(
            (ops["mm0"]["and_ops_per_inference"] + ops["mm1"]["and_ops_per_inference"])
            / ops["mm1"]["out_nonzero_per_inference"])
        assert pops["qn"].coproc_msgs == pops["kn"].coproc_msgs == pops["vn"].coproc_msgs == 1
        assert sorted(pops["mm1"].coproc_inputs) == ["kn", "qn", "vn"]
        from spikebill.api import depth                          # input -> qkv -> (co-processor) mm1 -> an -> on
        assert depth(g) == 4
        assert ("mm1", "an") in P and ("qn", "an") not in P and not any(p.dst.startswith("mm") for p in g.projections)
        rep = Simulator.from_torch(net, x, trace_kw=dict(neuron_types=(_Spk,), attention="coproc")).run(
            balance=False, ranges=True)
        v = rep.results["value"]
        assert v["split"]["coproc"] > 0 and "coproc" in v["time"]
        # co-processor energy = ops x e_coproc_op + operand messages x e_spike_out, per inference
        sim = Simulator(g, chip="loihi2")
        spikes, _, T, cs = sim._activity(None, None, None, "pipelined")
        opsv = sum(spikes[k] * p.coproc_ops for k, p in pops.items())
        msg = sum(spikes[k] * p.coproc_msgs for k, p in pops.items())
        e_cop = v["energy_J"] * v["split"]["coproc"]
        assert e_cop == pytest.approx((opsv * 0.03e-12 + msg * sim.chip["e_spike_out"]) * cs, rel=1e-9)
        assert rep.results["high"]["energy_J"] > v["energy_J"]
        assert "co-processor" in rep.paper_block()
        assert "cop" in rep.table() and sum(x["total"] for x in rep.by_population.values()) == pytest.approx(v["energy_J"])
    else:
        assert g.populations["mm0:and"].size == N * D * N and g.populations["mm0"].size == N * N
        assert g.populations["mm1:and"].size == N * N * D and g.populations["mm1"].size == N * D
        assert info["rates"]["mm0:and"] == pytest.approx(ops["mm0"]["and_density"])
        assert g.n_synapses(P[("qn", "mm0:and")]) == pytest.approx(N * D * N)      # each q element -> N ANDs
        assert P[("kn", "mm0:and")].kind == "sparse" and P[("mm0:and", "mm0")].fanout == 1.0
        assert ("mm0", "mm1:and") in P and ("vn", "mm1:and") in P and ("mm1", "an") in P
        assert ("qn", "an") not in P and not any("not mapped" in w for w in info["warnings"])
        assert Simulator(g, chip="loihi2").run(balance=False).results["value"]["energy_J"] > 0


def test_residual_stream_is_one_accumulator():
    """x = x + f(x) over several blocks: each weight layer appears once (no copy per later consumer); sums read by
    several nodes become graded accumulators; a weight layer with several targets writes one graded population."""
    D = 6

    class Blk(nn.Module):
        def __init__(s):
            super().__init__()
            s.n, s.fc = _Spk(), nn.Linear(D, D)

        def forward(s, x):
            return x + s.fc(s.n(x))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb, s.blocks = nn.Linear(D, D), nn.ModuleList(Blk() for _ in range(3))

        def forward(s, x):
            x = s.emb(x)
            for b in s.blocks:
                x = b(x)
            return x

    torch.manual_seed(0)
    x = (torch.rand(3, 2, 4, D) < 0.5).float()
    g, info = from_torch(Net(), x, neuron_types=(_Spk,))
    weighted = [(p.src, p.dst) for p in g.projections if p.kind != "one_to_one"]
    assert len(weighted) == 4                                  # emb + 3 fc, once each
    assert ("input", "emb:out") in weighted and ("blocks.0.n", "add0") in weighted
    o2o = {(p.src, p.dst) for p in g.projections if p.kind == "one_to_one"}
    assert {("emb:out", "blocks.0.n"), ("emb:out", "add0"), ("add0", "blocks.1.n"), ("add0", "add1")} <= o2o
    assert g.populations["add0"].size == 4 * D and g.populations["add0"].rate > 0
    assert "add2" not in g.populations                         # the last sum has no reader: folded


def test_spike_maxpool_uses_module_kernel():
    """MaxPool2d(3, 2, 1) on spikes: OR neurons with fan-in 9 (not the 2x2 the shapes suggest)."""
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.c, s.n, s.p = nn.Conv2d(1, 2, 3, padding=1), _Spk(), nn.MaxPool2d(3, 2, 1)

        def forward(s, x):
            return s.p(s.n(s.c(x)))

    x = (torch.rand(2, 1, 1, 8, 8) < 0.5).float()
    g, _ = from_torch(Net(), x, neuron_types=(_Spk,))
    q = [p for p in g.projections if p.dst == "p"][0]
    assert g.populations["p"].size == 2 * 4 * 4 and g.n_synapses(q) == pytest.approx(2 * 4 * 4 * 9)


def test_from_torch_per_step_rates_and_measured_trace():
    """info['rates_t'] holds the batch-mean rate per SNN step (mean over steps = info['rates']); measured_trace
    skews it by pipeline level and feeds power_trace."""
    torch.manual_seed(0)
    net = CSNN()
    x = (torch.rand(6, 4, 1, 28, 28) < 0.3).float()
    x[3] = 0.0                                              # a silent input step shows up in the trace
    sim = Simulator.from_torch(net, x, chip="loihi2")
    rt, r = sim.trace_info["rates_t"], sim.trace_info["rates"]
    for k in ("input", "l1", "l2", "l3"):
        assert len(rt[k]) == 6 and np.mean(rt[k]) == pytest.approx(r[k], rel=1e-6)
    assert rt["input"][3] == 0.0
    spk = []
    with torch.no_grad():
        net.l1.reset_hidden(); net.l2.reset_hidden(); net.l3.reset_hidden()
        for t in range(6):
            spk.append(float(net.l1(net.p1(net.c1(x[t]))).mean()))
    assert rt["l1"] == pytest.approx(spk, rel=1e-6)
    from spikebill.api import depth, levels
    tr = sim.measured_trace("pipelined")
    assert len(tr) == 6 + depth(sim.graph)
    lv = levels(sim.graph)
    assert tr[3 + lv["input"]]["input"] == 0.0
    for k, p in sim.graph.populations.items():
        assert sum(s.get(k, 0.0) for s in tr) == pytest.approx(p.size * sum(rt[k]), rel=1e-9)
    assert len(sim.measured_trace("lockstep")) == 6
    pt = sim.power_trace(tr)
    assert pt.steps == len(tr) and pt.energy_J > 0


class _SDSA(nn.Module):
    """SDT V1-style spike-driven self-attention on (B, N, D): g = LIF(sum_tokens(Q * K)), out = LIF(proj(g * V))."""
    def __init__(s, D=8):
        super().__init__()
        s.q, s.k, s.v = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        s.qn, s.kn, s.vn, s.gn, s.on = _Spk(), _Spk(), _Spk(), _Spk(), _Spk()
        s.proj = nn.Linear(D, D)

    def forward(s, x):
        q, k, v = s.qn(s.q(x)), s.kn(s.k(x)), s.vn(s.v(x))
        g = s.gn((q * k).sum(dim=-2, keepdim=True) - 1.0)
        return s.on(s.proj(g * v))


@pytest.mark.parametrize("mode", ["excluded", "andgate", "coproc"])
def test_hadamard_products_measured_and_mapped(mode):
    """Element-wise spike x spike products (SDSA) are measured like matmuls and costed in andgate / coproc
    (release gap 32); before, they passed through uncosted and unflagged."""
    torch.manual_seed(1)
    T, B, N, D = 3, 2, 6, 8
    net = _SDSA(D)
    x = (torch.rand(T, B, N, D) < 0.5).float()
    g, info = from_torch(net, x, neuron_types=(_Spk,), attention=mode)
    ops = {u["name"]: u for u in info["unmapped_ops"] if u.get("kind") == "hadamard"}
    assert set(ops) == {"hd0", "hd1"}
    ands = 0.0
    with torch.no_grad():
        for t in range(T):
            q, k = net.qn(net.q(x[t])), net.kn(net.k(x[t]))
            ands += float(((q != 0) & (k != 0)).sum())
    assert ops["hd0"]["and_ops_per_inference"] == pytest.approx(ands / B)
    assert ops["hd0"]["a_srcs"] == ["qn"] and ops["hd0"]["b_srcs"] == ["kn"]
    P = {(p.src, p.dst): p for p in g.projections}
    pops = g.populations
    if mode == "excluded":
        assert "hd0" not in pops and any("not mapped" in w for w in info["warnings"])
    elif mode == "andgate":
        assert "hd0:and" not in pops and pops["hd0"].size == N * D            # the product is its AND neurons
        assert info["rates"]["hd0"] == pytest.approx(ops["hd0"]["out_density"])
        assert g.n_synapses(P[("qn", "hd0")]) == pytest.approx(N * D) and ("kn", "hd0") in P
        assert ("vn", "hd1") in P and not any("not mapped" in w for w in info["warnings"])
        assert Simulator(g, chip="loihi2").run(balance=False).results["value"]["energy_J"] > 0
    else:
        assert pops["hd1"].coproc_ops > 0 and pops["qn"].coproc_msgs == 1 and pops["vn"].coproc_msgs == 1
        rep = Simulator(g, chip="loihi2").run(balance=False)
        assert rep.results["value"]["split"]["coproc"] > 0


def test_nonspiking_ops_flagged():
    """softmax (functional) and LayerNorm (module) on traced tensors are listed with element counts and warned
    about, not silently passed through (release gap 32)."""
    class Net(nn.Module):
        def __init__(s, D=8):
            super().__init__()
            s.fc, s.ln, s.n1, s.fc2, s.n2 = nn.Linear(D, D), nn.LayerNorm(D), _Spk(), nn.Linear(D, D), _Spk()

        def forward(s, x):
            h = s.n1(s.ln(s.fc(x)))
            return s.n2(s.fc2(torch.softmax(h, dim=-1)))
    T, B, N, D = 2, 3, 4, 8
    x = (torch.rand(T, B, N, D) < 0.5).float()
    g, info = from_torch(Net(D), x, neuron_types=(_Spk,))
    ns = {u["op"]: u for u in info["unmapped_ops"] if u.get("kind") == "nonspiking"}
    assert set(ns) == {"softmax", "LayerNorm"}
    assert ns["softmax"]["elems_per_inference"] == pytest.approx(T * N * D)
    assert ns["LayerNorm"]["elems_per_inference"] == pytest.approx(T * N * D)
    assert any("non-spiking" in w and "softmax" in w for w in info["warnings"])


@pytest.mark.parametrize("mode", ["s", "m"])
def test_spikingjelly_rates_match_framework(mode):
    """SpikingJelly single- and multi-step networks (README 'Supported models'): the traced firing rate of every
    neuron layer equals the mean of the spikes SpikingJelly itself produces on the same input."""
    sj = pytest.importorskip("spikingjelly.activation_based")
    from spikingjelly.activation_based import neuron, layer, functional
    torch.manual_seed(0)
    x = (torch.rand(4, 2, 1, 28, 28) < 0.3).float()
    net = nn.Sequential(layer.Conv2d(1, 4, 5, step_mode=mode), neuron.LIFNode(step_mode=mode),
                        layer.Flatten(step_mode=mode), layer.Linear(4 * 24 * 24, 10, step_mode=mode),
                        neuron.LIFNode(step_mode=mode))
    for n_, p in net.named_parameters():
        if "weight" in n_:
            nn.init.uniform_(p, 0, 0.2)
    outs = {1: [], 4: []}
    hs = [net[i].register_forward_hook(lambda m, a, o, i=i: outs[i].append(o)) for i in outs]
    functional.reset_net(net)
    with torch.no_grad():
        net(x) if mode == "m" else [net(x[t]) for t in range(x.shape[0])]
    for h in hs:
        h.remove()
    ref = sorted(torch.cat([o.reshape(-1) for o in v]).mean().item() for v in outs.values())
    g, info = from_torch(net, x, multi_step=mode == "m", reset=functional.reset_net)
    got = sorted(p.rate for k, p in g.populations.items() if k != "input")
    assert len(got) == 2 and np.allclose(got, ref, atol=1e-9), (got, ref)
    assert ref[1] > 0.1                                   # the network actually spikes


def test_reduction_before_linear_head():
    """A global mean over positions before a Linear head (spiking transformers, ResNets) is a reduction population
    of the head's input size, not a dense projection from every position (release gap 48, E30)."""
    class GapNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.c = nn.Conv2d(1, 16, 3, padding=1); s.l = snn.Leaky(beta=0.9, init_hidden=True, threshold=0.3)
            s.c2 = nn.Conv2d(16, 16, 1); s.head = nn.Linear(16, 10)

        def forward(s, x):
            y = s.c2(s.l(s.c(x)))                  # graded (T*B, 16, 16, 16)
            return s.head(y.flatten(2).mean(2))     # mean over 256 positions, then Linear 16 -> 10
    torch.manual_seed(0)
    g, info = from_torch(GapNet(), (torch.rand(4, 2, 1, 16, 16) < 0.5).float())
    assert "head:in" in g.populations and g.populations["head:in"].size == 16
    into_head = [p for p in g.projections if p.dst.startswith("head")]
    assert {p.src for p in into_head if p.dst != "head:in"} == {"head:in"}
    syn = sum(g.n_synapses(p) for p in into_head if p.src == "head:in")
    assert syn == pytest.approx(16 * 10)
    assert any("reduction" in w for w in info["warnings"])


def test_rates_per_input_feed_spread():
    """info['rates_per_input']: one {pop: rate} per batch sample (checked by hand for l1); the mean over samples is
    info['rates']; the list goes straight into Simulator.spread."""
    torch.manual_seed(0)
    net = CSNN()
    x = (torch.rand(6, 5, 1, 28, 28) < torch.linspace(0.1, 0.5, 5).view(1, 5, 1, 1, 1)).float()
    g, info = from_torch(net, x)
    per = info["rates_per_input"]
    assert len(per) == 5 and set(per[0]) == {"input", "l1", "l2", "l3"}
    spk = []
    with torch.no_grad():
        net.l1.reset_hidden()
        for t in range(6):
            spk.append(net.l1(net.p1(net.c1(x[t]))))
    by_b = torch.stack(spk).transpose(0, 1).flatten(1).double().mean(1)
    assert [r["l1"] for r in per] == pytest.approx(by_b.tolist(), rel=1e-9)
    for p in per[0]:
        assert np.mean([r[p] for r in per]) == pytest.approx(info["rates"][p], rel=1e-9)
    assert per[0]["input"] < per[-1]["input"]
    sp = Simulator(g, chip="loihi2").spread(per, balance=False)
    assert len(sp["energy_samples"]) == 5


@pytest.mark.parametrize("mode", ["s", "m"])
def test_spikingjelly_rates_per_input(mode):
    """Single- and multi-step SpikingJelly (with a SeqToANNContainer-style flattened (T*B) layer in multi-step):
    the same per-sample rates in both modes."""
    pytest.importorskip("spikingjelly.activation_based")
    from spikingjelly.activation_based import neuron, layer, functional
    torch.manual_seed(0)
    x = (torch.rand(4, 3, 1, 28, 28) < 0.3).float()
    net = nn.Sequential(layer.Conv2d(1, 4, 5, step_mode=mode), neuron.LIFNode(step_mode=mode),
                        layer.MaxPool2d(2, step_mode=mode), layer.Flatten(step_mode=mode),
                        layer.Linear(4 * 12 * 12, 10, step_mode=mode), neuron.LIFNode(step_mode=mode))
    for n_, p in net.named_parameters():
        if "weight" in n_:
            nn.init.uniform_(p, 0, 0.2)
    g, info = from_torch(net, x, multi_step=mode == "m", reset=functional.reset_net)
    per = info["rates_per_input"]
    assert len(per) == 3 and {"1", "5"} <= set(per[0])
    for p in per[0]:
        assert np.mean([r[p] for r in per]) == pytest.approx(info["rates"][p], rel=1e-9)
    ref = {}
    hs = [net[i].register_forward_hook(lambda m, a, o, i=i: ref.setdefault(i, []).append(o)) for i in (1, 5)]
    functional.reset_net(net)
    with torch.no_grad():
        net(x) if mode == "m" else [net(x[t]) for t in range(4)]
    for h in hs:
        h.remove()
    for i in (1, 5):
        o = torch.stack(ref[i]) if mode == "s" else ref[i][0]
        assert [r[str(i)] for r in per] == pytest.approx(o.transpose(0, 1).flatten(1).double().mean(1).tolist(), rel=1e-9)
