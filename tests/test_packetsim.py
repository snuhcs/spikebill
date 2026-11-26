import numpy as np
import pytest

from spikebill import packetsim, traffic


def net():
    return packetsim.Net(8, 8, 4, 4, t_link=1.0, t_chip=10.0)


def test_single_packet_no_contention():
    n = net()
    t, hops = packetsim.simulate(n, [(0, 0.0, (0, 0), [(2, 1)])], t_hop=0.5)
    assert hops == 3 and t == pytest.approx(3 * 1.5)


def test_chip_link_and_fifo():
    n = net()
    # (3,0) -> (4,0) crosses a chip boundary (t_chip 10); two packets at t=0 queue on it
    pk = [(0, 0.0, (3, 0), [(4, 0)]), (1, 0.0, (3, 0), [(4, 0)])]
    t, hops = packetsim.simulate(n, pk, t_hop=0.5)
    assert hops == 2 and t == pytest.approx(20 + 0.5)


def test_tree_link_count_matches_aggregate():
    n = net()
    rng = np.random.default_rng(0)
    dsts = [tuple(rng.integers(0, 8, 2)) for _ in range(12)]
    links = packetsim._tree_links(n, 2, 5, dsts)
    diff = traffic.new_diff(8, 8)

    class C:  # chip geometry for _route_tree
        tiles_x, tiles_y = 4, 4
    h, _, _ = traffic._route_tree(diff, np.array([2, 5]), np.array(dsts), 1.0, C)
    assert len(links) == h
    t, hops = packetsim.simulate(n, [(0, 0.0, (2, 5), dsts)], t_hop=0.0)
    assert hops == h


def test_inject_schedule():
    t = packetsim.inject_schedule([0, 0, 1, 0], 2.0, np.random.default_rng(0))
    assert sorted(t[[0, 1, 3]]) == [0.0, 2.0, 4.0] and t[2] == 0.0


def test_mixed_equals_tree_builder():
    n = net()
    rng = np.random.default_rng(1)
    P = [(0, float(rng.integers(0, 3)), tuple(rng.integers(0, 8, 2)), [tuple(rng.integers(0, 8, 2))]) for _ in range(200)]
    t_a, h_a = packetsim.simulate(n, P, t_hop=0.3)
    uni = tuple(np.array(v) for v in zip(*[(t, s[0], s[1], d[0][0], d[0][1]) for _, t, s, d in P]))
    t_b, h_b = packetsim.simulate_mixed(n, uni, [], t_hop=0.3)
    assert h_a == h_b and t_a == pytest.approx(t_b)


def test_3d_tree_matches_aggregate_and_unicast():
    n = packetsim.Net(8, 8, 4, 4, t_link=1.0, t_chip=10.0, Z=3, t_vlink=2.0, t_hop=0.5, t_vhop=0.25)
    rng = np.random.default_rng(4)
    dsts = [tuple(int(v) for v in (rng.integers(0, 8), rng.integers(0, 8), rng.integers(0, 3))) for _ in range(15)]
    links = packetsim._tree_links(n, 2, 5, dsts, 1)
    diff = traffic.new_diff(8, 8, 3)

    class C:
        tiles_x, tiles_y = 4, 4
    h, _, hv = traffic._route_tree(diff, np.array([2, 5, 1]), np.array(dsts), 1.0, C)
    assert len(links) == h + hv
    # single 3D unicast: (0,0,0) -> (1,1,2): 1 x hop, 1 y hop, 2 vertical hops
    t, hops = packetsim.simulate_mixed(n, (np.array([0.0]), np.array([0]), np.array([0]), np.array([1]), np.array([1]),
                                           np.array([0]), np.array([2])), [], t_hop=0.5)
    assert hops == 4 and t == pytest.approx(2 * (1.0 + 0.5) + 2 * (2.0 + 0.25))


def test_without_numba_same_result():
    """numba is optional (release gap 37): with it blocked the kernels run as plain Python with the same result."""
    import os
    import subprocess
    import sys
    code = ("import sys, warnings; sys.modules['numba'] = None\n"
            "import numpy as np\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    from spikebill import packetsim\n"
            "assert not packetsim.HAVE_NUMBA and any('numba' in str(x.message) for x in w)\n"
            "rng = np.random.default_rng(0)\n"
            "n = packetsim.Net(8, 8, 4, 4, t_link=1.0, t_chip=10.0)\n"
            "pk = [(i, float(i % 3), tuple(int(v) for v in rng.integers(0, 8, 2)),\n"
            "       [tuple(int(v) for v in rng.integers(0, 8, 2)) for _ in range(4)]) for i in range(40)]\n"
            "print(repr(tuple(float(x) for x in packetsim.simulate(n, pk, t_hop=0.5))))\n")
    import spikebill
    root = os.path.dirname(os.path.dirname(os.path.abspath(spikebill.__file__)))   # code/ here, src/ in the release
    env = dict(os.environ, PYTHONPATH=root)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    rng = np.random.default_rng(0)
    n = packetsim.Net(8, 8, 4, 4, t_link=1.0, t_chip=10.0)
    pk = [(i, float(i % 3), tuple(int(v) for v in rng.integers(0, 8, 2)),
           [tuple(int(v) for v in rng.integers(0, 8, 2)) for _ in range(4)]) for i in range(40)]
    assert r.stdout.strip() == repr(tuple(float(x) for x in packetsim.simulate(n, pk, t_hop=0.5)))
