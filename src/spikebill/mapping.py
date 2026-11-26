"""Partition populations onto cores under per-core limits and place the cores on the tile mesh.

Greedy model (first, conservative version):
- synapses are stored at the destination core (Loihi-style); a core holds neurons of one population only;
- neurons per core n is the largest value meeting: n <= max_neurons, synapse memory n * fanin * bits
  (or the shared weights for conv-like projections) <= syn_mem_bits, input axons <= max_fanin_axons;
- cores are filled tile by tile along a snake path over tiles and chips, in population order, so
  consecutive populations sit close together (locality-preserving). Chips form a near-square board.
  `order` permutes the fill order (spikebill.place searches it: the placement baseline).
Violations that no n can fix are recorded in `Mapping.warnings`, not silently dropped.

`from_placement` takes an explicit neuron-to-core assignment instead (e.g. the one another tool used).
Global core id = chip * cores_per_chip + tile * cores_per_tile + local core.
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PopPlacement:
    name: str
    size: int
    cores: np.ndarray       # indices into the Mapping's used-core arrays
    counts: np.ndarray      # neurons of this population on each of those cores
    limited_by: str = "explicit"
    act: np.ndarray = None  # share of the population's spikes on each core (default: by neuron count)
    layout: tuple = None    # (positions per core, channels per core) for weight-shared (token / conv) targets

    @property
    def share(self):
        return self.act if self.act is not None else self.counts / self.counts.sum()

    def set_rates(self, core_pos, rates):
        """Activity share per core from mean firing rates per neuron (core_pos: core position of each neuron)."""
        w = np.bincount(np.asarray(core_pos), weights=np.asarray(rates, dtype=float), minlength=self.n_cores)
        self.act = w / w.sum() if w.sum() > 0 else None

    @property
    def n_cores(self):
        return len(self.cores)

    @property
    def per_core(self):
        return int(self.counts.max())


@dataclass
class Mapping:
    pops: dict              # name -> PopPlacement
    core_id: np.ndarray     # (n_used,) global core id
    xy: np.ndarray          # (n_used, 2) global tile-mesh coordinates of each used core
    chip_of: np.ndarray     # (n_used,) chip index
    tile_of: np.ndarray     # (n_used,) global tile index
    n_chips: int
    board: tuple            # (chips_x, chips_y)
    mesh: tuple             # (W, H) global mesh size in tiles
    warnings: list = field(default_factory=list)
    z: np.ndarray = None    # (n_used,) tile layer of each used core (3D-stacked chips); zeros for planar chips
    layers: int = 1

    @property
    def n_cores(self):
        return len(self.xy)

    @property
    def n_tiles(self):
        return len(np.unique(self.tile_of))


def _incoming(graph, pop):
    return [p for p in graph.projections if p.dst == pop]


def shared_layout(graph, chip, pop_name):
    """Core layout for a population fed by weight-shared projections (share_group = positions per channel):
    a core holds t positions x c channels; weights are shared over its t positions.
      memory: c * sum(fan-in x bits of shared inputs) + t*c * (private bits per neuron) <= syn_mem_bits
      axons:  t * sum(axon fan-in of shared inputs) + t*c * (private fan-in) <= max_fanin_axons
    Returns (t, c, n = t*c, binding limit) maximising n, or None if the population has no shared input."""
    inc = _incoming(graph, pop_name)
    sh = [p for p in inc if p.shared_weights and p.share_group]
    if not sh:
        return None
    pop = graph.populations[pop_name]
    G = sh[0].share_group
    C_out = max(1, pop.size // G)
    mem_c = sum(graph.fanin(p) * p.weight_bits for p in sh)
    ax_t = sum((p.axon_fanin or graph.fanin(p)) for p in sh)
    priv = [p for p in inc if p not in sh]
    mem_n = sum(graph.fanin(p) * p.weight_bits for p in priv)
    ax_n = sum(graph.fanin(p) for p in priv)
    M, A, NM = chip["syn_mem_bits"], chip["max_fanin_axons"], int(chip["max_neurons"])
    best = None
    for t in range(1, G + 1):
        if t * ax_t > A:
            break
        c = C_out
        lim = "channels"
        cm = int(M // (mem_c + t * mem_n)) if (mem_c + t * mem_n) > 0 else C_out
        if cm < c:
            c, lim = cm, "syn_mem_bits"
        if ax_n > 0:
            ca = int((A - t * ax_t) // (t * ax_n))
            if ca < c:
                c, lim = ca, "max_fanin_axons"
        if t * c > NM:
            c, lim = NM // t, "max_neurons"
        if c >= 1 and (best is None or t * c > best[2]):
            best = (t, c, t * c, lim)
    return best


def neurons_per_core(graph, chip, pop_name):
    """Largest n meeting the core limits, and the name of the binding limit."""
    pop = graph.populations[pop_name]
    inc = _incoming(graph, pop_name)
    lay = shared_layout(graph, chip, pop_name)
    if lay is not None:
        t, c, n, lim = lay
        return max(1, min(n, pop.size)), lim, []
    if any(p.shared_weights and p.share_group for p in inc):
        return 1, "infeasible", [f"{pop_name}: no shared layout fits one position; using 1 neuron per core"]
    limits = {"max_neurons": int(chip["max_neurons"])}
    warnings = []

    def mem_bits(n):
        """Synapse memory of a core holding n neurons of this population."""
        tot = 0.0
        for p in inc:
            f = graph.fanin(p) * p.weight_bits
            if p.shared_weights and p.share_group:
                ch = min(n, math.ceil(n / p.share_group) + 1)       # channels held (channel-major layout)
                tot += f * ch
            elif p.shared_weights:
                tot += (p.n_weights or 0) * p.weight_bits
            else:
                tot += f * n
        return tot

    if inc:
        if mem_bits(1) > chip["syn_mem_bits"]:
            warnings.append(f"{pop_name}: one neuron needs {mem_bits(1):.3g} synapse bits > core memory; "
                            f"memory limit ignored (would need dendrite splitting)")
        else:
            lo, hi = 1, limits["max_neurons"]
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if mem_bits(mid) <= chip["syn_mem_bits"]:
                    lo = mid
                else:
                    hi = mid - 1
            if lo < limits["max_neurons"]:
                limits["syn_mem_bits"] = lo
    max_ax = chip["max_fanin_axons"]

    def axons(n):
        # distinct source axons seen by a core holding n neurons: every input of each neuron is distinct for
        # dense/random projections; local (conv, one_to_one) ones share inputs between neighbours, so about
        # the src neurons covering those n dst neurons plus one receptive field
        tot = 0.0
        for p in inc:
            S = graph.populations[p.src].size
            if p.kind in ("conv", "one_to_one"):
                tot += min(S, n * S / pop.size + graph.fanin(p))
            else:
                tot += min(S, n * graph.fanin(p))
        return tot

    if inc:
        lo, hi = 0, limits["max_neurons"]
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if axons(mid) <= max_ax:
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            warnings.append(f"{pop_name}: one neuron needs {axons(1):.3g} input axons > {max_ax:.0f}; "
                            f"input-axon limit ignored (would need relay neurons)")
        else:
            limits["max_fanin_axons"] = lo
    which = min(limits, key=limits.get)
    if limits[which] < 1:
        warnings.append(f"{pop_name}: {which} allows <1 neuron per core; using 1")
    n = max(1, min(limits[which], pop.size))
    return n, which, warnings


def _board_shape(n_chips):
    cx = math.ceil(math.sqrt(n_chips))
    cy = math.ceil(n_chips / cx)
    return cx, cy


def _snake(nx, ny):
    out = []
    for y in range(ny):
        xs = range(nx) if y % 2 == 0 else range(nx - 1, -1, -1)
        out.extend((x, y) for x in xs)
    return np.array(out)


def _geometry(chip, core_ids, n_chips):
    """Tile coordinates, chip and global tile index of global core ids (SANA-FE tile numbering in a chip)."""
    bx, by = _board_shape(n_chips)
    board = _snake(bx, by)
    cpc = chip.cores_per_chip
    ch = core_ids // cpc
    tile = (core_ids % cpc) // chip.cores_per_tile
    tx, ty = chip.tile_xy(tile)
    xy = np.stack([board[ch, 0] * chip.tiles_x + tx, board[ch, 1] * chip.tiles_y + ty], axis=1)
    return xy, ch, ch * chip.tiles_per_chip + tile, (bx, by), chip.tile_z(tile)


def _finish(chip, placements, core_ids, warnings):
    core_ids = np.asarray(core_ids, dtype=np.int64)
    n_chips = int(core_ids.max() // chip.cores_per_chip) + 1 if len(core_ids) else 1
    xy, ch, tile, board, z = _geometry(chip, core_ids, n_chips)
    return Mapping(placements, core_ids, xy, ch, tile, n_chips, board,
                   (board[0] * chip.tiles_x, board[1] * chip.tiles_y), warnings, z.astype(np.int64), chip.tiles_z)


def _tile_order(chip, n_chips):
    """Global core ids in fill order: tiles along a snake within each chip, chips in order."""
    snake = _snake(chip.tiles_x, chip.tiles_y)
    layer_ids = snake[:, 0] * chip.tiles_y + snake[:, 1]     # inverse of tile_xy within a layer
    # layers alternate direction so that consecutive tiles stay vertically adjacent between layers
    tile_ids = np.concatenate([(layer_ids if zz % 2 == 0 else layer_ids[::-1]) + zz * chip.tiles_per_layer
                               for zz in range(chip.tiles_z)])
    per_chip = (tile_ids[:, None] * chip.cores_per_tile + np.arange(chip.cores_per_tile)[None, :]).ravel()
    return np.concatenate([c * chip.cores_per_chip + per_chip for c in range(n_chips)])


def map_graph(graph, chip, caps=None, order=None):
    """caps: optional {population: max neurons per core} (load balancing, see spikebill.balance).
    order: optional fill order along the tile path (default: graph order; see spikebill.place). Entries are population
    names (all of its cores in one block) or (name, n_cores) segments: a population may then appear in several
    non-adjacent blocks whose core counts sum to its cores; its neurons fill the blocks in path order."""
    caps = caps or {}
    sizes, warnings, info = [], [], {}
    for name in graph.populations:
        pop = graph.populations[name]
        n, which, w = neurons_per_core(graph, chip, name)
        if name in caps and caps[name] < n:
            n, which = max(1, int(caps[name])), "balance"
        warnings += w
        info[name] = (pop.size, n, which, math.ceil(pop.size / n))
    if order is None:
        segs = [(k, v[3]) for k, v in info.items()]
    else:
        segs = [(e, info[e][3]) if isinstance(e, str) else (e[0], int(e[1])) for e in order]
        got = {}
        for k, c in segs:
            if k not in info or c < 1:
                raise ValueError(f"order: unknown population or empty segment {k!r}")
            got[k] = got.get(k, 0) + c
        if got != {k: v[3] for k, v in info.items()}:
            raise ValueError("order must cover every population's cores exactly once")
    total = sum(v[3] for v in info.values())
    slots = _tile_order(chip, math.ceil(total / chip.cores_per_chip))
    idx, i = {k: [] for k in info}, 0
    for k, c in segs:
        idx[k].append(np.arange(i, i + c))
        i += c
    placements = {}
    for name in graph.populations:                                  # graph order, whatever the fill order
        size, n, which, k = info[name]
        counts = np.full(k, n)
        counts[-1] = size - n * (k - 1)
        lay = shared_layout(graph, chip, name)
        placements[name] = PopPlacement(name, size, np.concatenate(idx[name]), counts, which,
                                        layout=(lay[0], lay[1]) if lay else None)
    return _finish(chip, placements, slots[:total], warnings)


def from_placement(graph, chip, placement):
    """placement: {pop: [(global core id, n neurons), ...]}; cores may not be shared between populations."""
    used, placements = [], {}
    for name in graph.populations:
        cores = [c for c, _ in placement[name]]
        counts = np.array([n for _, n in placement[name]])
        if counts.sum() != graph.populations[name].size:
            raise ValueError(f"{name}: placement holds {counts.sum()} neurons, population has "
                             f"{graph.populations[name].size}")
        placements[name] = PopPlacement(name, graph.populations[name].size,
                                        np.arange(len(used), len(used) + len(cores)), counts)
        used += cores
    if len(set(used)) != len(used):
        raise ValueError("a core holds neurons of more than one population")
    return _finish(chip, placements, used, [])
