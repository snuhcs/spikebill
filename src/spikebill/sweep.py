"""Run independent configurations in parallel (one process per configuration).

    from spikebill import sweep
    results = sweep.run(fn, [dict(chip="loihi2", which="low"), ...], workers=16)

`fn` must be a top-level (picklable) function taking one config dict and returning something picklable.
Results come back in the order of `configs`. Within one configuration, many steps are already batched by
`spikebill.compiled`; this module parallelises across configurations (parameter ranges, chips, mappings, models).
"""
import os
from concurrent.futures import ProcessPoolExecutor


def run(fn, configs, workers=None):
    configs = list(configs)
    if not configs:
        return []
    workers = workers or min(len(configs), max(1, (os.cpu_count() or 2) // 2))
    if workers == 1:
        return [fn(c) for c in configs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, configs))
