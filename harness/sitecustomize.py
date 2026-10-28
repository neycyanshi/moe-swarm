"""Disable the oracle's per-iteration argument clone, in this process and its children.

`flashinfer_bench.bench.timing.time_runnable` passes `setup=lambda: _clone_args(args)`,
so every timed iteration first allocates and copies each argument tensor -- for this
definition ~1.4 GB of expert weights, 200 iterations per trial. The clone exists to stop
a submission caching results or mutating its inputs; the three gates in `gates.py` cover
that for us, so we pay only its cost.

The clone's *duration* is excluded from the timing (setup runs, then a synchronize, then
the start event), but its aftermath is not: 1.4 GB of allocate-copy-free per iteration
leaves the caching allocator and the cache hierarchy in a different state every time, and
that state lands inside the next timed window. It is a roughly fixed perturbation, so it
swamps a 0.1 ms workload and is invisible at 8.85 ms -- which is why the small-T workloads
carried nearly all of the score's variance. Measured over 8 repeats of one unchanged
kernel on T=1,7,14,52: removing it took the score spread from 61% to 15%, the T=7 latency
spread from 116% to 15%, and the T=14 baseline spread from 152% to 4%.

The timed section runs in a `multiprocessing` **spawn** child, which re-imports from
scratch, so patching in the parent alone does nothing. A spawn child does inherit
`os.environ`, and Python imports `sitecustomize` at startup from anything on
`PYTHONPATH` -- which is why this lives in a file with this name rather than inside
`bench.py`. It stays inert unless `KS_NO_ARG_CLONE=1`, so putting this directory on
`PYTHONPATH` cannot quietly change an unrelated process.

Each child that patches successfully appends its pid to `KS_NO_ARG_CLONE_MARKER`, which
is how `bench.py` proves the metric it reports is the metric it actually measured,
instead of assuming the patch reached the process that held the stopwatch.
"""

import os

if os.environ.get("KS_NO_ARG_CLONE") == "1":
    try:
        import flashinfer_bench.bench.timing as _timing

        _timing._clone_args = lambda args: args
        _marker = os.environ.get("KS_NO_ARG_CLONE_MARKER")
        if _marker:
            with open(_marker, "a") as _fh:
                _fh.write(f"{os.getpid()}\n")
    except Exception:
        # Never break a process over this. bench.py finds the empty marker and refuses to
        # report a score, rather than labelling a clone-perturbed number as a clean one.
        pass
