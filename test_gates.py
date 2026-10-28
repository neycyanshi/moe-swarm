#!/usr/bin/env python3
"""
Regression tests for the three gates (docs/DESIGN.md section 3).

    python test_gates.py          # static gate everywhere; dynamic gates if CUDA

The point of step 3 in the build order is that a harness whose gates have never
caught anything is a hope, not a harness. So every gate is asserted in both
directions: it must trip on the cheat that targets it, and it must NOT trip on
`solution/python/kernel.py`. A gate that fires on the honest kernel is worse than
no gate, because it silently deletes real candidates.

Gate 1 needs no GPU, so the library-wrapper case is verifiable on a Mac. Gates 2
and 3 execute kernels and are skipped without CUDA.
"""

import sys
from pathlib import Path

import gates

ROOT = Path(__file__).resolve().parent
REAL = ROOT / "solution" / "python"
WRAPPER = ROOT / "cheats" / "library_wrapper"
CACHED = ROOT / "cheats" / "cached_result"
IGNORES = ROOT / "cheats" / "ignores_routing"

_failures = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"{name:<44} {status}{('  -- ' + detail) if detail and not cond else ''}")
    if not cond:
        _failures.append(name)


def tripped(trips, gate):
    return [t for t in trips if t["gate"] == gate]


def main():
    have_cuda = False
    try:
        import torch
        have_cuda = torch.cuda.is_available()
    except Exception:
        pass

    # ---- Gate 1: static, always runnable -------------------------------------------
    real_static = gates.gate_no_library_calls(REAL)
    check("gate1 clean on real kernel", not real_static,
          f"unexpected trips: {[t['detail'] for t in real_static]}")

    wrap_static = gates.gate_no_library_calls(WRAPPER)
    hits = tripped(wrap_static, "no_library_calls")
    check("gate1 trips on library_wrapper", bool(hits))
    check("gate1 names the flashinfer import",
          any("flashinfer" in t["detail"] for t in hits),
          f"details: {[t['detail'] for t in hits]}")

    cached_static = gates.gate_no_library_calls(CACHED)
    check("gate1 clean on cached_result", not cached_static,
          "the cached cheat must be invisible to the static gate, "
          "otherwise it is not testing gates 2 and 3")

    ignores_static = gates.gate_no_library_calls(IGNORES)
    check("gate1 clean on ignores_routing", not ignores_static)

    # The snippet that makes a dynamic trip evidence rather than an assertion.
    check("module state found in cached_result",
          [s for s in gates.module_level_state(CACHED) if "_CACHE" in s[2]])

    if not have_cuda:
        print("\nno CUDA: gates 2 and 3 skipped (run this on a GPU machine)")
        return 1 if _failures else 0

    # ---- Gates 2 and 3: dynamic ----------------------------------------------------
    real_dyn, skipped = gates.run_dynamic_gates(str(REAL), "kernel.py::kernel")
    check("gates2+3 clean on real kernel", not real_dyn,
          f"unexpected trips: {[(t['gate'], t['detail']) for t in real_dyn]}")

    # module_level_state is advisory (gates.py) and the real kernel is EXPECTED to report
    # some: it caches compiled Triton binaries and its workspace across calls, which the
    # agent brief allows outright -- reusing buffers is fine, reusing results is not, and
    # only the dynamic gate can tell those apart. So the property under test is that
    # reporting state is not itself a trip. This assertion used to read "no module state
    # in real kernel", which quietly outlawed every caching kernel, including both of the
    # kernels that now score highest.
    check("module state on the real kernel is advisory, not a trip", not real_dyn,
          f"state reported: {[s[2] for s in gates.module_level_state(REAL)]}")

    # Each cheat must trip EXACTLY the gate aimed at it. Gates 2 and 3 once shared a
    # predicate and both fired on one observation, which double-counted a single piece
    # of evidence in hacks.jsonl; these two assertions are what keep them separated.
    cached_dyn, _ = gates.run_dynamic_gates(str(CACHED), "kernel.py::kernel")
    check("gate3 trips on cached_result", bool(tripped(cached_dyn, "no_cross_call_state")))
    check("gate2 does NOT trip on cached_result", not tripped(cached_dyn, "input_sensitivity"),
          "caching is a state failure, not an input-sensitivity failure; "
          "reporting both double-counts one observation")

    ignores_dyn, _ = gates.run_dynamic_gates(str(IGNORES), "kernel.py::kernel")
    check("gate2 trips on ignores_routing", bool(tripped(ignores_dyn, "input_sensitivity")))
    check("gate3 does NOT trip on ignores_routing",
          not tripped(ignores_dyn, "no_cross_call_state"),
          "it is stateless and deterministic; only gate 2 should see it")

    # The wrapper must be clean dynamically -- that asymmetry is why gate 1 exists.
    wrap_dyn, _ = gates.run_dynamic_gates(str(WRAPPER), "kernel.py::kernel")
    check("gates2+3 clean on library_wrapper", not wrap_dyn,
          f"unexpected trips: {[(t['gate'], t['detail']) for t in wrap_dyn]}")

    return 1 if _failures else 0


if __name__ == "__main__":
    rc = main()
    print("\n" + ("all gate tests passed" if not _failures
                  else f"{len(_failures)} FAILED: {_failures}"))
    sys.exit(rc)
