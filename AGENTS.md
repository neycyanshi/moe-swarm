# Agent contract

## Your write-set

You may modify **exactly one file: `solution/python/kernel.py`**. Any other changed file
rejects your candidate before it is benchmarked. Do not touch `spec/`, `bench.py`,
`swarm.py`, `config.toml`, or `docs/`. Do not add dependencies.

## Read these first, in order

1. **`docs/DESIGN.md` §1** — the contract: shapes, signature, and the four traps that make
   a plausible-looking kernel wrong.
2. **`spec/reference_official.py`** — the official reference, verbatim. ~140 lines. This is
   the definition of correct; when in doubt it wins over any description of it.
3. **`docs/DESIGN.md` §2** — the roofline analysis. Read this before choosing an
   optimization, because it decides which ones are worth anything.

## Rules

- **Reimplement, don't wrap.** No runtime calls to `flashinfer`, `deep_gemm`, or any other
  kernel library, and no importing `spec/`. Wrapping the answer scores 1.0× and teaches
  nothing; a gate rejects it.
- **Recompute every call.** Reusing pre-allocated workspace buffers is fine. Reusing
  *results* across calls is not, and a gate checks for it.
- Correctness gates speed: a wrong kernel scores zero no matter how fast it is.

## Your loop

```bash
python bench.py            # correctness + speedup + per-workload roofline verdict
```

## What actually earns a win here

17 of the 19 scored workloads are **memory-bound**, and 16 have `seq_len ≤ 80`. This is a
weight-streaming problem: at `seq_len=56` the kernel moves 1.17 GB of expert weights to do
3.3 µs of arithmetic. The dominant lever is **bytes moved**, and the main control over it is
touching only the experts that routing actually activates. At `seq_len=1` roughly one local
expert is active — loading all 32 is 32× off the floor.

Read the roofline verdict in your bench output and address the bottleneck it names. Do not
pursue optimizations the numbers don't support.

## Before you finish

State your hypothesis, what you changed, the measured effect, and one thing you would try
next.
