# KERNELS — what the surviving kernels do, and how they got there

**A record, written after the fact.** The leaderboard records scores; this records
*kernels* — what the surviving ones turned out to be, once they existed. It is not a spec,
not a plan, and not an input to anything: agents never see it, `swarm.py` never reads it,
and nothing here constrains what the next kernel may do. Write it in the past tense about
code that is already measured.

Update it whenever a lineage slot changes. A round prunes its worktrees, so the only other
account of what a kernel tried is a 1.4 MB agent transcript, and the kernel files themselves
carry docstrings that have been wrong before (see the last section).

## Genealogy

```
touch_fewer_experts ─┬─ r2-a5-tile_quantization ─► r3-a0-combine_lineages
                     │                         ╳
                     │                            └─► r4-a5-small_t_parallelism
                     └─ touch_fewer_experts-simplified          ╳
                            └─ r1-a1-epilogue_scales ─► r2-a0 ─┐   └─► r5-a6-freestyle
                                                               │        (round 5) [lineage_2]
                                              r3-a3-one_launch ┘
                                                    └─► r4-a6-freestyle
                                                              └─► r5-a5-small_t_parallelism
                                                                    (round 5) [lineage_1]
```

**Three consecutive rounds have cross-seeded the two slots**, each survivor descending from
the other slot's kernel. They converged in round 3 and have stayed converged.

Scores before 2026-09-16 were measured on an unpinned clock and are not comparable with
these (`DESIGN.md` §3).

**An agent's name is its strategy *hint*, not a description of what it built.**
`tile_quantization` implemented split-K and CUDA graph capture; `combine_lineages`
implemented activation requantization; `small_t_parallelism` built graph-capture sessions in
two successive rounds. Read the code, not the name.

## The shared base

Everything below is common to both, inherited unchanged, and is most of each file:

- **Weights read from HBM exactly once, as fp8, into the MMA with no dequantized copy.** The 128x128 block scales are
  folded into the fp32 accumulator once per K-block (DeepGEMM-style promotion), which is the
  only place they can go if the weights are never materialized. Activation scales ride along
  the same way, so `hidden_states` stays fp8 and is gathered by token id inside the kernel.
- **GEMM1 owns both SwiGLU halves in one CTA** (output blocks `nb` and `nb+16`), so the
  `[n, 4096]` pre-activation never reaches HBM.
- **No grid indexed by expert id.** Routing activates 3 of 32 local experts at seq_len 1 and
  28 at 80, and a grid indexed by expert spends resident CTA slots on the empty ones. Both
  build a flat work list in-kernel from the 32-int count vector — cumsum to row-tile
  boundaries, then a 32-wide compare for which expert a flat tile index lands in.
- **Four launches, no host sync.** Routing, top-k and per-expert bucketing are one Triton
  kernel; the GEMMs read counts from device memory and derive their own row offsets. After
  the first call each key goes straight to four `CompiledKernel.run` invocations with
  pre-built argument tails, because `JITFunction.run` cost 53 us per call in
  specialization-key rebuilding alone.

## lineage_1 — `r5-a5-small_t_parallelism`, 3.0091x full / 3.0209 light

Seeded from `r4-a6-freestyle` and barely changed from it: the whole round added a
**capture threshold for the combine epilogue** (`_COMB_CAP`, `_COMB_T`), +119 lines. Two
measurements in a two-hour phase, so the kernel is thin on evidence from the agent itself;
the quiet re-rank (3.0212 / 3.0195 / 3.0209) and the full sweep are what stand behind it.

It inherits the whole accumulated stack: raw graph replay, split-K, requantization,
active-expert compaction, the bf16 path, banded A staging, dual-geometry GEMM1.

**Small T (T <= 16) is near its practical floor:** what is left is worth under +0.08 on
the score, about 2%. Each call pays the oracle's flush (~8 µs of dirty-L2 write-back that
lands in GEMM1) and four kernels' start-up, and a plain read of GEMM2's 44 MB after that
flush takes 16.4 µs against GEMM2's 20.2. Tried and rejected for GEMM2: more pipeline
stages (`s2` 5, 6), split-K through the combine plane (KS 2-8), a 64-wide tile, a
contiguous weight layout. Two small wins remain, in `BACKLOG.md`.

## lineage_2 — `r5-a6-freestyle`, 3.0019x full / 3.0178 light

Seeded from `r4-a5-small_t_parallelism`. Added a **second set of expert staging buffers**
(`_EBN1B`, `_EBN2B`, `_ES1B`, `_ES2B`, `_EW1A`, `_EW1B`, `_EW2B`) plus `_EKSK`, `_EMT` and a
`_PBT_T` threshold — double-buffering the per-expert weight and scale staging so a replay can
overlap one expert's load with the next expert's MMA.

At 1501 lines it is the largest kernel in the project.

## The prefill path, on top of lineage_1

From T=512 up, `kernel()` leaves the Triton path for a separate one built on DeepGEMM's
SM90 FP8 grouped GEMM. Every shorter call runs lineage_1 unchanged.

```
route (lineage_1's kernel) -> permute -> GEMM1 + SwiGLU + fp8 requant -> GEMM2 -> combine
```

**One file.** The CUDA source lives at the end of `kernel.py` as two strings:
`_PF_GEMM_SRC`, the GEMM, and `_PF_HOST_SRC`, its host launcher (tensor maps, cluster
launch, a pybind11 binding over raw pointers). `swarm.py` stores, seeds and scores a kernel
as `kernel.py` alone, and the separate `.cu` files this path started as would have failed
every T >= 512 workload there. At first use the two are spliced, written to a hash-named
path and compiled with `torch.utils.cpp_extension.load`; every process that sees the same
source shares the build. Cold, the build took 24 s and the first call another 1.9 s of
Triton JIT; warm, it is a cache hit. No torch header is included anywhere, which is what
keeps the build short. On anything but sm_90, or if the build fails, `kernel()` sends every
call to the Triton path and says why once on stderr.

**Adapted from DeepGEMM** (commit 1fc9c66, MIT; the one wgmma wrapper is CUTLASS's,
BSD-3): `sm90_fp8_gemm_1d2d_impl` and the psum-layout branch of its scheduler. Kept: the
warp-specialized TMA producer, two wgmma consumer warpgroups on a 128x128 tile, per-K-block
promotion of the 1x128 activation and 128x128 weight scales, 2-CTA TMA multicast of A, the
STSM + TMA-store epilogue. Changed:

- CUTLASS/CuTe primitives replaced by the PTX they lower to, so only CUDA headers are
  needed — CUTLASS cannot be located at eval time without importing flashinfer.
- The psum row offsets are derived in-kernel from the 32 routing counts, so no host sync
  and no offsets buffer.
- The producer bulk-loads each tile's B-scale row onto the tile's first stage barrier,
  double-buffered, instead of the math warps loading it behind a barrier per tile.
- Tile i-1 is stored while tile i's first wgmma batch runs (the first K block is peeled so
  only it carries the epilogue), and each consumer warpgroup stores its own 64 rows behind
  its own barrier. Worth 28 µs on GEMM2.
- Two accumulator sets: block k+1's wgmma batch is issued before block k is promoted, so
  the promotion FMAs run while the tensor cores work (`wgmma.wait_group 1`). The pair loop
  is fully unrolled; across a back-edge ptxas cannot pair the wait with its batch and
  serializes every wgmma. It frees each stage one block later, which costs lookahead, so
  GEMM1 is built both ways and `kernel.py` takes the overlapped one from T=4096: at
  T=14107 it is 899 → 867 µs, at T=901 (weight streaming) 277 → 292 µs. GEMM2 measured
  no worse overlapped at any length.
- GEMM1 fuses SwiGLU and the fp8 requantization. Tile j loads rows [64j, 64j+64) of both
  weight halves as two TMA boxes, so x1 and x2 for the same 64 columns sit in one thread,
  each half with its own B scale. The 128-column quantization block spans the two CTAs of
  a cluster, which exchange per-row amax through distributed shared memory before writing
  fp8 and the shared scale.

**Measured** GPU time per kernel, TFLOP/s counted on real (unpadded) rows:

| T | rows | GEMM1 | GEMM2 | combine | permute | route | call (light) | whole call |
|---|---|---|---|---|---|---|---|---|
| 11948 | 10,409 | 628 µs, 973 TF/s | 318 µs, 962 TF/s | 92 | 43 | 44 | 1133 µs | 809 TF/s |
| 14107 | 16,113 | 872 µs, 1085 TF/s | 451 µs, 1050 TF/s | 126 | 66 | 51 | 1580 µs | 898 TF/s |

The SM clock is locked at 1400 MHz, where dense fp8 peaks near 1.5 PFLOP/s. ncu put the
unfused GEMM1 at 84% of L2 throughput, which is what multicast addressed (970 → 854 µs).

Same session, same node, against lineage_1. Light divides both by the `node.json` table;
the full columns are lineage_1's second sweep of the session and the prefill build's last.
Below T=512 the two run identical code, so the light columns there are run-to-run noise
and the full columns show how much larger the sweep's noise is on this node:

| T | lineage_1 µs | with prefill µs | lineage_1 x | with prefill x | full: lineage_1 x | full: with prefill x |
|---|---|---|---|---|---|---|
| 1 | 70.5 | 70.2 | 6.907 | 6.940 | 6.852 | 6.922 |
| 7 | 115.4 | 115.4 | 4.715 | 4.717 | 4.676 | 4.862 |
| 14 | 170.0 | 169.7 | 3.501 | 3.505 | 2.300 | 1.930 |
| 15 | 95.1 | 95.5 | 5.447 | 5.426 | 5.004 | 5.732 |
| 16 | 180.2 | 180.2 | 3.365 | 3.366 | 2.213 | 2.160 |
| 32 | 284.1 | 282.2 | 2.554 | 2.571 | 2.455 | 2.514 |
| 52 | 233.4 | 233.3 | 2.852 | 2.854 | 1.738 | 3.631 |
| 53 | 296.1 | 295.5 | 2.510 | 2.515 | 1.432 | 1.359 |
| 54 | 288.7 | 288.3 | 2.508 | 2.511 | 1.448 | 1.467 |
| 55 | 296.0 | 295.4 | 2.496 | 2.501 | 1.534 | 1.436 |
| 56 | 323.1 | 323.9 | 2.370 | 2.364 | 1.334 | 2.287 |
| 57 | 315.4 | 314.5 | 2.456 | 2.463 | 1.250 | 2.338 |
| 58 | 323.8 | 322.9 | 2.439 | 2.446 | 1.244 | 1.293 |
| 59 | 259.0 | 258.8 | 2.667 | 2.668 | 1.601 | 1.698 |
| 62 | 243.0 | 244.2 | 2.769 | 2.755 | 1.648 | 2.754 |
| 80 | 354.6 | 355.1 | 2.389 | 2.385 | 2.728 | 2.331 |
| 901 | 502.3 | 459.0 | 1.964 | 2.149 | 1.705 | 0.990 |
| 11948 | 1651.3 | 1133.4 | 2.163 | 3.152 | 2.055 | 3.090 |
| 14107 | 2383.4 | 1579.6 | 1.985 | 2.995 | 1.825 | 2.608 |
| **score** | | | **3.0556** | **3.1728** | **2.3707** | **2.7054** |

Earlier sweeps of the session, before the overlapped promotion, read 2.385 (18/19: the
CUTLASS baseline failed correctness on T=1) against 2.454, and 2.371 against 2.633.

**Measured and rejected:**

- Ping-pong scheduling for GEMM2 (each consumer warpgroup on its own tile, so one's
  epilogue hides under the other's MMA): not built. Dropping the TMA store saves 11 µs at
  T=14107, which bounds what hiding the epilogue can win. Dropping the whole epilogue
  saved 87 µs, but only because the compiler then deletes the promotion FMAs as dead.
  That cost is what the two accumulator sets went after.
- Retiring a block early when the next stage has not landed (an `mbarrier.test_wait`
  branch around the wgmma issue): ptxas serialized the wgmmas, GEMM1 737 µs at T=901.

- Gathering A by token id in GEMM1's producer (cp.async into the swizzled layout) instead
  of a permute pass: +115 µs on GEMM1 against the 62 µs pass it removes.
- `/` in the fused epilogue: IEEE division brings a called slow path, and a `CALL`
  anywhere in the kernel cost GEMM1 90 µs. `__fdividef` has none.
- Writing single-expert tokens straight from GEMM2's epilogue: real routing gives 43% of
  tokens no local expert and puts 84% of rows in multi-expert tokens, so it would save
  about a sixth of the combine.
- The prefill path below T=512: at T=80 it took 393 µs against 355, where 28 experts with
  a few rows each pad to a 128-row block apiece.

## Why they are kept as a pair

Slot 2 is chosen by portfolio score — what it adds where the leader is weak — not by rank.

**That premise has failed for three rounds.** Each round has seeded both survivors from the
other slot, so they converged in round 3 and have stayed converged: round 4 saw both
independently attack launch overhead, and round 5's two winners differ from their
cross-seeded parents by a handful of symbols each. On the full sweep they are 0.24% apart
(3.0091 and 3.0019).

A portfolio of two kernels this similar buys nothing. The cheapest fix is the prompt: retire
whichever hint keeps merging them, and give agents disjoint workload regions so their edits
cannot collide even from the same seed. Adding a structural-distance term to slot 2's
selection was considered and rejected as machinery for something a hint already does.

## Hazard: docstrings outlive the code they describe

Each round seeds an agent with its lineage's file, so a docstring that no longer matches
its code misinforms the next generation. It has happened: an agent added split-K and CUDA
graph capture and left the paragraph saying both were absent.

The agent prompt now says the docstring is inherited and still true, that changing what the
kernel does means fixing any sentence that makes false, and that a "what is left to do"
section should not be kept at all — that half goes stale the moment anyone acts on it, and
duplicates the roofline verdict the harness measures fresh every round.

A harness check was considered and dropped: it could only verify that the docstring *moved*
when the code did, not that it matched, and since most agents time out mid-edit it would
have fired nearly every round and been ignored. Reading the diff is the check.
