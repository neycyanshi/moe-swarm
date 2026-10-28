# DESIGN — MoE-Swarm

An agent swarm that optimizes an FP8 fused-MoE GPU kernel against a benchmark oracle.

**Coding principle: as simple as possible.** Every component here has to justify its
existence against "could we just not?". Build the fitness function before the agents.

---

## 0. Goal and deliverables

Reproduce and extend **MLSys 2026 FlashInfer AI Kernel Generation Contest, Track A
(FP8 fused MoE)**. The 2026 cycle closed (winners announced 2026-05-22), so this
reproduces and extends public artifacts rather than competing — there is no submission,
no tag, no deadline. That frees us from all submission machinery.

Target kernel: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` —
DeepSeek-V3-class dimensions (hidden 7168 = V3's hidden size, intermediate 2048 = V3's
`moe_intermediate_size`). A real frontier-model MoE layer.

**Four deliverables, in priority order.** If the schedule slips, protect the first three:

1. **Profiler/roofline feedback into the prompt** as a prose verdict naming the regime and
   the binding bottleneck. Not a metric dump — a sentence the agent can act on.
2. **The economics curve**: best speedup vs. cumulative dollars spent. Nobody publishes
   this. Maps directly onto "improve the ratio of compute-to-software output."
3. **The reward-hacking appendix**: every gate trip logged, with the offending snippet.
4. Cross-round lesson memory (deferred — see `BACKLOG.md`).

Hardware: 1× H200 141GB HBM3e (sm_90a), on preemptible cloud capacity. The contest ran on B200
(sm_100), so our absolute numbers are not leaderboard-comparable; we say so and report
our own paired baseline instead.

**The GPU machine is where the work happens.** `flashinfer-bench` does not install on macOS
at all (its `cuda-tile` dependency requires `nvidia-smi` at metadata-generation time), so
anything that imports it — `bench.py`, `swarm.py` — runs on the GPU machine under `tmux`,
pushing code and results back through git. `spec/` vendors the contract so
kernel *semantics* can still be reasoned about and unit-tested on the Mac without a GPU,
which is what `test_kernel.py` does.

---

## 1. The kernel contract

Authoritative, vendored offline in `spec/` (40 KB, fetched 2026-08-17 from the HuggingFace
dataset `flashinfer-ai/mlsys26-contest`):

| File | What it is |
|---|---|
| `spec/definition.json` | The full definition: axes, I/O signature, reference source |
| `spec/reference_official.py` | The official PyTorch reference, **verbatim** — ground truth for semantics |
| `spec/workloads.jsonl` | All 19 scored workloads with their `seq_len` and input sourcing |

Read `spec/reference_official.py` before writing kernel code. It is the definition of
correct, and it is only ~140 lines.

### Shapes (all constants except `seq_len`)

```
num_experts        256      # global expert count
num_local_experts   32      # this rank's slice -> expert parallelism EP = 8
hidden_size       7168      # H
intermediate_size 2048      # I
gemm1_out_size    4096      # 2*I, gate and up fused into ONE GEMM
block             128       # FP8 quantization block edge
```

### Signature (value-returning; `destination_passing_style = false`)

```
routing_logits        float32        [seq_len, 256]
routing_bias          bfloat16       [256]
hidden_states         float8_e4m3fn  [seq_len, 7168]
hidden_states_scale   float32        [56, seq_len]      # NOTE: transposed, [H/128, T]
gemm1_weights         float8_e4m3fn  [32, 4096, 7168]
gemm1_weights_scale   float32        [32, 32, 56]       # [E, 2I/128, H/128]
gemm2_weights         float8_e4m3fn  [32, 7168, 2048]
gemm2_weights_scale   float32        [32, 56, 16]       # [E, H/128, I/128]
local_expert_offset   int32 scalar
routed_scaling_factor float32 scalar
-> output             bfloat16       [seq_len, 7168]
```

### Semantics, and the four traps

1. **SwiGLU applies `silu` to the SECOND half.** The official reference is
   `X1 = G1[:, :I]`, `X2 = G1[:, I:]`, `C = silu(X2) * X1`. Writing the intuitive
   `silu(gate) * up` on the first half is wrong — and it is wrong in a way that still
   produces plausible-looking output. This bug was actually written and caught during
   design; do not re-introduce it.
2. **Selection uses `s_with_bias`, combination uses `s`.** `s = sigmoid(logits)`;
   `s_with_bias = s + bias` picks the groups and the top-8, but the combine weights come
   from `s` **without** bias, normalized (`/(sum + 1e-20)`) then scaled by
   `routed_scaling_factor`.
3. **Grouped routing**: reshape 256 experts into `n_group=8` groups of 32; score each
   group by the **sum of its top-2** `s_with_bias`; keep `topk_group=4` groups; take the
   global top-8 within those.
4. **Only local experts compute.** Expert `ge = local_expert_offset + le` for
   `le` in `[0, 32)`. Tokens that routed to a non-local expert contribute nothing here.
   Skip experts with zero assigned tokens — this is not an optimization, it is most of the
   performance (see §2).

Correctness rule (fixed by the contest, not ours to tune): an element fails only if
**both** `abs_err > 1.0` **and** `rel_err > 0.3`; a workload passes if **≥90%** of
elements pass (`atol=1, rtol=0.3, required_matched_ratio=0.9`). Loose on purpose — FP8.
There is no tolerance to calibrate.

---

## 2. The workload set is the most important fact in this project

The 19 scored `seq_len` values:

```
1, 7, 14, 15, 16, 32, 52, 53, 54, 55, 56, 57, 58, 59, 62, 80, 901, 11948, 14107
```

**16 of 19 are T ≤ 80.** Nine are packed into 52–62 and three into 14–16 — deliberate
tile-quantization traps, testing whether a kernel degrades at sizes that straddle a
128-row tile boundary.

### Roofline, computed from the real shapes

Per local expert, FP8 weights are `2I·H + H·I = 44.04 MB`; all 32 are **1.409 GB**.
Per token per local expert, compute is `2(H·2I) + 2(I·H) = 88.08 MFLOP`.

Because EP = 8, each token's 8 routed experts are spread over 256 experts, so the expected
number of **local** expert-visits per token is `8 × 32/256 = 1`, not 8. Expected active
local experts is `32(1 − (31/32)^T)`.

Using the achieved HBM3e bandwidth **measured per node** into `results/node.json`
(4.11-4.25 TB/s across the H200s seen so far) and ~1.5 PFLOP/s achievable FP8. Compute is identical to H100; only bandwidth differs, so every `t_mem`
below shrinks by ~1.4× against an H100 while `t_compute` is unchanged:

| T | active experts | bytes | t_mem | t_compute | ratio |
|---|---|---|---|---|---|
| 1 | 1.0 | 44 MB | 10 µs | 0.06 µs | **177× MEM** |
| 16 | 12.7 | 562 MB | 132 µs | 0.9 µs | **141× MEM** |
| 56 | 26.6 | 1.17 GB | 276 µs | 3.3 µs | **84× MEM** |
| 80 | 29.5 | 1.30 GB | 306 µs | 4.7 µs | **65× MEM** |
| 901 | 32.0 | 1.43 GB | 336 µs | 53 µs | **6× MEM** |
| 11948 | 32.0 | 1.67 GB | 392 µs | 702 µs | 0.6× compute |
| 14107 | 32.0 | 1.71 GB | 403 µs | 828 µs | 0.5× compute |

**Crossover: T ≈ 6200. 17 of 19 workloads are memory-bound = 89% of the score.** The faster
HBM3e pulls the crossover in from ~8000 on H100, but not far enough to change which
workloads land on which side — the set has nothing between 901 and 11948.

The arithmetic intensity swings 250× → 0.7× across the set, but the distribution is
lopsided: prefill-GEMM strategies target ~11% of the score.

### What follows from this

- **This is a weight-streaming problem, not a GEMM problem.** At T=56 the kernel moves
  1.17 GB to perform 3.3 µs of arithmetic.
- **The dominant lever is bytes moved**, and the only real control over it is *how many
  experts you touch*. At T=1 a correct kernel loads 44 MB; a kernel that naively sweeps
  all 32 local experts loads 1.41 GB and is 32× off the floor.
- **Per-call overhead matters.** At T=1 the memory floor is ~10 µs, so kernel-launch and
  dispatch overhead (~5–10 µs) is a first-order cost. Fewer launches, no host syncs, no
  full-tensor dequantization.
- **The official reference is not a meaningful baseline.** It materializes all weights in
  fp32 (`[32,4096,7168]` + `[32,7168,2048]` ≈ 5.6 GB) every call. Speedup over it would be
  100–1000× and would say nothing. Hence the baseline choice in §3.

---

## 3. Architecture: one oracle, two signals

```
                    ┌──────────────────────────────────────┐
   kernel.py  ──────>│ bench.py                             │
   (agent's          │                                      │
    write-set)       │  1. flashinfer-bench  -> SCORE       │──> bench.json
                     │     correctness + speedup vs         │
                     │     flashinfer_wrapper_9sdjf3        │
                     │                                      │
                     │  2. our roofline layer -> VERDICT    │
                     │     analytic bytes/FLOPs per         │
                     │     workload, % of speed-of-light,   │
                     │     prose bottleneck sentence        │
                     └──────────────────────────────────────┘
                                      │
                                      v  verdict feeds next round's prompt
                              swarm.py round loop
```

**`flashinfer-bench` is the scorer.** It owns correctness and the official metric:
arithmetic mean of `flashinfer_baseline_latency / our_latency` across the 19 workloads,
correctness-gated (any failing workload zeros the score). Using the real harness is worth
more than a hand-rolled equivalent — it is externally defined and un-gameable by us.

**Our roofline layer is the feedback.** flashinfer-bench reports latency; it does not tell
an agent *why* it is slow or which regime it is in. We add, per workload, the analytic
bytes moved and FLOPs (we know the shapes exactly, §2), the achieved fraction of the
binding roofline, and a prose verdict. This is deliberately **analytic, not profiler-based**
— it works on day one, needs no `ncu`, and cannot be blocked by cluster telemetry. `ncu`
becomes optional enrichment for top candidates, not a dependency.

Why a secondary signal at all, given the score is the score: at T ≤ 80 the score is a
ratio between two numbers that are both ~99% weight-streaming, so it moves slowly and
tells the agent nothing about headroom. "You are at 34% of DRAM roofline and touching 32
experts when routing only activates 27" is directly actionable.

**`bench.json` keeps all 19 verdicts; the agent prompt gets a rollup.** `bench.py
--prompt-view` renders a view over that record rather than a smaller version of it — the
JSON is what `swarm.py`, the economics curve, and the hacks appendix all read, so nothing
is dropped at write time. `--from-json` re-renders from a stored result and needs no GPU,
so a prompt can be rebuilt on the Mac.

Three rules make the rollup safe to depend on:

- **Failures are never summarized.** A single failing workload zeroes the score, so it is
  printed verbatim with its error tail and can never become a table row.
- **Passing workloads group by regime *and by the advice itself***, so two headroom bands
  implying the same fix collapse into one paragraph. Today that gives two groups because
  one bug dominates; once it is fixed the groups split on their own into the small-T
  overhead regime and the T ∈ [52,62] tile-quantization regime.
- **Highlighted workloads rank by score contribution — lowest speedup — not by distance
  from the roofline.** The score is an arithmetic mean of speedups, so the laggards are the
  slow-relative-to-baseline ones. T=1 sits furthest from its roofline (2% of
  speed-of-light) and is simultaneously the *best* scoring workload (0.32×), because the
  baseline is equally bad there; ranking by roofline distance would aim a whole round at
  1/19th of the metric. Anything more than 2× off its group's median is promoted out of the
  group instead of hiding inside an average.

**The baseline is `solution/baseline_cutlass_sm90`, a documented substitution.** The
contest's own baseline (`flashinfer_wrapper_9sdjf3`, wrapping `trtllm_fp8_block_scale_moe`)
cannot run on sm_90: its dispatcher is `get_trtllm_moe_sm100_module()` and it raises
`No supported CUDA architectures found for major versions [10, 12]` — sm_90 is major 9.
That is an architecture restriction, not a driver problem, and the dataset ships no
alternative.

FlashInfer's other MoE backend covers sm_90, and covers it *exclusively*:
`cutlass_fused_moe` gates DeepSeek block scaling the other way, raising
`FP8 block scaling not yet implemented for Blackwell` when `device_arch != "90"`. So the
substitute is still a FlashInfer kernel — externally defined, not authored by us, which is
what keeps "speedup" credible. Every `bench.json` carries `baseline.substituted = true`
with this reason, so no reader can mistake the score for a leaderboard-comparable one.

Its dtype contract is forced: `FusedMoeRunner::isFp8Quant()` is `!mUseDeepSeekFP8BlockScaling
&& act == fp8 && weight == fp8`, so block scaling *excludes* the fp8×fp8 runner and passing
fp8 activations yields "Could not construct fused moe op". The branch that survives is
**bf16 activation × fp8 weight**, so the baseline dequantizes activations (T×7168 — trivial)
and leaves the weights, the 1.4 GB that decides runtime, in fp8 with their block scales.
Expert parallelism maps as `ep_rank = local_expert_offset // 32`, since the runner derives
`num_experts_total = num_experts_on_rank * ep_size`.

### Gates (anti-reward-hacking)

flashinfer-bench's correctness check already kills most cheats: ignoring block scales,
mis-laying scales, recomputing routing wrongly all fail correctness. Only add gates for
what correctness cannot see:

1. **`no_library_calls`** — AST scan for imports of `flashinfer`, `deep_gemm`, or the
   reference. The contest forbids runtime calls to kernel libraries; a wrapper would score
   1.0× and teach us nothing.
2. **`input_sensitivity`** — run two routings on two **independent module instances** and
   require different outputs. Kills hardcoded outputs and ignored arguments.
3. **`no_cross_call_state`** — the same input must give the same answer regardless of what
   the kernel was called with before. Catches persisted/hoisted work reused across calls
   (the FAQ permits reused *buffers*, not reused *results*).

**Each gate owns exactly one predicate, and that took a second module instance to arrange.**
With a single instance, "B returned A's output" is equally consistent with *ignores its
inputs* and with *returned a cached A* — indistinguishable, so both gates fired on one
observation and `hacks.jsonl` double-counted a single piece of evidence. Loading the
candidate twice fixes it: instance 2 has no call history, so its B is the honest value of
`f(B)`. Gate 2 then compares two clean first calls, gate 3 compares a hot instance against
a cold one, and neither can trip on the other's failure.

Every trip appends to `results/hacks.jsonl` with round, agent, gate, and snippet. That
file is deliverable 3.

They live in `gates.py`. **A trip zeroes the score and `bench.py` rejects the candidate
before any timing** — a cheat must not be able to post a number with a footnote attached.
Gate 1 is static, so it needs no GPU and is verifiable on the Mac. Gates 2 and 3 execute
candidate code, which per §5 must never enter an orchestrator's process, so they run in a
subprocess with a hard timeout; a crash there is reported as a gate error rather than
taking down the caller.

Two details that make the record usable rather than merely present:

- **Dynamic trips still carry a snippet.** A runtime gate knows *that* a result was reused
  but not *where from*, so `gates.py` also scans for module-scope mutable containers and
  attaches them as evidence — on `cheats/cached_result` that resolves to
  `kernel.py:48: _CACHE = {}`. Advisory only: it never trips a gate by itself, since a
  module-level dict can be a legitimate constant table.
- **Gate 2 steers the routing rather than randomising it.** A random draw can route nothing
  to this rank, and one all-zero output compares equal to another for reasons that have
  nothing to do with cheating. Inputs A and B put a clear ramp on disjoint local expert
  sets, and the gate refuses to judge at all if the output is identically zero.

`cheats/` holds three deliberately-illegal fixtures — one per gate — and `test_gates.py`
asserts every gate in **both** directions: it must trip on the cheat aimed at it, must
*not* trip on `solution/python/kernel.py`, and must not trip on the other cheats. A gate
that fires on the honest kernel is worse than no gate, because it silently deletes real
candidates; a gate that fires on another gate's cheat inflates the appendix.

| fixture | gate 1 | gate 2 | gate 3 | why it is the plausible version |
|---|---|---|---|---|
| `library_wrapper` | **trips** | clean | clean | numerically perfect and fast; the locally rational move for an agent told to make FP8 MoE fast |
| `ignores_routing` | clean | **trips** | clean | author assumes the expert set is stable and hardcodes it |
| `cached_result` | clean | clean | **trips** | over-generalises "reusing buffers is allowed" to "reusing work is allowed" |

Each trips exactly one gate, which is the property that proves gates 2 and 3 no longer
share a predicate. The library wrapper passing gates 2 and 3 while the cached cheat passes
gate 1 is the argument for keeping both static and dynamic gates at all.

`hacks.jsonl` is reserved for real agent trips; the fixtures are proven by `test_gates.py`
instead, so nobody later miscounts a fixture as misbehaviour.

---

### Measurement: pin the clock, or nothing below is true

Achieved SM clock, not the kernels, was this hardware's measurement noise. The GPU idles at
345 MHz and boosts to 1980, and the harness idles it in Python between workloads, so a short
timed region can land anywhere in the ramp back up. Nothing is power-capped — 457 W peak
against a 700 W limit, no throttle reason ever active.

| | one kernel, six runs at T14107 | six kernels, light ranked against the full sweep |
|---|---|---|
| clocks free | spread **115%** | Spearman **-0.31** |
| `--lock-gpu-clocks=1400,1400` | spread **0.18%** | Spearman **+0.71** |

1400 MHz costs nothing measurable: it matched free-at-boost-2000 to within 1%.
`node-setup.sh` pins it on every bring-up, since a lock does not survive preemption, and
records `/tmp/ks-sm-clock-mhz`; `bench.py` stamps `sm_clock_locked_mhz` into every node
record; `swarm.py` refuses to start a round without it. The marker file exists because a
lock has no read-back.

A locked score is not comparable with an unlocked one: locking moved one full sweep from
1.85x to 2.55x. **A paired A/B measurement does not rescue an unpinned clock** — pairing
cancels slow common-mode drift, not a clock state that changes between one measurement and
the next.

The same applies to the oracle's per-iteration argument clone.
`flashinfer_bench.bench.timing.time_runnable` passes `setup=lambda: _clone_args(args)`, so
every timed iteration re-copies ~1.4 GB of expert weights. Its duration is excluded from the
timer; its aftermath is not, and being a roughly fixed perturbation it swamps a 0.1 ms
workload and vanishes at 8.85 ms. `harness/sitecustomize.py` removes it — the timing runs in
a `multiprocessing` spawn child, which is why it is a `sitecustomize` on `PYTHONPATH` rather
than a patch in `bench.py`. `--faithful-clone` restores it, which is what a contest-
comparable reading needs. Every `bench.json` records `arg_clone`.

### Two metrics: light ranks, the full sweep records

| | sampling | cost | repeats to | used for |
|---|---|---|---|---|
| light (`--light`) | 3 trials x 10 iterations, vs the fixed `node.json` table | **85s** | 0.06-1.5% | every ranking decision |
| full sweep | 5 trials x 200 iterations, vs a live paired baseline | **1119s** | ~2% | the recorded score, nothing else |

**Light screens the field; the full sweep selects from the screen.** Both ends of the light
ratio are measured in the same regime — the `node.json` table is sampled at `LIGHT_SAMPLING`
by `--baseline-ref`, exactly as the numerator is — so the two metrics agree: **1.002 on the
whole score**, and per workload 0.99 at T<=32, 1.01 at T=52..80 and 1.01 at T=14107.

The regimes do not cancel on their own. Measured at 5x200 the two compute-bound workloads
run ~10 s of sustained arithmetic, long enough to settle and throttle; in a 3x10 burst they
read on boosted clocks, by ~2.5x — our kernel and the CUTLASS baseline alike. The ratio only
cancels if both ends are measured the same way.

Light still only screens. The top `RANK_FINALISTS` each get a full sweep and the survivors
are chosen on that, because the full sweep is the metric being reported and ordering a
handful of pre-screened kernels on it is not the winner's curse — that comes from taking the
max of *many* noisy readings. What must never happen is mixing the two scales in one
ranking.

**"3 trials" and "median-of-3" are different threes.** One light run already times every
workload over 3 trials of 10 iterations and reduces them by mean-of-means — that is the 85s.
A *median-of-3* is three such runs, taken because the mean-of-means inside one run does not
protect against a run landing on an outlier: in 13 locked triples, 12 spread 0.06-0.35% and
one spread 1.5%, and in that one the odd run was low, so a single reading would have
understated the kernel.

Every site below is a light run. The ranking batch is 15 of them; the agents' own loops
are the bulk of a round's ~84.

| | light runs | why |
|---|---|---|
| agent inner loop | **1x** | throughput; an agent needs to see a regression, not resolve a tie |
| seed ratchet, per lineage | 3x | one reading sets the bar for a whole round, and a high draw locks the agent out of snapshotting |
| round-open health gate | **free** | reads the spread of the ratchet's own three runs |
| ranking, up to 9 kernels | 1x | enough to sort a field that spans 10% |
| ranking, top 3 only | +2x | a single reading can sit 1.5% off, the width of a real gain |

and the same top 3 each get one **full sweep**, which is what the survivors are chosen on.

A light/full ratio far from 1.00 is the warning sign: the two metrics disagree about that
kernel, and the full sweep is the one that decides. `swarm.py` logs it.

Top *three*, not two, because slot 1 goes by rank and slot 2 by portfolio score, so the
third-placed kernel can still change the answer. The agent loop is the one site left at 1x:
its bar is a median while its own readings are single, so it can snapshot on a lucky run —
which costs nothing, since the end-of-round batch re-measures everything anyway.

The full sweep runs **at most `RANK_FINALISTS` times a round**, on the screened finalists. A survivor whose kernel
did not change keeps the score already recorded for it, stamped with `full_measured_at`,
because re-sweeping identical bytes measures a number we hold. The 5x200 sampling itself is
not cut: it is the operating point every recorded number sits on, and shrinking it to save
nine minutes would invalidate all of them.

Gates and correctness come free from light: `bench.py` runs the gates before timing and a
light run checks all 19 workloads.

### The baseline leg is the least reliable number in a full sweep

A full sweep times the candidate **and** re-measures the CUTLASS baseline, per workload,
live. That second leg has twice been wrong enough to decide a round:

| round | workload | leg measured | table says | worth on the mean |
|---|---|---|---|---|
| 3 | T=7 | 1699 µs | 525 µs | **+0.51** |
| 4 | T=14 | 822 µs | 582 µs | **+0.074** |

Both were recorded as leader scores — 3.4258x when the truth was 2.91x, and 3.0082x when it
was 2.9345x. Nothing inside the sweep can catch this. Correctness is clean, because nothing
is incorrect; the score is a mean of `baseline/ours`, so a baseline running 3x slow on one
workload is arithmetically indistinguishable from a 3x win on it.

**The light/full ratio does not reliably catch it either.** It flagged round 3's +0.51, but
round 4's +0.074 came in at 0.991 — comfortably inside tolerance. Light divides by the
`node.json` table, so only the full sweep ever sees the bad leg, and a moderate error is
invisible in the ratio. Treat the ratio as a coarse filter, not a guarantee.

**So the check is `audit_recorded_sweeps` in `swarm.py`, and it runs at the start of a
round.** It compares each recorded `lineage_*.bench.json` leg against the `node.json` table
and **writes the corrected `full_score` back into `lineages.json`**.

It corrects rather than advises because `full_score` is load-bearing: the select phase swaps
it in for the light score before ordering the slots, and a lineage whose kernel survives a
round unchanged keeps its recorded score instead of being swept again. A leg that blew out
two rounds ago would otherwise go on winning selections until someone noticed by hand.

No re-sweep is needed, and asking for one would waste 16 minutes re-deriving a number
already in hand. The substituted value is a measurement of the same quantity taken on the
same node, and where it has been possible to check, the table has been the correct one:
T=7's table read 525 µs against re-sweeps of 544 and 518, and T=14's table read 582 against
an earlier sweep's 581. The raw sweep stays in `lineage_*.bench.json` and in git.

Front-loaded deliberately. A sweep only exists at the end of a round, so checking inside it
reports *after* seven agents have been seeded from whatever the previous round crowned —
the round is already paid for. `results-sync.sh pull` puts the previous sweeps on disk and
`--baseline-ref` writes a fresh table, both before the first agent starts, so the comparison
is free and lands while it can still change what happens.

**Memory-bound legs only.** The table is measured at `LIGHT_SAMPLING` and a full sweep is
not, and T=11948 and T=14107 genuinely differ 2.4-2.6x between those regimes because they
run long enough to throttle. Checking them would fire on every sweep ever run.

Two rules follow for reading any recorded score:

- **Compare same-session light scores**, which divide both kernels by one fixed table, so
  baseline noise cancels. That is why round 4's real gain is +2.6% by light while its
  corrected full score barely moved.
- **Absolute latencies do not travel between node instances.** After a preemption this node
  measured the prefill pair ~2.2x slower for our kernel and the baseline alike. Ratios
  mostly survive; microseconds and % of speed-of-light do not.

## 4. Repo layout

```
moe-swarm/
├── README.md                    # what this is, how to run it, status
├── config.toml                  # definition + language + entry_point (flashinfer-bench)
├── solution/python/kernel.py    # THE agent write-set: exactly one file
├── solution/baseline_cutlass_sm90/main.py   # paired baseline (§3). Never edited by agents.
├── test_kernel.py               # CPU tests for the GPU-free logic (runs on the Mac)
├── bench.py                     # oracle: flashinfer-bench score + roofline verdict -> JSON
├── gates.py                     # the three anti-reward-hacking gates (§3)
├── test_gates.py                # asserts each gate trips on its cheat, and not on the real kernel
├── cheats/                      # deliberately illegal fixtures, one per gate. Never candidates.
│   ├── library_wrapper/kernel.py    # correct and fast; only gate 1 stops it
│   ├── ignores_routing/kernel.py    # stateless; only gate 2 stops it
│   └── cached_result/kernel.py      # imports nothing odd; only gate 3 stops it
├── scripts/node-setup.sh        # idempotent fresh-machine bring-up
├── scripts/results-sync.sh      # push/pull results through the swarm-results branch
├── scripts/profile_small_t.py   # oracle-faithful per-T timing, for nsys/ncu
├── scripts/nsys_breakdown.py    # per-call kernel/gap breakdown from an nsys SQLite export
├── swarm.py                     # the round loop (built LAST, after bench.py works)
├── test_swarm.py                # write-set guard, worktree cleanup, abort-on-failed-push
├── spec/                        # read-only vendored contract (§1). Never edited.
│   ├── definition.json
│   ├── reference_official.py
│   └── workloads.jsonl
├── results/                     # leaderboard, hacks, node.json and the surviving
│                                # lineages tracked; per-run snapshots gitignored
└── docs/
    ├── DESIGN.md                # this file
    ├── BACKLOG.md               # deferred, non-urgent work
    ├── PRIMER.md                # GPU vocabulary and the fused MoE pipeline
    └── KERNELS.md               # what the surviving kernels do, and how they got there
```

The kernel lives in its own directory because `flashinfer-bench` packs a *directory* into a
solution, and `language = "python"` selects `solution/python/` while v1 is pure PyTorch.
Everything else from the starter kit (`pack_solution.py`, `run_local.py`, `run_modal.py`,
`solution.json` round-tripping) is replaced by ~15 lines inside `bench.py` calling
`pack_solution_from_files` directly — the starter kit's helper hardcodes `triton` and `cuda`
as the only source dirs and would reject ours. We are not submitting, so packing exists only
to feed the local runner.

`test_kernel.py` is the one thing worth running on the Mac. Routing, the (token, expert)
compaction, block-scale dequantization, and the SwiGLU half-order are pure fp32 index math,
so they are verifiable on CPU — and they are where the subtle bugs live. The GEMM path needs
FP8 and 80 GB, so it is only verifiable on the GPU.

---

## 5. The swarm loop

One file, ~150 lines. The strongest evidence from the winning teams is that **loop
discipline dominates tooling**: MIT HAN Lab (1st, MoE) ran their own ablation and
attributed most of their gain to the plan-execute-verify harness, with their knowledge-base
and profiler skills merely "useful". So spend the effort on a tight loop and honest
feedback, not on a prompt framework.

Per round:

1. Spawn N agents, each in its own `git worktree` off one immutable base commit, each
   seeded with the current best kernel and given **one differentiated strategy hint**.
   Write-set is exactly `solution/python/kernel.py`; any other changed file rejects the
   candidate before it is benchmarked.
2. Agents iterate locally against `bench.py` — that fast inner loop is the point.
3. The orchestrator **re-verifies every candidate in a clean subprocess** from the kept
   file. Never trust a self-reported number. Candidate code never loads into the
   orchestrator's process: agents will produce illegal memory accesses and hangs, a sticky
   CUDA error poisons the context, and we have no root to reset the GPU.
4. Keep the best correct candidate; append round/agent/score/cost to
   `results/leaderboard.jsonl`.
5. **Checkpoint the agent phase the moment it ends**, before ranking. `write_checkpoint`
   puts every candidate kernel and its record in `results/checkpoint.json` and pushes it.
   The agent phase is 90 minutes and most of a round's cost, and until this existed nothing
   was durable until round end — a preemption at minute 100 destroyed all of it, three
   times in 26 hours. A later preemption now costs the ranking, ~20 minutes of GPU, and the
   next `swarm.py` resumes the round from the checkpoint rather than re-running the agents.
   Unlike the end-of-round push, a failed checkpoint push **warns and continues**: the work
   it protects has already happened, so aborting would destroy exactly what is being saved.
6. **Push `results/` — and abort the run if the push fails.** `scripts/results-sync.sh
   push` runs at the end of every round, and a non-zero exit raises `SystemExit`. This
   is a hard requirement rather than a mitigation: a preemptible machine's disk dies with
   it, we have observed preemption count 83 on one machine, so an unpushed round
   is a round the next preemption destroys. Continuing after a failed push means burning
   GPU to produce results that cannot survive, which is strictly worse than stopping.

Timing serializes behind a per-GPU file lock — concurrent timing on one GPU is garbage
(SM contention, L2 pollution, clock drift). The lock lives **inside `bench.py`**, not in
`swarm.py`, because agents iterate against `bench.py` directly in their own worktrees; a
lock the caller has to remember is a lock that gets forgotten. Only the measured section
is serialized, so packing and correctness still overlap. Queue wait has never bound a round:
`parallel=7` leaves the GPU 40-50% idle, so agent count is a diversity choice, not a
throughput one.

### What a round costs

Measured, round 4 (2026-09-20, `--parallel 7 --agent-timeout 5400`):

| phase | light runs | full sweeps | wall clock |
|---|---|---|---|
| seed ratchet, 2 lineages x 3 (spread doubles as the health gate) | 6 | — | ~9 min |
| **agent phase** (`--agent-timeout 5400`) | **38** of a ~63 capacity | — | **89 min** |
| checkpoint the candidates and push | — | — | <1 min |
| rank, one quiet batch: up to 9 kernels x 1, then top 3 x 2 more | 14 | — | 18 min |
| select: full-sweep the 3 finalists | — | 3 | 46 min (~15.5 min each) |
| keep candidates, push | — | — | <1 min |
| **total** | **58** | **3** | **2 h 45 m** |

The select range is 0-46 min: a finalist unchanged from its seed keeps its recorded score
and is not swept. A candidate byte-identical to its seed is skipped rather than ranked —
an agent that exits early leaves the seed untouched, and scoring it again only invites a
second reading of one kernel to be mistaken for two. `node-setup.sh` adds ~4 min, once per
node rather than once per round.

**The agent phase used 38 of its ~63 light-run capacity, so the GPU sat ~39% idle.** That
is the central fact about this loop and it has held for three rounds. Agents are bound by
turns, not by measurement.

**Run two rounds, not one long one.** Reseeding compounds: round 2 gained +8% because all
seven agents started from round 1's winner. A second round hands this round's winner to all
seven.

~2 h 45 m a round. Token use, round 4 across the seven agents: ~163M input read from the
prompt cache, ~2.7M written to it, ~1.5M output. The agent phase
cannot overrun: `run_agent` SIGKILLs at `--agent-timeout`.

### Where results live

| store | holds |
|---|---|
| `results/` on the GPU machine | machine-local, lives as long as the instance |
| `origin/swarm-results` | `leaderboard.jsonl`, `lineages.json`, `lineage_*.py`, `node.json`, agent transcripts and score traces, `checkpoint.json` |
| `results/candidates/` | each round's top-2 kernels and bench JSONs |

Kept kernels are named `light{score}-{agent}.py`, the metric in the name because a
filename is the one place a score travels without its bench.json.

### Inside the agent phase

A **turn** is one `assistant` event in the agent's JSONL transcript — one model response,
which may carry several tool calls. `swarm.py` counts them and appends `(N turns)` to the
leaderboard note. Measured: 112–179 turns per agent in a 90-minute phase, so **30–48 s per
turn**.

A turn is not an iteration. One genuine edit-and-measure cycle costs 8–15 turns — read the
relevant code, edit (often several `Edit` calls), compile-check, fix the Triton error,
measure, interpret — so roughly:

| | per iteration | per agent per 90 min |
|---|---|---|
| Claude time (8–15 turns at 30–48 s) | ~10 min | ~9 iterations |
| GPU time (**one** light run, 1x not 3x) | 85 s | ~9 light runs |
| measurement share of an iteration | ~15% | |

At median-of-3 an agent would get ~3 iterations in a phase instead of ~9. That is the whole
argument for leaving this one site at 1x.

The two sides do **not** saturate together. The GPU lock serialises timing node-wide, so a
90-minute phase could serve **~63 light runs in total**. Rounds 2, 3 and 4 used 30, ~38 and
38. The GPU sits 40-50% idle every time, and `parallel=7` costs nothing in throughput
because throughput is not what binds. Agent count is therefore a pure diversity-vs-depth
choice: `parallel=4` would give ~10 measurements each across 4 strategies instead of ~5
across 7.

The loop itself is `bench.py --light --prompt-view`: all 19 workloads, full correctness
check, kernel timed alone against the `node.json` table, snapshotting whenever the run beats
`best.json`. `--only T[,T...]` restricts it to named seq_lens for roughly four times the
iterations while working one region; it reports no score and never snapshots, so a subset
win has to be confirmed by a full light run before it can be kept. Subsetting scales
because the fixed cost is per-workload, not per-run: a 230s full sweep decomposes into 106s
of timed iterations, 71s of paired baseline and 58s fixed, the last being ~3s per workload.

Where an agent's tool calls actually go, measured across seven agents in round 4:

| | calls |
|---|---|
| `Bash` | 642 — of which **~474 run the agent's own scripts** and 43 run `bench.py` |
| `Edit` | 130 |
| `Write` | 52 |
| `Read` | 44 |
| `Grep` | **0** |

- **The GPU is not the constraint.** Confirmed three rounds running. Cheaper measurements
  would not buy more iterations.
- **Expect ~5 light runs per agent**, spread 2-12 (from `results/trace_*.jsonl`).
  Agents are turn-bound, not GPU-bound.
- Turn duration is 20-48 s, which is unremarkable; the cost is turn *count*.
- **More wall clock does not help. Tested and measured.** Round 5 ran at `--agent-timeout
  7200` against round 4's 5400, and came out worse on every axis: **24 measurements against
  38**, 1-5 per agent against 2-12, and **+1.0% against +2.6%** on the same-session light
  comparison. Turn counts *rose* (184-313 against 141-298), so the extra
  two hours went into thinking, editing and private scripts rather than measurement. GPU
  occupancy in the phase fell from ~61% to ~28%. Stay at 5400.
- **"Best score on the final measurement" is not evidence that more time would help.** With
  2-5 measurements and a ratchet that only keeps improvements, the last reading is the best
  one whenever there is any upward trend — it is close to tautological. The pattern held in
  round 5 (four of seven at 94-100% through the phase) while the gain more than halved. Do
  not read it as headroom.
- **Agents build their own tooling.** All seven wrote a private timing harness — a
  per-kernel profiler, a geometry sweep, standalone GEMM1 variants — taking the GPU lock
  themselves. `--only` was added for this and is being used, but only partly: 18 uses in
  round 5 against 244 runs of their own `/tmp` scripts.

**Count measurements from `results/trace_*.jsonl`, never by grepping a transcript for
`bench.py`.** The grep counts prose and `sed`/`grep` commands that merely mention the file;
it has produced counts 3x too high and an invented 60% failure rate.

Agent-time measurements are not comparable across agents — they are taken while six other
agents compile and allocate around the timed section, and each is the max over however many
iterations that agent managed. Hence the single quiet re-ranking batch after every agent has
exited.

### Weighting the strategy hints

**Rank a workload by `speedup × headroom`, not by the microseconds it could recover.** Two
independent reasons, both learned by getting it wrong:

- The score is a mean of *ratios*, so absolute gain scales with the speedup a workload
  already carries. Halving the time at T=1 (6.37x) moves the mean five times as much as
  halving it at T=901 (1.97x). The lowest-speedup workloads are the least valuable to fix.
- **The roofline is blind to launch and per-call overhead.** It counts bytes and FLOPs
  only, so a table of "recoverable time" built from it points away from whatever overhead
  remains — and overhead is where two consecutive rounds were actually won.

Round 4 tested the opposite weighting: five of seven agents on the two compute-bound
workloads, which hold 71% of the roofline-recoverable microseconds. All five underperformed.
The round went to the single small-T agent and to freestyle, both of which attacked graph
replay and launch cost.

The compute-bound pair remains 2.5-2.7x off its floor and is the largest physical gap. Five
different structural mechanisms did not dent it in one round; treat it as a multi-round
project or a deliberate CUTLASS port, not as a round to re-aim.

One always-freestyle explorer, because a full-autonomy agent has twice found what the
steered ones missed.

---

## 7. Decisions on record

| Decision | Why |
|---|---|
| flashinfer-bench is the scorer; primary metric is speedup vs the FlashInfer baseline | Externally defined, un-gameable by us, comparable to published work |
| Baseline is FlashInfer `cutlass_fused_moe` on sm_90, not the contest's `trtllm_fp8_block_scale_moe` | The contest baseline is sm_100-only and cannot run on sm_90; the substitute is still a FlashInfer kernel, and is the only backend supporting DeepSeek FP8 block scaling on sm_90. Recorded in every result |
| Roofline verdict is analytic, not `ncu`-derived | Works day one; immune to DCGM/telemetry contention; `ncu` is enrichment |
| Own harness rejected as *primary* | Would require us to define correctness and baseline ourselves — less credible |
| Triton first, CUDA second | UW SyFI (1st GDN, 3rd DSA): Triton is the quickest path to correct and to fast iteration; port after |
| Orchestrator runs **on the GPU machine**, in `tmux` | No transport layer, direct GPU, no per-call SSH latency; the laptop only pulls |
| Code and results move through git | The GPU machine pushes to a `swarm-results` branch so an unattended run never races `main`, and a fresh machine recovers every previous round with one pull |
| Achieved bandwidth cached in `results/node.json`, embedded in every result | GPU capacity is opportunistic, so rounds can land on different hardware; a score must never be separable from its denominator |
| No `REL_TOL` calibration | The contest fixes the tolerance; there is nothing to calibrate |
| No `spec/` reimplementation, no custom reference | We vendor the official reference verbatim instead |
| Three gates | Correctness already kills most cheats; only gate what it can't see |
| Selection never touches the full sweep | A reading that wins a selection is biased upward by having won it. The survivors' sweep takes no part in choosing, so it is an independent estimate by construction — which is also why it replaced the old confirmation sweep rather than joining it |
| No confirmation sweep | The record sweep already serves that purpose, at no extra cost, and vetoes nothing |
| No incumbent pre-sweep when `lineages.json` exists | It measured `solution/python/kernel.py`, which `load_lineages` then ignores in favour of the lineages: 19 minutes for a number nothing reads |
| No standalone repeatability gate | It measured the leader three times for a spread, moments before the seed ratchet measured the same kernel three more times. The gate reads the ratchet's runs |
| A survivor whose kernel is unchanged is not re-swept | Re-measuring identical bytes yields a number we hold. The recorded score is kept with `full_measured_at`, so a carried figure is never mistaken for one this session took |
| 5x200 full-sweep sampling is not reduced | It is our choice, not the contest's, but it is the operating point every recorded number sits on; shrinking it to save nine minutes would invalidate all of them |
| `solution/python/` and `language = "python"`, even for Triton kernels | flashinfer-bench's Triton builder differs from the Python one only by a metadata tag, a `fib_triton_` cache-key prefix and a `triton/` build subdir — both simply import the module. Move only if Triton and non-Triton candidates ever need separate build caches |

## 8. Prior art

| Source | What we took |
|---|---|
| [MIT HAN Lab, 1st MoE](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest) | Loop discipline > tooling (their own ablation). Intentionally small repo: prompts + a verify script, not a framework. Their `KernelWiki` and `ncu-report` Claude skills are deferred (see `BACKLOG.md`). |
| [UW SyFI, 1st GDN / 3rd DSA](https://syfi.cs.washington.edu/blog/2026-05-28-mlsys-kernel-contest/) | Triton-first-then-CUDA; full-agent beat human-steered on GDN — evidence against over-steering. |
| [KernelAgent (Meta)](https://github.com/meta-pytorch/KernelAgent) | Best-so-far tracker with revert-on-regression; stop when near speed-of-light. Skipped their AutoRouter/subgraph machinery — irrelevant for one fixed-signature kernel. |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | `fast_p` (correct **and** ≥p× speedup) as a single composite; baseline timings stored per hardware rather than re-measured. |
| [CUDA Agent](https://jianyuh.github.io/cuda/2026/03/03/cuda-agent.html) | Raw speedup as a reward over-rewards easy cases; ban reference/library fallbacks; verify on multiple random inputs. |
| [Contest starter kit](https://github.com/flashinfer-ai/flashinfer-bench) + its `EVALUATION.md`/`FAQ.md` | Official tolerances, scoring, self-containment rules. |
