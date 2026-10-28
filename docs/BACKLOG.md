# BACKLOG — deferred work

Real but deliberately not now. Promote an item when it becomes the thing standing between
us and a result.

Ordered roughly by expected value.

- **Sustained load degrades the full sweep and a clock lock cannot stop it.** Three
  back-to-back baseline sweeps at 5x200 drifted +20% at T=14107 while T=52 held to 0.5%;
  HW thermal slowdown overrides the lock and this node runs at 85-93 C. It does not appear
  to bias kernel comparison — six full sweeps back to back showed no trend. Untested:
  whether locking at 1200 MHz removes it.

- **An intermittent `baseline_latency_ms = None`.** Seen three times, never reproduced. It
  silently drops a workload, so an 18-workload score gets compared against 19-workload
  ones.

- **Two-tier measurement in the agent loop.** `snapshot_if_best` fires on a single light
  reading, so an agent's retained best is whichever of its ~9 runs read highest. A single
  run is enough to reject an obvious regression, and at 0.06-1.5% spread it is enough to see
  any gain worth having; what it cannot do is confirm a near-tie. Cheap fix: keep the single
  run for exploration, and require a median-of-3 before a snapshot is allowed to replace the
  incumbent. Most attempts fail, so the 3x cost is paid rarely. Untested: whether one run is
  in fact sufficient throughout — 10 consecutive locked light runs of one kernel (~15 min)
  would give the single-run distribution directly, instead of inferring it from triples.

- **Check the API spend limit before launching.** A round uses ~165M input and ~1.5M output tokens, and an agent that hits
  the limit mid-phase dies with a 429. `swarm.py` should check the remaining budget and
  refuse a round it cannot finish. Relatedly, `run_agent` SIGKILLs on timeout so the
  agent never emits its `result` event and `cost_usd` lands null; the sentinel path emits
  none either. The economics curve has to be rebuilt from per-turn `usage`.

---

## Resolved by design (kept for the record)

- ~~**Cross-validate our roofline layer against `flashinfer-bench`'s own timing.**~~ There is
  nothing to cross-validate: `bench.py` times nothing itself. Every workload latency comes
  from `flashinfer_bench`'s `run_all`, and the roofline layer only divides that latency by
  analytically derived bytes and FLOPs. The only timing we own is `measure_hbm_bandwidth`,
  which measures the node rather than a kernel.

- **Keep the `EXHAUSTED` sentinel.** Unused in two rounds, then four of seven agents used
  it. An agent that stops early is not necessarily right that it is done, so this is a cost
  control, not a signal.

- ~~**Bound-ness classification has no data source.**~~ Raised 2026-08-17: a strategy hint
  told the agent to classify DRAM- vs compute-bound from "the bench numbers," but nothing
  measured it — shapes lived in the trace set, not in the bench output, and `ncu` wasn't in
  the agent's allowed tools. **Resolved**: `bench.py`'s roofline layer now computes
  per-workload bytes/FLOPs analytically from known shapes and emits an explicit bound
  verdict (`DESIGN.md` §3). The classification is now a first-class output rather than
  something the agent was asked to invent.

---

## Deferred

- **Fold the split-K reduce into GEMM1 (T <= 16).** The last split CTA to arrive at a tile,
  found by a per-tile counter as in GEMM2's combine, applies SwiGLU and the fp8 quantization
  and zeroes the counter. Deleting the reduce kernel saves 3.4 / 3.2 / 2.9 µs at T=1 / 7 /
  15, +0.037 at most. Not in GEMM2: its 56 CTAs per row would each redo the SwiGLU.

- **Swap GEMM2's operands at BM=16** (W·Cᵀ: weights as wgmma's M=128, tokens as N=16).
  `mma.sync` has no fp8 form on sm_90, so today 35-43% of GEMM2's instructions are fp8→fp16
  conversions; the swap gets wgmma and was 7% faster on a GEMM2-shaped stream, ~+0.01.

- **Cross-round lesson memory.** Deliverable 4 in `DESIGN.md` §0, and the one the syhya
  retrospective names as the hard part ("retaining useful experience across iterations").
  Typed lessons (`worked` / `failed` / `invariant`), deduped, ranked recall, with `failed`
  highest-value because it stops N agents re-exploring one dead end. Tag each lesson with
  the regime it applies to — "persistent scheduling helps at T≥901 and hurts at T≤80" is
  worth more than either half. **Deferred because** HAN Lab's ablation says loop discipline
  dominates auxiliary context; prove the plain loop first, then measure what memory adds.

- **CUDA port.** Triton first per `DESIGN.md` §7. Once a Triton kernel is correct and
  scored, port the hot path. Needs `binding: "torch"` or TVM FFI in `config.toml`, and
  `-arch=sm_90a` for Hopper-specific features. Note the contest's B200 solutions use
  `sm_100a` features we can't run.

- **HAN Lab's Claude skills.** `KernelWiki` (kernel-design knowledge base, RAG-style) and
  `ncu-report-skill` (reads Nsight output) — both git submodules in their repo, installed
  into `~/.claude/skills/`. Their own ablation rates these "useful" but secondary to the
  plan-execute-verify loop, so they land after the loop works. `ncu-report` pairs naturally
  with the `ncu` enrichment item below.

- **`ncu` counter enrichment.** The roofline verdict is analytic on purpose and needs no
  profiler. Adding real counters for the top-3 candidates per round would buy the L2
  weight-reuse signal (`lts__t_sector_hit_rate`) and stall reasons, which analytic bytes
  cannot give. Note ncu's default `--clock-control base` overrides a pinned clock.

- **Economics curve panels beyond the headline.** Deliverable 2 needs only cumulative spend
  vs best score. The extra panels are nice-to-have: marginal speedup per 100M tokens (shows the
  plateau), stacked agent outcomes per round (win / no-improvement / incorrect /
  gate-failed / write-set-violation — quietly the most interesting, since usable-to-wasted
  agent-hours *is* the measure of a harness), and speedup split by regime to catch a mean
  win hiding a small-T regression.

- **`fast_p` composite metric.** KernelBench's "correct **and** ≥p× speedup" as a single
  number. The bench.flashinfer.ai leaderboard ranks by `fast_p` (the reference URL used
  `p=0.95`), so this is what a public comparison would use. Only matters if we compare
  against published solutions.

- **Multi-GPU task type.** N GPUs → N concurrent timing slots, since separate GPUs don't
  contend for SMs or L2. Per-device locks are already the design (`MOESWARM_GPU`-style env
  + one lock file per device). Worth requesting for an overnight run. Absolute latencies
  still drift across a shared power/thermal envelope, which is why measurement is paired.

- **Sanitizers.** `flashinfer_bench.agents.flashinfer_bench_run_sanitizer` wraps
  memcheck/racecheck/synccheck/initcheck. Cheap insurance once agents write real CUDA;
  pointless while we're in Triton.

- **TRT-LLM end-to-end landing.** Wire the best kernel into TRT-LLM's fused-MoE path and
  measure real tokens/sec on a small MoE model. The strongest possible ending to the
  writeup, and entirely optional to the swarm result. Note the kernel is shape-specialized
  to DeepSeek-V3 dims, so expect re-tuning for another model's shapes.

---

## Won't do (decided against, so we stop reconsidering)


- **Our own reference implementation / tolerance calibration.** The contest fixes
  `atol=1, rtol=0.3, matched_ratio=0.9` and ships a reference we vendor verbatim in
  `spec/`. There is nothing to calibrate.
- **Custom routing-distribution sweep (`uniform`/`zipf`/`single_hot`/`adversarial`).** The
  19 official workloads carry **real captured `routing_logits`**, so the routing skew is
  already realistic and fixed. Synthesizing distributions would test a benchmark we aren't
  scored on.
- **More anti-hacking gates.** flashinfer-bench's correctness check already kills
  scale-ignoring and routing-recomputing kernels. The three gates in `DESIGN.md` §3 cover
  what it can't see.

