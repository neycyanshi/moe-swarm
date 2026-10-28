#!/usr/bin/env python3
"""
swarm.py -- the round loop. Worthless unless bench.py is trustworthy: every number it
records comes from there (docs/DESIGN.md section 3).

    python swarm.py --rounds 1 --parallel 7            # one round, the normal case
    python swarm.py --rounds 1 --parallel 7 --dry-run  # verify every worktree, free

WHAT THIS FILE IS, AND MOSTLY IS NOT
------------------------------------
The strongest evidence from the winning teams is that loop discipline dominates
tooling: MIT HAN Lab ran their own ablation and attributed most of their gain to a
plan-execute-verify harness, with their knowledge-base and profiler skills merely
"useful". So this is a tight loop over an honest oracle, not a prompt framework.

Four invariants earn their place. Everything else was left out.

1. ONE IMMUTABLE BASE PER ROUND. Every agent branches from the same commit in its
   own `git worktree`, so agents cannot see or corrupt each other's edits, and a
   candidate is always a clean diff against a known parent.

2. THE WRITE-SET IS EXACTLY ONE FILE. A candidate that touched anything other than
   `solution/python/kernel.py` is rejected *before* it is benchmarked. Editing
   bench.py, gates.py or the spec is the cheapest possible way to win, so it must
   cost a candidate rather than earn one.

3. NEVER TRUST A SELF-REPORTED NUMBER. The orchestrator re-measures every candidate
   itself, in a fresh `bench.py` subprocess, from the file that was actually kept.
   Candidate code never loads into this process: agents will produce illegal memory
   accesses and hangs, a sticky CUDA error poisons the whole context, and we have no
   root to reset the GPU.

4. PUSHING RESULTS IS A HARD REQUIREMENT. Every round ends
   with `scripts/results-sync.sh push`, and a failed push ABORTS THE RUN. GPU machines
   are preemptible and their disk dies with them -- one reached preemption count 83 -- so
   an unpushed round is a round that will be silently destroyed. Continuing after a
   failed push means burning GPU to produce results that cannot survive, which is
   worse than stopping.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WRITE_SET = {"solution/python/kernel.py"}
WORKTREES = ROOT / ".worktrees"

# How often run_agent looks up from waiting: for the EXHAUSTED sentinel and the deadline.
POLL_SECONDS = 5

# How many lineages survive a round. One is what the loop used to keep, and it collapses
# every direction into whichever scored highest that round -- the first parallel round
# discarded a genuinely different kernel (1.4550x) that was better than the winner on the
# nine T=52..62 workloads the winner regresses. Two costs nothing and keeps that alive.
# How many of the ranked kernels are re-measured before the decision. One light run ranks
# all of them; a single reading can sit 1.5% off, so the top few are taken to a median of
# three. Three because slot 1 goes by rank and slot 2 by portfolio score, so the runner-up
# and the one behind it can both still change the answer.
RANK_FINALISTS = 3

# Light and the full sweep now measure the same thing: with the node.json table sampled in
# light's own regime, the two agree to 1.002 on the whole score and to within a few percent
# per workload. A finalist far off this is a kernel the two metrics disagree about, and the
# full sweep is the one that decides.
LIGHT_FULL_RATIO = 1.00

N_LINEAGES = 2

# How far a recorded sweep's baseline leg may sit from the per-node table before that leg is
# replaced by the table value. The baseline is re-measured live every full sweep and has
# twice been wrong enough to decide a round: +224% at T=7 (worth +0.51 on the recorded
# score) and +41% at T=14 (+0.074).
#
# Measured over 85 memory-bound legs across five sweeps, excluding those two: median 1.3%,
# p95 3.8%, worst 11.9%. So the good legs and the bad ones are separated by a wide gap, and
# anything in 12-40% would have the same outcome on the data we have.
#
# 10% rather than the middle of that gap, because the two errors are not symmetric. Flagging
# a good leg swaps one valid measurement of the baseline for another taken on the same node,
# and costs ~nothing; missing a bad one lets it decide slot ordering for rounds. The
# undetected error scales with the workload's speedup -- a leg just under the threshold at
# T=1 (6.37x) moves the score by speedup x threshold / 19, which is 0.067 at 20% and 0.034
# at 10%. At 10% exactly one of those 85 legs would be substituted.
BASELINE_DEVIATION = 0.10


# Weighted by where the recoverable TIME is, measured against the roofline rather than
# against the mean. On the 2.9191x leader the two compute-bound prefill workloads run at
# 37% and 40% of speed-of-light and hold 2474 us of the 3463 us recoverable across all 19
# workloads -- 71% of it. Every other workload combined holds 989 us. The mid range
# (T=32..80) is exhausted at 1.17-1.30x off its floor and is not worth an agent.
#
# Small T is score-heavy but time-poor: T=1 is 2.45x off its floor and worth +0.49 of the
# mean at the floor, yet that is only 47 us of real time, because an arithmetic mean of
# ratios divides by a small number there. One agent covers it. Chasing it harder would
# overfit the 19-workload suite rather than make the kernel better.
#
# `combine_lineages` is retired. It did its job and then some: round 3 seeded each agent
# from the opposite lineage and both re-created the other's features, so the two survivors
# converged and the portfolio slot stopped buying diversity. There is nothing left to merge.
#
# The last entry is a deliberate freestyle slot -- UW SyFI found a full-autonomy agent beat
# a human-steered one on GDN, so the loop should not be able to only ever explore what we
# already thought of.
STRATEGIES = [
    ("warp_specialize",
     "Target T=11948 and T=14107, the two compute-bound workloads. They run at 37% and 40% "
     "of speed-of-light and hold 71% of all the recoverable time in the suite -- 1035 us "
     "and 1439 us against arithmetic floors of 611 us and 946 us. Restructure the prefill "
     "GEMMs as producer/consumer warp specialization: dedicate warps to TMA loads and "
     "warps to the MMA, so the tensor cores stop waiting on global memory. Both workloads "
     "activate all 32 local experts, so there is no routing sparsity left to exploit -- "
     "this is purely about keeping the MMA pipeline fed. Do not regress the small-T path."),
    ("wgmma_pipeline",
     "Target T=11948 and T=14107 (37% and 40% of speed-of-light, 2474 us recoverable "
     "between them). Attack the MMA issue rate directly: wgmma with a deeper multi-stage "
     "TMA pipeline over K, enough stages that the tensor cores never stall on a K-block "
     "boundary. The current kernel is Triton with a 128x128 tile that already spills 40 "
     "bytes per thread; stage depth and register pressure trade against each other, so "
     "measure the spill when you change either. Do not regress the small-T path."),
    ("split_swiglu_ctas",
     "Target T=11948 and T=14107. The binding constraint on the prefill GEMMs is that "
     "128x128 is the largest tile keeping BOTH SwiGLU accumulators in registers, and it "
     "already spills. Break that constraint: split the two SwiGLU halves across separate "
     "CTAs (or separate warpgroups) so each holds one accumulator and a larger tile fits, "
     "then reconcile them in the epilogue or a short reduction. A larger tile is fewer "
     "weight re-reads and a better MMA-to-load ratio. The cost is the extra traffic or "
     "sync to rejoin the halves -- measure whether it pays. Do not regress the small-T path."),
    ("persistent_prefill",
     "Target T=11948 and T=14107. With all 32 local experts active and ~12k tokens spread "
     "unevenly across them, a grid sized per expert leaves SMs idle at the tail while other "
     "experts still have blocks to run. Use a persistent kernel or stream-K style "
     "decomposition: launch one wave of CTAs sized to the 132 SMs and have them pull tiles "
     "across expert boundaries from a device-side work queue, so load imbalance between "
     "experts stops costing a tail. Do not regress the small-T path."),
    ("cutlass_prefill",
     "Target T=11948 and T=14107 by dispatching ONLY those sizes into a CUTLASS/CuTe "
     "grouped GEMM, keeping the existing Triton path for every other T. These two hold 71% "
     "of the recoverable time in the suite and are the one region where a hand-written "
     "Triton tile is furthest from what the hardware can do -- 37% and 40% of speed-of-"
     "light. Dispatch on T at the top of kernel() and leave the small and mid-T paths "
     "untouched, so the worst case is that the new path is not taken."),
    ("small_t_parallelism",
     "Target T=1..16, which run at 41-75% of speed-of-light on the memory roofline. The "
     "cause is too little memory-level parallelism, not too little bandwidth: at T=1 "
     "routing activates 3 local experts, so GEMM1 is N=4096 K=7168 per expert and at "
     "BLOCK_N=128 that is 96 CTAs against 132 SMs -- it cannot fill one wave, and achieves "
     "1662 GB/s of the 4113 GB/s this node measures. Raise the number of concurrent weight "
     "loads in flight: deeper split-K than the current cap of 2, narrower N tiles, more "
     "pipeline stages on the weight stream. These are small absolute times (tens of "
     "microseconds), so measure carefully and do not let a win here cost anything at "
     "prefill."),
    ("freestyle",
     "No prescribed direction. Read the roofline verdict and decide for yourself where "
     "the headroom is."),
]

PROMPT = """Optimize an FP8 fused-MoE GPU kernel for a {gpu} (sm_90a, {bandwidth} GB/s measured).

The only file you may modify is solution/python/kernel.py. A candidate that changes
anything else is rejected before it is benchmarked.

Its module docstring is inherited from the kernel you were seeded with, and describes
design decisions that are still true. If you change what the kernel does, fix any sentence
it makes false -- the next agent is seeded with this file and will believe it. Do not keep
a "what is left to do" section: that goes stale the moment anyone acts on it, and the
roofline verdict below is measured fresh for you every round.

Everything you need is in this prompt, plus one file: spec/reference_official.py. The
rest of the repository is harness -- reading how scoring is implemented will not make
your kernel faster.

THE CONTRACT -- every shape fixed except seq_len = T
    routing_logits        float32        [T, 256]
    routing_bias          bfloat16       [256]
    hidden_states         float8_e4m3fn  [T, 7168]
    hidden_states_scale   float32        [56, T]           <- transposed, [H/128, T]
    gemm1_weights         float8_e4m3fn  [32, 4096, 7168]
    gemm1_weights_scale   float32        [32, 32, 56]
    gemm2_weights         float8_e4m3fn  [32, 7168, 2048]
    gemm2_weights_scale   float32        [32, 56, 16]
    local_expert_offset   int32 scalar
    routed_scaling_factor float32 scalar
    -> returns            bfloat16       [T, 7168]

256 global experts, 32 local (expert parallelism 8), hidden 7168, intermediate 2048,
FP8 block edge 128. Routing takes the top 8 experts from the best 4 of 8 groups. Only
local experts compute: expert local_expert_offset + le for le in [0, 32).

spec/reference_official.py is ground truth for semantics -- 140 lines, worth reading
once. Two of its rules produce plausible but wrong output if you get them backwards:
  - SwiGLU applies silu to the SECOND half:  C = silu(G1[:, I:]) * G1[:, :I]
  - Selection ranks on sigmoid(logits) + bias, but the combine weights come from
    sigmoid(logits) WITHOUT bias, normalized over the chosen 8 then scaled by
    routed_scaling_factor.

WHAT MAKES IT FAST
17 of the 19 scored workloads are memory-bound and carry ~89% of the score. This is a
weight-streaming problem, not a GEMM problem: at T=56 the kernel moves over a gigabyte
to perform microseconds of arithmetic. Bytes moved is the lever, and the main control
over it is how many experts you touch. The measured bandwidth above is what your
roofline is divided by.

MEASURE
    python bench.py --solution-dir solution/python --light --prompt-view

About 80s. Runs all 19 workloads and the full correctness check, timing your kernel
alone. This is your loop -- use it as often as you like. You do not need to produce a
speedup number yourself: when you stop, the harness re-scores your kernel against the
baseline in a clean subprocess and records that. Measuring the ratio yourself only
spends your budget twice.

Every passing run is snapshotted: the fastest version you measure is kept automatically
and is what gets scored, even if you are mid-edit when time runs out. So explore freely --
you cannot lose a good kernel by continuing past it, and you do not need to stop early to
protect one.

ITERATE FASTER ON ONE REGION
    python bench.py --solution-dir solution/python --light --prompt-view --only 11948,14107

Same run, restricted to the seq_lens you name, so four workloads cost roughly a quarter
of the time of 19. Use it while you are working one part of the range; it takes the same
GPU lock and its per-workload latencies are real measurements.

Two rules. It reports no score, because a mean over part of the set compares to nothing.
And it never snapshots, so a subset win is not kept until you confirm it with a full
`--light` run -- which you should do before you believe it, since a change that helps one
region often costs another.

Do not hand-write your own timing script. Every agent last round wrote one, and this flag
is that script, supported.

Correctness-gated: fast and wrong scores zero. Also rejected -- calling a kernel library
such as flashinfer or deep_gemm, producing output that ignores your inputs, or reusing a
result across calls. Reusing buffers across calls is fine; reusing results is not.

CURRENT STATE
{feedback}
{peer}
YOUR DIRECTION ({strategy})
{hint}

The current kernel scores {score:.4f}x. Make it faster across the 19 workloads with all
of them still correct.

IF THE DIRECTION IS A DEAD END, STOP. Write one line saying why to results/EXHAUSTED and
exit -- do not ship a regression, and do not keep going to fill the time. The snapshot
above is already safe, so stopping costs you nothing: whatever you measured as your best
is what gets scored either way. An hour spent past your last real improvement is an hour
the next round could have spent on a different direction.

A dead end means several consecutive attempts have failed to beat your best -- not that
you have a result you are pleased with. If your most recent measurement was your best so
far, there is more in this direction and you should keep going. Two of seven agents in the
last round stopped voluntarily with their best reading last, and the round lost that
ground."""


def sh(cmd, cwd=None, timeout=None, check=False):
    return subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True,
                          timeout=timeout, check=check)


def log(msg):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------------------

def current_feedback(bench_json):
    """The prompt-view rollup, or a neutral note on the first round."""
    if not Path(bench_json).exists():
        return "No previous measurement -- run bench.py first to see where you stand."
    out = sh([sys.executable, "bench.py", "--from-json", str(bench_json)])
    return out.stdout.strip() or "No previous measurement."


def node_facts(node_json):
    """GPU name and measured bandwidth for this node.

    Read rather than hardcoded: the same kernels run on H100 and H200 (both sm_90a), but
    their bandwidth differs by ~40%, and that number is the denominator of every roofline
    statement the agent is given. A prompt naming the wrong GPU is not cosmetic -- it
    invites the agent to reason about the wrong memory capacity and the wrong floor.
    """
    try:
        d = json.loads(Path(node_json).read_text())
        return d.get("gpu", "CUDA GPU"), d.get("achieved_hbm_bandwidth_gbs", "?")
    except Exception:
        return "CUDA GPU", "?"


def current_score(bench_json):
    try:
        return float(json.loads(Path(bench_json).read_text())["score"])
    except Exception:
        return 0.0


# --------------------------------------------------------------------------------------
# Lineages -- the N_LINEAGES best kernels, which are what the next round starts from.
#
# They live in results/ so they survive the task: results-sync.sh pushes that directory
# to the swarm-results branch every round, and node-setup.sh pulls it back. A kernel that
# only exists on node-local disk is a kernel the next preemption deletes.
# --------------------------------------------------------------------------------------

def lineage_paths(results_dir, i):
    """(kernel, bench) for lineage i, 1-based. Names are stable so a pull overwrites."""
    return results_dir / f"lineage_{i}.py", results_dir / f"lineage_{i}.bench.json"


def load_lineages(results_dir, fallback_kernel, fallback_score, fallback_bench):
    """Restore the surviving lineages, or start from the incumbent alone."""
    out = []
    meta = results_dir / "lineages.json"
    if meta.exists():
        try:
            for i, e in enumerate(json.loads(meta.read_text()), 1):
                kf, bf = lineage_paths(results_dir, i)
                if kf.exists() and kf.read_text().strip():
                    out.append({"score": float(e.get("score", 0.0)),
                                "kernel": kf.read_text(),
                                "origin": e.get("origin", "?"),
                                "full_score": e.get("full_score"),
                                "full_measured_at": e.get("full_measured_at"),
                                "bench": bf if bf.exists() else None,
                                "spd": workload_speedups(bf) if bf.exists() else None})
        except Exception:
            out = []
    if not out:
        out = [{"score": fallback_score, "kernel": fallback_kernel,
                "origin": "incumbent", "bench": fallback_bench}]
    return out[:N_LINEAGES]


def save_lineages(results_dir, lineages):
    results_dir.mkdir(parents=True, exist_ok=True)
    # Read every source before writing anything. The bench files are addressed by path,
    # and a lineage that shifts slot (a new candidate takes slot 1, the old slot 1 moves
    # to slot 2) would otherwise be copied from a path that this same loop has already
    # overwritten, giving two slots the same bench record.
    payload = []
    for ln in lineages:
        blob = None
        # The full sweep is the record, so keep that bench when there is one; the light
        # bench that did the selecting is a means, not a result.
        src = ln.get("full_bench") or ln.get("bench")
        if src and Path(src).exists():
            blob = Path(src).read_bytes()
        payload.append(blob)

    meta = []
    for i, (ln, blob) in enumerate(zip(lineages, payload), 1):
        kf, bf = lineage_paths(results_dir, i)
        kf.write_text(ln["kernel"])
        if blob is not None:
            bf.write_bytes(blob)
        ln["bench"] = bf if bf.exists() else None
        # `score` is the light median that selection ran on, `full_score` the recorded
        # full sweep. Both, named, because this project has twice lost a round to two
        # numbers on different metrics being compared as if they were the same one.
        meta.append({"file": kf.name, "score": ln["score"],
                     "full_score": ln.get("full_score"),
                     # When, because a carried-forward full score is a real measurement of
                     # this exact kernel but not one this session took.
                     "full_measured_at": ln.get("full_measured_at"),
                     "origin": ln["origin"]})
    (results_dir / "lineages.json").write_text(json.dumps(meta, indent=2) + "\n")


def peer_block(mine, peer):
    """What to tell an agent about the other lineage, and why it should care.

    Written into the prompt only when a second lineage exists and both have per-workload
    numbers. The file itself goes to results/other_lineage.py, which is outside the write
    set check (violations() skips results/), so reading it cannot cost the candidate.
    """
    if peer is None or not mine.get("spd") or not peer.get("spd"):
        return ""
    shared = sorted(set(mine["spd"]) & set(peer["spd"]), key=int)
    wins = [t for t in shared if peer["spd"][t] > mine["spd"][t] * 1.03]
    if not wins:
        return ""
    combined = portfolio_score([mine, peer])
    detail = ", ".join(f"T={t} {peer['spd'][t]:.2f}x vs your {mine['spd'][t]:.2f}x"
                       for t in wins[:8])
    return f"""
ANOTHER KERNEL, AND WHERE IT BEATS YOURS
`results/other_lineage.py` scores {peer['score']:.4f}x against your {mine['score']:.4f}x.
It is read-only reference material: you may read it freely, you may lift code from it,
and it is not part of your write set either way. It is faster than your starting kernel
on {len(wins)} of {len(shared)} workloads -- {detail}.

The score is an arithmetic mean of per-workload speedups, so it is separable: a kernel
that took the better path at every T would score {combined:.4f}x. That number is already
sitting in these two files and needs no new optimisation to collect.
"""


CHECKPOINT = "checkpoint.json"


def write_checkpoint(rnd, ran):
    """Persist the agent phase's output before anything expensive happens to it.

    The agent phase is 90 minutes and ~90% of a round's cost; ranking and the full sweeps
    that follow are another hour. Until round end nothing was written anywhere durable, so
    a preemption at minute 100 destroyed the whole round -- observed three times in 26
    hours on a project whose `preemptible: false` is not honoured.

    So the moment every agent has exited, each candidate kernel and its record go into one
    file that `results-sync.sh` pushes. A preemption after this point costs the ranking,
    which is ~20 minutes of GPU, instead of seven agents' work.

    One file, overwritten each round rather than one per round: it is ~400 KB of kernel
    source and there is no reason to accumulate it in git. `completed` is set at round end
    so a finished round is never resumed.
    """
    entries = []
    for rec, wt, status in ran:
        kernel = None
        try:
            src = wt / "solution" / "python" / "kernel.py"
            if src.exists():
                kernel = src.read_text()
        except Exception:
            pass
        entries.append({"rec": rec, "status": status, "kernel": kernel})
    path = ROOT / "results" / CHECKPOINT
    path.write_text(json.dumps({
        "round": rnd,
        "at": datetime.now(timezone.utc).isoformat(),
        "completed": False,
        "agents": entries,
    }, indent=1) + "\n")
    return path


def load_checkpoint(rnd):
    """Rebuild the agent phase's output for `rnd`, or None if there is nothing to resume.

    Returns the same `(rec, dir, status)` shape `ex.map(one, jobs)` produces. The directory
    is a plain scratch dir holding `solution/python/kernel.py`, not a git worktree: the
    write-set check that needs a real worktree already ran before the checkpoint was taken,
    and everything downstream only reads the kernel back out.
    """
    path = ROOT / "results" / CHECKPOINT
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        log(f"  WARNING: {path} is unreadable ({e}); starting the round fresh")
        return None
    if d.get("round") != rnd or d.get("completed"):
        return None
    ran = []
    for ent in d.get("agents", []):
        rec, status, kernel = ent.get("rec", {}), ent.get("status"), ent.get("kernel")
        wt = WORKTREES / f".resumed-{rec.get('agent', 'unknown')}"
        shutil.rmtree(wt, ignore_errors=True)
        (wt / "solution" / "python").mkdir(parents=True, exist_ok=True)
        if kernel is not None:
            (wt / "solution" / "python" / "kernel.py").write_text(kernel)
        ran.append((rec, wt, status))
    return ran or None


def complete_checkpoint(rnd):
    """Mark the round finished, so the next run does not resume a round that is done."""
    path = ROOT / "results" / CHECKPOINT
    if not path.exists():
        return
    try:
        d = json.loads(path.read_text())
        if d.get("round") == rnd:
            d["completed"] = True
            path.write_text(json.dumps(d, indent=1) + "\n")
    except Exception:
        pass


def audit_recorded_sweeps(node_json, results_dir):
    """Re-check the recorded full sweeps against the per-node baseline table.

    A full sweep re-measures the baseline live, and that leg is not reliable: T=7 once read
    1699 us against a table 525 and inflated a recorded score by +0.51, and T=14 read 822
    against 582 for +0.074. Nothing in the sweep itself can tell -- correctness is clean and
    the light/full ratio stays inside tolerance for the smaller case, because light divides
    by the table and only the full sweep sees the bad leg.

    Checked HERE, at the start of a round. A sweep only exists once a round has ended, so
    the earliest a bad leg can be acted on is the NEXT round -- and it has to be acted on
    before that round seeds seven agents from whatever the previous one crowned, or the
    money is spent before anyone knows. `results-sync.sh pull` puts the previous round's
    lineage_*.bench.json on disk and node.json already holds a table measured on this node,
    so the comparison is free and lands before the round commits anything. Round 3's T=7 leg
    would have been caught at round 4's launch instead of by hand two days later.

    Memory-bound legs only. The table is measured at LIGHT_SAMPLING and a full sweep is not,
    and the two compute-bound workloads genuinely differ 2.4-2.6x between those regimes
    because they run long enough to throttle. Checking them would fire on every sweep ever
    run, and a warning that always fires is a warning nobody reads.
    """
    try:
        table = json.loads(Path(node_json).read_text())["baseline_latency_ms"]
    except Exception:
        return
    found = sorted(Path(results_dir).glob("lineage_*.bench.json"))
    if not found:
        return
    log("auditing the recorded sweeps against results/node.json")
    corrections = {}
    for path in found:
        try:
            d = json.loads(path.read_text())
        except Exception as e:
            log(f"  {path.name}: unreadable ({e})")
            continue
        if d.get("light"):
            continue
        flagged, adj = [], []
        for w in d.get("workloads", []):
            ref = table.get(str(w.get("seq_len")))
            live, ours = w.get("baseline_latency_ms"), w.get("latency_ms")
            if not (ref and live and ours):
                adj = None
                break
            memory_bound = (w.get("roofline") or {}).get("bound") == "memory"
            dev = live / ref - 1.0
            if memory_bound and abs(dev) > BASELINE_DEVIATION:
                flagged.append((w["seq_len"], live * 1e3, ref * 1e3, dev))
                adj.append(ref / ours)
            else:
                adj.append(live / ours)
        recorded = d.get("score")
        if not flagged:
            log(f"  {path.name}: {recorded:.4f}x, legs agree with the table")
            continue
        log(f"  {path.name}: {recorded:.4f}x RECORDED, but "
            f"{len(flagged)} memory-bound leg(s) disagree with the table:")
        for t, live, ref, dev in sorted(flagged, key=lambda x: -abs(x[3])):
            log(f"      T={t:<6} sweep {live:8.1f} us  table {ref:8.1f} us  {dev:+.0%}")
        if adj:
            corrections[path.name.replace(".bench.json", ".py")] = sum(adj) / len(adj)

    if corrections:
        _apply_score_corrections(Path(results_dir), corrections)


def _apply_score_corrections(results_dir, corrections):
    """Write the corrected full scores back into lineages.json.

    Not advisory. `full_score` decides which kernel takes slot 1 -- the select phase swaps
    it in for the light score before ordering -- and a lineage whose kernel survives a round
    unchanged keeps its recorded score instead of being swept again. So a leg that blew out
    two rounds ago goes on winning selections until someone notices by hand. Round 3's
    3.4258x was 0.51 too high and would have done exactly that.

    Rewriting a recorded measurement deserves suspicion, so: only memory-bound legs, only
    beyond BASELINE_DEVIATION, and the replacement value is itself a measurement of the same
    quantity taken on the same node. The raw sweep stays in lineage_*.bench.json and in git.
    """
    path = results_dir / "lineages.json"
    try:
        entries = json.loads(path.read_text())
    except Exception:
        log("  WARNING: cannot read lineages.json; corrected scores not applied")
        return
    changed = False
    for e in entries:
        fixed = corrections.get(e.get("file"))
        if fixed is None or e.get("full_score") is None:
            continue
        was = e["full_score"]
        if abs(was - fixed) < 1e-9:
            continue
        e["full_score"] = fixed
        changed = True
        log(f"  corrected {e['file']} ({e.get('origin')}): full_score "
            f"{was:.4f} -> {fixed:.4f}. Selection uses this number, so it is fixed "
            f"before the round runs, not reported.")
    if changed:
        path.write_text(json.dumps(entries, indent=2) + "\n")


def archive_candidates(rnd, outcomes, node_json, keep=2):
    """Copy the round's best `keep` kernels out of their worktrees into results/candidates/.

    A worktree is scratch: it is pruned at round end and nothing in it survives. Named
    `light{score}-{agent}.py` with the score first, so `ls` sorts by quality and a glance
    says whether a file was worth anything. Kept whether or not they improved on anything: a
    round produced seven candidates that all beat a lineage, was told otherwise by a stale
    comparison, and had every worktree pruned before anyone could check. A kernel cannot be
    reconstructed from its transcript, so this is the only copy.
    """
    scored = sorted((r for r, _, k in outcomes if k and r.get("passed")
                     and not r.get("rejected") and not r.get("unchanged_from_seed")),
                    key=lambda r: -r["score"])[:keep]
    if not scored:
        return []

    d = ROOT / "results" / "candidates"
    d.mkdir(parents=True, exist_ok=True)
    by_agent = {r["agent"]: k for r, _, k in outcomes if k}
    kept = []
    for rec in scored:
        # "light" in the name because that is the metric the number is on. A filename is
        # the one place a score travels without its bench.json to explain it.
        name = f"light{rec['score']:.4f}-{rec['agent']}.py"
        f = d / f"r{rnd}-{name}"
        f.write_text(by_agent[rec["agent"]])
        kept.append(f)
        # The bench json alongside it: the score is not interpretable without per-workload
        # numbers, and re-deriving them means another 19-minute sweep.
        bj = Path(rec["bench"]) if rec.get("bench") else None
        if bj and bj.exists():
            shutil.copy(bj, d / f"r{rnd}-{name[:-3]}.bench.json")
    log(f"  kept {len(kept)} kernel(s) in results/candidates/: "
        + ", ".join(f.name for f in kept))
    return kept


def next_round_number(leaderboard):
    """One past the highest round already recorded, so rounds keep counting up.

    `--rounds 1` used to restart at 1 every invocation, which is how five distinct
    rounds all filed themselves as `round: 1` and became distinguishable only by agent
    name. The leaderboard is append-only, so it is the record of what round it is.
    """
    top = 0
    try:
        for line in Path(leaderboard).read_text().splitlines():
            if line.strip():
                top = max(top, int(json.loads(line).get("round", 0)))
    except Exception:
        pass
    return top + 1


def score_kernel(kernel_text, tag, out_json, timeout):
    """Full-sweep score for a kernel that has no worktree. None if the sweep failed."""
    d = WORKTREES / f".score_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    sol = d / "solution" / "python"
    sol.mkdir(parents=True, exist_ok=True)
    (sol / "kernel.py").write_text(kernel_text)
    subprocess.run(
        [sys.executable, "bench.py", "--solution-dir", str(sol), "--out", str(out_json),
         "--no-snapshot"],
        cwd=ROOT, capture_output=True, text=True, timeout=timeout,
    )
    try:
        return float(json.loads(Path(out_json).read_text())["score"])
    except Exception:
        return None


def seed_ratchet(lineage, slot, bench_timeout, repeats=3):
    """Measure a lineage in light mode so an agent's snapshot has to beat its own seed.

    Returns a directory holding best.json + best_kernel.py, or None.

    Without this, a worktree starts with no results/best.json, `best_so_far` is None, and
    the agent's FIRST measurement is snapshotted unconditionally -- even when it is worse
    than the kernel it was handed. Three of four agents in round 2 returned a kernel below
    their own seed that way, one of them 0.85x of it. With the ratchet primed, the worst an
    agent can hand back is what it started with.

    It has to be a LIGHT run. The seed's full-sweep score is measured at 5x200 against the
    baseline; the agent measures at 3x10 against the node's baseline table. Seeding the
    full-sweep number would put the bar in a different measurement regime from the agent's
    own runs -- light reads roughly 1.8x higher -- and the agent could never snapshot at
    all, losing snapshot-best entirely, which is worse than the problem being fixed.

    The bar is the MEDIAN of `repeats` light runs, not one. One run puts it anywhere in a
    7.4% band (eight repeats of one kernel gave score_estimate 3.3265-3.5721, CV 2.3%), and
    a high draw locks the agent out for the whole round: it then has to find a >7% gain
    before it may snapshot anything, so its real 3-5% improvements are discarded and it
    hands back the seed. Three runs cost ~4 minutes per lineage and cut the standard error
    to ~1.3%.
    """
    d = WORKTREES / f".seed{slot}"
    shutil.rmtree(d, ignore_errors=True)
    sol = d / "solution" / "python"
    sol.mkdir(parents=True, exist_ok=True)
    (sol / "kernel.py").write_text(lineage["kernel"])
    rd = d / "results"
    rd.mkdir(parents=True, exist_ok=True)

    scores, r = [], None
    for i in range(repeats):
        # A private results dir per run: writing them all into `rd` would let the first
        # run's best.json ratchet the later ones, which is the bias being removed.
        run_rd = d / f"run{i}"
        run_rd.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [sys.executable, "bench.py", "--solution-dir", str(sol), "--light",
             "--out", str(run_rd / "bench.json"), "--results-dir", str(run_rd)],
            cwd=ROOT, capture_output=True, text=True, timeout=bench_timeout,
        )
        try:
            snap = json.loads((run_rd / "bench.json").read_text()).get("snapshot") or {}
            if snap.get("key") == "score_estimate" and snap.get("score_estimate"):
                scores.append(float(snap["score_estimate"]))
        except Exception:
            pass

    if scores:
        bar = statistics.median(scores)
        (rd / "best_kernel.py").write_text(lineage["kernel"])
        (rd / "best.json").write_text(json.dumps({
            "key": "score_estimate",
            "score_estimate": bar,
            "n_workloads": None,
            "at": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")
        log(f"  ratchet for {lineage['origin']}: median {bar:.4f} of "
            + ", ".join(f"{v:.4f}" for v in scores))
        return rd, scores

    log(f"  WARNING: could not prime the ratchet for {lineage['origin']}; agents seeded "
        f"from it can return a regression. {(r.stdout or r.stderr or '')[-200:] if r else ''}")
    return None, scores


def workload_speedups(bench_path):
    """{seq_len: speedup_vs_baseline} from a bench.json, or None if unusable."""
    try:
        d = json.loads(Path(bench_path).read_text())
        s = {str(w["seq_len"]): w["speedup_vs_baseline"] for w in d["workloads"]
             if w.get("speedup_vs_baseline")}
        return s or None
    except Exception:
        return None


def light_workload_speedups(bench_path, node_json=None):
    """{seq_len: table_baseline / latency} from a LIGHT bench.json.

    A light run has no live baseline, so `speedup_vs_baseline` is null and
    `workload_speedups` returns nothing. The ranking key light mode does use is the fixed
    per-workload table in node.json, which is exactly this ratio -- the same arithmetic
    `snapshot_if_best` applies. Needed because slot 2 is chosen by portfolio score, which
    is per-workload.
    """
    try:
        node = json.loads(Path(node_json or (ROOT / "results" / "node.json")).read_text())
        table = node.get("baseline_latency_ms") or {}
        d = json.loads(Path(bench_path).read_text())
        s = {}
        for w in d["workloads"]:
            k, lat = str(w["seq_len"]), w.get("latency_ms")
            if lat and table.get(k):
                s[k] = table[k] / lat
        return s or None
    except Exception:
        return None


def light_evaluate(worktree, tag, round_, agent, timeout, repeats=3):
    """Rank a candidate on the median of `repeats` light runs. The selection metric.

    Pinned at 1400 MHz a light run repeats to 0.06-0.22% and is a scaled copy of the full
    sweep -- ratio 1.108, CV 0.27% across five kernels sharing a large-T code path -- at 85s
    against 1119s. Ranking six kernels it agreed with the full sweep on best and worst
    (Spearman +0.71); the kernels it ordered differently were within 0.62% of each other,
    which is a tie rather than a disagreement. Unpinned the same comparison gave -0.31, so
    this is only true on a clock-locked node, which `swarm.py` refuses to start without.

    Gates and correctness come free: bench.py runs the gates before timing and a light run
    checks all 19 workloads, so a cheat or a regression is caught here as it was by the
    full sweep.
    """
    sol = Path(worktree) / "solution" / "python"
    scores, passed, trips, benches = [], True, [], []
    for i in range(repeats):
        out = ROOT / "results" / f"light_{tag}_{i}.json"
        rd = WORKTREES / f".lrank_{tag}_{i}"
        shutil.rmtree(rd, ignore_errors=True)
        rd.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "bench.py", "--solution-dir", str(sol), "--light",
             "--out", str(out), "--results-dir", str(rd),
             "--round", str(round_), "--agent", agent],
            cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        )
        try:
            d = json.loads(out.read_text())
        except Exception:
            passed = False
            continue
        if not d.get("all_workloads_passed"):
            passed = False
        trips += d.get("gate_trips", []) or []
        se = (d.get("snapshot") or {}).get("score_estimate")
        if se:
            scores.append(float(se))
            benches.append(out)
    if not scores:
        return {"score": 0.0, "passed": False, "gate_trips": trips, "runs": [], "bench": None}
    med = statistics.median(scores)
    # The bench.json of the run closest to the median, so per-workload numbers and the
    # score they are quoted next to come from the same measurement.
    bench = min(zip(scores, benches), key=lambda x: abs(x[0] - med))[1]
    return {"score": med, "passed": passed, "gate_trips": trips,
            "runs": scores, "bench": bench}


def portfolio_score(members):
    """The score of an oracle that picks the best member at every workload.

    mean over T of max over members of spd[T], where spd is speedup_vs_baseline. The
    contest score is an arithmetic mean of per-workload ratios, so it is separable: what
    a kernel does at T=53 cannot affect T=1. That makes this the score a single kernel
    would reach by dispatching on seq_len between the members -- an upper bound only in
    that it ignores dispatch overhead.
    """
    sets = [m["spd"] for m in members if m.get("spd")]
    if not sets:
        return None
    keys = set().union(*(set(s) for s in sets))
    return sum(max(s.get(k, 0.0) for s in sets) for k in keys) / len(keys)


def merge_lineages(lineages, candidates):
    """The leader by score, then whoever most complements it. De-duplicated by kernel.

    Taking the top N by score is the obvious rule and it is the wrong one. Rank measures
    how good a kernel is alone; a second lineage exists to cover what the leader is bad
    at. Round 2 applied the obvious rule and evicted the only kernel that won the
    T=52..62 cluster in favour of a near-duplicate of the leader, destroying on its first
    outing the diversity the mechanism was added to protect.

    So the leader is chosen by score and everyone after it by portfolio score. Identical
    code from two agents is one lineage -- otherwise a round where everyone converges
    silently throws the slot away.
    """
    # Ascending, so that when two entries hold the same bytes the one kept is the lower
    # reading. Descending kept the higher, which is how round 1 recorded an unchanged
    # kernel at 1.9452x instead of the 1.8598x its own re-measurement gave: the max of two
    # readings of one kernel is noise, and it became the bar the next round was set.
    pool, seen = [], set()
    for ln in sorted(lineages + candidates, key=lambda x: x["score"]):
        h = hash(ln["kernel"])
        if h in seen:
            continue
        seen.add(h)
        if "spd" not in ln:
            ln["spd"] = light_workload_speedups(ln["bench"]) if ln.get("bench") else None
        pool.append(ln)
    if not pool:
        return []

    pool.sort(key=lambda x: -x["score"])
    chosen = [pool[0]]
    while len(chosen) < N_LINEAGES and len(chosen) < len(pool):
        taken = {id(c) for c in chosen}
        rest = [p for p in pool if id(p) not in taken]
        ranked = [(portfolio_score(chosen + [c]), c) for c in rest]
        if all(v is None for v, _ in ranked):
            chosen.append(rest[0])          # no per-workload data: fall back to rank
        else:
            chosen.append(max(ranked, key=lambda t: t[0] if t[0] is not None else -1.0)[1])
    return chosen




def make_worktree(base, name):
    """A worktree per agent, all off the same immutable base commit.

    Prune unconditionally first. `rm -rf .worktrees` deletes the directories but leaves
    git's registrations in .git/worktrees, and a registration whose directory is missing
    makes `git worktree add` fail outright ("missing but already registered"). Pruning
    only when the path exists misses exactly that case, which is the one a human hits.
    """
    path = WORKTREES / name
    if path.exists():
        remove_worktree(path)
    sh(["git", "worktree", "prune"])
    WORKTREES.mkdir(exist_ok=True)
    r = sh(["git", "worktree", "add", "--detach", str(path), base])
    if r.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {r.stderr.strip()}")
    return path


def remove_worktree(path):
    sh(["git", "worktree", "remove", "--force", str(path)])
    if Path(path).exists():
        shutil.rmtree(path, ignore_errors=True)
    sh(["git", "worktree", "prune"])


def violations(worktree):
    """Paths the agent changed that it was not allowed to change.

    `git status --porcelain` respects .gitignore, so scratch output under ignored paths
    does not count -- only real changes to tracked or newly added files.

    results/ is excluded because it is the harness's own output directory, not the
    agent's. bench.py writes results/bench.json there, and a tripped gate writes
    results/hacks.jsonl -- both inside the agent's worktree, because that is where it
    runs the measure command we gave it. hacks.jsonl is deliberately un-ignored (it is
    deliverable 3), so it surfaced in git status and failed a candidate for a file our
    own tooling created. That cost the first agent to finish a round: it completed
    cleanly in 134 turns and was discarded.

    The guard exists to stop a candidate editing the oracle -- bench.py, gates.py, the
    spec, the tests -- not to police measurement artifacts.
    """
    r = sh(["git", "status", "--porcelain"], cwd=worktree)
    changed = set()
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        p = line[3:].strip()
        if " -> " in p:                       # renames report "old -> new"
            p = p.split(" -> ", 1)[1]
        p = p.strip('"')
        if p.startswith("results/"):
            continue
        changed.add(p)
    return sorted(changed - WRITE_SET)


def run_agent(worktree, prompt, timeout, log_path):
    """Run one coding agent in its worktree, streaming its transcript to `log_path`.

    Returns (status, cost_usd, note) where status is "ok", "exhausted", "timeout" or
    "error".

    Streamed, not captured, and that distinction cost an hour to learn. With
    `--output-format json` the CLI emits nothing until it exits, so a working agent and
    a wedged one look identical from outside, and there is no way to tell whether a long
    run is thinking, measuring, or stuck. Worse, on TimeoutExpired subprocess.run throws
    the buffered output away, so a timed-out agent left no record of what it had done.

    `--output-format stream-json` emits one JSON event per line as it happens. Writing
    those straight to a file means progress is visible live (`tail -f`), and a killed
    agent still leaves its full transcript behind.

    An agent that has run out of ideas can stop early by writing `results/EXHAUSTED`.
    Two of four agents in the first parallel round found their only improvement inside
    the first five minutes and then spent fifty-five more producing nothing, about $18
    of a $47 round. Plateau detection was the obvious alternative and it is a trap: the
    round's *winner* went 29.7 minutes between improvements and then made its two
    largest gains, so any silence threshold short enough to save the idle agents also
    kills the best one. Asking the agent to declare it is cheaper and safer.
    """
    if shutil.which("claude") is None:
        return "error", None, "claude CLI not installed (npm i -g @anthropic-ai/claude-code)"

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    # Permission mode and model come from ~/.claude/settings.json, written by
    # scripts/node-setup.sh. One source of truth: a flag here would silently override
    # the node's configured mode.
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose"]
    sentinel = Path(worktree) / "results" / "EXHAUSTED"
    with open(log_path, "w") as out:
        proc = subprocess.Popen(cmd, cwd=worktree, stdout=out,
                                stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + timeout
        timed_out = exhausted = False
        while True:
            try:
                rc = proc.wait(timeout=POLL_SECONDS)
                note = "ok" if rc == 0 else f"agent exited {rc}"
                break
            except subprocess.TimeoutExpired:
                pass
            if sentinel.exists():
                # SIGTERM first: the CLI flushes its final `result` event on the way
                # out, and that event is the only place the run's cost is reported.
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                reason = ""
                try:
                    reason = sentinel.read_text().strip().replace("\n", " ")[:200]
                except OSError:
                    pass
                rc, exhausted = None, True
                note = f"agent stopped early: {reason or 'no reason given'}"
                break
            if time.time() >= deadline:
                proc.kill()
                proc.wait()
                rc, note, timed_out = None, f"agent exceeded {timeout}s", True
                break

    # The final `result` event carries usage. A SIGTERMed agent still emits one; a
    # SIGKILLed one does not, which is why the timeout path leaves cost null.
    cost, turns = None, 0
    try:
        for line in Path(log_path).read_text().splitlines():
            if not line.startswith("{"):
                continue
            ev = json.loads(line)
            if ev.get("type") == "assistant":
                turns += 1
            if ev.get("type") == "result":
                cost = ev.get("total_cost_usd", ev.get("cost_usd"))
    except Exception:
        pass
    if turns:
        note += f" ({turns} turns)"
    # Neither "timeout" nor "exhausted" is "failed": the worktree still holds whatever
    # the agent built, and only the oracle can say whether that is worth anything.
    status = ("exhausted" if exhausted else
              "ok" if rc == 0 else
              ("timeout" if timed_out else "error"))
    return status, cost, note


def sync_results(round_):
    """Hard requirement: a failed push aborts the run (see the module docstring, point 4)."""
    r = sh(["bash", "scripts/results-sync.sh", "push"], timeout=300)
    if r.returncode != 0:
        raise SystemExit(
            f"\nABORTING after round {round_}: results-sync push failed.\n"
            f"{(r.stderr or r.stdout).strip()}\n\n"
            "This is fatal by design. Node-local disk dies with the task, so an unpushed\n"
            "round is a round the next preemption destroys. Continuing would burn GPU\n"
            "producing results that cannot survive. Fix the push, then resume."
        )
    log(r.stdout.strip() or "results synced")


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=1,
                    help="agents per round; measurement still serializes on the GPU lock")
    ap.add_argument("--strategies", default=None,
                    help="comma-separated subset of: " + ",".join(s for s, _ in STRATEGIES))
    ap.add_argument("--agent-timeout", type=int, default=3600)
    ap.add_argument("--bench-timeout", type=int, default=3600)
    ap.add_argument("--bench-json", default="results/bench.json")
    ap.add_argument("--leaderboard", default="results/leaderboard.jsonl")
    ap.add_argument("--node-json", default="results/node.json")
    ap.add_argument("--keep-candidates", type=int, default=2,
                    help="how many of the round's best kernels to keep in "
                         "results/candidates/, regardless of whether "
                         "they improved on anything (default 2; 0 disables)")
    ap.add_argument("--keep-worktrees", action="store_true",
                    help="leave worktrees on disk for debugging (they fill the node)")
    ap.add_argument("--no-baseline-run", action="store_true",
                    help="skip the incumbent measurement even if results/bench.json is absent")
    ap.add_argument("--max-spread", type=float, default=0.05,
                    help="abort if the same kernel scores more than this fraction apart "
                         "on two consecutive sweeps (default 0.05). Costs two full sweeps "
                         "up front; 0 disables the check")
    ap.add_argument("--no-ratchet", action="store_true",
                    help="do not prime each agent's snapshot with its seed. Costs one "
                         "light run per lineage (~80s); without it an agent can hand back "
                         "a kernel worse than the one it started from")
    ap.add_argument("--no-sync", action="store_true",
                    help="skip results-sync; for local dry runs only, never on a node")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore results/checkpoint.json and re-run the agent phase even "
                         "if a preempted round left one behind")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every worktree, seed it and print the prompt, then stop "
                         "without spawning an agent. A 7-agent round is ~90 minutes and "
                         "~$110, and a round has already been lost to a worktree that was "
                         "set up wrong, so check the setup for free first")
    args = ap.parse_args()

    pool = [s for s in STRATEGIES
            if not args.strategies or s[0] in args.strategies.split(",")]
    if not pool:
        sys.exit(f"no strategies matched {args.strategies!r}")

    bench_json = ROOT / args.bench_json
    leaderboard = ROOT / args.leaderboard
    leaderboard.parent.mkdir(parents=True, exist_ok=True)

    # Measure the incumbent before spawning anyone. Without this a fresh node starts at
    # 0.0000x, so the agent is told to "improve on 0.0000x" with no feedback to work from,
    # and ANY correct kernel -- including one slower than what we already have -- is
    # recorded as an improvement. One full sweep up front removes both problems.
    # ...unless results/lineages.json already names the incumbents and their scores, which
    # is the normal case. Then this sweep measures solution/python/kernel.py, which
    # load_lineages is about to ignore in favour of the lineages -- 19 minutes for a number
    # nothing reads.
    have_lineages = (ROOT / "results" / "lineages.json").exists()
    if not bench_json.exists() and not args.no_baseline_run and not have_lineages:
        log("no previous measurement and no lineages; scoring the incumbent first (~19 min)")
        r = subprocess.run([sys.executable, "bench.py", "--out", str(bench_json)],
                           cwd=ROOT, capture_output=True, text=True, timeout=args.bench_timeout)
        if not bench_json.exists():
            sys.exit("could not establish a baseline:\n" + (r.stdout or r.stderr)[-800:])

    gpu, bandwidth = node_facts(ROOT / args.node_json)

    # Before spending a round seeding seven agents from the incumbents, check that the
    # incumbents' recorded scores are not resting on a bad baseline leg. Free -- it compares
    # the sweeps already on disk against the table already on disk. `--baseline-ref` runs the
    # same audit when it rebuilds the table, but that is once per node, so a second round on
    # the same node would never see it.
    audit_recorded_sweeps(ROOT / args.node_json, ROOT / "results")

    lineages = load_lineages(ROOT / "results",
                             (ROOT / "solution" / "python" / "kernel.py").read_text(),
                             current_score(bench_json),
                             bench_json if bench_json.exists() else None)
    best_score = lineages[0]["score"]
    first_round = next_round_number(leaderboard)
    log(f"starting from score {best_score:.4f}x, round(s) {first_round}"
        f"..{first_round + args.rounds - 1}, parallel={args.parallel}, "
        f"{len(lineages)} lineage(s): "
        + ", ".join(f"{ln['origin']} {ln['score']:.4f}x" for ln in lineages))

    # Refuse to spend a round on an unpinned clock. Achieved SM clock, not the kernels, was
    # this node's measurement noise: free, one unchanged kernel spread 115% on T14107 and
    # ranked six kernels at Spearman -0.31 against the full sweep; pinned at 1400 MHz it
    # spread 0.18% and ranked them at +0.71. A round measured unpinned optimises the clock.
    if not Path("/tmp/ks-sm-clock-mhz").exists():
        sys.exit(
            "\nABORTING: the SM clock is not pinned (no /tmp/ks-sm-clock-mhz).\n"
            "Run scripts/node-setup.sh, which pins it and records the marker, or set it by\n"
            "hand with `nvidia-smi --lock-gpu-clocks=1400,1400 && echo 1400 > "
            "/tmp/ks-sm-clock-mhz`.\nUnpinned, the same kernel varies by more than the "
            "gains this round is hunting."
        )

    for rnd in range(first_round, first_round + args.rounds):
        # Last round's scratch: light_evaluate and the record sweeps each leave a json per
        # kernel, ~30 a round, and nothing reads them once the candidates are promoted.
        # `bench_*.json` does not match `bench.json`, and transcripts, lineages, the
        # leaderboard and candidates/ are untouched.
        for f in [*(ROOT / "results").glob("light_*.json"),
                  *(ROOT / "results").glob("bench_*.json")]:
            f.unlink(missing_ok=True)

        base = sh(["git", "rev-parse", "HEAD"]).stdout.strip()
        # One feedback rollup per lineage: an agent seeded with lineage 2 must be told
        # about lineage 2's kernel, not about the leader's, or the roofline verdict it is
        # asked to act on describes a file it does not have.
        feedback = [current_feedback(ln["bench"] or bench_json) for ln in lineages]
        # Prime each lineage's ratchet once, not once per agent: the measurement is the
        # same for every agent seeded from the same kernel, and these runs now serialize
        # on the node-wide GPU lock like everything else.
        primed = [seed_ratchet(ln, i, args.bench_timeout)
                  for i, ln in enumerate(lineages, 1)] if not args.no_ratchet else \
                 [(None, [])] * len(lineages)
        ratchets = [p[0] for p in primed]

        # The health gate rides on the ratchet's own runs. It used to measure lineage[0]
        # three times for a spread, moments before the ratchet measured the same kernel
        # three more times -- the same number, paid for twice.
        gate = primed[0][1] if primed else []
        if args.max_spread > 0 and len(gate) >= 2 and min(gate) > 0:
            spread = max(gate) / min(gate) - 1.0
            log(f"  repeatability (from the ratchet): "
                + ", ".join(f"{v:.4f}" for v in gate)
                + f"  spread {100 * spread:.1f}% (limit {100 * args.max_spread:.0f}%)")
            if spread > args.max_spread:
                sys.exit(
                    f"\nABORTING: the same kernel scored {gate} -- {100 * spread:.1f}% "
                    f"apart, above the {100 * args.max_spread:.0f}% limit.\nPinned, this "
                    "reads ~0.1%, so a wide spread means the clock is not actually pinned.\n"
                    "Check /tmp/ks-sm-clock-mhz and nvidia-smi --query-gpu=clocks.sm."
                )
        log(f"round {rnd}: base {base[:8]}, {min(args.parallel, len(pool))} agent(s)")

        jobs = []
        for i in range(args.parallel):
            name, hint = pool[i % len(pool)]
            jobs.append((f"r{rnd}-a{i}-{name}", name, hint, i % len(lineages)))

        def one(job):
            agent_id, name, hint, li = job
            seed = lineages[li]
            wt = make_worktree(base, agent_id)
            # Seed from this agent's lineage, which is ahead of the base commit. Round 1
            # has one lineage and every agent gets it; from round 2 the agents are split
            # across the survivors so a direction that lost on headline score but wins
            # somewhere specific still gets developed.
            (wt / "solution" / "python" / "kernel.py").write_text(seed["kernel"])
            (wt / "results").mkdir(parents=True, exist_ok=True)
            # The worktree checks out the COMMITTED results/node.json, which is whatever
            # machine last committed one -- so an agent measuring in here reads another
            # node's bandwidth, and worse, no baseline_latency_ms table. Without the table
            # its bench.py ranks snapshots by geomean latency while the ratchet below was
            # written under the score key, the two keys are deliberately never compared,
            # and the ratchet silently becomes a no-op. That cost a whole 7-agent round:
            # every one of them returned a kernel ~0.3x below the seed it was handed.
            live_node = ROOT / args.node_json
            if live_node.exists():
                shutil.copy(live_node, wt / "results" / "node.json")
            # The other lineage, as reference material. Under results/ deliberately:
            # violations() skips that directory, so reading or even editing it cannot
            # reject the candidate, and the only file that counts is still kernel.py.
            peer = lineages[(li + 1) % len(lineages)] if len(lineages) > 1 else None
            if peer is not None:
                (wt / "results" / "other_lineage.py").write_text(peer["kernel"])
            # Prime the agent's snapshot ratchet with its own seed, so the floor on what it
            # can hand back is the kernel it was given.
            if ratchets[li]:
                for f in ("best.json", "best_kernel.py"):
                    shutil.copy(ratchets[li] / f, wt / "results" / f)
            prompt = PROMPT.format(feedback=feedback[li], strategy=name, hint=hint,
                                   score=seed["score"], gpu=gpu, bandwidth=bandwidth,
                                   peer=peer_block(seed, peer))
            t0 = time.time()
            agent_log = ROOT / "results" / f"agent_{agent_id}.jsonl"
            if args.dry_run:
                rd = wt / "results"
                nj = json.loads((rd / "node.json").read_text()) if (rd / "node.json").exists() else {}
                bj = json.loads((rd / "best.json").read_text()) if (rd / "best.json").exists() else {}
                print(f"\n{'=' * 78}\n{agent_id}   strategy={name}   "
                      f"seed={seed['origin']} {seed['score']:.4f}x\n{'=' * 78}")
                print(prompt)
                print("--- worktree results/ ---")
                for p in sorted(rd.iterdir()):
                    print(f"    {p.name:24s} {p.stat().st_size}B")
                print(f"    node.json baseline entries : {len(nj.get('baseline_latency_ms', {}))}")
                print(f"    node.json gpu / bandwidth  : {nj.get('gpu')} / "
                      f"{nj.get('achieved_hbm_bandwidth_gbs')}")
                print(f"    best.json ratchet key      : {bj.get('key')} = "
                      f"{bj.get(bj.get('key', ''), 'MISSING')}")
                print(f"    kernel.py == seed          : "
                      f"{(wt / 'solution' / 'python' / 'kernel.py').read_text() == seed['kernel']}")
                print(f"    violations()               : {violations(wt) or 'none'}")
                return ({"round": rnd, "agent": agent_id, "strategy": name, "base": base,
                         "seeded_from": seed["origin"], "seed_score": seed["score"],
                         "agent_status": "dry_run", "score": 0.0, "passed": False,
                         "rejected": "dry_run", "cost_usd": None, "agent_seconds": 0.0,
                         "note": "dry run", "agent_log": str(agent_log.relative_to(ROOT))},
                        wt, None)
            log(f"  {agent_id}: agent started from {seed['origin']} "
                f"({seed['score']:.4f}x) -- watch with: tail -f {agent_log}")
            status, cost, note = run_agent(wt, prompt, args.agent_timeout, agent_log)
            rec = {"round": rnd, "agent": agent_id, "strategy": name, "base": base,
                   "seeded_from": seed["origin"], "seed_score": seed["score"],
                   "seed_lineage": li,
                   "cost_usd": cost, "agent_seconds": round(time.time() - t0, 1),
                   "note": note, "agent_log": str(agent_log.relative_to(ROOT)),
                   "agent_status": status}

            # Only an agent that could not run at all is discarded unscored. A timeout is
            # just the clock and stopping early is just honesty: either way the worktree
            # holds real work, and ranking re-runs the gates and all 19 workloads on it, so
            # a broken kernel scores zero exactly as it would have anyway. A round once
            # threw away a working kernel purely because the clock ran out.
            if status == "error":
                rec.update(score=0.0, passed=False, rejected="agent_error")
                return rec, wt, None

            bad = violations(wt)
            if bad:
                # Rejected before benchmarking: the cheapest way to win is to edit the
                # oracle, so it must never reach the oracle.
                rec.update(score=0.0, passed=False, rejected="write_set_violation",
                           violating_paths=bad)
                return rec, wt, None

            # Prefer the best version the agent actually measured. bench.py snapshots
            # kernel.py whenever a passing run improves its ranking key -- the estimated
            # score against the per-node baseline table -- so this is the best it ever
            # had, not whatever it was mid-edit when the clock stopped, and not a later
            # version that broke what was working.
            # The score the agent itself last recorded, measured while six other agents
            # were on the node. Kept so a quiet-batch re-measurement can be compared
            # against it -- if they agree, the quiet batch is duplicated work and the
            # decide phase can drop it (docs/BACKLOG.md).
            try:
                bj = json.loads((wt / "results" / "best.json").read_text())
                if bj.get("key") == "score_estimate":
                    rec["agent_light_score"] = bj.get("score_estimate")
            except Exception:
                pass

            # The agent's score trace: one line per measurement it took, harvested out of
            # the worktree before it is pruned. This is the only record of whether an agent
            # was still improving when the timeout killed it.
            try:
                trace = wt / "results" / "score_trace.jsonl"
                if trace.exists():
                    dst = ROOT / "results" / f"trace_{agent_id}.jsonl"
                    shutil.copy(trace, dst)
                    rec["score_trace"] = f"results/{dst.name}"
                    rec["n_measurements"] = sum(1 for _ in trace.open())
            except Exception:
                pass

            best = wt / "results" / "best_kernel.py"
            live = wt / "solution" / "python" / "kernel.py"
            if best.exists() and best.read_text().strip():
                if best.read_text() != live.read_text():
                    rec["scored_snapshot"] = True
                    live.write_text(best.read_text())
                else:
                    rec["scored_snapshot"] = False
            else:
                rec["scored_snapshot"] = False
            return rec, wt, status

        # PHASE 1: every agent runs. PHASE 2: every candidate is scored, once they are all
        # gone. These were one phase, with each thread scoring its own agent as soon as it
        # finished -- so a candidate that finished early was timed against a machine with
        # six other agents still on it. The GPU lock cannot help: agents time their own
        # experiments through ad-hoc scripts that never take it. Scores measured that way
        # are not comparable with each other, let alone across rounds, and one lineage's
        # recorded latencies came out ~2x its quiet-machine numbers because of it.
        resumed = None if args.no_resume else load_checkpoint(rnd)
        if resumed is not None:
            log(f"  resuming round {rnd} from results/{CHECKPOINT}: "
                f"{len(resumed)} agent(s) recovered, skipping the agent phase")
            ran = resumed
        else:
            with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
                ran = list(ex.map(one, jobs))
            # A dry run verifies setup and must cost nothing else. Returning here keeps it
            # out of the leaderboard and off the round counter: it used to fall through and
            # append seven `score: 0.0, agent_status: dry_run` rows, which both polluted the
            # permanent record and burned the round number, so the real round that followed
            # was numbered one higher and its agents misnamed.
            if args.dry_run:
                for _, wt, _ in ran:
                    remove_worktree(wt)
                log(f"dry run: setup verified for {len(ran)} agent(s). "
                    f"Leaderboard, lineages and round number untouched; round {rnd} is "
                    f"still unused.")
                return 0
            # Durable before ranking: see write_checkpoint. A failed push here must NOT
            # abort -- unlike the end-of-round push, the work it would be protecting has
            # already happened, and stopping would throw away exactly what we are trying
            # to save.
            if not args.no_sync:
                write_checkpoint(rnd, ran)
                r = sh(["bash", "scripts/results-sync.sh", "push"], timeout=300)
                if r.returncode != 0:
                    log(f"  WARNING: checkpoint push failed; the agent phase is only on "
                        f"node-local disk and a preemption will destroy it. "
                        f"{(r.stderr or r.stdout)[-200:]}")
                else:
                    log(f"  checkpointed {len(ran)} candidate(s) before ranking")

        # Re-score the incumbents in THIS batch, next to the candidates. Measurements are
        # repeatable to ~3% CV within one scoring session and only ~7% across sessions, so
        # a lineage score carried over from an earlier round is not comparable with a
        # number measured today: round 5's leader was recorded at 1.9231x and re-measured
        # at 1.6977x, 13% lower, which is why seven candidates between 1.64x and 1.72x were
        # all recorded as regressions when none of them was distinguishable from it.
        # Two extra sweeps per round buys a comparison that means something.
        for i, ln in enumerate(lineages, 1):
            d = WORKTREES / f".lin{i}"
            shutil.rmtree(d, ignore_errors=True)
            (d / "solution" / "python").mkdir(parents=True, exist_ok=True)
            (d / "solution" / "python" / "kernel.py").write_text(ln["kernel"])
            ev = light_evaluate(d, f"lineage{i}", rnd, f"lineage{i}", args.bench_timeout,
                                repeats=1)
            if not ev["runs"]:
                log(f"  lineage {i} ({ln['origin']}): light re-score failed, keeping "
                    f"{ln['score']:.4f} from an earlier session")
                continue
            log(f"  lineage {i} ({ln['origin']}): {ln['score']:.4f} -> {ev['score']:.4f} "
                "light, re-measured in this batch ("
                + ", ".join(f"{v:.4f}" for v in ev["runs"]) + ")")
            ln["score"] = ev["score"]
            ln["bench"] = ev["bench"]
            ln["spd"] = light_workload_speedups(ev["bench"])

        outcomes = []
        for rec, wt, status in ran:
            if rec.get("rejected") or status is None:
                outcomes.append((rec, wt, None))
                continue
            # An agent that found its strategy a dead end and exited leaves the seed
            # untouched. Sweeping it again buys nothing and costs plenty: a second reading
            # of one kernel differs by up to 6.4%, and merge_lineages keeps the higher of
            # the two, so round 1 crowned an unchanged kernel "leader" at 1.9452x over its
            # own 1.8598x. Carry the score just measured for that same code instead.
            seed_ln = lineages[rec["seed_lineage"]] if "seed_lineage" in rec else None
            live_kernel = (wt / "solution" / "python" / "kernel.py")
            if (seed_ln is not None and live_kernel.exists()
                    and live_kernel.read_text() == seed_ln["kernel"]):
                log(f"  {rec['agent']}: kernel is byte-identical to its seed "
                    f"({seed_ln['origin']}); reusing {seed_ln['score']:.4f}x, no sweep")
                rec.update(score=seed_ln["score"], passed=True, gate_trips=[],
                           unchanged_from_seed=True)
                outcomes.append((rec, wt, live_kernel.read_text()))
                continue

            ev = light_evaluate(wt, rec["agent"], rnd, rec["agent"], args.bench_timeout,
                                repeats=1)
            out_json = ev["bench"]
            at = rec.get("agent_light_score")
            if ev["runs"]:
                log(f"  {rec['agent']}: light {ev['score']:.4f} ("
                    + ", ".join(f"{v:.4f}" for v in ev["runs"]) + ")"
                    + (f", agent-time read {at:.4f}" if at else ""))
            rec.update(score=ev["score"], passed=ev["passed"],
                       light_runs=ev["runs"],
                       bench=str(ev["bench"]) if ev["bench"] else None,
                       gate_trips=[t.get("gate") for t in ev["gate_trips"]])
            if ev["gate_trips"]:
                rec["rejected"] = "gate_trip"
            elif not ev["passed"]:
                rec["rejected"] = "incorrect"
            elif status == "timeout":
                rec["note"] += " -- scored anyway at timeout"
            elif status == "exhausted":
                rec["note"] += " -- scored after stopping early"
            outcomes.append((rec, wt, (wt / "solution" / "python" / "kernel.py").read_text()))

        # Pool this round's passing candidates with the incumbent lineages and keep the
        # best N_LINEAGES. A candidate can now survive without being the round's best,
        # which is the point: second place is still a direction worth developing.
        candidates = [
            {"score": rec["score"], "kernel": kernel_text, "origin": rec["agent"],
             "bench": Path(rec["bench"]) if rec.get("bench") else None}
            for rec, _, kernel_text in outcomes
            # An unchanged kernel is already in `lineages` under its own origin. Adding it
            # as a candidate too would only give merge_lineages a second entry for the same
            # bytes to relabel and re-rank, which is the trap this avoids.
            if kernel_text and rec["passed"] and not rec.get("rejected")
            and not rec.get("unchanged_from_seed")
        ]
        # Stage two: the contenders earn the precision. One run each ranks all nine, but a
        # single reading can sit 1.5% off, which is the width of a real gain -- so the top
        # few are measured twice more and re-ranked on the median of three. Spending 3x on a
        # kernel that lost by 10% buys nothing; spending it on the top of the list is the
        # whole decision.
        pool = sorted(lineages + candidates, key=lambda x: -x["score"])[:RANK_FINALISTS]
        for c in pool:
            d = WORKTREES / f".final_{abs(hash(c['origin'])) % 10**8}"
            shutil.rmtree(d, ignore_errors=True)
            (d / "solution" / "python").mkdir(parents=True, exist_ok=True)
            (d / "solution" / "python" / "kernel.py").write_text(c["kernel"])
            more = light_evaluate(d, f"final_{c['origin']}", rnd, c["origin"],
                                  args.bench_timeout, repeats=2)
            runs = [c["score"]] + more["runs"]
            if len(runs) < 2:
                continue
            c["score"] = statistics.median(runs)
            if more["bench"]:
                c["bench"] = more["bench"]
                c["spd"] = light_workload_speedups(more["bench"])
            log(f"  finalist {c['origin']}: median {c['score']:.4f} of "
                + ", ".join(f"{v:.4f}" for v in runs))

        # Carry the refined median back to the leaderboard record. `candidates` holds copies,
        # so without this a finalist would be filed under its median in lineages.json and its
        # single screen reading in the leaderboard and its promoted filename -- one kernel,
        # two numbers, in one round.
        refined = {c["origin"]: c for c in pool}
        for rec, _, _ in outcomes:
            c = refined.get(rec.get("agent"))
            if c:
                rec["score"] = c["score"]
                if c.get("bench"):
                    rec["bench"] = str(c["bench"])

        previous_leader = lineages[0]["origin"] if lineages else None
        # What each kernel already has a recorded full sweep for, so an unchanged finalist
        # is not swept again for a number we already hold.
        already = {ln["kernel"]: (ln.get("full_score"), ln.get("full_measured_at"))
                   for ln in lineages if ln.get("full_score")}

        # Every finalist gets a full sweep, and the survivors are chosen on it.
        #
        # Light screens the field, which it does well: against the full sweep its
        # per-workload speedup agrees to 0.99 at T<=32, 1.02 at T=52..80 and 0.98 at T=901.
        # It does not agree at the two compute-bound workloads -- 2.36x at T=11948 and 2.59x
        # at T=14107 -- because the node.json table is measured at 5x200, long enough for
        # those two to settle and throttle, while a light run reads them in a 3x10 burst at
        # boosted clocks. A full sweep measures numerator and denominator in one regime and
        # cancels it; light cannot. So light hands the prefill pair 19% of its mean where
        # the real metric gives it 11%, and any kernel that improves prefill looks roughly
        # 2.5x better than it is.
        #
        # Selection therefore happens on the full sweep, over the finalists only. Ordering a
        # handful of pre-screened kernels on an independent metric is not the winner's curse
        # -- that comes from taking the max of many noisy readings -- and mixing full-scored
        # finalists with light-scored also-rans would be worse than either, the scales
        # differing by ~1.1x.
        for i, c in enumerate(pool, 1):
            prior, when = already.get(c["kernel"], (None, None))
            if prior:
                c["full_score"], c["full_measured_at"] = prior, when
                log(f"  finalist {c['origin']}: kernel unchanged, keeping full "
                    f"{prior:.4f}x from {when or 'an earlier round'} -- no sweep")
                continue
            out_json = ROOT / "results" / f"bench_final{i}.json"
            full = score_kernel(c["kernel"], f"full{i}", out_json, args.bench_timeout)
            if full is None:
                log(f"  finalist {c['origin']}: full sweep failed")
                continue
            c["full_score"] = full
            c["full_bench"] = out_json
            c["full_measured_at"] = datetime.now(timezone.utc).date().isoformat()
            c["spd"] = workload_speedups(out_json) or c.get("spd")
            ratio = c["score"] / full if full else 0
            log(f"  finalist {c['origin']}: light {c['score']:.4f} -> full {full:.4f}x "
                f"(ratio {ratio:.3f})")
            if abs(ratio - LIGHT_FULL_RATIO) > 0.04:
                log(f"    NOTE: ratio is far from the validated {LIGHT_FULL_RATIO:.3f}, so "
                    "light and the full sweep disagree about this kernel. The full sweep "
                    "decides; see DESIGN.md section 3.")

        swept = [c for c in pool if c.get("full_score")]
        if swept:
            for c in pool:
                c["light_score"] = c["score"]
                c["score"] = c.get("full_score") or 0.0
            lineages = merge_lineages([], pool)
            for ln in lineages:                       # keep both numbers on the record
                ln["score"] = ln.get("light_score", ln["score"])
        else:
            log("  WARNING: no finalist produced a full sweep; selecting on light alone")
            lineages = merge_lineages(lineages, candidates)

        save_lineages(ROOT / "results", lineages)
        survivors = {ln["origin"] for ln in lineages}
        best_score = lineages[0]["score"]

        # The leader is also the repo's kernel and the default bench.json, so anything
        # reading the working tree -- a later bench.py, a human -- sees the best one.
        (ROOT / "solution" / "python" / "kernel.py").write_text(lineages[0]["kernel"])
        leader_bench = lineages[0].get("full_bench") or lineages[0].get("bench")
        if leader_bench and Path(leader_bench).exists():
            shutil.copy(leader_bench, bench_json)

        for rec, wt, kernel_text in outcomes:
            rec["kept"] = rec["agent"] in survivors
            rec["is_leader"] = rec["agent"] == lineages[0]["origin"]
            rec["best_score_after"] = best_score
            rec["timestamp"] = datetime.now(timezone.utc).isoformat()
            with leaderboard.open("a") as f:
                f.write(json.dumps(rec) + "\n")

            status = ("LEADER" if rec["is_leader"] else
                      "KEPT" if rec["kept"] else
                      rec.get("rejected", "no improvement").upper())
            log(f"  {rec['agent']}: {rec['score']:.4f}x  {status}"
                + (f"  ${rec['cost_usd']:.2f}" if rec.get("cost_usd") else ""))
            if not args.keep_worktrees:
                remove_worktree(wt)

        archive_candidates(rnd, outcomes, ROOT / args.node_json,
                           keep=args.keep_candidates)

        log(f"  lineages after round {rnd}: "
            + ", ".join(f"{ln['origin']} {ln['score']:.4f}x" for ln in lineages))

        if not args.no_sync:
            complete_checkpoint(rnd)
            sync_results(rnd)

    log(f"done. best {best_score:.4f}x -> {leaderboard}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
