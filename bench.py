#!/usr/bin/env python3
"""
bench.py -- the oracle. One score, one verdict, one JSON.

    python bench.py --calibrate     # measure achieved HBM bandwidth -> results/node.json
    python bench.py                 # score solution/python -> results/bench.json

TWO SIGNALS, ONE ORACLE (docs/DESIGN.md section 3)
--------------------------------------------------
1. flashinfer-bench is the SCORER. It owns correctness and the primary metric:
   the arithmetic mean of baseline_latency / our_latency across the 19 workloads,
   correctness-gated. We do not reimplement it and we do not grade ourselves.

2. Our roofline layer is the FEEDBACK. flashinfer-bench reports latency; it does
   not say why a kernel is slow or which regime it is in. We add, per workload,
   analytic bytes and FLOPs and a prose verdict. Analytic, not ncu-derived: every
   shape is known ahead of time, so this works on day one and cannot be blocked by
   profiler permissions or DCGM contention.

BASELINE SUBSTITUTION (stated, not silent)
------------------------------------------
The contest baseline `flashinfer_wrapper_9sdjf3` is sm_100-only and raises
"No supported CUDA architectures found for major versions [10, 12]" on sm_90.
We substitute solution/baseline_cutlass_sm90 -- still a FlashInfer kernel, and the
only backend that supports DeepSeek FP8 block scaling on sm_90. Every emitted
bench.json records `baseline.substituted = true` and why, so no reader can mistake
this score for a contest-comparable one.

The node record (GPU, driver, image, measured bandwidth) is embedded verbatim in
every bench.json. GPU capacity here is opportunistic, so successive rounds can land
on different hardware; a score must never be separable from the denominator that
produced it.
"""

import argparse
import fcntl
import json
import math
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch

import gates

ROOT = Path(__file__).resolve().parent

# harness/sitecustomize.py, which the spawn child that does the timing imports by name.
HARNESS_DIR = ROOT / "harness"

# Written by scripts/node-setup.sh when it pins the SM clock. Outside the repo because it
# describes the node, not the checkout, and every worktree must see the same value.
SM_CLOCK_MARKER = Path("/tmp/ks-sm-clock-mhz")

# One GPU per node, so the timing lock is per NODE, not per checkout. Every worktree's copy
# of bench.py must land on the same file or agents measure on top of each other.
DEFAULT_GPU_LOCK = Path("/tmp/moe-swarm-gpu.lock")
DEFINITION = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

# Shapes are fixed by the definition (docs/DESIGN.md section 1).
H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4

# Per local expert, in fp8: gemm1 (2I x H) + gemm2 (H x I).
WEIGHT_BYTES_PER_EXPERT = (2 * I * H) + (H * I)          # 44_040_192 = 44.04 MB
# Per (token, expert) assignment: 2*(H*2I) for gemm1 + 2*(I*H) for gemm2.
FLOPS_PER_ASSIGNMENT = 2 * (H * 2 * I) + 2 * (I * H)     # 88_080_384 = 88.08 MFLOP

# Achievable dense FP8 tensor-core throughput on Hopper SXM (H100 and H200 are identical
# here -- H200 changes memory, not compute). The datasheet peak is 1979 TFLOP/s; ~75% of
# that is what a good kernel actually sustains. This matches the figure used in
# docs/DESIGN.md section 2 so code and docs cannot drift apart.
ACHIEVABLE_FP8_FLOPS = 1.5e15

# Light sampling: warmup, iterations, trials. Used by `--light` and by `--baseline-ref`,
# which must match it -- the table is the denominator every light score divides by, and a
# denominator measured in a different regime does not cancel. At 5x200 the two
# compute-bound workloads run long enough to settle and throttle, so a 5x200 table against
# a 3x10 numerator read 2.4-2.6x high at T=11948 and T=14107 and left the whole light score
# 1.10x the full sweep. Matched, it is 1.002x.
LIGHT_SAMPLING = (5, 10, 3)

# Contest tolerance. Fixed by the contest -- there is nothing here to calibrate.
ATOL = 1.0
RTOL = 0.3
REQUIRED_MATCHED_RATIO = 0.9


# --------------------------------------------------------------------------------------
# Node record
# --------------------------------------------------------------------------------------

def measure_hbm_bandwidth(nbytes=2 << 30, iters=30):
    """Achieved device-to-device bandwidth from a large copy.

    A copy moves `nbytes` twice (one read, one write), which is what makes this a
    bandwidth measurement rather than a latency one. Expect 80-90% of datasheet:
    ~4.1-4.25 TB/s measured against H200's 4.8, ~3.02 against H100's 3.35. The datasheet
    number is not what the roofline verdict should be divided by.
    """
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()

    seconds = start.elapsed_time(end) / 1e3
    return (2 * nbytes * iters) / seconds  # bytes/s


def build_node_record(image_tag=None):
    props = torch.cuda.get_device_properties(0)
    driver = "unknown"
    try:
        import subprocess
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception:
        pass

    # Achieved SM clock is the measurement. Free, one unchanged kernel spread 115% on
    # T14107; pinned at 1400 MHz it spread 0.18%. Recorded so a locked score can never be
    # compared with an unlocked one by accident, the same reason `arg_clone` is recorded.
    #
    # There is no read-back for --lock-gpu-clocks: with the clock pinned at 1400,
    # clocks.max.sm and clocks.applications.graphics both still report 1980, and the
    # "Applications Clocks Setting" event reason stays Not Active. So the lock records
    # itself in a marker file and this reads that, alongside the live clock.
    sm_clock, sm_locked = None, None
    try:
        import subprocess as _sp
        sm_clock = int(_sp.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()[0])
    except Exception:
        pass
    try:
        sm_locked = int(SM_CLOCK_MARKER.read_text().strip())
    except Exception:
        pass

    return {
        "gpu": props.name,
        "sm_clock_mhz": sm_clock,
        "sm_clock_locked_mhz": sm_locked,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_memory / 1e9, 1),
        "driver_version": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "image": image_tag or os.environ.get("KS_IMAGE_TAG", "unknown"),
        "achieved_hbm_bandwidth_gbs": round(measure_hbm_bandwidth() / 1e9, 1),
        "achievable_fp8_flops": ACHIEVABLE_FP8_FLOPS,
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }


def load_node_record(path):
    if not path.exists():
        sys.exit(
            f"No node record at {path}.\n"
            "Run `python bench.py --calibrate` first -- the roofline verdict needs a\n"
            "measured bandwidth, not the datasheet figure."
        )
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------
# Workload facts and the analytic roofline
# --------------------------------------------------------------------------------------

def _routing_active_experts(logits, bias, offset, scaling):
    """Exact count of local experts the routing actually activates, and assignments.

    This is what separates a useful verdict from a vague one: at T=1 a correct
    kernel touches 1 expert and a naive one sweeps all 32, a 32x difference in
    bytes that no latency number alone will explain to an agent.
    """
    logits = logits.to(torch.float32)
    bias = bias.to(torch.float32).reshape(-1)
    T = logits.shape[0]

    s = torch.sigmoid(logits)
    s_with_bias = s + bias
    grouped = s_with_bias.view(T, N_GROUP, E_GLOBAL // N_GROUP)
    top2 = torch.topk(grouped, k=2, dim=2, largest=True, sorted=False).values
    group_scores = top2.sum(dim=2)
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    keep = group_mask.unsqueeze(2).expand(T, N_GROUP, E_GLOBAL // N_GROUP).reshape(T, E_GLOBAL)
    pruned = s_with_bias.masked_fill(keep == 0, torch.finfo(torch.float32).min)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=False).indices

    local = (topk_idx >= offset) & (topk_idx < offset + E_LOCAL)
    assignments = int(local.sum())
    active = int(torch.unique(topk_idx[local]).numel()) if assignments else 0
    return active, assignments


def expected_active_experts(T):
    """E[distinct local experts] when routing is uninformative.

    EP=8 means each token's 8 global picks land in this rank's 32-expert slice about
    once on average, not 8 times -- which is why the memory/compute crossover sits
    near T=8000 and not near T=600.
    """
    return E_LOCAL * (1.0 - (1.0 - 1.0 / E_LOCAL) ** T)


def workload_facts(ds_root, definition):
    """Map workload uuid -> {seq_len, local_expert_offset, active_experts, assignments}.

    Reads the dataset's own workload file so uuids line up with the traces, and
    resolves routing_logits from safetensors where the workload provides them so the
    expert count is exact rather than assumed.
    """
    wl_path = ds_root / "workloads" / "moe" / f"{definition}.jsonl"
    if not wl_path.exists():
        matches = list((ds_root / "workloads").rglob(f"{definition}.jsonl"))
        if not matches:
            sys.exit(f"Cannot find workloads for {definition} under {ds_root}")
        wl_path = matches[0]

    facts = {}
    for line in wl_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        wl = row["workload"]
        uuid = wl["uuid"]
        T = int(wl["axes"]["seq_len"])
        inputs = wl.get("inputs", {})

        offset = 0
        off_spec = inputs.get("local_expert_offset", {})
        if off_spec.get("type") == "scalar":
            offset = int(off_spec.get("value", 0))
        scaling = 1.0
        sc_spec = inputs.get("routed_scaling_factor", {})
        if sc_spec.get("type") == "scalar":
            scaling = float(sc_spec.get("value", 1.0))

        active, assignments, exact = None, None, False
        try:
            lg, bs = inputs.get("routing_logits", {}), inputs.get("routing_bias", {})
            if lg.get("type") == "safetensors" and bs.get("type") == "safetensors":
                from safetensors.torch import load_file
                # Paths in the workload file are relative to the dataset root
                # (they start "./blob/..."), not to the workload file's directory.
                lg_file = Path(lg["path"])
                lg_file = lg_file if lg_file.is_absolute() else (ds_root / lg["path"]).resolve()
                tensors = load_file(str(lg_file))
                bs_tensors = tensors
                if bs["path"] != lg["path"]:
                    bs_file = Path(bs["path"])
                    bs_file = bs_file if bs_file.is_absolute() else (ds_root / bs["path"]).resolve()
                    bs_tensors = load_file(str(bs_file))
                active, assignments = _routing_active_experts(
                    tensors[lg["tensor_key"]].cuda(), bs_tensors[bs["tensor_key"]].cuda(),
                    offset, scaling,
                )
                exact = True
        except Exception as e:
            print(f"  note: exact routing unavailable for T={T} ({e}); using analytic expectation",
                  file=sys.stderr)
            active, assignments, exact = None, None, False

        if active is None:
            active = expected_active_experts(T)
            assignments = T * TOP_K * E_LOCAL / E_GLOBAL

        facts[uuid] = {
            "seq_len": T,
            "local_expert_offset": offset,
            "active_local_experts": active,
            "assignments": assignments,
            "expert_count_is_exact": exact,
        }
    return facts


def roofline(fact, node):
    """Analytic floor for one workload, plus which term binds."""
    T = fact["seq_len"]
    active = fact["active_local_experts"]
    assignments = fact["assignments"]

    bw = node["achieved_hbm_bandwidth_gbs"] * 1e9
    flops_cap = node.get("achievable_fp8_flops", ACHIEVABLE_FP8_FLOPS)

    # Minimum traffic: the weights of the experts routing actually activates, read
    # once in fp8, plus fp8 activations in and bf16 output out.
    weight_bytes = active * WEIGHT_BYTES_PER_EXPERT
    act_bytes = T * H * 1
    out_bytes = T * H * 2
    total_bytes = weight_bytes + act_bytes + out_bytes

    flops = assignments * FLOPS_PER_ASSIGNMENT

    t_mem = total_bytes / bw
    t_compute = flops / flops_cap
    bound = "memory" if t_mem >= t_compute else "compute"

    return {
        "bytes_min": int(total_bytes),
        "weight_bytes": int(weight_bytes),
        "flops": int(flops),
        "t_mem_us": t_mem * 1e6,
        "t_compute_us": t_compute * 1e6,
        "bound": bound,
        "intensity_ratio": (t_mem / t_compute) if t_compute > 0 else float("inf"),
        "floor_us": max(t_mem, t_compute) * 1e6,
    }


def bottleneck_advice(bound, ratio):
    """The actionable half of a verdict: what to change, given regime and headroom.

    Shared by the per-workload verdict and the rolled-up prompt view so the two can
    never drift into giving different advice about the same measurement.
    """
    if bound == "compute":
        return ("Tile efficiency and tensor-core occupancy decide this one, not bytes.")
    if ratio >= 10.0:
        return (
            "Weights are dequantized through fp32 -- 4 bytes per element written and "
            "re-read, not 2 -- and the per-expert loop issues one launch pair per expert. "
            "In order of expected gain: apply block scales in the GEMM epilogue; keep the "
            "[n,2048] intermediate off HBM; collapse to one grouped launch."
        )
    if ratio >= 3.0:
        return (
            "Weight rematerialization dominates: fp8 in, wider dtype out, re-read. "
            "Applying block scales in the GEMM epilogue removes it."
        )
    if ratio >= 1.5:
        return (
            "Remaining gap is per-call overhead and the [n,2048] intermediate "
            "round-tripping through HBM between the two GEMMs."
        )
    return "Close to the analytic floor; further gains need a better floor, not a better kernel."


def verdict(fact, rl, latency_ms):
    """One prose sentence an agent can act on -- not a metric dump."""
    T = fact["seq_len"]
    active = fact["active_local_experts"]
    floor_us = rl["floor_us"]

    if latency_ms is None:
        return f"T={T}: no measurement (kernel did not produce a timed result)."

    measured_us = latency_ms * 1e3
    pct = 100.0 * floor_us / measured_us if measured_us > 0 else 0.0
    ratio = measured_us / floor_us if floor_us > 0 else float("inf")

    active_s = f"{active:.1f}" if isinstance(active, float) else str(active)
    head = (
        f"T={T}: {rl['bound']}-bound ({rl['intensity_ratio']:.0f}x). "
        f"Routing activates {active_s} of {E_LOCAL} local experts -> "
        f"{rl['bytes_min']/1e6:.0f} MB minimum traffic; floor {floor_us:.0f} us, "
        f"measured {measured_us:.0f} us = {pct:.0f}% of speed-of-light."
    )
    if rl["bound"] == "memory":
        head += f" Moving roughly {ratio:.0f}x the minimum bytes."
    else:
        head += (f" {rl['flops']/1e9:.0f} GFLOP against a {rl['t_compute_us']:.0f} us"
                 f" arithmetic floor.")
    return head + " " + bottleneck_advice(rl["bound"], ratio)


# --------------------------------------------------------------------------------------
# The prompt view: a rollup over bench.json, not a reduction of it
# --------------------------------------------------------------------------------------

def _verdict_from_row(w):
    """Regenerate a verdict from a stored workload row.

    The row carries everything verdict() needs, so recomputing beats reprinting
    `w["verdict"]`: a bench.json written by an older revision would otherwise render
    with stale wording or a stale ratio next to freshly computed numbers.
    """
    fact = {"seq_len": w["seq_len"], "active_local_experts": w["active_local_experts"]}
    return verdict(fact, w["roofline"], w.get("latency_ms"))


def render_prompt_view(report, worst_n=3, outlier_factor=2.0):
    """Render the agent-facing view of a bench.json.

    bench.json keeps all 19 verdicts because swarm.py, the economics curve and the
    reward-hacking appendix all read it. This is a *view* over that record, not a
    smaller version of it.

    Three rules earn their keep here:

    1. Failures are never summarized. A single failing workload zeroes the score, so
       it cannot be allowed to become a table row.
    2. Passing workloads group by (regime, headroom band). Today that collapses to
       two groups because one bug dominates everything; once it is fixed the groups
       split on their own into the small-T overhead regime and the T in [52,62]
       tile-quantization regime.
    3. The highlighted workloads are ranked by SCORE CONTRIBUTION -- lowest speedup --
       not by distance from the roofline. The score is an arithmetic mean of speedups,
       so the workloads dragging it down are the slow-relative-to-baseline ones. T=1
       is typically furthest from its roofline and among the best scoring, because the
       baseline is equally bad there; pointing an agent at it spends a round on 1/19th
       of the metric.
    """
    lines = []
    ws = report["workloads"]
    node = report.get("node", {})
    base = report.get("baseline", {})

    lines.append(
        f"SCORE {report['score']:.3f}x vs {base.get('name') or 'baseline'}"
        f"{' (substituted)' if base.get('substituted') else ''}. "
        f"{report['workloads_scored']}/{report['workloads_total']} correct."
    )
    lines.append(
        f"{node.get('gpu','?')} sm_{node.get('compute_capability','?').replace('.','')}, "
        f"{node.get('achieved_hbm_bandwidth_gbs','?')} GB/s measured."
    )

    failed = [w for w in ws if w.get("status") != "PASSED"]
    passed = [w for w in ws if w.get("status") == "PASSED" and w.get("latency_ms")]

    if failed:
        lines.append("")
        lines.append(f"CORRECTNESS FAILURES ({len(failed)}) -- the score is zero until these pass:")
        for w in failed:
            lines.append(f"  T={w['seq_len']}  {w.get('status')}")
            lines.append(f"    {w['verdict']}")
            if w.get("error"):
                for el in str(w["error"]).strip().splitlines()[-6:]:
                    lines.append(f"    | {el}")

    if not passed:
        return "\n".join(lines)

    def ratio_of(w):
        floor = w["roofline"]["floor_us"]
        return (w["latency_ms"] * 1e3 / floor) if floor > 0 else float("inf")

    # Group by regime and by the advice itself, so two headroom bands that imply the
    # same fix collapse into one paragraph instead of repeating it. Promote anything
    # far off its group's median out into OUTLIERS.
    groups = {}
    for w in passed:
        bound = w["roofline"]["bound"]
        groups.setdefault((bound, bottleneck_advice(bound, ratio_of(w))), []).append(w)

    promoted = []
    for key, members in list(groups.items()):
        if len(members) < 3:
            continue
        ratios = sorted(ratio_of(m) for m in members)
        median = ratios[len(ratios) // 2]
        keep = []
        for m in members:
            (promoted if ratio_of(m) > outlier_factor * median else keep).append(m)
        groups[key] = keep

    lines.append("")
    total = len(passed)
    by_bound = {}
    for w in passed:
        by_bound[w["roofline"]["bound"]] = by_bound.get(w["roofline"]["bound"], 0) + 1
    regime = ", ".join(
        f"{n} {b}-bound ({100.0*n/total:.0f}% of score)" for b, n in sorted(by_bound.items())
    )
    lines.append(f"REGIME  {regime}.")

    for key in sorted(groups, key=lambda k: -len(groups[k])):
        members = groups[key]
        if not members:
            continue
        bound, advice = key
        ts = sorted(m["seq_len"] for m in members)
        span = f"T={ts[0]}" if len(ts) == 1 else f"T={ts[0]}..{ts[-1]} ({len(ts)} workloads)"
        lo, hi = min(ratio_of(m) for m in members), max(ratio_of(m) for m in members)
        band = f"{lo:.0f}x off floor" if hi - lo < 1 else f"{lo:.0f}-{hi:.0f}x off floor"
        lines.append("")
        lines.append(f"{bound.upper()}-BOUND, {band}  {span}")
        lines.append(f"  {advice}")

    if promoted:
        lines.append("")
        lines.append("OUTLIERS -- far off their group, look at these separately:")
        for w in sorted(promoted, key=lambda w: -ratio_of(w)):
            lines.append(f"  {_verdict_from_row(w)}")

    scored = [w for w in passed if w.get("speedup_vs_baseline")]
    if scored:
        lines.append("")
        lines.append(f"WORST {min(worst_n, len(scored))} BY SCORE CONTRIBUTION")
        for w in sorted(scored, key=lambda w: w["speedup_vs_baseline"])[:worst_n]:
            lines.append(
                f"  T={w['seq_len']:<6} {w['speedup_vs_baseline']:.3f}x  "
                f"{w['latency_ms']:>7.2f}ms  floor {w['roofline']['floor_us']/1e3:.2f}ms  "
                f"{w['active_local_experts']}/{E_LOCAL} experts"
            )

    lines.append("")
    lines.append("ALL WORKLOADS  T / experts / ms / speedup / %SoL / bound")
    for w in ws:
        sp = f"{w['speedup_vs_baseline']:.3f}" if w.get("speedup_vs_baseline") else "-"
        ms = f"{w['latency_ms']:.2f}" if w.get("latency_ms") else "-"
        pct = f"{w['pct_of_speed_of_light']:.1f}" if w.get("pct_of_speed_of_light") else "-"
        status = "" if w.get("status") == "PASSED" else f"  <{w.get('status')}>"
        lines.append(
            f"  {w['seq_len']}\t{w['active_local_experts']}\t{ms}\t{sp}\t{pct}\t"
            f"{w['roofline']['bound']}{status}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Scoring via flashinfer-bench
# --------------------------------------------------------------------------------------

@contextmanager
def gpu_lock(lock_path=None):
    """Serialize GPU timing across processes with an advisory file lock.

    One GPU and N agents means candidates are written concurrently but must be
    *measured* one at a time. flock is the right primitive here: it is released by the
    kernel when the holder dies, so a candidate that hangs or segfaults cannot wedge
    the queue for everyone else.

    The default path is node-global, and that is the whole point. It used to be
    `ROOT / "results" / ".gpu.lock"`, and ROOT is this file's own directory -- so an agent
    running its worktree's copy of bench.py took a lock in its *worktree*, private to
    itself. Four agents therefore timed kernels simultaneously on one GPU for a whole
    round, which is exactly the "garbage, and silently so" this lock exists to prevent.
    Only the orchestrator's own scoring runs, which execute with cwd=ROOT, ever shared it.
    """
    path = Path(lock_path or os.environ.get("KS_GPU_LOCK") or DEFAULT_GPU_LOCK)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        waited = time.time()
        fcntl.flock(fh, fcntl.LOCK_EX)
        held = time.time() - waited
        if held > 1.0:
            print(f"(waited {held:.0f}s for the GPU lock)", file=sys.stderr)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def snapshot_if_best(per_workload, solution_dir, results_dir, baseline_ref=None):
    """Keep the best kernel an agent has measured, so it cannot lose it by continuing.

    An agent measured a kernel that beat the incumbent with 19/19 correct, kept editing
    for another twenty minutes, and ended with one that trips a gate. Nothing preserved
    the good version.

    RANKING KEY: the score itself -- the arithmetic mean of `baseline_ref[T] / latency[T]`
    over the 19 workloads, higher is better, read against the per-node table that
    `bench.py --baseline-ref` measures once. That is the same statistic the leaderboard
    ranks by, so the version an agent keeps is the version that would score best.

    The key used to be the geometric mean of the candidate's latencies, justified by

        mean(log ref_w) - mean(log cur_w)

    making the reference a constant. That reasoning is correct -- but only for the
    GEOMETRIC mean of speedups. The official metric is the ARITHMETIC mean, and the two
    rank differently: an arithmetic mean of ratios is dominated by the few workloads
    running at 3-6x, while a geometric mean of latencies weights all 19 equally in log
    space. One agent duly improved its geometric mean and scored *below* the kernel it
    was seeded with.

    Plain mean latency would be wrong for the same family of reasons -- T=14107 runs
    ~20 ms and T=1 ~2 ms, so it is dominated by the two prefill workloads.

    With no baseline table the geometric mean is still the best key available, so it
    stays as the fallback. The two are never compared against each other: they rank in
    opposite directions, so best.json records which key it holds.
    """
    lats = [w["latency_ms"] for w in per_workload if w.get("latency_ms")]
    if len(lats) != len(per_workload) or not lats:
        return None

    ratios = []
    if baseline_ref:
        for w in per_workload:
            ref = baseline_ref.get(str(w["seq_len"]))
            if ref and w.get("latency_ms"):
                ratios.append(ref / w["latency_ms"])

    if len(ratios) == len(per_workload):
        key_name = "score_estimate"
        key = sum(ratios) / len(ratios)
        improved = lambda new, old: new > old      # noqa: E731 -- higher is better
    else:
        key_name = "geomean_latency_ms"
        key = math.exp(sum(math.log(x) for x in lats) / len(lats))
        improved = lambda new, old: new < old      # noqa: E731 -- lower is better

    results_dir.mkdir(parents=True, exist_ok=True)
    best_json, best_kernel = results_dir / "best.json", results_dir / "best_kernel.py"
    prev = None
    if best_json.exists():
        try:
            d = json.loads(best_json.read_text())
            stored = d.get("key", "geomean_latency_ms")
            if stored == key_name:
                prev = d.get(key_name)
            else:
                # Never compare across keys -- they rank in opposite directions. But say so
                # loudly: this branch silently disables a primed ratchet, and it fires when
                # the baseline table is missing here and was present when best.json was
                # written, which is a misconfiguration rather than a normal state.
                print(f"WARNING: {best_json} was written under '{stored}' but this run "
                      f"ranks by '{key_name}'; ignoring the previous best. If a ratchet "
                      f"was seeded here it is now inert -- check results/node.json has "
                      f"baseline_latency_ms.", file=sys.stderr)
        except Exception:
            prev = None
    if prev is not None and not improved(key, prev):
        _append_score_trace(results_dir, key_name, key, kept=False)
        return {"key": key_name, key_name: key, "kept": False, "best_so_far": prev}

    src = Path(solution_dir) / "kernel.py"
    if src.exists():
        best_kernel.write_text(src.read_text())
    best_json.write_text(json.dumps({
        "key": key_name,
        key_name: key,
        "n_workloads": len(lats),
        "at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    _append_score_trace(results_dir, key_name, key, kept=True)
    return {"key": key_name, key_name: key, "kept": True, "best_so_far": prev}


def _append_score_trace(results_dir, key_name, key, kept):
    """Append one line per measurement, so a round can be read as a trajectory.

    The leaderboard records only an agent's final score, which cannot answer whether it was
    still improving when `--agent-timeout` killed it -- the question that decides whether
    more wall clock is worth buying. One line per `bench.py` run does answer it.

    Best-effort: a trace that fails to write must never fail a measurement.
    """
    try:
        with (Path(results_dir) / "score_trace.jsonl").open("a") as fh:
            fh.write(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "key": key_name,
                key_name: key,
                "kept": kept,
            }) + "\n")
    except Exception:
        pass



def pack(source_dir, name, entry_point, definition, author="moe-swarm"):
    from flashinfer_bench import BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files

    spec = BuildSpec(
        language="python",
        target_hardware=["cuda"],
        entry_point=entry_point,
        destination_passing_style=False,
    )
    return pack_solution_from_files(
        path=str(source_dir), spec=spec, name=name, definition=definition, author=author
    )


def _disable_arg_cloning(marker):
    """Take the oracle's per-iteration argument clone out of the timing loop.

    See harness/sitecustomize.py for what the clone costs and why the gates make it
    redundant here. Both halves are needed: the environment reaches the spawn child that
    actually holds the stopwatch, and the in-process patch covers a non-isolated run.
    """
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if str(HARNESS_DIR) not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(HARNESS_DIR)] + parts)
    os.environ["KS_NO_ARG_CLONE"] = "1"
    os.environ["KS_NO_ARG_CLONE_MARKER"] = str(marker)
    try:
        import flashinfer_bench.bench.timing as timing

        timing._clone_args = lambda args: args
    except Exception as exc:
        sys.exit(f"cannot disable the per-iteration argument clone: {exc}")


def run_benchmark(solutions, ds_root, definition, iterations, warmup, trials, timeout,
                  lock_path=None, clone_args=False, only=None):
    """Time every solution over every workload, holding the per-GPU lock.

    Concurrent timing on one GPU is garbage -- SM contention, L2 pollution and clock
    drift make the numbers unusable, and silently so. The lock lives here rather than
    in swarm.py because agents iterate against bench.py directly in their own
    worktrees; a lock the caller has to remember is a lock that gets forgotten.
    Correctness and packing stay outside it, so only the measured section serializes.

    `only` restricts the run to a set of seq_lens. See --only.
    """
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

    marker = None
    if not clone_args:
        marker = Path(tempfile.mkdtemp(prefix="ks-noclone-")) / "patched"
        _disable_arg_cloning(marker)

    trace_set = TraceSet.from_path(str(ds_root))
    if definition not in trace_set.definitions:
        sys.exit(f"Definition {definition} not in trace set at {ds_root}")

    d = trace_set.definitions[definition]
    workloads = trace_set.workloads.get(definition, [])
    if not workloads:
        sys.exit(f"No workloads for {definition}")

    if only:
        facts = workload_facts(Path(ds_root), definition)
        keep = {u for u, f in facts.items() if f["seq_len"] in only}
        workloads = [w for w in workloads if w.workload.uuid in keep]
        if not workloads:
            sys.exit(f"--only {sorted(only)} matched no workload. Available: "
                     + ",".join(str(f["seq_len"]) for f in
                                sorted(facts.values(), key=lambda f: f["seq_len"])))

    bench_set = TraceSet(
        root=trace_set.root,
        definitions={d.name: d},
        solutions={d.name: list(solutions)},
        workloads={d.name: workloads},
        traces={d.name: []},
    )
    cfg = BenchmarkConfig(
        warmup_runs=warmup,
        iterations=iterations,
        num_trials=trials,
        atol=ATOL,
        rtol=RTOL,
        required_matched_ratio=REQUIRED_MATCHED_RATIO,
        use_isolated_runner=True,
        timeout_seconds=timeout,
    )
    with gpu_lock(lock_path):
        result = Benchmark(bench_set, cfg).run_all(dump_traces=False)

    # Prove the metric reported is the metric measured. The stopwatch is in a spawn child,
    # so an unset PYTHONPATH or a package layout change would silently give back
    # clone-perturbed numbers under a clean label -- the one confusion this whole
    # measurement effort exists to prevent.
    if marker is not None and not (marker.exists() and marker.read_text().strip()):
        sys.exit("the argument clone was meant to be off, but no timing process "
                 f"reported patching it (marker {marker} empty). Refusing to report.")

    out = {}
    for trace in result.traces.get(definition, []):
        ev = trace.evaluation
        if ev is None:
            continue
        entry = {"status": ev.status.value, "latency_ms": None, "reference_latency_ms": None}
        if ev.performance:
            entry["latency_ms"] = ev.performance.latency_ms
            entry["reference_latency_ms"] = ev.performance.reference_latency_ms
        if ev.correctness:
            entry["max_abs_error"] = ev.correctness.max_absolute_error
            entry["max_rel_error"] = ev.correctness.max_relative_error
        if ev.status.value != "PASSED":
            entry["log"] = (getattr(ev, "log", "") or "")[-1200:]
        out.setdefault(trace.solution, {})[trace.workload.uuid] = entry
    return out


# --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="measure HBM bandwidth -> results/node.json")
    ap.add_argument("--baseline-ref", action="store_true",
                    help="time the baseline alone over every workload at light sampling "
                         "and store the per-workload latencies in results/node.json. This "
                         "is the denominator every light score divides by, so it is "
                         "measured in light's own regime; an agent's inner loop then ranks "
                         "by score without paying for a baseline run")
    ap.add_argument("--solution-dir", default="solution/python")
    ap.add_argument("--entry-point", default="kernel.py::kernel")
    ap.add_argument("--baseline-dir", default="solution/baseline_cutlass_sm90")
    ap.add_argument("--baseline-entry", default="main.py::run")
    ap.add_argument("--no-baseline", action="store_true", help="skip the paired baseline (correctness only)")
    ap.add_argument("--out", default="results/bench.json")
    ap.add_argument("--node-json", default="results/node.json")
    ap.add_argument("--prompt-view", action="store_true",
                    help="print the rolled-up agent-facing view instead of all 19 verdicts")
    ap.add_argument("--from-json", metavar="PATH",
                    help="render --prompt-view from an existing bench.json without re-running")
    ap.add_argument("--image-tag", default=None)
    ap.add_argument("--no-gates", action="store_true",
                    help="skip the anti-reward-hacking gates (debugging only)")
    ap.add_argument("--hacks", default="results/hacks.jsonl")
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--agent", default=None)
    # 5x200 rather than 3x50, measured: three sweeps of one unchanged kernel spread 44.5%
    # at 3x50 and 17.8% at 5x200. Still not tight enough to resolve a 10% gain, but it is
    # the difference between unusable and marginal. Costs ~21 min per sweep against ~4.
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--faithful-clone", action="store_true",
                    help="keep the oracle's per-iteration argument clone. It re-copies "
                         "~1.4 GB of weights before every timed iteration, which is a "
                         "roughly fixed perturbation and so dominates the small-T "
                         "workloads: one unchanged kernel then repeats to 61%% rather "
                         "than 15%%. Scores measured with it are NOT comparable to "
                         "scores measured without it -- see the arg_clone field")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--light", action="store_true",
                    help="agent inner loop: all 19 workloads and the correctness check, but "
                         "candidate only and 3x10 sampling instead of 3x50. Directional, "
                         "not a score -- swarm.py always re-verifies with the full sweep")
    ap.add_argument("--only", metavar="T[,T...]",
                    help="restrict the run to these seq_lens, e.g. --only 11948,14107. For "
                         "the inner loop while working one region: four workloads instead "
                         "of 19 is roughly four times the iterations per hour. Implies "
                         "--no-snapshot and never writes a score -- a subset mean is not "
                         "comparable with anything, and must not be allowed to ratchet "
                         "best.json. Confirm with a full `--light` run before believing a "
                         "win. Round 4's agents each hand-wrote this as a /tmp script; it "
                         "is supported now so they do not have to")
    ap.add_argument("--no-snapshot", action="store_true",
                    help="do not keep results/best_kernel.py; set by swarm.py when it "
                         "re-verifies, so a candidate's own best cannot be overwritten")
    ap.add_argument("--gpu-lock", default=None,
                    help=f"path to the per-NODE timing lock (default {DEFAULT_GPU_LOCK}, "
                         "or $KS_GPU_LOCK). Node-global on purpose: a repo-relative "
                         "default gives every worktree its own lock and lets agents time "
                         "on top of each other")
    ap.add_argument("--results-dir", default=None,
                    help="where snapshot_if_best keeps best.json and best_kernel.py "
                         "(default results/). swarm.py points this at a worktree to seed "
                         "the agent's ratchet with its own starting kernel")
    args = ap.parse_args()

    # A subset run is a stopwatch, not a score. It must not snapshot (best.json is the
    # ratchet that decides what an agent hands back) and must not report a number anyone
    # could mistake for one, because a mean over 4 of 19 workloads compares to nothing.
    only = None
    if args.only:
        try:
            only = {int(x) for x in args.only.replace(",", " ").split()}
        except ValueError:
            sys.exit(f"--only wants seq_lens, got {args.only!r}")
        args.no_snapshot = True

    # Rendering a stored result needs no GPU, so this check comes first: swarm.py and
    # anything on the Mac can re-render a prompt from a pulled bench.json.
    if args.from_json:
        print(render_prompt_view(json.loads(Path(args.from_json).read_text())))
        return 0

    if not torch.cuda.is_available():
        sys.exit("No CUDA device. bench.py needs an NVIDIA Hopper GPU (see scripts/node-setup.sh).")

    node_path = ROOT / args.node_json
    node_path.parent.mkdir(parents=True, exist_ok=True)

    if args.calibrate:
        record = build_node_record(args.image_tag)
        node_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        print(f"\nwrote {node_path}")
        return 0

    node = load_node_record(node_path)

    ds_root = os.environ.get("FIB_DATASET_PATH")
    if not ds_root:
        sys.exit("FIB_DATASET_PATH is not set (run scripts/node-setup.sh, then source ~/ks-env.sh).")
    ds_root = Path(ds_root)

    if args.baseline_ref:
        # snapshot_if_best ranks by the score itself, which needs a per-workload baseline.
        # An agent cannot re-time the baseline every inner loop -- that is exactly the
        # doubling the light run exists to avoid -- so it is measured once per node, at
        # LIGHT_SAMPLING, because it is divided into light-sampled numerators.
        name = "baseline-cutlass-sm90"
        sols = [pack(ROOT / args.baseline_dir, name, args.baseline_entry, DEFINITION)]
        print(f"Timing {name} alone over the {DEFINITION} workloads ...", flush=True)
        bw, bi, bt = LIGHT_SAMPLING
        res = run_benchmark(sols, ds_root, DEFINITION, bi, bw, bt, args.timeout,
                            lock_path=args.gpu_lock, clone_args=args.faithful_clone)
        got = res.get(name, {})
        facts = workload_facts(ds_root, DEFINITION)
        table, missing = {}, []
        for uuid, fact in sorted(facts.items(), key=lambda kv: kv[1]["seq_len"]):
            e = got.get(uuid, {})
            if e.get("status") == "PASSED" and e.get("latency_ms"):
                table[str(fact["seq_len"])] = e["latency_ms"]
            else:
                missing.append((fact["seq_len"], e.get("status")))
        record = load_node_record(node_path)
        record["baseline_latency_ms"] = table
        record["baseline_ref_at"] = datetime.now(timezone.utc).isoformat()
        node_path.write_text(json.dumps(record, indent=2) + "\n")
        for t, lat in sorted(table.items(), key=lambda kv: int(kv[0])):
            print(f"  T={t:>6}  {lat * 1000:8.0f} us")
        if missing:
            # A workload with no baseline cannot contribute a ratio, and a partial table
            # would silently rank against a different set than the score does.
            print(f"\nWARNING: no baseline for {missing}. snapshot_if_best needs the "
                  "whole set and will fall back to the geometric mean until this is "
                  "complete -- rerun; the baseline fails intermittently at T=1.")
        print(f"\nwrote {len(table)}/{len(facts)} baseline latencies to {node_path}")
        return 0

    # Gates run before any timing, on the candidate only -- never on the baseline, which
    # is a library wrapper by design. A tripped gate cannot be allowed to post a score,
    # so this short-circuits: the whole point is that a cheat scores zero, not that it
    # scores well with a footnote.
    gate_trips = []
    if not args.no_gates:
        gate_trips, gates_skipped = gates.run_gates(
            ROOT / args.solution_dir, args.entry_point, timeout=args.timeout
        )
        if gate_trips:
            gates.log_trips(gate_trips, ROOT / args.hacks, args.round, args.agent,
                            args.solution_dir)
            report = {
                "definition": DEFINITION,
                "score": 0.0,
                "score_metric": "zeroed: a gate tripped before benchmarking",
                "all_workloads_passed": False,
                "workloads_scored": 0,
                "workloads_total": 0,
                "gate_trips": gate_trips,
                "node": node,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "workloads": [],
            }
            out_path = ROOT / args.out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"REJECTED before benchmarking -- {len(gate_trips)} gate trip(s):")
            for t in gate_trips:
                loc = f" [{t['file']}:{t['line']}]" if t.get("file") else ""
                print(f"  {t['gate']}{loc}: {t['detail']}")
            print(f"\nlogged to {ROOT / args.hacks}; wrote {out_path}")
            return 1
        if gates_skipped:
            print("warning: dynamic gates were skipped", file=sys.stderr)

    # Light mode trades sampling for turnaround, and nothing else. All 19 workloads still
    # run and correctness is still checked, so a fast-and-wrong kernel is caught exactly as
    # before; only the paired baseline and 80% of the timed calls go away. Measured on
    # H200: 250s -> ~78s. Three trials are kept deliberately -- a single trial has no
    # outlier rejection, and an agent misled by one noisy batch wastes a whole round.
    if args.light:
        args.no_baseline = True
        args.warmup, args.iterations, args.trials = LIGHT_SAMPLING

    solutions = [pack(ROOT / args.solution_dir, "moe-swarm-candidate", args.entry_point, DEFINITION)]
    baseline_name = None
    if not args.no_baseline:
        baseline_name = "baseline-cutlass-sm90"
        solutions.append(pack(ROOT / args.baseline_dir, baseline_name, args.baseline_entry, DEFINITION))

    print(f"Running {len(solutions)} solution(s) over the {DEFINITION} workloads ...", flush=True)
    t0 = time.time()
    results = run_benchmark(solutions, ds_root, DEFINITION, args.iterations, args.warmup,
                            args.trials, args.timeout, lock_path=args.gpu_lock,
                            clone_args=args.faithful_clone, only=only)
    elapsed = time.time() - t0

    ours = results.get("moe-swarm-candidate", {})
    base = results.get(baseline_name, {}) if baseline_name else {}
    facts = workload_facts(ds_root, DEFINITION)
    if only:
        facts = {u: f for u, f in facts.items() if f["seq_len"] in only}

    per_workload = []
    speedups = []
    all_passed = bool(ours)
    for uuid, fact in sorted(facts.items(), key=lambda kv: kv[1]["seq_len"]):
        o = ours.get(uuid, {"status": "MISSING", "latency_ms": None})
        b = base.get(uuid, {})
        rl = roofline(fact, node)

        passed = o.get("status") == "PASSED"
        all_passed = all_passed and passed
        speedup = None
        if passed and o.get("latency_ms") and b.get("latency_ms"):
            speedup = b["latency_ms"] / o["latency_ms"]
            speedups.append(speedup)

        per_workload.append({
            "uuid": uuid,
            "seq_len": fact["seq_len"],
            "local_expert_offset": fact["local_expert_offset"],
            "active_local_experts": fact["active_local_experts"],
            "status": o.get("status"),
            "latency_ms": o.get("latency_ms"),
            "baseline_latency_ms": b.get("latency_ms"),
            "baseline_status": b.get("status"),
            "reference_latency_ms": o.get("reference_latency_ms"),
            "speedup_vs_baseline": speedup,
            "roofline": rl,
            "pct_of_speed_of_light": (
                100.0 * rl["floor_us"] / (o["latency_ms"] * 1e3) if o.get("latency_ms") else None
            ),
            "verdict": verdict(fact, rl, o.get("latency_ms")),
            "error": o.get("log"),
        })

    # Correctness-gated: a single failing workload zeroes the score, as the contest does.
    score = (sum(speedups) / len(speedups)) if (speedups and all_passed) else 0.0
    if only:
        score = 0.0

    report = {
        "definition": DEFINITION,
        "score": score,
        "score_metric": (f"SUBSET RUN over seq_lens {sorted(only)} -- NOT a score and not "
                         "comparable with any other number. Per-workload latencies below "
                         "are real; confirm a win with a full --light run."
                         if only else
                         "LIGHT RUN -- directional only, not a score: candidate timed alone "
                         "at 3x10, so there is no baseline to form a ratio against"
                         if args.light else
                         "arithmetic mean of baseline_latency / our_latency over all workloads, correctness-gated"),
        "light": bool(args.light),
        "sampling": {"warmup": args.warmup, "iterations": args.iterations, "trials": args.trials},
        # False means the oracle's per-iteration argument clone was removed from the timing
        # loop. It changes the absolute score, so a number carrying one value of this field
        # cannot be compared with a number carrying the other.
        "arg_clone": bool(args.faithful_clone),
        "all_workloads_passed": all_passed,
        "workloads_scored": len(speedups),
        "workloads_total": len(facts),
        "baseline": {
            "name": baseline_name,
            "substituted": True,
            "contest_baseline": "flashinfer_wrapper_9sdjf3",
            "reason": (
                "The contest baseline wraps trtllm_fp8_block_scale_moe, whose dispatcher is "
                "get_trtllm_moe_sm100_module(); on sm_90 it raises 'No supported CUDA "
                "architectures found for major versions [10, 12]'. Substituted FlashInfer's "
                "cutlass_fused_moe, which supports DeepSeek FP8 block scaling on sm_90 only. "
                "Scores are therefore NOT comparable to the published contest leaderboard."
            ),
        },
        "tolerance": {"atol": ATOL, "rtol": RTOL, "required_matched_ratio": REQUIRED_MATCHED_RATIO},
        "gates_passed": not args.no_gates,
        "node": node,
        "elapsed_s": round(elapsed, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workloads": per_workload,
    }

    # Snapshot before printing, so the agent sees whether its run became the new best.
    snap = None
    if all_passed and not gate_trips and not args.no_snapshot:
        snap = snapshot_if_best(per_workload, ROOT / args.solution_dir,
                                Path(args.results_dir) if args.results_dir
                                else ROOT / "results",
                                baseline_ref=node.get("baseline_latency_ms"))
        report["snapshot"] = snap

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    if snap:
        # Whichever key this run ranked by. Printing 'geomean_latency_ms' unconditionally
        # crashed every snapshotting run on a node that has a baseline table -- which is
        # every set-up node, and so every agent's inner loop.
        k = snap["key"]
        unit, better = ("x", "higher") if k == "score_estimate" else (" ms", "lower")
        label = "score estimate" if k == "score_estimate" else "geomean latency"
        print(f"\nbest-so-far: {label} {snap[k]:.3f}{unit} ({better} is better)"
              + (" -- NEW BEST, saved to results/best_kernel.py" if snap["kept"]
                 else f" (best remains {snap['best_so_far']:.3f}{unit})"))

    if args.prompt_view:
        print()
        print(render_prompt_view(report))
    else:
        print(f"\nscore {score:.3f}x  ({len(speedups)}/{len(facts)} scored, all_passed={all_passed})")
        print(f"node  {node['gpu']} driver {node['driver_version']} "
              f"bw {node['achieved_hbm_bandwidth_gbs']} GB/s\n")
        for w in per_workload:
            print(f"  {w['verdict']}")
    print(f"\nwrote {out_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
