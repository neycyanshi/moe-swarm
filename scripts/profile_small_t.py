#!/usr/bin/env python3
"""
Profile the kernel at chosen seq_lens, one NVTX range per workload, for nsys.

    python scripts/profile_small_t.py --T 1 7 15                  # plain timing
    nsys profile -t cuda,nvtx -o /tmp/prof/small_t --force-overwrite true \
        python scripts/profile_small_t.py --T 1 7 15              # timeline
    nsys stats -r nvtx_sum,cuda_gpu_kern_sum /tmp/prof/small_t.nsys-rep

Routing logits, bias, expert offset and scaling factor come from the real workload file,
so routing activates exactly the experts the scored run does. Everything else is random,
as it is in the oracle. Without a real routing the expert count is wrong and the profile
describes a different workload.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "solution" / "python"))

DEFINITION = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
H, I, E = 7168, 2048, 32


def load_inputs(T, ds_root):
    from safetensors.torch import load_file
    rows = [json.loads(l) for l in (ROOT / "spec" / "workloads.jsonl").read_text().splitlines() if l.strip()]
    wl = next(r["workload"] for r in rows if int(r["workload"]["axes"]["seq_len"]) == T)
    inp = wl["inputs"]
    tensors = load_file(str((ds_root / inp["routing_logits"]["path"]).resolve()))
    dev = "cuda"
    fp8 = torch.float8_e4m3fn
    return dict(
        routing_logits=tensors[inp["routing_logits"]["tensor_key"]].to(dev),
        routing_bias=tensors[inp["routing_bias"]["tensor_key"]].to(dev),
        hidden_states=torch.randn(T, H, device=dev).to(fp8),
        hidden_states_scale=torch.rand(H // 128, T, device=dev) + 0.5,
        gemm1_weights=torch.randn(E, 2 * I, H, device=dev).to(fp8),
        gemm1_weights_scale=torch.rand(E, 2 * I // 128, H // 128, device=dev) + 0.5,
        gemm2_weights=torch.randn(E, H, I, device=dev).to(fp8),
        gemm2_weights_scale=torch.rand(E, H // 128, I // 128, device=dev) + 0.5,
        local_expert_offset=int(inp["local_expert_offset"]["value"]),
        routed_scaling_factor=float(inp["routed_scaling_factor"]["value"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[1, 7, 15])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    ds_root = Path(os.environ["FIB_DATASET_PATH"])
    from kernel import kernel
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")

    for T in args.T:
        x = load_inputs(T, ds_root)
        for _ in range(args.warmup):        # compile, autotune, graph capture
            kernel(**x)
        torch.cuda.synchronize()

        # Time each call the way the oracle's do_bench does: zero a 256 MB buffer, sync,
        # then events around the one call. The memset leaves L2 full of dirty lines whose
        # write-back lands in GEMM1 (~8 us at T=1); a back-to-back loop never sees it.
        ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(args.iters)]
        torch.cuda.nvtx.range_push(f"T={T}")
        for start, end in ev:
            flush.zero_()
            torch.cuda.synchronize()
            start.record()
            kernel(**x)
            end.record()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        us = sorted(s.elapsed_time(e) * 1e3 for s, e in ev)
        print(f"T={T:<4} {sum(us) / len(us):8.1f} us/call mean, {us[len(us) // 2]:.1f} median "
              f"(oracle-style, {args.iters} calls)")
        del x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
