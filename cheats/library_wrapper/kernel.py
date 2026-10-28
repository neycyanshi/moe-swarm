"""
CHEAT 1 of 2 -- the library wrapper. Deliberately illegal. Not a candidate.

This is a fixture for `gates.py`, kept in the repo so gate 1 has a permanent
regression test (docs/DESIGN.md section 6, step 3).

Why this is the cheat worth writing first: it is the one an agent reaches for
without being told to cheat. Asked to make an FP8 fused MoE fast on sm_90, calling
the vendor kernel that already does exactly that is the locally rational move. It
would score ~1.0x against the paired baseline -- which is itself this same
FlashInfer kernel -- pass correctness perfectly, and teach us nothing at all.

Nothing except gate 1 stops it:
  - correctness:        passes, the numbers are right
  - gate 2 (inputs):    passes, output tracks the routing
  - gate 3 (state):     passes, no state persists
  - gate 1 (libraries): TRIPS on `from flashinfer.fused_moe import ...`

That asymmetry is the whole argument for having a static gate at all. No amount of
runtime probing distinguishes "wrote a great kernel" from "called someone else's".
"""

import torch

from flashinfer.fused_moe import cutlass_fused_moe   # <-- gate 1 trips here

H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
NHB = H // BLOCK


def _route(routing_logits, routing_bias, routed_scaling_factor):
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
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

    mask = torch.zeros_like(s)
    mask.scatter_(1, topk_idx, 1.0)
    weights = s * mask
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20) * routed_scaling_factor
    return topk_idx, weights


@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    T = routing_logits.shape[0]
    offset = int(local_expert_offset)
    rsf = float(routed_scaling_factor)

    topk_idx, weights = _route(routing_logits, routing_bias, rsf)
    tse = topk_idx.to(torch.int32).contiguous()
    tfs = torch.gather(weights, 1, topk_idx).to(torch.float32).contiguous()

    a_scale = hidden_states_scale.to(torch.float32).t().contiguous()
    act = hidden_states.view(T, NHB, BLOCK).to(torch.float32) * a_scale.unsqueeze(-1)
    act = act.to(torch.bfloat16).view(T, H).contiguous()

    out = cutlass_fused_moe(
        input=act,
        token_selected_experts=tse,
        token_final_scales=tfs,
        fc1_expert_weights=gemm1_weights,
        fc2_expert_weights=gemm2_weights,
        output_dtype=torch.bfloat16,
        quant_scales=[gemm1_weights_scale, gemm2_weights_scale],
        use_deepseek_fp8_block_scale=True,
        ep_size=E_GLOBAL // E_LOCAL,
        ep_rank=offset // E_LOCAL,
    )
    return out[0] if isinstance(out, (list, tuple)) else out
