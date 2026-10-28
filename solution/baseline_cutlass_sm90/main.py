"""
Paired baseline: FlashInfer CUTLASS fused MoE on sm_90.

WHY THIS EXISTS
---------------
The contest's own baseline (`flashinfer_wrapper_9sdjf3`, wrapping
`trtllm_fp8_block_scale_moe`) cannot run here. Its dispatcher is
`get_trtllm_moe_sm100_module()` and it raises

    RuntimeError: No supported CUDA architectures found for major versions [10, 12].

on sm_90, because sm_90 is CUDA major 9. That is a hard architecture restriction,
not a driver or install problem.

FlashInfer's other MoE backend does cover sm_90, and in fact covers it
*exclusively* -- `cutlass_fused_moe` gates DeepSeek block scaling the other way:

    if use_deepseek_fp8_block_scale:
        if device_arch != "90":
            raise NotImplementedError("FP8 block scaling not yet implemented for Blackwell.")

So this baseline is still a FlashInfer kernel, externally defined and not authored
by us -- which is what keeps "speedup" credible. It is a documented substitution
for the sm_100-only contest baseline, not a redefinition of the metric.

THE DTYPE CONTRACT, AND WHY ACTIVATIONS ARE bf16
------------------------------------------------
`FusedMoeRunner` picks a kernel by (activation dtype, weight dtype, output dtype).
Its fp8-activation branch is guarded by

    bool isFp8Quant() const {
      return !mUseDeepSeekFP8BlockScaling && mActivationDtype == dl_float8_e4m3fn && ...
    }

i.e. DeepSeek block scaling *excludes* the fp8 x fp8 runner, and nothing else
constructs one -- passing fp8 activations yields a null runner and
"Could not construct fused moe op". The branch that is unconditional on the
block-scale flag is bf16 activation x fp8 weight. The binding also rejects
`input_sf` unless the quant mode is mxfp8 or nvfp4, which is consistent: with bf16
activations there is no activation scale to pass.

So we dequantize activations to bf16 here (T x 7168 elements -- negligible) and
leave the weights, which are the 1.4 GB that actually decides the runtime, in fp8
with their 128x128 block scales. Validated against solution/python/kernel.py at
T in {16, 80, 901} and local_expert_offset in {0, 32, 192}: matched_ratio 1.0 under
the contest tolerance, max abs error ~0.03 on outputs of magnitude ~0.7.

Expert parallelism: the runner derives `num_experts_total = num_experts_on_rank *
ep_size` (= 32 * 8 = 256) and selects this rank's slice from `ep_rank`, so
`ep_rank = local_expert_offset // 32`.
"""

import torch

H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
NHB = H // BLOCK

from flashinfer.fused_moe import cutlass_fused_moe


def _route(routing_logits, routing_bias, routed_scaling_factor):
    """DeepSeek-V3 no-aux grouped routing. Mirrors spec/reference_official.py.

    Selection uses `s + bias`; the combine weights come from `s` WITHOUT bias.
    """
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
def run(
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

    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = int(local_expert_offset.item())
    else:
        local_expert_offset = int(local_expert_offset)
    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())
    else:
        routed_scaling_factor = float(routed_scaling_factor)

    topk_idx, weights = _route(routing_logits, routing_bias, routed_scaling_factor)
    token_selected_experts = topk_idx.to(torch.int32).contiguous()
    token_final_scales = torch.gather(weights, 1, topk_idx).to(torch.float32).contiguous()

    # hidden_states_scale is [H/BLOCK, T] -- transposed relative to hidden_states.
    a_scale = hidden_states_scale.to(torch.float32).t().contiguous()      # [T, NHB]
    act = hidden_states.view(T, NHB, BLOCK).to(torch.float32)
    act = act * a_scale.unsqueeze(-1)
    act = act.to(torch.bfloat16).view(T, H).contiguous()

    out = cutlass_fused_moe(
        input=act,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        fc1_expert_weights=gemm1_weights,
        fc2_expert_weights=gemm2_weights,
        output_dtype=torch.bfloat16,
        quant_scales=[gemm1_weights_scale, gemm2_weights_scale],
        use_deepseek_fp8_block_scale=True,
        ep_size=E_GLOBAL // E_LOCAL,
        ep_rank=local_expert_offset // E_LOCAL,
    )
    return out[0] if isinstance(out, (list, tuple)) else out
