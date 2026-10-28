"""
FP8 block-scale fused MoE -- correctness-first v1.

Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Ground truth: spec/reference_official.py (read that, not this, for semantics).

This is the swarm's starting line, not a target design. It is written to be *obviously
correct* and to avoid the reference's three worst performance sins, nothing more:

  1. The reference dequantizes ALL 32 local experts to fp32 up front (~5.6 GB per call).
     Here each expert is dequantized inside the loop and freed, so peak weight memory is
     one expert (~88 MB in bf16) instead of 5.6 GB.
  2. The reference builds full-size expanded scale tensors with repeat_interleave. Here a
     reshape makes the 128x128 block broadcast free.
  3. The reference calls .any() per expert -- 32 device-to-host syncs. At seq_len=1 the
     memory floor is ~15 us, so 32 syncs would dominate the whole kernel. Here routing is
     compacted once into a sorted (token, expert) permutation using 2 syncs total.

KNOWN HEADROOM, roughly in order (see docs/DESIGN.md section 2):
  - Weights are read as fp8 (44 MB/expert) then written+reread as bf16 (176 MB/expert of
    extra traffic). A real fp8 GEMM with block scales applied in the epilogue would move
    ~5x fewer bytes. This is the single biggest win and the reason to go to Triton.
  - GEMM1 and GEMM2 round-trip the [n, 2048] intermediate through HBM. Fusing them keeps
    it in registers/shared -- the actual "fused" in fused MoE.
  - Still 2 host syncs, which matter at seq_len <= 16.
  - The per-expert Python loop serializes 32 launches; a grouped/persistent kernel would
    issue one.
"""

import torch
import torch.nn.functional as F

# Fixed by the definition. Asserted below rather than trusted.
H = 7168          # hidden_size
I = 2048          # intermediate_size
E_GLOBAL = 256    # num_experts
E_LOCAL = 32      # num_local_experts (expert parallelism = 8)
BLOCK = 128       # FP8 quantization block edge
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4

NHB = H // BLOCK        # 56 hidden blocks
NIB = I // BLOCK        # 16 intermediate blocks
NG1B = (2 * I) // BLOCK  # 32 gemm1-out blocks


def _dequant_2d(w, scale, bf16=True):
    """Dequantize a block-scaled FP8 matrix.

    w:     [M, K] float8, M and K both multiples of BLOCK
    scale: [M/BLOCK, K/BLOCK] float32

    Reshaping to [M/B, B, K/B, B] lets the scale broadcast against axes 0 and 2 with no
    expanded intermediate, which is what repeat_interleave would have allocated.
    """
    m, k = w.shape
    out = w.view(m // BLOCK, BLOCK, k // BLOCK, BLOCK).to(torch.float32)
    out.mul_(scale.view(m // BLOCK, 1, k // BLOCK, 1))
    out = out.to(torch.bfloat16) if bf16 else out
    return out.view(m, k)


def _route(routing_logits, routing_bias, routed_scaling_factor):
    """DeepSeek-V3 no-aux routing. Mirrors spec/reference_official.py exactly.

    Returns:
        topk_idx: [T, TOP_K] int64  -- global expert ids
        weights:  [T, E_GLOBAL] float32 -- combine weights, zero off the top-k
    """
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    T = logits.shape[0]

    s = torch.sigmoid(logits)                       # [T, E]  combine weights come from this
    s_with_bias = s + bias                          # [T, E]  selection uses this

    # Group score = sum of the top-2 within each group of 32 experts.
    grouped = s_with_bias.view(T, N_GROUP, E_GLOBAL // N_GROUP)
    top2 = torch.topk(grouped, k=2, dim=2, largest=True, sorted=False).values
    group_scores = top2.sum(dim=2)                  # [T, N_GROUP]

    # Keep TOPK_GROUP groups, mask the rest out of contention.
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    keep = group_mask.unsqueeze(2).expand(T, N_GROUP, E_GLOBAL // N_GROUP).reshape(T, E_GLOBAL)

    pruned = s_with_bias.masked_fill(keep == 0, torch.finfo(torch.float32).min)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=False).indices  # pick topk from all total E_GLOBAL experts

    # Combine weights: s WITHOUT bias, normalized over the chosen k, then scaled.
    mask = torch.zeros_like(s)
    mask.scatter_(1, topk_idx, 1.0)
    weights = s * mask
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20) * routed_scaling_factor
    return topk_idx, weights


def _compact(topk_idx, weights, local_start, device):
    """Flatten routing into (token, local_expert, weight) triples sorted by expert.

    Costs exactly 2 device-to-host syncs (the nonzero compaction and the count readback),
    versus one per expert if we tested emptiness inside the loop. Returns token ids and
    weights already grouped by expert, plus per-expert counts on the host so the loop can
    slice without further syncs.
    """
    T = topk_idx.shape[0]

    flat_tok = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K).reshape(-1)
    flat_exp = topk_idx.reshape(-1)  # expand expert indices to [T*TOP_K] for compacting
    flat_w = torch.gather(weights, 1, topk_idx).reshape(-1)

    is_local = (flat_exp >= local_start) & (flat_exp < local_start + E_LOCAL)  # Throw away non-local experts, leaving only the ones this rank is responsible for.
    sel = is_local.nonzero(as_tuple=True)[0]  # sync 1: host tensor sel shape is device data-dependent, average ~T*TOP_K*E_LOCAL/E_GLOBAL = T
    tok = flat_tok[sel]
    le = flat_exp[sel] - local_start
    w = flat_w[sel]

    # Group by expert. Each token selects an expert at most once, so no duplicates.
    perm = torch.argsort(le, stable=True)
    counts = torch.bincount(le, minlength=E_LOCAL).tolist()  # sync 2: bincount need max the size lives on host cpu, tolist is also explicit sync to host values
    return tok[perm], w[perm], counts


# One call = one expert-parallel rank. With 256 global experts and 32 local, a forward
# pass runs this 8 times, once per rank, each with its own local_expert_offset (0, 32,
# ... 224). Every rank routes over all 256 experts -- redundant but cheaper than
# communicating the routing -- and handles only the top-k picks landing in its own slice.
# The returned [T, H] is therefore a PARTIAL sum; the caller all-reduces across ranks.
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
    device = hidden_states.device
    local_start = int(local_expert_offset)

    assert hidden_states.shape == (T, H)
    assert hidden_states_scale.shape == (NHB, T)
    assert gemm1_weights.shape == (E_LOCAL, 2 * I, H)
    assert gemm1_weights_scale.shape == (E_LOCAL, NG1B, NHB)
    assert gemm2_weights.shape == (E_LOCAL, H, I)
    assert gemm2_weights_scale.shape == (E_LOCAL, NHB, NIB)

    topk_idx, weights = _route(routing_logits, routing_bias, routed_scaling_factor)  # routing_logits [T, E_GLOBAL], same for all ranks.
    tok, tok_w, counts = _compact(topk_idx, weights, local_start, device)
    # tok, tok_w: [N] on device sorted by local expert; counts: 32 host ints summing to N,
    # where N is this rank's share of the assignments, bounded by T*TOP_K, on average ~T.

    # Activations: scale is [H/BLOCK, T], i.e. transposed relative to hidden_states.
    a_scale = hidden_states_scale.to(torch.float32).t().contiguous()      # [T, NHB]
    act = hidden_states.view(T, NHB, BLOCK).to(torch.float32)
    act.mul_(a_scale.unsqueeze(-1))
    act = act.to(torch.bfloat16).view(T, H)

    out = torch.zeros((T, H), dtype=torch.float32, device=device)

    offset = 0
    for le in range(E_LOCAL):
        n = counts[le]
        if n == 0:
            continue
        tok_e = tok[offset:offset + n]
        w_e = tok_w[offset:offset + n]
        offset += n

        # Dequantize this expert only, then let it fall out of scope.
        w13 = _dequant_2d(gemm1_weights[le], gemm1_weights_scale[le])     # [2I, H]
        g1 = act.index_select(0, tok_e) @ w13.t()                         # [n, 2I]
        del w13

        # SwiGLU: silu is applied to the SECOND half and multiplied by the first.
        # Getting this backwards still produces plausible output. It is not backwards here.
        x1, x2 = g1[:, :I], g1[:, I:]
        c = (F.silu(x2.float()) * x1.float()).to(torch.bfloat16)          # [n, I]

        w2 = _dequant_2d(gemm2_weights[le], gemm2_weights_scale[le])      # [H, I]
        o = c @ w2.t()                                                    # [n, H]
        del w2

        out.index_add_(0, tok_e, o.float() * w_e.unsqueeze(1))

    return out.to(torch.bfloat16)
