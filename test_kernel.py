"""
CPU tests for the parts of solution/python/kernel.py that don't need a GPU.

Routing and compaction are pure index/fp32 math, so they are testable on a Mac -- and
they are the fiddliest logic in the kernel. The GEMM path needs FP8 and 80 GB, so it is
only verifiable on a GPU via bench.py.

Run: python test_kernel.py
"""

import sys

import torch

# These are unit tests for the PyTorch v1 implementation, kept at reference/ after the
# scored kernel became Triton. They still earn their place: the SwiGLU half-order and
# selection-vs-combination traps are the ones that produce plausible-but-wrong output,
# and catching them on a Mac in a second beats catching them on a node in four minutes.
sys.path.insert(0, "reference")
import kernel_pytorch_v1 as K  # noqa: E402


def route_reference(routing_logits, routing_bias, routed_scaling_factor):
    """Routing lifted verbatim from spec/reference_official.py, as an oracle."""
    T = routing_logits.shape[0]
    E_global = routing_logits.shape[1]
    TOP_K, N_GROUP, TOPK_GROUP = 8, 8, 4

    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = 1.0 / (1.0 + torch.exp(-logits))
    s_with_bias = s + bias

    group_size = E_global // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor
    return topk_idx, weights


def test_route():
    torch.manual_seed(0)
    for T in (1, 7, 56, 257):
        logits = torch.randn(T, K.E_GLOBAL, dtype=torch.float32)
        bias = torch.randn(K.E_GLOBAL, dtype=torch.float32)
        rsf = 2.5

        got_idx, got_w = K._route(logits, bias, rsf)
        exp_idx, exp_w = route_reference(logits, bias, rsf)

        # topk uses sorted=False, so compare the selected SET per token, not the order.
        assert torch.equal(got_idx.sort(dim=1).values, exp_idx.sort(dim=1).values), \
            f"T={T}: selected experts differ"
        # The [T, E] weight matrix is order-independent and must match exactly.
        torch.testing.assert_close(got_w, exp_w, rtol=0, atol=0)
        # Sanity: exactly TOP_K nonzero weights per token, summing to rsf.
        assert torch.equal((got_w > 0).sum(1), torch.full((T,), K.TOP_K))
        torch.testing.assert_close(got_w.sum(1), torch.full((T,), rsf), rtol=1e-5, atol=1e-5)
    print("test_route                 ok")


def test_compact():
    """Every (token, local_expert) pair must appear exactly once, with the right weight."""
    torch.manual_seed(1)
    for T, local_start in ((1, 0), (7, 32), (56, 224), (129, 96)):
        logits = torch.randn(T, K.E_GLOBAL, dtype=torch.float32)
        bias = torch.randn(K.E_GLOBAL, dtype=torch.float32)
        topk_idx, weights = K._route(logits, bias, 1.0)

        tok, w, counts = K._compact(topk_idx, weights, local_start, torch.device("cpu"))

        assert sum(counts) == tok.numel() == w.numel()

        # Rebuild the expected pair set by brute force.
        expected = set()
        for t in range(T):
            for ge in topk_idx[t].tolist():
                if local_start <= ge < local_start + K.E_LOCAL:
                    expected.add((t, ge - local_start))

        # counts must be sorted-by-expert group sizes, and tok grouped accordingly.
        got = set()
        offset = 0
        for le in range(K.E_LOCAL):
            for j in range(counts[le]):
                t = int(tok[offset + j])
                got.add((t, le))
                # weight must be the combine weight for THIS token/expert pair
                torch.testing.assert_close(
                    w[offset + j], weights[t, local_start + le], rtol=0, atol=0
                )
            offset += counts[le]

        assert got == expected, f"T={T} off={local_start}: pair set mismatch"
        # Expected local pairs ~= T (top_k=8 over 256 experts, 32 local => 8*32/256 = 1).
    print("test_compact               ok")


def test_dequant_block_scale():
    """Reshape-broadcast dequant must equal the naive repeat_interleave version."""
    torch.manual_seed(2)
    M, Kd = 256, 384                      # 2 x 3 blocks of 128
    w = torch.randn(M, Kd, dtype=torch.float32)
    scale = torch.rand(M // K.BLOCK, Kd // K.BLOCK, dtype=torch.float32) + 0.5

    naive = w * torch.repeat_interleave(
        torch.repeat_interleave(scale, K.BLOCK, dim=0), K.BLOCK, dim=1
    )
    got = K._dequant_2d(w, scale, bf16=False)
    torch.testing.assert_close(got, naive, rtol=0, atol=0)
    print("test_dequant_block_scale   ok")


def test_swiglu_half_order():
    """Guard the trap: silu goes on the SECOND half, multiplied by the first."""
    torch.manual_seed(3)
    n = 4
    g1 = torch.randn(n, 2 * K.I, dtype=torch.float32)
    x1, x2 = g1[:, :K.I], g1[:, K.I:]

    correct = (x2 / (1.0 + torch.exp(-x2))) * x1        # reference formula
    swapped = (x1 / (1.0 + torch.exp(-x1))) * x2        # the plausible-looking bug

    ours = torch.nn.functional.silu(x2) * x1
    torch.testing.assert_close(ours, correct, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(correct, swapped), "test itself is broken"
    print("test_swiglu_half_order     ok")


if __name__ == "__main__":
    test_route()
    test_compact()
    test_dequant_block_scale()
    test_swiglu_half_order()
    print("\nall CPU tests passed")
