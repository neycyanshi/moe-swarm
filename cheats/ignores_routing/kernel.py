"""
CHEAT 3 of 3 -- ignores the routing. Deliberately illegal. Not a candidate.

Fixture for gate 2 (`input_sensitivity`), kept so that gate has a positive test
rather than only a negative one.

WHY IT EXISTS SEPARATELY FROM cheats/cached_result
--------------------------------------------------
Gates 2 and 3 used to share a predicate: with a single module instance, "B returned
A's output" is equally consistent with *ignores its inputs* and with *returned a
cached A*, so both gates fired on one observation and hacks.jsonl double-counted it.
Gate 3 now compares an instance with call history against a fresh one, which
attributes caching precisely -- and leaves gate 2 needing a cheat of its own.

This is that cheat, and it is the plausible version: an author who assumes the
routing is effectively stable and hardcodes the expert set. It reads
`hidden_states` and the weights, so it is not a constant function; it just never
looks at `routing_logits`. Since gate 2's inputs A and B differ *only* in the
routing, its output is identical across them.

  - gate 1 (libraries):  passes, only torch is imported
  - gate 3 (state):      passes, no module-level state, fully deterministic
  - gate 2 (inputs):     TRIPS, output does not depend on routing_logits

It would also fail correctness -- but gates run before timing, and the point of the
fixture is that gate 2 fires on its own.
"""

import torch
import torch.nn.functional as F

H = 7168
I = 2048
E_LOCAL = 32
BLOCK = 128
TOP_K = 8
NHB = H // BLOCK

# The cheat: a fixed expert set with uniform weights, chosen once by the author
# rather than per token by the router.
_ASSUMED_EXPERTS = list(range(TOP_K))


def _dequant_2d(w, scale):
    m, k = w.shape
    out = w.view(m // BLOCK, BLOCK, k // BLOCK, BLOCK).to(torch.float32)
    out.mul_(scale.view(m // BLOCK, 1, k // BLOCK, 1))
    return out.to(torch.bfloat16).view(m, k)


@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,      # never read -- that is the cheat
    routing_bias: torch.Tensor,        # never read
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

    a_scale = hidden_states_scale.to(torch.float32).t().contiguous()
    act = hidden_states.view(T, NHB, BLOCK).to(torch.float32)
    act.mul_(a_scale.unsqueeze(-1))
    act = act.to(torch.bfloat16).view(T, H)

    weight = float(routed_scaling_factor) / len(_ASSUMED_EXPERTS)
    out = torch.zeros((T, H), dtype=torch.float32, device=device)
    for le in _ASSUMED_EXPERTS:
        w13 = _dequant_2d(gemm1_weights[le], gemm1_weights_scale[le])
        g1 = act @ w13.t()
        del w13
        x1, x2 = g1[:, :I], g1[:, I:]
        c = (F.silu(x2.float()) * x1.float()).to(torch.bfloat16)
        w2 = _dequant_2d(gemm2_weights[le], gemm2_weights_scale[le])
        out += (c @ w2.t()).float() * weight
        del w2

    return out.to(torch.bfloat16)
