# Primer

Background for reading.

---

## 1. Vocabulary

**GEMM** — GEneral Matrix Multiply, `C[M,N] = A[M,K] @ B[K,N]`. Three axes with different
characters, and the whole art is in telling them apart:

| axis | meaning | can you tile it? |
|---|---|---|
| **M** | rows of output (here: tokens) | yes, freely |
| **N** | columns of output (here: neurons) | yes, freely |
| **K** | the summed-over axis | only with *partial sums* |

M and N tile freely because output elements are independent. K is different: an output
element is not finished until every one of its K products has been added, so cutting K
leaves you holding incomplete accumulators. Remember this asymmetry — §3 is entirely a
consequence of it.

**GEMV** — matrix × *vector*: a GEMM with `M = 1`. The distinction matters because
arithmetic intensity is `2MNK / (bytes of A + B + C)`. Each weight element gets used
exactly **M** times, so at M=1 you stream the entire weight matrix to do one multiply per
element. GEMM at large M is compute-bound; GEMV is always memory-bound. Our workloads sit
near the GEMV end (§2), which is why this project is a weight-streaming problem.

**MMA** — Matrix-Multiply-Accumulate, the tensor-core instruction: `D += A @ B` on a small
fixed tile, one instruction. On Hopper it is `wgmma`, issued by a warpgroup, and its shape
is **quantized: M is fixed at 64**. Ask for 2 rows and the hardware still does 64. That
quantization, not bandwidth, is the waste discussed in §4.

**The execution hierarchy**, bottom to top:

```
thread
warp        = 32 threads, scheduled as one
warpgroup   = 4 warps = 128 threads          <- the unit that issues wgmma
CTA / block = 1..1024 threads, lives on ONE SM, shares that SM's shared memory
cluster     = up to 16 CTAs (Hopper), can read each other's shared memory
grid        = all CTAs of a launch
```

**CTA** (Cooperative Thread Array) is just CUDA's "thread block" — same thing, PTX
spelling. **SM** (Streaming Multiprocessor) is the physical core that runs CTAs; an
H100/H200 SXM has **132 SMs**, each with **228 KB shared memory (SMEM)** and a 256 KB register file.
The practical consequence: *a kernel that generates fewer than 132 independent tiles cannot
fill the GPU*, no matter how efficient each tile is.

**Split-K** — the fix for exactly that. When M and N are too small to produce 132 tiles,
partition the K axis across several CTAs, each computing a partial sum, then reduce them
(atomics, or a second pass). You buy parallelism by paying for partial-accumulator traffic.

**TMA** (Tensor Memory Accelerator, Hopper) — an async bulk HBM→SMEM copy engine. One
descriptor moves a whole tile without the compute warps babysitting it, which is what makes
weight streaming overlap with math.

---

## 2. Where "2 rows per expert" comes from

You do **not** need the DeepSeek-V3 paper for this. It is `spec/reference_official.py:134-166`,
which is 30 lines:

```python
for le in range(E_local):                        # 32 local experts
    sel = (topk_idx == local_start + le).any(1)  # [T] bool: which tokens picked this expert
    token_idx = sel.nonzero().squeeze(1)         # [Tk] int64 row addresses (nonzero gives [Tk,1])
    Tk = token_idx.numel()                       # <-- this is M
    A_e = A.index_select(0, token_idx)           # [Tk, 7168]
    G1  = A_e @ W13_e.t()                        # [Tk, 4096]
    C   = silu(G1[:, 2048:]) * G1[:, :2048]      # [Tk, 2048]
    O   = C @ W2_e.t()                           # [Tk, 7168]
```

The shape chain in those first three lines is `[T,8]` bool → `[T]` bool mask → `[Tk]`
integer addresses. The last step matters: a *mask* is indexed by token, whereas `token_idx`
is a *list of addresses* whose values are tokens — which is what lets `index_select` collect
scattered rows into one dense `[Tk, 7168]` block.

`Tk` — the number of tokens that routed to expert `le` — **is the M of that expert's two
GEMMs**. The arithmetic:

- Each token picks **top-8 of 256** experts (`TOP_K=8`, `E_global=256`).
- Expert parallelism 8 means only **32** experts are on this GPU, i.e. 1/8 of them.
- So expected local expert-visits per token = `8 × 32/256 = 1`. Summed over all local
  experts, `sum(Tk) ≈ T` — the surprise being **T, not 8T**.
- Experts with `Tk = 0` are skipped, so divide by the ones that run: at T=56, ~26.6 are
  active, giving `56 / 26.6 ≈ 2.1` rows per GEMM.

That is the whole meaning of "2 rows per expert GEMM": nine of the 19 scored workloads have
T∈[52,62], so their per-expert GEMM is `[2, 7168] @ [7168, 4096]` — two rows against a
44 MB weight matrix, run 27 times.

The DeepSeek-V3 paper explains *why* the routing looks like that (aux-loss-free balancing,
node-limited group selection). It changes no shape and is optional.

---

## 3. The fusion: keep the `[·,2048]` intermediate on-chip

`kernel.py` materializes three tensors to HBM per expert, each its own kernel launch:

```
A_e[Tk,7168] @ W13ᵀ  ->  G1[Tk,4096]      write + read back
SwiGLU(G1)           ->  C [Tk,2048]      write + read back    <- pure waste
C @ W2ᵀ              ->  O [Tk,7168]      the real output
```

`C` exists only to be GEMM2's left operand. Fusion means one kernel computes it into SMEM
and consumes it there, so it never touches HBM.

**Why it is affordable — you must hold one of two things whole, so pick the cheaper.** GEMM2
is `C[Tk,2048] @ W2ᵀ[2048,7168]`, so 2048 is its **K** and 7168 its **N**, and by §1 K cannot
be tiled without partial sums. You *can* consume each 128-wide chunk of `C` the moment GEMM1
produces it — but only by accumulating into every output column at once, which means keeping
all of `O` live instead. That is the whole choice:

| loop order | held whole | per row |
|---|---|---|
| N outer, K inner (the sketch below) | `C[BM, 2048]` bf16 | 2048 × 2 B = **4 KB** |
| K outer, N inner | `O[BM, 7168]` fp32 | 7168 × 4 B = **28 KB** |

`C` wins by 7×: N is 3.5× wider than K, and accumulators must be fp32 where `C` can be
bf16. Nothing else needs to be resident — 7168 as GEMM1's K is streamed in BK chunks and
reduced away, and as GEMM2's N it is tiled in BN chunks, each finished and stored.

So the on-chip requirement is `BM × 2048`, and **BM is the free knob** — which answers "can't
its tile have fewer rows?": yes, and that is exactly the dimension you shrink. BM=8 needs
8 × 2048 × 2 B = **32 KB** of the 228 KB SMEM; our real BM≈2 needs 8 KB.

Split-K on GEMM2 is the second row distributed across CTAs — cut K into `S` chunks of
`BK = 2048/S`, run them concurrently, reduce at the end. The input slice does shrink to
`[BM, BK]`, but each of the S CTAs still owns a full `[BM, 7168]` fp32 partial, because
output shape is independent of K. So it multiplies the 28 KB by S rather than avoiding it.
Useful when you are starved for parallelism and N is small — GEMM1 (§4), not GEMM2.

**The schedule.** GEMM1's own N is 4096, too wide to accumulate at once. Count it: a
`[8, 4096]` fp32 accumulator is 32,768 values, and MMA accumulators live in registers
partitioned privately across the 128 threads of the issuing warpgroup, one 32-bit register
per fp32 — so **32768 / 128 = 256 registers per thread**, just over the hard limit of 255,
with nothing left for addresses or staging. Spreading it over a 512-thread CTA gets to 64
regs/thread, but then those 32,768 registers are half the SM's 65,536-register file, so no
second CTA fits and there is nothing to hide latency with. Either way: tile it — but note
SwiGLU pairs column *j* with column *j+2048*, so walk the two halves together:

```python
# Triton sketch, one program = one (expert, row-tile)
c = tl.zeros([BM, 2048], dtype=tl.bfloat16)                  # the fused intermediate, SMEM

for j in range(16):                                          # 16 chunks of 128 columns
    a1 = tl.zeros([BM, 128], tl.float32)
    a2 = tl.zeros([BM, 128], tl.float32)
    for k in range(0, 7168, BK):                             # GEMM1's K: streamed
        a = load_A(rows, k, BK)                              # [BM, BK]
        a1 += tl.dot(a, load_W13(j*128,        k, BK))       # gate half
        a2 += tl.dot(a, load_W13(2048 + j*128, k, BK))       # up   half
    c[:, j*128:(j+1)*128] = (a1 * tl.sigmoid(a2) * a2)       # SwiGLU, in-register
                                                             # ^ no store to HBM

for n in range(0, 7168, BN):                                 # GEMM2's N: tiled
    o = tl.zeros([BM, BN], tl.float32)
    for k in range(0, 2048, BK):                             # K read from SMEM, not HBM
        o += tl.dot(c[:, k:k+BK], load_W2(n, BN, k, BK))
    tl.atomic_add(out + rows*7168 + n, o * routing_weight)
```

The two things to see: `c` is written and read entirely inside the program, and the second
loop's `k` reads SMEM where the unfused version read HBM.

**What it actually buys**, which differs sharply by regime:

- **Small T (16 of 19 workloads):** the bytes are negligible — at T=56, `C` totals 229 KB
  against 1.17 GB of weights, 0.02%. The win is **launch count and the store→load
  dependency stall**: ~3 launches × ~27 active experts ≈ 80 launches at 5–10 µs each,
  against a 15 µs memory floor at T=1.
- **Large T:** real bandwidth. At T=14107, writing then re-reading `C` is ~116 MB against
  1.71 GB, ~7%.

---

## 4. Why small M hurts twice

At `Tk≈2`, two separate penalties stack:

1. **MMA quantization.** `wgmma` does M=64 rows minimum. Two real rows means ~97% of the
   tensor-core work is padding. Fix: small-M tiles, or a GEMV-shaped path that skips tensor
   cores entirely.
2. **Too few CTAs.** Tiles = `active_experts × ceil(Tk/BM) × ceil(7168/BN)`. Fusing §3 makes
   N a *sequential* loop inside one program — every N-tile needs the same complete
   `C[BM,2048]`, which lives in one CTA's SMEM — so the N factor drops out, `Tk≈2` collapses
   the row factor to 1, and you are left with ~27 CTAs. A CTA cannot span SMs, so at most 27
   of the 132 SMs are occupied.

**But idle SMs are not themselves the cost here** — this kernel needs bandwidth, not
arithmetic. The quantity to fill is **bytes in flight**: by Little's law, saturating the
measured 4.25 TB/s at ~600 ns HBM latency needs ≈2.6 MB outstanding. 27 CTAs with
double-buffered 128×128 fp8 weight tiles (16 KB) hold 27 × 2 × 16 KB ≈ 864 KB — a third of
that, so ~34% of peak bandwidth. H200's faster HBM3e raises the bar, not lowers it.
Four-deep buffering reaches ~1.7 MB and roughly closes the gap.

So deepen the TMA pipeline first; it is far cheaper than restructuring for more CTAs. If more
CTAs are genuinely needed, a Hopper **cluster** is the right tool — up to 16 CTAs can read
each other's SMEM, so one computes `C` and the rest share it. Do *not* parallelize N by
recomputing GEMM1 per N-tile: that re-streams the 44 MB `W13` 56 times, and bytes moved is
the thing being minimized.

---

## 5. Reading order

Two separate tracks. **Start with the repo** — `spec/reference_official.py` is a
specification in plain PyTorch, with no tiles, no shared memory and no MMA in it, so none of
the GPU material below is a prerequisite for reading it:

1. `spec/reference_official.py` (168 lines, ground truth) → `DESIGN.md` §1-2 →
   `KERNELS.md` → `solution/python/kernel.py`. Needs only `torch` indexing (`topk`,
   `scatter_`, `index_add_`, `matmul`) plus §2 and §3 of this file. `KERNELS.md` says what
   the current kernels actually do before you meet 830 lines of Triton, and is more
   reliable than their own docstrings.

Then, to make it *fast*, top-down:

2. **Simon Boehm, "How to Optimize a CUDA Matmul Kernel"** (siboehm.com) — the one
   essential item: GEMM tiling from naive to near-cuBLAS. Fusion is a tiling decision, so
   this comes before any fusion material.
3. **CUDA C++ Programming Guide**, "Programming Model" + "Hardware Implementation" — the
   thread/block/cluster hierarchy and memory spaces. Skim for vocabulary.
4. **Hopper architecture whitepaper** — tells you where the 132 SMs / 228 KB / M=64 numbers
   come from. Skim.
5. **CUTLASS `efficient_gemm.md`** — the formal tile hierarchy; consult when a term like
   "split-K", "warp tile" or "epilogue" needs pinning down. Reference, not a read-through.
6. **Triton tutorials 03 (matmul) then 06 (fused attention)** — fusion in practice; 06 is
   the same structural trick as §3 applied to softmax.

Skip the DeepSeek-V3 paper until you want to know *why* the routing is shaped this way.
