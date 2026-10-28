#!/usr/bin/env bash
# Bring a fresh Linux + NVIDIA Hopper (H100/H200, sm_90) machine to the point where
# `python bench.py` and `python swarm.py` work.
#
# Idempotent: run it on any new machine, as many times as you like.
#
#   bash scripts/node-setup.sh
#   source ~/ks-env.sh
#   python bench.py --light --prompt-view
#
# Assumes: CUDA 12.8+ driver, Python 3.10+ with a CUDA build of PyTorch, git, and an
# Anthropic API key in ANTHROPIC_API_KEY if you want to run agents.

set -euo pipefail

REPO="${KS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
DATASET="$REPO/mlsys26-contest"
step() { printf '\n=== %s\n' "$1"; }

cd "$REPO"

# 1. One env file, so every shell and every agent subprocess inherits the same settings.
step "~/ks-env.sh"
cat > ~/ks-env.sh <<EOF
export FIB_DATASET_PATH="$DATASET"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="\$HOME/.local/bin:\$PATH"
EOF
# shellcheck source=/dev/null
source ~/ks-env.sh
touch ~/.bashrc
grep -q 'ks-env.sh' ~/.bashrc || echo '[ -f ~/ks-env.sh ] && . ~/ks-env.sh' >> ~/.bashrc
echo "wrote ~/ks-env.sh and sourced it from .bashrc"

step "torch sees the GPU"
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(torch.cuda.get_device_name(0), 'sm', ''.join(map(str, torch.cuda.get_device_capability(0))))
"

# 2. The scorer.
step "flashinfer-bench"
python -c "import flashinfer_bench" 2>/dev/null || pip install -q flashinfer-bench
python -c "import flashinfer_bench as f; print('flashinfer-bench', f.__version__)"

# 3. The 19 scored workloads. A plain clone smudges every LFS blob in the dataset (3.6 GB
#    across all tracks); the MoE slice is ~27 MiB, so clone pointers and pull only that.
step "contest dataset"
if [[ -d "$DATASET/workloads" ]]; then
    echo "already present ($(du -sh "$DATASET" | cut -f1))"
else
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
        https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest "$DATASET"
    ( cd "$DATASET" && git lfs pull --include="blob/workloads/moe/**" )
fi
# Fail here rather than three layers down in flashinfer-bench: an unfetched LFS pointer
# looks like a corrupt safetensors blob.
python - <<'PY'
import glob, sys
bad = [f for f in glob.glob("mlsys26-contest/blob/workloads/moe/**/*.safetensors", recursive=True)
       if open(f, "rb").read(9) == b"version h"]
if bad:
    sys.exit(f"ERROR: {len(bad)} LFS pointer(s) not fetched, e.g. {bad[0]}")
print("MoE safetensors are real blobs")
PY

# 4. The coding agent. Without it swarm.py runs zero agents; bench.py and the gates work
#    regardless.
step "coding agent"
if ! command -v claude >/dev/null 2>&1; then
    npm install -g @anthropic-ai/claude-code >/dev/null 2>&1 \
        || echo "WARNING: could not install Claude Code (npm i -g @anthropic-ai/claude-code)"
fi
command -v claude >/dev/null 2>&1 && claude --version 2>&1 | head -1

# 5. Pin the SM clock before anything is measured. Clock ramp, not the kernels, was the
#    measurement noise: unpinned, one kernel spread 115% across six runs; locked at
#    1400 MHz it spread 0.18%, at <1% cost against boost. Needs root.
step "pin the SM clock"
KS_SM_CLOCK="${KS_SM_CLOCK:-1400}"
if nvidia-smi --lock-gpu-clocks="${KS_SM_CLOCK},${KS_SM_CLOCK}" >/dev/null 2>&1; then
    echo "${KS_SM_CLOCK}" > /tmp/ks-sm-clock-mhz
    echo "SM clock locked to ${KS_SM_CLOCK} MHz"
else
    rm -f /tmp/ks-sm-clock-mhz
    echo "WARNING: could not lock the SM clock; measurements will carry clock-ramp noise"
    echo "         and swarm.py will refuse to start a round."
fi

# 6. Previous rounds, if a results branch exists.
step "previous results"
bash scripts/results-sync.sh pull || echo "no results branch yet"

# 7. The roofline denominator and the baseline table, measured on THIS machine. Must
#    follow the results pull, which may restore another machine's node.json.
step "calibrate this machine"
python bench.py --calibrate | grep -E '"gpu"|bandwidth' \
    || echo "WARNING: calibrate failed; bench.py will refuse to score until it succeeds."

# The baseline fails numerically at T=1 roughly one run in five, and a partial table makes
# ranking fall back to a worse statistic, so retry.
step "baseline reference table"
for attempt in 1 2 3; do
    out="$(python bench.py --baseline-ref 2>&1 | tail -4 || true)"
    echo "$out"
    grep -q "wrote 19/19" <<<"$out" && break
    [[ $attempt -lt 3 ]] && echo "  incomplete; retrying ($attempt/3)"
done

step "done"
cat <<'EOF'
Next:
  source ~/ks-env.sh
  python bench.py --light --prompt-view          # score + roofline verdict
  tmux new -s swarm 'python swarm.py --rounds 1 --parallel 7 --agent-timeout 5400'

The first bench.py run JIT-compiles the CUTLASS baseline (~70 min, once per machine).
EOF
