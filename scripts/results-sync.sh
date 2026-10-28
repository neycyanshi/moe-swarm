#!/usr/bin/env bash
# Sync a GPU machine's results back through git, and recover them on a fresh machine.
#
# GPU machines are disposable (spot / preemptible capacity): local disk dies with the
# instance, so every round's results go to a git branch before anything else happens.
#
# WHY A DEDICATED BRANCH
#   The machine pushes to `swarm-results`, never to main. An unattended overnight run must
#   not race a developer for main, and must never be tempted to force-push.
#
#   Commits are built on top of origin/swarm-results in a throwaway worktree, NOT on the
#   node's checked-out HEAD. "Only one node writes it, so it is always a fast-forward" is
#   false the moment a second node exists: a fresh node starts from main, which does not
#   descend from the branch tip, and its push is rejected non-fast-forward. Observed
#   exactly that, and because swarm.py treats a failed push as fatal it stopped the run.
#
# USAGE (on the GPU machine)
#   scripts/results-sync.sh push     # after every round -- cheap, do not batch it
#   scripts/results-sync.sh pull     # on a fresh node, to recover previous rounds

set -euo pipefail

REPO="${KS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
BRANCH="${KS_SYNC_BRANCH:-swarm-results}"

cd "$REPO"

case "${1:-}" in
    push)
        git fetch -q origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" 2>/dev/null || true

        # Build the commit in a throwaway worktree parented on the remote branch, so the
        # push is a fast-forward no matter which node or which base commit we are on.
        WT="$REPO/.worktrees/_sync"
        git worktree remove --force "$WT" 2>/dev/null || true
        rm -rf "$WT"
        if git rev-parse -q --verify "origin/$BRANCH" >/dev/null; then
            git worktree add -q --detach "$WT" "origin/$BRANCH"
        else
            git worktree add -q --detach "$WT" HEAD
        fi

        mkdir -p "$WT/results" "$WT/solution/python"
        cp -R "$REPO/results/." "$WT/results/" 2>/dev/null || true
        cp "$REPO/solution/python/kernel.py" "$WT/solution/python/kernel.py" 2>/dev/null || true

        # Scope every step to these paths. `git commit` without a pathspec commits the
        # whole index, which would sweep in whatever an agent happened to have staged.
        #
        # Named individually and force-added, because the worktree is parented on
        # origin/$BRANCH and therefore carries THAT branch's .gitignore -- which is
        # whatever it was when the branch last moved, not what main says today. When
        # results/lineage_* were un-ignored on main, the sync worktree kept ignoring them
        # and pushed nothing: two rounds of surviving kernels existed only on node-local
        # disk while the push reported success. `git add -A -- results` cannot be used
        # with -f either, or it would sweep in every per-candidate bench json that
        # .gitignore is there to keep out.
        #
        # Agent transcripts go up GZIPPED. Raw they are 1.4 MB each and were deliberately
        # excluded for that reason; compressed they are ~10% of that, which is worth
        # paying. A preemption destroyed round 3's transcripts along with the only record
        # of how many measurements each agent took and whether it was still improving at
        # the timeout -- questions that then could not be answered at all.
        (
            cd "$WT"
            PATHS=(solution/python/kernel.py results/leaderboard.jsonl results/hacks.jsonl
                   results/node.json results/lineages.json results/checkpoint.json)
            for f in results/lineage_*.py results/lineage_*.bench.json \
                     results/trace_*.jsonl; do
                [[ -e "$f" ]] && PATHS+=("$f")
            done
            # Gzip the worktree's copy, never the live results/ -- an agent may still be
            # appending to its transcript. The .jsonl itself is never added; only the .gz.
            for f in results/agent_*.jsonl; do
                [[ -e "$f" ]] || continue
                gzip -9 -c "$f" > "$f.gz" && PATHS+=("$f.gz")
            done
            EXISTING=()
            for f in "${PATHS[@]}"; do
                [[ -e "$f" ]] && EXISTING+=("$f")
            done
            git add -f -- "${EXISTING[@]}" 2>/dev/null || true
            if git diff --cached --quiet -- "${EXISTING[@]}"; then
                echo "nothing to sync."
                exit 0
            fi
            git commit -q -m "swarm: results as of $(date -u +%Y-%m-%dT%H:%M:%SZ)" -- "${EXISTING[@]}"
            git push -q origin "HEAD:refs/heads/$BRANCH"
            echo "pushed $(git rev-parse --short HEAD) -> origin/$BRANCH ($((${#EXISTING[@]})) paths)"
        )
        rc=$?
        git worktree remove --force "$WT" 2>/dev/null || true
        exit $rc
        ;;
    pull)
        if ! git fetch -q origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" 2>/dev/null; then
            echo "no origin/$BRANCH yet -- starting fresh."
            exit 0
        fi
        # Take only results/ from the branch. Code comes from main; clobbering the
        # working tree here would undo whatever the node is mid-way through.
        if git checkout "origin/$BRANCH" -- results 2>/dev/null; then
            # checkout stages what it writes. results/ is not tracked on main, so leaving
            # them staged only makes the tree look dirty and a later pull look blocked.
            git reset -q -- results 2>/dev/null || true
            echo "recovered results/ from origin/$BRANCH ($(git rev-parse --short "origin/$BRANCH")):"
            ls -la results/ | tail -n +2
        else
            echo "origin/$BRANCH has no results/ yet -- starting fresh."
        fi
        ;;
    *) sed -n '2,24p' "$0"; exit 1 ;;
esac
