#!/usr/bin/env python3
"""Reconstruct an agent's kernel.py by replaying its edits from the transcript.

    scripts/replay-agent-edits.py <transcript.jsonl> <base_ref> <out.py> [--until "TEXT"]

Worktrees are pruned at the end of a round, so a rejected candidate's code is gone -- but
every Write/Edit tool call is recorded verbatim in results/agent_<id>.jsonl. Replaying
them onto the base file rebuilds what the agent had.

REPLAY IS NOT GUARANTEED FAITHFUL. On the one case it was built for, the reconstructed
kernel -- extracted verbatim from a single Write, with no intervening edits -- did not
compile, although the agent had benchmarked that same file successfully. Treat output as
a lead to investigate, not as a candidate to trust. snapshot-best
(results/best_kernel.py) is the reliable path; this is forensics.

--until stops the replay at the first tool_result containing TEXT, which recovers a
version that WORKED rather than the state the agent happened to end on:

    scripts/replay-agent-edits.py results/agent_<id>.jsonl <base_sha> /tmp/k.py \
        --until "SCORE 1."
"""
import argparse, json, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("transcript")
ap.add_argument("base_ref")
ap.add_argument("out")
ap.add_argument("--until", default=None,
                help="stop at the first tool_result containing this text")
ap.add_argument("--file", default="solution/python/kernel.py")
args = ap.parse_args()

ev = [json.loads(l) for l in open(args.transcript) if l.strip().startswith("{")]

cut = len(ev)
if args.until:
    for i, e in enumerate(ev):
        if e.get("type") != "user":
            continue
        for b in e.get("message", {}).get("content", []) or []:
            if isinstance(b, dict) and b.get("type") == "tool_result" \
               and args.until in str(b.get("content", "")):
                cut = i
                break
        if cut != len(ev):
            break
    if cut == len(ev):
        print(f"WARNING: {args.until!r} never appeared; replaying the whole transcript")

text = subprocess.run(["git", "show", f"{args.base_ref}:{args.file}"],
                      capture_output=True, text=True, check=True).stdout

ops = skipped = 0
for e in ev[:cut]:
    if e.get("type") != "assistant":
        continue
    for b in e.get("message", {}).get("content", []) or []:
        if b.get("type") != "tool_use":
            continue
        i = b.get("input", {}) or {}
        if "kernel.py" not in str(i.get("file_path", "")):
            continue
        if b["name"] == "Write":
            text = i.get("content", text)
            ops += 1
        elif b["name"] in ("Edit", "MultiEdit"):
            for ed in (i.get("edits") or [i]):
                o, n = ed.get("old_string", ""), ed.get("new_string", "")
                if o and o in text:
                    text = text.replace(o, n, 1)
                    ops += 1
                else:
                    skipped += 1

open(args.out, "w").write(text)
print(f"replayed {ops} ops ({skipped} did not apply) over events 0..{cut} "
      f"-> {args.out}, {len(text.splitlines())} lines")
