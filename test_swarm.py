#!/usr/bin/env python3
"""
Regression tests for swarm.py's invariants (docs/DESIGN.md section 5).

    python test_swarm.py          # no GPU, no agent, no network

These cover the three things that must not silently break, because each of them
fails *quietly* rather than loudly if it regresses:

  - the write-set guard, which is the only thing stopping a candidate from winning
    by editing the oracle instead of the kernel;
  - worktree cleanup, since node disk is small and a leak fills it mid-run;
  - abort-on-failed-push, which is a hard requirement (swarm.py docstring, point 4) and
    would otherwise degrade into a warning nobody reads at 3am.

Every check must behave identically on a Mac and on a node, and none may touch the
real remote. An earlier version of the push test leaned on a node-only path
being absent; on an actual GPU node it therefore pushed for real.

Deliberately not covered here: anything needing a GPU or a live agent. Those are
bench.py's and test_gates.py's job.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import swarm

_failures = []


def check(name, cond, detail=""):
    print(f"{name:<48} {'ok' if cond else 'FAIL'}"
          f"{('  -- ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        _failures.append(name)


def main():
    base = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    wt = swarm.make_worktree(base, "selftest")
    try:
        check("worktree is created off the base commit", wt.exists())
        check("a clean worktree has no violations", swarm.violations(wt) == [])

        (wt / "solution" / "python" / "kernel.py").write_text("# edited\n")
        check("editing the write-set file is allowed", swarm.violations(wt) == [])

        (wt / "bench.py").write_text("# tampered\n")
        (wt / "scratch.txt").write_text("junk\n")
        bad = swarm.violations(wt)
        check("editing the oracle is caught", "bench.py" in bad, bad)
        check("stray new files are caught", "scratch.txt" in bad, bad)
        check("the write-set file is still not flagged",
              "solution/python/kernel.py" not in bad, bad)

        # bench.py writes into results/ from inside the agent's own worktree, and
        # hacks.jsonl is un-ignored because it is deliverable 3. Counting that as the
        # agent's edit failed a candidate that had completed cleanly.
        (wt / "results").mkdir(exist_ok=True)
        (wt / "results" / "hacks.jsonl").write_text('{"gate": "x"}\n')
        (wt / "results" / "bench.json").write_text("{}\n")
        bad2 = swarm.violations(wt)
        check("harness output in results/ is not a violation",
              not [p for p in bad2 if p.startswith("results/")], bad2)
    finally:
        swarm.remove_worktree(wt)
    check("worktree is removed again", not wt.exists())

    # Force the push to fail, rather than relying on the environment to fail for us.
    # The first version of this test leaned on results-sync.sh's default KS_REPO
    # being absent, which is true on a Mac and false on a node --
    # so on the node the push SUCCEEDED, the assertion failed, and the test pushed real
    # commits to origin/swarm-results as a side effect. Pointing KS_REPO at a
    # non-repository makes the failure deterministic everywhere and touches no remote.
    prev = os.environ.get("KS_REPO")
    os.environ["KS_REPO"] = str(Path(tempfile.mkdtemp()) / "not-a-repo")
    try:
        swarm.sync_results(1)
        check("a failed results push aborts the run", False,
              "sync_results returned instead of raising")
    except SystemExit as e:
        check("a failed results push aborts the run", True)
        check("the abort explains why it is fatal", "unpushed" in str(e), str(e)[:80])
    finally:
        if prev is None:
            os.environ.pop("KS_REPO", None)
        else:
            os.environ["KS_REPO"] = prev

    # merge_lineages must not relabel one kernel with whichever reading came out higher.
    # Round 1 recorded an unchanged kernel at 1.9452x instead of the 1.8598x its own
    # re-measurement gave, and that number became the bar the next round was set.
    same = "KERNEL SOURCE"
    merged = swarm.merge_lineages(
        [{"score": 1.8598, "kernel": same, "origin": "incumbent", "bench": None}],
        [{"score": 1.9452, "kernel": same, "origin": "lucky-candidate", "bench": None}],
    )
    check("identical kernels collapse to one lineage", len(merged) == 1, f"{len(merged)}")
    check("the lower reading of identical code is the one kept",
          merged and abs(merged[0]["score"] - 1.8598) < 1e-9,
          f"{merged[0]['score'] if merged else None}")

    # A genuinely better candidate must still win, or the fix above would freeze the swarm.
    merged = swarm.merge_lineages(
        [{"score": 1.8598, "kernel": "OLD", "origin": "incumbent", "bench": None}],
        [{"score": 1.9452, "kernel": "NEW", "origin": "real-gain", "bench": None}],
    )
    check("a different, higher-scoring kernel still leads",
          merged and merged[0]["origin"] == "real-gain",
          merged[0]["origin"] if merged else None)

    # The end-of-round paths only run after the agents have, so a KeyError or a
    # non-serialisable value there costs a whole round. Exercise them on representative
    # dicts instead of finding out at 2 a.m.
    tmp = Path(tempfile.mkdtemp())
    (tmp / "lineage_1.py").write_text("K1")
    (tmp / "lineage_2.py").write_text("K2")
    lineages = [
        {"score": 3.07, "kernel": "K1", "origin": "r2-a5-tile", "bench": None,
         "full_score": 2.94, "full_measured_at": "2026-09-16"},
        {"score": 3.14, "kernel": "K2", "origin": "r2-a0-comb", "bench": None,
         "full_score": 2.82, "full_measured_at": "2026-09-16"},
    ]
    swarm.save_lineages(tmp, lineages)
    back = swarm.load_lineages(tmp, "fallback", 0.0, None)
    check("lineages round-trip keeps both slots", len(back) == 2, str(len(back)))
    check("lineages round-trip keeps the light score",
          back and abs(back[0]["score"] - 3.07) < 1e-9, str(back[0]["score"]))
    check("lineages round-trip keeps the full score",
          back and back[0].get("full_score") == 2.94, str(back[0].get("full_score")))
    check("lineages round-trip keeps full_measured_at",
          back and back[0].get("full_measured_at") == "2026-09-16",
          str(back[0].get("full_measured_at")))

    # A leaderboard record must be JSON-serialisable. Path objects reaching it is the
    # realistic failure -- bench paths are carried as strings for exactly this reason.
    rec = {"round": 2, "agent": "r2-a0", "strategy": "s", "base": "abc",
           "seeded_from": "x", "seed_score": 3.0, "seed_lineage": 0, "cost_usd": None,
           "agent_seconds": 1.0, "note": "n", "agent_log": "results/a.jsonl",
           "agent_status": "ok", "score": 3.14, "passed": True, "light_runs": [3.1, 3.2],
           "bench": str(tmp / "b.json"), "gate_trips": [], "scored_snapshot": False,
           "agent_light_score": 3.13, "kept": True, "is_leader": True,
           "best_score_after": 3.14, "timestamp": "2026-09-16T00:00:00+00:00"}
    try:
        json.dumps(rec)
        check("a leaderboard record serialises", True)
    except TypeError as e:
        check("a leaderboard record serialises", False, str(e))

    # An unchanged-from-seed candidate carries no bench path; promotion must tolerate it.
    rec_skipped = dict(rec, agent="r2-a2", unchanged_from_seed=True, score=3.0)
    rec_skipped.pop("bench")
    try:
        json.dumps(rec_skipped)
        check("a skipped candidate's record serialises", True)
    except TypeError as e:
        check("a skipped candidate's record serialises", False, str(e))

    # The checkpoint is the only thing standing between a preemption and a lost agent
    # phase, and it runs exclusively on the path where something has already gone wrong,
    # so nothing else would exercise it.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        old_root, old_wts = swarm.ROOT, swarm.WORKTREES
        try:
            swarm.ROOT, swarm.WORKTREES = tmp, tmp / ".worktrees"
            (tmp / "results").mkdir()
            wt = tmp / "agent0"
            (wt / "solution" / "python").mkdir(parents=True)
            (wt / "solution" / "python" / "kernel.py").write_text("# candidate\n")
            ran = [({"agent": "r4-a0-x", "seed_lineage": 0}, wt, "timeout")]

            swarm.write_checkpoint(4, ran)
            check("checkpoint is written", (tmp / "results" / "checkpoint.json").exists())

            back = swarm.load_checkpoint(4)
            check("checkpoint restores one agent", back is not None and len(back) == 1)
            rec, rwt, status = back[0]
            check("checkpoint restores the record", rec["agent"] == "r4-a0-x")
            check("checkpoint restores the status", status == "timeout")
            check("checkpoint restores the kernel",
                  (rwt / "solution" / "python" / "kernel.py").read_text() == "# candidate\n")

            check("a different round is not resumed", swarm.load_checkpoint(5) is None)

            swarm.complete_checkpoint(4)
            check("a completed round is not resumed", swarm.load_checkpoint(4) is None)

            (tmp / "results" / "checkpoint.json").write_text("{ not json")
            check("a corrupt checkpoint starts fresh instead of raising",
                  swarm.load_checkpoint(4) is None)
        finally:
            swarm.ROOT, swarm.WORKTREES = old_root, old_wts

    # The audit rewrites a recorded measurement, and `full_score` decides slot ordering, so
    # a wrong correction is as damaging as the bad leg it repairs.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        def sweep(score, legs):
            # legs: seq_len -> (baseline_ms, ours_ms, bound)
            return {"score": score, "light": False, "workloads": [
                {"seq_len": t, "baseline_latency_ms": b, "latency_ms": o,
                 "roofline": {"bound": bound}} for t, (b, o, bound) in legs.items()]}

        table = {"1": 0.500, "7": 0.525, "11948": 3.500}
        (tmp / "node.json").write_text(json.dumps({"baseline_latency_ms": table}))

        # T=7's leg reads 3x the table; T=11948 is compute-bound and must be ignored even
        # though it is 2x off, because the table and a full sweep disagree there by design.
        (tmp / "lineage_1.bench.json").write_text(json.dumps(sweep(9.0, {
            1: (0.500, 0.250, "memory"),
            7: (1.575, 0.250, "memory"),
            11948: (7.000, 1.750, "compute")})))
        (tmp / "lineage_2.bench.json").write_text(json.dumps(sweep(2.0, {
            1: (0.500, 0.250, "memory")})))
        (tmp / "lineages.json").write_text(json.dumps([
            {"file": "lineage_1.py", "full_score": 9.0, "origin": "bad"},
            {"file": "lineage_2.py", "full_score": 2.0, "origin": "clean"}]))

        swarm.audit_recorded_sweeps(tmp / "node.json", tmp)
        after = {e["file"]: e["full_score"] for e in
                 json.loads((tmp / "lineages.json").read_text())}

        # T=7 repaired to 0.525/0.250 = 2.1; T=1 stays 2.0; T=11948 keeps its live leg at
        # 7.000/1.750 = 4.0. Mean = (2.0 + 2.1 + 4.0) / 3.
        check("a blown memory-bound leg is corrected",
              abs(after["lineage_1.py"] - (2.0 + 2.1 + 4.0) / 3) < 1e-9, after)
        check("a compute-bound leg is left alone",
              abs(after["lineage_1.py"] - (2.0 + 2.1 + 2.0) / 3) > 1e-6, after)
        check("a clean sweep is not rewritten", after["lineage_2.py"] == 2.0, after)

    return 1 if _failures else 0


if __name__ == "__main__":
    rc = main()
    print("\n" + ("all swarm tests passed" if not _failures
                  else f"{len(_failures)} FAILED: {_failures}"))
    sys.exit(rc)
