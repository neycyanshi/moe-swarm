#!/usr/bin/env python3
"""
Break an nsys capture of scripts/profile_small_t.py into where each call's time goes.

    nsys export -t sqlite -o /tmp/prof/small_t.sqlite /tmp/prof/small_t.nsys-rep   # if needed
    python scripts/nsys_breakdown.py /tmp/prof/small_t.sqlite

Per NVTX range (one per seq_len), every GPU kernel is attributed to the CPU launch that
caused it via CUPTI's correlation id -- a CUDA graph replay is one launch owning all of its
kernels -- so one launch is one call of `kernel()`. For each call:

    kernels   sum of kernel durations
    inside    time between the call's first kernel starting and its last ending, not
              spent in a kernel: dependency bubbles and launch latency inside the call
    between   idle GPU time from this call's last kernel to the next call's first --
              host-side cost the GPU waits on
"""

import sqlite3
import statistics
import sys
from collections import defaultdict


def main(path):
    db = sqlite3.connect(path)
    q = lambda s, *a: db.execute(s, a).fetchall()

    ranges = q("SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'T=%' ORDER BY start")
    names = dict(q("SELECT id, value FROM StringIds"))
    launches = q("SELECT start, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start")
    kernels = defaultdict(list)
    for s, e, corr, nid in q("SELECT start, end, correlationId, shortName "
                             "FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        kernels[corr].append((s, e, names.get(nid, "?")))

    for text, r0, r1 in ranges:
        calls = [sorted(kernels[c]) for t, c in launches if r0 <= t <= r1 and kernels.get(c)]
        if not calls:
            print(f"{text}: no kernels attributed"); continue
        per_name = defaultdict(list)
        kern, inside, between = [], [], []
        for i, ks in enumerate(calls):
            k = sum(e - s for s, e, _ in ks)
            kern.append(k)
            inside.append((ks[-1][1] - ks[0][0]) - k)
            if i + 1 < len(calls):
                between.append(max(0, calls[i + 1][0][0] - ks[-1][1]))
            for s, e, n in ks:
                per_name[n].append(e - s)
        n = len(calls)
        us = lambda v: statistics.median(v) / 1e3
        total = us(kern) + us(inside) + (us(between) if between else 0)
        print(f"\n== {text}  ({n} calls, {len(calls[0])} kernels per call)  "
              f"~{total:.1f} us per call")
        print(f"   kernels {us(kern):7.1f} us   inside-call gaps {us(inside):6.1f} us   "
              f"between-call gaps {us(between) if between else 0:6.1f} us")
        for name, ds in sorted(per_name.items(), key=lambda kv: -sum(kv[1])):
            print(f"   {sum(ds) / n / 1e3:7.1f} us  x{len(ds) // n:<2} {name[:72]}")


if __name__ == "__main__":
    main(sys.argv[1])
