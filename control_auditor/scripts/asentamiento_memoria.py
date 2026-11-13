"""Predeclared tau=4 null-only burn-in sensitivity. No response trials.

24 runs: four fixed seeds x three gains x two burn durations. The decision rule
uses absolute differences between equally weighted replicate means. Arrays remain
inside workers; only compact per-replicate metrics are retained.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import motor_controles as m

ROOT = Path(__file__).resolve().parents[1]
PLAN = {
    "date": "2026-11-12",
    "question": "Sensitivity of tau=4 closed-loop null attention to burn12 vs24",
    "stat": "full", "tau": 4.0, "intercept": m.DEFAULT_INTERCEPT,
    "seeds": [900000, 900001, 900002, 900003],
    "gains": [1.0, 1.0 / 0.083, 48.0], "burns": [12.0, 24.0],
    "duration": 12.0, "sample_every": 0.1, "N": 250, "k": 15,
    "noise": float(np.sqrt(2)), "workers": 4,
    "estimator": "directed nonself recipient-sender edges, equal replicate weighting",
    "decision_rule": {
        "difference_direction": "burn24 minus burn12",
        "use_burn24_if_absolute_mean_edge_FPR_difference_gt": 0.01,
        "or_absolute_mean_edge_weight_difference_gt": 0.015,
        "scope": "flag each affected gain at tau4; if common block required use24 for tau4 and its comparison references",
    },
    "limitations": [
        "four independent seeds only; no test of full stationarity",
        "same-seed evaluation segments are12-24s and24-36s, so both memory transient and trajectory variation contribute",
        "edge samples within a replicate are not independent replicates",
        "no response trial or response-derived tuning is permitted in this check",
    ],
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def work(job):
    seed, gain, burn = job
    variant = {"stat": "full", "tau": 4.0, "gain": gain, "intercept": m.DEFAULT_INTERCEPT}
    r = m.run_null_scores(seed, burn=burn, duration=12, variant=variant, sample_every=0.1)
    return {"seed": seed, "gain": gain, "burn": burn, "duration": 12.0,
            "samples_time": len(r["times"]), "samples_nonself_edges": int(r["edge_posterior"].size),
            "edge_FPR": r["edge_fpr_p_gt_half"], "edge_mean_weight": r["edge_mean_perceived"],
            "edge_mean_posterior": r["edge_mean_posterior"],
            "edge_FPR_perceived": r["edge_perceived_fpr_p_gt_half"],
            "emitter_FPR": r["fpr_p_gt_half"], "emitter_mean_posterior": r["mean_posterior"],
            "seconds": r["seconds"]}


def main():
    started = perf_counter()
    source_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in (Path(m.__file__), ROOT / "fuentes/flocking.py", Path(__file__))}
    plan_path = ROOT / "piloto/asentamiento_memoria_plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        assert old["plan"] == PLAN and old["source_hashes"] == source_hashes, "Plan or source changed: do not silently overwrite predeclaration"
    else:
        plan_path.write_text(json.dumps({"plan": PLAN, "source_hashes": source_hashes}, indent=2) + "\n")
    jobs = [(seed, gain, burn) for seed in PLAN["seeds"] for gain in PLAN["gains"] for burn in PLAN["burns"]]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(work, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"completed {len(rows)}/24 seed={row['seed']} gain={row['gain']:.6g} burn={row['burn']:g} FPR={row['edge_FPR']:.6f} mean={row['edge_mean_weight']:.6f}", flush=True)
    rows.sort(key=lambda r: (r["gain"], r["seed"], r["burn"]))
    summaries = []
    for gain in PLAN["gains"]:
        by_burn = {}
        for burn in PLAN["burns"]:
            subset = [r for r in rows if r["gain"] == gain and r["burn"] == burn]
            assert len(subset) == 4
            by_burn[str(int(burn))] = {
                "replicates": 4, "edge_FPR_mean": float(np.mean([r["edge_FPR"] for r in subset])),
                "edge_weight_mean": float(np.mean([r["edge_mean_weight"] for r in subset])),
                "edge_FPR_between_seed_sd": float(np.std([r["edge_FPR"] for r in subset], ddof=1)),
                "edge_weight_between_seed_sd": float(np.std([r["edge_mean_weight"] for r in subset], ddof=1)),
            }
        deltas = []
        for seed in PLAN["seeds"]:
            a, b = [next(r for r in rows if r["gain"] == gain and r["seed"] == seed and r["burn"] == burn) for burn in (12, 24)]
            deltas.append({"seed": seed, "edge_FPR_delta24_minus12": b["edge_FPR"] - a["edge_FPR"],
                           "edge_weight_delta24_minus12": b["edge_mean_weight"] - a["edge_mean_weight"]})
        df = by_burn["24"]["edge_FPR_mean"] - by_burn["12"]["edge_FPR_mean"]
        dm = by_burn["24"]["edge_weight_mean"] - by_burn["12"]["edge_weight_mean"]
        flag_f, flag_m = abs(df) > 0.01, abs(dm) > 0.015
        summaries.append({"gain": gain, "by_burn": by_burn, "paired_seed_differences": deltas,
                          "edge_FPR_delta24_minus12": df, "edge_weight_delta24_minus12": dm,
                          "FPR_threshold_exceeded": flag_f, "weight_threshold_exceeded": flag_m,
                          "use_burn24": bool(flag_f or flag_m),
                          "recommended_burn": 24 if flag_f or flag_m else 12})
    after = {str(p.relative_to(ROOT)): file_hash(p) for p in (Path(m.__file__), ROOT / "fuentes/flocking.py", Path(__file__))}
    assert source_hashes == after, "Source changed while null burn sensitivity was running"
    result = {"plan": PLAN, "source_hashes": source_hashes, "sources_unchanged": True,
              "n_runs": len(rows), "runs": rows, "summary_by_gain": summaries,
              "maximum_absolute_FPR_difference": max(abs(r["edge_FPR_delta24_minus12"]) for r in summaries),
              "maximum_absolute_mean_weight_difference": max(abs(r["edge_weight_delta24_minus12"]) for r in summaries),
              "use_common_burn24_for_tau4_block_and_references": any(r["use_burn24"] for r in summaries),
              "seconds_wall": perf_counter() - started, "seconds_sum_runs": sum(r["seconds"] for r in rows),
              "runtime": {"python": sys.version, "numpy": np.__version__}}
    out = ROOT / "piloto/asentamiento_memoria.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"maximum_absolute_FPR_difference": result["maximum_absolute_FPR_difference"],
                      "maximum_absolute_mean_weight_difference": result["maximum_absolute_mean_weight_difference"],
                      "use_common_burn24_for_tau4_block_and_references": result["use_common_burn24_for_tau4_block_and_references"],
                      "seconds_wall": result["seconds_wall"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
