"""Small scientific anchors; no campaign and no edits to the frozen engine."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import motor_controles as m


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "fuentes/flocking.py"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    checks = []
    for protocol in ("forzado", "maniobra"):
        for stat in ("full", "binary"):
            seed, phi = 1107, 2.0
            p = m.make_params(seed)
            legacy_p = m.Params(**{**asdict(p), "detector": "amplitud" if stat == "full" else "ciclo", "tau_det": 1.0})
            variant = {"stat": stat, "tau": 0.0 if stat == "full" else 1.0,
                       "gain": 1.0 if stat == "full" else 1.0 / p.tau_obs}
            started = perf_counter()
            legacy = m.engine.run_response(legacy_p, "WiSt", phi, 3.0, t_s=3.0,
                                           pulse=protocol == "maniobra", every_coh=10000)
            fresh = m.run_trial(variant, seed, phi, protocol=protocol, burn=3.0, t_R=3.0, diagnostics=False)
            delta = float(np.max(np.abs(legacy["proj"] - fresh["proj"])))
            assert delta < 1e-12, (stat, protocol, delta)
            checks.append({"check": "legacy_trajectory_equivalence", "stat": stat, "protocol": protocol,
                           "N": p.N, "burn": 3, "test_seconds": 3, "max_abs_projection_difference": delta,
                           "R_difference": abs(legacy["r"] - fresh["R"]), "elapsed": perf_counter() - started})
    variant = {"stat": "full", "tau": 1.0, "gain": 7.0, "intercept": -2.0}
    quiet = m.run_trial(variant, 441, 1.7, burn=0.5, t_R=0.5, diagnostics=False, params_overrides={"N": 40, "k": 7})
    diagnostic = m.run_trial(variant, 441, 1.7, burn=0.5, t_R=0.5, diagnostics=True, collect_scores=True, params_overrides={"N": 40, "k": 7})
    assert np.array_equal(quiet["proj"], diagnostic["proj"])
    sparse = m.run_trial(variant, 441, 1.7, burn=0.5, t_R=0.5, diagnostics=True,
                         diagnostics_every=10, params_overrides={"N": 40, "k": 7})
    assert np.array_equal(quiet["proj"], sparse["proj"])
    checks.append({"check": "diagnostics_do_not_change_trajectory", "pass": True})
    p = m.make_params(42, N=40, k=7)
    f = m.ClaseFlockControl(p, variant)
    f.step()
    idx, dist = f._neighbors()
    f.will[f.leader] = 0.82
    a = f.w_hat(idx, dist).copy()
    state = f._score_memory.copy()
    b = f.w_hat(idx, dist)
    assert np.array_equal(a, b) and np.array_equal(state, f._score_memory)
    own = idx == np.arange(p.N)[:, None]
    perceived = f._perceived_will(idx, dist)
    assert np.array_equal(perceived[own], f.will)
    checks.append({"check": "one_memory_update_per_step_and_exact_own_will", "pass": True})
    null = m.run_null_scores(19, burn=0.3, duration=0.3, variant=variant, sample_every=0.03, params_overrides={"N": 40, "k": 7})
    assert np.array_equal(null["posterior"], m.posterior_from_score(null["scores"], variant["gain"], variant["intercept"]))
    assert null["scores"].shape == (10, 40)
    assert null["edge_scores"].shape == (10, 40, 7)
    assert np.array_equal(null["edge_posterior"], m.posterior_from_score(null["edge_scores"], variant["gain"], variant["intercept"]))
    assert np.array_equal(null["edge_posterior"], null["edge_perceived"])
    checks.append({"check": "null_stream_shape_and_exact_score_logit_mapping", "pass": True})
    replay = m.run_replay_source(19, burn=0.3, duration=0.3, params_overrides={"N": 40, "k": 7})
    assert replay["full"].shape == (60, 40) and replay["evaluation_mask"].sum() == 30
    assert replay["edge_indices"].shape == (60, 40, 7)
    assert np.all(replay["edge_indices"] != np.arange(40)[None, :, None])
    for stat in ("full", "binary"):
        shadow = m.ClaseFlockControl(m.make_params(19, N=40, k=7), {"stat": stat, "tau": 1.0}, model="ViSt")
        values = []
        for _ in range(60):
            ix, _ = shadow._neighbors()
            values.append(shadow._llr_kinematic(ix).copy())
            shadow.step()
        initial = 0 if stat == "full" else replay["binary_initial_score"]
        offline = m.ema_scores(replay[stat], 1.0, replay["dt"], initial=initial)
        delta = float(np.max(np.abs(offline - values)))
        assert delta < 1e-12
        checks.append({"check": "offline_replay_EMA_matches_shadow_control", "stat": stat, "max_abs_score_difference": delta})
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    assert before == after
    result = {"pass": True, "checks": checks, "frozen_engine_sha256_before": before,
              "frozen_engine_sha256_after": after, "n_checks": len(checks)}
    target = root / "piloto/motor_controles_anclas.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
