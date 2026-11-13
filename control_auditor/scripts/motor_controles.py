"""Frozen-engine controls separating statistic, temporal memory, gain and prior.

The raw score has no automatic effective-sample multiplier. ``intercept`` includes
the log prior odds. Sources are loaded from ../fuentes; no global monkeypatching.
Streams are evaluated immediately before the action that consumes their score.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
_NAME = "prioridad3_flocking_frozen"
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, ROOT / "fuentes/flocking.py")
    engine = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = engine
    spec.loader.exec_module(engine)
else:
    engine = sys.modules[_NAME]
Params, Flock = engine.Params, engine.Flock
DEFAULT_PRIOR = 0.173
DEFAULT_INTERCEPT = float(np.log(DEFAULT_PRIOR / (1 - DEFAULT_PRIOR)))


def make_params(seed, N=250, k=15, noise=np.sqrt(2.0), tauW=4.0, **overrides):
    """Reference parameters, with explicit seed, independent alpha and memory."""
    values = dict(N=int(N), k=int(k), sigma_xi=float(noise), tau_W=float(tauW),
                  seed=int(seed), sigma_theta=0.361, nu=2.8, sigma_W=0.811,
                  pi0=DEFAULT_PRIOR, tau_obs=0.083, will_mode="bayes",
                  detector="amplitud", lambda_fp=0.0)
    values.update(overrides)
    return Params(**values)


def normalize_variant(variant=None, p=None):
    v = dict(variant or {})
    v["stat"] = v.get("stat", "full")
    v["tau"] = float(v.get("tau", v.get("tau_mem", 0.0)))
    v["gain"] = float(v.get("gain", v.get("alpha", 1.0)))
    prior = p.pi0 if p is not None else DEFAULT_PRIOR
    v["intercept"] = float(v.get("intercept", v.get("b", np.log(prior / (1 - prior)))))
    if v["stat"] not in ("full", "binary", "oracle", "CoLd", "ViSt"):
        raise ValueError(f"Unknown statistic/baseline {v['stat']!r}")
    if v["tau"] < 0 or v["gain"] < 0 or not all(np.isfinite(v[x]) for x in ("tau", "gain", "intercept")):
        raise ValueError("tau and gain must be finite and nonnegative; intercept finite")
    return v


def likelihood_constants(p):
    c = engine._t_scale_from_mad(p.nu)
    s0, s1 = p.sigma_theta, float(np.hypot(p.sigma_theta, p.sigma_W))
    t0, t1 = s0 * c, s1 * c
    threshold = p.c_det * p.sigma_theta
    q0 = float(np.clip(2 * student_t.sf(threshold / t0, p.nu), 1e-6, 1 - 1e-6))
    q1 = float(np.clip(2 * student_t.sf(threshold / t1, p.nu), 1e-6, 1 - 1e-6))
    return {"t0": t0, "t1": t1, "q0": q0, "q1": q1,
            "hit_llr": float(np.log(q1 / q0)),
            "miss_llr": float(np.log((1 - q1) / (1 - q0))), "threshold": threshold}


def instantaneous_scores(eps, p, constants=None):
    """Exact Student amplitude LLR and one-observation binary LLR."""
    c = constants or likelihood_constants(p)
    t0, t1, nu = c["t0"], c["t1"], p.nu
    full = np.log(t0 / t1) + 0.5 * (nu + 1) * (
        np.log1p(eps ** 2 / (nu * t0 ** 2)) - np.log1p(eps ** 2 / (nu * t1 ** 2)))
    exceedance = np.abs(eps) > c["threshold"]
    binary = exceedance * c["hit_llr"] + (1 - exceedance) * c["miss_llr"]
    return full, binary, exceedance


def posterior_from_score(score, gain=1.0, intercept=DEFAULT_INTERCEPT):
    logit = intercept + gain * np.asarray(score)
    return 1 / (1 + np.exp(-np.clip(logit, -60, 60)))


def ema_scores(stream, tau, dt, initial=0.0):
    """Filter a full-resolution stream before subsampling; no gain multiplier.

    For binary use initial=q0*hit_llr+(1-q0)*miss_llr. Full starts at0,
    matching the frozen sequential detector. Retain the entire burn-in stream.
    """
    stream = np.asarray(stream, float)
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    if tau == 0:
        return stream.copy()
    a = np.exp(-dt / tau)
    state = np.broadcast_to(np.asarray(initial, float), stream.shape[1:]).copy()
    out = np.empty_like(stream)
    for i, x in enumerate(stream):
        state = a * state + (1 - a) * x
        out[i] = state
    return out


class ClaseFlockControl(Flock):
    """Emitter-level score memory, updated once per engine step.

    Signature: ClaseFlockControl(params, variant=None, model='WiSt'). Baselines
    should use ``new_flock`` or run_trial. Own will is restored by the untouched
    base _perceived_will. The binary memory uses the probability accumulator to
    reproduce the old cycle bit for bit when alpha=tau/tau_obs.
    """
    def __init__(self, p, variant=None, model="WiSt"):
        super().__init__(p, model)
        self.variant = normalize_variant(variant, p)
        if self.variant["stat"] not in ("full", "binary"):
            raise ValueError("ClaseFlockControl supports full/binary; route baselines through new_flock")
        if p.call_on or p.gain_on:
            raise ValueError("Acoustic and divisive-gain channels are outside this control experiment")
        self._constants = likelihood_constants(p)
        self._score_paso = -1
        self._score_memory = np.zeros(p.N)
        self._binary_memory = np.full(p.N, self._constants["q0"])
        self._memory_decay = np.exp(-p.dt / self.variant["tau"]) if self.variant["tau"] else 0.0
        self.last_score = np.zeros(p.N)
        self.last_instant_score = np.zeros(p.N)
        self.last_logit = np.full(p.N, self.variant["intercept"])
        self.last_posterior = posterior_from_score(self.last_score, self.variant["gain"], self.variant["intercept"])
        self.last_residual = np.zeros(p.N)
        self.last_indices = None
        self.last_perceived = None

    def _llr_kinematic(self, idx):
        if self._score_paso != self._paso:
            eps = self.omega - self.omega[idx].mean(axis=1)
            full, binary, exceedance = instantaneous_scores(eps, self.p, self._constants)
            self.last_instant_score = full if self.variant["stat"] == "full" else binary
            if self.variant["tau"] == 0:
                score = self.last_instant_score
            elif self.variant["stat"] == "full":
                a = self._memory_decay
                self._score_memory = a * self._score_memory + (1 - a) * full
                score = self._score_memory
            else:
                a = self._memory_decay
                self._binary_memory = a * self._binary_memory + (1 - a) * exceedance
                d = self._binary_memory
                score = d * self._constants["hit_llr"] + (1 - d) * self._constants["miss_llr"]
            self.last_score = score
            self.last_logit = self.variant["intercept"] + self.variant["gain"] * score
            self.last_posterior = 1 / (1 + np.exp(-np.clip(self.last_logit, -60, 60)))
            self.last_residual = eps
            self.last_indices = idx
            self._score_paso = self._paso
        return self.last_score

    def w_hat(self, idx, dist=None):
        self._llr_kinematic(idx)
        return self.last_posterior[idx]

    def _perceived_will(self, idx, dist):
        # Base restores own will exactly and retains the original channel logic.
        perceived = super()._perceived_will(idx, dist)
        self.last_indices, self.last_perceived = idx, perceived
        return perceived


FlockControl = ClaseFlockControl


def new_flock(p, variant=None, model="WiSt"):
    v = normalize_variant(variant, p)
    if v["stat"] in ("ViSt", "CoLd", "oracle"):
        pp = Params(**{**asdict(p), "will_mode": "oracle"})
        return Flock(pp, "WiSt" if v["stat"] == "oracle" else v["stat"]), v
    if model not in ("WiSt", "WiSp"):
        raise ValueError("Controlled observers require WiSt/WiSp; use baseline stat for ViSt/CoLd")
    return ClaseFlockControl(p, v, model), v


def _diagnostic_row(f, informed=True):
    h, leader = f.last_posterior, f.leader
    others = np.arange(f.p.N) != leader if informed else np.ones(f.p.N, bool)
    out = {"leader_posterior": float(h[leader]) if informed else np.nan,
           "leader_posterior_above_half": float(h[leader] > 0.5) if informed else np.nan,
           "nonleader_posterior_mean": float(h[others].mean()),
           "nonleader_fpr_p_gt_half": float((h[others] > 0.5).mean()),
           "nonleader_logit_mean": float(f.last_logit[others].mean())}
    idx, perceived = f.last_indices, f.last_perceived
    own = idx == np.arange(f.p.N)[:, None]
    nonself = ~own
    on_leader = (idx == leader) & nonself if informed else np.zeros_like(nonself)
    total = float(perceived[nonself].sum())
    lead = float(perceived[on_leader].sum())
    out.update(total_nonself_weight_per_receiver=total / f.p.N,
               nonself_pair_weight_mean=float(perceived[nonself].mean()),
               nonself_pair_fpr_p_gt_half=float((perceived[nonself] > 0.5).mean()),
               leader_weight_per_receiver=lead / f.p.N if informed else np.nan,
               nonleader_weight_per_receiver=(total - lead) / f.p.N,
               leader_attention_fraction=lead / total if informed and total > 0 else np.nan,
               nonleader_pair_weight_mean=float(perceived[nonself & ~on_leader].mean()))
    return out


def edge_view(values, indices):
    """Directed recipient->sender edges, excluding each recipient's self entry."""
    own = indices == np.arange(len(indices))[:, None]
    return np.asarray(values)[indices][~own].reshape(len(indices), -1)


def _edge_perceived(f):
    own = f.last_indices == np.arange(f.p.N)[:, None]
    return f.last_perceived[~own].reshape(f.p.N, -1)


def _mean_diagnostics(rows, mask=None):
    if not rows:
        return None
    if mask is not None:
        rows = [r for r, keep in zip(rows, mask) if keep]
    if not rows:
        return None
    out = {}
    for key in rows[0]:
        arr = np.asarray([r[key] for r in rows])
        out[key] = float(np.mean(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else None
    return out


def run_trial(variant, seed, phi, protocol="forzado", tauW=4.0, t_R=15.8,
              burn=24.0, diagnostics=True, params_overrides=None,
              collect_scores=False, score_every=0.1, null_last=4.0, model="WiSt",
              diagnostics_every=1):
    """Direct analogue of frozen run_response, with independent score gain.

    variant=None means the oracle baseline here. Other baselines:
    {'stat':'ViSt'}, {'stat':'CoLd'}, {'stat':'oracle'}. Closed-loop observer
    diagnostics use cached pre-action scores and actual nonself perceived weights;
    they never call the detector extra times. nulo has no informed/forced agent.
    """
    if protocol not in ("forzado", "maniobra", "nulo"):
        raise ValueError("protocol must be forzado, maniobra or nulo")
    p = make_params(seed, tauW=tauW, **(params_overrides or {}))
    f, v = new_flock(p, variant or {"stat": "oracle"}, model)
    started = perf_counter()
    for _ in range(int(round(burn / p.dt))):
        f.step()
    target = np.array([np.cos(phi), np.sin(phi)])
    informed = protocol != "nulo"
    if informed:
        f.forced = target
        f.will[f.leader] = 1.0
        f.d_des[f.leader] = target
    count = int(round(t_R / p.dt))
    if count < 1:
        raise ValueError("t_R must be at least one time step")
    proj, polarization = np.empty(count), np.empty(count)
    rows, row_steps, stream_score, stream_post, stream_times = [], [], [], [], []
    edge_scores, edge_post, edge_perceived = [], [], []
    every = max(1, int(round(score_every / p.dt)))
    diagnostic_stride = max(1, int(diagnostics_every))
    released, release_time = False, None
    for step in range(count):
        f.step()
        if protocol == "maniobra" and not released:
            vel = f.vel[f.leader]
            alignment = float(vel @ target) / max(float(np.linalg.norm(vel)), 1e-9)
            if alignment > np.cos(np.deg2rad(10)) or step * p.dt > 2.5:
                f.forced, released, release_time = None, True, step * p.dt
        if informed and not released:
            f.will[f.leader] = 1.0
        proj[step] = f.V() @ target / p.v0
        polarization[step] = f.polarization()
        if isinstance(f, ClaseFlockControl):
            if diagnostics and step % diagnostic_stride == 0:
                rows.append(_diagnostic_row(f, informed))
                row_steps.append(step)
            if collect_scores and step % every == 0:
                stream_score.append(f.last_score.copy())
                stream_post.append(f.last_posterior.copy())
                stream_times.append(step * p.dt)
                edge_scores.append(edge_view(f.last_score, f.last_indices).copy())
                edge_post.append(edge_view(f.last_posterior, f.last_indices).copy())
                edge_perceived.append(_edge_perceived(f).copy())
    first = min(count, int(round(4 / p.dt)))
    tail_start = max(0, count - int(round(null_last / p.dt)))
    out = {"R": float(proj.mean()), "r": float(proj.mean()), "R_first4": float(proj[:first].mean()),
           "R_tail": float(proj[tail_start:].mean()), "proj": proj,
           "pol": float(polarization.mean()), "leader": f.leader, "release_time": release_time,
           "variant": v, "protocol": protocol, "seed": int(seed), "phi": float(phi),
           "params": asdict(p), "burn": burn, "t_R": t_R, "seconds": perf_counter() - started,
           "diagnostics": _mean_diagnostics(rows),
           "diagnostics_first4": _mean_diagnostics(rows, [s < first for s in row_steps]),
           "diagnostics_tail": _mean_diagnostics(rows, [s >= tail_start for s in row_steps]),
           "diagnostics_every_steps": diagnostic_stride, "diagnostics_sample_every": diagnostic_stride * p.dt,
           "diagnostic_alignment": "pre-action cached posterior; actual perceived neighbour weights; projected velocity after action"}
    if collect_scores:
        out.update(scores=np.asarray(stream_score), posterior=np.asarray(stream_post), times=np.asarray(stream_times),
                   edge_scores=np.asarray(edge_scores), edge_posterior=np.asarray(edge_post),
                   edge_perceived=np.asarray(edge_perceived), edge_estimator="directed_nonself_recipient_sender_pairs")
    if out["diagnostics"] is not None:
        fraction = out["diagnostics"]["leader_posterior_above_half"]
        out["leader_posterior_time_above_half"] = None if fraction is None else fraction * count * p.dt
    return out


def run_null_scores(seed, burn=24.0, duration=12.0, variant=None,
                    sample_every=0.1, params_overrides=None, model="WiSt"):
    """Closed-loop all-uninformed run; returns raw score before gain/prior.

    Arrays shape (sample_times, emitters). No artificial informed labels are
    created. Burn-in is simulated in the same closed loop, not on a replay.
    """
    p = make_params(seed, **(params_overrides or {}))
    if model not in ("WiSt", "WiSp"):
        raise ValueError("Closed-loop null scores require WiSt/WiSp; use run_replay_source for ViSt")
    f = ClaseFlockControl(p, variant, model)
    started = perf_counter()
    for _ in range(int(round(burn / p.dt))):
        f.step()
    every = max(1, int(round(sample_every / p.dt)))
    scores, posterior, logits, times = [], [], [], []
    edge_scores, edge_post, edge_perceived = [], [], []
    for step in range(int(round(duration / p.dt))):
        f.step()
        if step % every == 0:
            scores.append(f.last_score.copy())
            posterior.append(f.last_posterior.copy())
            logits.append(f.last_logit.copy())
            times.append(step * p.dt)
            edge_scores.append(edge_view(f.last_score, f.last_indices).copy())
            edge_post.append(edge_view(f.last_posterior, f.last_indices).copy())
            edge_perceived.append(_edge_perceived(f).copy())
    score = np.asarray(scores)
    post = np.asarray(posterior)
    edge_s, edge_p, edge_w = np.asarray(edge_scores), np.asarray(edge_post), np.asarray(edge_perceived)
    return {"scores": score, "posterior": post, "logit": np.asarray(logits), "times": np.asarray(times),
            "variant": f.variant, "params": asdict(p), "seed": int(seed), "burn": burn, "duration": duration,
            "sample_every": every * p.dt, "all_uninformed": True,
            "mean_posterior": float(post.mean()), "fpr_p_gt_half": float((post > 0.5).mean()),
            "edge_scores": edge_s, "edge_posterior": edge_p, "edge_perceived": edge_w,
            "edge_mean_posterior": float(edge_p.mean()), "edge_fpr_p_gt_half": float((edge_p > 0.5).mean()),
            "edge_mean_perceived": float(edge_w.mean()), "edge_perceived_fpr_p_gt_half": float((edge_w > 0.5).mean()),
            "edge_estimator": "directed_nonself_recipient_sender_pairs",
            "seconds": perf_counter() - started, "score_alignment": "pre-action score at times relative to end of burn-in"}


def run_replay_source(seed, burn=24.0, duration=12.0, sample_every=None, params_overrides=None,
                      record_edge_indices=True):
    """One common ViSt source for offline memory/prior/gain calibration.

    Streams INCLUDE burn-in. Default records every engine step: filter those
    streams before using evaluation_mask and subsampling. Requesting a larger
    sample_every before filtering changes the discretized memory approximation.
    """
    p = make_params(seed, **(params_overrides or {}))
    f = Flock(p, "ViSt")
    constants = likelihood_constants(p)
    every = 1 if sample_every is None else max(1, int(round(sample_every / p.dt)))
    full, binary, residual, times, edge_indices = [], [], [], [], []
    started = perf_counter()
    for step in range(int(round((burn + duration) / p.dt))):
        if step % every == 0:
            idx, _ = f._neighbors()
            eps = f.omega - f.omega[idx].mean(axis=1)
            a, b, _ = instantaneous_scores(eps, p, constants)
            full.append(a)
            binary.append(b)
            residual.append(eps)
            times.append(step * p.dt)
            if record_edge_indices:
                own = idx == np.arange(p.N)[:, None]
                edge_indices.append(idx[~own].reshape(p.N, -1).astype(np.int32))
        f.step()
    times = np.asarray(times)
    return {"full": np.asarray(full), "binary": np.asarray(binary), "residual": np.asarray(residual),
            "times": times, "evaluation_mask": times >= burn - 1e-10,
            "dt": every * p.dt, "engine_dt": p.dt, "q0": constants["q0"], "q1": constants["q1"],
            "binary_initial_score": constants["q0"] * constants["hit_llr"] + (1 - constants["q0"]) * constants["miss_llr"],
            "full_initial_score": 0.0, "params": asdict(p), "seed": int(seed), "burn": burn,
            "duration": duration, "seconds": perf_counter() - started,
            "edge_indices": np.asarray(edge_indices) if record_edge_indices else None,
            "edge_estimator": "directed_nonself_recipient_sender_pairs",
            "warning": "Filter at engine_dt before subsampling" if every > 1 else None}
