# -*- coding: utf-8 -*-
"""
data_io.py
----------
CSV loader, fitting helpers, and multi-dataset global-fit export utilities.

This module preserves the original single-dataset fitting entry points and
adds a new multi-dataset S-only global fitting path built around the same
NdSb3TM simulator.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from time import perf_counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from config import (
    LEGACY_PARAM_ALIASES,
    MULTI_FIT_DEFAULT_GLOBAL_KEYS,
    MULTI_FIT_DEFAULT_LOCAL_KEYS,
    MULTI_FIT_DEFAULT_OBSERVABLE_MODE,
    default_params,
    normalize_params_dict,
)
from physics_engine import DebyeCl
from solver import NdSb3TM


POSITIVE_FIT_KEYS = {
    # Canonical base couplings
    "G_el0", "G_sl0", "G_es0",

    # sinks / time constants
    "tau_l_sink", "tau_e_sink", "tau_s_sink",
    "tau_m0", "tau_m_crit_amp", "tau_m_max",
    "eps_crit",

    # pump / scales
    "S_scale", "fluence_multiplier", "pulse_width",

    # electron thermo
    "gamma_PM_molar", "gap0_meV",

    # eta / OP
    "Gamma_eta", "eta_dT",

    # positive coupling-shape params
    "G_es_m_power",
    "G_sl_TR_boost", "G_sl_TR_w",
    "G_sl_TN_boost", "G_sl_TN_w",

    # spin-sector scales
    "A_sw_2q", "A_sw_1q",
    "J_renorm", "J_2q_scale", "J_1q_scale",
    "S_eff", "mag_gap_meV", "Cs_scale",

    # peaks
    "lambda_amp", "lambda_w",
    "latent_amp", "latent_w",

    # ARPES mapping positive params
    "S_amp", "S_power",
    "A_obs",
    "K_dk",

}

LOCAL_KEY_BOUNDS = {
    "dt_local": (-1e-12, 1e-12),
    "A_obs": (1e-4, 8e-1),
    "B_obs": (-2e-1, 8e-2),
}

VARPRO_READOUT_BOUNDS = {
    "A_obs": (1e-4, 8e-1),
    "B_obs": (-2e-1, 8e-2),
}

USE_VARPRO_READOUT = True
NEG_DELAY_WEIGHT_ENABLE = False
NEG_DELAY_THRESHOLD_PS = -1.0
NEG_DELAY_WEIGHT = 0.35


def _build_time_weights_from_t(t_sec):
    """
    Return pointwise weights for residual/readout.
    Default:
      - t < NEG_DELAY_THRESHOLD_PS ps  -> NEG_DELAY_WEIGHT
      - else                           -> 1.0
    """
    t_sec = np.asarray(t_sec, dtype=float)
    weights = np.ones_like(t_sec, dtype=float)
    if not NEG_DELAY_WEIGHT_ENABLE:
        return weights

    threshold_sec = NEG_DELAY_THRESHOLD_PS * 1e-12
    mask = t_sec < threshold_sec
    weights[mask] = float(NEG_DELAY_WEIGHT)
    return np.clip(weights, 0.0, np.inf)


# ============================================================
# CSV loader (auto-detect columns; supports Te/Ts/Tl/S)
# ============================================================
def load_csv_auto(path: str):
    arr = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")
    if arr is None or arr.dtype.names is None:
        raise ValueError("CSV must have a header row (names=True).")

    names = list(arr.dtype.names)
    ln = [n.strip().lower() for n in names]
    s_ln = set(ln)

    def find_col(candidates):
        for k in candidates:
            if k in s_ln:
                return names[ln.index(k)]
        return None

    t_col = find_col(["tps", "t_ps", "time_ps", "t(ps)", "time(ps)", "time", "t"])
    if t_col is None:
        raise ValueError(f"Cannot find time column. Available columns: {names}")
    t_raw = np.array(arr[t_col], dtype=float)

    te_col = find_col(["tek", "te_k", "te", "te(k)", "temp_e", "electron_temp"])
    ts_col = find_col(["tsk", "ts_k", "ts", "ts(k)", "temp_s", "spin_temp", "tspin"])
    tl_col = find_col(["tlk", "tl_k", "tl", "tl(k)", "temp_l", "lattice_temp"])

    Te = np.array(arr[te_col], dtype=float) if te_col is not None else None
    Ts = np.array(arr[ts_col], dtype=float) if ts_col is not None else None
    Tl = np.array(arr[tl_col], dtype=float) if tl_col is not None else None

    s_col = find_col(["s", "sw", "spec", "weight", "spectral_weight", "intensity", "amp"])
    S = np.array(arr[s_col], dtype=float) if s_col is not None else None

    t_max = np.nanmax(np.abs(t_raw))
    if t_max > 1e-6:
        t_sec = t_raw * 1e-12
        t_unit = "ps"
    else:
        t_sec = t_raw
        t_unit = "s"

    mask = np.isfinite(t_sec)
    for v in [Te, Ts, Tl, S]:
        if v is not None:
            mask &= np.isfinite(v)

    t_sec = t_sec[mask]

    def filt(v):
        return v[mask] if v is not None else None

    Te, Ts, Tl, S = map(filt, [Te, Ts, Tl, S])

    idx = np.argsort(t_sec)
    t_sec = t_sec[idx]

    def sortv(v):
        return v[idx] if v is not None else None

    Te, Ts, Tl, S = map(sortv, [Te, Ts, Tl, S])

    t_u, inv = np.unique(t_sec, return_inverse=True)

    def avg(vals):
        out = np.zeros_like(t_u)
        cnt = np.zeros_like(t_u)
        for i, k in enumerate(inv):
            out[k] += vals[i]
            cnt[k] += 1
        return out / np.maximum(cnt, 1)

    Te_u = avg(Te) if Te is not None else None
    Ts_u = avg(Ts) if Ts is not None else None
    Tl_u = avg(Tl) if Tl is not None else None
    S_u = avg(S) if S is not None else None

    return t_u, Te_u, Ts_u, Tl_u, S_u, names, t_unit


def preprocess_signal_baseline(t, S):
    """
    Simple baseline subtraction for S-like traces.

    If at least 2 points satisfy t < 0, baseline = mean(S[t < 0]).
    Otherwise baseline = mean(first min(3, len(S)) points).

    Returns:
        S_sub, baseline_value, baseline_npts, baseline_method
    """
    t_arr = np.asarray(t, dtype=float)
    s_arr = np.asarray(S, dtype=float)
    if t_arr.shape != s_arr.shape:
        raise ValueError("preprocess_signal_baseline: t and S must have the same shape.")
    if t_arr.ndim != 1:
        raise ValueError("preprocess_signal_baseline: t and S must be 1D arrays.")

    neg_mask = t_arr < 0.0
    if int(np.count_nonzero(neg_mask)) >= 2:
        baseline_pts = s_arr[neg_mask]
        baseline_method = "negative_delay_mean"
    else:
        baseline_pts = s_arr[: min(3, s_arr.size)]
        baseline_method = "early_points_mean"

    baseline_value = float(np.mean(baseline_pts)) if baseline_pts.size > 0 else 0.0
    baseline_npts = int(baseline_pts.size)
    s_sub = np.asarray(s_arr - baseline_value, dtype=float)
    return s_sub, baseline_value, baseline_npts, baseline_method


def load_s_dataset_csv(path: str | Path) -> dict:
    t, Te, Ts, Tl, S, _, _ = load_csv_auto(str(path))
    if S is None:
        raise ValueError(f"{Path(path).name} has no S column.")

    S_sub, baseline_value, baseline_npts, baseline_method = preprocess_signal_baseline(t, S)
    return {
        "name": Path(path).name,
        "path": str(path),
        "t": t,
        "Te": Te,
        "Ts": Ts,
        "Tl": Tl,
        "S_raw": np.asarray(S, dtype=float),
        "S": S_sub,
        "baseline_value": baseline_value,
        "baseline_npts": baseline_npts,
        "baseline_method": baseline_method,
        "fluence_ratio": parse_fluence_ratio_from_name(str(path)),
    }


def load_s_dataset_csv_raw(path: str | Path) -> dict:
    t, Te, Ts, Tl, S, _, _ = load_csv_auto(str(path))
    if S is None:
        raise ValueError(f"{Path(path).name} has no S column.")

    s_raw = np.asarray(S, dtype=float)
    return {
        "name": Path(path).name,
        "path": str(path),
        "t": t,
        "Te": Te,
        "Ts": Ts,
        "Tl": Tl,
        "S_raw": s_raw,
        "S": s_raw,
        "baseline_value": 0.0,
        "baseline_npts": 0,
        "baseline_method": "none",
        "fluence_ratio": parse_fluence_ratio_from_name(str(path)),
    }


def load_dk_dataset_csv(path: str | Path, resolution_limit: float | None = None) -> dict:
    """Load a delta-k dataset CSV without S baseline subtraction."""
    arr = np.genfromtxt(str(path), delimiter=",", names=True, encoding="utf-8-sig")
    if arr is None or arr.dtype.names is None:
        raise ValueError(f"{Path(path).name} must have a header row.")

    names = list(arr.dtype.names)
    lowered = [n.strip().lower() for n in names]
    lowered_set = set(lowered)

    def _find_col(candidates):
        for key in candidates:
            if key in lowered_set:
                return names[lowered.index(key)]
        return None

    dk_col = _find_col([
        "delta_k", "deltak", "dk", "delta_k_ainv", "deltak_ainv",
        "k_split", "ksplit", "split_k",
    ])
    if dk_col is None:
        raise ValueError(
            f"{Path(path).name} is missing delta-k column. "
            "Expected one of: delta_k, deltak, dk, delta_k_Ainv, deltak_Ainv, "
            "k_split, ksplit, split_k."
        )
    resolved_col = _find_col(["resolved", "is_resolved", "dk_resolved", "split_resolved"])
    sigma_col = _find_col([
        "sigma_dk", "sigmadeltak12_k", "delta_k_err", "deltak_err",
        "dk_err", "err", "error", "sigma",
    ])

    t_col = _find_col(["tps", "t_ps", "time_ps", "t(ps)", "time(ps)", "time", "t"])
    if t_col is None:
        raise ValueError(f"{Path(path).name} has no recognizable time column.")

    te_col = _find_col(["tek", "te_k", "te", "te(k)", "temp_e", "electron_temp"])
    ts_col = _find_col(["tsk", "ts_k", "ts", "ts(k)", "temp_s", "spin_temp", "tspin"])
    tl_col = _find_col(["tlk", "tl_k", "tl", "tl(k)", "temp_l", "lattice_temp"])

    t_raw = np.asarray(arr[t_col], dtype=float)
    t_sec = t_raw * 1e-12 if np.nanmax(np.abs(t_raw)) > 1e-6 else t_raw
    delta_raw = np.asarray(arr[dk_col], dtype=float)
    sigma_raw = np.asarray(arr[sigma_col], dtype=float) if sigma_col is not None else None

    Te_raw = np.asarray(arr[te_col], dtype=float) if te_col is not None else None
    Ts_raw = np.asarray(arr[ts_col], dtype=float) if ts_col is not None else None
    Tl_raw = np.asarray(arr[tl_col], dtype=float) if tl_col is not None else None

    mask = np.isfinite(t_sec) & np.isfinite(delta_raw)
    if Te_raw is not None:
        mask &= np.isfinite(Te_raw)
    if Ts_raw is not None:
        mask &= np.isfinite(Ts_raw)
    if Tl_raw is not None:
        mask &= np.isfinite(Tl_raw)

    t_sec = t_sec[mask]
    delta_raw = delta_raw[mask]
    sigma_raw = sigma_raw[mask] if sigma_raw is not None else None
    Te_raw = Te_raw[mask] if Te_raw is not None else None
    Ts_raw = Ts_raw[mask] if Ts_raw is not None else None
    Tl_raw = Tl_raw[mask] if Tl_raw is not None else None

    idx = np.argsort(t_sec)
    t_sorted = t_sec[idx]
    delta_sorted = delta_raw[idx]
    Te_sorted = Te_raw[idx] if Te_raw is not None else None
    Ts_sorted = Ts_raw[idx] if Ts_raw is not None else None
    Tl_sorted = Tl_raw[idx] if Tl_raw is not None else None
    sigma_sorted = sigma_raw[idx] if sigma_raw is not None else None

    if resolved_col is not None:
        raw = np.asarray(arr[resolved_col])[mask][idx]
        if np.issubdtype(raw.dtype, np.number):
            resolved_sorted = np.asarray(raw, dtype=float) > 0.5
        else:
            resolved_sorted = np.array(
                [str(v).strip().lower() in {"1", "true", "t", "yes", "y"} for v in raw],
                dtype=bool,
            )
    else:
        if resolution_limit is None:
            resolution_limit = float(
                normalize_params_dict(default_params()).get("dk_resolution_limit", 0.003)
            )
        resolved_sorted = np.isfinite(delta_sorted) & (delta_sorted > float(resolution_limit))

    t_u, inv = np.unique(t_sorted, return_inverse=True)
    delta_u = np.zeros_like(t_u, dtype=float)
    cnt = np.zeros_like(t_u, dtype=float)
    resolved_u = np.zeros_like(t_u, dtype=bool)
    Te_u = np.zeros_like(t_u, dtype=float) if Te_sorted is not None else None
    Ts_u = np.zeros_like(t_u, dtype=float) if Ts_sorted is not None else None
    Tl_u = np.zeros_like(t_u, dtype=float) if Tl_sorted is not None else None
    sigma_sum_sq_u = np.zeros_like(t_u, dtype=float) if sigma_sorted is not None else None
    for i, g in enumerate(inv):
        delta_u[g] += delta_sorted[i]
        cnt[g] += 1.0
        resolved_u[g] = bool(resolved_u[g] or resolved_sorted[i])
        if Te_u is not None:
            Te_u[g] += Te_sorted[i]
        if Ts_u is not None:
            Ts_u[g] += Ts_sorted[i]
        if Tl_u is not None:
            Tl_u[g] += Tl_sorted[i]
        if sigma_sum_sq_u is not None:
            sig_i = sigma_sorted[i]
            if np.isfinite(sig_i):
                sigma_sum_sq_u[g] += sig_i ** 2
    delta_u = delta_u / np.maximum(cnt, 1.0)
    if Te_u is not None:
        Te_u = Te_u / np.maximum(cnt, 1.0)
    if Ts_u is not None:
        Ts_u = Ts_u / np.maximum(cnt, 1.0)
    if Tl_u is not None:
        Tl_u = Tl_u / np.maximum(cnt, 1.0)
    sigma_u = None
    if sigma_sum_sq_u is not None:
        sigma_u = np.sqrt(sigma_sum_sq_u) / np.maximum(cnt, 1.0)

    return {
        "name": Path(path).name,
        "path": str(path),
        "t": t_u,
        "Te": np.asarray(Te_u, dtype=float) if Te_u is not None else None,
        "Ts": np.asarray(Ts_u, dtype=float) if Ts_u is not None else None,
        "Tl": np.asarray(Tl_u, dtype=float) if Tl_u is not None else None,
        "delta_k": np.asarray(delta_u, dtype=float),
        "sigma_dk": np.asarray(sigma_u, dtype=float) if sigma_u is not None else None,
        "S": np.asarray(delta_u, dtype=float),
        "is_resolved": np.asarray(resolved_u, dtype=bool),
        "fluence_ratio": parse_fluence_ratio_from_name(str(path)),
    }


# ============================================================
# Bounds helper
# ============================================================
def _get_bounds_for_keys(keys):
    lb, ub = [], []

    for k in keys:
        if k in ["G_el", "G_el0"]:
            lb.append(1e13); ub.append(2e15)
        elif k in ["G_es", "G_es0"]:
            lb.append(1e14); ub.append(2e16)
        elif k in ["G_sl", "G_sl0"]:
            lb.append(1e12); ub.append(5e16)
        elif k == "G_el_Tpow":
            lb.append(-3.0); ub.append(3.0)
        elif k == "G_es_floor_frac":
            lb.append(0.0); ub.append(1.0)
        elif k == "G_es_m_power":
            lb.append(0.0); ub.append(6.0)
        elif k == "G_es_eta_coupling":
            lb.append(0.0); ub.append(5.0)
        elif k == "G_sl_TR_boost":
            lb.append(0.0); ub.append(10.0)
        elif k == "G_sl_TR_w":
            lb.append(0.05); ub.append(5.0)
        elif k == "G_sl_TN_boost":
            lb.append(0.0); ub.append(10.0)
        elif k == "G_sl_TN_w":
            lb.append(0.05); ub.append(5.0)
        elif k == "G_sl_eta_coupling":
            lb.append(0.0); ub.append(5.0)
        elif k == "tau_l_sink":
            lb.append(2e-12); ub.append(1e-8)
        elif k == "tau_e_sink":
            lb.append(1e-13); ub.append(5e-9)
        elif k == "tau_s_sink":
            lb.append(1e-12); ub.append(5e-8)
        elif k == "S_scale":
            lb.append(1e-7); ub.append(1.0)
        elif k == "fluence_multiplier":
            lb.append(0.1); ub.append(100.0)
        elif k == "pulse_width":
            lb.append(10e-15); ub.append(5e-12)
        elif k == "t0_pulse":
            lb.append(-2e-12); ub.append(2e-12)
        elif k == "dt0_ps":
            lb.append(-1.0); ub.append(0.5)
        elif k == "sigma_irf_ps":
            lb.append(0.02); ub.append(1.0)
        elif k == "alpha_dt_per_F":
            lb.append(-0.5); ub.append(0.5)
        elif k == "alpha_gap":
            lb.append(0.0); ub.append(2.0)
        elif k == "gap0_meV":
            lb.append(0.0); ub.append(200.0)
        elif k == "gap_eta_coupling":
            lb.append(0.0); ub.append(5.0)
        elif k == "gamma_PM_molar":
            lb.append(1e-4); ub.append(1e-1)
        elif k == "gamma_min_frac":
            lb.append(0.01); ub.append(1.0)
        elif k.endswith("_amp"):
            lb.append(0.0); ub.append(50.0)
        elif k.endswith("_w"):
            lb.append(0.05); ub.append(5.0)
        elif k in ["A_sw_2q", "A_sw_1q"]:
            lb.append(0.0); ub.append(1.0)
        elif k in ["J_renorm", "J_2q_scale", "J_1q_scale"]:
            lb.append(0.1); ub.append(10.0)
        elif k == "S_eff":
            lb.append(0.1); ub.append(10.0)
        elif k == "mag_gap_meV":
            lb.append(0.0); ub.append(20.0)
        elif k == "magnon_grid":
            lb.append(8); ub.append(80)
        elif k == "Cs_scale":
            lb.append(0.01); ub.append(100.0)
        elif k in ["cef_E1_meV", "cef_E2_meV"]:
            lb.append(0.0); ub.append(100.0)
        elif k in ["cef_g0", "cef_g1", "cef_g2"]:
            lb.append(1); ub.append(20)
        elif k == "nu":
            lb.append(0.2); ub.append(3.0)
        elif k == "eps_crit":
            lb.append(0.001); ub.append(0.5)
        elif k.startswith("tau_m"):
            lb.append(1e-13); ub.append(5e-9)
        elif k == "Gamma_eta":
            lb.append(1e10); ub.append(1e12)
        elif k == "Gamma_eta_low_frac":
            lb.append(1e-6); ub.append(1.0)
        elif k == "eta_dT":
            lb.append(0.01); ub.append(10.0)
        elif k == "eta_representation":
            lb.append(-np.inf); ub.append(np.inf)
        elif k in ["a_eta0", "b_eta", "c_eta"]:
            lb.append(0.0); ub.append(1e4)
        elif k == "g_m2eta2":
            lb.append(-1e3); ub.append(1e3)
        elif k == "eta_clip":
            lb.append(0.1); ub.append(5.0)
        elif k == "eta_sign":
            lb.append(-1.0); ub.append(1.0)
        elif k == "S_offset":
            lb.append(-2.0); ub.append(2.0)
        elif k == "S_amp":
            lb.append(0.0); ub.append(10.0)
        elif k == "S_power":
            lb.append(0.5); ub.append(6.0)
        elif k == "T_bath":
            lb.append(0.1); ub.append(300.0)
        elif k == "T_init_eff":
            lb.append(0.1); ub.append(400.0)
        elif k == "rep_rate_Hz":
            lb.append(0.0); ub.append(1e9)
        elif k == "preheat_max_dT":
            lb.append(0.0); ub.append(500.0)
        elif k == "TN":
            lb.append(0.1); ub.append(100.0)
        elif k == "TR":
            lb.append(0.1); ub.append(100.0)
        elif k == "ThetaD":
            lb.append(10.0); ub.append(1000.0)
        elif k == "A_obs":
            lb.append(1e-5); ub.append(8e-1)
        elif k == "B0_obs":
            lb.append(0.0); ub.append(4e-2)
        elif k == "B1_obs":
            lb.append(-1e-2); ub.append(1e-2)
        elif k == "K_dk":
            lb.append(1e-6); ub.append(1.0)
        elif k == "B_dk":
            lb.append(-0.2); ub.append(0.2)
        else:
            lb.append(-np.inf); ub.append(np.inf)

    return np.array(lb, float), np.array(ub, float)


def normalize_fit_keys(fit_keys):
    normalized = []
    for key in fit_keys:
        canonical = LEGACY_PARAM_ALIASES.get(key, key)
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _to_fit_x(key, value):
    value = float(value)
    if key in POSITIVE_FIT_KEYS:
        return np.log10(max(value, 1e-300))
    return value


def _from_fit_x(key, value):
    value = float(value)
    if key in POSITIVE_FIT_KEYS:
        return float(10.0 ** value)
    return value


def _pack_params(params, keys):
    arr = []
    for key in keys:
        if key not in params:
            raise KeyError(f"fit parameter '{key}' not found in parameter dict.")
        arr.append(_to_fit_x(key, params[key]))
    return np.array(arr, dtype=float)


def _unpack_params(params, keys, x):
    for key, value in zip(keys, x):
        params[key] = _from_fit_x(key, value)


def _build_fit_bounds(keys, bounds_lookup):
    lb_phys, ub_phys = bounds_lookup(keys)
    lb = np.array([_to_fit_x(k, v) for k, v in zip(keys, lb_phys)], float)
    ub = np.array([_to_fit_x(k, v) for k, v in zip(keys, ub_phys)], float)
    return lb, ub


def _get_multi_local_bounds(keys):
    lb, ub = [], []
    for key in keys:
        if key not in LOCAL_KEY_BOUNDS:
            raise KeyError(
                f"Unsupported local fit key '{key}'. Add an explicit bound in LOCAL_KEY_BOUNDS."
            )
        lo, hi = LOCAL_KEY_BOUNDS[key]
        lb.append(lo)
        ub.append(hi)
    return np.array(lb, dtype=float), np.array(ub, dtype=float)


# ============================================================
# Observable mapping helpers
# ============================================================
def _compute_chi2q(sim):
    eta_repr = str(sim.get("eta_representation", "")).strip().lower()
    phi = sim.get("phi")
    if eta_repr == "cos2phi" and phi is not None:
        phi = np.asarray(phi, dtype=float)
        return np.abs(np.sin(2.0 * phi))

    eta = np.asarray(sim["eta"], dtype=float)
    eta_clip = np.clip(eta, -1.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - eta_clip ** 2))


def _gaussian_irf_kernel(dt_ps, sigma_ps, half_width_sigma=5):
    sigma_ps = float(sigma_ps)
    if sigma_ps <= 0.0:
        return np.array([1.0], dtype=float)
    dt_ps = float(abs(dt_ps))
    if dt_ps <= 0.0 or not np.isfinite(dt_ps):
        dt_ps = sigma_ps / 5.0
    half_width_ps = float(max(half_width_sigma, 1.0) * sigma_ps)
    n_half = int(np.ceil(half_width_ps / dt_ps))
    x = np.arange(-n_half, n_half + 1, dtype=float) * dt_ps
    kernel = np.exp(-0.5 * (x / sigma_ps) ** 2)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0 or not np.isfinite(kernel_sum):
        return np.array([1.0], dtype=float)
    return kernel / kernel_sum


def _convolve_with_irf(y, t_ps, sigma_ps):
    y_arr = np.asarray(y, dtype=float)
    if float(sigma_ps) <= 0.0:
        return y_arr.copy()
    t_ps = np.asarray(t_ps, dtype=float)
    if y_arr.ndim != 1 or t_ps.ndim != 1 or y_arr.size != t_ps.size:
        raise ValueError("_convolve_with_irf expects 1D y/t_ps with the same length.")
    if y_arr.size < 3:
        return y_arr.copy()
    diffs = np.diff(t_ps)
    finite_diffs = diffs[np.isfinite(diffs)]
    if finite_diffs.size == 0:
        return y_arr.copy()
    dt_est = float(np.mean(np.abs(finite_diffs)))
    if not np.isfinite(dt_est) or dt_est <= 0.0:
        return y_arr.copy()
    kernel = _gaussian_irf_kernel(dt_est, float(sigma_ps))
    return np.convolve(y_arr, kernel, mode="same")


def solve_linear_readout_ab(y_obs, template_u, weights=None, bounds=None):
    y_obs = np.asarray(y_obs, dtype=float)
    template_u = np.asarray(template_u, dtype=float)
    if y_obs.shape != template_u.shape or y_obs.ndim != 1:
        raise ValueError("solve_linear_readout_ab expects y_obs/template_u as 1D arrays with same shape.")

    X = np.column_stack([template_u, np.ones_like(template_u)])
    y = y_obs
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != y_obs.shape:
            raise ValueError("weights must match y_obs shape.")
        sw = np.sqrt(np.clip(w, 0.0, np.inf))
        Xw = X * sw[:, None]
        yw = y * sw
    else:
        w = None
        Xw = X
        yw = y

    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    A_best = float(coef[0])
    B_best = float(coef[1])

    if bounds is not None:
        a_lo, a_hi = bounds.get("A_obs", (-np.inf, np.inf))
        b_lo, b_hi = bounds.get("B_obs", (-np.inf, np.inf))
        A_best = float(np.clip(A_best, a_lo, a_hi))
        B_best = float(np.clip(B_best, b_lo, b_hi))

    y_fit = B_best + A_best * template_u
    residual = y_fit - y_obs
    rss = float(np.sum(residual ** 2))
    wrss = float(np.sum((residual ** 2) * w)) if w is not None else None
    return {
        "A_best": A_best,
        "B_best": B_best,
        "y_fit": y_fit,
        "rss": rss,
        "wrss": wrss,
    }


def build_observable_template_only(
    p_global,
    p_dataset,
    observable_mode,
    *,
    debye_obj=None,
    with_diag=False,
    F_ref=1.0,
):
    mode = str(observable_mode).strip().lower()
    if mode not in {"raw_m_chi2q", "raw_chi2q", "raw_eta"}:
        raise ValueError(f"build_observable_template_only unsupported mode: {observable_mode}")

    fluence_ratio = float(p_dataset.get("fluence_ratio", p_global.get("fluence_multiplier", 1.0)))
    dt_i_ps = float(p_global.get("dt0_ps", 0.0))
    if "alpha_dt_per_F" in p_global:
        dt_i_ps += float(p_global.get("alpha_dt_per_F", 0.0)) * (fluence_ratio - float(F_ref))
    sigma_irf_ps = float(p_global.get("sigma_irf_ps", 0.0))

    t = np.asarray(p_dataset["t"], dtype=float)
    t_model = t - dt_i_ps * 1e-12
    p_run = dict(p_global)
    p_run["fluence_multiplier"] = fluence_ratio
    model = NdSb3TM(p_run, debye_obj=debye_obj)
    sim = model.simulate_aligned(t_model, with_diag=with_diag)

    if mode == "raw_eta":
        u = np.asarray(sim["eta"], dtype=float)
    elif mode == "raw_chi2q":
        u = np.asarray(_compute_chi2q(sim), dtype=float)
    else:
        m = np.asarray(sim["m"], dtype=float)
        chi2q = np.asarray(_compute_chi2q(sim), dtype=float)
        u = m * chi2q

    template_u = _convolve_with_irf(u, t * 1e12, sigma_irf_ps)
    return {
        "template_u": template_u,
        "dt_i_ps": float(dt_i_ps),
        "sigma_irf_ps": float(sigma_irf_ps),
        "sim": sim,
        "t_model": t_model,
    }

def build_delta_k_template_only(
    p_global,
    p_dataset,
    observable_mode,
    *,
    debye_obj=None,
    with_diag=False,
    F_ref=1.0,
):
    """Build only the delta-k template from simulated states."""
    mode = str(observable_mode).strip().lower()
    if mode not in {
        "dk_chi2q",
        "dk_affine_chi2q",
        "dk_m_chi2q",
        "dk_affine_m_chi2q",
        "dk_rel_chi2q",
        "dk_rel_m_chi2q",
        "dk_raw_chi2q",
        "dk_raw_m_chi2q",
    }:
        raise ValueError(f"build_delta_k_template_only unsupported mode: {observable_mode}")

    fluence_ratio = float(p_dataset.get("fluence_ratio", p_global.get("fluence_multiplier", 1.0)))
    dt_i_ps = float(p_global.get("dt0_ps", 0.0))
    if "alpha_dt_per_F" in p_global:
        dt_i_ps += float(p_global.get("alpha_dt_per_F", 0.0)) * (fluence_ratio - float(F_ref))
    sigma_irf_ps = float(p_global.get("sigma_irf_ps", 0.0))

    t = np.asarray(p_dataset["t"], dtype=float)
    t_model = t - dt_i_ps * 1e-12
    p_run = dict(p_global)
    p_run["fluence_multiplier"] = fluence_ratio
    model = NdSb3TM(p_run, debye_obj=debye_obj)
    sim = model.simulate_aligned(t_model, with_diag=with_diag)

    chi2q = np.asarray(_compute_chi2q(sim), dtype=float)

    if mode in {"dk_chi2q", "dk_affine_chi2q", "dk_rel_chi2q", "dk_raw_chi2q"}:
        u = chi2q
    else:
        u = np.asarray(sim["m"], dtype=float) * chi2q

    template_u = _convolve_with_irf(u, t * 1e12, sigma_irf_ps)
    return {
        "template_u": template_u,
        "dt_i_ps": float(dt_i_ps),
        "sigma_irf_ps": float(sigma_irf_ps),
        "sim": sim,
        "t_model": t_model,
    }


def build_delta_k_residual(
    y_obs,
    y_model,
    is_resolved,
    sigma_resolved,
    sigma_censored,
    resolution_limit,
    sigma_point=None,
    sigma_floor=None,
):
    """Build censored residuals for resolved/unresolved delta-k observations."""
    y_obs = np.asarray(y_obs, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    is_resolved = np.asarray(is_resolved, dtype=bool)
    if y_obs.shape != y_model.shape or y_obs.shape != is_resolved.shape:
        raise ValueError(
            "build_delta_k_residual shape mismatch: y_obs, y_model, is_resolved must match."
        )
    sigma_resolved = float(max(sigma_resolved, 1e-12))
    sigma_censored = float(max(sigma_censored, 1e-12))
    resolution_limit = float(resolution_limit)
    sigma_floor = float(max(sigma_floor if sigma_floor is not None else 1e-12, 1e-12))

    sigma_eff = np.full_like(y_obs, sigma_resolved, dtype=float)
    if sigma_point is not None:
        sigma_point = np.asarray(sigma_point, dtype=float)
        if sigma_point.shape != y_obs.shape:
            raise ValueError(
                "build_delta_k_residual shape mismatch: sigma_point must match y_obs shape."
            )
        valid = np.isfinite(sigma_point) & (sigma_point > 0.0)
        sigma_eff[valid] = np.maximum(sigma_point[valid], sigma_floor)

    r = np.zeros_like(y_obs, dtype=float)
    resolved_mask = is_resolved
    unresolved_mask = ~resolved_mask
    r[resolved_mask] = (y_model[resolved_mask] - y_obs[resolved_mask]) / sigma_eff[resolved_mask]
    over = y_model[unresolved_mask] > resolution_limit
    if np.any(over):
        idx = np.where(unresolved_mask)[0][over]
        r[idx] = (y_model[idx] - resolution_limit) / sigma_censored
    return r


def build_observable(sim, p_global, p_local, observable_mode):
    """
    Map simulator output to the measured S-like observable.

    Parameters
    ----------
    sim : dict
        Output dictionary from ``NdSb3TM.simulate_aligned``.
    p_global : dict
        Global best-fit or working parameter dictionary.
    p_local : dict
        Dataset-specific local parameter dictionary.
    observable_mode : str
        One of ``"m"``, ``"eta"``, ``"eta_m2"``, ``"eta_m1_mult"``,
        ``"eta_m2_mult"``, ``"chi2q"``, ``"m_chi2q"``, ``"raw_chi2q"``,
        ``"raw_m_chi2q"``, or ``"raw_eta"``.
    """
    mode = str(observable_mode).strip().lower()
    if mode == "m":
        return np.asarray(sim["S_m"], dtype=float)

    eta = np.asarray(sim["eta"], dtype=float)
    m = np.asarray(sim["m"], dtype=float)
    A_obs = float(p_local.get("A_obs", p_global.get("A_obs", 1.0)))
    F = float(p_global.get("fluence_multiplier", 1.0))

    def _legacy_b_eff():
        return float(p_global.get("B0_obs", 0.0)) + float(p_global.get("B1_obs", 0.0)) * (F - 1.0)

    def _pick_b_eff(allow_global_b_obs=True):
        if "B_obs" in p_local:
            return float(p_local["B_obs"])
        if allow_global_b_obs and "B_obs" in p_global:
            return float(p_global["B_obs"])
        return _legacy_b_eff()

    B_obs = _pick_b_eff(allow_global_b_obs=True)

    if mode == "eta":
        return B_obs + A_obs * eta
    if mode == "eta_m2":
        lam_m2 = float(p_global.get("lam_m2", 0.0))
        return B_obs + A_obs * (eta + lam_m2 * (m ** 2))
    if mode == "eta_m1_mult":
        return B_obs + A_obs * (m * eta)
    if mode == "eta_m2_mult":
        return B_obs + A_obs * ((m ** 2) * eta)
    if mode == "chi2q":
        chi2q = _compute_chi2q(sim)
        return B_obs + A_obs * chi2q
    if mode == "m_chi2q":
        chi2q = _compute_chi2q(sim)
        return B_obs + A_obs * (m * chi2q)
    if mode == "raw_chi2q":
        chi2q = _compute_chi2q(sim)
        return _pick_b_eff(allow_global_b_obs=False) + A_obs * chi2q
    if mode == "raw_m_chi2q":
        chi2q = _compute_chi2q(sim)
        B_eff = _pick_b_eff(allow_global_b_obs=False)

        return B_eff + A_obs * (m * chi2q)
    if mode == "raw_eta":
        return _pick_b_eff(allow_global_b_obs=False) + A_obs * eta

    raise ValueError(
        "Unsupported observable_mode='{mode}'. Expected one of: "
        "m, eta, eta_m2, eta_m1_mult, eta_m2_mult, chi2q, m_chi2q, "
        "raw_chi2q, raw_m_chi2q, raw_eta."
        .format(mode=observable_mode)
    )


# ============================================================
# Filename helper
# ============================================================
def parse_fluence_ratio_from_name(name_or_path: str) -> float:
    """Parse a fluence ratio from names such as ``deltak12k_2p5mW.csv``."""
    base = os.path.basename(name_or_path)
    match = re.search(r"_(\d+)p(\d+)mW(?:\.[^.]+)?$", base, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Cannot parse fluence_ratio from filename '{base}'. Expected a suffix like '_2p5mW.csv'."
        )
    return float(f"{match.group(1)}.{match.group(2)}")


# ============================================================
# Single-dataset fitting (legacy API preserved)
# ============================================================
def fit_params(t, Te, S, p0, fit_keys, sigma_Te=2.0, sigma_S=0.02):
    """
    Fit using Te(t) and/or S(t). Times are seconds.

    This function is kept intact as the legacy single-dataset fitting entry
    point used by the existing GUI buttons. Its S-channel is the legacy
    S_m proxy path and does not use multi-fit observable_mode.
    """
    if Te is None and S is None:
        raise ValueError("Need at least Te or S data to fit.")

    p0 = normalize_params_dict(p0)
    raw_ab_mode = mode in {"dk_raw_chi2q", "dk_raw_m_chi2q"}
    fit_keys = normalize_fit_keys(fit_keys)

    debye_obj = DebyeCl(thetaD=float(p0["ThetaD"]))
    lb, ub = _build_fit_bounds(fit_keys, _get_bounds_for_keys)

    x0 = _pack_params(p0, fit_keys)

    span = ub - lb
    eps = np.full_like(x0, 1e-12, dtype=float)
    finite = np.isfinite(lb) & np.isfinite(ub)
    eps[finite] = np.minimum(1e-12, 0.1 * np.maximum(span[finite], 0.0))
    x0 = np.minimum(np.maximum(x0, lb + eps), ub - eps)

    t = np.asarray(t, float)
    if t.ndim != 1 or t.size < 5:
        raise ValueError("time array must be 1D with at least 5 points.")

    if Te is not None:
        Te = np.asarray(Te, float)
        if Te.shape != t.shape:
            raise ValueError("Te must have same shape as t.")
    if S is not None:
        S = np.asarray(S, float)
        if S.shape != t.shape:
            raise ValueError("S must have same shape as t.")

    sigma_Te = float(max(sigma_Te, 1e-12))
    sigma_S = float(max(sigma_S, 1e-12))

    def residual(x):
        p = dict(p0)
        _unpack_params(p, fit_keys, x)
        model = NdSb3TM(p, debye_obj=debye_obj)
        sim = model.simulate_aligned(t, with_diag=False)

        r = []
        if Te is not None:
            r.append((sim["Te"] - Te) / sigma_Te)
        if S is not None:
            r.append((sim["S_m"] - S) / sigma_S)
        if not r:
            raise RuntimeError("No residual components constructed.")
        return np.concatenate(r)

    res = least_squares(
        residual,
        x0,
        bounds=(lb, ub),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=200,
    )

    p_best = dict(p0)
    _unpack_params(p_best, fit_keys, res.x)
    return p_best, res


# ============================================================
# Multi-dataset global fitting
# ============================================================
def fit_params_multi(
    datasets,
    p0,
    global_keys=None,
    local_keys=None,
    observable_mode=MULTI_FIT_DEFAULT_OBSERVABLE_MODE,
    sigma_S=0.02,
    max_nfev=300,
    progress_every=5,
    progress_callback=None,
    optimizer_verbose=0,
    enable_timing=True,
):
    """
    Jointly fit multiple S datasets.

    Default mainline uses variable projection for readout in raw_m_chi2q mode:
      - nonlinear: global parameters only
      - linear (per-dataset each residual call): A_obs, B_obs
    Legacy local nonlinear readout can be restored by setting USE_VARPRO_READOUT=False.
    """
    if not datasets:
        raise ValueError("fit_params_multi: datasets must be a non-empty list.")

    p0 = normalize_params_dict(p0)
    if global_keys is None:
        global_keys = MULTI_FIT_DEFAULT_GLOBAL_KEYS
    if local_keys is None:
        local_keys = MULTI_FIT_DEFAULT_LOCAL_KEYS
    global_keys = normalize_fit_keys(global_keys)
    requested_local_keys = list(local_keys)

    sigma_S = float(max(sigma_S, 1e-12))
    debye_obj = DebyeCl(thetaD=float(p0["ThetaD"]))
    use_varpro = bool(p0.get("USE_VARPRO_READOUT", USE_VARPRO_READOUT)) and str(observable_mode).strip().lower() == "raw_m_chi2q"

    validated = []
    for i, dataset in enumerate(datasets):
        name = dataset.get("name") or f"dataset_{i}"
        t = np.asarray(dataset.get("t"), dtype=float)
        if t.ndim != 1 or t.size < 5:
            raise ValueError(f"Dataset '{name}' must provide a 1D time array with at least 5 points.")
        S = dataset.get("S")
        if S is None:
            raise ValueError(f"Dataset '{name}' is missing S data; multi-dataset fitting currently requires S.")
        S = np.asarray(S, dtype=float)
        if S.shape != t.shape:
            raise ValueError(f"Dataset '{name}' has mismatched shapes: t{t.shape} vs S{S.shape}.")
        fluence_ratio = dataset.get("fluence_ratio")
        if fluence_ratio is None:
            raise ValueError(f"Dataset '{name}' is missing required key 'fluence_ratio'.")

        local_defaults = {"dt_local": 0.0}
        for key in requested_local_keys:
            if key in dataset:
                local_defaults[key] = float(dataset[key])
            elif key in p0:
                local_defaults[key] = float(p0[key])
            else:
                local_defaults[key] = 0.0

        validated.append({
            "name": str(name),
            "path": dataset.get("path"),
            "t": t,
            "S": S,
            "Te": dataset.get("Te"),
            "Ts": dataset.get("Ts"),
            "Tl": dataset.get("Tl"),
            "fluence_ratio": float(fluence_ratio),
            "local_init": local_defaults,
        })

    if use_varpro:
        effective_local_keys = []
    else:
        effective_local_keys = list(requested_local_keys)

    global_lb, global_ub = _build_fit_bounds(global_keys, _get_bounds_for_keys)
    if effective_local_keys:
        local_lb, local_ub = _build_fit_bounds(effective_local_keys, _get_multi_local_bounds)
    else:
        local_lb = np.array([], dtype=float)
        local_ub = np.array([], dtype=float)

    x0_parts = [_pack_params(p0, global_keys)]
    lb_parts = [global_lb]
    ub_parts = [global_ub]
    for dataset in validated:
        x0_parts.append(_pack_params(dataset["local_init"], effective_local_keys))
        lb_parts.append(local_lb)
        ub_parts.append(local_ub)

    x0 = np.concatenate(x0_parts)
    lb = np.concatenate(lb_parts)
    ub = np.concatenate(ub_parts)
    span = ub - lb
    eps = np.full_like(x0, 1e-12, dtype=float)
    finite = np.isfinite(lb) & np.isfinite(ub)
    eps[finite] = np.minimum(1e-12, 0.1 * np.maximum(span[finite], 0.0))
    x0 = np.minimum(np.maximum(x0, lb + eps), ub - eps)

    n_global = len(global_keys)
    n_local = len(effective_local_keys)
    enable_timing = bool(enable_timing)
    progress_every = int(progress_every) if progress_every is not None else 0
    residual_call_count = 0
    fit_start_time = perf_counter()
    dataset_timing_stats = {
        dataset["name"]: {
            "dataset_name": dataset["name"],
            "wall_time_total_sec": 0.0,
            "call_count": 0,
            "avg_wall_time_sec": 0.0,
        }
        for dataset in validated
    }

    def _emit_progress(message):
        if progress_callback is None:
            print(message, flush=True)
            return
        progress_callback(message)

    def _build_timing_summary():
        elapsed_sec = max(perf_counter() - fit_start_time, 0.0)
        avg_residual_sec = elapsed_sec / residual_call_count if residual_call_count > 0 else 0.0
        estimated_total_calls = int(max(int(max_nfev), 1) * 5)
        rough_eta_sec = avg_residual_sec * max(estimated_total_calls - residual_call_count, 0)
        return {
            "residual_call_count": int(residual_call_count),
            "elapsed_sec": float(elapsed_sec),
            "avg_residual_sec": float(avg_residual_sec),
            "estimated_total_calls": estimated_total_calls,
            "rough_eta_sec": float(rough_eta_sec),
            "per_dataset": {
                name: {
                    "dataset_name": stats["dataset_name"],
                    "wall_time_total_sec": float(stats["wall_time_total_sec"]),
                    "call_count": int(stats["call_count"]),
                    "avg_wall_time_sec": float(stats["avg_wall_time_sec"]),
                }
                for name, stats in dataset_timing_stats.items()
            },
        }

    def _split_x(x):
        x = np.asarray(x, dtype=float)
        x_global = x[:n_global]
        x_local = []
        offset = n_global
        for _ in validated:
            x_local.append(x[offset:offset + n_local])
            offset += n_local
        return x_global, x_local

    def _evaluate_dataset(p_global, dataset, local_x, with_diag=False):
        eval_start = perf_counter() if enable_timing else None

        p_dataset = dict(p0)
        p_dataset.update(p_global)
        p_dataset["fluence_ratio"] = float(dataset["fluence_ratio"])
        p_dataset["t"] = np.asarray(dataset["t"], dtype=float)
        point_weights = _build_time_weights_from_t(dataset["t"])

        if use_varpro:
            template_info = build_observable_template_only(
                p_global,
                p_dataset,
                observable_mode,
                debye_obj=debye_obj,
                with_diag=with_diag,
                F_ref=1.0,
            )
            y_obs = np.asarray(dataset["S"], dtype=float)
            weights = (point_weights / (sigma_S ** 2)).astype(float, copy=False)
            lin = solve_linear_readout_ab(
                y_obs,
                template_info["template_u"],
                weights=weights,
                bounds=VARPRO_READOUT_BOUNDS,
            )
            p_local = {
                "A_obs": float(lin["A_best"]),
                "B_obs": float(lin["B_best"]),
                "dt_local": 0.0,
            }
            S_fit = np.asarray(lin["y_fit"], dtype=float)
            dt_i_ps = float(template_info["dt_i_ps"])
            sigma_irf_ps = float(template_info["sigma_irf_ps"])
            sim = template_info["sim"]
            t_model = np.asarray(template_info["t_model"], dtype=float)
        else:
            p_local = dict(dataset["local_init"])
            _unpack_params(p_local, effective_local_keys, local_x)
            p_dataset["fluence_multiplier"] = float(dataset["fluence_ratio"])
            dt_local = float(p_local.get("dt_local", 0.0))
            t_model = np.asarray(dataset["t"], dtype=float) - dt_local
            model = NdSb3TM(p_dataset, debye_obj=debye_obj)
            sim = model.simulate_aligned(t_model, with_diag=with_diag)
            S_fit = build_observable(sim, p_dataset, p_local, observable_mode)
            dt_i_ps = float(dt_local * 1e12)
            sigma_irf_ps = float(p_global.get("sigma_irf_ps", np.nan))

        residual = ((S_fit - dataset["S"]) / sigma_S) * np.sqrt(point_weights)

        wall_time_sec = max(perf_counter() - eval_start, 0.0) if enable_timing and eval_start is not None else 0.0
        stats = dataset_timing_stats[dataset["name"]]
        if enable_timing:
            stats["call_count"] += 1
            stats["wall_time_total_sec"] += wall_time_sec
            stats["avg_wall_time_sec"] = stats["wall_time_total_sec"] / max(stats["call_count"], 1)

        return {
            "name": dataset["name"],
            "path": dataset["path"],
            "fluence_ratio": dataset["fluence_ratio"],
            "t": np.asarray(dataset["t"], dtype=float),
            "t_model": t_model,
            "S_exp": np.asarray(dataset["S"], dtype=float),
            "S_fit": np.asarray(S_fit, dtype=float),
            "Te_fit": np.asarray(sim["Te"], dtype=float),
            "Ts_fit": np.asarray(sim["Ts"], dtype=float),
            "Tl_fit": np.asarray(sim["Tl"], dtype=float),
            "m_fit": np.asarray(sim["m"], dtype=float),
            "eta_fit": np.asarray(sim["eta"], dtype=float),
            "phi_fit": np.asarray(sim["phi"], dtype=float),
            "chi2q_fit": np.asarray(_compute_chi2q(sim), dtype=float),
            "residual": np.asarray(residual, dtype=float),
            "point_weights": np.asarray(point_weights, dtype=float),
            "sim": sim,
            "local_params": p_local,
            "params": p_dataset,
            "diag": sim.get("diag"),
            "dt_i_ps": dt_i_ps,
            "sigma_irf_ps": sigma_irf_ps,
            "wall_time_sec": float(wall_time_sec),
            "avg_wall_time_sec": float(stats["avg_wall_time_sec"]),
        }

    def residual(x):
        nonlocal residual_call_count

        x_global, x_local_list = _split_x(x)
        p_global = dict(p0)
        _unpack_params(p_global, global_keys, x_global)

        residual_call_count += 1
        parts = []
        for dataset, local_x in zip(validated, x_local_list):
            evaluated = _evaluate_dataset(p_global, dataset, local_x, with_diag=False)
            parts.append(evaluated["residual"])

        if progress_every > 0 and (residual_call_count % progress_every == 0):
            timing_summary = _build_timing_summary()
            _emit_progress(
                "[multi-fit] residual_call={residual_call} | elapsed={elapsed:.2f} s | "
                "avg_residual={avg_residual:.4f} s | datasets={datasets} | rough_eta={rough_eta:.2f} s".format(
                    residual_call=timing_summary["residual_call_count"],
                    elapsed=timing_summary["elapsed_sec"],
                    avg_residual=timing_summary["avg_residual_sec"],
                    datasets=len(validated),
                    rough_eta=timing_summary["rough_eta_sec"],
                )
            )

        return np.concatenate(parts)

    res = least_squares(
        residual,
        x0,
        bounds=(lb, ub),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(max(max_nfev, 1)),
        verbose=int(optimizer_verbose),
    )

    x_global_best, x_local_best = _split_x(res.x)
    best_global_params = dict(p0)
    _unpack_params(best_global_params, global_keys, x_global_best)

    dataset_fits = []
    best_local_params = {}
    dataset_summary = []
    for dataset, local_x in zip(validated, x_local_best):
        evaluated = _evaluate_dataset(best_global_params, dataset, local_x, with_diag=True)
        local_best = dict(evaluated["local_params"])
        best_local_params[dataset["name"]] = local_best

        B0 = float(best_global_params.get("B0_obs", np.nan))
        B1 = float(best_global_params.get("B1_obs", np.nan))
        F = float(dataset["fluence_ratio"])
        if "B_obs" in local_best:
            b_obs_summary = float(local_best["B_obs"])
            b_eff_summary = b_obs_summary
        else:
            b_obs_summary = float(np.nan)
            b_eff_summary = B0 + B1 * (F - 1.0)

        a_obs_summary = float(local_best.get("A_obs", best_global_params.get("A_obs", np.nan)))
        rms = float(np.sqrt(np.mean((evaluated["S_fit"] - evaluated["S_exp"]) ** 2)))
        wrms = float(np.sqrt(np.mean(evaluated["residual"] ** 2)))
        evaluated["rms"] = rms
        evaluated["wrms"] = wrms
        dataset_fits.append(evaluated)
        dataset_summary.append({
            "dataset_name": dataset["name"],
            "dataset": dataset["name"],
            "path": dataset["path"],
            "fluence_ratio": float(dataset["fluence_ratio"]),
            "n_points": int(evaluated["t"].size),
            "rms": rms,
            "wrms": wrms,
            "dt_i_ps": float(evaluated["dt_i_ps"]),
            "dt_local_ps": float(local_best.get("dt_local", 0.0) * 1e12),
            "A_obs": a_obs_summary,
            "B_obs": b_obs_summary,
            "B_eff_obs": b_eff_summary,
            "sigma_irf_ps": float(evaluated.get("sigma_irf_ps", np.nan)),
            "B0_obs": float(best_global_params.get("B0_obs", np.nan)),
            "B1_obs": float(best_global_params.get("B1_obs", np.nan)),
            "wall_time_sec": float(evaluated["wall_time_sec"]),
            "avg_wall_time_sec": float(evaluated["avg_wall_time_sec"]),
        })

    timing_summary = _build_timing_summary()

    fit_bundle = {
        "target_kind": "S",
        "observable_mode": str(observable_mode),
        "use_varpro_readout": bool(use_varpro),
        "global_keys": list(global_keys),
        "local_keys": list(effective_local_keys),
        "legacy_local_keys_requested": list(requested_local_keys),
        "best_global_params": best_global_params,
        "best_local_params": best_local_params,
        "dataset_fits": dataset_fits,
        "dataset_summary": dataset_summary,
        "timing_summary": timing_summary,
    }
    return fit_bundle, res


def fit_params_multi_dk(
    datasets,
    p0,
    global_keys=None,
    local_keys=None,
    observable_mode="dk_chi2q",
    sigma_dk=None,
    sigma_dk_censored=None,
    dk_resolution_limit=None,
    max_nfev=300,
    progress_every=5,
    progress_callback=None,
    optimizer_verbose=0,
    enable_timing=True,
):
    """Jointly fit multiple datasets using delta-k specific readout/loss."""
    if not datasets:
        raise ValueError("fit_params_multi_dk: datasets must be a non-empty list.")

    mode = str(observable_mode).strip().lower()
    if mode not in {
        "dk_chi2q",
        "dk_affine_chi2q",
        "dk_m_chi2q",
        "dk_affine_m_chi2q",
        "dk_rel_chi2q",
        "dk_rel_m_chi2q",
        "dk_raw_chi2q",
        "dk_raw_m_chi2q",
    }:
        raise ValueError(f"fit_params_multi_dk unsupported observable_mode: {observable_mode}")

    p0 = normalize_params_dict(p0)
    sigma_resolved = float(max(sigma_dk if sigma_dk is not None else p0.get("sigma_dk", 0.002), 1e-12))
    sigma_censored = float(max(sigma_dk_censored if sigma_dk_censored is not None else p0.get("sigma_dk_censored", 0.002), 1e-12))
    resolution_limit = float(dk_resolution_limit if dk_resolution_limit is not None else p0.get("dk_resolution_limit", 0.003))

    if global_keys is None:
        base = [k for k in MULTI_FIT_DEFAULT_GLOBAL_KEYS if k not in {"A_obs", "B_obs", "B0_obs", "B1_obs"}]
        global_keys = list(base) + (["K_dk"] if "K_dk" not in base else [])
    if local_keys is None:
        local_keys = []
    if local_keys:
        raise ValueError("fit_params_multi_dk currently supports only empty local_keys.")

    global_keys = normalize_fit_keys(global_keys)

    if raw_ab_mode:
        global_keys = [
            k for k in global_keys
            if k not in {"K_dk", "B_dk", "A_obs", "B_obs", "B0_obs", "B1_obs"}
        ]
    else:
        if "K_dk" not in global_keys:
            global_keys = list(global_keys) + ["K_dk"]

    if raw_ab_mode:
        global_keys = [k for k in global_keys if k != "B_dk"]
    elif mode in {"dk_affine_chi2q", "dk_affine_m_chi2q"}:
        if "B_dk" not in global_keys:
            global_keys = list(global_keys) + ["B_dk"]
    elif "B_dk" in global_keys:
        global_keys = [k for k in global_keys if k != "B_dk"]

    debye_obj = DebyeCl(thetaD=float(p0["ThetaD"]))
    validated = []
    for i, dataset in enumerate(datasets):
        name = dataset.get("name") or f"dataset_{i}"
        t = np.asarray(dataset.get("t"), dtype=float)
        y = np.asarray(dataset.get("delta_k", dataset.get("S")), dtype=float)
        resolved = np.asarray(dataset.get("is_resolved", np.isfinite(y) & (y > resolution_limit)), dtype=bool)
        if t.ndim != 1 or y.ndim != 1 or t.shape != y.shape:
            raise ValueError(f"Dataset '{name}' must provide matching 1D t and delta_k arrays.")
        if resolved.shape != y.shape:
            raise ValueError(f"Dataset '{name}' has mismatched is_resolved shape: {resolved.shape} vs {y.shape}.")
        sigma_point = dataset.get("sigma_dk", None)
        if sigma_point is not None:
            sigma_point = np.asarray(sigma_point, dtype=float)
            if sigma_point.shape != y.shape:
                raise ValueError(
                    f"Dataset '{name}' has mismatched sigma_dk shape: {sigma_point.shape} vs {y.shape}."
                )
        fluence_ratio = dataset.get("fluence_ratio")
        if fluence_ratio is None:
            raise ValueError(f"Dataset '{name}' is missing required key 'fluence_ratio'.")
        validated.append({
            "name": str(name),
            "path": dataset.get("path"),
            "t": t,
            "delta_k": y,
            "is_resolved": resolved,
            "Te": dataset.get("Te"),
            "Ts": dataset.get("Ts"),
            "Tl": dataset.get("Tl"),
            "fluence_ratio": float(fluence_ratio),
            "sigma_dk": sigma_point,
        })

    lb, ub = _build_fit_bounds(global_keys, _get_bounds_for_keys)
    x0 = _pack_params(p0, global_keys)
    span = ub - lb
    eps = np.full_like(x0, 1e-12, dtype=float)
    finite = np.isfinite(lb) & np.isfinite(ub)
    eps[finite] = np.minimum(1e-12, 0.1 * np.maximum(span[finite], 0.0))
    x0 = np.minimum(np.maximum(x0, lb + eps), ub - eps)

    residual_call_count = 0
    fit_start_time = perf_counter()
    progress_every = int(progress_every) if progress_every is not None else 0

    def _emit_progress(message):
        if progress_callback is None:
            print(message, flush=True)
        else:
            progress_callback(message)

    def _eval_dataset(p_global, dataset, with_diag=False):
        p_dataset = dict(p0)
        p_dataset.update(p_global)
        p_dataset["fluence_ratio"] = float(dataset["fluence_ratio"])
        p_dataset["t"] = np.asarray(dataset["t"], dtype=float)

        tmpl = build_delta_k_template_only(
            p_global,
            p_dataset,
            mode,
            debye_obj=debye_obj,
            with_diag=with_diag,
            F_ref=1.0,
        )
        u = np.asarray(tmpl["template_u"], dtype=float)
        y_obs = np.asarray(dataset["delta_k"], dtype=float)

        raw_ab_mode = mode in {"dk_raw_chi2q", "dk_raw_m_chi2q"}
        affine_mode = mode in {"dk_affine_chi2q", "dk_affine_m_chi2q"}
        relative_mode = mode in {"dk_rel_chi2q", "dk_rel_m_chi2q"}

        if raw_ab_mode:
            mask = np.isfinite(u) & np.isfinite(y_obs)
            sigma_point = dataset.get("sigma_dk", None)
            if sigma_point is not None:
                sigma_point = np.asarray(sigma_point, dtype=float)
                valid_sigma = np.isfinite(sigma_point) & (sigma_point > 0.0)
                mask = mask & valid_sigma
            if np.count_nonzero(mask) < 2:
                raise ValueError(f"Dataset '{dataset['name']}' has insufficient valid points for raw AB readout.")
            X = np.column_stack([u[mask], np.ones(np.count_nonzero(mask))])
            if sigma_point is not None:
                sigma_floor = float(max(p0.get("sigma_dk_floor", 1e-12), 1e-12))
                w = 1.0 / np.maximum(sigma_point[mask], sigma_floor)
                Xw = X * w[:, None]
                yw = y_obs[mask] * w
                coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            else:
                coef, *_ = np.linalg.lstsq(X, y_obs[mask], rcond=None)
            A_obs_i = float(coef[0])
            B_obs_i = float(coef[1])
            y_model = B_obs_i + A_obs_i * u

            if sigma_point is not None:
                sigma_eff = np.maximum(np.asarray(sigma_point, dtype=float), p0.get("sigma_dk_floor", 1e-12))
            else:
                sigma_eff = np.full_like(y_obs, sigma_resolved, dtype=float)
            residual = (y_model - y_obs) / sigma_eff
        elif relative_mode:
            K_dk = float(p_global.get("K_dk", p0.get("K_dk", 0.02)))
            t_ps = np.asarray(dataset["t"], dtype=float) * 1e12
            neg_mask = t_ps < 0.0
            if int(np.count_nonzero(neg_mask)) >= 2:
                base_mask = neg_mask
            else:
                base_mask = np.zeros_like(t_ps, dtype=bool)
                base_mask[:min(3, t_ps.size)] = True

            u0 = float(np.mean(u[base_mask]))
            dk0 = float(np.mean(y_obs[base_mask]))
            A_obs_i = K_dk
            B_obs_i = dk0 - K_dk * u0
            y_model = dk0 + K_dk * (u - u0)
            y_model = np.clip(y_model, 0.0, np.inf)
            residual = build_delta_k_residual(
                y_obs,
                y_model,
                dataset["is_resolved"],
                sigma_resolved,
                sigma_censored,
                resolution_limit,
                sigma_point=dataset.get("sigma_dk", None),
                sigma_floor=p0.get("sigma_dk_floor", 1e-12),
            )
        else:
            K_dk = float(p_global.get("K_dk", p0.get("K_dk", 0.02)))
            B_dk = float(p_global.get("B_dk", p0.get("B_dk", 0.0))) if affine_mode else 0.0
            A_obs_i = K_dk
            B_obs_i = B_dk
            y_model = B_dk + K_dk * u
            y_model = np.clip(y_model, 0.0, np.inf)
            residual = build_delta_k_residual(
                y_obs,
                y_model,
                dataset["is_resolved"],
                sigma_resolved,
                sigma_censored,
                resolution_limit,
                sigma_point=dataset.get("sigma_dk", None),
                sigma_floor=p0.get("sigma_dk_floor", 1e-12),
            )
        return {
            "name": dataset["name"],
            "path": dataset["path"],
            "fluence_ratio": float(dataset["fluence_ratio"]),
            "t": np.asarray(dataset["t"], dtype=float),
            "t_model": np.asarray(tmpl["t_model"], dtype=float),
            "delta_k_exp": np.asarray(dataset["delta_k"], dtype=float),
            "delta_k_fit": np.asarray(y_model, dtype=float),
            "is_resolved": np.asarray(dataset["is_resolved"], dtype=bool),
            "Te_fit": np.asarray(tmpl["sim"]["Te"], dtype=float),
            "Ts_fit": np.asarray(tmpl["sim"]["Ts"], dtype=float),
            "Tl_fit": np.asarray(tmpl["sim"]["Tl"], dtype=float),
            "m_fit": np.asarray(tmpl["sim"]["m"], dtype=float),
            "eta_fit": np.asarray(tmpl["sim"]["eta"], dtype=float),
            "phi_fit": np.asarray(tmpl["sim"]["phi"], dtype=float),
            "chi2q_fit": np.asarray(_compute_chi2q(tmpl["sim"]), dtype=float),
            "residual": np.asarray(residual, dtype=float),
            "sigma_dk": np.asarray(dataset["sigma_dk"], dtype=float) if dataset.get("sigma_dk", None) is not None else None,
            "sim": tmpl["sim"],
            "diag": tmpl["sim"].get("diag"),
            "dt_i_ps": float(tmpl["dt_i_ps"]),
            "sigma_irf_ps": float(tmpl["sigma_irf_ps"]),
            "A_obs": float(A_obs_i),
            "B_obs": float(B_obs_i),
            "readout_mode": "per_dataset_AB" if raw_ab_mode else "global_K",
        }

    def residual(x):
        nonlocal residual_call_count
        p_global = dict(p0)
        _unpack_params(p_global, global_keys, x)
        residual_call_count += 1
        parts = [_eval_dataset(p_global, ds, with_diag=False)["residual"] for ds in validated]
        if progress_every > 0 and residual_call_count % progress_every == 0:
            elapsed = max(perf_counter() - fit_start_time, 0.0)
            _emit_progress(
                f"[multi-fit-dk] residual_call={residual_call_count} | elapsed={elapsed:.2f} s | datasets={len(validated)}"
            )
        return np.concatenate(parts)

    res = least_squares(
        residual,
        x0,
        bounds=(lb, ub),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(max(max_nfev, 1)),
        verbose=int(optimizer_verbose),
    )

    best_global_params = dict(p0)
    _unpack_params(best_global_params, global_keys, res.x)
    dataset_fits = []
    dataset_summary = []
    for ds in validated:
        evaluated = _eval_dataset(best_global_params, ds, with_diag=True)
        resolved_mask = np.asarray(evaluated["is_resolved"], dtype=bool)
        if np.any(resolved_mask):
            rms = float(np.sqrt(np.mean((evaluated["delta_k_fit"][resolved_mask] - evaluated["delta_k_exp"][resolved_mask]) ** 2)))
        else:
            rms = float("nan")
        wrms = float(np.sqrt(np.mean(evaluated["residual"] ** 2)))
        evaluated["rms"] = rms
        evaluated["wrms"] = wrms
        dataset_fits.append(evaluated)
        dataset_summary.append({
            "dataset_name": ds["name"],
            "dataset": ds["name"],
            "path": ds["path"],
            "fluence_ratio": float(ds["fluence_ratio"]),
            "n_points": int(evaluated["t"].size),
            "n_resolved": int(np.count_nonzero(resolved_mask)),
            "n_unresolved": int(evaluated["t"].size - np.count_nonzero(resolved_mask)),
            "rms": rms,
            "wrms": wrms,
            "dt_i_ps": float(evaluated["dt_i_ps"]),
            "sigma_irf_ps": float(evaluated["sigma_irf_ps"]),
            "A_obs": float(evaluated.get("A_obs", np.nan)),
            "B_obs": float(evaluated.get("B_obs", np.nan)),
            "readout_mode": evaluated.get("readout_mode", ""),
        })

    elapsed_sec = max(perf_counter() - fit_start_time, 0.0)
    timing_summary = {
        "residual_call_count": int(residual_call_count),
        "elapsed_sec": float(elapsed_sec),
        "avg_residual_sec": float(elapsed_sec / residual_call_count) if residual_call_count > 0 else 0.0,
    }
    return {
        "observable_mode": mode,
        "target_kind": "delta_k",
        "global_keys": list(global_keys),
        "local_keys": [],
        "best_global_params": best_global_params,
        "best_local_params": {},
        "dataset_fits": dataset_fits,
        "dataset_summary": dataset_summary,
        "timing_summary": timing_summary,
        "use_varpro_readout": False,
    }, res


# ============================================================
# Export helpers for multi-fit
# ============================================================
def _format_ratio_token(value: float) -> str:
    return f"{float(value):.1f}".replace(".", "p")


def _slugify_dataset_name(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def _json_safe_dict(params: dict):
    safe = {}
    for key, value in params.items():
        if isinstance(value, (np.generic,)):
            safe[key] = value.item()
        elif isinstance(value, (float, int, str, bool)) or value is None:
            safe[key] = value
        else:
            try:
                safe[key] = float(value)
            except Exception:
                safe[key] = str(value)
    return safe


def export_multi_fit_results(fit_bundle, optimizer_result, export_root="fit_results"):
    """Export multi-dataset global-fit JSON/CSV/PNG artifacts."""
    if fit_bundle is None or optimizer_result is None:
        raise ValueError("export_multi_fit_results requires both fit_bundle and optimizer_result.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dataset_fits = fit_bundle["dataset_fits"]
    if not dataset_fits:
        raise ValueError("fit_bundle['dataset_fits'] is empty; nothing to export.")

    prefix = _slugify_dataset_name(dataset_fits[0]["name"]).split("_")[0] or "globalfit"
    ratios = sorted(item["fluence_ratio"] for item in dataset_fits)
    ratio_token = f"{_format_ratio_token(ratios[0])}to{_format_ratio_token(ratios[-1])}mW"

    out_dir = Path(export_root) / f"{prefix}_globalfit_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_name = f"globalfit_{prefix}_{fit_bundle['observable_mode']}_{ratio_token}_{timestamp}.json"
    json_path = out_dir / json_name
    summary_csv_path = out_dir / f"summary_globalfit_{prefix}_{timestamp}.csv"
    overlay_png_path = out_dir / f"overlay_globalfit_{prefix}_{timestamp}.png"
    params_png_path = out_dir / f"params_globalfit_{prefix}_{timestamp}.png"

    optimizer_summary = {
        "success": bool(optimizer_result.success),
        "status": int(optimizer_result.status),
        "cost": float(optimizer_result.cost),
        "nfev": int(optimizer_result.nfev),
        "message": str(optimizer_result.message),
    }
    target_kind = str(fit_bundle.get("target_kind", "S"))
    json_payload = {
        "target_kind": target_kind,
        "observable_mode": fit_bundle["observable_mode"],
        "global_keys": list(fit_bundle["global_keys"]),
        "local_keys": list(fit_bundle["local_keys"]),
        "best_global_params": _json_safe_dict(fit_bundle["best_global_params"]),
        "best_local_params": {k: _json_safe_dict(v) for k, v in fit_bundle["best_local_params"].items()},
        "optimizer_summary": optimizer_summary,
        "dataset_summary": fit_bundle["dataset_summary"],
        "timing_summary": fit_bundle.get("timing_summary"),
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with summary_csv_path.open("w", newline="", encoding="utf-8") as fh:
        if target_kind == "delta_k":
            fieldnames = [
                "dataset",
                "dataset_name",
                "path",
                "fluence_ratio",
                "n_points",
                "n_resolved",
                "n_unresolved",
                "rms",
                "wrms",
                "dt_i_ps",
                "sigma_irf_ps",
                "A_obs",
                "B_obs",
                "readout_mode",
            ]
        else:
            fieldnames = [
                "dataset",
                "dataset_name",
                "path",
                "fluence_ratio",
                "n_points",
                "rms",
                "wrms",
                "dt_i_ps",
                "dt_local_ps",
                "A_obs",
                "B_obs",
                "B_eff_obs",
                "sigma_irf_ps",
                "B0_obs",
                "B1_obs",
                "wall_time_sec",
                "avg_wall_time_sec",
            ]
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(fit_bundle["dataset_summary"])

        fitcurve_paths = []
        for item in dataset_fits:
            dataset_token = _slugify_dataset_name(item["name"])
            curve_path = out_dir / f"fitcurve_{dataset_token}_{timestamp}.csv"
            if target_kind == "delta_k":
                sigma_col = np.asarray(item["sigma_dk"], dtype=float) if item.get("sigma_dk") is not None else np.full_like(np.asarray(item["delta_k_exp"], dtype=float), np.nan, dtype=float)
                stacked = np.column_stack([
                    item["t"] * 1e12,
                    item["delta_k_exp"],
                    sigma_col,
                    item["delta_k_fit"],
                    item["m_fit"],
                    item["eta_fit"],
                    item["chi2q_fit"],
                    item["residual"],
                    item["is_resolved"].astype(int),
                    np.full_like(item["t"], item.get("A_obs", np.nan), dtype=float),
                    np.full_like(item["t"], item.get("B_obs", np.nan), dtype=float),
                ])
                header = "t_ps,delta_k_exp,sigma_dk,delta_k_fit,m_fit,eta_fit,chi2q_fit,residual,is_resolved,A_obs,B_obs"
            else:
                stacked = np.column_stack([
                    item["t"] * 1e12,
                    item["S_exp"],
                    item["S_fit"],
                    item["Te_fit"],
                    item["Ts_fit"],
                    item["Tl_fit"],
                    item["m_fit"],
                    item["eta_fit"],
                    item["chi2q_fit"],
                    item["residual"],
                ])
                header = "t_ps,S_exp,S_fit,Te_fit,Ts_fit,Tl_fit,m_fit,eta_fit,chi2q_fit,residual"
            np.savetxt(
                curve_path,
                stacked,
                delimiter=",",
                header=header,
                comments="",
            )
            fitcurve_paths.append(str(curve_path))        

    fig, ax = plt.subplots(figsize=(10, 6))
    for item in dataset_fits:
        label = f"{item['name']} ({item['fluence_ratio']:.1f} mW)"
        if target_kind == "delta_k":
            ax.scatter(item["t"] * 1e12, item["delta_k_exp"], s=14, alpha=0.5, label=f"exp {label}")
            ax.plot(item["t"] * 1e12, item["delta_k_fit"], linewidth=1.8, label=f"fit {label}")
        else:
            ax.scatter(item["t"] * 1e12, item["S_exp"], s=14, alpha=0.5, label=f"exp {label}")
            ax.plot(item["t"] * 1e12, item["S_fit"], linewidth=1.8, label=f"fit {label}")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("delta_k" if target_kind == "delta_k" else "S")
    ax.set_title(f"Global fit overlay ({fit_bundle['observable_mode']})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(overlay_png_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    global_keys = fit_bundle["global_keys"]
    global_values = [float(fit_bundle["best_global_params"].get(k, np.nan)) for k in global_keys]
    axes[0].bar(np.arange(len(global_keys)), global_values)
    axes[0].set_xticks(np.arange(len(global_keys)))
    axes[0].set_xticklabels(global_keys, rotation=45, ha="right")
    axes[0].set_title("Global parameters")
    axes[0].grid(alpha=0.3, axis="y")

    flu = [row["fluence_ratio"] for row in fit_bundle["dataset_summary"]]
    axes[1].plot(flu, [row.get("dt_i_ps", np.nan) for row in fit_bundle["dataset_summary"]], marker="o", label="dt_i_ps")
    if target_kind == "delta_k":
        axes[1].plot(flu, [row.get("n_resolved", np.nan) for row in fit_bundle["dataset_summary"]], marker="s", label="n_resolved")
        axes[1].plot(flu, [row.get("n_unresolved", np.nan) for row in fit_bundle["dataset_summary"]], marker="d", label="n_unresolved")
        axes[1].plot(flu, [row.get("A_obs", np.nan) for row in fit_bundle["dataset_summary"]], marker="^", label="A_obs")
        axes[1].plot(flu, [row.get("B_obs", np.nan) for row in fit_bundle["dataset_summary"]], marker="v", label="B_obs")
    else:
        axes[1].plot(flu, [row["A_obs"] for row in fit_bundle["dataset_summary"]], marker="s", label="A_obs")
        if any(np.isfinite(float(row.get("B_obs", np.nan))) for row in fit_bundle["dataset_summary"]):
            axes[1].plot(flu, [row.get("B_obs", np.nan) for row in fit_bundle["dataset_summary"]], marker="d", label="B_obs")
        axes[1].plot(flu, [row["B_eff_obs"] for row in fit_bundle["dataset_summary"]], marker="^", label="B_eff_obs")
    axes[1].set_xlabel("fluence ratio / mW label")
    axes[1].set_title("Per-dataset effective readout")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(params_png_path, dpi=180)
    plt.close(fig)

    return {
        "export_dir": str(out_dir),
        "json": str(json_path),
        "summary_csv": str(summary_csv_path),
        "fitcurve_csvs": fitcurve_paths,
        "overlay_png": str(overlay_png_path),
        "params_png": str(params_png_path),
    }
