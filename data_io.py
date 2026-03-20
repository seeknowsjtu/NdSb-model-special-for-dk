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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from config import (
    LEGACY_PARAM_ALIASES,
    MULTI_FIT_DEFAULT_GLOBAL_KEYS,
    MULTI_FIT_DEFAULT_LOCAL_KEYS,
    MULTI_FIT_DEFAULT_OBSERVABLE_MODE,
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

    # multi-fit local observable scale
    "A_obs",
}

LOCAL_KEY_BOUNDS = {
    "dt_local": (-2e-12, 2e-12),
    "A_obs": (-10.0, 10.0),
    "B_obs": (-2.0, 2.0),
}


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


# ============================================================
# Bounds helper
# ============================================================
def _get_bounds_for_keys(keys):
    lb, ub = [], []

    for k in keys:
        if k in ["G_el", "G_el0"]:
            lb.append(1e11); ub.append(5e15)
        elif k in ["G_es", "G_es0"]:
            lb.append(1e13); ub.append(1e18)
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
            lb.append(2e-10); ub.append(1e-8)
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
            lb.append(1e8); ub.append(1e14)
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
        One of ``"m"``, ``"eta"``, or ``"eta_m2"``.
    """
    mode = str(observable_mode).strip().lower()
    if mode == "m":
        return np.asarray(sim["S_m"], dtype=float)

    eta = np.asarray(sim["eta"], dtype=float)
    m = np.asarray(sim["m"], dtype=float)
    A_obs = float(p_local.get("A_obs", 1.0))
    B_obs = float(p_local.get("B_obs", 0.0))

    if mode == "eta":
        return B_obs + A_obs * eta
    if mode == "eta_m2":
        lam_m2 = float(p_global.get("lam_m2", 0.0))
        return B_obs + A_obs * (eta + lam_m2 * (m ** 2))

    raise ValueError(
        f"Unsupported observable_mode='{observable_mode}'. Expected one of: m, eta, eta_m2."
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
    point used by the existing GUI buttons.
    """
    if Te is None and S is None:
        raise ValueError("Need at least Te or S data to fit.")

    p0 = normalize_params_dict(p0)
    fit_keys = normalize_fit_keys(fit_keys)

    debye_obj = DebyeCl(thetaD=float(p0["ThetaD"]))
    lb, ub = _build_fit_bounds(fit_keys, _get_bounds_for_keys)

    x0 = _pack_params(p0, fit_keys)
    x0 = np.minimum(np.maximum(x0, lb + 1e-12), ub - 1e-12)

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
):
    """
    Jointly fit multiple S datasets using shared global parameters and
    per-dataset local parameters.

    Notes
    -----
    * Each dataset keeps its own original time array; no interpolation, no
      padding, and no resampling are performed.
    * ``dataset['fluence_ratio']`` is written directly into
      ``p['fluence_multiplier']`` and is never treated as a fit parameter.
    * Residual vectors from all datasets are concatenated directly.
    """
    if not datasets:
        raise ValueError("fit_params_multi: datasets must be a non-empty list.")

    p0 = normalize_params_dict(p0)
    if global_keys is None:
        global_keys = MULTI_FIT_DEFAULT_GLOBAL_KEYS
    if local_keys is None:
        local_keys = MULTI_FIT_DEFAULT_LOCAL_KEYS
    global_keys = normalize_fit_keys(global_keys)
    local_keys = list(local_keys)

    sigma_S = float(max(sigma_S, 1e-12))
    debye_obj = DebyeCl(thetaD=float(p0["ThetaD"]))

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

        local_defaults = {
            "dt_local": 0.0,
            "A_obs": float(dataset.get("A_obs", 1.0)),
            "B_obs": float(dataset.get("B_obs", 0.0)),
        }
        for key in local_keys:
            if key not in local_defaults:
                local_defaults[key] = float(dataset.get(key, p0.get(key, 0.0)))

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

    global_lb, global_ub = _build_fit_bounds(global_keys, _get_bounds_for_keys)
    local_lb, local_ub = _build_fit_bounds(local_keys, _get_multi_local_bounds)

    x0_parts = [_pack_params(p0, global_keys)]
    lb_parts = [global_lb]
    ub_parts = [global_ub]
    for dataset in validated:
        x0_parts.append(_pack_params(dataset["local_init"], local_keys))
        lb_parts.append(local_lb)
        ub_parts.append(local_ub)

    x0 = np.concatenate(x0_parts)
    lb = np.concatenate(lb_parts)
    ub = np.concatenate(ub_parts)
    x0 = np.minimum(np.maximum(x0, lb + 1e-12), ub - 1e-12)

    n_global = len(global_keys)
    n_local = len(local_keys)

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
        p_local = dict(dataset["local_init"])
        _unpack_params(p_local, local_keys, local_x)

        p_dataset = dict(p0)
        p_dataset.update(p_global)
        p_dataset["fluence_multiplier"] = float(dataset["fluence_ratio"])

        dt_local = float(p_local.get("dt_local", 0.0))
        t_model = np.asarray(dataset["t"], dtype=float) - dt_local
        model = NdSb3TM(p_dataset, debye_obj=debye_obj)
        sim = model.simulate_aligned(t_model, with_diag=with_diag)
        S_fit = build_observable(sim, p_dataset, p_local, observable_mode)
        residual = (S_fit - dataset["S"]) / sigma_S

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
            "residual": np.asarray(residual, dtype=float),
            "sim": sim,
            "local_params": p_local,
            "params": p_dataset,
            "diag": sim.get("diag"),
        }

    def residual(x):
        x_global, x_local_list = _split_x(x)
        p_global = dict(p0)
        _unpack_params(p_global, global_keys, x_global)

        parts = []
        for dataset, local_x in zip(validated, x_local_list):
            evaluated = _evaluate_dataset(p_global, dataset, local_x, with_diag=False)
            parts.append(evaluated["residual"])
        return np.concatenate(parts)

    res = least_squares(
        residual,
        x0,
        bounds=(lb, ub),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(max(max_nfev, 1)),
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

        rms = float(np.sqrt(np.mean((evaluated["S_fit"] - evaluated["S_exp"]) ** 2)))
        wrms = float(np.sqrt(np.mean(evaluated["residual"] ** 2)))
        evaluated["rms"] = rms
        evaluated["wrms"] = wrms
        dataset_fits.append(evaluated)
        dataset_summary.append({
            "dataset_name": dataset["name"],
            "path": dataset["path"],
            "fluence_ratio": float(dataset["fluence_ratio"]),
            "n_points": int(evaluated["t"].size),
            "rms": rms,
            "wrms": wrms,
            "dt_local_ps": float(local_best.get("dt_local", 0.0) * 1e12),
            "A_obs": float(local_best.get("A_obs", np.nan)),
            "B_obs": float(local_best.get("B_obs", np.nan)),
        })

    fit_bundle = {
        "observable_mode": str(observable_mode),
        "global_keys": list(global_keys),
        "local_keys": list(local_keys),
        "best_global_params": best_global_params,
        "best_local_params": best_local_params,
        "dataset_fits": dataset_fits,
        "dataset_summary": dataset_summary,
    }
    return fit_bundle, res


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
    json_payload = {
        "observable_mode": fit_bundle["observable_mode"],
        "global_keys": list(fit_bundle["global_keys"]),
        "local_keys": list(fit_bundle["local_keys"]),
        "best_global_params": _json_safe_dict(fit_bundle["best_global_params"]),
        "best_local_params": {k: _json_safe_dict(v) for k, v in fit_bundle["best_local_params"].items()},
        "optimizer_summary": optimizer_summary,
        "dataset_summary": fit_bundle["dataset_summary"],
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with summary_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset_name", "fluence_ratio", "n_points", "rms", "wrms", "dt_local_ps", "A_obs", "B_obs"],
        )
        writer.writeheader()
        writer.writerows(fit_bundle["dataset_summary"])

    fitcurve_paths = []
    for item in dataset_fits:
        dataset_token = _slugify_dataset_name(item["name"])
        curve_path = out_dir / f"fitcurve_{dataset_token}_{timestamp}.csv"
        stacked = np.column_stack([
            item["t"] * 1e12,
            item["S_exp"],
            item["S_fit"],
            item["Te_fit"],
            item["Ts_fit"],
            item["Tl_fit"],
            item["m_fit"],
            item["eta_fit"],
            item["residual"],
        ])
        np.savetxt(
            curve_path,
            stacked,
            delimiter=",",
            header="t_ps,S_exp,S_fit,Te_fit,Ts_fit,Tl_fit,m_fit,eta_fit,residual",
            comments="",
        )
        fitcurve_paths.append(str(curve_path))

    fig, ax = plt.subplots(figsize=(10, 6))
    for item in dataset_fits:
        label = f"{item['name']} ({item['fluence_ratio']:.1f} mW)"
        ax.scatter(item["t"] * 1e12, item["S_exp"], s=14, alpha=0.5, label=f"exp {label}")
        ax.plot(item["t"] * 1e12, item["S_fit"], linewidth=1.8, label=f"fit {label}")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("S")
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
    axes[1].plot(flu, [row["dt_local_ps"] for row in fit_bundle["dataset_summary"]], marker="o", label="dt_local_ps")
    axes[1].plot(flu, [row["A_obs"] for row in fit_bundle["dataset_summary"]], marker="s", label="A_obs")
    axes[1].plot(flu, [row["B_obs"] for row in fit_bundle["dataset_summary"]], marker="^", label="B_obs")
    axes[1].set_xlabel("fluence ratio / mW label")
    axes[1].set_title("Local parameters by dataset")
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
