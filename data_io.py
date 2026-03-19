# -*- coding: utf-8 -*-
"""
data_io.py
----------
CSV loader (auto-detect columns) and fitting utilities (bounded least squares).

Updated for the NdSb effective-coupling model:
- supports new fit keys: G_el0, G_es0, G_sl0
- supports state-dependent coupling parameters such as
  G_es_floor_frac, G_es_m_power, G_sl_TR_boost, G_sl_TN_boost, etc.
- keeps backward compatibility with legacy G_el / G_es / G_sl names
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

from config import LEGACY_PARAM_ALIASES, normalize_params_dict
from physics_engine import DebyeCl
from solver import NdSb3TM


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

    # time
    t_col = find_col(["tps", "t_ps", "time_ps", "t(ps)", "time(ps)", "time", "t"])
    if t_col is None:
        raise ValueError(f"Cannot find time column. Available columns: {names}")
    t_raw = np.array(arr[t_col], dtype=float)

    # temperatures
    te_col = find_col(["tek", "te_k", "te", "te(k)", "temp_e", "electron_temp"])
    ts_col = find_col(["tsk", "ts_k", "ts", "ts(k)", "temp_s", "spin_temp", "tspin"])
    tl_col = find_col(["tlk", "tl_k", "tl", "tl(k)", "temp_l", "lattice_temp"])

    Te = np.array(arr[te_col], dtype=float) if te_col is not None else None
    Ts = np.array(arr[ts_col], dtype=float) if ts_col is not None else None
    Tl = np.array(arr[tl_col], dtype=float) if tl_col is not None else None

    # S / intensity / spectral weight
    s_col = find_col(["s", "sw", "spec", "weight", "spectral_weight", "intensity", "amp"])
    S = np.array(arr[s_col], dtype=float) if s_col is not None else None

    # time unit auto-detect: if values look like ps -> convert to s
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

    # unique + average duplicates
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
        # ----------------------------------------------------
        # Effective couplings (new model)
        # ----------------------------------------------------
        if k in ["G_el", "G_el0"]:
            lb.append(1e11); ub.append(5e15)

        elif k in ["G_es", "G_es0"]:
            lb.append(1e13); ub.append(1e18)

        elif k in ["G_sl", "G_sl0"]:
            lb.append(1e12); ub.append(5e16)

        # Optional e-l temperature exponent
        elif k == "G_el_Tpow":
            lb.append(-3.0); ub.append(3.0)

        # e-s modulation
        elif k == "G_es_floor_frac":
            lb.append(0.0); ub.append(1.0)
        elif k == "G_es_m_power":
            lb.append(0.0); ub.append(6.0)
        elif k == "G_es_eta_coupling":
            lb.append(0.0); ub.append(5.0)

        # s-l enhancement near TR / TN
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

        # ----------------------------------------------------
        # Time constants / sinks
        # ----------------------------------------------------
        elif k == "tau_l_sink":
            lb.append(2e-10); ub.append(1e-8)
        elif k == "tau_e_sink":
            lb.append(1e-13); ub.append(5e-9)
        elif k == "tau_s_sink":
            lb.append(1e-12); ub.append(5e-8)

        # ----------------------------------------------------
        # Pump scale & timing
        # ----------------------------------------------------
        elif k == "S_scale":
            lb.append(1e-7); ub.append(1.0)
        elif k == "fluence_multiplier":
            lb.append(0.1); ub.append(100.0)
        elif k == "pulse_width":
            lb.append(10e-15); ub.append(5e-12)
        elif k == "t0_pulse":
            lb.append(-2e-12); ub.append(2e-12)

        # ----------------------------------------------------
        # Electron thermodynamics
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Spin heat capacity peak params
        # ----------------------------------------------------
        elif k.endswith("_amp"):
            lb.append(0.0); ub.append(50.0)
        elif k.endswith("_w"):
            lb.append(0.05); ub.append(5.0)

        # ----------------------------------------------------
        # Spin-wave / magnon sector
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # CEF levels / degeneracies
        # ----------------------------------------------------
        elif k in ["cef_E1_meV", "cef_E2_meV"]:
            lb.append(0.0); ub.append(100.0)
        elif k in ["cef_g0", "cef_g1", "cef_g2"]:
            lb.append(1); ub.append(20)

        # ----------------------------------------------------
        # Order parameter dynamics
        # ----------------------------------------------------
        elif k == "nu":
            lb.append(0.2); ub.append(3.0)
        elif k == "eps_crit":
            lb.append(0.001); ub.append(0.5)
        elif k.startswith("tau_m"):
            lb.append(1e-13); ub.append(5e-9)

        # eta dynamics / free-energy
        elif k == "Gamma_eta":
            lb.append(1e8); ub.append(1e14)
        elif k == "Gamma_eta_low_frac":
            lb.append(1e-6); ub.append(1.0)
        elif k == "eta_dT":
            lb.append(0.01); ub.append(10.0)
        elif k in ["a_eta0", "b_eta", "c_eta"]:
            lb.append(0.0); ub.append(1e4)
        elif k == "g_m2eta2":
            lb.append(-1e3); ub.append(1e3)
        elif k == "eta_clip":
            lb.append(0.1); ub.append(5.0)
        elif k == "eta_sign":
            lb.append(-1.0); ub.append(1.0)

        # ----------------------------------------------------
        # ARPES mapping
        # ----------------------------------------------------
        elif k == "S_offset":
            lb.append(-2.0); ub.append(2.0)
        elif k == "S_amp":
            lb.append(0.0); ub.append(10.0)
        elif k == "S_power":
            lb.append(0.5); ub.append(6.0)

        # ----------------------------------------------------
        # Thermodynamic anchors
        # ----------------------------------------------------
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


# ============================================================
# Fitting (bounded least squares)
# ============================================================
def fit_params(t, Te, S, p0, fit_keys, sigma_Te=2.0, sigma_S=0.02):
    """
    Fit using Te(t) and/or S(t). Times are seconds.

    Improvements:
      - log-parameterization for strictly-positive parameters
      - robust loss (soft_l1)
      - supports both legacy and new effective-coupling parameter sets
    """
    if Te is None and S is None:
        raise ValueError("Need at least Te or S data to fit.")

    p0 = normalize_params_dict(p0)
    fit_keys = normalize_fit_keys(fit_keys)

    debye_obj = DebyeCl(thetaD=float(p0["ThetaD"]))

    POS_KEYS = {
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
    }

    def _to_x(k, v):
        v = float(v)
        if k in POS_KEYS:
            return np.log10(max(v, 1e-300))
        return v

    def _from_x(k, x):
        x = float(x)
        if k in POS_KEYS:
            return float(10.0 ** x)
        return x

    def pack(p, keys):
        arr = []
        for k in keys:
            if k not in p:
                raise KeyError(f"fit_params: key '{k}' not found in parameter dict.")
            arr.append(_to_x(k, p[k]))
        return np.array(arr, dtype=float)

    def unpack(p, keys, x):
        for kk, xx in zip(keys, x):
            p[kk] = _from_x(kk, xx)

    lb_phys, ub_phys = _get_bounds_for_keys(fit_keys)
    lb = np.array([_to_x(k, v) for k, v in zip(fit_keys, lb_phys)], float)
    ub = np.array([_to_x(k, v) for k, v in zip(fit_keys, ub_phys)], float)

    x0 = pack(p0, fit_keys)
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
        unpack(p, fit_keys, x)

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
    unpack(p_best, fit_keys, res.x)
    return p_best, res
