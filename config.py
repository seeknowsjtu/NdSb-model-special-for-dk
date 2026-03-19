# -*- coding: utf-8 -*-
"""
config.py
---------
Constants, material parameters, default_params(), and small utilities.
Split out from ndsb_3tm_gui_magnon_compact.py.
"""
from __future__ import annotations
import numpy as np


LEGACY_PARAM_ALIASES = {
    "G_el": "G_el0",
    "G_es": "G_es0",
    "G_sl": "G_sl0",
}

# ============================================================
# Constants
# ============================================================
kB_J = 1.380649e-23
NA = 6.02214076e23
R = NA * kB_J
hbar_Js = 1.054571817e-34
eV_J = 1.602176634e-19
kB_eV = kB_J / eV_J
kB_meV = 1e3 * kB_eV  # meV/K

# ============================================================
# NdSb molar number density (NaCl: 4 f.u. per cubic cell)
# Using lattice constant near 20 K: a ≈ 6.319 Å
# ============================================================
a_lattice = 6.319e-10      # m
molar_mass = 266.0e-3      # kg/mol
rho = (4.0 * molar_mass / NA) / (a_lattice**3)   # kg/m^3
n_mol_vol = rho / molar_mass                      # mol/m^3


# ============================================================
# Defaults aligned to the user's report (updated TN/TR etc.)
# ============================================================
def default_params() -> dict:
    p: dict = {}

    # ========================================================
    # Thermodynamic anchors
    # ========================================================
    p["T_bath"] = 4.0
    p["TN"] = 15.0
    p["TR"] = 13.0
    p["ThetaD"] = 200.0

    # ========================================================
    # eta (spin reorientation / 1q-2q sector) dynamics
    # ========================================================
    p["eta_enable"] = 1
    p["eta_mode"] = "second"       # "second" or "first"
    p["Gamma_eta"] = 5e11          # 1/s
    p["Gamma_eta_low_frac"] = 1e-3
    p["eta_dT"] = 0.3

    # Landau-like eta free-energy coefficients
    # a(T,m) = a_eta0 * (Ts-TR)/TR + g_m2eta2 * m^2
    p["a_eta0"] = 2.0
    p["b_eta"] = 20.0
    p["c_eta"] = 200.0             # used only for first-order mode
    p["g_m2eta2"] = 0.0

    # Numerical clamp / sign convention
    p["eta_clip"] = 1.2
    p["eta_sign"] = +1

    # ========================================================
    # Electron thermodynamics
    # Ce = gamma(Te, Ts, m, eta) * Te   [volumetric]
    # ========================================================
    p["gamma_PM_molar"] = 2.0e-3   # J/mol/K^2
    p["alpha_gap"] = 0.0
    p["gap0_meV"] = 40.0
    p["gamma_min_frac"] = 0.2

    # Optional eta-dependent enhancement of the order-induced gap
    p["gap_eta_coupling"] = 0.0

    # ========================================================
    # Pump
    # ========================================================
    p["fluence_multiplier"] = 8.0
    p["fluence_base_J_m2"] = 100e-6 * 10000   # 100 uJ/cm^2 -> J/m^2
    p["delta_opt"] = 20e-9
    p["pulse_width"] = 120e-15
    p["t0_pulse"] = 0.0
    p["S_scale"] = 5e-4

    # ========================================================
    # Effective transfer channels (W/m^3/K)
    #
    # New NdSb-inspired interpretation:
    #   G_el0 : hot-electron -> lattice main channel
    #   G_es0 : exchange-mediated electron -> spin-sector auxiliary channel
    #   G_sl0 : spin/order <-> lattice main relaxation channel
    #
    # Legacy aliases are normalized at input boundaries via normalize_params_dict().
    # ========================================================

    # ---- base couplings ----
    p["G_el0"] = 5.0e13
    p["G_es0"] = 1.0e16
    p["G_sl0"] = 3.0e14

    # ---- optional temperature dependence for e-l channel ----
    # currently unused in solver first pass, but reserved
    p["G_el_Tpow"] = 0.0

    # ---- exchange-mediated e-s auxiliary channel ----
    # G_es_eff = G_es0 * [floor + (1-floor)*m^p] * (1 + c_eta*|eta|)
    p["G_es_floor_frac"] = 0.10
    p["G_es_m_power"] = 2.0
    p["G_es_eta_coupling"] = 0.0

    # ---- spin/order-lattice main channel ----
    # G_sl_eff = G_sl0 * [1 + TR boost + TN boost] * (1 + c_eta*|eta|)
    p["G_sl_TR_boost"] = 0.5
    p["G_sl_TR_w"] = 1.2
    p["G_sl_TN_boost"] = 0.5
    p["G_sl_TN_w"] = 1.5
    p["G_sl_eta_coupling"] = 0.0

    # ========================================================
    # Sinks to bath (stabilize long tail)
    # ========================================================
    p["tau_e_sink"] = 200e-12
    p["tau_s_sink"] = 5e-9
    p["tau_l_sink"] = 800e-12

    # ========================================================
    # Spin heat capacity sector
    # ========================================================
    # Choose model: "magnon" or "AT3"
    p["sw_model"] = "magnon"

    # Legacy empirical spin-wave coefficient: C_sw = A*T^3 (J/mol/K^4)
    p["A_sw_2q"] = 2.5e-3
    p["A_sw_1q"] = 1.8e-3

    # Exchange constants from neutron literature (Kelvin), with renormalization
    p["J1_old_K"] = -0.0575
    p["J2_old_K"] = 0.1367
    p["J_renorm"] = 15.0 / 13.6
    p["J_2q_scale"] = 1.00
    p["J_1q_scale"] = 1.00

    # Effective spin for LSWT heat capacity
    p["S_eff"] = 0.5

    # Magnon anisotropy / low-energy cutoff (effective)
    p["mag_gap_meV"] = 0.05

    # Magnon BZ sampling
    p["magnon_grid"] = 32

    # Overall spin-sector scaling
    p["Cs_scale"] = 1.0

    # Peaks near TN and TR
    p["lambda_amp"] = 3.0
    p["lambda_w"] = 0.7
    p["latent_amp"] = 2.0
    p["latent_w"] = 0.5

    # CEF energies (meV) & degeneracies
    p["cef_E1_meV"] = 1.6
    p["cef_E2_meV"] = 8.8
    p["cef_g0"] = 4
    p["cef_g1"] = 2
    p["cef_g2"] = 4

    # ========================================================
    # Order parameter dynamics
    # ========================================================
    p["tau_m0"] = 25e-12
    p["tau_m_crit_amp"] = 120e-12
    p["nu"] = 1.0
    p["eps_crit"] = 0.03
    p["tau_m_max"] = 400e-12

    # ARPES proxy
    # S(t) = offset + amp * m^power
    p["S_offset"] = 0.0
    p["S_amp"] = 1.0
    p["S_power"] = 2.0

    p["G_es_mode"] = "m2"
    p["G_es_TR_sharpness"] = 0.5
    p["G_sl_mode"] = "gaussian"

    return p

# ============================================================
# Utilities
# ============================================================
def clipT(T: float, Tmin: float = 1e-6, Tmax: float = 8e4) -> float:
    return float(np.clip(float(T), Tmin, Tmax))


def normalize_params_dict(params: dict | None) -> dict:
    """
    Normalize user/input parameter dictionaries so downstream code can rely on
    canonical internal keys only.

    Canonical effective-coupling keys:
        G_el0, G_es0, G_sl0

    Legacy aliases:
        G_el, G_es, G_sl
    """
    normalized = {} if params is None else dict(params)
    for legacy_key, canonical_key in LEGACY_PARAM_ALIASES.items():
        if canonical_key not in normalized and legacy_key in normalized:
            normalized[canonical_key] = normalized[legacy_key]
    return normalized


def safe_float(x, default):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def fmt_num(v) -> str:
    try:
        v = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(v):
        return str(v)
    av = abs(v)
    if av != 0 and (av < 1e-3 or av >= 1e4):
        return f"{v:.3e}"
    return f"{v:.6g}"
