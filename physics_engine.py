# -*- coding: utf-8 -*-
"""
physics_engine.py
-----------------
Thermodynamics and spin/magnon heat capacity models:
- Debye lattice Cv (cached PCHIP)
- Schottky CEF tail
- Gaussian peaks (lambda & latent-like)
- Magnon LSWT Cv + LUT accelerator

Also includes physics-informed helper functions for the NdSb effective model:
- exchange_scale(...)
- spin_lattice_enhancement(...)

Split out from ndsb_3tm_gui_magnon_compact.py.
"""
from __future__ import annotations
import math
from typing import Dict, Tuple, Optional, TypedDict

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

from config import R, NA, eV_J, kB_J, kB_meV, n_mol_vol

# ============================================================
# Debye lattice heat capacity (molar -> volumetric), cached
# ============================================================
_DEBYE_CACHE: Dict[float, Tuple[np.ndarray, np.ndarray, PchipInterpolator, float]] = {}


def _debye_integrand(x: float) -> float:
    # Stable form: x^4 * exp(-x) / (1-exp(-x))^2
    if x == 0.0:
        return 0.0
    u = math.exp(-x)
    denom = -math.expm1(-x)  # 1-exp(-x)
    return (x**4) * u / (denom * denom)


def _build_debye_interpolator(thetaD: float, T_min=0.5, T_max=2000.0, n=360):
    Ts = np.geomspace(T_min, T_max, n)
    Cm = np.empty_like(Ts)

    lowT_coeff = (12.0 * math.pi**4 / 5.0) * R  # molar Debye prefactor

    for i, T in enumerate(Ts):
        y = thetaD / T
        if y < 1e-5:
            Cm[i] = 3.0 * R
            continue
        upper = min(float(y), 220.0)
        val, _ = quad(lambda xx: _debye_integrand(xx), 0.0, upper,
                      epsabs=0, epsrel=1e-10, limit=250)
        Cm_i = 9.0 * R * (T/thetaD)**3 * val
        Cm[i] = min(max(Cm_i, 0.0), 3.0*R)

    interp = PchipInterpolator(Ts, Cm, extrapolate=False)
    return Ts, Cm, interp, lowT_coeff


class DebyeCl:
    def __init__(self, thetaD: float = 200.0):
        thetaD = float(thetaD)
        self.thetaD = thetaD
        key = float(thetaD)

        if key in _DEBYE_CACHE:
            self.Ts, self.Cm, self.interp, self.lowT_coeff = _DEBYE_CACHE[key]
        else:
            Ts, Cm, interp, lowT_coeff = _build_debye_interpolator(thetaD)
            _DEBYE_CACHE[key] = (Ts, Cm, interp, lowT_coeff)
            self.Ts, self.Cm, self.interp, self.lowT_coeff = _DEBYE_CACHE[key]

    def C_molar(self, T: float) -> float:
        T = float(T)
        if not np.isfinite(T) or T <= 0:
            T = 1e-9
        if T < self.Ts[0]:
            Cm = self.lowT_coeff * (T/self.thetaD)**3
        elif T > self.Ts[-1]:
            Cm = 3.0 * R
        else:
            Cm = float(self.interp(T))
            if not np.isfinite(Cm):
                Cm = 3.0*R if T > self.thetaD else self.lowT_coeff*(T/self.thetaD)**3
        return float(np.clip(Cm, 1e-12, 3.0*R))

    def C_vol(self, T: float) -> float:
        return self.C_molar(T) * n_mol_vol  # J/m^3/K


# ============================================================
# Schottky / peaks
# ============================================================
def schottky_C_molar(T: float,
                     energies_meV=(0.0, 1.6, 8.8),
                     degeneracies=(4, 2, 4)) -> float:
    """
    Multi-level Schottky heat capacity (molar), energies in meV.
    Cv_molar = NA * Var(E)_J / (kB_J T^2).
    Implemented in a numerically stable way by shifting energies.
    """
    T = float(T)
    if not np.isfinite(T) or T <= 0:
        return 0.0

    E = np.asarray(energies_meV, dtype=float)  # meV
    g = np.asarray(degeneracies, dtype=float)
    if E.size != g.size or E.size < 2:
        return 0.0

    beta = 1.0 / (kB_meV * T)  # 1/meV
    Emin = float(np.min(E))
    dE = (E - Emin)            # meV, non-negative
    x = -dE * beta             # <= 0
    w = g * np.exp(x)
    Z = float(np.sum(w))
    if not np.isfinite(Z) or Z <= 0:
        return 0.0
    p = w / Z

    m1 = float(np.sum(p * dE))
    m2 = float(np.sum(p * dE * dE))
    var_meV2 = max(0.0, m2 - m1*m1)

    # Convert variance: (meV)^2 -> J^2
    meV_J = 1e-3 * eV_J
    var_J2 = var_meV2 * (meV_J * meV_J)

    Cv = (NA * var_J2) / (kB_J * T*T)
    return float(max(Cv, 0.0))


def gaussian_peak_C_molar(T: float, T0: float, amp: float, width: float) -> float:
    """
    Simple Gaussian peak in heat capacity (molar): amp * exp(-(T-T0)^2/(2 w^2)).
    amp in J/mol/K, width in K.
    """
    T = float(T)
    if width <= 0:
        return 0.0
    x = (T - T0) / float(width)
    return float(max(0.0, amp * math.exp(-0.5 * x * x)))


# ============================================================
# Magnon (spin-wave) heat capacity from J1/J2 dispersion
# ============================================================
class _MagnonCommonKw(TypedDict):
    S_eff: float
    gap_meV: float
    gridN: int
    n_branches: int
    T_min: float
    T_max: float
    nT: int


_MAG_CACHE: Dict[tuple, np.ndarray] = {}  # key -> energies_meV ndarray (flattened)


def _fcc_J_of_k(kx, ky, kz, J1_K, J2_K):
    """
    Exchange Fourier transform J(k) for fcc (conventional cubic cell), in Kelvin units.
    kx,ky,kz in radians (i.e., k·a in the conventional cubic basis).
      - 12 NN at (±1/2,±1/2,0) etc  -> sum = 4[cos(kx/2)cos(ky/2)+...]
      - 6 NNN at (±1,0,0) etc       -> sum = 2[cos(kx)+cos(ky)+cos(kz)]
    """
    c1x = np.cos(0.5 * kx); c1y = np.cos(0.5 * ky); c1z = np.cos(0.5 * kz)
    term1 = c1x*c1y + c1y*c1z + c1z*c1x
    term2 = np.cos(kx) + np.cos(ky) + np.cos(kz)
    return 4.0 * J1_K * term1 + 2.0 * J2_K * term2


def _magnon_energies_meV(J1_K, J2_K, S_eff=0.5, gap_meV=0.0, gridN=32):
    """
    Precompute magnon energies on a uniform grid in the reduced BZ:
        h,k,l in [-0.5, 0.5)  (in 2π/a units).
    Returns flattened energies in meV (including a possible gap).
    """
    gridN = int(gridN)
    if gridN < 8:
        gridN = 8
    if gridN > 80:
        gridN = 80  # keep UI responsive

    key = (round(float(J1_K), 8), round(float(J2_K), 8),
           round(float(S_eff), 6), round(float(gap_meV), 6), int(gridN))
    if key in _MAG_CACHE:
        return _MAG_CACHE[key]

    h = (np.arange(gridN, dtype=float) / gridN) - 0.5
    kx, ky, kz = np.meshgrid(2*np.pi*h, 2*np.pi*h, 2*np.pi*h, indexing="ij")

    # Type-I ordering vector Q = (0,0,1) => add 2π to kz
    JQ = float(_fcc_J_of_k(0.0, 0.0, 2*np.pi, J1_K, J2_K))
    Jq = _fcc_J_of_k(kx, ky, kz, J1_K, J2_K)
    JqQ = _fcc_J_of_k(kx, ky, kz + 2*np.pi, J1_K, J2_K)

    # LSWT energy in Kelvin
    a = (JQ - Jq)
    b = (JQ - JqQ)
    Ek = 2.0 * float(S_eff) * np.sqrt(np.clip(a*b, 0.0, None))

    # Convert to meV and add anisotropy gap
    E = Ek * kB_meV
    if gap_meV and gap_meV > 0:
        E = np.sqrt(E*E + float(gap_meV)*float(gap_meV))

    E = E.astype(np.float64).ravel()
    E = E[np.isfinite(E)]
    _MAG_CACHE[key] = E
    return E


def magnon_C_molar(T, J1_K, J2_K, S_eff=0.5, gap_meV=0.0, gridN=32, n_branches=2):
    """
    Molar magnon heat capacity (J/mol/K) from LSWT energies on a grid.
    """
    T = float(T)
    if not np.isfinite(T) or T <= 0:
        return 0.0

    E = _magnon_energies_meV(J1_K, J2_K, S_eff=S_eff, gap_meV=gap_meV, gridN=gridN)
    if E.size == 0:
        return 0.0

    x = E / (kB_meV * T)
    exm1 = np.expm1(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        f = (x*x) * (exm1 + 1.0) / (exm1*exm1)

    f = f[np.isfinite(f)]
    if f.size == 0:
        return 0.0

    Cv = float(n_branches) * float(R) * float(np.mean(f))
    return float(max(Cv, 0.0))


class MagnonCvLUT:
    """
    Precompute magnon Cv(T) on a log-spaced temperature grid and interpolate.
    Accelerates stiff ODE + fitting.
    """
    def __init__(self, J1_K, J2_K, S_eff=0.5, gap_meV=0.0, gridN=32,
                 n_branches=2, T_min=0.4, T_max=80.0, nT=220):
        self.n_branches = float(n_branches)
        self.E = _magnon_energies_meV(J1_K, J2_K, S_eff=S_eff, gap_meV=gap_meV, gridN=gridN)
        if self.E.size == 0:
            self.Ts = np.array([T_min, T_max], float)
            self.Cv = np.zeros_like(self.Ts)
            self.interp = PchipInterpolator(self.Ts, self.Cv, extrapolate=True)
            return

        T_min = max(float(T_min), 1e-3)
        T_max = max(float(T_max), T_min*1.01)
        self.Ts = np.geomspace(T_min, T_max, int(nT)).astype(np.float64)

        E = self.E.astype(np.float64)
        Cv = np.empty_like(self.Ts)

        for i, T in enumerate(self.Ts):
            x = E / (kB_meV * T)
            exm1 = np.expm1(x)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                f = (x*x) * (exm1 + 1.0) / (exm1*exm1)
            small = (np.abs(x) < 1e-4)
            if np.any(small):
                f[small] = 1.0 - (x[small]*x[small]) / 12.0
            f = f[np.isfinite(f)]
            Cv[i] = float(self.n_branches) * float(R) * (float(np.mean(f)) if f.size else 0.0)

        self.Cv = np.clip(Cv, 0.0, 10.0*R)
        self.interp = PchipInterpolator(self.Ts, self.Cv, extrapolate=True)

    def __call__(self, T: float) -> float:
        T = float(T)
        if not np.isfinite(T) or T <= 0:
            return 0.0
        if T <= self.Ts[0]:
            return float(self.Cv[0])
        if T >= self.Ts[-1]:
            return float(self.Cv[-1])
        v = float(self.interp(T))
        return float(max(v, 0.0))


# ============================================================
# Physics-informed helper functions for effective couplings
# ============================================================

def exchange_scale(
    Ts: float,
    m: float,
    eta: float = 0.0,
    *,
    mode: str = "m2",
    floor_frac: float = 0.10,
    m_power: float = 2.0,
    eta_coupling: float = 0.0,
    TR: Optional[float] = None,
    TR_sharpness: float = 0.5,
) -> float:
    """
    Effective exchange-strength-like scaling factor for use in G_es_eff.

    Motivation:
    - In rare-earth 4f antiferromagnets, ultrafast angular momentum transfer
      is often discussed in relation to exchange / RKKY strength.
    - Here we encode a minimal phenomenological dependence on magnetic order m
      and optional eta-dependent enhancement.

    Parameters
    ----------
    Ts : float
        Spin-sector temperature (K). Currently only used if mode requests it.
    m : float
        Instantaneous magnetic order parameter.
    eta : float, optional
        Spin-reorientation / 1q-2q variable.
    mode : str
        Scaling mode:
          - "m2"       : floor + (1-floor)*m^p
          - "m"        : floor + (1-floor)*m
          - "const"    : 1
          - "m2_stepTR": m^p scaling times a smooth TR crossover
    floor_frac : float
        Minimal residual fraction when m -> 0.
    m_power : float
        Power p in m^p.
    eta_coupling : float
        Multiplicative enhancement factor via |eta|.
    TR : float or None
        Characteristic reorientation temperature (K), used in "m2_stepTR".
    TR_sharpness : float
        Width of smooth TR crossover in Kelvin.

    Returns
    -------
    float
        Dimensionless factor >= 0.
    """
    Ts = float(Ts)
    m = float(np.clip(m, 0.0, 1.0))
    eta = abs(float(eta))
    floor_frac = float(np.clip(floor_frac, 0.0, 1.0))
    m_power = float(max(m_power, 0.0))
    eta_coupling = float(max(eta_coupling, 0.0))
    TR_sharpness = float(max(TR_sharpness, 1e-6))

    mode = str(mode).strip().lower()

    if mode == "const":
        base = 1.0
    elif mode == "m":
        base = floor_frac + (1.0 - floor_frac) * m
    else:
        # default: "m2" and fallback
        base = floor_frac + (1.0 - floor_frac) * (m ** m_power)

    if mode == "m2_steptr" and TR is not None:
        # Smooth crossover around TR, useful if one wants a weak phase sensitivity
        s = 1.0 / (1.0 + math.exp((float(TR) - Ts) / TR_sharpness))
        # keep effect modest and positive
        base *= (0.75 + 0.25 * s)

    eta_fac = 1.0 + eta_coupling * eta
    eta_fac = max(0.0, eta_fac)

    return float(max(base * eta_fac, 0.0))


def spin_lattice_enhancement(
    Ts: float,
    eta: float = 0.0,
    *,
    TR: Optional[float] = None,
    TN: Optional[float] = None,
    TR_boost: float = 0.0,
    TR_width: float = 1.0,
    TN_boost: float = 0.0,
    TN_width: float = 1.0,
    eta_coupling: float = 0.0,
    mode: str = "gaussian",
) -> float:
    """
    Effective enhancement factor for spin/order <-> lattice coupling.

    Motivation:
    - NdSb likely has enhanced sensitivity of spin/order-lattice relaxation
      near the reorientation scale TR and the Néel temperature TN.
    - This helper returns a dimensionless multiplicative factor >= 1
      (unless eta_coupling is used with 0).

    Parameters
    ----------
    Ts : float
        Spin-sector temperature (K).
    eta : float
        Spin-reorientation variable.
    TR, TN : float or None
        Characteristic temperatures (K).
    TR_boost, TN_boost : float
        Peak amplitudes near TR and TN.
    TR_width, TN_width : float
        Characteristic widths in Kelvin.
    eta_coupling : float
        Optional multiplicative enhancement via |eta|.
    mode : str
        "gaussian" (default) or "lorentzian".

    Returns
    -------
    float
        Dimensionless factor >= 0.
    """
    Ts = float(Ts)
    eta = abs(float(eta))

    TR_boost = float(max(TR_boost, 0.0))
    TN_boost = float(max(TN_boost, 0.0))
    TR_width = float(max(TR_width, 1e-6))
    TN_width = float(max(TN_width, 1e-6))
    eta_coupling = float(max(eta_coupling, 0.0))
    mode = str(mode).strip().lower()

    def peak(T, T0, amp, w):
        if T0 is None:
            return 0.0
        x = (T - float(T0)) / w
        if mode.startswith("lor"):
            return amp / (1.0 + x * x)
        return amp * math.exp(-0.5 * x * x)

    fac = 1.0
    fac += peak(Ts, TR, TR_boost, TR_width)
    fac += peak(Ts, TN, TN_boost, TN_width)

    eta_fac = 1.0 + eta_coupling * eta
    eta_fac = max(0.0, eta_fac)

    return float(max(fac * eta_fac, 0.0))