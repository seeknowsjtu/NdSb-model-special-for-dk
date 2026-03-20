# -*- coding: utf-8 -*-
"""
solver.py
---------
NdSb 3TM (+ m + eta/phi) model, ODE RHS, simulation helper, and energy diagnostics.

Split out from ndsb_3tm_gui_magnon_compact.py.
"""
from __future__ import annotations
import math
from typing import Optional, Dict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from config import (
    kB_meV, n_mol_vol,
    clipT,
    normalize_params_dict,
)
from physics_engine import (
    DebyeCl,
    schottky_C_molar,
    gaussian_peak_C_molar,
    magnon_C_molar,
    MagnonCvLUT,
    exchange_scale,
    spin_lattice_enhancement,
)


class NdSb3TM:
    def __init__(self, params: dict, debye_obj: Optional[DebyeCl] = None):
        self.p = normalize_params_dict(params)
        self.debye = debye_obj if debye_obj is not None else DebyeCl(thetaD=self.p["ThetaD"])

        # ---- magnon Cv LUT (optional acceleration) ----
        self._use_magnon = str(self.p.get("sw_model", "AT3")).strip().lower().startswith("mag")
        self._mag_lut_2q = None
        self._mag_lut_1q = None

        if self._use_magnon:
            try:
                Jren = float(self.p.get("J_renorm", 1.0))
                J2q = Jren * float(self.p.get("J_2q_scale", 1.0))
                J1q = Jren * float(self.p.get("J_1q_scale", 1.0))

                J1_old = float(self.p.get("J1_old_K", -0.0575))
                J2_old = float(self.p.get("J2_old_K", 0.1367))

                common = {
                    "S_eff": float(self.p.get("S_eff", 0.5)),
                    "gap_meV": float(self.p.get("mag_gap_meV", 0.0)),
                    "gridN": int(self.p.get("magnon_grid", 32)),
                    "n_branches": 2,
                    "T_min": 0.4,
                    "T_max": max(40.0, float(self.p.get("TN", 15.0))*5.0),
                    "nT": 220,
                }

                self._mag_lut_2q = MagnonCvLUT(J1_old*J2q, J2_old*J2q, **common)
                self._mag_lut_1q = MagnonCvLUT(J1_old*J1q, J2_old*J1q, **common)
            except Exception:
                self._mag_lut_2q = None
                self._mag_lut_1q = None

    # ---- pump ----
    def energy_density(self) -> float:
        flu = self.p["fluence_base_J_m2"] * self.p["fluence_multiplier"]  # J/m^2
        return (flu / self.p["delta_opt"]) * self.p["S_scale"]            # J/m^3

    def laser_S(self, t: float) -> float:
        sigma = max(float(self.p["pulse_width"]) / 2.355, 1e-18)
        edens = self.energy_density()
        x = (t - float(self.p["t0_pulse"])) / sigma
        return (edens / (math.sqrt(2.0*math.pi)*sigma)) * math.exp(-0.5*(x*x))

    def estimate_hot_steady_T_init(self) -> Dict[str, float]:
        """
        Estimate the effective pre-pulse sample temperature under repeated
        pumping. This only changes the initial condition; bath sink terms still
        relax toward the true external bath temperature T_bath.
        """
        Tb = float(self.p["T_bath"])
        Tmax = Tb + max(float(self.p.get("preheat_max_dT", 100.0)), 0.0)
        manual_T = float(np.clip(float(self.p.get("T_init_eff", Tb)), Tb, Tmax))
        mode = str(self.p.get("hot_init_mode", "manual")).strip().lower()

        if mode == "manual":
            return {"T_init_eff_used": manual_T}

        if mode != "avg_power":
            return {"T_init_eff_used": manual_T}

        P_avg = max(self.energy_density(), 0.0) * max(float(self.p.get("rep_rate_Hz", 0.0)), 0.0)
        tau_l = max(float(self.p.get("tau_l_sink", 1.0)), 1e-15)
        tol = max(float(self.p.get("preheat_solver_tol", 1e-6)), 1e-12)
        use_lattice_only = int(self.p.get("preheat_use_lattice_only", 1)) == 1

        def balance(T):
            T = float(np.clip(T, Tb, Tmax))
            if use_lattice_only:
                Ceff = self.Cl(T)
            else:
                Ceff = self.Cl(T) + self.Ce(T, T, m=self.m_eq(T), eta=self.eta_init(T)) + self.Cs(T)
            return Ceff * (T - Tb) / tau_l - P_avg

        if P_avg <= 0.0:
            return {"T_init_eff_used": Tb, "P_avg_preheat": P_avg}

        f_hi = balance(Tmax)
        if f_hi <= 0.0:
            return {"T_init_eff_used": Tmax, "P_avg_preheat": P_avg}

        try:
            root = root_scalar(balance, bracket=(Tb, Tmax), method="bisect", xtol=tol)
            if root.converged:
                T_used = float(np.clip(root.root, Tb, Tmax))
                return {"T_init_eff_used": T_used, "P_avg_preheat": P_avg}
        except Exception:
            pass

        return {"T_init_eff_used": manual_T, "P_avg_preheat": P_avg}

    # ---- order parameter equilibrium & dynamics ----
    def m_eq(self, Ts: float) -> float:
        TN = float(self.p["TN"])
        if Ts >= TN:
            return 0.0
        x = 1.0 - Ts / TN
        return math.sqrt(max(x, 0.0))

    def tau_m(self, Ts: float) -> float:
        TN = float(self.p["TN"])
        tau0 = float(self.p["tau_m0"])
        amp = float(self.p["tau_m_crit_amp"])
        nu = float(self.p["nu"])
        eps = float(self.p["eps_crit"])
        r = abs(1.0 - Ts / TN)
        slow = amp / ((r + eps)**nu)
        tau = tau0 + slow
        return float(np.clip(tau, 1e-13, float(self.p["tau_m_max"])))

    # ---- eta / phi representation helpers ----
    def eta_representation(self) -> str:
        rep = str(self.p.get("eta_representation", "scalar")).strip().lower()
        return "cos2phi" if rep in {"cos2phi", "phi", "angle", "director"} else "scalar"

    def _eta_gamma(self, Ts: float) -> float:
        TR = float(self.p["TR"])
        Gamma0 = float(self.p.get("Gamma_eta", 5e11))
        Gamma_low_frac = float(self.p.get("Gamma_eta_low_frac", 1e-3))
        dT = float(self.p.get("eta_dT", 0.3))
        s = 1.0 / (1.0 + math.exp((TR - Ts) / max(dT, 1e-6)))
        return float(Gamma0 * (Gamma_low_frac + (1.0 - Gamma_low_frac) * s))

    def _phi_wrap(self, phi: float) -> float:
        phi = float(phi)
        period = math.pi
        phi = math.fmod(phi, period)
        if phi < 0.0:
            phi += period
        return phi

    def _phi_to_eta(self, phi: float) -> float:
        return float(np.clip(math.cos(2.0 * self._phi_wrap(phi)), -1.0, 1.0))

    def _eta_to_phi_proxy(self, eta: float) -> float:
        eta_clip = float(np.clip(eta, -1.0, 1.0))
        return 0.5 * math.acos(eta_clip)

    # ---- eta potential (2nd/1st order) & dynamics ----
    def a_eta(self, Ts: float, m: float) -> float:
        TR = float(self.p["TR"])
        a0 = float(self.p.get("a_eta0", 1.0))
        g = float(self.p.get("g_m2eta2", 0.0))
        return a0 * ((float(Ts) - TR) / max(TR, 1e-12)) + g * (float(m)**2)

    def dF_deta(self, eta: float, Ts: float, m: float) -> float:
        a = self.a_eta(Ts, m)
        b = float(self.p.get("b_eta", 1.0))
        c = float(self.p.get("c_eta", 0.0))
        mode = str(self.p.get("eta_mode", "second")).strip().lower()

        if mode.startswith("first"):
            return a*eta - b*(eta**3) + c*(eta**5)
        else:
            return a*eta + b*(eta**3)

    def deta_dt(self, eta: float, Ts: float, m: float) -> float:
        if int(self.p.get("eta_enable", 1)) == 0:
            return 0.0

        Gamma = self._eta_gamma(Ts)
        return -Gamma * self.dF_deta(float(eta), float(Ts), float(m))

    def F_eta(self, eta: float, Ts: float, m: float) -> float:
        a = self.a_eta(Ts, m)
        b = float(self.p.get("b_eta", 1.0))
        c = float(self.p.get("c_eta", 0.0))
        mode = str(self.p.get("eta_mode", "second")).strip().lower()

        eta = float(eta)
        if mode.startswith("first"):
            return 0.5*a*eta**2 - 0.25*b*eta**4 + (1.0/6.0)*c*eta**6
        else:
            return 0.5*a*eta**2 + 0.25*b*eta**4

    def eta_eq(self, Ts: float, m: float, prefer_sign: int = +1) -> float:
        a = float(self.a_eta(Ts, m))
        b = float(self.p.get("b_eta", 1.0))
        c = float(self.p.get("c_eta", 0.0))
        mode = str(self.p.get("eta_mode", "second")).strip().lower()

        if not np.isfinite(a) or not np.isfinite(b) or (mode.startswith("first") and not np.isfinite(c)):
            return 0.0

        cand = [0.0]

        if mode.startswith("first"):
            if c <= 0:
                return 0.0
            disc = b*b - 4.0*a*c
            if disc >= 0.0:
                sdisc = math.sqrt(disc)
                for sgn in (+1.0, -1.0):
                    x = (b + sgn*sdisc) / (2.0*c)
                    if x > 0 and np.isfinite(x):
                        e = math.sqrt(x)
                        cand.extend([+e, -e])
        else:
            if b > 0.0 and a < 0.0:
                e = math.sqrt(max(0.0, -a/b))
                cand.extend([+e, -e])

        vals = [(self.F_eta(e, Ts, m), e) for e in cand]
        vals.sort(key=lambda x: x[0])
        Fmin = vals[0][0]
        near = [e for (Fv, e) in vals if abs(Fv - Fmin) <= 1e-12 * (1.0 + abs(Fmin))]

        prefer_sign = +1 if prefer_sign >= 0 else -1
        if len(near) >= 2:
            near_sorted = sorted(
                near,
                key=lambda e: (
                    0 if (e == 0) else (0 if (math.copysign(1.0, e) == prefer_sign) else 1),
                    abs(e),
                ),
            )
            return float(near_sorted[0])

        return float(near[0])

    def eta_init(self, Ts: float) -> float:
        if self.eta_representation() == "cos2phi":
            return self._phi_to_eta(self.phi_init(Ts))
        m0 = self.m_eq(Ts)
        prefer_sign = int(self.p.get("eta_sign", +1))
        eta0 = self.eta_eq(Ts, m0, prefer_sign=prefer_sign)
        clipv = float(self.p.get("eta_clip", 1.2))
        return float(np.clip(eta0, -clipv, clipv))

    # ---- angle-based reorientation model: eta = cos(2 phi) ----
    def F_phi(self, phi: float, Ts: float, m: float) -> float:
        K = self.a_eta(Ts, m)
        K4 = float(self.p.get("b_eta", 1.0))
        s2 = math.sin(self._phi_wrap(phi)) ** 2
        return float(K * s2 + K4 * (s2 ** 2))

    def dF_dphi(self, phi: float, Ts: float, m: float) -> float:
        K = self.a_eta(Ts, m)
        K4 = float(self.p.get("b_eta", 1.0))
        s = math.sin(self._phi_wrap(phi))
        c = math.cos(self._phi_wrap(phi))
        return float(2.0 * s * c * (K + 2.0 * K4 * (s ** 2)))

    def phi_eq(self, Ts: float, m: float, prefer_sign: int = +1) -> float:
        K = float(self.a_eta(Ts, m))
        K4 = float(self.p.get("b_eta", 1.0))
        prefer_sign = +1 if prefer_sign >= 0 else -1

        cand = [0.0, 0.5 * math.pi]
        if K4 > 0.0:
            x = -K / (2.0 * K4)
            if 0.0 < x < 1.0 and np.isfinite(x):
                phi = math.asin(math.sqrt(x))
                cand.extend([phi, math.pi - phi])

        vals = [(self.F_phi(phi, Ts, m), self._phi_wrap(phi)) for phi in cand]
        vals.sort(key=lambda x: x[0])
        Fmin = vals[0][0]
        near = [phi for (Fv, phi) in vals if abs(Fv - Fmin) <= 1e-12 * (1.0 + abs(Fmin))]

        if len(near) <= 1:
            return float(near[0])

        def branch_rank(phi: float):
            eta = self._phi_to_eta(phi)
            sign_rank = 0 if eta == 0.0 or math.copysign(1.0, eta) == prefer_sign else 1
            return (sign_rank, abs(phi - (0.0 if prefer_sign >= 0 else 0.5 * math.pi)))

        near_sorted = sorted(near, key=branch_rank)
        return float(near_sorted[0])

    def phi_init(self, Ts: float) -> float:
        m0 = self.m_eq(Ts)
        prefer_sign = int(self.p.get("eta_sign", +1))
        return self.phi_eq(Ts, m0, prefer_sign=prefer_sign)

    def dphi_dt(self, phi: float, Ts: float, m: float) -> float:
        if int(self.p.get("eta_enable", 1)) == 0:
            return 0.0
        Gamma = self._eta_gamma(Ts)
        return -Gamma * self.dF_dphi(float(phi), float(Ts), float(m))

    def _op_state_to_eta_phi(self, op_state: float) -> Dict[str, float]:
        if self.eta_representation() == "cos2phi":
            phi = self._phi_wrap(op_state)
            eta = self._phi_to_eta(phi)
            return {"phi": float(phi), "eta": float(eta), "op_state": float(phi)}
        eta_clip = float(self.p.get("eta_clip", 1.2))
        eta = float(np.clip(op_state, -eta_clip, eta_clip))
        phi = self._eta_to_phi_proxy(eta)
        return {"phi": float(phi), "eta": float(eta), "op_state": float(eta)}

    def _d_op_dt(self, op_state: float, Ts: float, m: float) -> float:
        if self.eta_representation() == "cos2phi":
            return self.dphi_dt(op_state, Ts, m)
        return self.deta_dt(op_state, Ts, m)

    # ---- electron specific heat ----
    def gap_meV(self, Ts: float, m: Optional[float] = None, eta: Optional[float] = None) -> float:
        m_use = self.m_eq(Ts) if m is None else float(np.clip(m, 0.0, 1.2))
        eta_use = 0.0 if eta is None else abs(float(eta))

        gap0 = float(self.p["gap0_meV"])
        eta_c = float(self.p.get("gap_eta_coupling", 0.0))

        g = gap0 * (m_use ** 2) * (1.0 + eta_c * eta_use)
        return float(max(0.0, g))

    def gamma_molar(self, Te: float, Ts: float, m: Optional[float] = None, eta: Optional[float] = None) -> float:
        g0 = float(self.p["gamma_PM_molar"])
        alpha = float(self.p["alpha_gap"])
        Te = max(float(Te), 1e-6)
        red = alpha * (self.gap_meV(Ts, m=m, eta=eta) / (kB_meV * Te))
        g = g0 * (1.0 - red)
        gmin = float(self.p["gamma_min_frac"]) * g0
        return float(np.clip(g, gmin, 5.0*g0))

    def Ce(self, Te: float, Ts: float, m: Optional[float] = None, eta: Optional[float] = None) -> float:
        Te = max(float(Te), 1e-6)
        gamma_vol = self.gamma_molar(Te, Ts, m=m, eta=eta) * n_mol_vol
        return float(max(gamma_vol * Te, 1e-12))

    # ---- lattice specific heat ----
    def Cl(self, Tl: float) -> float:
        return float(max(self.debye.C_vol(Tl), 1e-12))

    # ---- spin specific heat ----
    def Cs_molar(self, Ts: float) -> float:
        Ts = float(Ts)
        if not np.isfinite(Ts) or Ts <= 0:
            return 1e-12

        TN = float(self.p["TN"])
        TR = float(self.p["TR"])

        # Schottky tail from CEF levels
        E1 = float(self.p["cef_E1_meV"])
        E2 = float(self.p["cef_E2_meV"])
        g0 = int(self.p["cef_g0"])
        g1 = int(self.p["cef_g1"])
        g2 = int(self.p["cef_g2"])
        C_sch = schottky_C_molar(Ts, energies_meV=(0.0, E1, E2), degeneracies=(g0, g1, g2))

        # Spin-wave / magnon contribution below TN
        C_sw = 0.0
        if Ts < TN:
            sw_model = str(self.p.get("sw_model", "AT3")).strip().lower()
            if sw_model.startswith("mag"):
                if (Ts < TR) and (self._mag_lut_2q is not None):
                    C_sw = float(self._mag_lut_2q(Ts))
                elif (Ts >= TR) and (self._mag_lut_1q is not None):
                    C_sw = float(self._mag_lut_1q(Ts))
                else:
                    Jscale = float(self.p.get("J_renorm", 1.0))
                    Jscale *= float(self.p.get("J_2q_scale", 1.0)) if Ts < TR else float(self.p.get("J_1q_scale", 1.0))
                    J1 = float(self.p.get("J1_old_K", -0.0575)) * Jscale
                    J2 = float(self.p.get("J2_old_K", 0.1367)) * Jscale
                    C_sw = magnon_C_molar(
                        Ts,
                        J1_K=J1,
                        J2_K=J2,
                        S_eff=float(self.p.get("S_eff", 0.5)),
                        gap_meV=float(self.p.get("mag_gap_meV", 0.0)),
                        gridN=int(self.p.get("magnon_grid", 32)),
                    )
            else:
                A = float(self.p["A_sw_2q"]) if Ts < TR else float(self.p["A_sw_1q"])
                C_sw = max(0.0, A * Ts**3)

        # Peaks (Gaussian approximations)
        C_lambda = gaussian_peak_C_molar(Ts, TN, float(self.p["lambda_amp"]), float(self.p["lambda_w"]))
        C_latent = gaussian_peak_C_molar(Ts, TR, float(self.p["latent_amp"]), float(self.p["latent_w"]))

        C_tot = float(self.p["Cs_scale"]) * (C_sch + C_sw + C_lambda + C_latent)
        return float(max(C_tot, 1e-12))

    def Cs(self, Ts: float) -> float:
        return self.Cs_molar(Ts) * n_mol_vol  # J/m^3/K

    # ---- effective couplings ----
    def G_el_eff(self, Te: float, Ts: float, Tl: float, m: float, eta: float) -> float:
        """
        Hot-electron -> lattice main channel.
        First version: keep it simple as a constant base coupling.
        """
        return float(max(self.p.get("G_el0", 0.0), 0.0))

    def G_es_eff(self, Te: float, Ts: float, Tl: float, m: float, eta: float) -> float:
        """
        Effective exchange-mediated electron -> spin-sector auxiliary channel.
        Interpreted as 5d/6s <-> 4f exchange-assisted transfer.
        """
        G0 = float(max(self.p.get("G_es0", 0.0), 0.0))

        fac = exchange_scale(
            Ts=Ts,
            m=m,
            eta=eta,
            mode=str(self.p.get("G_es_mode", "m2")),
            floor_frac=float(self.p.get("G_es_floor_frac", 0.10)),
            m_power=float(self.p.get("G_es_m_power", 2.0)),
            eta_coupling=float(self.p.get("G_es_eta_coupling", 0.0)),
            TR=float(self.p.get("TR", 13.0)),
            TR_sharpness=float(self.p.get("G_es_TR_sharpness", 0.5)),
        )

        return float(G0 * fac)
    
    def G_sl_eff(self, Te: float, Ts: float, Tl: float, m: float, eta: float) -> float:
        """
        Main spin/order <-> lattice channel.
        Enhanced near TR and TN.
        """
        G0 = float(max(self.p.get("G_sl0", 0.0), 0.0))

        fac = spin_lattice_enhancement(
            Ts=Ts,
            eta=eta,
            TR=float(self.p.get("TR", 13.0)),
            TN=float(self.p.get("TN", 15.0)),
            TR_boost=float(self.p.get("G_sl_TR_boost", 0.0)),
            TR_width=float(self.p.get("G_sl_TR_w", 1.0)),
            TN_boost=float(self.p.get("G_sl_TN_boost", 0.0)),
            TN_width=float(self.p.get("G_sl_TN_w", 1.0)),
            eta_coupling=float(self.p.get("G_sl_eta_coupling", 0.0)),
            mode=str(self.p.get("G_sl_mode", "gaussian")),
        )

        return float(G0 * fac)

    def power_terms(self, t: float, Te: float, Ts: float, Tl: float, m: float, eta: float) -> Dict[str, float]:
        """
        Return effective couplings and power-transfer terms (W/m^3).
        Positive convention:
            P_es > 0 means energy flows e -> s
            P_el > 0 means energy flows e -> l
            P_sl > 0 means energy flows s -> l
        """
        Ges = self.G_es_eff(Te, Ts, Tl, m, eta)
        Gel = self.G_el_eff(Te, Ts, Tl, m, eta)
        Gsl = self.G_sl_eff(Te, Ts, Tl, m, eta)

        P_es = Ges * (Te - Ts)
        P_el = Gel * (Te - Tl)
        P_sl = Gsl * (Ts - Tl)

        return {
            "G_es_eff": float(Ges),
            "G_el_eff": float(Gel),
            "G_sl_eff": float(Gsl),
            "P_es": float(P_es),
            "P_el": float(P_el),
            "P_sl": float(P_sl),
        }

    def _clip_state(self, Te: float, Ts: float, Tl: float, m: float, op_state: float) -> Dict[str, float]:
        state = {
            "Te": clipT(Te, 1e-6, 8e4),
            "Ts": clipT(Ts, 1e-6, 5e4),
            "Tl": clipT(Tl, 1e-6, 5e4),
            "m": float(np.clip(m, 0.0, 1.2)),
        }
        if self.eta_representation() == "cos2phi":
            state["op_state"] = self._phi_wrap(op_state)
        else:
            eta_clip = float(self.p.get("eta_clip", 1.2))
            state["op_state"] = float(np.clip(op_state, -eta_clip, eta_clip))
        return state

    def _state_eval(self, t: float, Te: float, Ts: float, Tl: float, m: float, op_state: float) -> Dict[str, float]:
        state = self._clip_state(Te, Ts, Tl, m, op_state)
        Te = state["Te"]
        Ts = state["Ts"]
        Tl = state["Tl"]
        m = state["m"]
        op_state = state["op_state"]
        op_map = self._op_state_to_eta_phi(op_state)
        eta = op_map["eta"]
        phi = op_map["phi"]

        Ce = self.Ce(Te, Ts, m=m, eta=eta)
        Cs = self.Cs(Ts)
        Cl = self.Cl(Tl)
        pw = self.power_terms(t, Te, Ts, Tl, m, eta)

        Tb = float(self.p["T_bath"])
        tau_e = max(float(self.p["tau_e_sink"]), 1e-15)
        tau_s = max(float(self.p["tau_s_sink"]), 1e-15)
        tau_l = max(float(self.p["tau_l_sink"]), 1e-15)

        Qe = Ce * (Te - Tb) / tau_e
        Qs = Cs * (Ts - Tb) / tau_s
        Ql = Cl * (Tl - Tb) / tau_l
        source = self.laser_S(t)
        meq = self.m_eq(Ts)
        tau_m = self.tau_m(Ts)

        return {
            **state,
            "Ce": Ce,
            "Cs": Cs,
            "Cl": Cl,
            "Qe": Qe,
            "Qs": Qs,
            "Ql": Ql,
            "Qtot": Qe + Qs + Ql,
            "Sin": source,
            "meq": meq,
            "tau_m": tau_m,
            "phi": phi,
            "op_state": op_state,
            "dTe": (source - pw["P_es"] - pw["P_el"] - Qe) / Ce,
            "dTs": (pw["P_es"] - pw["P_sl"] - Qs) / Cs,
            "dTl": (pw["P_el"] + pw["P_sl"] - Ql) / Cl,
            "dm": -(m - meq) / tau_m,
            "deta": -2.0 * math.sin(2.0 * phi) * self.dphi_dt(phi, Ts, m) if self.eta_representation() == "cos2phi" else self.deta_dt(eta, Ts, m),
            "dphi": self.dphi_dt(phi, Ts, m) if self.eta_representation() == "cos2phi" else 0.0,
            "dop": self._d_op_dt(op_state, Ts, m),
            **pw,
            **op_map,
        }

    # ---- RHS ----
    def rhs(self, t: float, y):
        Te, Ts, Tl, m, op_state = y
        eval_state = self._state_eval(t, Te, Ts, Tl, m, op_state)
        return [eval_state["dTe"], eval_state["dTs"], eval_state["dTl"], eval_state["dm"], eval_state["dop"]]

    def simulate_aligned(self, t_eval, y0=None, method="Radau",
                         rtol=1e-5, atol=1e-8, max_step=None, with_diag: bool = False):
        init_info = {"T_init_eff_used": float(self.p["T_bath"])}
        if y0 is None:
            Tb = float(self.p["T_bath"])
            if int(self.p.get("use_hot_steady_init", 0)) == 1:
                init_info = self.estimate_hot_steady_T_init()
                T0_eff = float(init_info["T_init_eff_used"])
            else:
                T0_eff = Tb
            op0 = self.phi_init(T0_eff) if self.eta_representation() == "cos2phi" else self.eta_init(T0_eff)
            y0 = [T0_eff, T0_eff, T0_eff, self.m_eq(T0_eff), op0]
        else:
            init_info["T_init_eff_used"] = float(y0[0])

        t_eval = np.array(t_eval, dtype=float)
        t_eval = t_eval[np.isfinite(t_eval)]
        t_eval = np.unique(t_eval)
        if t_eval.size < 2:
            raise ValueError("t_eval needs at least two distinct time points (seconds).")

        t0, t1 = float(t_eval[0]), float(t_eval[-1])

        if max_step is None:
            pw = max(float(self.p["pulse_width"]), 1e-18)
            window = max(float(t1 - t0), 1e-15)
            max_step = min(max(pw/5.0, 2e-15), max(5e-14, window/400.0))

        sol = solve_ivp(
            self.rhs, (t0, t1), y0,
            method=method, t_eval=t_eval,
            rtol=rtol, atol=atol,
            max_step=max_step,
        )
        if sol.y.shape[1] != t_eval.size:
            raise RuntimeError(f"Solver returned {sol.y.shape[1]} points, expected {t_eval.size}")

        Te, Ts, Tl, m, op_state = sol.y
        m = np.clip(m, 0.0, 1.0)
        if self.eta_representation() == "cos2phi":
            phi = np.mod(op_state, math.pi)
            eta = np.clip(np.cos(2.0 * phi), -1.0, 1.0)
            aux = phi
        else:
            eta_clip = float(self.p.get("eta_clip", 1.2))
            eta = np.clip(op_state, -eta_clip, eta_clip)
            phi = np.array([self._eta_to_phi_proxy(v) for v in eta], dtype=float)
            aux = eta

        S_m = float(self.p["S_offset"]) + float(self.p["S_amp"]) * (m ** float(self.p["S_power"]))

        out = {
            "t": sol.t,
            "Te": Te,
            "Ts": Ts,
            "Tl": Tl,
            "m": m,
            "eta": eta,
            "phi": phi,
            "aux": aux,
            "eta_representation": self.eta_representation(),
            "S_m": S_m,
        }
        out["T_bath"] = float(self.p["T_bath"])
        out["T_init_eff_used"] = float(init_info["T_init_eff_used"])
        if "P_avg_preheat" in init_info:
            out["P_avg_preheat"] = float(init_info["P_avg_preheat"])

        out["diag"] = None
        if with_diag:
            try:
                out["diag"] = energy_diagnostics(self, out)
            except Exception:
                out["diag"] = None

        return out


def energy_diagnostics(model: NdSb3TM, sim: dict):
    """
    Energy-balance diagnostics for the 3TM (+m+eta) simulation.

    Checks:
        d/dt(Ue + Us + Ul) = S_in(t) - (Qe + Qs + Ql)

    Also records effective couplings and inter-reservoir power-transfer terms.

    Returns
    -------
    dict
        Contains:
            t, Sin, Qtot, Udot, mismatch,
            Ein, Eloss, dU, Emismatch,
            G_es_eff, G_el_eff, G_sl_eff,
            P_es, P_el, P_sl,
            closure_error_J_m3, rel_error,
            mismatch_rms_W_m3, mismatch_max_W_m3
    """
    t = np.asarray(sim["t"], dtype=float)
    Te = np.asarray(sim["Te"], dtype=float)
    Ts = np.asarray(sim["Ts"], dtype=float)
    Tl = np.asarray(sim["Tl"], dtype=float)
    m = np.asarray(sim["m"], dtype=float)
    op_state = np.asarray(sim.get("aux", sim.get("eta", np.zeros_like(t))), dtype=float)

    n = t.size
    if n < 2:
        raise ValueError("energy_diagnostics: need at least 2 time points.")

    Sin = np.empty(n, dtype=float)
    Qtot = np.empty(n, dtype=float)
    Udot = np.empty(n, dtype=float)
    mismatch = np.empty(n, dtype=float)

    G_es_eff = np.empty(n, dtype=float)
    G_el_eff = np.empty(n, dtype=float)
    G_sl_eff = np.empty(n, dtype=float)

    P_es = np.empty(n, dtype=float)
    P_el = np.empty(n, dtype=float)
    P_sl = np.empty(n, dtype=float)

    Qe_arr = np.empty(n, dtype=float)
    Qs_arr = np.empty(n, dtype=float)
    Ql_arr = np.empty(n, dtype=float)

    Ce_arr = np.empty(n, dtype=float)
    Cs_arr = np.empty(n, dtype=float)
    Cl_arr = np.empty(n, dtype=float)

    for i in range(n):
        state = model._state_eval(float(t[i]), Te[i], Ts[i], Tl[i], m[i], op_state[i])
        Ce = state["Ce"]
        Cs = state["Cs"]
        Cl = state["Cl"]
        Ud = Ce * state["dTe"] + Cs * state["dTs"] + Cl * state["dTl"]

        Sin[i] = state["Sin"]
        Qtot[i] = state["Qtot"]
        Udot[i] = Ud
        mismatch[i] = Ud - (state["Sin"] - state["Qtot"])

        G_es_eff[i] = state["G_es_eff"]
        G_el_eff[i] = state["G_el_eff"]
        G_sl_eff[i] = state["G_sl_eff"]

        P_es[i] = state["P_es"]
        P_el[i] = state["P_el"]
        P_sl[i] = state["P_sl"]

        Qe_arr[i] = state["Qe"]
        Qs_arr[i] = state["Qs"]
        Ql_arr[i] = state["Ql"]

        Ce_arr[i] = Ce
        Cs_arr[i] = Cs
        Cl_arr[i] = Cl

    def cumtrapz(y, x):
        y = np.asarray(y, float)
        x = np.asarray(x, float)
        out = np.zeros_like(y)
        dx = np.diff(x)
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)
        return out

    Ein = cumtrapz(Sin, t)
    Eloss = cumtrapz(Qtot, t)
    dU = cumtrapz(Udot, t)
    Emis = cumtrapz(mismatch, t)

    Ees = cumtrapz(P_es, t)
    Eel = cumtrapz(P_el, t)
    Esl = cumtrapz(P_sl, t)

    closure_error = float(dU[-1] - (Ein[-1] - Eloss[-1]))
    denom = max(abs(Ein[-1]) + abs(Eloss[-1]) + abs(dU[-1]), 1e-30)
    rel_error = float(closure_error / denom)

    return {
        "t": t,

        # source / sinks / balance
        "Sin": Sin,
        "Qtot": Qtot,
        "Qe": Qe_arr,
        "Qs": Qs_arr,
        "Ql": Ql_arr,
        "Udot": Udot,
        "mismatch": mismatch,

        # cumulative energies
        "Ein": Ein,
        "Eloss": Eloss,
        "dU": dU,
        "Emismatch": Emis,
        "E_es": Ees,
        "E_el": Eel,
        "E_sl": Esl,

        # effective couplings
        "G_es_eff": G_es_eff,
        "G_el_eff": G_el_eff,
        "G_sl_eff": G_sl_eff,

        # inter-reservoir power flow
        "P_es": P_es,
        "P_el": P_el,
        "P_sl": P_sl,

        # heat capacities for reference
        "Ce": Ce_arr,
        "Cs": Cs_arr,
        "Cl": Cl_arr,

        # summary numbers
        "closure_error_J_m3": closure_error,
        "rel_error": rel_error,
        "mismatch_rms_W_m3": float(np.sqrt(np.mean(mismatch * mismatch))),
        "mismatch_max_W_m3": float(np.max(np.abs(mismatch))),
    }
