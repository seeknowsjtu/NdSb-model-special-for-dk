# -*- coding: utf-8 -*-
"""
gui_component.py
----------------
A scrollable ttk.Frame used by the GUI.

Split out from ndsb_3tm_gui_magnon_compact.py.
"""
from __future__ import annotations
from dataclasses import dataclass

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None
    ttk = None


@dataclass(frozen=True)
class ParameterSpec:
    tab: str
    group: str
    key: str
    label: str
    value_type: str = "float"
    note: str = ""
    options: tuple[str, ...] = ()


PARAMETER_FORM_SPECS = (
    ParameterSpec("Basic", "Temperatures / hot steady-state init", "T_bath", "Bath temperature T_bath (K)"),
    ParameterSpec("Basic", "Temperatures / hot steady-state init", "use_hot_steady_init", "Use hot steady-state init (0/1)", value_type="int"),
    ParameterSpec(
        "Basic",
        "Temperatures / hot steady-state init",
        "hot_init_mode",
        "Hot init mode",
        value_type="str",
        note="Choose from menu",
        options=("manual", "avg_power"),
    ),
    ParameterSpec("Basic", "Temperatures / hot steady-state init", "T_init_eff", "Effective pre-pulse temperature T_init_eff (K)"),
    ParameterSpec("Basic", "Temperatures / hot steady-state init", "rep_rate_Hz", "Repetition rate (Hz)"),
    ParameterSpec("Basic", "Temperatures / hot steady-state init", "preheat_use_lattice_only", "Use lattice-only preheat balance (0/1)", value_type="int"),
    ParameterSpec("Basic", "Temperatures / hot steady-state init", "preheat_max_dT", "Max preheat rise above bath (K)"),
    ParameterSpec("Basic", "Pump", "fluence_multiplier", "Fluence multiplier"),
    ParameterSpec("Basic", "Pump", "delta_opt", "Optical depth δ (m)"),
    ParameterSpec("Basic", "Pump", "pulse_width", "Pulse FWHM (s)"),
    ParameterSpec("Basic", "Pump", "t0_pulse", "Pulse center t0 (s)"),
    ParameterSpec("Basic", "Pump", "S_scale", "Absorption scale S_scale"),
    ParameterSpec("Basic", "Pump", "A_obs", "Readout scale A_obs"),
    ParameterSpec("Basic", "Pump", "B0_obs", "Readout background B0_obs"),
    ParameterSpec("Basic", "Pump", "B1_obs", "Readout fluence slope B1_obs"),
    ParameterSpec("Basic", "Effective transfer channels", "G_el0", "G_e-l base (hot e → lattice)"),
    ParameterSpec("Basic", "Effective transfer channels", "G_es0", "G_e-s exch base (5d/6s ↔ 4f)"),
    ParameterSpec("Basic", "Effective transfer channels", "G_sl0", "G_s-l base (spin/order ↔ lattice)"),
    ParameterSpec("Basic", "Effective transfer channels", "G_es_floor_frac", "G_es floor fraction"),
    ParameterSpec("Basic", "Effective transfer channels", "G_es_m_power", "G_es m-power"),
    ParameterSpec("Basic", "Effective transfer channels", "G_sl_TR_boost", "G_sl boost near T_R"),
    ParameterSpec("Basic", "Effective transfer channels", "G_sl_TN_boost", "G_sl boost near T_N"),
    ParameterSpec("Basic", "Bath coupling / sinks", "tau_e_sink", "τ_e_sink (s)"),
    ParameterSpec("Basic", "Bath coupling / sinks", "tau_s_sink", "τ_s_sink (s)"),
    ParameterSpec("Basic", "Bath coupling / sinks", "tau_l_sink", "τ_l_sink (s)"),
    ParameterSpec("Basic", "Thermodynamics / electrons", "TN", "T_N (K)"),
    ParameterSpec("Basic", "Thermodynamics / electrons", "TR", "T_R (K)"),
    ParameterSpec("Basic", "Thermodynamics / electrons", "ThetaD", "Θ_D (K)"),
    ParameterSpec("Basic", "Thermodynamics / electrons", "gamma_PM_molar", "γ_PM (J/mol/K^2)"),
    ParameterSpec("Basic", "Thermodynamics / electrons", "alpha_gap", "α_gap"),
    ParameterSpec("Basic", "Thermodynamics / electrons", "gap0_meV", "Δ0 (meV)"),
    ParameterSpec("Basic", "Thermodynamics / electrons", "gap_eta_coupling", "gap-eta coupling"),
    ParameterSpec(
        "Basic",
        "Spin heat capacity",
        "sw_model",
        "C_sw model",
        value_type="str",
        note="Choose from menu",
        options=("magnon", "AT3"),
    ),
    ParameterSpec("Basic", "Spin heat capacity", "Cs_scale", "Cs scale"),
    ParameterSpec("Spin / Magnon", "J1/J2 renormalization", "J1_old_K", "J1 old (K)"),
    ParameterSpec("Spin / Magnon", "J1/J2 renormalization", "J2_old_K", "J2 old (K)"),
    ParameterSpec("Spin / Magnon", "J1/J2 renormalization", "J_renorm", "J renorm"),
    ParameterSpec("Spin / Magnon", "J1/J2 renormalization", "J_2q_scale", "J extra scale (T<TR)"),
    ParameterSpec("Spin / Magnon", "J1/J2 renormalization", "J_1q_scale", "J extra scale (TR<T<TN)"),
    ParameterSpec("Spin / Magnon", "Magnon integration (LSWT)", "S_eff", "S_eff"),
    ParameterSpec("Spin / Magnon", "Magnon integration (LSWT)", "mag_gap_meV", "Magnon gap / cutoff (meV)"),
    ParameterSpec("Spin / Magnon", "Magnon integration (LSWT)", "magnon_grid", "Magnon grid N (8–80)", value_type="int"),
    ParameterSpec("Spin / Magnon", "Legacy fallback (AT^3)", "A_sw_2q", "A_sw (T<TR) (J/mol/K^4)", note="Legacy AT^3 branch"),
    ParameterSpec("Spin / Magnon", "Legacy fallback (AT^3)", "A_sw_1q", "A_sw (TR<T<TN) (J/mol/K^4)", note="Legacy AT^3 branch"),
    ParameterSpec("Eta", "Eta switches / interpretation", "eta_enable", "eta_enable (0/1)", value_type="int"),
    ParameterSpec(
        "Eta",
        "Eta switches / interpretation",
        "eta_representation",
        "eta_representation",
        value_type="str",
        note="Choose from menu",
        options=("scalar", "cos2phi"),
    ),
    ParameterSpec(
        "Eta",
        "Eta switches / interpretation",
        "eta_mode",
        "eta_mode",
        value_type="str",
        note="Choose from menu",
        options=("second", "first"),
    ),
    ParameterSpec("Eta", "Eta switches / interpretation", "eta_sign", "eta_sign (+1/-1)", value_type="int", note="In cos2phi mode, selects phi≈0 vs phi≈pi/2 branch"),
    ParameterSpec("Eta", "Eta switches / interpretation", "eta_clip", "eta_clip"),
    ParameterSpec("Eta", "Eta kinetics", "Gamma_eta", "Gamma_eta (1/s)"),
    ParameterSpec("Eta", "Eta kinetics", "Gamma_eta_low_frac", "Gamma_eta_low_frac"),
    ParameterSpec("Eta", "Eta kinetics", "eta_dT", "eta_dT (K)"),
    ParameterSpec("Eta", "Eta Landau coefficients", "a_eta0", "a_eta0 / K(T,m) prefactor"),
    ParameterSpec("Eta", "Eta Landau coefficients", "b_eta", "b_eta / K4 (sin^4 phi)"),
    ParameterSpec("Eta", "Eta Landau coefficients", "c_eta", "c_eta (scalar first-order only)"),
    ParameterSpec("Eta", "Eta Landau coefficients", "g_m2eta2", "g_m2eta2 / gm"),
    ParameterSpec("Advanced", "Effective-coupling shaping", "G_el_Tpow", "G_el temperature power"),
    ParameterSpec("Advanced", "Effective-coupling shaping", "G_es_eta_coupling", "G_es eta coupling"),
    ParameterSpec("Advanced", "Effective-coupling shaping", "G_sl_TR_w", "G_sl width near T_R (K)"),
    ParameterSpec("Advanced", "Effective-coupling shaping", "G_sl_TN_w", "G_sl width near T_N (K)"),
    ParameterSpec("Advanced", "Effective-coupling shaping", "G_sl_eta_coupling", "G_sl eta coupling"),
    ParameterSpec("Advanced", "Heat-capacity features", "lambda_amp", "λ-peak amp (J/mol/K)"),
    ParameterSpec("Advanced", "Heat-capacity features", "lambda_w", "λ-peak width (K)"),
    ParameterSpec("Advanced", "Heat-capacity features", "latent_amp", "TR bump amp (J/mol/K)"),
    ParameterSpec("Advanced", "Heat-capacity features", "latent_w", "TR bump width (K)"),
    ParameterSpec("Advanced", "CEF (Schottky)", "cef_E1_meV", "CEF E1 (meV)"),
    ParameterSpec("Advanced", "CEF (Schottky)", "cef_E2_meV", "CEF E2 (meV)"),
    ParameterSpec("Advanced", "Order parameter dynamics", "tau_m0", "τ_m0 (s)"),
    ParameterSpec("Advanced", "Order parameter dynamics", "tau_m_crit_amp", "τ_m crit amp (s)"),
    ParameterSpec("Advanced", "Order parameter dynamics", "nu", "ν"),
    ParameterSpec("Advanced", "Order parameter dynamics", "eps_crit", "ε_crit"),
    ParameterSpec("Advanced", "Order parameter dynamics", "tau_m_max", "τ_m max (s)"),
    ParameterSpec("Advanced", "ARPES proxy: S(t)=offset + amp*m^power", "S_offset", "S_offset"),
    ParameterSpec("Advanced", "ARPES proxy: S(t)=offset + amp*m^power", "S_amp", "S_amp"),
    ParameterSpec("Advanced", "ARPES proxy: S(t)=offset + amp*m^power", "S_power", "S_power"),
)


class ScrollableFrame(ttk.Frame):  # type: ignore[misc]
    """A vertically scrollable frame (ttk) for long parameter forms."""
    def __init__(self, master, width=420, height=520, *args, **kwargs):
        if tk is None or ttk is None:
            raise RuntimeError("tkinter is required for ScrollableFrame.")
        super().__init__(master, *args, **kwargs)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, width=width, height=height)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.interior = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        def _on_interior_config(event=None):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            try:
                self.canvas.itemconfigure(self._win, width=self.canvas.winfo_width())
            except Exception:
                pass

        def _on_canvas_config(event=None):
            try:
                self.canvas.itemconfigure(self._win, width=event.width)
            except Exception:
                pass

        self.interior.bind("<Configure>", _on_interior_config)
        self.canvas.bind("<Configure>", _on_canvas_config)

        # mousewheel support (Windows/macOS/Linux)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-5>", self._on_mousewheel, add="+")
        self._active = False
        self.canvas.bind("<Enter>", lambda e: setattr(self, "_active", True))
        self.canvas.bind("<Leave>", lambda e: setattr(self, "_active", False))

    def _on_mousewheel(self, event):
        if not getattr(self, "_active", False):
            return
        try:
            if hasattr(event, "delta") and event.delta:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif getattr(event, "num", None) == 4:
                self.canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                self.canvas.yview_scroll(3, "units")
        except Exception:
            pass
