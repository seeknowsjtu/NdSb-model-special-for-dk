# -*- coding: utf-8 -*-
"""
main.py
-------
GUI entry + CLI demo for NdSb 3TM (+m +eta).

Run:
    python main.py

Dependencies:
    numpy, scipy, matplotlib. Tkinter optional (GUI).
"""
from __future__ import annotations
import os
from typing import Any, cast, TYPE_CHECKING

import numpy as np

from config import default_params, safe_float, fmt_num
from solver import NdSb3TM
from data_io import load_csv_auto, fit_params

# ============================================================
# GUI availability
# ============================================================
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    GUI_AVAILABLE = False
    tk = None
    ttk = None
    filedialog = None
    messagebox = None
else:
    GUI_AVAILABLE = True

if TYPE_CHECKING:
    import tkinter as tk  # noqa: F811
    from tkinter import ttk, filedialog, messagebox  # noqa: F811

if GUI_AVAILABLE:
    assert tk is not None and ttk is not None and filedialog is not None and messagebox is not None
    tk = cast(Any, tk)
    ttk = cast(Any, ttk)
    filedialog = cast(Any, filedialog)
    messagebox = cast(Any, messagebox)

# Matplotlib backend
import matplotlib

if GUI_AVAILABLE:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        GUI_AVAILABLE = False
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")

from matplotlib.figure import Figure
if GUI_AVAILABLE:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui_component import ScrollableFrame


# ============================================================
# GUI App
# ============================================================
if GUI_AVAILABLE:
    class NdSb3TMApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("NdSb effective nonequilibrium model — simulate + fit")
            self.geometry("1460x900")

            self.p = default_params()
            self.p_fit = None
            self.fit_res = None
            self.current_view = "p"

            self.data: dict[str, Any] = {
                "t": None, "Te": None, "Ts": None, "Tl": None, "S": None, "path": None
            }

            self._build_ui()
            self._refresh_entries_from_params(self.p, view="p")
            self._plot_empty()
            self._log("Ready. Load CSV with headers like: tps, teK, tsK, tlK, S.")

        # ====================================================
        # UI build
        # ====================================================
        def _build_ui(self):
            self.columnconfigure(0, weight=0)
            self.columnconfigure(1, weight=1)
            self.rowconfigure(0, weight=1)

            left = ttk.Frame(self, padding=10)
            left.grid(row=0, column=0, sticky="nsw")
            left.rowconfigure(1, weight=1)
            left.columnconfigure(0, weight=1)

            right = ttk.Frame(self, padding=10)
            right.grid(row=0, column=1, sticky="nsew")
            right.rowconfigure(0, weight=1)
            right.columnconfigure(0, weight=1)

            header = ttk.Frame(left)
            header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
            ttk.Label(header, text="NdSb Model Parameters", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
            self.lbl_view = ttk.Label(header, text="VIEW: p", font=("Segoe UI", 10))
            self.lbl_view.grid(row=0, column=1, padx=12, sticky="w")

            nb = ttk.Notebook(left)
            nb.grid(row=1, column=0, sticky="nsew", pady=(0, 6))

            tab_basic = ttk.Frame(nb)
            tab_spin = ttk.Frame(nb)
            tab_eta = ttk.Frame(nb)
            tab_adv = ttk.Frame(nb)

            nb.add(tab_basic, text="Basic")
            nb.add(tab_spin, text="Spin / Magnon")
            nb.add(tab_eta, text="Eta")
            nb.add(tab_adv, text="Advanced")

            sf_basic = ScrollableFrame(tab_basic, width=500, height=450)
            sf_spin = ScrollableFrame(tab_spin, width=500, height=450)
            sf_eta = ScrollableFrame(tab_eta, width=500, height=450)
            sf_adv = ScrollableFrame(tab_adv, width=500, height=450)

            sf_basic.pack(fill="both", expand=True)
            sf_spin.pack(fill="both", expand=True)
            sf_eta.pack(fill="both", expand=True)
            sf_adv.pack(fill="both", expand=True)

            self.entries: dict[str, Any] = {}

            def add_entry(parent, key, label, width_label=34, width_entry=22):
                row = ttk.Frame(parent)
                row.pack(fill="x", pady=2)
                ttk.Label(row, text=label, width=width_label).pack(side="left")
                ent = ttk.Entry(row, width=width_entry)
                ent.pack(side="left", padx=(6, 0))
                self.entries[key] = ent
                return ent

            def add_section(parent, title):
                ttk.Separator(parent).pack(fill="x", pady=(8, 6))
                ttk.Label(parent, text=title, font=("Segoe UI", 10, "bold")).pack(anchor="w")

            # ---------------- BASIC ----------------
            add_section(sf_basic.interior, "Pump")
            add_entry(sf_basic.interior, "fluence_multiplier", "Fluence multiplier")
            add_entry(sf_basic.interior, "delta_opt", "Optical depth δ (m)")
            add_entry(sf_basic.interior, "pulse_width", "Pulse FWHM (s)")
            add_entry(sf_basic.interior, "t0_pulse", "Pulse center t0 (s)")
            add_entry(sf_basic.interior, "S_scale", "Absorption scale S_scale")

            add_section(sf_basic.interior, "Effective transfer channels")
            add_entry(sf_basic.interior, "G_el0", "G_e-l base (hot e → lattice)")
            add_entry(sf_basic.interior, "G_es0", "G_e-s exch base (5d/6s ↔ 4f)")
            add_entry(sf_basic.interior, "G_sl0", "G_s-l base (spin/order ↔ lattice)")
            add_entry(sf_basic.interior, "G_es_floor_frac", "G_es floor fraction")
            add_entry(sf_basic.interior, "G_es_m_power", "G_es m-power")
            add_entry(sf_basic.interior, "G_sl_TR_boost", "G_sl boost near T_R")
            add_entry(sf_basic.interior, "G_sl_TN_boost", "G_sl boost near T_N")

            add_section(sf_basic.interior, "Bath coupling / sinks")
            add_entry(sf_basic.interior, "tau_e_sink", "τ_e_sink (s)")
            add_entry(sf_basic.interior, "tau_s_sink", "τ_s_sink (s)")
            add_entry(sf_basic.interior, "tau_l_sink", "τ_l_sink (s)")

            add_section(sf_basic.interior, "Thermodynamics / electrons")
            add_entry(sf_basic.interior, "TN", "T_N (K)")
            add_entry(sf_basic.interior, "TR", "T_R (K)")
            add_entry(sf_basic.interior, "ThetaD", "Θ_D (K)")
            add_entry(sf_basic.interior, "gamma_PM_molar", "γ_PM (J/mol/K^2)")
            add_entry(sf_basic.interior, "alpha_gap", "α_gap")
            add_entry(sf_basic.interior, "gap0_meV", "Δ0 (meV)")
            add_entry(sf_basic.interior, "gap_eta_coupling", "gap-eta coupling")

            add_section(sf_basic.interior, "Spin heat capacity")
            add_entry(sf_basic.interior, "sw_model", "C_sw model (magnon/AT3)")
            add_entry(sf_basic.interior, "Cs_scale", "Cs scale")

            # ---------------- SPIN / MAGNON ----------------
            add_section(sf_spin.interior, "J1/J2 renormalization")
            add_entry(sf_spin.interior, "J1_old_K", "J1 old (K)")
            add_entry(sf_spin.interior, "J2_old_K", "J2 old (K)")
            add_entry(sf_spin.interior, "J_renorm", "J renorm")
            add_entry(sf_spin.interior, "J_2q_scale", "J extra scale (T<TR)")
            add_entry(sf_spin.interior, "J_1q_scale", "J extra scale (TR<T<TN)")

            add_section(sf_spin.interior, "Magnon integration (LSWT)")
            add_entry(sf_spin.interior, "S_eff", "S_eff")
            add_entry(sf_spin.interior, "mag_gap_meV", "Magnon gap / cutoff (meV)")
            add_entry(sf_spin.interior, "magnon_grid", "Magnon grid N (8–80)")

            add_section(sf_spin.interior, "Legacy fallback (AT^3)")
            add_entry(sf_spin.interior, "A_sw_2q", "A_sw (T<TR) (J/mol/K^4)")
            add_entry(sf_spin.interior, "A_sw_1q", "A_sw (TR<T<TN) (J/mol/K^4)")

            # ---------------- ETA ----------------
            add_section(sf_eta.interior, "Eta switches / interpretation")
            add_entry(sf_eta.interior, "eta_enable", "eta_enable (0/1)")
            add_entry(sf_eta.interior, "eta_mode", "eta_mode (second/first)")
            add_entry(sf_eta.interior, "eta_sign", "eta_sign (+1/-1)")
            add_entry(sf_eta.interior, "eta_clip", "eta_clip")

            add_section(sf_eta.interior, "Eta kinetics")
            add_entry(sf_eta.interior, "Gamma_eta", "Gamma_eta (1/s)")
            add_entry(sf_eta.interior, "Gamma_eta_low_frac", "Gamma_eta_low_frac")
            add_entry(sf_eta.interior, "eta_dT", "eta_dT (K)")

            add_section(sf_eta.interior, "Eta Landau coefficients")
            add_entry(sf_eta.interior, "a_eta0", "a_eta0")
            add_entry(sf_eta.interior, "b_eta", "b_eta")
            add_entry(sf_eta.interior, "c_eta", "c_eta (first-order only)")
            add_entry(sf_eta.interior, "g_m2eta2", "g_m2eta2")

            # ---------------- ADVANCED ----------------
            add_section(sf_adv.interior, "Effective-coupling shaping")
            add_entry(sf_adv.interior, "G_el_Tpow", "G_el temperature power")
            add_entry(sf_adv.interior, "G_es_eta_coupling", "G_es eta coupling")
            add_entry(sf_adv.interior, "G_sl_TR_w", "G_sl width near T_R (K)")
            add_entry(sf_adv.interior, "G_sl_TN_w", "G_sl width near T_N (K)")
            add_entry(sf_adv.interior, "G_sl_eta_coupling", "G_sl eta coupling")

            add_section(sf_adv.interior, "Heat-capacity features")
            add_entry(sf_adv.interior, "lambda_amp", "λ-peak amp (J/mol/K)")
            add_entry(sf_adv.interior, "lambda_w", "λ-peak width (K)")
            add_entry(sf_adv.interior, "latent_amp", "TR bump amp (J/mol/K)")
            add_entry(sf_adv.interior, "latent_w", "TR bump width (K)")

            add_section(sf_adv.interior, "CEF (Schottky)")
            add_entry(sf_adv.interior, "cef_E1_meV", "CEF E1 (meV)")
            add_entry(sf_adv.interior, "cef_E2_meV", "CEF E2 (meV)")

            add_section(sf_adv.interior, "Order parameter dynamics")
            add_entry(sf_adv.interior, "tau_m0", "τ_m0 (s)")
            add_entry(sf_adv.interior, "tau_m_crit_amp", "τ_m crit amp (s)")
            add_entry(sf_adv.interior, "nu", "ν")
            add_entry(sf_adv.interior, "eps_crit", "ε_crit")
            add_entry(sf_adv.interior, "tau_m_max", "τ_m max (s)")

            add_section(sf_adv.interior, "ARPES proxy: S(t)=offset + amp*m^power")
            add_entry(sf_adv.interior, "S_offset", "S_offset")
            add_entry(sf_adv.interior, "S_amp", "S_amp")
            add_entry(sf_adv.interior, "S_power", "S_power")

            # ---------------- Simulation controls ----------------
            ctrl = ttk.Frame(left)
            ctrl.grid(row=2, column=0, sticky="ew")

            ttk.Label(ctrl, text="Simulation time", font=("Segoe UI", 10, "bold")).grid(
                row=0, column=0, columnspan=6, sticky="w", pady=(4, 2)
            )
            self.ent_t0ps = ttk.Entry(ctrl, width=8)
            self.ent_t0ps.insert(0, "-5")
            self.ent_t1ps = ttk.Entry(ctrl, width=8)
            self.ent_t1ps.insert(0, "200")
            self.ent_npts = ttk.Entry(ctrl, width=8)
            self.ent_npts.insert(0, "2500")

            ttk.Label(ctrl, text="t0 (ps)").grid(row=1, column=0, sticky="w")
            self.ent_t0ps.grid(row=1, column=1, padx=3, sticky="w")
            ttk.Label(ctrl, text="t1 (ps)").grid(row=1, column=2, sticky="w")
            self.ent_t1ps.grid(row=1, column=3, padx=3, sticky="w")
            ttk.Label(ctrl, text="N").grid(row=1, column=4, sticky="w")
            self.ent_npts.grid(row=1, column=5, padx=3, sticky="w")

            # ---------------- Buttons ----------------
            btnfrm = ttk.Frame(left)
            btnfrm.grid(row=3, column=0, sticky="ew", pady=6)
            for i in range(4):
                btnfrm.columnconfigure(i, weight=1)
            ttk.Button(btnfrm, text="Simulate", command=self.on_simulate).grid(row=0, column=0, padx=3, pady=2, sticky="ew")
            ttk.Button(btnfrm, text="Load CSV...", command=self.on_load_csv).grid(row=0, column=1, padx=3, pady=2, sticky="ew")
            ttk.Button(btnfrm, text="Show Main (p)", command=self.on_show_p).grid(row=0, column=2, padx=3, pady=2, sticky="ew")
            ttk.Button(btnfrm, text="Show Fit (p_fit)", command=self.on_show_pfit).grid(row=0, column=3, padx=3, pady=2, sticky="ew")

            fitfrm = ttk.Frame(left)
            fitfrm.grid(row=4, column=0, sticky="ew", pady=6)
            for i in range(4):
                fitfrm.columnconfigure(i, weight=1)
            ttk.Button(fitfrm, text="Fit Te", command=self.on_fit_te).grid(row=0, column=0, padx=3, pady=2, sticky="ew")
            ttk.Button(fitfrm, text="Fit S", command=self.on_fit_s).grid(row=0, column=1, padx=3, pady=2, sticky="ew")
            ttk.Button(fitfrm, text="Fit Te + S", command=self.on_fit_both).grid(row=0, column=2, padx=3, pady=2, sticky="ew")
            ttk.Button(fitfrm, text="Apply Fit → Params", command=self.on_apply_fit).grid(row=0, column=3, padx=3, pady=2, sticky="ew")

            ttk.Separator(left).grid(row=5, column=0, sticky="ew", pady=8)

            ttk.Label(left, text="Status", font=("Segoe UI", 10, "bold")).grid(row=6, column=0, sticky="w")
            self.txt = tk.Text(left, width=66, height=11)
            self.txt.grid(row=7, column=0, sticky="ew")

            # ---------------- Plots (2x2) ----------------
            self.fig = Figure(figsize=(10.2, 7.8), dpi=100)
            gs = self.fig.add_gridspec(2, 2)

            self.axT = self.fig.add_subplot(gs[0, 0])   # temperatures
            self.axM = self.fig.add_subplot(gs[0, 1])   # m / eta / S
            self.axG = self.fig.add_subplot(gs[1, 0])   # effective couplings
            self.axP = self.fig.add_subplot(gs[1, 1])   # power flows

            self.fig.tight_layout(pad=2.0)

            self.canvas = FigureCanvasTkAgg(self.fig, master=right)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # ====================================================
        # Helpers
        # ====================================================
        def _log(self, msg: str):
            self.txt.insert("end", msg + "\n")
            self.txt.see("end")

        def _set_view_label(self):
            self.lbl_view.config(text=f"VIEW: {self.current_view}")

        def _current_params_dict(self):
            if self.current_view == "p_fit" and self.p_fit is not None:
                return self.p_fit
            return self.p

        def _read_entries_to_params(self):
            target = self._current_params_dict()
            for k, ent in self.entries.items():
                s = ent.get().strip()
                if s == "":
                    continue

                if k in ["sw_model", "eta_mode"]:
                    target[k] = s
                    continue

                if k in ["magnon_grid", "eta_enable", "eta_sign"]:
                    try:
                        target[k] = int(round(float(s)))
                    except Exception:
                        self._log(f"[warn] cannot parse {k}='{s}', keep {target.get(k)}")
                    continue

                try:
                    target[k] = float(s)
                except Exception:
                    self._log(f"[warn] cannot parse {k}='{s}', keep {target.get(k)}")

            for k in ["cef_g0", "cef_g1", "cef_g2"]:
                if k in target:
                    try:
                        target[k] = int(round(float(target[k])))
                    except Exception:
                        pass

            # backward compatibility aliases
            if "G_el0" in target:
                target["G_el"] = target["G_el0"]
            if "G_es0" in target:
                target["G_es"] = target["G_es0"]
            if "G_sl0" in target:
                target["G_sl"] = target["G_sl0"]

            if self.current_view == "p_fit":
                self.p_fit = dict(target)

        def _refresh_entries_from_params(self, p_dict, view: str):
            self.current_view = view
            self._set_view_label()
            for k, ent in self.entries.items():
                ent.delete(0, "end")
                if k in p_dict:
                    ent.insert(0, fmt_num(p_dict.get(k)))

        def _time_grid(self):
            t0ps = safe_float(self.ent_t0ps.get(), -5.0)
            t1ps = safe_float(self.ent_t1ps.get(), 200.0)
            n = int(max(50, round(safe_float(self.ent_npts.get(), 2500))))
            if t1ps <= t0ps:
                t1ps = t0ps + 1.0
            return np.linspace(t0ps, t1ps, n) * 1e-12

        # ====================================================
        # Plotting
        # ====================================================
        def _plot_empty(self):
            self.axT.clear()
            self.axM.clear()
            self.axG.clear()
            self.axP.clear()

            self.axT.set_title("Temperatures")
            self.axT.set_xlabel("time (ps)")
            self.axT.set_ylabel("T (K)")
            self.axT.grid(alpha=0.3)

            self.axM.set_title("Order parameter / proxy")
            self.axM.set_xlabel("time (ps)")
            self.axM.set_ylabel("m / eta / S")
            self.axM.grid(alpha=0.3)

            self.axG.set_title("Effective couplings")
            self.axG.set_xlabel("time (ps)")
            self.axG.set_ylabel("G_eff (W/m$^3$/K)")
            self.axG.grid(alpha=0.3)

            self.axP.set_title("Power flow")
            self.axP.set_xlabel("time (ps)")
            self.axP.set_ylabel("P (W/m$^3$)")
            self.axP.grid(alpha=0.3)

            self.canvas.draw()

        def _plot_sim(self, sim, params_used: dict):
            tps = sim["t"] * 1e12
            TN = float(params_used["TN"])
            TR = float(params_used["TR"])

            # -------- temperatures --------
            self.axT.clear()
            self.axT.plot(tps, sim["Te"], label="Te (hot e)")
            self.axT.plot(tps, sim["Ts"], label="Ts (spin-sector)")
            self.axT.plot(tps, sim["Tl"], label="Tl (lattice)", linestyle="--")
            self.axT.axhline(TN, linestyle=":", label="TN")
            self.axT.axhline(TR, linestyle="--", label="TR")
            self.axT.set_title("Temperatures")
            self.axT.set_xlabel("time (ps)")
            self.axT.set_ylabel("T (K)")
            self.axT.grid(alpha=0.3)

            # -------- order / proxy --------
            self.axM.clear()
            self.axM.plot(tps, sim["m"], label="m(t)")
            self.axM.plot(tps, sim["eta"], label="eta(t)")
            self.axM.plot(tps, sim["S_m"], label="S(t)=offset+amp*m^p")
            self.axM.set_title("Order parameter / proxy")
            self.axM.set_xlabel("time (ps)")
            self.axM.set_ylabel("m / eta / S")
            self.axM.grid(alpha=0.3)

            # -------- diag: effective couplings & power --------
            self.axG.clear()
            self.axP.clear()

            diag = sim.get("diag", None)
            if diag is not None:
                self.axG.plot(tps, diag["G_el_eff"], label="G_el_eff")
                self.axG.plot(tps, diag["G_es_eff"], label="G_es_eff")
                self.axG.plot(tps, diag["G_sl_eff"], label="G_sl_eff")
                self.axG.set_title("Effective couplings")
                self.axG.set_xlabel("time (ps)")
                self.axG.set_ylabel("G_eff (W/m$^3$/K)")
                self.axG.grid(alpha=0.3)

                self.axP.plot(tps, diag["P_el"], label="P_el")
                self.axP.plot(tps, diag["P_es"], label="P_es")
                self.axP.plot(tps, diag["P_sl"], label="P_sl")
                self.axP.set_title("Power flow")
                self.axP.set_xlabel("time (ps)")
                self.axP.set_ylabel("P (W/m$^3$)")
                self.axP.grid(alpha=0.3)
            else:
                self.axG.set_title("Effective couplings")
                self.axG.set_xlabel("time (ps)")
                self.axG.set_ylabel("G_eff (W/m$^3$/K)")
                self.axG.grid(alpha=0.3)

                self.axP.set_title("Power flow")
                self.axP.set_xlabel("time (ps)")
                self.axP.set_ylabel("P (W/m$^3$)")
                self.axP.grid(alpha=0.3)

            # -------- data overlay --------
            if self.data["t"] is not None:
                td = self.data["t"] * 1e12
                if self.data["Te"] is not None:
                    self.axT.scatter(td, self.data["Te"], s=20, marker="o", label="Te data")
                if self.data["Ts"] is not None:
                    self.axT.scatter(td, self.data["Ts"], s=20, marker="^", label="Ts data")
                if self.data["Tl"] is not None:
                    self.axT.scatter(td, self.data["Tl"], s=20, marker="s", label="Tl data")
                if self.data["S"] is not None:
                    self.axM.scatter(td, self.data["S"], s=20, marker="o", label="S data")

            self.axT.legend(loc="best", fontsize=8)
            self.axM.legend(loc="best", fontsize=8)
            if diag is not None:
                self.axG.legend(loc="best", fontsize=8)
                self.axP.legend(loc="best", fontsize=8)

            self.fig.tight_layout(pad=2.0)
            self.canvas.draw()

        # ====================================================
        # Buttons
        # ====================================================
        def on_show_p(self):
            self._refresh_entries_from_params(self.p, view="p")
            self._log("[view] switched to main parameters (p).")

        def on_show_pfit(self):
            if self.p_fit is None:
                messagebox.showwarning("No fit yet", "Please run a fit first.")
                return
            self._refresh_entries_from_params(self.p_fit, view="p_fit")
            self._log("[view] switched to fitted parameters (p_fit).")

        def on_simulate(self):
            try:
                self._read_entries_to_params()
                t = self._time_grid()
                params_used = dict(self._current_params_dict())
                model = NdSb3TM(params_used)
                sim = model.simulate_aligned(t)
                self._plot_sim(sim, params_used)

                self._log(
                    f"[sim:{self.current_view}] max Te={np.max(sim['Te']):.2f} K | "
                    f"max Ts={np.max(sim['Ts']):.2f} K | max Tl={np.max(sim['Tl']):.2f} K | "
                    f"final m={sim['m'][-1]:.3f} | final eta={sim['eta'][-1]:.3f}"
                )

                diag = sim.get("diag", None)
                if diag is not None:
                    self._log(
                        "[energy] Ein={:.3e} J/m^3 | Eloss={:.3e} J/m^3 | dU={:.3e} J/m^3".format(
                            diag["Ein"][-1], diag["Eloss"][-1], diag["dU"][-1]
                        )
                    )
                    self._log(
                        "[energy] closure_error={:.3e} J/m^3 | rel_error={:.3e} | mismatch_rms={:.3e} W/m^3 | mismatch_max={:.3e} W/m^3".format(
                            diag["closure_error_J_m3"], diag["rel_error"],
                            diag["mismatch_rms_W_m3"], diag["mismatch_max_W_m3"]
                        )
                    )
            except Exception as e:
                messagebox.showerror("Simulate error", str(e))
                self._log(f"[error] {e}")

        def on_apply_fit(self):
            if self.p_fit is None:
                messagebox.showwarning("No fit yet", "Please run a fit first.")
                return
            self.p = dict(self.p_fit)
            self._refresh_entries_from_params(self.p, view="p")
            self._log("[apply] fitted params applied to main parameters (p).")
            self.on_simulate()

        def on_load_csv(self):
            path = filedialog.askopenfilename(
                title="Choose CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not path:
                return

            try:
                t, Te, Ts, Tl, S, names, unit = load_csv_auto(path)
                self.data.update({"t": t, "Te": Te, "Ts": Ts, "Tl": Tl, "S": S, "path": path})

                self._log(f"[load] {os.path.basename(path)} | time unit: {unit} | columns: {names}")
                self._log(
                    f"[load] N={len(t)} | Te={'yes' if Te is not None else 'no'} | "
                    f"Ts={'yes' if Ts is not None else 'no'} | "
                    f"Tl={'yes' if Tl is not None else 'no'} | "
                    f"S={'yes' if S is not None else 'no'}"
                )
            except Exception as e:
                messagebox.showerror("Load CSV failed", str(e))
                self._log(f"[error] {e}")

        def _fit(self, mode: str):
            if self.data["t"] is None:
                messagebox.showwarning("No data", "Please load a CSV first.")
                return

            self._read_entries_to_params()
            t = self.data["t"]
            Te = self.data["Te"]
            S = self.data["S"]

            if mode == "Te" and Te is None:
                messagebox.showwarning("No Te", "CSV does not contain Te column.")
                return
            if mode == "S" and S is None:
                messagebox.showwarning("No S", "CSV does not contain S column.")
                return
            if mode == "Both" and (Te is None or S is None):
                messagebox.showwarning("Missing", "Need both Te and S columns for Fit Te+S.")
                return

            if mode == "Te":
                fit_keys = [
                    "G_el0", "G_es0",
                    "S_scale", "t0_pulse"
                ]
                Te_fit, S_fit = Te, None

            elif mode == "S":
                fit_keys = [
                    "tau_m0", "tau_m_crit_amp", "nu",
                    "S_offset", "S_amp", "S_power",
                    "t0_pulse"
                ]
                Te_fit, S_fit = None, S

            else:
                fit_keys = [
                    "G_el0", "G_es0", "G_sl0",
                    "G_es_m_power",
                    "G_sl_TR_boost", "G_sl_TN_boost",
                    "tau_l_sink", "S_scale",
                    "tau_m0", "tau_m_crit_amp", "nu",
                    "S_offset", "S_amp", "S_power",
                    "t0_pulse"
                ]
                Te_fit, S_fit = Te, S

            p_start = dict(self._current_params_dict())

            try:
                self._log(f"[fit-{mode}] keys={fit_keys}")
                p_best, res = fit_params(t, Te_fit, S_fit, p_start, fit_keys, sigma_Te=2.0, sigma_S=0.02)

                self.p_fit = dict(p_best)
                self.fit_res = res

                self._refresh_entries_from_params(self.p_fit, view="p_fit")

                self._log(f"[fit-{mode}] success: cost={res.cost:.3e}, nfev={res.nfev}, status={res.status}")
                for k in fit_keys:
                    self._log(f"    {k} = {fmt_num(self.p_fit[k])}")

                t_sim = self._time_grid()
                model = NdSb3TM(self.p_fit)
                sim = model.simulate_aligned(t_sim)
                self._plot_sim(sim, self.p_fit)

                self._log(
                    f"[fit-{mode}] preview: max Te={np.max(sim['Te']):.2f} K | "
                    f"max Ts={np.max(sim['Ts']):.2f} K | final m={sim['m'][-1]:.3f}"
                )

            except Exception as e:
                messagebox.showerror("Fit error", str(e))
                self._log(f"[error] {e}")

        def on_fit_te(self):
            self._fit("Te")

        def on_fit_s(self):
            self._fit("S")

        def on_fit_both(self):
            self._fit("Both")


def _cli_demo():
    """Headless quick demo: run a short simulation and print maxima."""
    p = default_params()
    model = NdSb3TM(p)
    t = np.linspace(-2e-12, 100e-12, 1200)
    sim = model.simulate_aligned(t)
    print("GUI not available (tkinter missing). Headless demo:")
    print("  max Te =", float(np.max(sim["Te"])))
    print("  max Ts =", float(np.max(sim["Ts"])))
    print("  max Tl =", float(np.max(sim["Tl"])))
    print("  final m =", float(sim["m"][-1]))
    print("  final eta =", float(sim["eta"][-1]))


if __name__ == "__main__":
    if GUI_AVAILABLE:
        app = NdSb3TMApp()
        app.mainloop()
    else:
        _cli_demo()