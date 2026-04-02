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

from config import (
    MULTI_FIT_DEFAULT_GLOBAL_KEYS,
    MULTI_FIT_DEFAULT_LOCAL_KEYS,
    MULTI_FIT_DEFAULT_OBSERVABLE_MODE,
    default_params,
    safe_float,
    fmt_num,
    normalize_params_dict,
)
from solver import NdSb3TM
from data_io import (
    export_multi_fit_results,
    fit_params,
    fit_params_multi,
    fit_params_multi_dk,
    load_csv_auto,
    load_dk_dataset_csv,
    load_s_dataset_csv,
    normalize_fit_keys,
    preprocess_signal_baseline,
)

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

from gui_component import ScrollableFrame, PARAMETER_FORM_SPECS


FIT_PRESETS = {
    "Te": ["G_el0", "G_es0", "S_scale", "t0_pulse"],
    # Legacy single-dataset S_m proxy fits (separate from multi-fit observable_mode).
    "S": ["tau_m0", "tau_m_crit_amp", "nu", "S_offset", "S_amp", "S_power", "t0_pulse"],
    "Both": [
        "G_el0", "G_es0", "G_sl0",
        "G_es_m_power",
        "G_sl_TR_boost", "G_sl_TN_boost",
        "tau_l_sink", "S_scale",
        "tau_m0", "tau_m_crit_amp", "nu",
        "S_offset", "S_amp", "S_power",
        "t0_pulse",
    ],
}


MULTI_FIT_PRESET = {
    "target_kind": "S",
    "global_keys": list(MULTI_FIT_DEFAULT_GLOBAL_KEYS),
    "local_keys": list(MULTI_FIT_DEFAULT_LOCAL_KEYS),
    "observable_mode": MULTI_FIT_DEFAULT_OBSERVABLE_MODE,
}

MULTI_FIT_DK_PRESET = {
    "target_kind": "delta_k",
    "global_keys": [k for k in MULTI_FIT_DEFAULT_GLOBAL_KEYS if k not in {"A_obs", "B_obs"}] + ["K_dk"],
    "local_keys": [],
    "observable_mode": "dk_chi2q",
}


# ============================================================
# GUI App
# ============================================================
if GUI_AVAILABLE:
    class NdSb3TMApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("NdSb effective nonequilibrium model — simulate + fit")
            self.geometry("1460x900")

            self.p = normalize_params_dict(default_params())
            self.p_fit = None
            self.fit_res = None
            self.current_view = "p"

            self.data: dict[str, Any] = {
                "t": None, "Te": None, "Ts": None, "Tl": None, "S": None, "path": None
            }
            self.datasets: list[dict[str, Any]] = []
            self.multi_fit_params = None
            self.multi_fit_res = None
            self.multi_fit_exports = None

            self._build_ui()
            self._refresh_entries_from_params(self.p, view="p")
            self._plot_empty()
            self._log("Ready. Load CSV with headers like: tps, teK, tsK, tlK, S.")
            self._log("Multi-fit preset: observable=raw_m_chi2q | local_keys=[] (A/B via varpro).")

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
            self.entry_specs: dict[str, Any] = {}
            self.entry_vars: dict[str, Any] = {}

            def add_entry(parent, key, label, width_label=34, width_entry=22):
                row = ttk.Frame(parent)
                row.pack(fill="x", pady=2)
                ttk.Label(row, text=label, width=width_label).pack(side="left")
                spec = next((item for item in PARAMETER_FORM_SPECS if item.key == key), None)
                if spec is not None and getattr(spec, "options", ()):
                    var = tk.StringVar(value="")
                    ent = ttk.Combobox(
                        row,
                        width=width_entry,
                        textvariable=var,
                        values=spec.options,
                        state="readonly",
                    )
                    self.entry_vars[key] = var
                else:
                    ent = ttk.Entry(row, width=width_entry)
                ent.pack(side="left", padx=(6, 0))
                self.entries[key] = ent
                return ent

            def add_section(parent, title):
                ttk.Separator(parent).pack(fill="x", pady=(8, 6))
                ttk.Label(parent, text=title, font=("Segoe UI", 10, "bold")).pack(anchor="w")

            tab_map = {
                "Basic": sf_basic.interior,
                "Spin / Magnon": sf_spin.interior,
                "Eta": sf_eta.interior,
                "Advanced": sf_adv.interior,
            }
            last_group_by_tab: dict[str, str | None] = {key: None for key in tab_map}
            for spec in PARAMETER_FORM_SPECS:
                parent = tab_map[spec.tab]
                if last_group_by_tab[spec.tab] != spec.group:
                    add_section(parent, spec.group)
                    last_group_by_tab[spec.tab] = spec.group
                label = spec.label if not spec.note else f"{spec.label} [{spec.note}]"
                ent = add_entry(parent, spec.key, label)
                self.entry_specs[spec.key] = spec

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
            ttk.Button(btnfrm, text="Load CSVs...", command=self.on_load_csvs).grid(row=0, column=2, padx=3, pady=2, sticky="ew")
            ttk.Button(btnfrm, text="Clear Datasets", command=self.on_clear_datasets).grid(row=0, column=3, padx=3, pady=2, sticky="ew")
            ttk.Button(btnfrm, text="Show Main (p)", command=self.on_show_p).grid(row=1, column=0, padx=3, pady=2, sticky="ew")
            ttk.Button(btnfrm, text="Show Fit (p_fit)", command=self.on_show_pfit).grid(row=1, column=1, padx=3, pady=2, sticky="ew")
            ttk.Button(btnfrm, text="Fit Multi-S", command=self.on_fit_multi_s).grid(row=1, column=2, padx=3, pady=2, sticky="ew")
            ttk.Button(btnfrm, text="Export Multi-Fit", command=self.on_export_multi_fit).grid(row=1, column=3, padx=3, pady=2, sticky="ew")
            ttk.Button(btnfrm, text="Load delta-k CSVs...", command=self.on_load_dk_csvs).grid(row=2, column=2, padx=3, pady=2, sticky="ew")
            ttk.Button(btnfrm, text="Fit Multi-delta-k", command=self.on_fit_multi_dk).grid(row=2, column=3, padx=3, pady=2, sticky="ew")

            fitfrm = ttk.Frame(left)
            fitfrm.grid(row=4, column=0, sticky="ew", pady=6)
            for i in range(4):
                fitfrm.columnconfigure(i, weight=1)
            ttk.Button(fitfrm, text="Fit Te", command=self.on_fit_te).grid(row=0, column=0, padx=3, pady=2, sticky="ew")
            ttk.Button(fitfrm, text="Fit S (legacy m-proxy)", command=self.on_fit_s).grid(row=0, column=1, padx=3, pady=2, sticky="ew")
            ttk.Button(fitfrm, text="Fit Te + S (legacy S)", command=self.on_fit_both).grid(row=0, column=2, padx=3, pady=2, sticky="ew")
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
            self.update_idletasks()

        def _set_view_label(self):
            self.lbl_view.config(text=f"VIEW: {self.current_view}")

        def _current_params_dict(self):
            if self.current_view == "p_fit" and self.p_fit is not None:
                return self.p_fit
            return self.p

        def _read_entries_to_params(self):
            target = dict(self._current_params_dict())
            for k, ent in self.entries.items():
                raw_value = self.entry_vars[k].get() if k in self.entry_vars else ent.get()
                s = raw_value.strip()
                if s == "":
                    continue

                spec = self.entry_specs.get(k)
                value_type = getattr(spec, "value_type", "float")

                if value_type == "str":
                    target[k] = s
                    continue

                if value_type == "int":
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

            target = normalize_params_dict(target)

            if self.current_view == "p_fit":
                self.p_fit = dict(target)
            else:
                self.p = dict(target)

        def _refresh_entries_from_params(self, p_dict, view: str):
            self.current_view = view
            self._set_view_label()
            p_dict = normalize_params_dict(p_dict)
            for k, ent in self.entries.items():
                if k in self.entry_vars:
                    self.entry_vars[k].set("")
                else:
                    ent.delete(0, "end")
                if k in p_dict:
                    value = fmt_num(p_dict.get(k))
                    if k in self.entry_vars:
                        self.entry_vars[k].set(value)
                    else:
                        ent.insert(0, value)

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
            if np.isfinite(np.asarray(sim.get("phi", []), dtype=float)).any():
                self.axM.plot(tps, sim["phi"], label="phi(t) [rad]", linestyle="--", alpha=0.8)
            self.axM.plot(tps, sim["S_m"], label="S(t)=offset+amp*m^p")
            self.axM.set_title("Order parameter / proxy")
            self.axM.set_xlabel("time (ps)")
            self.axM.set_ylabel("m / eta / phi / S")
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

        def _plot_multi_fit_preview(self, fit_bundle):
            """Preview the latest multi-dataset fit on the existing 2x2 canvas."""
            dataset_fits = fit_bundle.get("dataset_fits", [])
            if not dataset_fits:
                self._plot_empty()
                return

            first = dataset_fits[0]
            tps = first["t"] * 1e12
            global_p = fit_bundle.get("best_global_params", {})
            TN = float(global_p.get("TN", np.nan))
            TR = float(global_p.get("TR", np.nan))

            self.axT.clear()
            self.axT.plot(tps, first["Te_fit"], label=f"{first['name']} Te_fit")
            self.axT.plot(tps, first["Ts_fit"], label=f"{first['name']} Ts_fit")
            self.axT.plot(tps, first["Tl_fit"], label=f"{first['name']} Tl_fit", linestyle="--")
            self.axT.axhline(TN, linestyle=":", label="TN")
            self.axT.axhline(TR, linestyle="--", label="TR")
            self.axT.set_title(f"Temperatures (preview: {first['name']})")
            self.axT.set_xlabel("time (ps)")
            self.axT.set_ylabel("T (K)")
            self.axT.grid(alpha=0.3)
            self.axT.legend(loc="best", fontsize=8)

            target_kind = str(fit_bundle.get("target_kind", "S"))
            self.axM.clear()
            for item in dataset_fits:
                label = f"{item['name']} | {item['fluence_ratio']:.1f}"
                t_ps = item["t"] * 1e12
                if target_kind == "delta_k":
                    y_exp = np.asarray(item["delta_k_exp"], dtype=float)
                    y_fit = np.asarray(item["delta_k_fit"], dtype=float)
                    resolved = np.asarray(item.get("is_resolved", np.ones_like(y_exp, dtype=bool)), dtype=bool)
                    self.axM.scatter(t_ps[resolved], y_exp[resolved], s=12, alpha=0.55, label=f"exp(resolved) {label}")
                    if np.any(~resolved):
                        self.axM.scatter(t_ps[~resolved], y_exp[~resolved], s=12, alpha=0.25, marker="x", label=f"exp(unresolved) {label}")
                    self.axM.plot(t_ps, y_fit, linewidth=1.6, label=f"fit {label}")
                else:
                    self.axM.scatter(t_ps, item["S_exp"], s=12, alpha=0.45, label=f"exp {label}")
                    self.axM.plot(t_ps, item["S_fit"], linewidth=1.6, label=f"fit {label}")
            self.axM.set_title(f"Multi-{target_kind} global fit ({fit_bundle['observable_mode']})")
            self.axM.set_xlabel("time (ps)")
            self.axM.set_ylabel("delta_k" if target_kind == "delta_k" else "S")
            self.axM.grid(alpha=0.3)
            self.axM.legend(loc="best", fontsize=7, ncol=2)

            self.axG.clear()
            diag = first.get("diag")
            if diag is not None:
                self.axG.plot(tps, diag["G_el_eff"], label="G_el_eff")
                self.axG.plot(tps, diag["G_es_eff"], label="G_es_eff")
                self.axG.plot(tps, diag["G_sl_eff"], label="G_sl_eff")
                self.axG.legend(loc="best", fontsize=8)
            self.axG.set_title("Effective couplings (preview first dataset)")
            self.axG.set_xlabel("time (ps)")
            self.axG.set_ylabel("G_eff (W/m$^3$/K)")
            self.axG.grid(alpha=0.3)

            self.axP.clear()
            flu = [item["fluence_ratio"] for item in dataset_fits]
            wrms = [item["wrms"] for item in dataset_fits]
            self.axP.plot(flu, wrms, marker="o", label="WRMS")
            self.axP.set_title("Multi-fit dataset summary")
            self.axP.set_xlabel("fluence ratio")
            self.axP.set_ylabel("WRMS")
            self.axP.grid(alpha=0.3)
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
                params_used = normalize_params_dict(self._current_params_dict())
                model = NdSb3TM(params_used)
                sim = model.simulate_aligned(t, with_diag=True)
                self._plot_sim(sim, params_used)

                self._log(
                    f"[init] T_bath={sim['T_bath']:.2f} K | "
                    f"T_init_eff_used={sim['T_init_eff_used']:.2f} K"
                    + (
                        f" | P_avg_preheat={sim['P_avg_preheat']:.3e} W/m^3"
                        if "P_avg_preheat" in sim else ""
                    )
                )

                self._log(
                    f"[sim:{self.current_view}] max Te={np.max(sim['Te']):.2f} K | "
                    f"max Ts={np.max(sim['Ts']):.2f} K | max Tl={np.max(sim['Tl']):.2f} K | "
                    f"final m={sim['m'][-1]:.3f} | final eta={sim['eta'][-1]:.3f}"
                    + (
                        f" | final phi={sim['phi'][-1]:.3f} rad"
                        if sim.get("eta_representation", "scalar") == "cos2phi" else ""
                    )
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
                baseline_value = None
                baseline_npts = None
                baseline_method = None
                if S is not None:
                    S_raw = np.asarray(S, dtype=float)
                    S, baseline_value, baseline_npts, baseline_method = preprocess_signal_baseline(t, S_raw)
                    self.data["S_raw"] = S_raw
                self.data.update({"t": t, "Te": Te, "Ts": Ts, "Tl": Tl, "S": S, "path": path})

                self._log(f"[load] {os.path.basename(path)} | time unit: {unit} | columns: {names}")
                self._log(
                    f"[load] N={len(t)} | Te={'yes' if Te is not None else 'no'} | "
                    f"Ts={'yes' if Ts is not None else 'no'} | "
                    f"Tl={'yes' if Tl is not None else 'no'} | "
                    f"S={'yes' if S is not None else 'no'}"
                )
                if baseline_method is not None:
                    self._log(
                        f"[load] baseline={baseline_value:.4e} | baseline_npts={baseline_npts} | "
                        f"baseline_method={baseline_method}"
                    )
            except Exception as e:
                messagebox.showerror("Load CSV failed", str(e))
                self._log(f"[error] {e}")

        def on_load_csvs(self):
            paths = filedialog.askopenfilenames(
                title="Choose CSV files",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not paths:
                return

            loaded = []
            try:
                for path in paths:
                    dataset = load_s_dataset_csv(path)
                    loaded.append(dataset)
                    self._log(
                        f"[multi-load] name={dataset['name']} | N={len(dataset['t'])} | "
                        f"fluence_ratio={dataset['fluence_ratio']:.3f} | "
                        f"baseline={dataset['baseline_value']:.4e} | "
                        f"baseline_npts={dataset['baseline_npts']} | "
                        f"baseline_method={dataset['baseline_method']}"
                    )
            except Exception as e:
                messagebox.showerror("Load CSVs failed", str(e))
                self._log(f"[error] {e}")
                return

            existing_by_path = {item.get("path"): item for item in self.datasets if item.get("path")}
            unnamed_existing = [item for item in self.datasets if not item.get("path")]
            replaced = 0
            for dataset in loaded:
                dataset_path = dataset.get("path")
                if dataset_path in existing_by_path:
                    replaced += 1
                existing_by_path[dataset_path] = dataset

            merged = unnamed_existing + list(existing_by_path.values())
            self.datasets = merged
            self._log(
                f"[multi-load] added={len(loaded)} | replaced={replaced} | total datasets={len(self.datasets)}"
            )

        def on_load_dk_csvs(self):
            paths = filedialog.askopenfilenames(
                title="Choose delta-k CSV files",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not paths:
                return

            loaded = []
            try:
                for path in paths:
                    dataset = load_dk_dataset_csv(path)
                    loaded.append(dataset)
                    n_resolved = int(np.count_nonzero(np.asarray(dataset["is_resolved"], dtype=bool)))
                    self._log(
                        f"[multi-load-dk] name={dataset['name']} | N={len(dataset['t'])} | "
                        f"fluence_ratio={dataset['fluence_ratio']:.3f} | "
                        f"resolved={n_resolved} | unresolved={len(dataset['t']) - n_resolved}"
                    )
            except Exception as e:
                messagebox.showerror("Load delta-k CSVs failed", str(e))
                self._log(f"[error] {e}")
                return

            existing_by_path = {item.get("path"): item for item in self.datasets if item.get("path")}
            unnamed_existing = [item for item in self.datasets if not item.get("path")]
            replaced = 0
            for dataset in loaded:
                dataset_path = dataset.get("path")
                if dataset_path in existing_by_path:
                    replaced += 1
                existing_by_path[dataset_path] = dataset

            self.datasets = unnamed_existing + list(existing_by_path.values())
            self._log(
                f"[multi-load-dk] added={len(loaded)} | replaced={replaced} | total datasets={len(self.datasets)}"
            )

        def on_clear_datasets(self):
            self.datasets = []
            self.multi_fit_params = None
            self.multi_fit_res = None
            self.multi_fit_exports = None
            self._log("[multi-load] cleared all datasets and multi-fit results.")

        def on_fit_multi_s(self):
            if not self.datasets:
                messagebox.showwarning("No datasets", "Please load multiple CSV files first.")
                return

            self._read_entries_to_params()
            p_start = normalize_params_dict(self._current_params_dict())
            preset = dict(MULTI_FIT_PRESET)

            try:
                self._log(
                    f"[multi-fit] observable={preset['observable_mode']} | global_keys={preset['global_keys']} | "
                    f"local_keys={preset['local_keys']}"
                )
                fit_bundle, res = fit_params_multi(
                    self.datasets,
                    p_start,
                    global_keys=preset["global_keys"],
                    local_keys=preset["local_keys"],
                    observable_mode=preset["observable_mode"],
                    sigma_S=0.02,
                    progress_every=5,
                    progress_callback=self._log,
                    optimizer_verbose=0,
                    enable_timing=True,
                )
                self.multi_fit_params = fit_bundle
                self.multi_fit_res = res
                self.p_fit = dict(fit_bundle["best_global_params"])
                self.fit_res = res
                self._refresh_entries_from_params(self.p_fit, view="p_fit")
                self._plot_multi_fit_preview(fit_bundle)

                self._log(
                    f"[multi-fit] success={res.success} | cost={res.cost:.3e} | nfev={res.nfev} | status={res.status}"
                )
                for key in fit_bundle["global_keys"]:
                    self._log(f"    [global] {key} = {fmt_num(fit_bundle['best_global_params'].get(key, np.nan))}")
                for row in fit_bundle["dataset_summary"]:
                    dt_i_ps = float(row.get("dt_i_ps", np.nan))
                    a_obs = float(row.get("A_obs", np.nan))
                    b_obs = float(row.get("B_obs", np.nan))
                    b_eff_obs = float(row.get("B_eff_obs", np.nan))
                    sigma_irf_ps = float(row.get("sigma_irf_ps", np.nan))
                    self._log(
                        f"    [local] {row['dataset_name']} | rms={row['rms']:.4g} | wrms={row['wrms']:.4g} | "
                        f"dt_i_ps={dt_i_ps:.4g} | A_obs={a_obs:.4g} | "
                        f"B_obs={b_obs:.4g} | B_eff_obs={b_eff_obs:.4g} | "
                        f"sigma_irf_ps={sigma_irf_ps:.4g}"
                    )
            except Exception as e:
                messagebox.showerror("Multi-fit error", str(e))
                self._log(f"[error] {e}")

        def on_fit_multi_dk(self):
            if not self.datasets:
                messagebox.showwarning("No datasets", "Please load delta-k CSV files first.")
                return

            self._read_entries_to_params()
            p_start = normalize_params_dict(self._current_params_dict())
            preset = dict(MULTI_FIT_DK_PRESET)

            try:
                self._log(
                    f"[multi-fit-dk] target_kind=delta_k | observable={preset['observable_mode']} | "
                    f"global_keys={preset['global_keys']} | local_keys={preset['local_keys']}"
                )
                fit_bundle, res = fit_params_multi_dk(
                    self.datasets,
                    p_start,
                    global_keys=preset["global_keys"],
                    local_keys=preset["local_keys"],
                    observable_mode=preset["observable_mode"],
                    sigma_dk=float(p_start.get("sigma_dk", 0.002)),
                    sigma_dk_censored=float(p_start.get("sigma_dk_censored", 0.002)),
                    dk_resolution_limit=float(p_start.get("dk_resolution_limit", 0.003)),
                    progress_every=5,
                    progress_callback=self._log,
                    optimizer_verbose=0,
                    enable_timing=True,
                )
                self.multi_fit_params = fit_bundle
                self.multi_fit_res = res
                self.p_fit = dict(fit_bundle["best_global_params"])
                self.fit_res = res
                self._refresh_entries_from_params(self.p_fit, view="p_fit")
                self._plot_multi_fit_preview(fit_bundle)

                self._log(
                    f"[multi-fit-dk] success={res.success} | cost={res.cost:.3e} | nfev={res.nfev} | status={res.status}"
                )
                self._log(f"    [global] K_dk = {fmt_num(fit_bundle['best_global_params'].get('K_dk', np.nan))}")
                if "B_dk" in fit_bundle["best_global_params"]:
                    self._log(f"    [global] B_dk = {fmt_num(fit_bundle['best_global_params'].get('B_dk', np.nan))}")
                for row in fit_bundle["dataset_summary"]:
                    self._log(
                        f"    [dk] {row['dataset_name']} | rms={row['rms']:.4g} | wrms={row['wrms']:.4g} | "
                        f"resolved={int(row.get('n_resolved', 0))} | unresolved={int(row.get('n_unresolved', 0))}"
                    )
            except Exception as e:
                messagebox.showerror("Multi-fit delta-k error", str(e))
                self._log(f"[error] {e}")

        def on_export_multi_fit(self):
            if self.multi_fit_params is None or self.multi_fit_res is None:
                messagebox.showwarning("No multi-fit", "Please run Fit Multi-S first.")
                return
            try:
                exports = export_multi_fit_results(self.multi_fit_params, self.multi_fit_res)
                self.multi_fit_exports = exports
                self._log(f"[multi-export] export_dir={exports['export_dir']}")
                self._log(f"[multi-export] json={exports['json']}")
                self._log(f"[multi-export] summary_csv={exports['summary_csv']}")
                self._log(f"[multi-export] overlay_png={exports['overlay_png']}")
                self._log(f"[multi-export] params_png={exports['params_png']}")
            except Exception as e:
                messagebox.showerror("Export failed", str(e))
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

            fit_keys = normalize_fit_keys(FIT_PRESETS[mode])

            if mode == "Te":
                Te_fit, S_fit = Te, None

            elif mode == "S":
                Te_fit, S_fit = None, S

            else:
                Te_fit, S_fit = Te, S

            p_start = normalize_params_dict(self._current_params_dict())

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
                sim = model.simulate_aligned(t_sim, with_diag=True)
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
            self._log("[fit-S] using legacy single-dataset S_m proxy, not multi-fit observable_mode.")
            self._fit("S")

        def on_fit_both(self):
            self._log("[fit-Both] using legacy single-dataset S_m proxy for S.")
            self._fit("Both")


def _cli_demo():
    """Headless quick demo: run a short simulation and print maxima."""
    p = default_params()
    model = NdSb3TM(p)
    t = np.linspace(-2e-12, 100e-12, 1200)
    sim = model.simulate_aligned(t, with_diag=False)
    print("GUI not available (tkinter missing). Headless demo:")
    print("  T_bath =", float(sim["T_bath"]))
    print("  T_init_eff_used =", float(sim["T_init_eff_used"]))
    if "P_avg_preheat" in sim:
        print("  P_avg_preheat =", float(sim["P_avg_preheat"]))
    print("  max Te =", float(np.max(sim["Te"])))
    print("  max Ts =", float(np.max(sim["Ts"])))
    print("  max Tl =", float(np.max(sim["Tl"])))
    print("  final m =", float(sim["m"][-1]))
    print("  final eta =", float(sim["eta"][-1]))
    if sim.get("eta_representation", "scalar") == "cos2phi":
        print("  final phi =", float(sim["phi"][-1]))


if __name__ == "__main__":
    if GUI_AVAILABLE:
        app = NdSb3TMApp()
        app.mainloop()
    else:
        _cli_demo()
