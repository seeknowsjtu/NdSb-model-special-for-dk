# -*- coding: utf-8 -*-
"""
plot_order_params_from_mainfit.py
---------------------------------
Read a main-fit JSON, reuse best_global_params, and plot
model-predicted order parameters vs time for different fluences.

Outputs:
1) combined_order_panels_*.png
2) m_only_*.png
3) chi2q_only_*.png
4) m2_chi2q2_only_*.png
5) composite_orders_*.png
6) m_vs_meq_only_*.png
7) Ts_vs_TN_*.png
8) m_meq_tau_diagnostic_*.png
9) order_summary_*.csv
10) order_diagnostic_summary_*.csv
11) order_meta_*.json

Order channels:
    m
    chi2q
    eta
    phi
    m_chi2q
    m2_chi2q2
    m2_chi2q
    m2

Diagnostic channels:
    Te
    Ts
    Tl
    meq
    tau_m
    dm_dt
    Ts_minus_TN
    above_TN

Usage:
    python plot_order_params_from_mainfit.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import default_params, normalize_params_dict
from physics_engine import DebyeCl
from solver import NdSb3TM


# ============================================================
# User settings
# ============================================================
DATA_DIR = Path(".")
OUTPUT_DIR = Path("fit_results/model_order_params")
OUTPUT_TAG = "from_mainfit"

# Main-fit JSON
# Use raw string to avoid Windows path escape problems.
MAINFIT_JSON = Path(
    r"F:\python4git\simulate_dk\fit_results\real_multi_fit_round1\deltak12k_globalfit_20260328_083202\globalfit_deltak12k_raw_m_chi2q_1p0to2p5mW_20260328_083202.json"
)

# Fluences to simulate
FLUENCE_LIST = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0]

# Time axis for plotting, in ps
TMIN_PS = -2.0
TMAX_PS = 105.0
DT_PS = 0.25

# If True:
#   plot on experimental delay axis, internally simulate at
#   t_model = t_plot - dt_i_ps
APPLY_DT_ALIGNMENT = True

# Used in dt_i_ps = dt0_ps + alpha_dt_per_F * (F - F_REF)
F_REF = 1.0

# Plot cosmetics
FIG_DPI = 180
LINE_WIDTH = 2.0


# ============================================================
# Small helpers
# ============================================================
def find_latest_json(pattern: str) -> Path | None:
    cands = sorted(Path(".").glob(pattern))
    return cands[-1] if cands else None


def resolve_mainfit_json() -> Path:
    if MAINFIT_JSON.exists():
        return MAINFIT_JSON

    latest = find_latest_json("fit_results/**/globalfit_*.json")
    if latest is not None:
        return latest

    raise FileNotFoundError(
        "Cannot find the main-fit JSON. "
        "Please set MAINFIT_JSON explicitly."
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_best_global_params(payload: dict) -> dict:
    if "best_global_params" not in payload:
        raise KeyError("JSON missing 'best_global_params'.")
    p = normalize_params_dict(default_params())
    p.update(payload["best_global_params"])
    return p


def get_observable_mode(payload: dict) -> str:
    return str(payload.get("observable_mode", "unknown"))


def get_target_kind(payload: dict) -> str:
    return str(payload.get("target_kind", "unknown"))


def build_time_axis_ps() -> np.ndarray:
    n = int(np.floor((TMAX_PS - TMIN_PS) / DT_PS)) + 1
    return TMIN_PS + np.arange(n, dtype=float) * DT_PS


def compute_dt_i_ps(p_global: dict, fluence_ratio: float) -> float:
    dt0_ps = float(p_global.get("dt0_ps", 0.0))
    alpha = float(p_global.get("alpha_dt_per_F", 0.0))
    return dt0_ps + alpha * (float(fluence_ratio) - float(F_REF))


def color_for_index(i: int):
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
    return cycle[i % len(cycle)]


def compute_chi2q_from_sim(sim: dict) -> np.ndarray:
    """
    Same logic as data_io._compute_chi2q.

    If eta_representation == cos2phi:
        chi2q = |sin(2 phi)|
    Otherwise:
        chi2q = sqrt(1 - eta^2)
    """
    eta_repr = str(sim.get("eta_representation", "")).strip().lower()
    phi = sim.get("phi", None)

    if eta_repr == "cos2phi" and phi is not None:
        phi = np.asarray(phi, dtype=float)
        return np.abs(np.sin(2.0 * phi))

    eta = np.asarray(sim["eta"], dtype=float)
    eta_clip = np.clip(eta, -1.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - eta_clip ** 2))


def compute_m_diagnostics(model: NdSb3TM, Ts: np.ndarray, m: np.ndarray) -> dict:
    """
    Compute equilibrium order parameter and m dynamics diagnostics.

    meq(Ts) comes from model.m_eq(Ts).
    tau_m(Ts) comes from model.tau_m(Ts).
    dm_dt = -(m - meq) / tau_m, in 1/s.
    dm_dt_per_ps = dm_dt * 1e-12.
    """
    Ts = np.asarray(Ts, dtype=float)
    m = np.asarray(m, dtype=float)

    meq = np.empty_like(m, dtype=float)
    tau_m = np.empty_like(m, dtype=float)

    for i in range(m.size):
        meq[i] = float(model.m_eq(float(Ts[i])))
        tau_m[i] = float(model.tau_m(float(Ts[i])))

    tau_m = np.maximum(tau_m, 1e-30)
    dm_dt = -(m - meq) / tau_m
    dm_dt_per_ps = dm_dt * 1e-12

    return {
        "meq": meq,
        "tau_m": tau_m,
        "tau_m_ps": tau_m * 1e12,
        "dm_dt": dm_dt,
        "dm_dt_per_ps": dm_dt_per_ps,
    }


def simulate_one_fluence(
    p_global: dict,
    fluence_ratio: float,
    t_plot_ps: np.ndarray,
) -> dict:
    """
    Simulate one fluence using the best-fit global params.
    """
    p_run = dict(p_global)
    p_run["fluence_multiplier"] = float(fluence_ratio)

    dt_i_ps = compute_dt_i_ps(p_run, fluence_ratio)

    if APPLY_DT_ALIGNMENT:
        t_model_sec = (t_plot_ps - dt_i_ps) * 1e-12
    else:
        t_model_sec = t_plot_ps * 1e-12

    debye_obj = DebyeCl(thetaD=float(p_run["ThetaD"]))
    model = NdSb3TM(p_run, debye_obj=debye_obj)

    sim = model.simulate_aligned(
        t_model_sec,
        with_diag=True,
    )

    required = ["Te", "Ts", "Tl", "m", "eta", "phi"]
    missing = [k for k in required if k not in sim]
    if missing:
        raise KeyError(
            f"simulate_aligned result missing keys {missing}. "
            f"Available keys include: {list(sim.keys())[:20]} ..."
        )

    Te = np.asarray(sim["Te"], dtype=float)
    Ts = np.asarray(sim["Ts"], dtype=float)
    Tl = np.asarray(sim["Tl"], dtype=float)

    m = np.asarray(sim["m"], dtype=float)
    eta = np.asarray(sim["eta"], dtype=float)
    phi = np.asarray(sim["phi"], dtype=float)
    chi2q = compute_chi2q_from_sim(sim)

    m_diag = compute_m_diagnostics(model=model, Ts=Ts, m=m)

    TN = float(p_run.get("TN", np.nan))
    TR = float(p_run.get("TR", np.nan))

    row = {
        "fluence_ratio": float(fluence_ratio),
        "dt_i_ps": float(dt_i_ps),
        "TN": TN,
        "TR": TR,
        "t_plot_ps": np.asarray(t_plot_ps, dtype=float),
        "t_model_sec": np.asarray(t_model_sec, dtype=float),

        # temperatures
        "Te": Te,
        "Ts": Ts,
        "Tl": Tl,

        # order variables
        "m": m,
        "eta": eta,
        "phi": phi,
        "chi2q": chi2q,

        # composite order variables
        "m_chi2q": m * chi2q,
        "m2_chi2q2": (m * chi2q) ** 2,
        "m2_chi2q": (m ** 2) * chi2q,
        "m2": m ** 2,

        # m dynamics diagnostics
        "meq": m_diag["meq"],
        "tau_m": m_diag["tau_m"],
        "tau_m_ps": m_diag["tau_m_ps"],
        "dm_dt": m_diag["dm_dt"],
        "dm_dt_per_ps": m_diag["dm_dt_per_ps"],
        "Ts_minus_TN": Ts - TN,
        "above_TN": (Ts > TN).astype(float),

        "sim": sim,
    }

    return row


def get_channel_label(key: str) -> str:
    labels = {
        "Te": "Te (K)",
        "Ts": "Ts (K)",
        "Tl": "Tl (K)",
        "m": "m",
        "meq": "m_eq[Ts]",
        "eta": "eta",
        "phi": "phi",
        "chi2q": "chi2q",
        "m_chi2q": "m * chi2q",
        "m2_chi2q2": "(m * chi2q)^2",
        "m2_chi2q": "m^2 * chi2q",
        "m2": "m^2",
        "tau_m": "tau_m (s)",
        "tau_m_ps": "tau_m (ps)",
        "dm_dt": "dm/dt (1/s)",
        "dm_dt_per_ps": "dm/dt (per ps)",
        "Ts_minus_TN": "Ts - TN (K)",
        "above_TN": "Ts > TN",
    }
    return labels.get(key, key)


# ============================================================
# Plotting
# ============================================================
def plot_combined_order_panels(rows: list[dict], outpath: Path, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), sharex=True)

    for i, row in enumerate(rows):
        c = color_for_index(i)
        label = f"{row['fluence_ratio']:.1f} mW"

        axes[0].plot(row["t_plot_ps"], row["m"], lw=LINE_WIDTH, color=c, label=label)
        axes[1].plot(row["t_plot_ps"], row["chi2q"], lw=LINE_WIDTH, color=c, label=label)
        axes[2].plot(row["t_plot_ps"], row["m2_chi2q2"], lw=LINE_WIDTH, color=c, label=label)

    axes[0].set_ylabel("m")
    axes[1].set_ylabel("chi2q")
    axes[2].set_ylabel("(m * chi2q)^2")
    axes[2].set_xlabel("time (ps)")

    axes[0].set_title(f"Model order parameters from main-fit JSON{title_suffix}")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    axes[0].legend(ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_single_channel(rows: list[dict], key: str, outpath: Path, title_suffix: str = "") -> None:
    plt.figure(figsize=(8.5, 5.8))

    for i, row in enumerate(rows):
        c = color_for_index(i)
        plt.plot(
            row["t_plot_ps"],
            row[key],
            lw=LINE_WIDTH,
            color=c,
            label=f"{row['fluence_ratio']:.1f} mW",
        )

    plt.xlabel("time (ps)")
    plt.ylabel(get_channel_label(key))
    plt.title(f"{get_channel_label(key)} vs time{title_suffix}")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()


def plot_composite_orders(rows: list[dict], outpath: Path, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 12.0), sharex=True)

    keys = ["m_chi2q", "m2_chi2q2", "m2_chi2q", "m2"]

    for ax, key in zip(axes, keys):
        for i, row in enumerate(rows):
            c = color_for_index(i)
            label = f"{row['fluence_ratio']:.1f} mW"
            ax.plot(row["t_plot_ps"], row[key], lw=LINE_WIDTH, color=c, label=label)

        ax.set_ylabel(get_channel_label(key))
        ax.grid(True, alpha=0.25)

    axes[0].set_title(f"Composite order-parameter observables{title_suffix}")
    axes[-1].set_xlabel("time (ps)")
    axes[0].legend(ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_m_vs_meq_only(rows: list[dict], outpath: Path, title_suffix: str = "") -> None:
    plt.figure(figsize=(8.8, 6.0))

    for i, row in enumerate(rows):
        c = color_for_index(i)
        label = f"{row['fluence_ratio']:.1f} mW"

        plt.plot(
            row["t_plot_ps"],
            row["m"],
            lw=LINE_WIDTH,
            color=c,
            label=f"m {label}",
        )
        plt.plot(
            row["t_plot_ps"],
            row["meq"],
            lw=1.5,
            ls="--",
            color=c,
            label=f"m_eq {label}",
        )

    plt.xlabel("time (ps)")
    plt.ylabel("m, m_eq")
    plt.title(f"m vs m_eq[Ts]{title_suffix}")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()


def plot_Ts_vs_TN(rows: list[dict], outpath: Path, title_suffix: str = "") -> None:
    plt.figure(figsize=(8.8, 6.0))

    TN_values = []
    TR_values = []

    for i, row in enumerate(rows):
        c = color_for_index(i)
        label = f"{row['fluence_ratio']:.1f} mW"

        plt.plot(
            row["t_plot_ps"],
            row["Ts"],
            lw=LINE_WIDTH,
            color=c,
            label=label,
        )
        TN_values.append(float(row["TN"]))
        TR_values.append(float(row["TR"]))

    TN = float(np.nanmean(TN_values)) if TN_values else np.nan
    TR = float(np.nanmean(TR_values)) if TR_values else np.nan

    if np.isfinite(TN):
        plt.axhline(TN, color="k", lw=1.2, ls="--", label=f"TN = {TN:g} K")
    if np.isfinite(TR):
        plt.axhline(TR, color="gray", lw=1.2, ls=":", label=f"TR = {TR:g} K")

    plt.xlabel("time (ps)")
    plt.ylabel("Ts (K)")
    plt.title(f"Spin temperature vs TN/TR{title_suffix}")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()


def plot_m_meq_tau_diagnostic(rows: list[dict], outpath: Path, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 12.0), sharex=True)

    TN_values = []
    TR_values = []

    for i, row in enumerate(rows):
        c = color_for_index(i)
        label = f"{row['fluence_ratio']:.1f} mW"

        axes[0].plot(row["t_plot_ps"], row["m"], lw=LINE_WIDTH, color=c, label=f"m {label}")
        axes[0].plot(row["t_plot_ps"], row["meq"], lw=1.3, ls="--", color=c, label=f"m_eq {label}")

        axes[1].plot(row["t_plot_ps"], row["Ts"], lw=LINE_WIDTH, color=c, label=label)

        axes[2].plot(row["t_plot_ps"], row["tau_m_ps"], lw=LINE_WIDTH, color=c, label=label)

        axes[3].plot(row["t_plot_ps"], row["dm_dt_per_ps"], lw=LINE_WIDTH, color=c, label=label)

        TN_values.append(float(row["TN"]))
        TR_values.append(float(row["TR"]))

    TN = float(np.nanmean(TN_values)) if TN_values else np.nan
    TR = float(np.nanmean(TR_values)) if TR_values else np.nan

    if np.isfinite(TN):
        axes[1].axhline(TN, color="k", lw=1.2, ls="--", label=f"TN = {TN:g} K")
    if np.isfinite(TR):
        axes[1].axhline(TR, color="gray", lw=1.2, ls=":", label=f"TR = {TR:g} K")

    axes[0].set_ylabel("m, m_eq")
    axes[1].set_ylabel("Ts (K)")
    axes[2].set_ylabel("tau_m (ps)")
    axes[3].set_ylabel("dm/dt (per ps)")
    axes[3].set_xlabel("time (ps)")

    axes[0].set_title(f"m dynamics diagnostic{title_suffix}")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


# ============================================================
# CSV / JSON exports
# ============================================================
def write_summary_csv(rows: list[dict], outpath: Path) -> None:
    """
    Wide-format CSV:
        t_ps,
        m_1p0mW, chi2q_1p0mW, m_chi2q_1p0mW, ...
    """
    if not rows:
        return

    t_ps = rows[0]["t_plot_ps"]
    for row in rows[1:]:
        if row["t_plot_ps"].shape != t_ps.shape or not np.allclose(row["t_plot_ps"], t_ps):
            raise ValueError("All rows must share the same t_plot_ps grid for CSV export.")

    channels = [
        "m",
        "chi2q",
        "eta",
        "phi",
        "m_chi2q",
        "m2_chi2q2",
        "m2_chi2q",
        "m2",
    ]

    header = ["t_ps"]
    for row in rows:
        token = f"{row['fluence_ratio']:.1f}".replace(".", "p") + "mW"
        for ch in channels:
            header.append(f"{ch}_{token}")

    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(len(t_ps)):
            line = [float(t_ps[i])]
            for row in rows:
                for ch in channels:
                    line.append(float(row[ch][i]))
            writer.writerow(line)


def write_diagnostic_summary_csv(rows: list[dict], outpath: Path) -> None:
    """
    Wide-format diagnostic CSV:
        t_ps,
        Te_1p0mW, Ts_1p0mW, Tl_1p0mW,
        m_1p0mW, meq_1p0mW, tau_m_ps_1p0mW,
        dm_dt_per_ps_1p0mW, Ts_minus_TN_1p0mW, above_TN_1p0mW, ...
    """
    if not rows:
        return

    t_ps = rows[0]["t_plot_ps"]
    for row in rows[1:]:
        if row["t_plot_ps"].shape != t_ps.shape or not np.allclose(row["t_plot_ps"], t_ps):
            raise ValueError("All rows must share the same t_plot_ps grid for CSV export.")

    channels = [
        "Te",
        "Ts",
        "Tl",
        "m",
        "meq",
        "tau_m_ps",
        "dm_dt_per_ps",
        "Ts_minus_TN",
        "above_TN",
    ]

    header = ["t_ps"]
    for row in rows:
        token = f"{row['fluence_ratio']:.1f}".replace(".", "p") + "mW"
        for ch in channels:
            header.append(f"{ch}_{token}")

    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(len(t_ps)):
            line = [float(t_ps[i])]
            for row in rows:
                for ch in channels:
                    line.append(float(row[ch][i]))
            writer.writerow(line)


def write_meta_json(
    rows: list[dict],
    p_global: dict,
    source_json: Path,
    target_kind: str,
    observable_mode: str,
    outpath: Path,
) -> None:
    payload = {
        "source_mainfit_json": str(source_json),
        "target_kind": target_kind,
        "observable_mode": observable_mode,
        "fluence_rows": [
            {
                "fluence_ratio": float(row["fluence_ratio"]),
                "dt_i_ps": float(row["dt_i_ps"]),
                "TN": float(row["TN"]),
                "TR": float(row["TR"]),
            }
            for row in rows
        ],
        "order_channels": [
            "m",
            "chi2q",
            "eta",
            "phi",
            "m_chi2q",
            "m2_chi2q2",
            "m2_chi2q",
            "m2",
        ],
        "diagnostic_channels": [
            "Te",
            "Ts",
            "Tl",
            "meq",
            "tau_m",
            "tau_m_ps",
            "dm_dt",
            "dm_dt_per_ps",
            "Ts_minus_TN",
            "above_TN",
        ],
        "best_global_params": {
            k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in p_global.items()
        },
    }

    with outpath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================
# Main
# ============================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = resolve_mainfit_json()
    payload = load_json(json_path)
    p_global = get_best_global_params(payload)

    target_kind = get_target_kind(payload)
    observable_mode = get_observable_mode(payload)
    title_suffix = f" ({target_kind}, {observable_mode})"

    t_plot_ps = build_time_axis_ps()

    rows = []
    for flu in FLUENCE_LIST:
        row = simulate_one_fluence(
            p_global=p_global,
            fluence_ratio=flu,
            t_plot_ps=t_plot_ps,
        )
        rows.append(row)

    # Original order-parameter plots
    plot_combined_order_panels(
        rows,
        outpath=OUTPUT_DIR / f"combined_order_panels_{OUTPUT_TAG}.png",
        title_suffix=title_suffix,
    )

    plot_single_channel(
        rows,
        key="m",
        outpath=OUTPUT_DIR / f"m_only_{OUTPUT_TAG}.png",
        title_suffix=title_suffix,
    )

    plot_single_channel(
        rows,
        key="chi2q",
        outpath=OUTPUT_DIR / f"chi2q_only_{OUTPUT_TAG}.png",
        title_suffix=title_suffix,
    )

    plot_single_channel(
        rows,
        key="m2_chi2q2",
        outpath=OUTPUT_DIR / f"m2_chi2q2_only_{OUTPUT_TAG}.png",
        title_suffix=title_suffix,
    )

    plot_composite_orders(
        rows,
        outpath=OUTPUT_DIR / f"composite_orders_{OUTPUT_TAG}.png",
        title_suffix=title_suffix,
    )

    # New diagnostics
    plot_m_vs_meq_only(
        rows,
        outpath=OUTPUT_DIR / f"m_vs_meq_only_{OUTPUT_TAG}.png",
        title_suffix=title_suffix,
    )

    plot_Ts_vs_TN(
        rows,
        outpath=OUTPUT_DIR / f"Ts_vs_TN_{OUTPUT_TAG}.png",
        title_suffix=title_suffix,
    )

    plot_m_meq_tau_diagnostic(
        rows,
        outpath=OUTPUT_DIR / f"m_meq_tau_diagnostic_{OUTPUT_TAG}.png",
        title_suffix=title_suffix,
    )

    # Data exports
    write_summary_csv(
        rows,
        outpath=OUTPUT_DIR / f"order_summary_{OUTPUT_TAG}.csv",
    )

    write_diagnostic_summary_csv(
        rows,
        outpath=OUTPUT_DIR / f"order_diagnostic_summary_{OUTPUT_TAG}.csv",
    )

    write_meta_json(
        rows,
        p_global=p_global,
        source_json=json_path,
        target_kind=target_kind,
        observable_mode=observable_mode,
        outpath=OUTPUT_DIR / f"order_meta_{OUTPUT_TAG}.json",
    )

    print(f"[order-plot] source mainfit json = {json_path}")
    print(f"[order-plot] target_kind = {target_kind}")
    print(f"[order-plot] observable_mode = {observable_mode}")
    print(f"[order-plot] output dir = {OUTPUT_DIR}")
    print("[order-plot] files:")
    print(f"  - {OUTPUT_DIR / f'combined_order_panels_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'm_only_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'chi2q_only_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'm2_chi2q2_only_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'composite_orders_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'm_vs_meq_only_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'Ts_vs_TN_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'm_meq_tau_diagnostic_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'order_summary_{OUTPUT_TAG}.csv'}")
    print(f"  - {OUTPUT_DIR / f'order_diagnostic_summary_{OUTPUT_TAG}.csv'}")
    print(f"  - {OUTPUT_DIR / f'order_meta_{OUTPUT_TAG}.json'}")


if __name__ == "__main__":
    main()