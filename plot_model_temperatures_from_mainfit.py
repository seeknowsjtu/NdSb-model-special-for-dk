# -*- coding: utf-8 -*-
"""
plot_model_temperatures_from_mainfit.py
---------------------------------------
Read the 3-group main-fit JSON, reuse best_global_params, and plot
model-predicted temperatures vs time for different fluences.

Outputs:
1) combined_temperature_panels_*.png
2) Te_only_*.png
3) Tl_only_*.png
4) custom_only_*.png
5) temperature_summary_*.csv

Default custom channel:
    Ts   (spin temperature)

You can change CUSTOM_CHANNEL below to, e.g.:
    "Ts", "Te", "Tl", "m", "eta", "phi", "meq", "tau_m", "Ce", "Cs", "Cl"

Usage:
    python plot_model_temperatures_from_mainfit.py
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
OUTPUT_DIR = Path("fit_results/model_temperatures")
OUTPUT_TAG = "from_mainfit3"

# Main-fit JSON from the 3-group fit
MAINFIT_JSON = Path("F:\python4git\simulate\fit_results\real_multi_fit_smoke\deltak12k_globalfit_20260327_100946\globalfit_deltak12k_raw_m_chi2q_1p0to2p5mW_20260327_100946.json")
# 改成你的实际路径即可；如果不存在，会自动找最近的 1.0~2.5mW json

# Fluences to simulate
FLUENCE_LIST = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0]

# Time axis for plotting (experimental delay axis, in ps)
TMIN_PS = -2.0
TMAX_PS = 105.0
DT_PS = 0.25

# If True:
#   plot on the experimental delay axis, and internally simulate at
#   t_model = t_plot - dt_i_ps
# If False:
#   plot intrinsic model time axis directly without dt0 alignment
APPLY_DT_ALIGNMENT = True

# Custom channel to plot in the 3rd panel
CUSTOM_CHANNEL = "Ts"

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

    latest = find_latest_json("fit_results/**/globalfit_*raw_m_chi2q*1p0to2p5mW*.json")
    if latest is not None:
        return latest

    raise FileNotFoundError(
        "Cannot find the 3-group main-fit JSON. "
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
    return str(payload.get("observable_mode", "raw_m_chi2q"))


def build_time_axis_ps() -> np.ndarray:
    n = int(np.floor((TMAX_PS - TMIN_PS) / DT_PS)) + 1
    return TMIN_PS + np.arange(n, dtype=float) * DT_PS


def compute_dt_i_ps(p_global: dict, fluence_ratio: float) -> float:
    dt0_ps = float(p_global.get("dt0_ps", 0.0))
    alpha = float(p_global.get("alpha_dt_per_F", 0.0))
    return dt0_ps + alpha * (float(fluence_ratio) - float(F_REF))


def get_channel_label(channel: str) -> str:
    temp_like = {"Te", "Ts", "Tl"}
    if channel in temp_like:
        return f"{channel} (K)"
    if channel in {"Ce", "Cs", "Cl"}:
        return channel
    if channel in {"m", "eta", "phi", "meq", "tau_m"}:
        return channel
    return channel


def color_for_index(i: int):
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
    return cycle[i % len(cycle)]


def simulate_one_fluence(
    p_global: dict,
    fluence_ratio: float,
    t_plot_ps: np.ndarray,
    custom_channel: str = "Ts",
) -> dict:
    """
    Simulate one fluence using the best-fit global params.

    We plot on the experimental delay axis t_plot_ps.
    If APPLY_DT_ALIGNMENT=True, we internally simulate at
        t_model = t_plot - dt_i_ps
    so the resulting curves are aligned to the same delay convention
    used by the main-fit template builder.
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

    needed_keys = ["Te", "Tl", custom_channel]
    missing = [k for k in needed_keys if k not in sim]
    if missing:
        raise KeyError(
            f"simulate_aligned result missing keys {missing}. "
            f"Available keys include: {list(sim.keys())[:20]} ..."
        )

    row = {
        "fluence_ratio": float(fluence_ratio),
        "dt_i_ps": float(dt_i_ps),
        "t_plot_ps": np.asarray(t_plot_ps, dtype=float),
        "Te": np.asarray(sim["Te"], dtype=float),
        "Tl": np.asarray(sim["Tl"], dtype=float),
        custom_channel: np.asarray(sim[custom_channel], dtype=float),
        "custom_channel": custom_channel,
        "sim": sim,
    }

    # If Ts exists and custom channel isn't Ts, still keep it for convenience
    if "Ts" in sim:
        row["Ts"] = np.asarray(sim["Ts"], dtype=float)

    return row


# ============================================================
# Plotting
# ============================================================
def plot_combined_panels(rows: list[dict], custom_channel: str, outpath: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), sharex=True)

    for i, row in enumerate(rows):
        c = color_for_index(i)
        label = f"{row['fluence_ratio']:.1f} mW"

        axes[0].plot(row["t_plot_ps"], row["Te"], lw=LINE_WIDTH, color=c, label=label)
        axes[1].plot(row["t_plot_ps"], row["Tl"], lw=LINE_WIDTH, color=c, label=label)
        axes[2].plot(row["t_plot_ps"], row[custom_channel], lw=LINE_WIDTH, color=c, label=label)

    axes[0].set_ylabel("Te (K)")
    axes[1].set_ylabel("Tl (K)")
    axes[2].set_ylabel(get_channel_label(custom_channel))
    axes[2].set_xlabel("time (ps)")

    axes[0].set_title("Model temperatures from 3-group global fit")
    for ax in axes:
        ax.grid(True, alpha=0.25)

    axes[0].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_single_channel(rows: list[dict], key: str, ylabel: str, title: str, outpath: Path) -> None:
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
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()


def write_summary_csv(rows: list[dict], custom_channel: str, outpath: Path) -> None:
    """
    Wide-format CSV:
        t_ps,
        Te_1p0mW, Tl_1p0mW, Ts_1p0mW, ...
    """
    if not rows:
        return

    t_ps = rows[0]["t_plot_ps"]
    for row in rows[1:]:
        if row["t_plot_ps"].shape != t_ps.shape or not np.allclose(row["t_plot_ps"], t_ps):
            raise ValueError("All rows must share the same t_plot_ps grid for CSV export.")

    header = ["t_ps"]
    for row in rows:
        token = f"{row['fluence_ratio']:.1f}".replace(".", "p") + "mW"
        header += [
            f"Te_{token}",
            f"Tl_{token}",
            f"{custom_channel}_{token}",
        ]

    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(t_ps)):
            line = [float(t_ps[i])]
            for row in rows:
                line += [
                    float(row["Te"][i]),
                    float(row["Tl"][i]),
                    float(row[custom_channel][i]),
                ]
            writer.writerow(line)


def write_meta_json(rows: list[dict], p_global: dict, source_json: Path, custom_channel: str, outpath: Path) -> None:
    payload = {
        "source_mainfit_json": str(source_json),
        "custom_channel": custom_channel,
        "fluence_rows": [
            {
                "fluence_ratio": float(row["fluence_ratio"]),
                "dt_i_ps": float(row["dt_i_ps"]),
            }
            for row in rows
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

    t_plot_ps = build_time_axis_ps()

    rows = []
    for flu in FLUENCE_LIST:
        row = simulate_one_fluence(
            p_global=p_global,
            fluence_ratio=flu,
            t_plot_ps=t_plot_ps,
            custom_channel=CUSTOM_CHANNEL,
        )
        rows.append(row)

    # figures
    plot_combined_panels(
        rows,
        custom_channel=CUSTOM_CHANNEL,
        outpath=OUTPUT_DIR / f"combined_temperature_panels_{OUTPUT_TAG}.png",
    )
    plot_single_channel(
        rows,
        key="Te",
        ylabel="Te (K)",
        title="Electron temperature vs time",
        outpath=OUTPUT_DIR / f"Te_only_{OUTPUT_TAG}.png",
    )
    plot_single_channel(
        rows,
        key="Tl",
        ylabel="Tl (K)",
        title="Lattice temperature vs time",
        outpath=OUTPUT_DIR / f"Tl_only_{OUTPUT_TAG}.png",
    )
    plot_single_channel(
        rows,
        key=CUSTOM_CHANNEL,
        ylabel=get_channel_label(CUSTOM_CHANNEL),
        title=f"{CUSTOM_CHANNEL} vs time",
        outpath=OUTPUT_DIR / f"{CUSTOM_CHANNEL}_only_{OUTPUT_TAG}.png",
    )

    # data export
    write_summary_csv(
        rows,
        custom_channel=CUSTOM_CHANNEL,
        outpath=OUTPUT_DIR / f"temperature_summary_{OUTPUT_TAG}.csv",
    )
    write_meta_json(
        rows,
        p_global=p_global,
        source_json=json_path,
        custom_channel=CUSTOM_CHANNEL,
        outpath=OUTPUT_DIR / f"temperature_meta_{OUTPUT_TAG}.json",
    )

    print(f"[temp-plot] source mainfit json = {json_path}")
    print(f"[temp-plot] output dir = {OUTPUT_DIR}")
    print(f"[temp-plot] custom channel = {CUSTOM_CHANNEL}")
    print("[temp-plot] files:")
    print(f"  - {OUTPUT_DIR / f'combined_temperature_panels_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'Te_only_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'Tl_only_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'{CUSTOM_CHANNEL}_only_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'temperature_summary_{OUTPUT_TAG}.csv'}")
    print(f"  - {OUTPUT_DIR / f'temperature_meta_{OUTPUT_TAG}.json'}")


if __name__ == "__main__":
    main()