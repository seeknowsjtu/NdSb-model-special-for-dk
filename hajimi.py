# -*- coding: utf-8 -*-
"""
plot_reproduce_old_raw_m_chi2q.py
---------------------------------
Strictly reproduce the old raw_m_chi2q-style overlay.

This script is NOT a new delta-k relative model.
It reproduces the old S-like readout:

    y_fit = B_obs_i + A_obs_i * (m * chi2q)

where A_obs_i and B_obs_i are taken from the old JSON best_local_params.

Use case:
    old JSON:
        observable_mode = raw_m_chi2q

    old data:
        F:\\python4git\\simulate_dk\\mdc data\\deltak12k_1p0mW.csv
        F:\\python4git\\simulate_dk\\mdc data\\deltak12k_2p0mW.csv
        F:\\python4git\\simulate_dk\\mdc data\\deltak12k_2p5mW.csv

Outputs:
    old_raw_m_chi2q_overlay_*.png
    old_raw_m_chi2q_fitcurves_*.csv
    old_raw_m_chi2q_summary_*.csv
    each_dataset/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import default_params, normalize_params_dict
from physics_engine import DebyeCl
from solver import NdSb3TM


# ============================================================
# User settings
# ============================================================

ROOT_DIR = Path(r"F:\python4git\simulate_dk")

JSON_PATH = ROOT_DIR / (
    r"fit_results\real_multi_fit_round1"
    r"\deltak12k_globalfit_20260328_083202"
    r"\globalfit_deltak12k_raw_m_chi2q_1p0to2p5mW_20260328_083202.json"
)

CSV_FILES = [
    ROOT_DIR / r"mdc data\deltak12k_1p0mW.csv",
    ROOT_DIR / r"mdc data\deltak12k_2p0mW.csv",
    ROOT_DIR / r"mdc data\deltak12k_2p5mW.csv",
]

OUT_DIR = ROOT_DIR / r"fit_results\reproduce_old_raw_m_chi2q"
OUTPUT_TAG = "20260328"

FIG_DPI = 200
LINE_WIDTH = 2.0
MARKER_SIZE = 4.5

# The old fit used the global dt0_ps and sigma_irf_ps in the JSON.
# Keep these False/None unless you deliberately want to test something.
FORCE_DT0_PS: float | None = None
FORCE_SIGMA_IRF_PS: float | None = None


# ============================================================
# Basic helpers
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON does not exist:\n{path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_best_global_params(payload: dict) -> dict:
    if "best_global_params" not in payload:
        raise KeyError("JSON is missing 'best_global_params'.")

    p = normalize_params_dict(default_params())
    p.update(payload["best_global_params"])

    if FORCE_DT0_PS is not None:
        p["dt0_ps"] = float(FORCE_DT0_PS)

    if FORCE_SIGMA_IRF_PS is not None:
        p["sigma_irf_ps"] = float(FORCE_SIGMA_IRF_PS)

    return p


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for key in candidates:
        if key.lower() in col_map:
            return col_map[key.lower()]
    return None


def parse_fluence_from_filename(path: Path) -> float:
    name = path.name

    # examples:
    #   deltak12k_1p0mW.csv
    #   deltak12k_2p5mW.csv
    import re
    m = re.search(r"(\d+)p(\d+)mW", name, flags=re.IGNORECASE)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")

    m = re.search(r"(\d+)mW", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    raise ValueError(f"Cannot parse fluence from filename: {name}")


def load_old_mdc_csv(path: Path) -> dict:
    """
    Load old mdc data.

    The old run treated the measured delta-k-like trace as an S-like observable.
    Therefore this loader accepts either:
        delta_k / deltak / dk / ...
    or:
        S / s
    """
    if not path.exists():
        raise FileNotFoundError(f"Data CSV does not exist:\n{path}")

    df = pd.read_csv(path)

    t_col = find_column(df, ["t_ps", "tps", "time_ps", "time", "t"])
    y_col = find_column(
        df,
        [
            "delta_k",
            "deltak",
            "dk",
            "delta_k_ainv",
            "deltak_ainv",
            "k_split",
            "ksplit",
            "split_k",
            "S",
            "s",
        ],
    )
    sigma_col = find_column(
        df,
        [
            "sigma_dk",
            "sigmadeltak12_k",
            "delta_k_err",
            "deltak_err",
            "dk_err",
            "err",
            "error",
            "sigma",
        ],
    )

    if t_col is None:
        raise ValueError(f"{path.name}: cannot find time column. Columns={list(df.columns)}")
    if y_col is None:
        raise ValueError(
            f"{path.name}: cannot find observable column. "
            f"Expected delta_k/dk/S-like column. Columns={list(df.columns)}"
        )

    t_ps = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

    if sigma_col is not None:
        sigma = pd.to_numeric(df[sigma_col], errors="coerce").to_numpy(dtype=float)
    else:
        sigma = None

    mask = np.isfinite(t_ps) & np.isfinite(y)
    if sigma is not None:
        mask &= np.isfinite(sigma)

    t_ps = t_ps[mask]
    y = y[mask]
    sigma = sigma[mask] if sigma is not None else None

    idx = np.argsort(t_ps)

    return {
        "path": path,
        "name": path.name,
        "fluence_ratio": parse_fluence_from_filename(path),
        "t_ps": t_ps[idx],
        "t_sec": t_ps[idx] * 1e-12,
        "y_exp": y[idx],
        "sigma": sigma[idx] if sigma is not None else None,
        "y_column": y_col,
    }


# ============================================================
# Model / observable helpers
# ============================================================

def compute_chi2q_from_sim(sim: dict) -> np.ndarray:
    """
    Same convention as data_io._compute_chi2q.

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


def gaussian_irf_kernel(dt_ps: float, sigma_ps: float, half_width_sigma: float = 5.0) -> np.ndarray:
    sigma_ps = float(sigma_ps)
    if sigma_ps <= 0.0:
        return np.array([1.0], dtype=float)

    dt_ps = abs(float(dt_ps))
    if dt_ps <= 0.0 or not np.isfinite(dt_ps):
        dt_ps = sigma_ps / 5.0

    half_width_ps = max(float(half_width_sigma), 1.0) * sigma_ps
    n_half = int(np.ceil(half_width_ps / dt_ps))

    x = np.arange(-n_half, n_half + 1, dtype=float) * dt_ps
    k = np.exp(-0.5 * (x / sigma_ps) ** 2)
    s = float(np.sum(k))

    if not np.isfinite(s) or s <= 0.0:
        return np.array([1.0], dtype=float)

    return k / s


def convolve_with_irf(y: np.ndarray, t_ps: np.ndarray, sigma_ps: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    t_ps = np.asarray(t_ps, dtype=float)

    if float(sigma_ps) <= 0.0 or y.size < 3:
        return y.copy()

    diffs = np.diff(t_ps)
    diffs = diffs[np.isfinite(diffs)]

    if diffs.size == 0:
        return y.copy()

    dt_ps = float(np.mean(np.abs(diffs)))
    if not np.isfinite(dt_ps) or dt_ps <= 0.0:
        return y.copy()

    kernel = gaussian_irf_kernel(dt_ps, sigma_ps)
    return np.convolve(y, kernel, mode="same")


def get_local_ab_from_json(payload: dict, dataset_name: str) -> tuple[float, float]:
    """
    Strictly read A_obs/B_obs from JSON best_local_params.
    """
    local_all = payload.get("best_local_params", {})
    if dataset_name not in local_all:
        raise KeyError(
            f"Cannot find local params for {dataset_name} in JSON.\n"
            f"Available keys: {list(local_all.keys())}"
        )

    local = local_all[dataset_name]

    if "A_obs" not in local or "B_obs" not in local:
        raise KeyError(f"Local params for {dataset_name} must contain A_obs and B_obs.")

    return float(local["A_obs"]), float(local["B_obs"])


def simulate_old_raw_m_chi2q_for_dataset(
    payload: dict,
    p_global: dict,
    dataset: dict,
) -> dict:
    """
    Old readout:
        y_fit = B_obs_i + A_obs_i * convolve[m(t) * chi2q(t)]
    """
    p_run = dict(p_global)
    fluence = float(dataset["fluence_ratio"])
    p_run["fluence_multiplier"] = fluence

    dt0_ps = float(p_global.get("dt0_ps", 0.0))
    alpha_dt = float(p_global.get("alpha_dt_per_F", 0.0))
    dt_i_ps = dt0_ps + alpha_dt * (fluence - 1.0)

    sigma_irf_ps = float(p_global.get("sigma_irf_ps", 0.0))

    t_model_sec = np.asarray(dataset["t_sec"], dtype=float) - dt_i_ps * 1e-12

    debye_obj = DebyeCl(thetaD=float(p_run["ThetaD"]))
    model = NdSb3TM(p_run, debye_obj=debye_obj)

    sim = model.simulate_aligned(t_model_sec, with_diag=True)

    m = np.asarray(sim["m"], dtype=float)
    eta = np.asarray(sim["eta"], dtype=float)
    phi = np.asarray(sim["phi"], dtype=float)
    chi2q = compute_chi2q_from_sim(sim)

    u_raw = m * chi2q
    u = convolve_with_irf(u_raw, dataset["t_ps"], sigma_irf_ps)

    A_obs, B_obs = get_local_ab_from_json(payload, dataset["name"])
    y_fit = B_obs + A_obs * u

    residual = y_fit - dataset["y_exp"]

    out = dict(dataset)
    out.update(
        {
            "A_obs": A_obs,
            "B_obs": B_obs,
            "dt_i_ps": dt_i_ps,
            "sigma_irf_ps": sigma_irf_ps,
            "u": u,
            "u_raw": u_raw,
            "y_fit": y_fit,
            "residual": residual,
            "m": m,
            "eta": eta,
            "phi": phi,
            "chi2q": chi2q,
            "Te": np.asarray(sim["Te"], dtype=float),
            "Ts": np.asarray(sim["Ts"], dtype=float),
            "Tl": np.asarray(sim["Tl"], dtype=float),
        }
    )

    return out


# ============================================================
# Plot / export
# ============================================================

def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda x: float(x["fluence_ratio"]))


def plot_overlay(rows: list[dict], out_path: Path, json_mode: str) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.5))

    for row in rows:
        label = f"{row['name']} ({row['fluence_ratio']:.1f} mW)"

        sigma = row.get("sigma", None)
        if sigma is not None:
            ax.errorbar(
                row["t_ps"],
                row["y_exp"],
                yerr=sigma,
                fmt="o",
                markersize=MARKER_SIZE,
                capsize=2,
                alpha=0.55,
                label=f"exp {label}",
            )
        else:
            ax.plot(
                row["t_ps"],
                row["y_exp"],
                "o",
                markersize=MARKER_SIZE,
                alpha=0.55,
                label=f"exp {label}",
            )

        ax.plot(
            row["t_ps"],
            row["y_fit"],
            "-",
            lw=LINE_WIDTH,
            label=f"fit {label}",
        )

    ax.set_xlabel("time (ps)")
    ax.set_ylabel("S / delta-k-like observable")
    ax.set_title(f"Reproduced old global fit overlay ({json_mode})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_each(rows: list[dict], out_dir: Path) -> None:
    each_dir = out_dir / "each_dataset"
    each_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))

        sigma = row.get("sigma", None)
        if sigma is not None:
            ax.errorbar(
                row["t_ps"],
                row["y_exp"],
                yerr=sigma,
                fmt="o",
                markersize=MARKER_SIZE,
                capsize=2,
                alpha=0.65,
                label="exp",
            )
        else:
            ax.plot(
                row["t_ps"],
                row["y_exp"],
                "o",
                markersize=MARKER_SIZE,
                alpha=0.65,
                label="exp",
            )

        ax.plot(
            row["t_ps"],
            row["y_fit"],
            "-",
            lw=LINE_WIDTH,
            label="fit",
        )

        ax.set_xlabel("time (ps)")
        ax.set_ylabel("S / delta-k-like observable")
        ax.set_title(
            f"{row['name']} | {row['fluence_ratio']:.1f} mW\n"
            f"A_obs={row['A_obs']:.6g}, B_obs={row['B_obs']:.6g}, "
            f"dt_i={row['dt_i_ps']:.4g} ps"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        out_path = each_dir / f"{Path(row['name']).stem}_old_raw_m_chi2q.png"
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)


def write_fitcurves_csv(rows: list[dict], out_path: Path) -> None:
    all_rows = []

    for row in rows:
        sigma = row.get("sigma", None)
        if sigma is None:
            sigma = np.full_like(row["y_exp"], np.nan, dtype=float)

        for i in range(len(row["t_ps"])):
            all_rows.append(
                {
                    "dataset": row["name"],
                    "fluence_ratio": row["fluence_ratio"],
                    "t_ps": row["t_ps"][i],
                    "y_exp": row["y_exp"][i],
                    "sigma": sigma[i],
                    "y_fit": row["y_fit"][i],
                    "residual": row["residual"][i],
                    "u_m_chi2q": row["u"][i],
                    "u_raw_m_chi2q": row["u_raw"][i],
                    "m": row["m"][i],
                    "eta": row["eta"][i],
                    "phi": row["phi"][i],
                    "chi2q": row["chi2q"][i],
                    "Te": row["Te"][i],
                    "Ts": row["Ts"][i],
                    "Tl": row["Tl"][i],
                    "A_obs": row["A_obs"],
                    "B_obs": row["B_obs"],
                    "dt_i_ps": row["dt_i_ps"],
                    "sigma_irf_ps": row["sigma_irf_ps"],
                    "y_column": row["y_column"],
                }
            )

    pd.DataFrame(all_rows).to_csv(out_path, index=False)


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    summary_rows = []

    for row in rows:
        rms = float(np.sqrt(np.mean(row["residual"] ** 2)))
        sigma = row.get("sigma", None)
        if sigma is not None:
            wrms = float(np.sqrt(np.mean((row["residual"] / np.maximum(sigma, 1e-12)) ** 2)))
        else:
            wrms = np.nan

        summary_rows.append(
            {
                "dataset": row["name"],
                "fluence_ratio": row["fluence_ratio"],
                "n_points": len(row["t_ps"]),
                "A_obs": row["A_obs"],
                "B_obs": row["B_obs"],
                "rms": rms,
                "wrms_if_sigma": wrms,
                "dt_i_ps": row["dt_i_ps"],
                "sigma_irf_ps": row["sigma_irf_ps"],
                "y_column": row["y_column"],
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_path, index=False)


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = load_json(JSON_PATH)
    p_global = get_best_global_params(payload)

    json_mode = str(payload.get("observable_mode", "unknown"))
    json_target = str(payload.get("target_kind", "unknown"))

    if json_mode != "raw_m_chi2q":
        print(
            f"[warning] JSON observable_mode is {json_mode}, "
            "but this script is designed to reproduce raw_m_chi2q."
        )

    print(f"[info] source JSON = {JSON_PATH}")
    print(f"[info] JSON target_kind = {json_target}")
    print(f"[info] JSON observable_mode = {json_mode}")
    print("[info] readout formula = B_obs_i + A_obs_i * convolved(m * chi2q)")
    print(f"[info] dt0_ps = {float(p_global.get('dt0_ps', np.nan)):.6g}")
    print(f"[info] sigma_irf_ps = {float(p_global.get('sigma_irf_ps', np.nan)):.6g}")

    datasets = []
    for path in CSV_FILES:
        ds = load_old_mdc_csv(path)
        datasets.append(ds)
        print(
            f"[load] {ds['name']} | F={ds['fluence_ratio']:.2f} | "
            f"N={len(ds['t_ps'])} | y_col={ds['y_column']}"
        )

    datasets = sort_rows(datasets)

    rows = []
    for ds in datasets:
        row = simulate_old_raw_m_chi2q_for_dataset(payload, p_global, ds)
        rows.append(row)
        print(
            f"[fit] {row['name']} | A={row['A_obs']:.6g} | "
            f"B={row['B_obs']:.6g} | rms={np.sqrt(np.mean(row['residual'] ** 2)):.6e}"
        )

    rows = sort_rows(rows)

    overlay_path = OUT_DIR / f"old_raw_m_chi2q_overlay_{OUTPUT_TAG}.png"
    fitcurves_path = OUT_DIR / f"old_raw_m_chi2q_fitcurves_{OUTPUT_TAG}.csv"
    summary_path = OUT_DIR / f"old_raw_m_chi2q_summary_{OUTPUT_TAG}.csv"

    plot_overlay(rows, overlay_path, json_mode=json_mode)
    plot_each(rows, OUT_DIR)
    write_fitcurves_csv(rows, fitcurves_path)
    write_summary_csv(rows, summary_path)

    print("[done] outputs:")
    print(f"  - {overlay_path}")
    print(f"  - {OUT_DIR / 'each_dataset'}")
    print(f"  - {fitcurves_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()