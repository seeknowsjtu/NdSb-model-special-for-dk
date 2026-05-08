# -*- coding: utf-8 -*-
"""
plot_reproduce_old_raw_m_chi2q_6groups.py
-----------------------------------------
Reproduce / extend the old raw_m_chi2q-style readout for six MDC delta-k datasets.

This script is NOT the newer delta-k relative model. It uses the old S-like readout
form used by the 20260328 raw_m_chi2q result:

    y_fit = B_obs_i + A_obs_i * convolved[m(t) * chi2q(t)]

Behavior:
    1) For datasets that exist in the old JSON best_local_params, use JSON A_obs/B_obs.
       Typically: 1.0, 2.0, 2.5 mW.

    2) For datasets missing from the old JSON, keep the same JSON global dynamics,
       simulate u(t)=convolved[m*chi2q], then fit only per-dataset A_obs/B_obs:

           y_exp ~= B_obs_i + A_obs_i * u(t)

       Typically: 3.0, 3.5, 4.0 mW.

Outputs:
    old_raw_m_chi2q_overlay_*.png
    old_raw_m_chi2q_ab_vs_fluence_*.png
    old_raw_m_chi2q_residual_overlay_*.png
    old_raw_m_chi2q_fitcurves_*.csv
    old_raw_m_chi2q_summary_*.csv
    each_dataset/*.png

Usage:
    python plot_reproduce_old_raw_m_chi2q_6groups.py
"""

from __future__ import annotations

import json
import re
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

# Six old MDC datasets. These names are important because the first three names
# match the keys in JSON best_local_params.
CSV_FILES = [
    ROOT_DIR / r"mdc data\deltak12k_1p0mW.csv",
    ROOT_DIR / r"mdc data\deltak12k_2p0mW.csv",
    ROOT_DIR / r"mdc data\deltak12k_2p5mW.csv",
    ROOT_DIR / r"mdc data\deltak12k_3p0mW.csv",
    ROOT_DIR / r"mdc data\deltak12k_3p5mW.csv",
    ROOT_DIR / r"mdc data\deltak12k_4p0mW.csv",
]

OUT_DIR = ROOT_DIR / r"fit_results\reproduce_old_raw_m_chi2q_6groups"
OUTPUT_TAG = "old_json_extend_1p0to4p0mW"

FIG_DPI = 200
LINE_WIDTH = 2.0
MARKER_SIZE = 4.5

# Keep these as None for faithful reproduction/extension from the JSON.
# Set to a number only for a controlled diagnostic test.
FORCE_DT0_PS: float | None = None
FORCE_SIGMA_IRF_PS: float | None = None

# Fitting policy for datasets missing JSON A/B.
# First three should use JSON A/B; later datasets will fit A/B if missing.
FIT_AB_FOR_MISSING_JSON_LOCAL = True

# Use sigma column as weights when fitting A/B for missing local datasets.
USE_SIGMA_WEIGHT_FOR_AB_FIT = True

# Optional bounds for fitted A/B on missing local datasets.
# Use None for no bound. These are deliberately broad.
A_FIT_BOUNDS: tuple[float, float] | None = None  # e.g. (0.0, 0.8)
B_FIT_BOUNDS: tuple[float, float] | None = None  # e.g. (-0.2, 0.2)


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
    m = re.search(r"(\d+)p(\d+)mW", name, flags=re.IGNORECASE)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")

    m = re.search(r"(\d+)mW", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    raise ValueError(f"Cannot parse fluence from filename: {name}")


def load_old_mdc_csv(path: Path) -> dict:
    """
    Load old MDC data.

    The old run treated the measured delta-k-like trace as an S-like observable.
    Therefore this loader accepts either delta_k/deltak/dk-like columns or S-like columns.
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
        "sigma_column": sigma_col,
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


def get_json_local_ab(payload: dict, dataset_name: str) -> tuple[float, float, str] | None:
    """
    Read A_obs/B_obs from JSON best_local_params if available.
    Return None if not found.
    """
    local_all = payload.get("best_local_params", {})
    if dataset_name not in local_all:
        return None

    local = local_all[dataset_name]
    if "A_obs" not in local or "B_obs" not in local:
        raise KeyError(f"Local params for {dataset_name} must contain A_obs and B_obs.")

    return float(local["A_obs"]), float(local["B_obs"]), "json_best_local_params"


def fit_ab_for_dataset(u: np.ndarray, y: np.ndarray, sigma: np.ndarray | None) -> tuple[float, float]:
    """
    Fit y = B + A*u.

    Returns:
        A, B
    """
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(u) & np.isfinite(y)

    if sigma is not None and USE_SIGMA_WEIGHT_FOR_AB_FIT:
        sigma = np.asarray(sigma, dtype=float)
        mask &= np.isfinite(sigma) & (sigma > 0.0)

    if np.count_nonzero(mask) < 2:
        raise ValueError("Need at least two finite points to fit A/B.")

    x = u[mask]
    yy = y[mask]

    X = np.column_stack([x, np.ones_like(x)])

    if sigma is not None and USE_SIGMA_WEIGHT_FOR_AB_FIT:
        w = 1.0 / np.maximum(sigma[mask], 1e-12)
        X_fit = X * w[:, None]
        y_fit = yy * w
    else:
        X_fit = X
        y_fit = yy

    coef, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    A = float(coef[0])
    B = float(coef[1])

    if A_FIT_BOUNDS is not None:
        A = float(np.clip(A, A_FIT_BOUNDS[0], A_FIT_BOUNDS[1]))
    if B_FIT_BOUNDS is not None:
        B = float(np.clip(B, B_FIT_BOUNDS[0], B_FIT_BOUNDS[1]))

    return A, B


def simulate_old_raw_m_chi2q_for_dataset(
    payload: dict,
    p_global: dict,
    dataset: dict,
) -> dict:
    """
    Old readout / extension:
        y_fit = B_obs_i + A_obs_i * convolve[m(t) * chi2q(t)]

    A/B source:
        - JSON local params if present.
        - Otherwise fit A/B from this dataset if FIT_AB_FOR_MISSING_JSON_LOCAL=True.
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

    ab_from_json = get_json_local_ab(payload, dataset["name"])
    if ab_from_json is not None:
        A_obs, B_obs, readout_source = ab_from_json
    else:
        if not FIT_AB_FOR_MISSING_JSON_LOCAL:
            raise KeyError(
                f"{dataset['name']} is missing from JSON best_local_params and "
                "FIT_AB_FOR_MISSING_JSON_LOCAL=False."
            )
        A_obs, B_obs = fit_ab_for_dataset(u, dataset["y_exp"], dataset.get("sigma"))
        readout_source = "fitted_AB_missing_json_local"

    y_fit = B_obs + A_obs * u
    residual = y_fit - dataset["y_exp"]

    out = dict(dataset)
    out.update(
        {
            "A_obs": float(A_obs),
            "B_obs": float(B_obs),
            "readout_source": readout_source,
            "dt_i_ps": float(dt_i_ps),
            "sigma_irf_ps": float(sigma_irf_ps),
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
        label = f"{row['fluence_ratio']:.1f} mW"
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

        source_tag = "JSON A/B" if row["readout_source"] == "json_best_local_params" else "fit A/B"
        ax.plot(
            row["t_ps"],
            row["y_fit"],
            "-",
            lw=LINE_WIDTH,
            label=f"fit {label} ({source_tag})",
        )

    ax.set_xlabel("time (ps)")
    ax.set_ylabel("delta-k-like observable")
    ax.set_title(f"Old raw_m_chi2q readout: JSON dynamics + per-dataset A/B ({json_mode})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_residual_overlay(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.5))

    for row in rows:
        label = f"{row['fluence_ratio']:.1f} mW"
        ax.axhline(0.0, lw=0.8, alpha=0.3)
        ax.plot(
            row["t_ps"],
            row["residual"],
            "o-",
            markersize=3.0,
            lw=1.2,
            label=label,
        )

    ax.set_xlabel("time (ps)")
    ax.set_ylabel("fit - exp")
    ax.set_title("Residuals: old raw_m_chi2q readout")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_ab_vs_fluence(rows: list[dict], out_path: Path) -> None:
    rows = sort_rows(rows)
    flu = np.array([r["fluence_ratio"] for r in rows], dtype=float)
    A = np.array([r["A_obs"] for r in rows], dtype=float)
    B = np.array([r["B_obs"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)

    for i, row in enumerate(rows):
        marker = "o" if row["readout_source"] == "json_best_local_params" else "s"
        label = "JSON A/B" if row["readout_source"] == "json_best_local_params" else "fitted A/B"
        # avoid duplicate labels
        if label in [h.get_label() for h in axes[0].lines]:
            label = None
        axes[0].plot(flu[i], A[i], marker=marker, markersize=7, linestyle="None", label=label)
        axes[1].plot(flu[i], B[i], marker=marker, markersize=7, linestyle="None", label=label)

    axes[0].plot(flu, A, "--", lw=1.0, alpha=0.5)
    axes[1].plot(flu, B, "--", lw=1.0, alpha=0.5)

    axes[0].set_ylabel("A_obs")
    axes[1].set_ylabel("B_obs")
    axes[1].set_xlabel("fluence label (mW)")
    axes[0].set_title("Per-dataset readout parameters")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_each(rows: list[dict], out_dir: Path) -> None:
    each_dir = out_dir / "each_dataset"
    each_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax, axr = axes
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
            ax.plot(row["t_ps"], row["y_exp"], "o", markersize=MARKER_SIZE, alpha=0.65, label="exp")

        ax.plot(row["t_ps"], row["y_fit"], "-", lw=LINE_WIDTH, label="fit")
        ax.set_ylabel("delta-k-like observable")
        ax.set_title(
            f"{row['name']} | {row['fluence_ratio']:.1f} mW | {row['readout_source']}\n"
            f"A_obs={row['A_obs']:.6g}, B_obs={row['B_obs']:.6g}, "
            f"dt_i={row['dt_i_ps']:.4g} ps, sigma_irf={row['sigma_irf_ps']:.3g} ps"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        axr.axhline(0.0, lw=0.9, alpha=0.4)
        axr.plot(row["t_ps"], row["residual"], "o-", markersize=3.0, lw=1.1)
        axr.set_xlabel("time (ps)")
        axr.set_ylabel("fit-exp")
        axr.grid(True, alpha=0.3)

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
                    "readout_source": row["readout_source"],
                    "dt_i_ps": row["dt_i_ps"],
                    "sigma_irf_ps": row["sigma_irf_ps"],
                    "y_column": row["y_column"],
                    "sigma_column": row["sigma_column"],
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
                "readout_source": row["readout_source"],
                "rms": rms,
                "wrms_if_sigma": wrms,
                "dt_i_ps": row["dt_i_ps"],
                "sigma_irf_ps": row["sigma_irf_ps"],
                "y_column": row["y_column"],
                "sigma_column": row["sigma_column"],
                "u_min": float(np.nanmin(row["u"])),
                "u_max": float(np.nanmax(row["u"])),
                "y_exp_min": float(np.nanmin(row["y_exp"])),
                "y_exp_max": float(np.nanmax(row["y_exp"])),
                "y_fit_min": float(np.nanmin(row["y_fit"])),
                "y_fit_max": float(np.nanmax(row["y_fit"])),
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
            "but this script is designed for raw_m_chi2q readout."
        )

    print(f"[info] source JSON = {JSON_PATH}")
    print(f"[info] JSON target_kind = {json_target}")
    print(f"[info] JSON observable_mode = {json_mode}")
    print("[info] readout formula = B_obs_i + A_obs_i * convolved(m * chi2q)")
    print("[info] first datasets use JSON A/B if available; missing datasets fit A/B only")
    print(f"[info] dt0_ps = {float(p_global.get('dt0_ps', np.nan)):.6g}")
    print(f"[info] sigma_irf_ps = {float(p_global.get('sigma_irf_ps', np.nan)):.6g}")

    datasets = []
    for path in CSV_FILES:
        ds = load_old_mdc_csv(path)
        datasets.append(ds)
        print(
            f"[load] {ds['name']} | F={ds['fluence_ratio']:.2f} | "
            f"N={len(ds['t_ps'])} | y_col={ds['y_column']} | sigma_col={ds['sigma_column']}"
        )

    datasets = sort_rows(datasets)

    rows = []
    for ds in datasets:
        row = simulate_old_raw_m_chi2q_for_dataset(payload, p_global, ds)
        rows.append(row)
        print(
            f"[fit] {row['name']} | source={row['readout_source']} | "
            f"A={row['A_obs']:.6g} | B={row['B_obs']:.6g} | "
            f"rms={np.sqrt(np.mean(row['residual'] ** 2)):.6e}"
        )

    rows = sort_rows(rows)

    overlay_path = OUT_DIR / f"old_raw_m_chi2q_overlay_{OUTPUT_TAG}.png"
    residual_path = OUT_DIR / f"old_raw_m_chi2q_residual_overlay_{OUTPUT_TAG}.png"
    ab_path = OUT_DIR / f"old_raw_m_chi2q_ab_vs_fluence_{OUTPUT_TAG}.png"
    fitcurves_path = OUT_DIR / f"old_raw_m_chi2q_fitcurves_{OUTPUT_TAG}.csv"
    summary_path = OUT_DIR / f"old_raw_m_chi2q_summary_{OUTPUT_TAG}.csv"

    plot_overlay(rows, overlay_path, json_mode=json_mode)
    plot_each(rows, OUT_DIR)
    plot_residual_overlay(rows, residual_path)
    plot_ab_vs_fluence(rows, ab_path)
    write_fitcurves_csv(rows, fitcurves_path)
    write_summary_csv(rows, summary_path)

    print("[done] outputs:")
    print(f"  - {overlay_path}")
    print(f"  - {residual_path}")
    print(f"  - {ab_path}")
    print(f"  - {OUT_DIR / 'each_dataset'}")
    print(f"  - {fitcurves_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
