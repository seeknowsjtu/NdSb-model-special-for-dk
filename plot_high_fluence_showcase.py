# -*- coding: utf-8 -*-
"""
plot_high_fluence_showcase.py
-----------------------------
Use the 3-group main fit (1.0 / 2.0 / 2.5 mW) as the fixed global model,
then project / showcase the higher-fluence traces (3.0 / 3.5 / 4.0 mW).

Outputs:
1) A combined overview figure:
   - training groups: exp points + solid fitted curves
   - showcase groups: exp points + dashed projected curves

2) A high-fluence-only figure:
   - 3 stacked panels for 3.0 / 3.5 / 4.0 mW

3) A readout-trend figure:
   - A_obs and B_obs vs fluence
   - training groups shown as fitted anchors
   - showcase groups shown as projected points

4) A CSV summary of showcase projection results.

Usage:
    python plot_high_fluence_showcase.py
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
from data_io import load_s_dataset_csv_raw, build_observable
from physics_engine import DebyeCl
from solver import NdSb3TM


# ============================================================
# User settings
# ============================================================
DATA_DIR = Path(".")
MAINFIT_JSON = Path("F:\python4git\simulate\fit_results\real_multi_fit_smoke\deltak12k_globalfit_20260327_100946\globalfit_deltak12k_raw_m_chi2q_1p0to2p5mW_20260327_100946.json")
# 你可以把上面改成你实际的 3-group JSON 文件名

TRAIN_CSV_FILES = [
    "deltak12k_1p0mW.csv",
    "deltak12k_2p0mW.csv",
    "deltak12k_2p5mW.csv",
]

SHOWCASE_CSV_FILES = [
    "deltak12k_3p0mW.csv",
    "deltak12k_3p5mW.csv",
    "deltak12k_4p0mW.csv",
]

OUTPUT_DIR = Path("fit_results/high_fluence_showcase")
OUTPUT_TAG = "mainfit3_showcase3454"

OBSERVABLE_MODE_DEFAULT = "raw_m_chi2q"

# bounds used only for linear readout projection on showcase traces
A_BOUNDS = (1.0e-4, 1.5e-1)
B_BOUNDS = (-2.0e-2, 8.0e-2)

# For dt(F) = dt0 + alpha*(F - F_ref)
F_REF = 1.0


# ============================================================
# Small helpers
# ============================================================
def find_latest_json(pattern: str) -> Path | None:
    candidates = sorted(Path(".").glob(pattern))
    return candidates[-1] if candidates else None


def resolve_mainfit_json() -> Path:
    if MAINFIT_JSON.exists():
        return MAINFIT_JSON

    latest = find_latest_json("fit_results/**/globalfit_deltak12k_raw_m_chi2q_1p0to2p5mW_*.json")
    if latest is not None:
        return latest

    raise FileNotFoundError(
        "Cannot find the 3-group main-fit JSON. "
        "Please set MAINFIT_JSON to your actual exported file."
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(path: Path) -> dict:
    ds = load_s_dataset_csv_raw(path)
    return ds


def build_template_fallback(dataset: dict, p_global: dict, observable_mode: str) -> tuple[np.ndarray, float, float]:
    """
    Fallback template builder that uses the current solver + build_observable.

    It constructs a unit-amplitude / zero-background template:
        template_u = build_observable(sim, p_dataset, {"A_obs":1, "B_obs":0}, mode)

    Notes:
    - dt_i is global dt0 + optional alpha_dt_per_F * (F - F_ref)
    - If your project's newer data_io.py exposes a dedicated
      build_observable_template_only(), this script will try to use it first.
    """
    fluence_ratio = float(dataset["fluence_ratio"])
    dt0_ps = float(p_global.get("dt0_ps", 0.0))
    alpha_dt_per_F = float(p_global.get("alpha_dt_per_F", 0.0))
    sigma_irf_ps = float(p_global.get("sigma_irf_ps", 0.0))
    dt_i_ps = dt0_ps + alpha_dt_per_F * (fluence_ratio - F_REF)

    p_dataset = dict(p_global)
    p_dataset["fluence_multiplier"] = fluence_ratio

    t = np.asarray(dataset["t"], dtype=float)
    t_model = t - dt_i_ps * 1e-12

    debye_obj = DebyeCl(thetaD=float(p_dataset["ThetaD"]))
    model = NdSb3TM(p_dataset, debye_obj=debye_obj)
    sim = model.simulate_aligned(t_model, with_diag=False)

    # Force template-only form: B=0, A=1
    template_u = build_observable(
        sim,
        p_dataset,
        {"A_obs": 1.0, "B_obs": 0.0},
        observable_mode,
    )
    return np.asarray(template_u, dtype=float), dt_i_ps, sigma_irf_ps


def build_template(dataset: dict, p_global: dict, observable_mode: str) -> tuple[np.ndarray, float, float]:
    """
    Prefer the dedicated template-only helper if your current data_io.py exposes it.
    Otherwise fallback to a robust local implementation.
    """
    try:
        import data_io  # local project module
        if hasattr(data_io, "build_observable_template_only"):
            template_u, dt_i_ps, sigma_irf_ps = data_io.build_observable_template_only(
                p_global,
                dataset,
                observable_mode,
            )
            return (
                np.asarray(template_u, dtype=float),
                float(dt_i_ps),
                float(sigma_irf_ps),
            )
    except Exception:
        pass

    return build_template_fallback(dataset, p_global, observable_mode)


def solve_linear_ab(
    y_obs: np.ndarray,
    template_u: np.ndarray,
    a_bounds: tuple[float, float] = A_BOUNDS,
    b_bounds: tuple[float, float] = B_BOUNDS,
) -> tuple[float, float, np.ndarray, float]:
    """
    Solve y ≈ A * u + B with ordinary least squares, then clip to bounds.
    """
    y = np.asarray(y_obs, dtype=float).ravel()
    u = np.asarray(template_u, dtype=float).ravel()

    if y.shape != u.shape:
        raise ValueError(f"shape mismatch: y{y.shape} vs u{u.shape}")

    X = np.column_stack([u, np.ones_like(u)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a_raw, b_raw = float(coef[0]), float(coef[1])

    a = float(np.clip(a_raw, a_bounds[0], a_bounds[1]))
    b = float(np.clip(b_raw, b_bounds[0], b_bounds[1]))
    y_fit = a * u + b
    rms = float(np.sqrt(np.mean((y_fit - y) ** 2)))
    return a, b, y_fit, rms


def get_best_global_params(payload: dict) -> dict:
    if "best_global_params" not in payload:
        raise KeyError("JSON missing key 'best_global_params'.")
    p = normalize_params_dict(default_params())
    p.update(payload["best_global_params"])
    return p


def get_observable_mode(payload: dict) -> str:
    return str(payload.get("observable_mode", OBSERVABLE_MODE_DEFAULT))


def dataset_summary_map(payload: dict) -> dict[str, dict]:
    rows = payload.get("dataset_summary", [])
    return {str(row.get("dataset_name", row.get("dataset", ""))): row for row in rows}


def best_local_map(payload: dict) -> dict[str, dict]:
    return payload.get("best_local_params", {})


def project_dataset(
    dataset: dict,
    p_global: dict,
    observable_mode: str,
    use_known_ab: dict | None = None,
) -> dict:
    template_u, dt_i_ps, sigma_irf_ps = build_template(dataset, p_global, observable_mode)

    if use_known_ab is not None:
        a = float(use_known_ab.get("A_obs", np.nan))
        b = float(use_known_ab.get("B_obs", np.nan))
        y_fit = a * template_u + b
        rms = float(np.sqrt(np.mean((y_fit - dataset["S"]) ** 2)))
    else:
        a, b, y_fit, rms = solve_linear_ab(dataset["S"], template_u)

    return {
        "dataset_name": dataset["name"],
        "fluence_ratio": float(dataset["fluence_ratio"]),
        "n_points": int(len(dataset["t"])),
        "t": np.asarray(dataset["t"], dtype=float),
        "S_obs": np.asarray(dataset["S"], dtype=float),
        "template_u": np.asarray(template_u, dtype=float),
        "S_fit": np.asarray(y_fit, dtype=float),
        "dt_i_ps": float(dt_i_ps),
        "sigma_irf_ps": float(sigma_irf_ps),
        "A_obs": float(a),
        "B_obs": float(b),
        "B_eff_obs": float(b),
        "rms": float(rms),
        "wrms": float("nan"),
    }


def safe_wrms(row_from_json: dict | None, fallback_rms: float, sigma_s: float = 0.02) -> float:
    if row_from_json is not None and "wrms" in row_from_json:
        try:
            return float(row_from_json["wrms"])
        except Exception:
            pass
    return float(fallback_rms / sigma_s)


# ============================================================
# Plotting
# ============================================================
def color_for_index(i: int):
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
    return cycle[i % len(cycle)]


def plot_overview(train_rows: list[dict], showcase_rows: list[dict], outpath: Path) -> None:
    plt.figure(figsize=(12, 7))

    all_rows = train_rows + showcase_rows
    for i, row in enumerate(all_rows):
        c = color_for_index(i)
        flu = row["fluence_ratio"]

        if row in train_rows:
            plt.scatter(
                row["t"] * 1e12,
                row["S_obs"],
                s=22,
                alpha=0.7,
                color=c,
                label=f"exp {row['dataset_name']} ({flu:.1f} mW)",
            )
            plt.plot(
                row["t"] * 1e12,
                row["S_fit"],
                lw=2.0,
                color=c,
                label=f"fit {row['dataset_name']} ({flu:.1f} mW)",
            )
        else:
            plt.scatter(
                row["t"] * 1e12,
                row["S_obs"],
                s=28,
                alpha=0.8,
                facecolors="none",
                edgecolors=c,
                linewidths=1.2,
                label=f"exp {row['dataset_name']} ({flu:.1f} mW)",
            )
            plt.plot(
                row["t"] * 1e12,
                row["S_fit"],
                lw=2.0,
                ls="--",
                color=c,
                label=f"projection {row['dataset_name']} ({flu:.1f} mW)",
            )

    plt.xlabel("time (ps)")
    plt.ylabel("S")
    plt.title("Global fit (training) + high-fluence showcase")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_high_only(showcase_rows: list[dict], outpath: Path) -> None:
    fig, axes = plt.subplots(len(showcase_rows), 1, figsize=(8.5, 9.0), sharex=True)

    if len(showcase_rows) == 1:
        axes = [axes]

    for i, row in enumerate(showcase_rows):
        ax = axes[i]
        c = color_for_index(i + 3)
        ax.scatter(
            row["t"] * 1e12,
            row["S_obs"],
            s=28,
            facecolors="none",
            edgecolors=c,
            linewidths=1.2,
        )
        ax.plot(
            row["t"] * 1e12,
            row["S_fit"],
            ls="--",
            lw=2.0,
            color=c,
        )
        ax.set_ylabel("S")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"{row['dataset_name']} | F={row['fluence_ratio']:.1f} mW | "
            f"N={row['n_points']} | rms={row['rms']:.4e}"
        )

    axes[-1].set_xlabel("time (ps)")
    fig.suptitle("High-fluence showcase (fixed global parameters)", y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_readout_trends(train_rows: list[dict], showcase_rows: list[dict], outpath: Path) -> None:
    rows = sorted(train_rows + showcase_rows, key=lambda r: r["fluence_ratio"])
    train_set = {r["dataset_name"] for r in train_rows}

    flu = np.array([r["fluence_ratio"] for r in rows], dtype=float)
    a = np.array([r["A_obs"] for r in rows], dtype=float)
    b = np.array([r["B_obs"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 7.0), sharex=True)

    # A_obs
    for r in rows:
        is_train = r["dataset_name"] in train_set
        markerface = color_for_index(0) if is_train else "none"
        axes[0].plot(
            r["fluence_ratio"],
            r["A_obs"],
            marker="o",
            markersize=7,
            markerfacecolor=markerface,
            markeredgecolor=color_for_index(0),
            linestyle="None",
        )
    axes[0].plot(flu, a, lw=1.5)
    axes[0].set_ylabel("A_obs")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title("Effective readout trends")

    # B_obs
    for r in rows:
        is_train = r["dataset_name"] in train_set
        markerface = color_for_index(2) if is_train else "none"
        axes[1].plot(
            r["fluence_ratio"],
            r["B_obs"],
            marker="s",
            markersize=6.5,
            markerfacecolor=markerface,
            markeredgecolor=color_for_index(2),
            linestyle="None",
        )
    axes[1].plot(flu, b, lw=1.5)
    axes[1].set_xlabel("fluence ratio / mW label")
    axes[1].set_ylabel("B_obs")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def write_summary_csv(rows: list[dict], outpath: Path) -> None:
    fieldnames = [
        "dataset_name",
        "fluence_ratio",
        "n_points",
        "dt_i_ps",
        "sigma_irf_ps",
        "A_obs",
        "B_obs",
        "B_eff_obs",
        "rms",
        "wrms",
    ]
    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ============================================================
# Main
# ============================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = resolve_mainfit_json()
    payload = load_json(json_path)
    p_global = get_best_global_params(payload)
    observable_mode = get_observable_mode(payload)

    summary_map = dataset_summary_map(payload)
    local_map = best_local_map(payload)

    train_datasets = [load_dataset(DATA_DIR / fn) for fn in TRAIN_CSV_FILES]
    showcase_datasets = [load_dataset(DATA_DIR / fn) for fn in SHOWCASE_CSV_FILES]

    train_rows: list[dict] = []
    for ds in train_datasets:
        known_ab = local_map.get(ds["name"], None)
        row = project_dataset(ds, p_global, observable_mode, use_known_ab=known_ab)
        row["wrms"] = safe_wrms(summary_map.get(ds["name"]), row["rms"])
        train_rows.append(row)

    showcase_rows: list[dict] = []
    for ds in showcase_datasets:
        row = project_dataset(ds, p_global, observable_mode, use_known_ab=None)
        row["wrms"] = safe_wrms(summary_map.get(ds["name"]), row["rms"])
        showcase_rows.append(row)

    # Save figures
    plot_overview(
        train_rows,
        showcase_rows,
        OUTPUT_DIR / f"overview_{OUTPUT_TAG}.png",
    )
    plot_high_only(
        showcase_rows,
        OUTPUT_DIR / f"high_only_{OUTPUT_TAG}.png",
    )
    plot_readout_trends(
        train_rows,
        showcase_rows,
        OUTPUT_DIR / f"readout_trends_{OUTPUT_TAG}.png",
    )
    write_summary_csv(
        showcase_rows,
        OUTPUT_DIR / f"showcase_summary_{OUTPUT_TAG}.csv",
    )

    # Also dump the readout rows to a small JSON for convenience
    out_json = OUTPUT_DIR / f"showcase_summary_{OUTPUT_TAG}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_mainfit_json": str(json_path),
                "observable_mode": observable_mode,
                "train_rows": train_rows,
                "showcase_rows": showcase_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        )

    print(f"[showcase] source mainfit json = {json_path}")
    print(f"[showcase] output dir = {OUTPUT_DIR}")
    print("[showcase] files:")
    print(f"  - {OUTPUT_DIR / f'overview_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'high_only_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'readout_trends_{OUTPUT_TAG}.png'}")
    print(f"  - {OUTPUT_DIR / f'showcase_summary_{OUTPUT_TAG}.csv'}")
    print(f"  - {OUTPUT_DIR / f'showcase_summary_{OUTPUT_TAG}.json'}")


if __name__ == "__main__":
    main()