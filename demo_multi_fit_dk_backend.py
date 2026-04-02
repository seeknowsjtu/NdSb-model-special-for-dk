# -*- coding: utf-8 -*-
"""Minimal backend-only demo for delta-k multi-dataset fitting.

Run:
    python demo_multi_fit_dk_backend.py
"""
from __future__ import annotations

import numpy as np

from config import default_params, normalize_params_dict
from data_io import (
    build_delta_k_template_only,
    build_delta_k_residual,
    export_multi_fit_results,
    fit_params_multi_dk,
)
from physics_engine import DebyeCl


def _make_dataset(base_params: dict, fluence_ratio: float, mode: str, noise: float, seed: int) -> dict:
    t = np.linspace(-1.2, 12.0, 220) * 1e-12
    params = dict(base_params)
    params["fluence_multiplier"] = float(fluence_ratio)
    tmpl = build_delta_k_template_only(params, {"t": t, "fluence_ratio": fluence_ratio}, mode, debye_obj=DebyeCl(thetaD=float(params["ThetaD"])))
    y_clean = np.clip(float(params["K_dk"]) * np.asarray(tmpl["template_u"], dtype=float), 0.0, np.inf)

    rng = np.random.default_rng(seed)
    y_obs = y_clean + rng.normal(0.0, noise, size=t.shape)
    is_resolved = np.isfinite(y_obs) & (y_obs > float(params["dk_resolution_limit"]))

    return {
        "name": f"demo_dk_{fluence_ratio:.1f}mW.csv",
        "path": f"demo/dk_{fluence_ratio:.1f}mW.csv",
        "t": t,
        "delta_k": y_obs,
        "S": y_obs,
        "is_resolved": is_resolved,
        "Te": None,
        "Ts": None,
        "Tl": None,
        "fluence_ratio": float(fluence_ratio),
    }


def _run_mode(mode: str, datasets: list[dict], p0: dict) -> None:
    fit_bundle, res = fit_params_multi_dk(
        datasets,
        p0,
        global_keys=["S_scale", "G_es0", "G_el0", "G_sl0", "dt0_ps", "sigma_irf_ps", "K_dk"],
        local_keys=[],
        observable_mode=mode,
        sigma_dk=float(p0.get("sigma_dk", 0.002)),
        sigma_dk_censored=float(p0.get("sigma_dk_censored", 0.002)),
        dk_resolution_limit=float(p0.get("dk_resolution_limit", 0.003)),
        max_nfev=60,
        progress_every=5,
        optimizer_verbose=0,
        enable_timing=True,
    )
    exports = export_multi_fit_results(fit_bundle, res, export_root=f"fit_results/demo_multi_dk_{mode}")

    print(f"mode={mode} success={res.success} status={res.status} cost={res.cost:.6e} nfev={res.nfev}")
    print(f"  K_dk={fit_bundle['best_global_params'].get('K_dk', float('nan')):.6g}")
    for row in fit_bundle["dataset_summary"]:
        print(
            f"  dataset={row['dataset_name']} wrms={row['wrms']:.4e} "
            f"resolved={row['n_resolved']} unresolved={row['n_unresolved']}"
        )

    # quick explicit residual demonstration for unresolved points
    first = fit_bundle["dataset_fits"][0]
    demo_res = build_delta_k_residual(
        first["delta_k_exp"],
        first["delta_k_fit"],
        first["is_resolved"],
        sigma_resolved=float(p0.get("sigma_dk", 0.002)),
        sigma_censored=float(p0.get("sigma_dk_censored", 0.002)),
        resolution_limit=float(p0.get("dk_resolution_limit", 0.003)),
    )
    print(f"  residual_check_norm={np.linalg.norm(demo_res):.4e}")
    print(f"  export_dir={exports['export_dir']}")


def main() -> None:
    p_true = normalize_params_dict(default_params())
    p_true["eta_representation"] = "cos2phi"
    p_true["K_dk"] = 0.028
    p_true["dk_resolution_limit"] = 0.003
    p_true["sigma_dk"] = 0.002
    p_true["sigma_dk_censored"] = 0.002

    datasets = [
        _make_dataset(p_true, fluence_ratio=1.0, mode="dk_chi2q", noise=0.0010, seed=1),
        _make_dataset(p_true, fluence_ratio=2.5, mode="dk_chi2q", noise=0.0012, seed=2),
        _make_dataset(p_true, fluence_ratio=4.0, mode="dk_chi2q", noise=0.0014, seed=3),
    ]

    p0 = dict(p_true)
    p0["K_dk"] *= 0.8
    p0["G_el0"] *= 1.05
    p0["G_es0"] *= 0.95

    _run_mode("dk_chi2q", datasets, p0)
    _run_mode("dk_m_chi2q", datasets, p0)


if __name__ == "__main__":
    main()
