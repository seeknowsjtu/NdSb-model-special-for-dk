# -*- coding: utf-8 -*-
"""Minimal backend-only demo for multi-dataset fitting and export.

Run:
    python demo_multi_fit_backend.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import default_params, normalize_params_dict
from data_io import build_observable, export_multi_fit_results, fit_params_multi
from physics_engine import DebyeCl
from solver import NdSb3TM


def _make_dataset(base_params: dict, fluence_ratio: float, dt_local: float, noise: float, seed: int) -> dict:
    t = np.linspace(-1.5, 12.0, 220) * 1e-12
    params = dict(base_params)
    params["fluence_multiplier"] = float(fluence_ratio)

    model = NdSb3TM(params, debye_obj=DebyeCl(thetaD=float(params["ThetaD"])))
    sim = model.simulate_aligned(t - dt_local, with_diag=False)
    rng = np.random.default_rng(seed)
    S = build_observable(sim, params, {"dt_local": dt_local, "A_obs": 1.0, "B_obs": 0.0}, "eta")
    S = np.asarray(S, dtype=float) + rng.normal(0.0, noise, size=t.shape)

    return {
        "name": f"demo_{fluence_ratio:.1f}mW.csv",
        "path": str(Path("demo") / f"demo_{fluence_ratio:.1f}mW.csv"),
        "t": t,
        "S": S,
        "Te": None,
        "Ts": None,
        "Tl": None,
        "fluence_ratio": float(fluence_ratio),
    }


def main() -> None:
    p_true = normalize_params_dict(default_params())
    datasets = [
        _make_dataset(p_true, fluence_ratio=1.0, dt_local=0.10e-12, noise=0.003, seed=1),
        _make_dataset(p_true, fluence_ratio=2.5, dt_local=-0.08e-12, noise=0.003, seed=2),
        _make_dataset(p_true, fluence_ratio=4.0, dt_local=0.04e-12, noise=0.003, seed=3),
    ]

    p0 = dict(p_true)
    p0["G_el0"] *= 1.08
    p0["G_es0"] *= 0.92
    p0["tau_m0"] *= 1.10
    p0["Gamma_eta"] *= 0.90
    p0["S_scale"] *= 1.05

    fit_bundle, res = fit_params_multi(
        datasets,
        p0,
        local_keys=["dt_local"],
        observable_mode="eta",
        sigma_S=0.01,
        max_nfev=80,
    )
    exports = export_multi_fit_results(fit_bundle, res, export_root="fit_results/demo_multi")

    print(f"success={res.success} status={res.status} cost={res.cost:.6e} nfev={res.nfev}")
    print(f"observable_mode={fit_bundle['observable_mode']} local_keys={fit_bundle['local_keys']}")
    for row in fit_bundle["dataset_summary"]:
        print(
            f"dataset={row['dataset_name']} fluence_ratio={row['fluence_ratio']:.2f} "
            f"dt_local_ps={row['dt_local_ps']:.4f} rms={row['rms']:.4e}"
        )
    print("exports:")
    for key, value in exports.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
