# -*- coding: utf-8 -*-
"""Minimal backend-only demo for current varpro multi-dataset mainline.

This demo validates global dt0_ps + sigma_irf_ps fitting/export using
observable_mode="raw_m_chi2q" with varpro readout.

Run:
    python demo_multi_fit_backend.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import MULTI_FIT_DEFAULT_GLOBAL_KEYS, default_params, normalize_params_dict
from data_io import build_observable, export_multi_fit_results, fit_params_multi
from physics_engine import DebyeCl
from solver import NdSb3TM


def _make_dataset(base_params: dict, fluence_ratio: float, dt_trace_ps: float, noise: float, seed: int) -> dict:
    t = np.linspace(-1.5, 12.0, 220) * 1e-12
    params = dict(base_params)
    params["fluence_multiplier"] = float(fluence_ratio)

    model = NdSb3TM(params, debye_obj=DebyeCl(thetaD=float(params["ThetaD"])))
    sim = model.simulate_aligned(t - float(dt_trace_ps) * 1e-12, with_diag=False)
    rng = np.random.default_rng(seed)
    S = build_observable(sim, params, {}, "raw_m_chi2q")
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
    p_true["dt0_ps"] = 0.08
    p_true["sigma_irf_ps"] = 0.20
    p_true["A_obs"] = 0.04
    p_true["B_obs"] = 0.02

    datasets = [
        _make_dataset(p_true, fluence_ratio=1.0, dt_trace_ps=0.10, noise=0.003, seed=1),
        _make_dataset(p_true, fluence_ratio=2.5, dt_trace_ps=-0.08, noise=0.003, seed=2),
        _make_dataset(p_true, fluence_ratio=4.0, dt_trace_ps=0.04, noise=0.003, seed=3),
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
        global_keys=list(MULTI_FIT_DEFAULT_GLOBAL_KEYS),
        local_keys=[],
        observable_mode="raw_m_chi2q",
        sigma_S=0.01,
        max_nfev=80,
        progress_every=5,
        optimizer_verbose=2,
        enable_timing=True,
    )
    exports = export_multi_fit_results(fit_bundle, res, export_root="fit_results/demo_multi")

    print(f"success={res.success} status={res.status} cost={res.cost:.6e} nfev={res.nfev}")
    print(f"observable_mode={fit_bundle['observable_mode']} local_keys={fit_bundle['local_keys']}")
    for row in fit_bundle["dataset_summary"]:
        print(
            f"dataset={row['dataset_name']} fluence_ratio={row['fluence_ratio']:.2f} "
            f"dt_i_ps={row['dt_i_ps']:.4f} A_obs={row['A_obs']:.4e} "
            f"B_obs={row['B_obs']:.4e} B_eff_obs={row['B_eff_obs']:.4e} "
            f"rms={row['rms']:.4e} wrms={row['wrms']:.4e}"
        )
    print("exports:")
    for key, value in exports.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
