from __future__ import annotations
from pathlib import Path
import time
import traceback
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_KIND = "delta_k"  # "S" or "delta_k"
EXPERIMENT_MODE = "dk_rel_m_chi2q"
# S modes: "raw_eta", "raw_chi2q", "raw_m_chi2q"
# delta-k modes: "dk_chi2q", "dk_affine_chi2q", "dk_m_chi2q", "dk_affine_m_chi2q", "dk_rel_chi2q", "dk_rel_m_chi2q"

from config import (
    default_params,
    normalize_params_dict,
    MULTI_FIT_DEFAULT_GLOBAL_KEYS,
    MULTI_FIT_DEFAULT_LOCAL_KEYS,
    MULTI_FIT_DEFAULT_OBSERVABLE_MODE,
)
from data_io import (
    build_delta_k_residual,
    build_delta_k_template_only,
    fit_params_multi,
    fit_params_multi_dk,
    export_multi_fit_results,
    _get_bounds_for_keys,
    build_observable,
    _build_time_weights_from_t,
    load_dk_dataset_csv,
    load_s_dataset_csv_raw,
)
from physics_engine import DebyeCl
from solver import NdSb3TM

# =========================
# 1. 数据文件列表
# =========================
CSV_FILES = [
    "mdc data/deltak12k_1p0mW.csv",
    "mdc data/deltak12k_2p0mW.csv",
    "mdc data/deltak12k_2p5mW.csv",
    "mdc data/deltak12k_3p0mW.csv",
    "mdc data/deltak12k_3p5mW.csv",
    "mdc data/deltak12k_4p0mW.csv",
    # "spectral data/S_0p5mW.csv",
    # # "spectral data/S_1p2mW.csv",
    # "spectral data/S_2p0mW.csv",
    # # "spectral data/S_2p7mW.csv",
    # # "spectral data/S_3p5mW.csv",    
    # # "spectral data/S_4p2mW.csv",
    # # "spectral data/S_5p0mW.csv",
    # # "spectral data/S_6p5mW.csv",
    # # "spectral data/S_7p2mW.csv",
    # "spectral data/S_8p0mW.csv",
]

DATA_DIR = Path(".")

# =========================
# 2. 运行控制
# =========================
RUN_MODE = "fit"               # "fit" or "scan"
SCAN_REOPTIMIZE_READOUT = False
SCAN_EXPORT_PLOTS = True
SCAN_EXPORT_ROOT = "fit_results/scan_runs"

HEARTBEAT_SEC = 10

SMOKE_TEST = True              # 扫描时建议显式关掉
STOP_AFTER_SMOKE = True       # True: smoke 后退出；False: smoke 后继续 full fit
SMOKE_MAX_NFEV = 80             # 扫描模式下基本不会用到
FULL_MAX_NFEV = 150             # 扫描模式下基本不会用到

# multi-fit 相关设置
SIGMA_S = 0.02
SIGMA_DK = 0.002
SIGMA_DK_CENSORED = 0.002
DK_RESOLUTION_LIMIT = 0.017
PROGRESS_EVERY = 5
OPTIMIZER_VERBOSE = 2
ENABLE_TIMING = True

# Full-fit nonlinear global parameters.
# We fix dt0_ps and sigma_irf_ps because the time zero / IRF are experimentally calibrated.
# K_dk and B_dk do not need to be listed here: fit_params_multi_dk() adds them automatically
# for dk_affine_* modes.
FULL_FIT_GLOBAL_KEYS = [
    k for k in MULTI_FIT_DEFAULT_GLOBAL_KEYS
    if k not in {"dt0_ps", "sigma_irf_ps"}
] + [
    "tau_m0",
    "tau_m_crit_amp",
]

FULL_FIT_LOCAL_KEYS = []

# Scan reoptimize 只重调 readout 子集；命名上明确区别于 full fit。
SCAN_REOPT_GLOBAL_KEYS: list[str] = []

# Scan/readout compatibility knob: 在 USE_VARPRO_READOUT 主线下，A/B 由 varpro 线性读出；
# 这里不是 full-fit nonlinear local keys，只用于扫描阶段的兼容性重优化入口。
SCAN_REOPT_LOCAL_KEYS = ["A_obs", "B_obs"]
ROUND1_GLOBAL_BOUND_WARNING_KEYS = list(FULL_FIT_GLOBAL_KEYS)

# Legacy/manual initialization compatibility:
# 保留 A_obs / B0_obs / B1_obs / pulse_width 作为基线覆盖入口，
# 但它们不是当前 varpro full-fit 默认 nonlinear keys。
BASELINE_OVERRIDE = {
    "S_scale": 0.0611,
    "A_obs": 0.0425,
    "B0_obs": 0.0232,
    "B1_obs": 0.0,
    "pulse_width": 1.5e-13,
    "G_es0": 5.649878739136659e14,
    "G_el0": 3.0e14,
    "G_sl0": 3.04e14,
    "tau_e_sink": 2e-10,
    "tau_s_sink": 5e-09,
    "tau_l_sink": 1.5e-10,
}

SCAN_SPECS = {
    "K_dk": [0.072, 0.076, 0.080, 0.084, 0.088, 0.092],
}

# =========================
# 3. 读入一个数据集
# =========================
def load_dataset(path: Path) -> dict:
    if TARGET_KIND == "S":
        return load_s_dataset_csv_raw(path)
    if TARGET_KIND == "delta_k":
        return load_dk_dataset_csv(path)
    raise ValueError(f"Unsupported TARGET_KIND: {TARGET_KIND}")

# =========================
# 4. 打印数据摘要
# =========================
def print_dataset_summary(datasets: list[dict]) -> None:
    print("Loaded datasets:")
    total_points = 0
    for ds in datasets:
        n = len(ds["t"])
        total_points += n
        print(
            f"  {ds['name']:>22s} | "
            f"N={n:3d} | "
            f"fluence_ratio={ds['fluence_ratio']:.2f} | "
            f"baseline={ds.get('baseline_value', 0.0):.4e} "
            f"(n={ds.get('baseline_npts', 0)}, {ds.get('baseline_method', 'n/a')})"
        )
        if TARGET_KIND == "delta_k":
            n_resolved = int(np.count_nonzero(np.asarray(ds.get("is_resolved", []), dtype=bool)))
            print(f"    resolved={n_resolved} | unresolved={n - n_resolved}")
    print(f"Total points across all datasets = {total_points}")


def configure_mode(p0: dict) -> tuple[dict, str]:
    mode = EXPERIMENT_MODE.strip().lower()
    p = dict(p0)
    if TARGET_KIND == "S":
        if mode == "raw_eta":
            p["eta_representation"] = "scalar"
            observable_mode = "raw_eta"
        elif mode in {"raw_chi2q", "raw_m_chi2q"}:
            p["eta_representation"] = "cos2phi"
            observable_mode = mode
        else:
            raise ValueError(f"Unsupported S EXPERIMENT_MODE: {EXPERIMENT_MODE}")
    elif TARGET_KIND == "delta_k":
        if mode not in {"dk_chi2q", "dk_affine_chi2q", "dk_m_chi2q", "dk_affine_m_chi2q", "dk_rel_chi2q", "dk_rel_m_chi2q"}:
            raise ValueError(f"Unsupported delta_k EXPERIMENT_MODE: {EXPERIMENT_MODE}")
        p["eta_representation"] = "cos2phi"
        observable_mode = mode
    else:
        raise ValueError(f"Unsupported TARGET_KIND: {TARGET_KIND}")

    return p, observable_mode
# =========================
# 5. 构造起始参数
# =========================
def make_initial_params() -> dict:
    p0 = normalize_params_dict(default_params())

    # common
    p0["use_hot_steady_init"] = 1
    p0["hot_init_mode"] = "avg_power"
    p0["rep_rate_Hz"] = 5.0e5
    p0["preheat_max_dT"] = 30.0
    p0["pulse_width"] = 1.5e-13
    p0["USE_VARPRO_READOUT"] = True
    p0["target_kind"] = TARGET_KIND

    if TARGET_KIND == "delta_k":
        p0["dk_mode"] = EXPERIMENT_MODE

        # ---- seed from previous 1.0-4.0 mW global fit ----
        # These are only initial guesses. Parameters listed in FULL_FIT_GLOBAL_KEYS
        # will still be optimized.
        p0["S_scale"] = 0.0507204193329889
        p0["G_es0"] = 4.631928715729768e15
        p0["G_el0"] = 1.4525590752943357e13
        p0["G_sl0"] = 1.6286283118989362e14
        p0["tau_l_sink"] = 3.679158176527993e-11
        p0["tau_s_sink"] = 8.63801703975068e-10

        # delta-k readout initial values.
        # fit_params_multi_dk() will automatically include K_dk,
        # and for dk_affine_m_chi2q it will also include B_dk.
        p0["K_dk"] = 0.10
        p0["B_dk"] = 0.0

        # fixed timing / resolution parameters
        # Use your calibrated value here.
        p0["dt0_ps"] = 0.0
        p0["sigma_irf_ps"] = 0.20

        # magnetic order-parameter dynamics.
        # These two are now floated by FULL_FIT_GLOBAL_KEYS.
        p0["tau_m0"] = 25e-12
        p0["tau_m_crit_amp"] = 120e-12

        # keep these for compatibility; they are not the active dk readout here
        p0["A_obs"] = 0.04
        p0["B_obs"] = 0.02
        p0["B0_obs"] = 0.02
        p0["B1_obs"] = 0.0

        # fixed background sink not used in FULL_FIT_GLOBAL_KEYS here
        p0["tau_e_sink"] = 3.0e-12
    else:
        # keep your old S-mode initialization
        p0["G_sl0"] = 3.04e14
        p0["tau_m0"] = 3.0e-11
        p0["tau_m_crit_amp"] = 0.0
        p0["lam_m2"] = 0.0
        p0["S_scale"] = 0.0611
        p0["A_obs"] = 0.05
        p0["B_obs"] = 0.30
        p0["B0_obs"] = 0.30
        p0["G_es0"] = 5.65e14
        p0["G_el0"] = 3.0e14
        p0["tau_l_sink"] = 1.5e-10
        p0["B1_obs"] = 0.0

    return p0

def make_baseline_params() -> dict:
    p0 = make_initial_params()
    p0.update(BASELINE_OVERRIDE)
    return p0


def infer_observable_scale_from_datasets(datasets: list[dict]) -> tuple[float, float]:
    all_s = np.concatenate([np.asarray(ds["S"], dtype=float) for ds in datasets])
    global_min = float(np.min(all_s))
    global_max = float(np.max(all_s))
    b_obs0 = max(0.0, global_min)
    a_obs0 = max(global_max - b_obs0, 1e-4)
    return a_obs0, b_obs0


def evaluate_fixed_model(
    datasets: list[dict],
    p0: dict,
    observable_mode: str,
    sigma_S: float,
) -> tuple[float, float, list[dict], list[dict]]:
    sigma_S = float(max(sigma_S, 1e-12))
    debye_obj = DebyeCl(thetaD=float(p0["ThetaD"]))
    rows = []
    plot_payloads = []
    all_sq = 0.0
    all_count = 0

    for dataset in datasets:
        p_work = dict(p0)
        p_work["fluence_multiplier"] = float(dataset["fluence_ratio"])
        if TARGET_KIND == "S":
            model = NdSb3TM(p_work, debye_obj=debye_obj)
            sim = model.simulate_aligned(np.asarray(dataset["t"], dtype=float), with_diag=False)
            y_fit = build_observable(sim, p_work, {}, observable_mode)
            y_obs = np.asarray(dataset["S"], dtype=float)
            point_weights = _build_time_weights_from_t(dataset["t"])
            residual = ((y_fit - y_obs) / sigma_S) * np.sqrt(point_weights)
            rms = float(np.sqrt(np.mean((y_fit - y_obs) ** 2)))
        else:
            p_work["sigma_dk"] = SIGMA_DK
            p_work["sigma_dk_censored"] = SIGMA_DK_CENSORED
            p_work["dk_resolution_limit"] = DK_RESOLUTION_LIMIT
            templ = build_delta_k_template_only(
                p_work, dataset, observable_mode, debye_obj=debye_obj, with_diag=False
            )
            K_dk = float(p_work.get("K_dk", 0.02))
            affine_mode = observable_mode in {"dk_affine_chi2q", "dk_affine_m_chi2q"}
            B_dk = float(p_work.get("B_dk", 0.0)) if affine_mode else 0.0
            y_fit = np.clip(B_dk + K_dk * np.asarray(templ["template_u"], dtype=float), 0.0, np.inf)
            y_obs = np.asarray(dataset["delta_k"], dtype=float)
            residual = build_delta_k_residual(
                y_obs=y_obs,
                y_model=y_fit,
                is_resolved=np.asarray(dataset["is_resolved"], dtype=bool),
                sigma_resolved=SIGMA_DK,
                sigma_censored=SIGMA_DK_CENSORED,
                resolution_limit=DK_RESOLUTION_LIMIT,
            )
            resolved_mask = np.asarray(dataset["is_resolved"], dtype=bool)
            rms = float(np.sqrt(np.mean((y_fit[resolved_mask] - y_obs[resolved_mask]) ** 2))) if np.any(resolved_mask) else float("nan")
            sim = templ["sim"]
        wrms = float(np.sqrt(np.mean(residual ** 2)))

        rows.append(
            {
                "dataset_name": dataset["name"],
                "fluence_ratio": float(dataset["fluence_ratio"]),
                "rms": rms,
                "wrms": wrms,
            }
        )
        plot_payloads.append(
            {
                "dataset_name": dataset["name"],
                "t": np.asarray(dataset["t"], dtype=float),
                "S_obs": y_obs,
                "S_fit": np.asarray(y_fit, dtype=float),
            }
        )
        all_sq += float(np.sum(residual ** 2))
        all_count += int(residual.size)

    if all_count <= 0:
        raise ValueError("No points found during evaluate_fixed_model.")

    total_cost = 0.5 * all_sq
    total_wrms = float(np.sqrt(all_sq / all_count))
    return total_cost, total_wrms, rows, plot_payloads


def run_scan_suite(datasets: list[dict], p0: dict):
    p_mode, observable_mode = configure_mode(p0)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    export_dir = Path(SCAN_EXPORT_ROOT)
    export_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = export_dir / f"scan_summary_{timestamp}.csv"
    summary_rows = []

    for scan_param, values in SCAN_SPECS.items():
        x_vals = []
        y_vals = []
        for scan_value in values:
            t0 = time.perf_counter()
            p_scan = dict(p_mode)
            p_scan[scan_param] = float(scan_value)

            if not SCAN_REOPTIMIZE_READOUT:
                total_cost, total_wrms, dataset_rows, _ = evaluate_fixed_model(
                    datasets=datasets,
                    p0=p_scan,
                    observable_mode=observable_mode,
                    sigma_S=SIGMA_S if TARGET_KIND == "S" else SIGMA_DK,
                )
                wrms_map = {float(r["fluence_ratio"]): float(r["wrms"]) for r in dataset_rows}
                row = {
                    "scan_param": scan_param,
                    "scan_value": float(scan_value),
                    "mode": observable_mode,
                    "reoptimize_readout": False,
                    "total_cost": float(total_cost),
                    "total_wrms": float(total_wrms),
                    "wrms_1p0": wrms_map.get(1.0, float("nan")),
                    "wrms_2p0": wrms_map.get(2.0, float("nan")),
                    "wrms_2p5": wrms_map.get(2.5, float("nan")),
                    "S_scale": p_scan.get("S_scale", float("nan")),
                    "A_obs": p_scan.get("A_obs", float("nan")),
                    "B_obs": p_scan.get("B_obs", float("nan")),
                    "B0_obs": p_scan.get("B0_obs", float("nan")),
                }
            else:
                if TARGET_KIND == "S":
                    fit_bundle, res = fit_params_multi(
                        datasets,
                        p_scan,
                        global_keys=SCAN_REOPT_GLOBAL_KEYS,
                        local_keys=SCAN_REOPT_LOCAL_KEYS,
                        observable_mode=observable_mode,
                        sigma_S=SIGMA_S,
                        max_nfev=60,
                        progress_every=PROGRESS_EVERY,
                        optimizer_verbose=OPTIMIZER_VERBOSE,
                        enable_timing=ENABLE_TIMING,
                    )
                else:
                    fit_bundle, res = fit_params_multi_dk(
                        datasets,
                        p_scan,
                        global_keys=[k for k in FULL_FIT_GLOBAL_KEYS if k != "A_obs"] + ["K_dk"],
                        local_keys=[],
                        observable_mode=observable_mode,
                        sigma_dk=SIGMA_DK,
                        sigma_dk_censored=SIGMA_DK_CENSORED,
                        dk_resolution_limit=DK_RESOLUTION_LIMIT,
                        max_nfev=60,
                        progress_every=PROGRESS_EVERY,
                        optimizer_verbose=OPTIMIZER_VERBOSE,
                        enable_timing=ENABLE_TIMING,
                    )
                wrms_map = {
                    float(r["fluence_ratio"]): float(r["wrms"])
                    for r in fit_bundle["dataset_summary"]
                }
                row = {
                    "scan_param": scan_param,
                    "scan_value": float(scan_value),
                    "mode": observable_mode,
                    "reoptimize_readout": True,
                    "total_cost": float(res.cost),
                    "total_wrms": float(np.sqrt(2.0 * res.cost / sum(len(d["t"]) for d in datasets))),
                    "wrms_1p0": wrms_map.get(1.0, float("nan")),
                    "wrms_2p0": wrms_map.get(2.0, float("nan")),
                    "wrms_2p5": wrms_map.get(2.5, float("nan")),
                    "S_scale": float(fit_bundle["best_global_params"].get("S_scale", float("nan"))),
                    "A_obs": float(np.nanmean([float(r.get("A_obs", np.nan)) for r in fit_bundle["dataset_summary"]])),
                    "B_obs": float(np.nanmean([float(r.get("B_obs", np.nan)) for r in fit_bundle["dataset_summary"]])),
                    "B0_obs": float(fit_bundle["best_global_params"].get("B0_obs", float("nan"))),
                }

            dt = time.perf_counter() - t0
            summary_rows.append(row)
            x_vals.append(float(scan_value))
            y_vals.append(float(row["total_wrms"]))
            print(
                f"[scan] {scan_param:>10s} = {scan_value:.6e} | "
                f"total_wrms = {row['total_wrms']:.6e} | "
                f"cost = {row['total_cost']:.6e} | "
                f"dt = {dt:.2f}s",
                flush=True,
            )

        if SCAN_EXPORT_PLOTS:
            plt.figure(figsize=(6, 4))
            plt.plot(x_vals, y_vals, marker="o")
            plt.xscale("log")
            plt.xlabel(scan_param)
            plt.ylabel("total_wrms")
            plt.title(f"{scan_param} scan ({observable_mode})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = export_dir / f"scan_wrms_{scan_param}_{timestamp}.png"
            plt.savefig(fig_path, dpi=160)
            plt.close()

    fieldnames = [
        "scan_param",
        "scan_value",
        "mode",
        "reoptimize_readout",
        "total_cost",
        "total_wrms",
        "wrms_1p0",
        "wrms_2p0",
        "wrms_2p5",
        "S_scale",
        "A_obs",
        "B_obs",
        "B0_obs",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[scan] summary rows = {len(summary_rows)}")
    print(f"[scan] csv = {summary_csv}")
    print(f"[scan] plot_dir = {export_dir}")
    return summary_rows, summary_csv

# =========================
# 6. 单次拟合任务
# =========================
def run_fit(datasets: list[dict], p0: dict, max_nfev: int, export_root: str):
    p_mode, observable_mode = configure_mode(p0)
    if TARGET_KIND == "S":
        fit_bundle, res = fit_params_multi(
            datasets,
            p_mode,
            global_keys=FULL_FIT_GLOBAL_KEYS,
            local_keys=FULL_FIT_LOCAL_KEYS,
            observable_mode=observable_mode,
            sigma_S=SIGMA_S,
            max_nfev=max_nfev,
            progress_every=PROGRESS_EVERY,
            optimizer_verbose=OPTIMIZER_VERBOSE,
            enable_timing=ENABLE_TIMING,
        )
    else:
        fit_bundle, res = fit_params_multi_dk(
            datasets,
            p_mode,
            global_keys=[k for k in FULL_FIT_GLOBAL_KEYS if k != "A_obs"] + ["K_dk"],
            local_keys=[],
            observable_mode=observable_mode,
            sigma_dk=SIGMA_DK,
            sigma_dk_censored=SIGMA_DK_CENSORED,
            dk_resolution_limit=DK_RESOLUTION_LIMIT,
            max_nfev=max_nfev,
            progress_every=PROGRESS_EVERY,
            optimizer_verbose=OPTIMIZER_VERBOSE,
            enable_timing=ENABLE_TIMING,
        )

    return fit_bundle, res, export_root
# =========================
# 7. 带心跳的执行器
# =========================
def run_with_heartbeat(datasets: list[dict], p0: dict, max_nfev: int, export_root: str):
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(run_fit, datasets, p0, max_nfev, export_root)

        while not future.done():
            elapsed = time.perf_counter() - start
            print(
                f"[multi-fit] still running... "
                f"elapsed = {elapsed:.1f} s | "
                f"max_nfev = {max_nfev}",
                flush=True,
            )
            time.sleep(HEARTBEAT_SEC)

        end = time.perf_counter()

        # 若拟合线程里抛异常，这里会再次抛出
        fit_bundle, res, export_root = future.result()

    print(f"[multi-fit] finished in {end - start:.1f} s", flush=True)
    try:
        exports = export_multi_fit_results(
            fit_bundle,
            res,
            export_root=export_root,
        )
    except Exception as exc:
        warning_msg = f"export failed: {exc}"
        print(f"[warning] {warning_msg}", flush=True)
        exports = {"export_error": str(exc)}
    return fit_bundle, res, exports, end - start

# =========================
# 8. 打印结果
# =========================
def _print_bound_warnings(fit_bundle) -> None:
    warning_lines = []

    for row in fit_bundle["dataset_summary"]:
        dt_i_ps = float(row.get("dt_i_ps", float("nan")))
        if abs(dt_i_ps) >= 0.95:
            warning_lines.append(
                f"[warning] dt_i near bound: {row['dataset_name']} dt_i_ps={dt_i_ps:.3f}"
            )

    global_lb, global_ub = _get_bounds_for_keys(ROUND1_GLOBAL_BOUND_WARNING_KEYS)
    best_global = fit_bundle["best_global_params"]
    for key, lb, ub in zip(ROUND1_GLOBAL_BOUND_WARNING_KEYS, global_lb, global_ub):
        value = float(best_global.get(key, float("nan")))
        if not (lb < ub):
            continue
        margin = 0.05 * (ub - lb)
        if value <= lb + margin or value >= ub - margin:
            warning_lines.append(
                f"[warning] global param near bound: {key}={value:.8g} within 5% of [{lb:.8g}, {ub:.8g}]"
            )

    if warning_lines:
        print("\n=== bound warnings ===")
        for line in warning_lines:
            print(line)


def print_fit_summary(fit_bundle, res, exports, wall_time_sec: float) -> None:
    print("\n=== optimizer summary ===")
    print(f"success = {res.success}")
    print(f"status  = {res.status}")
    print(f"cost    = {res.cost:.6e}")
    print(f"nfev    = {res.nfev}")
    print(f"target_kind = {fit_bundle.get('target_kind', TARGET_KIND)}")
    print(f"mode    = {fit_bundle['observable_mode']}")
    print(f"varpro  = {fit_bundle.get('use_varpro_readout')}")
    print(f"locals  = {fit_bundle['local_keys']}")
    print(f"wall_time_sec = {wall_time_sec:.1f}")

    print("\n=== fitted global params ===")
    for k in fit_bundle["global_keys"]:
        v = fit_bundle["best_global_params"].get(k, float("nan"))
        print(f"  {k:20s} = {v:.8g}")
    if fit_bundle.get("target_kind") == "delta_k":
        print(f"  {'K_dk':20s} = {fit_bundle['best_global_params'].get('K_dk', float('nan')):.8g}")
        if "B_dk" in fit_bundle["best_global_params"]:
            print(f"  {'B_dk':20s} = {fit_bundle['best_global_params'].get('B_dk', float('nan')):.8g}")

    print("\n=== dataset summary ===")
    print("  dataset | fluence | rms | wrms | dt_i_ps | sigma_irf_ps")
    for row in fit_bundle["dataset_summary"]:
        dt_i_ps = row.get("dt_i_ps", float("nan"))
        line = (
            f"  {row['dataset_name']:>22s} | "
            f"fluence={row['fluence_ratio']:.2f} | "
            f"rms={row['rms']:.4e} | "
            f"wrms={row['wrms']:.4e} | "
            f"dt_i_ps={dt_i_ps:.4f} | "
            f"sigma_irf_ps={float(row.get('sigma_irf_ps', float('nan'))):.6g}"
        )
        if fit_bundle.get("target_kind") == "delta_k":
            line += (
                f" | n_resolved={int(row.get('n_resolved', 0))}"
                f" | n_unresolved={int(row.get('n_unresolved', 0))}"
            )
        else:
            line += (
                f" | A_obs={float(row.get('A_obs', float('nan'))):.6g}"
                f" | B_obs={float(row.get('B_obs', float('nan'))):.6g}"
                f" | B_eff_obs={float(row.get('B_eff_obs', float('nan'))):.6g}"
            )
        print(line)

    _print_bound_warnings(fit_bundle)

    timing = fit_bundle.get("timing_summary", {})
    if timing:
        print("\n=== timing summary ===")
        print(f"residual_call_count = {timing.get('residual_call_count')}")
        print(f"elapsed_sec         = {timing.get('elapsed_sec')}")
        print(f"avg_residual_sec    = {timing.get('avg_residual_sec')}")
        print(f"estimated_total_calls = {timing.get('estimated_total_calls')}")
        print(f"rough_eta_sec       = {timing.get('rough_eta_sec')}")

        per_dataset = timing.get("per_dataset", {})
        if per_dataset:
            print("\n=== per-dataset timing ===")
            for name, info in per_dataset.items():
                print(
                    f"  {name:>22s} | "
                    f"call_count={info.get('call_count')} | "
                    f"avg_wall_time_sec={info.get('avg_wall_time_sec'):.4f}"
                )

    print("\n=== exports ===")
    for k, v in exports.items():
        print(f"  {k}: {v}")

# =========================
# 9. 主程序
# =========================
def main() -> None:
    try:
        missing_files = [str(DATA_DIR / fn) for fn in CSV_FILES if not (DATA_DIR / fn).exists()]
        if missing_files:
            raise FileNotFoundError(
                "Missing CSV files in CSV_FILES:\n  - " + "\n  - ".join(missing_files)
            )
        datasets = [load_dataset(DATA_DIR / fn) for fn in CSV_FILES]
        datasets.sort(key=lambda d: d["fluence_ratio"])
        print_dataset_summary(datasets)

        p0 = make_initial_params()
        p0, observable_mode_preview = configure_mode(p0)
        print(
            f"[mode] TARGET_KIND = {TARGET_KIND} | EXPERIMENT_MODE = {EXPERIMENT_MODE} | "
            f"default_observable = {MULTI_FIT_DEFAULT_OBSERVABLE_MODE} | "
            f"observable_mode = {observable_mode_preview} | "
            f"eta_representation = {p0['eta_representation']}"
        )
        # a_obs0, b_obs0 = infer_observable_scale_from_datasets(datasets)
        # p0["A_obs"] = a_obs0
        # p0["B0_obs"] = b_obs0

        mode = RUN_MODE.strip().lower()
        if mode == "scan":
            print("\n===== SCAN RUN START =====")
            run_scan_suite(datasets, p0)
            print("===== SCAN RUN END =====")
            return
        if mode != "fit":
            raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")

        # ---- 第一步：smoke test ----
        if SMOKE_TEST:
            print("\n===== SMOKE TEST START =====")
            smoke_bundle, smoke_res, smoke_exports, smoke_t = run_with_heartbeat(
                datasets,
                p0,
                max_nfev=SMOKE_MAX_NFEV,
                export_root="fit_results/real_multi_fit_smoke",
            )
            print_fit_summary(smoke_bundle, smoke_res, smoke_exports, smoke_t)
            print("===== SMOKE TEST END =====\n")
            # SMOKE_TEST=True 表示“先跑 smoke”；是否在此退出由 STOP_AFTER_SMOKE 决定。
            if STOP_AFTER_SMOKE:
                return

            # 用 smoke test 的最优全局参数作为正式跑的起点
            p0 = dict(p0)
            p0.update(smoke_bundle["best_global_params"])

        # ---- 第二步：正式第一轮 ----
        print("\n===== FULL FIT START =====")
        fit_bundle, res, exports, wall_time_sec = run_with_heartbeat(
            datasets,
            p0,
            max_nfev=FULL_MAX_NFEV,
            export_root="fit_results/real_multi_fit_round1",
        )
        print_fit_summary(fit_bundle, res, exports, wall_time_sec)
        print("===== FULL FIT END =====")

    except Exception as e:
        print("\n[ERROR] multi-fit failed:")
        print(str(e))
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
