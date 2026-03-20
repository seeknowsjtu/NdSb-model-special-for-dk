from __future__ import annotations
from pathlib import Path
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from config import default_params, normalize_params_dict
from data_io import (
    load_csv_auto,
    parse_fluence_ratio_from_name,
    fit_params_multi,
    export_multi_fit_results,
)

# =========================
# 1. 数据文件列表
# =========================
CSV_FILES = [
    "deltak12k_1p0mW.csv",
    "deltak12k_2p0mW.csv",
    "deltak12k_2p5mW.csv",
    "deltak12k_3p0mW.csv",
    "deltak12k_3p5mW.csv",
    "deltak12k_4p0mW.csv",
]

DATA_DIR = Path(".")

# =========================
# 2. 运行控制
# =========================
HEARTBEAT_SEC = 10          # 每隔多少秒提示“还在跑”
SMOKE_TEST = True           # 先做一个小步数试跑
SMOKE_MAX_NFEV = 20         # smoke test 的最大 nfev
FULL_MAX_NFEV = 120         # 正式跑的最大 nfev

# =========================
# 3. 读入一个数据集
# =========================
def load_dataset(path: Path) -> dict:
    t, Te, Ts, Tl, S, names, unit = load_csv_auto(str(path))
    if S is None:
        raise ValueError(f"{path.name} has no S column.")

    fluence_ratio = parse_fluence_ratio_from_name(str(path))

    return {
        "name": path.name,
        "path": str(path),
        "t": t,
        "Te": Te,
        "Ts": Ts,
        "Tl": Tl,
        "S": S,
        "fluence_ratio": fluence_ratio,
    }

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
            f"fluence_ratio={ds['fluence_ratio']:.2f}"
        )
    print(f"Total points across all datasets = {total_points}")

# =========================
# 5. 构造起始参数
# =========================
def make_initial_params() -> dict:
    p0 = normalize_params_dict(default_params())

    # 轻微扰动初值（可注释掉）
    p0["G_el0"] *= 1.02
    p0["G_es0"] *= 0.98
    p0["tau_m0"] *= 1.05
    p0["Gamma_eta"] *= 0.95

    return p0

# =========================
# 6. 单次拟合任务
# =========================
def run_fit(datasets: list[dict], p0: dict, max_nfev: int, export_root: str):
    fit_bundle, res = fit_params_multi(
        datasets,
        p0,
        local_keys=["dt_local"],
        observable_mode="eta",
        sigma_S=0.02,
        max_nfev=max_nfev,
    )

    exports = export_multi_fit_results(
        fit_bundle,
        res,
        export_root=export_root,
    )
    return fit_bundle, res, exports

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

        # 如果拟合线程里抛异常，这里会再次抛出
        fit_bundle, res, exports = future.result()

    print(f"[multi-fit] finished in {end - start:.1f} s", flush=True)
    return fit_bundle, res, exports, end - start

# =========================
# 8. 打印结果
# =========================
def print_fit_summary(fit_bundle, res, exports, wall_time_sec: float) -> None:
    print("\n=== optimizer summary ===")
    print(f"success = {res.success}")
    print(f"status  = {res.status}")
    print(f"cost    = {res.cost:.6e}")
    print(f"nfev    = {res.nfev}")
    print(f"mode    = {fit_bundle['observable_mode']}")
    print(f"locals  = {fit_bundle['local_keys']}")
    print(f"wall_time_sec = {wall_time_sec:.1f}")

    print("\n=== fitted global params ===")
    for k in fit_bundle["global_keys"]:
        v = fit_bundle["best_global_params"][k]
        print(f"  {k:20s} = {v:.8g}")

    print("\n=== dataset summary ===")
    for row in fit_bundle["dataset_summary"]:
        dt_local_ps = row.get("dt_local_ps", float("nan"))
        print(
            f"  {row['dataset_name']:>22s} | "
            f"fluence={row['fluence_ratio']:.2f} | "
            f"rms={row['rms']:.4e} | "
            f"wrms={row['wrms']:.4e} | "
            f"dt_local_ps={dt_local_ps:.4f}"
        )

    print("\n=== exports ===")
    for k, v in exports.items():
        print(f"  {k}: {v}")

# =========================
# 9. 主程序
# =========================
def main() -> None:
    try:
        datasets = [load_dataset(DATA_DIR / fn) for fn in CSV_FILES]
        datasets.sort(key=lambda d: d["fluence_ratio"])
        print_dataset_summary(datasets)

        p0 = make_initial_params()

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