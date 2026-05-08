from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================
DATA_DIR = Path(r"F:\python4git\simulate_dk\SW6_DK_SIGMA_CSV")
OUT_DIR = DATA_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

# If True: save one figure containing all curves.
SAVE_OVERLAY = True

# If True: save one separate figure for each csv file.
SAVE_EACH = True

# If True and sigma_dk exists: plot error bars.
USE_ERRORBAR = True


def parse_power_from_filename(name: str) -> float:
    """
    Parse power from names like:
        dk_2p5mW_sigma.csv
        deltak12k_2p5mW.csv
        S_0p5mW.csv
    """
    m = re.search(r"(\d+)p(\d+)mW", name, flags=re.IGNORECASE)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")

    m = re.search(r"(\d+)mW", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    return np.nan


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Find a column by case-insensitive matching.
    """
    col_map = {c.strip().lower(): c for c in df.columns}
    for key in candidates:
        if key.lower() in col_map:
            return col_map[key.lower()]
    return None


def load_one_csv(path: Path) -> dict:
    df = pd.read_csv(path)

    t_col = find_column(df, ["t_ps", "tps", "time_ps", "time", "t"])
    dk_col = find_column(df, ["delta_k", "deltak", "dk", "delta_k_ainv", "deltak_ainv"])
    sig_col = find_column(df, ["sigma_dk", "sigmadeltak12_k", "delta_k_err", "dk_err", "err", "error"])

    if t_col is None:
        raise ValueError(f"{path.name}: cannot find time column. Columns = {list(df.columns)}")
    if dk_col is None:
        raise ValueError(f"{path.name}: cannot find delta_k column. Columns = {list(df.columns)}")

    t = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
    dk = pd.to_numeric(df[dk_col], errors="coerce").to_numpy(dtype=float)

    if sig_col is not None:
        sigma = pd.to_numeric(df[sig_col], errors="coerce").to_numpy(dtype=float)
    else:
        sigma = None

    mask = np.isfinite(t) & np.isfinite(dk)
    if sigma is not None:
        mask = mask & np.isfinite(sigma)

    t = t[mask]
    dk = dk[mask]
    sigma = sigma[mask] if sigma is not None else None

    idx = np.argsort(t)

    return {
        "path": path,
        "name": path.name,
        "power_mW": parse_power_from_filename(path.name),
        "t_ps": t[idx],
        "delta_k": dk[idx],
        "sigma_dk": sigma[idx] if sigma is not None else None,
    }


def plot_one(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    label = data["name"]

    if USE_ERRORBAR and data["sigma_dk"] is not None:
        ax.errorbar(
            data["t_ps"],
            data["delta_k"],
            yerr=data["sigma_dk"],
            fmt="o-",
            markersize=4,
            linewidth=1,
            capsize=2,
            label=label,
        )
    else:
        ax.plot(
            data["t_ps"],
            data["delta_k"],
            "o-",
            markersize=4,
            linewidth=1,
            label=label,
        )

    ax.set_xlabel("Delay time (ps)")
    ax.set_ylabel(r"$\Delta k$ ($\mathrm{\AA}^{-1}$)")
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = OUT_DIR / f"{data['path'].stem}_dk_vs_time.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")


def plot_overlay(all_data: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for data in all_data:
        if np.isfinite(data["power_mW"]):
            label = f"{data['power_mW']:g} mW"
        else:
            label = data["name"]

        if USE_ERRORBAR and data["sigma_dk"] is not None:
            ax.errorbar(
                data["t_ps"],
                data["delta_k"],
                yerr=data["sigma_dk"],
                fmt="o-",
                markersize=4,
                linewidth=1,
                capsize=2,
                label=label,
            )
        else:
            ax.plot(
                data["t_ps"],
                data["delta_k"],
                "o-",
                markersize=4,
                linewidth=1,
                label=label,
            )

    ax.set_xlabel("Delay time (ps)")
    ax.set_ylabel(r"$\Delta k$ ($\mathrm{\AA}^{-1}$)")
    ax.set_title("All delta-k traces")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out_path = OUT_DIR / "all_dk_vs_time_overlay.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Saved: {out_path}")


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {DATA_DIR}")

    all_data = []
    for path in csv_files:
        try:
            data = load_one_csv(path)
            all_data.append(data)
            print(f"Loaded: {path.name} | N = {len(data['t_ps'])}")
        except Exception as exc:
            print(f"Skipped {path.name}: {exc}")

    all_data = sorted(
        all_data,
        key=lambda d: d["power_mW"] if np.isfinite(d["power_mW"]) else 9999,
    )

    if SAVE_EACH:
        for data in all_data:
            plot_one(data)

    if SAVE_OVERLAY:
        plot_overlay(all_data)

    print("Done.")


if __name__ == "__main__":
    main()