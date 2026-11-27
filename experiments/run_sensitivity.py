import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.engine import SimulationEngine


MARS_TEMPS = [0.0, -20.0, -40.0, -60.0, -80.0, -100.0]
VENUS_DAYS = [24.0, 48.0, 96.0, 144.0, 192.0, 288.0]
SEEDS = list(range(10))
TICKS = 10_000


def _run_mars_temp(args: tuple[float, int]) -> dict:
    temp_c, seed = args
    eng = SimulationEngine(planet_key="mars", seed=seed)
    eng.env.temp_mean_c[0] = np.float32(temp_c)
    eng.ticks_per_run = TICKS
    stats = eng.run(logger=None)
    return {
        "param": temp_c,
        "seed": seed,
        "survived": int(stats["extinct_tick"] is None and stats["pop_size"] > 0),
        "pop_size": stats["pop_size"],
        "mean_thermal_tol": stats["mean_thermal_tol"],
    }


def _run_venus_day(args: tuple[float, int]) -> dict:
    day_hours, seed = args
    eng = SimulationEngine(planet_key="venus", seed=seed)
    eng.env.day_length_hours = float(day_hours)
    eng.ticks_per_run = TICKS
    stats = eng.run(logger=None)
    return {
        "param": day_hours,
        "seed": seed,
        "survived": int(stats["extinct_tick"] is None and stats["pop_size"] > 0),
        "pop_size": stats["pop_size"],
        "mean_rad_shield": stats["mean_rad_shield"],
        "mean_dormancy_thresh": stats["mean_dormancy_thresh"],
    }


def plot_fig4(mars_df: pd.DataFrame, venus_df: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "figure.dpi": 300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # --- Panel A: Mars Subsurface Temperature Sweep ---
    m_surv = mars_df[mars_df["survived"] == 1]
    m_pop = mars_df.groupby("param")["pop_size"].agg(["mean", "std"]).reset_index()
    m_therm = m_surv.groupby("param")["mean_thermal_tol"].agg(["mean", "std"]).reset_index()

    ax1_r = ax1.twinx()
    l1 = ax1.errorbar(
        m_pop["param"], m_pop["mean"], yerr=m_pop["std"].fillna(0),
        color="#d95f02", marker="o", capsize=3, linewidth=1.8, label="Population"
    )
    l2 = ax1_r.errorbar(
        m_therm["param"], m_therm["mean"], yerr=m_therm["std"].fillna(0),
        color="#5e4fa2", marker="s", linestyle="--", capsize=3, linewidth=1.8, label="Thermal Tol"
    )

    ax1.set_title("(a) Mars Subsurface Temperature Sweep")
    ax1.set_xlabel("Subsurface Mean Temperature (°C)")
    ax1.set_ylabel("Final Population", color="#d95f02")
    ax1_r.set_ylabel("Evolved Thermal Tolerance [0–1]", color="#5e4fa2")
    ax1_r.set_ylim(0, 1.1)
    ax1.invert_xaxis()  # Colder to the right
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend([l1, l2], ["Population", "Thermal Tol"], frameon=False, loc="center left")

    # --- Panel B: Venus Cloud Day-Length Sweep ---
    v_surv = venus_df[venus_df["survived"] == 1]
    v_pop = venus_df.groupby("param")["pop_size"].agg(["mean", "std"]).reset_index()
    v_rad = v_surv.groupby("param")["mean_rad_shield"].agg(["mean", "std"]).reset_index()
    v_dorm = v_surv.groupby("param")["mean_dormancy_thresh"].agg(["mean", "std"]).reset_index()

    ax2_r = ax2.twinx()
    v1 = ax2.errorbar(
        v_pop["param"], v_pop["mean"], yerr=v_pop["std"].fillna(0),
        color="#e6ab02", marker="o", capsize=3, linewidth=1.8, label="Population"
    )
    v2 = ax2_r.errorbar(
        v_rad["param"], v_rad["mean"], yerr=v_rad["std"].fillna(0),
        color="#1b9e77", marker="^", linestyle="--", capsize=3, linewidth=1.6, label="Rad Shield"
    )
    v3 = ax2_r.errorbar(
        v_dorm["param"], v_dorm["mean"], yerr=v_dorm["std"].fillna(0),
        color="#7570b3", marker="s", linestyle=":", capsize=3, linewidth=1.6, label="Dormancy"
    )

    ax2.set_title("(b) Venus Cloud Super-Rotation Period Sweep")
    ax2.set_xlabel("Diurnal Cycle Length (Hours)")
    ax2.set_ylabel("Final Population", color="#b8860b")
    ax2_r.set_ylabel("Evolved Trait Value [0–1]", color="#333333")
    ax2_r.set_ylim(0, 1.1)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend([v1, v2, v3], ["Population", "Rad Shield", "Dormancy"], frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_sensitivity_analysis.png", dpi=300)
    fig.savefig(out_dir / "fig4_sensitivity_analysis.pdf")
    plt.close(fig)


def main() -> None:
    workers = max(1, (os.cpu_count() or 2) // 2)
    raw_dir = Path("data/raw")
    fig_dir = Path("data/figures")
    raw_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    print(f"Running sensitivity sweeps ({len(SEEDS)} seeds/point, {TICKS} ticks) on {workers} workers...")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        mars_tasks = [(t, s) for t in MARS_TEMPS for s in SEEDS]
        mars_results = list(ex.map(_run_mars_temp, mars_tasks))

        venus_tasks = [(d, s) for d in VENUS_DAYS for s in SEEDS]
        venus_results = list(ex.map(_run_venus_day, venus_tasks))

    mars_df = pd.DataFrame(mars_results)
    venus_df = pd.DataFrame(venus_results)

    mars_df.to_csv(raw_dir / "sensitivity_mars_temp.csv", index=False)
    venus_df.to_csv(raw_dir / "sensitivity_venus_day.csv", index=False)

    print("\n--- Mars Subsurface Temperature Sweep ---")
    print(mars_df.groupby("param")[["survived", "pop_size", "mean_thermal_tol"]].mean().round(3))

    print("\n--- Venus Cloud Day-Length Sweep ---")
    print(venus_df.groupby("param")[["survived", "pop_size", "mean_rad_shield", "mean_dormancy_thresh"]].mean().round(3))

    plot_fig4(mars_df, venus_df, fig_dir)
    elapsed = time.perf_counter() - start
    print(f"\nDone in {elapsed:.1f}s. Saved Figure 4 to {fig_dir / 'fig4_sensitivity_analysis.png'}")


if __name__ == "__main__":
    main()