import argparse
import time
from pathlib import Path

from src.engine import SimulationEngine
from src.logger import TelemetryLogger


def run_planet(planet_key: str, seed: int, ticks: int | None = None) -> dict:
    output_csv = Path("data/raw") / f"{planet_key}_seed{seed}.csv"
    engine = SimulationEngine(planet_key=planet_key, seed=seed)

    if ticks is not None:
        engine.ticks_per_run = ticks

    start = time.perf_counter()
    with TelemetryLogger(output_csv, planet_key=planet_key, seed=seed) as logger:
        final_stats = engine.run(logger=logger)
    elapsed = time.perf_counter() - start

    final_stats["planet"] = planet_key
    final_stats["elapsed_s"] = elapsed
    return final_stats


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 92)
    print(
        f"{'Planet':<8} | {'Pop':>6} | {'Sub/Surf/Air':<17} | "
        f"{'Metab':>5} | {'Rad':>5} | {'Acid':>5} | {'Therm':>5} | {'Dorm':>5} | {'Time':>6}"
    )
    print("-" * 92)

    for r in results:
        layers = f"{r['layer_0_subsurface']}/{r['layer_1_surface']}/{r['layer_2_airborne']}"
        status = f"EXTINCT@{r['extinct_tick']}" if r["extinct_tick"] else f"{r['pop_size']}"
        print(
            f"{r['planet'].upper():<8} | {status:>6} | {layers:<17} | "
            f"{r['mean_metabolism_mode']:>5.2f} | {r['mean_rad_shield']:>5.2f} | "
            f"{r['mean_acid_resist']:>5.2f} | {r['mean_thermal_tol']:>5.2f} | "
            f"{r['mean_dormancy_thresh']:>5.2f} | {r['elapsed_s']:>5.2f}s"
        )
    print("=" * 92)
    print("Metab: 0.00 = Pure Chemosynthesis, 1.00 = Pure Photosynthesis\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-seed planetary evolution test.")
    parser.add_argument(
        "--planets",
        nargs="+",
        default=["earth", "mars", "venus"],
        help="Planets to simulate (default: earth mars venus)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--ticks", type=int, default=None, help="Override ticks_per_run")
    args = parser.parse_args()

    results = [run_planet(p, args.seed, args.ticks) for p in args.planets]
    print_summary(results)


if __name__ == "__main__":
    main()