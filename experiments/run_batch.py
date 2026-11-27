import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.engine import SimulationEngine
from src.logger import TelemetryLogger


def _run_job(args: tuple[str, int, int | None]) -> dict:
    """Top-level worker function (must be picklable for Windows multiprocessing)."""
    planet_key, seed, ticks_override = args
    output_csv = Path("data/raw") / f"{planet_key}_seed{seed}.csv"

    engine = SimulationEngine(planet_key=planet_key, seed=seed)
    if ticks_override is not None:
        engine.ticks_per_run = ticks_override

    start = time.perf_counter()
    with TelemetryLogger(output_csv, planet_key=planet_key, seed=seed) as logger:
        final_stats = engine.run(logger=logger)
    elapsed = time.perf_counter() - start

    return {
        "planet": planet_key,
        "seed": seed,
        "elapsed_s": round(elapsed, 2),
        **final_stats,
    }


def save_batch_summary(results: list[dict], summary_path: Path) -> None:
    if not results:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Sort deterministically by planet then seed
    results.sort(key=lambda r: (r["planet"], r["seed"]))
    fieldnames = list(results[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    default_workers = max(1, (os.cpu_count() or 2) // 2)

    parser = argparse.ArgumentParser(description="Run multi-seed planetary evolution batch.")
    parser.add_argument(
        "--planets",
        nargs="+",
        default=["earth", "mars", "venus"],
        help="Planets to simulate (default: earth mars venus)",
    )
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds per planet (default: 30)")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed integer (default: 0)")
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Parallel worker processes (default: {default_workers})",
    )
    parser.add_argument("--ticks", type=int, default=None, help="Override ticks_per_run (for quick dry runs)")
    args = parser.parse_args()

    tasks = [
        (planet, seed, args.ticks)
        for planet in args.planets
        for seed in range(args.start_seed, args.start_seed + args.seeds)
    ]
    total_jobs = len(tasks)

    print(f"Starting batch: {total_jobs} runs ({args.seeds} seeds x {len(args.planets)} planets) "
          f"using {args.workers} workers...")
    batch_start = time.perf_counter()

    results: list[dict] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_job, task): task for task in tasks}

        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1

            status = f"EXTINCT@{res['extinct_tick']}" if res["extinct_tick"] else f"Pop={res['pop_size']}"
            print(
                f"[{completed:>3}/{total_jobs}] {res['planet'].upper():<5} seed={res['seed']:02d} | "
                f"{status:<12} | Metab={res['mean_metabolism_mode']:.2f} "
                f"Acid={res['mean_acid_resist']:.2f} Therm={res['mean_thermal_tol']:.2f} | "
                f"{res['elapsed_s']:.1f}s"
            )

    summary_path = Path("data/raw/batch_summary.csv")
    save_batch_summary(results, summary_path)

    total_time = time.perf_counter() - batch_start
    print(f"\nBatch complete in {total_time / 60:.2f} minutes.")
    print(f"Individual CSVs saved to data/raw/ and summary written to {summary_path}")


if __name__ == "__main__":
    main()