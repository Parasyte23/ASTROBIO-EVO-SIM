from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PLANETS = ["earth", "mars", "venus"]
COLORS = {
    "earth": "#2b8cbe",
    "mars": "#d95f02",
    "venus": "#e6ab02",
}
LAYER_COLORS = ["#5e4fa2", "#3288bd", "#66c2a5"]  # Subsurface, Surface, Airborne


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    })


def load_time_series(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for planet in PLANETS:
        for csv_path in sorted(raw_dir.glob(f"{planet}_seed*.csv")):
            df = pd.read_csv(csv_path, usecols=["planet", "seed", "tick", "pop_size"])
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No seed CSV files found in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def plot_fig1_population(ts_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    for planet in PLANETS:
        sub = ts_df[ts_df["planet"] == planet]
        agg = sub.groupby("tick")["pop_size"].agg(["mean", "std"]).reset_index()
        ticks = agg["tick"].to_numpy()
        mean = agg["mean"].to_numpy()
        std = agg["std"].fillna(0.0).to_numpy()

        ax.plot(ticks, mean, label=planet.capitalize(), color=COLORS[planet], linewidth=1.8)
        ax.fill_between(
            ticks,
            np.maximum(0, mean - std),
            mean + std,
            color=COLORS[planet],
            alpha=0.2,
        )

    ax.set_title("Population Dynamics Across Planetary Regimes (n = 30 seeds, Mean ± SD)")
    ax.set_xlabel("Simulation Time (Ticks / Hours)")
    ax.set_ylabel("Total Living Population")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="center right")

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_population_dynamics.png", dpi=300)
    fig.savefig(out_dir / "fig1_population_dynamics.pdf")
    plt.close(fig)


def plot_fig2_habitats(summary_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    planets_upper = [p.upper() for p in PLANETS]
    sub_vals = []
    surf_vals = []
    air_vals = []

    for p in planets_upper:
        row = summary_df[summary_df["planet"] == p].iloc[0]
        sub_vals.append(row["sub_frac_mean"])
        surf_vals.append(row["surf_frac_mean"])
        air_vals.append(row["air_frac_mean"])

    x = np.arange(len(PLANETS))
    width = 0.55

    ax.bar(x, sub_vals, width, label="Subsurface (Layer 0)", color=LAYER_COLORS[0])
    ax.bar(x, surf_vals, width, bottom=sub_vals, label="Surface (Layer 1)", color=LAYER_COLORS[1])
    bottom_air = np.array(sub_vals) + np.array(surf_vals)
    ax.bar(x, air_vals, width, bottom=bottom_air, label="Airborne (Layer 2)", color=LAYER_COLORS[2])

    ax.set_title("Evolved Vertical Habitat Partitioning (Tick 20,000)")
    ax.set_ylabel("Population Share (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PLANETS])
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_habitat_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig2_habitat_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_traits(summary_df: pd.DataFrame, out_dir: Path) -> None:
    traits = [
        ("mean_metabolism_mode", "Photosynthesis\nMode"),
        ("mean_rad_shield", "Radiation\nShield"),
        ("mean_acid_resist", "Acid\nResistance"),
        ("mean_thermal_tol", "Thermal\nTolerance"),
        ("mean_dormancy_thresh", "Dormancy\nThreshold"),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(traits))
    width = 0.25

    for i, planet in enumerate(PLANETS):
        row = summary_df[summary_df["planet"] == planet.upper()].iloc[0]
        means = [row[f"{col}_mean"] for col, _ in traits]
        sds = [row[f"{col}_sd"] for col, _ in traits]
        offset = (i - 1) * width

        ax.bar(
            x + offset,
            means,
            width,
            yerr=sds,
            capsize=3,
            label=planet.capitalize(),
            color=COLORS[planet],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_title("Convergent Trait Evolution Across Planetary Environments (Mean ± SD)")
    ax.set_ylabel("Normalized Trait Value [0.0 – 1.0]")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in traits])
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_trait_convergence.png", dpi=300)
    fig.savefig(out_dir / "fig3_trait_convergence.pdf")
    plt.close(fig)


def main() -> None:
    setup_style()
    raw_dir = Path("data/raw")
    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary_table.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Run `python -m analysis.stats` first to generate summary_table.csv.")

    summary_df = pd.read_csv(summary_path)
    ts_df = load_time_series(raw_dir)

    plot_fig1_population(ts_df, out_dir)
    plot_fig2_habitats(summary_df, out_dir)
    plot_fig3_traits(summary_df, out_dir)

    print(f"Generated 3 publication figures (PNG + PDF) in {out_dir}")


if __name__ == "__main__":
    main()