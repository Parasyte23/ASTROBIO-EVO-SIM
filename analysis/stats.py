from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


PLANETS = ["earth", "mars", "venus"]

TRAIT_METRICS = [
    ("pop_size", "Population"),
    ("sub_frac", "Subsurface %"),
    ("surf_frac", "Surface %"),
    ("air_frac", "Airborne %"),
    ("mean_metabolism_mode", "Metabolism (Photo)"),
    ("mean_rad_shield", "Rad Shield"),
    ("mean_acid_resist", "Acid Resist"),
    ("mean_thermal_tol", "Thermal Tol"),
    ("mean_dormancy_thresh", "Dormancy Thresh"),
    ("dormant_frac", "Dormant Frac"),
]


def load_batch_summary(summary_path: str | Path = "data/raw/batch_summary.csv") -> pd.DataFrame:
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run `python -m experiments.run_batch --seeds 30` first."
        )

    df = pd.read_csv(path)
    df["survived"] = df["extinct_tick"].isna() & (df["pop_size"] > 0)

    # Compute habitat layer fractions safely (avoid division by zero on extinct runs)
    safe_pop = df["pop_size"].replace(0, np.nan)
    df["sub_frac"] = (df["layer_0_subsurface"] / safe_pop) * 100.0
    df["surf_frac"] = (df["layer_1_surface"] / safe_pop) * 100.0
    df["air_frac"] = (df["layer_2_airborne"] / safe_pop) * 100.0

    return df


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Computes survival rates and mean +/- SD across surviving seeds per planet."""
    rows = []
    for planet in PLANETS:
        sub = df[df["planet"] == planet]
        if sub.empty:
            continue

        n_total = len(sub)
        surv = sub[sub["survived"]]
        n_surv = len(surv)
        extinct_rate = ((n_total - n_surv) / n_total) * 100.0

        row = {
            "planet": planet.upper(),
            "n_seeds": n_total,
            "n_survived": n_surv,
            "extinction_rate_pct": round(extinct_rate, 2),
        }

        for col, _ in TRAIT_METRICS:
            if n_surv > 0:
                mean_val = float(surv[col].mean())
                std_val = float(surv[col].std(ddof=1)) if n_surv > 1 else 0.0
                row[f"{col}_mean"] = round(mean_val, 4)
                row[f"{col}_sd"] = round(std_val, 4)
            else:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_sd"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compute_significance_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs Kruskal-Wallis H-test across all 3 planets and pairwise Mann-Whitney U
    tests with Bonferroni correction (3 pairwise comparisons).
    """
    surv_df = df[df["survived"]]
    pairs = [("earth", "mars"), ("earth", "venus"), ("mars", "venus")]
    rows = []

    for col, label in TRAIT_METRICS:
        groups = [surv_df[surv_df["planet"] == p][col].dropna().to_numpy() for p in PLANETS]

        # Kruskal-Wallis requires at least 2 non-empty groups with variation
        valid_groups = [g for g in groups if len(g) > 0]
        if len(valid_groups) >= 2 and np.unique(np.concatenate(valid_groups)).size > 1:
            kw_stat, kw_p = stats.kruskal(*valid_groups)
        else:
            kw_stat, kw_p = np.nan, np.nan

        row = {
            "metric": label,
            "kruskal_H": round(float(kw_stat), 4) if np.isfinite(kw_stat) else np.nan,
            "kruskal_p": float(kw_p) if np.isfinite(kw_p) else np.nan,
        }

        for p1, p2 in pairs:
            g1 = surv_df[surv_df["planet"] == p1][col].dropna().to_numpy()
            g2 = surv_df[surv_df["planet"] == p2][col].dropna().to_numpy()
            pair_key = f"{p1}_vs_{p2}"

            if len(g1) > 0 and len(g2) > 0:
                u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                p_adj = min(1.0, float(p_val) * len(pairs))  # Bonferroni correction
                row[f"{pair_key}_U"] = round(float(u_stat), 2)
                row[f"{pair_key}_p_adj"] = p_adj
            else:
                row[f"{pair_key}_U"] = np.nan
                row[f"{pair_key}_p_adj"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _fmt_p(p: float) -> str:
    if pd.isna(p):
        return "N/A"
    if p < 1e-4:
        return "< 0.0001 ***"
    if p < 0.001:
        return f"{p:.4f} ***"
    if p < 0.01:
        return f"{p:.4f} **"
    if p < 0.05:
        return f"{p:.4f} *"
    return f"{p:.4f} ns"


def print_tables(desc_df: pd.DataFrame, sig_df: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print("TABLE 1: DESCRIPTIVE STATISTICS (MEAN +/- SD ACROSS SURVIVING SEEDS)")
    print("=" * 96)
    for _, r in desc_df.iterrows():
        print(
            f"\n{r['planet']} (n={int(r['n_survived'])}/{int(r['n_seeds'])} survived, "
            f"extinction={r['extinction_rate_pct']:.1f}%)"
        )
        print(
            f"  Pop: {r['pop_size_mean']:.1f} +/- {r['pop_size_sd']:.1f} | "
            f"Layers (Sub/Surf/Air): {r['sub_frac_mean']:.1f}% / "
            f"{r['surf_frac_mean']:.1f}% / {r['air_frac_mean']:.1f}%"
        )
        print(
            f"  Metab: {r['mean_metabolism_mode_mean']:.3f} +/- {r['mean_metabolism_mode_sd']:.3f} | "
            f"Rad: {r['mean_rad_shield_mean']:.3f} +/- {r['mean_rad_shield_sd']:.3f} | "
            f"Acid: {r['mean_acid_resist_mean']:.3f} +/- {r['mean_acid_resist_sd']:.3f}"
        )
        print(
            f"  Therm: {r['mean_thermal_tol_mean']:.3f} +/- {r['mean_thermal_tol_sd']:.3f} | "
            f"Dorm: {r['mean_dormancy_thresh_mean']:.3f} +/- {r['mean_dormancy_thresh_sd']:.3f}"
        )

    print("\n" + "=" * 96)
    print("TABLE 2: STATISTICAL SIGNIFICANCE (KRUSKAL-WALLIS & BONFERRONI MANN-WHITNEY U)")
    print("=" * 96)
    header = f"{'Metric':<20} | {'Kruskal H':>9} | {'KW p-val':<12} | {'E vs M':<12} | {'E vs V':<12} | {'M vs V':<12}"
    print(header)
    print("-" * len(header))
    for _, r in sig_df.iterrows():
        h_str = f"{r['kruskal_H']:.2f}" if pd.notna(r["kruskal_H"]) else "N/A"
        print(
            f"{r['metric']:<20} | {h_str:>9} | {_fmt_p(r['kruskal_p']):<12} | "
            f"{_fmt_p(r['earth_vs_mars_p_adj']):<12} | "
            f"{_fmt_p(r['earth_vs_venus_p_adj']):<12} | "
            f"{_fmt_p(r['mars_vs_venus_p_adj']):<12}"
        )
    print("=" * 96)
    print("Significance levels (Bonferroni-adjusted): * p<0.05, ** p<0.01, *** p<0.001, ns = not significant\n")


def main() -> None:
    df = load_batch_summary()
    desc_df = compute_descriptive_stats(df)
    sig_df = compute_significance_tests(df)

    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    desc_df.to_csv(out_dir / "summary_table.csv", index=False)
    sig_df.to_csv(out_dir / "significance_tests.csv", index=False)

    print_tables(desc_df, sig_df)
    print(f"Saved CSV tables to {out_dir / 'summary_table.csv'} and {out_dir / 'significance_tests.csv'}")


if __name__ == "__main__":
    main()