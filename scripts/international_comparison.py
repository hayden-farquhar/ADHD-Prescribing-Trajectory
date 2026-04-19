"""
international_comparison.py
============================
International comparison of ADHD medication prescribing trends:
  - Australia (PBS, unique patients dispensed, 2004–2024)
  - Denmark (National registers, medicated patients, 2000–2022; Grøntved et al. 2025)
  - UK (CPRD, prescribing prevalence per 10,000, 2004–2015; Renoux et al. 2016)

Outputs:
  - Figure 3: Indexed growth comparison (base year 2005 = 100)
  - Table 2: Cross-national comparison summary
  - data/international/comparison_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "adhd_annual_national.parquet"
FIG_DIR = PROJECT_DIR / "outputs" / "figures"
TAB_DIR = PROJECT_DIR / "outputs" / "tables"
INT_DIR = PROJECT_DIR / "data" / "international"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
INT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

AU_COL = "#1b9e77"   # teal
DK_COL = "#d95f02"   # orange
UK_COL = "#7570b3"   # purple


# ═══════════════════════════════════════════════════════════════════════════════
# DATA: Australia (from PBS/AIHW parquet)
# ═══════════════════════════════════════════════════════════════════════════════
def load_australia():
    """Load Australian PBS data: total unique patients dispensed ADHD meds."""
    df = pd.read_parquet(DATA_PATH)
    sub = df[
        (df["medication"].str.contains("All ADHD", case=False))
        & (df["measure"] == "Patients")
        & (df["sex"] == "People")
    ]
    total = sub.groupby("fy_start")["value"].sum().sort_index()
    # Also get sex-stratified
    male = df[
        (df["medication"].str.contains("All ADHD", case=False))
        & (df["measure"] == "Patients")
        & (df["sex"] == "Male")
    ].groupby("fy_start")["value"].sum().sort_index()
    female = df[
        (df["medication"].str.contains("All ADHD", case=False))
        & (df["measure"] == "Patients")
        & (df["sex"] == "Female")
    ].groupby("fy_start")["value"].sum().sort_index()

    return pd.DataFrame({
        "year": total.index,
        "total": total.values,
        "male": male.values,
        "female": female.values,
        "country": "Australia",
        "metric": "Unique patients dispensed (PBS)",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# DATA: Denmark (from Grøntved et al. 2025, Acta Psychiatr Scand)
# ═══════════════════════════════════════════════════════════════════════════════
def load_denmark():
    """
    Danish data from Grøntved et al. (2025).
    Source: National register-based open cohort study, 2000–2022.
    Metric: Number of prevalent ADHD cases who redeemed medication.

    Data extracted from Table 1 and main text of the paper:
      - Medicated patients by year (select years available from published text)
      - Full annual series is in supplementary Table S1 (Excel, not publicly accessible)

    Note: These are the directly reported/extractable figures. Years between
    key data points are linearly interpolated for visualisation only.
    Sex-stratified annual data not available from published text (only 2022).
    """
    # Key data points extracted from published paper
    # (year, medicated_patients, prevalent_cases, prevalence_pct)
    raw = pd.DataFrame({
        "year":      [2000,  2005,   2010,   2015,   2020,    2022],
        "total":     [2024,  6727,   32519,  42307,  61782,   86707],
        "prevalent": [5383,  14960,  52395,  95318,  143632,  179632],
        "prev_pct":  [0.10,  0.28,   0.95,   1.68,   2.47,    3.03],
    })
    raw["country"] = "Denmark"
    raw["metric"] = "Medicated patients (national registers)"
    raw["male"] = np.nan
    raw["female"] = np.nan

    # Interpolate to annual for smoother plotting
    all_years = np.arange(2000, 2023)
    interp = pd.DataFrame({"year": all_years})
    interp["total"] = np.interp(all_years, raw["year"], raw["total"])
    interp["country"] = "Denmark"
    interp["metric"] = "Medicated patients (national registers)"
    interp["male"] = np.nan
    interp["female"] = np.nan
    interp["interpolated"] = ~interp["year"].isin(raw["year"])

    return interp, raw


# ═══════════════════════════════════════════════════════════════════════════════
# DATA: United Kingdom (from Renoux et al. 2016, Br J Clin Pharmacol)
# ═══════════════════════════════════════════════════════════════════════════════
def load_uk():
    """
    UK data from Renoux et al. (2016).
    Source: Clinical Practice Research Datalink (CPRD), 1995–2015.
    Metric: Prescribing prevalence per 10,000 population (ages 6–45).

    Data extracted from Figure 2 / Table in the published paper.
    Sex-stratified rates only available for 2000 and 2014.
    """
    raw = pd.DataFrame({
        "year":  [2000,  2001,  2002,  2003,  2004,  2005,  2006,  2007,
                  2008,  2009,  2010,  2011,  2012,  2013,  2014,  2015],
        "total": [42.7,  50.4,  64.0,  86.3,  107.2, 127.5, 155.7, 189.8,
                  209.4, 225.0, 240.4, 268.9, 295.9, 322.6, 353.1, 394.4],
    })
    raw["country"] = "United Kingdom"
    raw["metric"] = "Prescribing prevalence per 10,000 (CPRD)"
    raw["male"] = np.nan
    raw["female"] = np.nan
    # Sex-stratified at two time points
    raw.loc[raw["year"] == 2000, "male"] = 73.5
    raw.loc[raw["year"] == 2000, "female"] = 10.4
    raw.loc[raw["year"] == 2014, "male"] = 568.7
    raw.loc[raw["year"] == 2014, "female"] = 134.3

    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Indexed growth comparison
# ═══════════════════════════════════════════════════════════════════════════════
def figure3(au, dk_interp, dk_raw, uk):
    """
    Indexed growth comparison using 2005 as base year (= 100).
    2005 chosen because it's the earliest year with data for all three countries.
    """
    print("Generating Figure 3...")

    base_year = 2005

    # Australia: index to 2005
    au_base = au.loc[au["year"] == base_year, "total"].values[0]
    au["index"] = au["total"] / au_base * 100

    # Denmark (interpolated): index to 2005
    dk_base = dk_interp.loc[dk_interp["year"] == base_year, "total"].values[0]
    dk_interp["index"] = dk_interp["total"] / dk_base * 100

    # Denmark raw points for markers
    dk_raw_base = dk_raw.loc[dk_raw["year"] == base_year, "total"].values[0]
    dk_raw["index"] = dk_raw["total"] / dk_raw_base * 100

    # UK: index to 2005
    uk_base = uk.loc[uk["year"] == base_year, "total"].values[0]
    uk["index"] = uk["total"] / uk_base * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"wspace": 0.3})

    # ── Panel A: Full indexed comparison ──────────────────────────────────────
    # Australia
    ax1.plot(au["year"], au["index"], "o-", color=AU_COL, markersize=4,
             linewidth=2, label="Australia (PBS)", zorder=4)

    # Denmark — interpolated line + raw point markers
    dk_plot = dk_interp[(dk_interp["year"] >= 2004) & (dk_interp["year"] <= 2022)]
    ax1.plot(dk_plot["year"], dk_plot["index"], "-", color=DK_COL,
             linewidth=2, alpha=0.6, zorder=3)
    dk_raw_plot = dk_raw[dk_raw["year"] >= 2004]
    ax1.plot(dk_raw_plot["year"], dk_raw_plot["index"], "D", color=DK_COL,
             markersize=6, zorder=5, label="Denmark (registers)")

    # UK
    uk_plot = uk[uk["year"] >= 2004]
    ax1.plot(uk_plot["year"], uk_plot["index"], "s-", color=UK_COL, markersize=4,
             linewidth=2, label="United Kingdom (CPRD)", zorder=4)

    ax1.axhline(100, color="gray", linewidth=0.5, linestyle=":")
    ax1.set_xlabel("Year")
    ax1.set_ylabel(f"Index ({base_year} = 100)")
    ax1.set_title("A. Indexed growth in ADHD medication use", loc="left", fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.2)

    # Annotate final values
    au_final = au["index"].iloc[-1]
    ax1.annotate(f"{au_final:.0f}", xy=(au["year"].iloc[-1], au_final),
                 xytext=(au["year"].iloc[-1] + 0.3, au_final),
                 fontsize=8, color=AU_COL, fontweight="bold", va="center")

    dk_final = dk_raw_plot["index"].iloc[-1]
    ax1.annotate(f"{dk_final:.0f}", xy=(dk_raw_plot["year"].iloc[-1], dk_final),
                 xytext=(dk_raw_plot["year"].iloc[-1] + 0.3, dk_final),
                 fontsize=8, color=DK_COL, fontweight="bold", va="center")

    uk_final = uk_plot["index"].iloc[-1]
    ax1.annotate(f"{uk_final:.0f}", xy=(uk_plot["year"].iloc[-1], uk_final),
                 xytext=(uk_plot["year"].iloc[-1] + 0.3, uk_final),
                 fontsize=8, color=UK_COL, fontweight="bold", va="center")

    # Note on Denmark interpolation
    ax1.text(0.02, 0.02,
             "Denmark: diamonds = reported data;\nline = linear interpolation between reports",
             transform=ax1.transAxes, fontsize=7, color="gray", va="bottom")

    # ── Panel B: Overlapping period comparison (2005–2015) ────────────────────
    overlap_years = range(2005, 2016)

    au_overlap = au[au["year"].isin(overlap_years)]
    dk_overlap = dk_interp[dk_interp["year"].isin(overlap_years)]
    uk_overlap = uk[uk["year"].isin(overlap_years)]

    ax2.plot(au_overlap["year"], au_overlap["index"], "o-", color=AU_COL,
             markersize=4, linewidth=2, label="Australia")
    ax2.plot(dk_overlap["year"], dk_overlap["index"], "D-", color=DK_COL,
             markersize=4, linewidth=2, label="Denmark")
    ax2.plot(uk_overlap["year"], uk_overlap["index"], "s-", color=UK_COL,
             markersize=4, linewidth=2, label="United Kingdom")

    ax2.axhline(100, color="gray", linewidth=0.5, linestyle=":")
    ax2.set_xlabel("Year")
    ax2.set_ylabel(f"Index ({base_year} = 100)")
    ax2.set_title("B. Overlapping period (2005–2015)", loc="left", fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.2)

    fig.savefig(FIG_DIR / "figure3_international_comparison.png")
    fig.savefig(FIG_DIR / "figure3_international_comparison.pdf")
    plt.close()
    print("  -> figure3_international_comparison.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 2: Cross-national summary
# ═══════════════════════════════════════════════════════════════════════════════
def table2(au, dk_raw, uk):
    """Cross-national comparison summary table."""
    print("Generating Table 2...")

    # Compute growth metrics for each country over comparable periods

    # -- Australia: full period 2004-2024 and overlap period 2005-2015 --
    au_2004 = au.loc[au["year"] == 2004, "total"].values[0]
    au_2024 = au.loc[au["year"] == 2023, "total"].values[0]  # fy_start 2023 = FY 2023-24
    au_2005 = au.loc[au["year"] == 2005, "total"].values[0]
    au_2015 = au.loc[au["year"] == 2015, "total"].values[0]
    au_full_growth = (au_2024 / au_2004 - 1) * 100
    au_full_cagr = ((au_2024 / au_2004) ** (1 / 19) - 1) * 100
    au_overlap_growth = (au_2015 / au_2005 - 1) * 100
    au_overlap_cagr = ((au_2015 / au_2005) ** (1 / 10) - 1) * 100
    au_mf_start = au.loc[au["year"] == 2004, "male"].values[0] / au.loc[au["year"] == 2004, "female"].values[0]
    au_mf_end = au.loc[au["year"] == 2023, "male"].values[0] / au.loc[au["year"] == 2023, "female"].values[0]

    # -- Denmark: full reported period and overlap period --
    dk_2005 = dk_raw.loc[dk_raw["year"] == 2005, "total"].values[0]
    dk_2022 = dk_raw.loc[dk_raw["year"] == 2022, "total"].values[0]
    dk_2000 = dk_raw.loc[dk_raw["year"] == 2000, "total"].values[0]
    dk_2015 = dk_raw.loc[dk_raw["year"] == 2015, "total"].values[0]
    dk_full_growth = (dk_2022 / dk_2000 - 1) * 100
    dk_full_cagr = ((dk_2022 / dk_2000) ** (1 / 22) - 1) * 100
    dk_overlap_growth = (dk_2015 / dk_2005 - 1) * 100
    dk_overlap_cagr = ((dk_2015 / dk_2005) ** (1 / 10) - 1) * 100

    # -- UK: full reported period and overlap period --
    uk_2000 = uk.loc[uk["year"] == 2000, "total"].values[0]
    uk_2015 = uk.loc[uk["year"] == 2015, "total"].values[0]
    uk_2005 = uk.loc[uk["year"] == 2005, "total"].values[0]
    uk_full_growth = (uk_2015 / uk_2000 - 1) * 100
    uk_full_cagr = ((uk_2015 / uk_2000) ** (1 / 15) - 1) * 100
    uk_overlap_growth = (uk_2015 / uk_2005 - 1) * 100
    uk_overlap_cagr = ((uk_2015 / uk_2005) ** (1 / 10) - 1) * 100
    uk_mf_2000 = uk.loc[uk["year"] == 2000, "male"].values[0] / uk.loc[uk["year"] == 2000, "female"].values[0]
    uk_mf_2014 = uk.loc[uk["year"] == 2014, "male"].values[0] / uk.loc[uk["year"] == 2014, "female"].values[0]

    rows = [
        {
            "Country": "Australia",
            "Data source": "PBS/AIHW",
            "Metric": "Unique patients dispensed",
            "Full period": "2004–2024",
            "Full growth (%)": f"{au_full_growth:,.1f}",
            "Full CAGR (%)": f"{au_full_cagr:.1f}",
            "Overlap period growth (2005–2015, %)": f"{au_overlap_growth:,.1f}",
            "Overlap CAGR (%)": f"{au_overlap_cagr:.1f}",
            "M:F ratio (start)": f"{au_mf_start:.2f}:1",
            "M:F ratio (end)": f"{au_mf_end:.2f}:1",
        },
        {
            "Country": "Denmark",
            "Data source": "National registers",
            "Metric": "Medicated patients",
            "Full period": "2000–2022",
            "Full growth (%)": f"{dk_full_growth:,.1f}",
            "Full CAGR (%)": f"{dk_full_cagr:.1f}",
            "Overlap period growth (2005–2015, %)": f"{dk_overlap_growth:,.1f}",
            "Overlap CAGR (%)": f"{dk_overlap_cagr:.1f}",
            "M:F ratio (start)": "—*",
            "M:F ratio (end)": "~1.4:1†",
        },
        {
            "Country": "United Kingdom",
            "Data source": "CPRD",
            "Metric": "Prescribing prevalence per 10,000",
            "Full period": "2000–2015",
            "Full growth (%)": f"{uk_full_growth:,.1f}",
            "Full CAGR (%)": f"{uk_full_cagr:.1f}",
            "Overlap period growth (2005–2015, %)": f"{uk_overlap_growth:,.1f}",
            "Overlap CAGR (%)": f"{uk_overlap_cagr:.1f}",
            "M:F ratio (start)": f"{uk_mf_2000:.1f}:1",
            "M:F ratio (end)": f"{uk_mf_2014:.1f}:1",
        },
    ]

    table = pd.DataFrame(rows)
    table.to_csv(TAB_DIR / "table2_international_comparison.csv", index=False)

    with open(TAB_DIR / "table2_international_comparison.md", "w") as f:
        f.write("# Table 2. Cross-national comparison of ADHD medication prescribing trends\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n\nCAGR = compound annual growth rate.\n")
        f.write("Overlap period = 2005–2015, the longest period with data from all three countries.\n")
        f.write("* Sex-stratified annual data not available from published text for Denmark.\n")
        f.write("† In 2022, males comprised 58.3% of diagnosed ADHD cases in Denmark (Grøntved et al. 2025).\n")
        f.write("\n**Sources:**\n")
        f.write("- Australia: Australian Institute of Health and Welfare, PBS dispensing data.\n")
        f.write("- Denmark: Grøntved S et al. (2025) Acta Psychiatr Scand; 152(1):27–38.\n")
        f.write("- United Kingdom: Renoux C et al. (2016) Br J Clin Pharmacol; 82(3):858–868.\n")

    print("  -> table2_international_comparison.csv/md")
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Save combined data for reproducibility
# ═══════════════════════════════════════════════════════════════════════════════
def save_comparison_data(au, dk_raw, dk_interp, uk):
    """Save all comparison data as a single CSV."""
    print("Saving comparison data...")

    # Australia
    au_out = au[["year", "country", "metric", "total", "male", "female"]].copy()
    au_out["data_type"] = "observed"

    # Denmark raw
    dk_out = dk_raw[["year", "country", "metric", "total", "male", "female"]].copy()
    dk_out["data_type"] = "observed"

    # Denmark interpolated (mark which are interpolated)
    dk_i_out = dk_interp[["year", "country", "metric", "total", "male", "female", "interpolated"]].copy()
    dk_i_out["data_type"] = dk_i_out["interpolated"].map({True: "interpolated", False: "observed"})
    dk_i_out = dk_i_out.drop(columns=["interpolated"])

    # UK
    uk_out = uk[["year", "country", "metric", "total", "male", "female"]].copy()
    uk_out["data_type"] = "observed"

    combined = pd.concat([au_out, dk_out, uk_out], ignore_index=True)
    combined.to_csv(INT_DIR / "comparison_data.csv", index=False)

    # Also save the interpolated Denmark data separately
    dk_i_out.to_csv(INT_DIR / "denmark_interpolated.csv", index=False)

    print("  -> data/international/comparison_data.csv")
    print("  -> data/international/denmark_interpolated.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Project 63 — International Comparison")
    print("=" * 60)

    au = load_australia()
    dk_interp, dk_raw = load_denmark()
    uk = load_uk()

    print(f"\nAustralia: {len(au)} annual data points ({au['year'].min()}–{au['year'].max()})")
    print(f"Denmark: {len(dk_raw)} reported data points ({dk_raw['year'].min()}–{dk_raw['year'].max()})")
    print(f"  (interpolated to {len(dk_interp)} annual points)")
    print(f"UK: {len(uk)} annual data points ({uk['year'].min()}–{uk['year'].max()})")
    print()

    figure3(au, dk_interp, dk_raw, uk)
    t2 = table2(au, dk_raw, uk)
    save_comparison_data(au, dk_raw, dk_interp, uk)

    print("\n" + "=" * 60)
    print("International comparison complete.")
    print("=" * 60)

    # Print key comparisons
    print("\n--- KEY COMPARISONS ---")
    print(t2.to_string(index=False))


if __name__ == "__main__":
    main()
