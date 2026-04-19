"""
generate_figures.py
===================
Publication-quality figures and tables for Project 63:
Twenty-Year Trajectory of ADHD Medication Prescribing in Australia, 2004–2024.

Outputs:
  Main text:
    - Figure 1: Overall trajectory + projection (2-panel)
    - Figure 2: Sex decomposition + M:F ratio (2-panel)
    - Table 1: Growth summary by medication and sex (CSV + formatted)

  Supplementary:
    - Figure S1: Age group decomposition (stacked area + indexed growth)
    - Figure S2: Medication-specific trends by sex (5-panel grid)
    - Figure S3: Prescribing rate per 1,000 population by sex
    - Table S1: Full growth statistics (all medications × sex)
    - Table S2: Annual patient counts (all years × medication × sex)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "adhd_annual_national.parquet"
FIG_DIR = PROJECT_DIR / "outputs" / "figures"
TAB_DIR = PROJECT_DIR / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

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

MALE_COL = "#2166ac"
FEMALE_COL = "#b2182b"
TOTAL_COL = "#333333"
PROJ_COL = "#d62728"
COVID_COL = "#f0f0f0"

MED_COLOURS = {
    "Dexamfetamine": "#e41a1c",
    "Methylphenidate": "#377eb8",
    "Atomoxetine": "#4daf4a",
    "Lisdexamfetamine": "#984ea3",
    "Guanfacine": "#ff7f00",
}

AGE_COLOURS = {
    "0 - 11": "#1f77b4",
    "12 - 17": "#ff7f0e",
    "18 - 24": "#2ca02c",
    "25 - 44": "#d62728",
    "45+": "#9467bd",
}

AGE_LABELS = {
    "0 - 11": "Children (0–11)",
    "12 - 17": "Adolescents (12–17)",
    "18 - 24": "Young adults (18–24)",
    "25 - 44": "Adults (25–44)",
    "45+": "Older adults (45+)",
}


def load_data():
    return pd.read_parquet(DATA_PATH)


def get_patients(df, medication="All ADHD", sex="People"):
    """Sum patients across age groups for a given medication/sex."""
    sub = df[
        (df["medication"].str.contains(medication, case=False, na=False))
        & (df["measure"] == "Patients")
        & (df["sex"] == sex)
    ]
    total = sub.groupby("fy_start")["value"].sum().sort_index()
    return total


def fy_label(year):
    """Convert fy_start int to financial year label."""
    return f"{year}–{str(year + 1)[-2:]}"


def fy_labels_for_axis(years):
    """Short labels for x-axis ticks."""
    return [f"'{str(y)[-2:]}" for y in years]


def add_covid_shading(ax, label=True):
    """Add subtle COVID-19 period shading."""
    ax.axvspan(2019.5, 2021.5, alpha=0.06, color="gray", zorder=0)
    if label:
        ylim = ax.get_ylim()
        ax.text(2020.5, ylim[1] * 0.97, "COVID-19", ha="center", va="top",
                fontsize=7, color="gray", style="italic")


def run_projection(ts, steps=3, n_boot=5000):
    """Holt additive exponential smoothing with simulation-based PIs.

    Uses a simulation approach that accounts for:
    1. Residual noise (resampled from in-sample errors)
    2. Parameter uncertainty (fitted to perturbed training series)
    3. Trend extrapolation uncertainty (variance grows with horizon)

    This produces wider, more realistic intervals than naive residual
    bootstrapping, reflecting genuine uncertainty in an accelerating trend.
    """
    model = ExponentialSmoothing(ts, trend="add", seasonal=None)
    result = model.fit(optimized=True)
    forecast = result.forecast(steps=steps)

    residuals = (ts - result.fittedvalues).values
    sigma = np.std(residuals)
    boot = np.zeros((n_boot, steps))
    rng = np.random.default_rng(42)

    for b in range(n_boot):
        # Perturb the training series to capture parameter uncertainty
        perturbed = ts.copy() + rng.normal(0, sigma * 0.5, size=len(ts))
        try:
            pmodel = ExponentialSmoothing(perturbed, trend="add", seasonal=None)
            presult = pmodel.fit(optimized=True)
            pfore = presult.forecast(steps=steps).values
        except Exception:
            pfore = forecast.values.copy()

        # Add residual noise scaled by horizon
        for h in range(steps):
            horizon_scale = np.sqrt(1 + h)  # variance grows with sqrt(horizon)
            noise = rng.choice(residuals) * horizon_scale
            boot[b, h] = pfore[h] + noise

    return forecast, {
        "lo90": np.percentile(boot, 5, axis=0),
        "hi90": np.percentile(boot, 95, axis=0),
        "lo95": np.percentile(boot, 2.5, axis=0),
        "hi95": np.percentile(boot, 97.5, axis=0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Overall trajectory + projection
# ═══════════════════════════════════════════════════════════════════════════════
def figure1(df):
    print("Generating Figure 1...")
    total = get_patients(df)
    forecast, pi = run_projection(total)
    forecast_years = np.array([2024, 2025, 2026])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[3, 1.2],
                                    gridspec_kw={"hspace": 0.28})

    # Panel A: Absolute counts + projection
    ax1.bar(total.index, total.values, color=TOTAL_COL, alpha=0.75, width=0.7,
            label="Observed", zorder=3)
    ax1.bar(forecast_years, forecast.values, color=PROJ_COL, alpha=0.45, width=0.7,
            label="Projected", zorder=3)

    # 95% PI whiskers
    for i, yr in enumerate(forecast_years):
        ax1.plot([yr, yr], [pi["lo95"][i], pi["hi95"][i]],
                 color=PROJ_COL, linewidth=1.5, zorder=4)
        ax1.plot([yr - 0.15, yr + 0.15], [pi["lo95"][i]] * 2,
                 color=PROJ_COL, linewidth=1.5, zorder=4)
        ax1.plot([yr - 0.15, yr + 0.15], [pi["hi95"][i]] * 2,
                 color=PROJ_COL, linewidth=1.5, zorder=4)
    # 90% PI thicker
    for i, yr in enumerate(forecast_years):
        ax1.plot([yr, yr], [pi["lo90"][i], pi["hi90"][i]],
                 color=PROJ_COL, linewidth=3.5, alpha=0.4, zorder=4)

    # Annotations
    ax1.annotate(f"{total.values[0]:,.0f}", xy=(total.index[0], total.values[0]),
                 xytext=(total.index[0] + 1.5, total.values[0] + 40000),
                 fontsize=8, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax1.annotate(f"{total.values[-1]:,.0f}", xy=(total.index[-1], total.values[-1]),
                 xytext=(total.index[-1] - 2, total.values[-1] + 60000),
                 fontsize=8, ha="right",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax1.annotate(
        f"{forecast.values[-1]:,.0f}\n(95% PI: {pi['lo95'][-1]:,.0f}–{pi['hi95'][-1]:,.0f})",
        xy=(2026, forecast.values[-1]),
        xytext=(2024, forecast.values[-1] + 100000),
        fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", lw=0.5),
    )

    add_covid_shading(ax1)
    ax1.set_ylabel("Patients dispensed ADHD medication")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax1.set_xticks(list(total.index) + list(forecast_years))
    ax1.set_xticklabels(fy_labels_for_axis(list(total.index) + list(forecast_years)), rotation=45)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_title("A. ADHD medication recipients in Australia, 2004–2027", loc="left", fontweight="bold")
    ax1.grid(axis="y", alpha=0.2)

    # Panel B: Year-on-year growth rate
    yoy = total.pct_change() * 100
    colours = [MALE_COL if g > 0 else PROJ_COL for g in yoy.values]
    ax2.bar(total.index[1:], yoy.values[1:], color=colours[1:], alpha=0.7, width=0.7, zorder=3)
    ax2.axhline(0, color="black", linewidth=0.5, zorder=2)
    add_covid_shading(ax2, label=False)
    ax2.set_ylabel("Year-on-year growth (%)")
    ax2.set_xticks(list(total.index))
    ax2.set_xticklabels(fy_labels_for_axis(total.index), rotation=45)
    ax2.set_title("B. Annual growth rate", loc="left", fontweight="bold")
    ax2.grid(axis="y", alpha=0.2)

    fig.savefig(FIG_DIR / "figure1_trajectory.png")
    fig.savefig(FIG_DIR / "figure1_trajectory.pdf")
    plt.close()
    print("  -> figure1_trajectory.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Sex decomposition + M:F ratio
# ═══════════════════════════════════════════════════════════════════════════════
def figure2(df):
    print("Generating Figure 2...")
    male = get_patients(df, sex="Male")
    female = get_patients(df, sex="Female")
    ratio = male / female

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    gridspec_kw={"wspace": 0.35})

    # Panel A: Stacked area by sex
    ax1.fill_between(male.index, 0, male.values, alpha=0.5, color=MALE_COL, label="Male")
    ax1.fill_between(male.index, male.values, male.values + female.values,
                     alpha=0.5, color=FEMALE_COL, label="Female")
    ax1.plot(male.index, male.values + female.values, color=TOTAL_COL, linewidth=1, alpha=0.6)
    ax1.set_ylabel("Patients")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax1.legend(loc="upper left")
    add_covid_shading(ax1)
    ax1.set_title("A. ADHD patients by sex", loc="left", fontweight="bold")
    ax1.grid(axis="y", alpha=0.2)
    ax1.set_xticks(male.index[::2])
    ax1.set_xticklabels(fy_labels_for_axis(male.index[::2]), rotation=45)

    # Panel B: M:F ratio
    ax2.plot(ratio.index, ratio.values, "o-", color=TOTAL_COL, markersize=4, linewidth=1.5, zorder=3)
    ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", zorder=2, label="Parity (1:1)")
    ax2.set_ylabel("Male : Female ratio")
    ax2.set_ylim(0.8, 3.5)

    # Annotate start and end
    ax2.annotate(f"{ratio.values[0]:.2f}:1", xy=(ratio.index[0], ratio.values[0]),
                 xytext=(ratio.index[0] + 1.5, ratio.values[0] + 0.15),
                 fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax2.annotate(f"{ratio.values[-1]:.2f}:1", xy=(ratio.index[-1], ratio.values[-1]),
                 xytext=(ratio.index[-1] - 3, ratio.values[-1] + 0.3),
                 fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    add_covid_shading(ax2, label=False)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title("B. Male-to-female ratio", loc="left", fontweight="bold")
    ax2.grid(axis="y", alpha=0.2)
    ax2.set_xticks(ratio.index[::2])
    ax2.set_xticklabels(fy_labels_for_axis(ratio.index[::2]), rotation=45)

    fig.savefig(FIG_DIR / "figure2_sex_decomposition.png")
    fig.savefig(FIG_DIR / "figure2_sex_decomposition.pdf")
    plt.close()
    print("  -> figure2_sex_decomposition.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE S1: Age group decomposition
# ═══════════════════════════════════════════════════════════════════════════════
def figure_s1(df):
    print("Generating Figure S1...")
    age_data = df[
        (df["medication"].str.contains("All ADHD", case=False))
        & (df["measure"] == "Patients")
        & (df["sex"] == "People")
    ]
    pivot = age_data.pivot_table(index="fy_start", columns="age_group",
                                  values="value", aggfunc="sum", fill_value=0)
    # Reorder columns
    col_order = ["0 - 11", "12 - 17", "18 - 24", "25 - 44", "45+"]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.3})

    # Panel A: Stacked area
    colours = [AGE_COLOURS[c] for c in pivot.columns]
    labels = [AGE_LABELS[c] for c in pivot.columns]
    ax1.stackplot(pivot.index, *[pivot[c] for c in pivot.columns],
                  labels=labels, colors=colours, alpha=0.7)
    ax1.set_ylabel("Patients")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("A. ADHD patients by age group", loc="left", fontweight="bold")
    ax1.grid(axis="y", alpha=0.2)
    ax1.set_xticks(pivot.index[::2])
    ax1.set_xticklabels(fy_labels_for_axis(pivot.index[::2]), rotation=45)

    # Panel B: Indexed growth (base year = 100)
    for col in pivot.columns:
        vals = pivot[col]
        first_nonzero = vals[vals > 0].index[0]
        indexed = vals.loc[first_nonzero:] / vals.loc[first_nonzero] * 100
        ax2.plot(indexed.index, indexed.values, "o-", markersize=3, linewidth=1.5,
                 color=AGE_COLOURS[col], label=AGE_LABELS[col])

    ax2.axhline(100, color="black", linewidth=0.5, linestyle=":")
    ax2.set_ylabel("Index (baseline year = 100)")
    ax2.set_title("B. Indexed growth by age group", loc="left", fontweight="bold")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(axis="y", alpha=0.2)
    ax2.set_xticks(pivot.index[::2])
    ax2.set_xticklabels(fy_labels_for_axis(pivot.index[::2]), rotation=45)

    fig.savefig(FIG_DIR / "figure_s1_age_decomposition.png")
    fig.savefig(FIG_DIR / "figure_s1_age_decomposition.pdf")
    plt.close()
    print("  -> figure_s1_age_decomposition.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE S2: Medication-specific trends by sex (5-panel grid)
# ═══════════════════════════════════════════════════════════════════════════════
def figure_s2(df):
    print("Generating Figure S2...")
    meds = [
        ("N06BA02 Dexamfetamine", "Dexamfetamine"),
        ("N06BA04 Methylphenidate", "Methylphenidate"),
        ("N06BA09 Atomoxetine", "Atomoxetine"),
        ("N06BA12 Lisdexamfetamine", "Lisdexamfetamine"),
        ("C02AC02 Guanfacine", "Guanfacine"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    axes_flat = axes.flatten()

    for idx, (med_full, med_short) in enumerate(meds):
        ax = axes_flat[idx]
        male = get_patients(df, medication=med_full, sex="Male")
        female = get_patients(df, medication=med_full, sex="Female")

        ax.plot(male.index, male.values, "o-", markersize=3, linewidth=1.5,
                color=MALE_COL, label="Male")
        ax.plot(female.index, female.values, "s-", markersize=3, linewidth=1.5,
                color=FEMALE_COL, label="Female")

        ax.set_title(med_short, loc="left", fontweight="bold",
                     color=MED_COLOURS[med_short])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks(male.index[::3])
        ax.set_xticklabels(fy_labels_for_axis(male.index[::3]), rotation=45)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplot
    axes_flat[5].set_visible(False)

    fig.suptitle("Supplementary Figure S2: Medication-specific trends by sex",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(FIG_DIR / "figure_s2_medication_trends.png")
    fig.savefig(FIG_DIR / "figure_s2_medication_trends.pdf")
    plt.close()
    print("  -> figure_s2_medication_trends.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE S3: Prescribing rate per 1,000 population
# ═══════════════════════════════════════════════════════════════════════════════
def figure_s3(df):
    print("Generating Figure S3...")
    rate_data = df[
        (df["medication"].str.contains("All ADHD", case=False))
        & (df["measure"] == "Patient rate per 1,000 population")
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for sex, colour, marker in [("Male", MALE_COL, "o"), ("Female", FEMALE_COL, "s"),
                                 ("People", TOTAL_COL, "D")]:
        sub = rate_data[rate_data["sex"] == sex]
        agg = sub.groupby("fy_start")["value"].mean().sort_index()
        label = sex if sex != "People" else "Total"
        ax.plot(agg.index, agg.values, f"{marker}-", markersize=4, linewidth=1.5,
                color=colour, label=label)

    add_covid_shading(ax)
    ax.set_ylabel("Patients per 1,000 population")
    ax.set_title("Supplementary Figure S3: ADHD prescribing rate per 1,000 population",
                 loc="left", fontweight="bold", fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_xticks(agg.index[::2])
    ax.set_xticklabels(fy_labels_for_axis(agg.index[::2]), rotation=45)

    fig.savefig(FIG_DIR / "figure_s3_rates.png")
    fig.savefig(FIG_DIR / "figure_s3_rates.pdf")
    plt.close()
    print("  -> figure_s3_rates.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Condensed growth summary (main text)
# ═══════════════════════════════════════════════════════════════════════════════
def table1(df):
    """Compute growth statistics directly from the parquet data."""
    print("Generating Table 1...")

    med_order = [
        ("All ADHD medications", "All ADHD medications"),
        ("N06BA02 Dexamfetamine", "Dexamfetamine"),
        ("N06BA04 Methylphenidate", "Methylphenidate"),
        ("N06BA09 Atomoxetine", "Atomoxetine"),
        ("N06BA12 Lisdexamfetamine", "Lisdexamfetamine"),
        ("C02AC02 Guanfacine", "Guanfacine"),
    ]

    rows = []
    for med_full, med_short in med_order:
        for sex in ["People", "Male", "Female"]:
            ts = get_patients(df, medication=med_full, sex=sex)
            if len(ts) < 2:
                continue

            first_val = ts.iloc[0]
            last_val = ts.iloc[-1]
            first_yr = ts.index[0]
            last_yr = ts.index[-1]
            n_years = last_yr - first_yr

            if first_val <= 0 or n_years <= 0:
                continue

            total_growth = (last_val / first_val - 1) * 100
            cagr = ((last_val / first_val) ** (1 / n_years) - 1) * 100

            recent = ts[ts.index >= 2019]
            recent_growth = ((recent.iloc[-1] / recent.iloc[0]) - 1) * 100 if len(recent) >= 2 else np.nan

            sex_label = "Total" if sex == "People" else sex
            rows.append({
                "Medication": med_short,
                "Sex": sex_label,
                "Period": f"{fy_label(first_yr)} to {fy_label(last_yr)}",
                "Patients (first year)": f"{first_val:,.0f}",
                "Patients (last year)": f"{last_val:,.0f}",
                "Total growth (%)": f"{total_growth:,.1f}",
                "CAGR (%)": f"{cagr:.1f}",
                "5-year growth (%)": f"{recent_growth:.1f}",
            })

    table = pd.DataFrame(rows)
    table.to_csv(TAB_DIR / "table1_growth_summary.csv", index=False)

    with open(TAB_DIR / "table1_growth_summary.md", "w") as f:
        f.write("# Table 1. Growth in ADHD medication recipients in Australia by medication and sex\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n\nCAGR = compound annual growth rate. 5-year growth = 2019–20 to 2023–24.\n")
        f.write("Note: 'All ADHD medications' counts are de-duplicated; individual medication totals may sum to more than the all-medication total.\n")

    print("  -> table1_growth_summary.csv/md")
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE S1: Full annual patient counts
# ═══════════════════════════════════════════════════════════════════════════════
def table_s1(df):
    print("Generating Table S1...")
    patients = df[
        (df["measure"] == "Patients")
        & (df["medication"].str.contains("All ADHD", case=False))
    ]

    # By year, sex, age group
    pivot = patients.pivot_table(
        index=["fy_start", "sex"],
        columns="age_group",
        values="value",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    # Add total column
    age_cols = [c for c in pivot.columns if c not in ("fy_start", "sex")]
    pivot["Total"] = pivot[age_cols].sum(axis=1)

    # Financial year label
    pivot["Financial year"] = pivot["fy_start"].apply(fy_label)
    pivot = pivot.drop(columns=["fy_start"])

    # Reorder columns
    col_order = ["Financial year", "sex"] + sorted(age_cols) + ["Total"]
    pivot = pivot[col_order].rename(columns={"sex": "Sex"})
    pivot = pivot.sort_values(["Financial year", "Sex"])

    pivot.to_csv(TAB_DIR / "table_s1_annual_counts.csv", index=False)
    print("  -> table_s1_annual_counts.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE S2: Male:Female ratio by year and medication
# ═══════════════════════════════════════════════════════════════════════════════
def table_s2(df):
    print("Generating Table S2...")
    meds = ["All ADHD medications", "N06BA02 Dexamfetamine", "N06BA04 Methylphenidate",
            "N06BA09 Atomoxetine", "N06BA12 Lisdexamfetamine", "C02AC02 Guanfacine"]
    med_short = {
        "All ADHD medications": "All ADHD",
        "N06BA02 Dexamfetamine": "Dexamfetamine",
        "N06BA04 Methylphenidate": "Methylphenidate",
        "N06BA09 Atomoxetine": "Atomoxetine",
        "N06BA12 Lisdexamfetamine": "Lisdexamfetamine",
        "C02AC02 Guanfacine": "Guanfacine",
    }

    rows = []
    for med in meds:
        male = get_patients(df, medication=med, sex="Male")
        female = get_patients(df, medication=med, sex="Female")
        common_idx = male.index.intersection(female.index)
        for yr in common_idx:
            m_val = male.loc[yr]
            f_val = female.loc[yr]
            ratio = m_val / f_val if f_val > 0 else np.nan
            rows.append({
                "Medication": med_short[med],
                "Financial year": fy_label(yr),
                "Male patients": int(m_val),
                "Female patients": int(f_val),
                "M:F ratio": f"{ratio:.2f}",
            })

    table = pd.DataFrame(rows)
    table.to_csv(TAB_DIR / "table_s2_sex_ratios.csv", index=False)
    print("  -> table_s2_sex_ratios.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Project 63 — Publication Figure & Table Generation")
    print("=" * 60)

    df = load_data()
    print(f"Loaded {len(df)} rows\n")

    # Main text
    figure1(df)
    figure2(df)
    table1(df)

    # Supplementary
    figure_s1(df)
    figure_s2(df)
    figure_s3(df)
    table_s1(df)
    table_s2(df)

    print("\n" + "=" * 60)
    print("All outputs written to:", FIG_DIR.parent)
    print("=" * 60)


if __name__ == "__main__":
    main()
