# Closing the Gap: Twenty Years of ADHD Medication Prescribing by Sex in Australia, with International Comparison

Reproducibility repository for:

> Farquhar H. Closing the Gap: Twenty Years of ADHD Medication Prescribing by Sex in Australia, with International Comparison. [Journal TBD]. 2026.

## Data

- `data/adhd_annual_national.parquet` — National PBS dispensing data for ADHD medications (2004–2024), sourced from the [Australian Institute of Health and Welfare](https://www.aihw.gov.au/). Contains annual counts of unique patients dispensed each medication, stratified by sex, age group, and medication (ATC code). Processed from the publicly available AIHW PBS data tables as part of a registered parent study (OSF: [doi:10.17605/OSF.IO/QPZAJ](https://doi.org/10.17605/OSF.IO/QPZAJ)).

- `data/international/comparison_data.csv` — Extracted comparison data from Denmark (Grøntved et al. 2025, *Acta Psychiatr Scand*) and the United Kingdom (Renoux et al. 2016, *Br J Clin Pharmacol*), used for the indexed growth comparison in Figure 3.

## Scripts

Both scripts are self-contained and produce all manuscript figures and tables from the data files above.

### `scripts/generate_figures.py`

Generates all Australian analysis outputs:
- **Figure 1:** Overall prescribing trajectory with 3-year projection and prediction intervals
- **Figure 2:** Sex decomposition (stacked area + male-to-female ratio)
- **Table 1:** Growth summary by medication and sex (CSV + markdown)
- **Figure S1:** Age group decomposition
- **Figure S2:** Medication-specific trends by sex
- **Figure S3:** Prescribing rate per 1,000 population
- **Tables S1–S2:** Annual counts and sex ratios

### `scripts/international_comparison.py`

Generates the international comparison outputs:
- **Figure 3:** Indexed growth comparison (Australia, Denmark, UK)
- **Table 2:** Cross-national comparison summary

## Usage

```bash
pip install -r requirements.txt
python scripts/generate_figures.py
python scripts/international_comparison.py
```

Outputs are written to `outputs/figures/` and `outputs/tables/`.

## Requirements

Python 3.10+. See `requirements.txt`.

## Pre-registration

This analysis was pre-specified as a secondary aim of a registered parent study: [OSF doi:10.17605/OSF.IO/QPZAJ](https://doi.org/10.17605/OSF.IO/QPZAJ).

## License

Data are derived from publicly available AIHW sources. Code is released under the MIT License.
