"""
ARTI 406 – Machine Learning Project
Predicting Hospital Bed Demand Based on Population Growth in Saudi Arabia
Group 6

population_pre.py  –  Data Loading, Cleaning, EDA, and Population Forecasting
===============================================================================

Datasets:
  1. Population_2010_-_2022.csv
  2. Number_of_hospital_beds_per_1_000_population.csv
  3. Hospital_beds_in_the_government_sector_by_...specialty.csv
  4. Number_of_hospitals_per_10_000_population.csv

Population forecasting method: Linear Regression (per region, Year → Population)
  - Justified: 13-year trends are smooth and near-linear; simple and interpretable
  - Train on 2010–2019, evaluate on 2020–2022, then forecast 2023–2030

Model comparison (for population step):
  - LinearRegression  → strong baseline, interpretable
  - Ridge             → handles slight multicollinearity
  - XGBoost           → cited in literature, best on tabular data

Output files:
  - population_cleaned.csv
  - population_forecast_2023_2030.csv
  - merged_dataset.csv   (population + all bed metrics, ready for bed demand model)
  - model_evaluation_population.csv
  - Several EDA and forecast plots (*.png)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – LOAD ALL DATASETS
# ─────────────────────────────────────────────────────────────────────────────

def load(path):
    """Load a utf-16 tab-separated CSV, strip whitespace from column names."""
    df = pd.read_csv(path, encoding="utf-16", sep="\t")
    df.columns = df.columns.str.strip()
    return df
pop_raw        = load("DataSets/Population 2010 - 2022.csv")
beds_per_1000  = load("DataSets/Number of hospital beds per 1,000 population.csv")
beds_specialty = load("DataSets/Hospital beds in the government sector by administrative region and specialty.csv")
hosp_per_10000 = load("DataSets/Number of hospitals per 10,000 population.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
for name, df in [("Population", pop_raw), ("Beds/1000", beds_per_1000),
                 ("Beds by Specialty", beds_specialty), ("Hospitals/10000", hosp_per_10000)]:
    print(f"\n{name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print("  Columns:", list(df.columns))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – CLEAN POPULATION DATA
# ─────────────────────────────────────────────────────────────────────────────

# Drop the "Total" summary row
pop = pop_raw[pop_raw["Region"].str.strip() != "Total"].copy()

# Year columns contain comma-formatted numbers (e.g. "6,224,033") → convert to int
year_cols = [str(y) for y in range(2010, 2023)]
for col in year_cols:
    pop[col] = pop[col].astype(str).str.replace(",", "").str.strip()
    pop[col] = pd.to_numeric(pop[col], errors="coerce")

pop["Region"] = pop["Region"].str.strip()

print("\n\nPOPULATION DATA (cleaned wide format):")
print(pop.to_string(index=False))
print("\nNull values:", pop.isnull().sum().sum())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – RESHAPE TO LONG FORMAT (for ML)
# ─────────────────────────────────────────────────────────────────────────────

pop_long = pop.melt(id_vars="Region", value_vars=year_cols,
                    var_name="Year", value_name="Population")
pop_long["Year"] = pop_long["Year"].astype(int)
pop_long = pop_long.sort_values(["Region", "Year"]).reset_index(drop=True)

print("\n\nPOPULATION LONG FORMAT (first 20 rows):")
print(pop_long.head(20).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – CLEAN HEALTHCARE DATASETS
# ─────────────────────────────────────────────────────────────────────────────

# Standardise region names to match population dataset
REGION_MAP = {
    "Riyadh":           "Ar Riyadh",
    "Makkah":           "Makkah Al Mukarramah",
    "Madinah":          "Al Madinah Al Munawwarah",
    "Qassim":           "Al Qaseem",
    "Eastern Region":   "Eastern Region",
    "Aseer":            "Aseer",
    "Tabuk":            "Tabuk",
    "Hail":             "Hail",
    "Northern Borders": "Northern Borders",
    "Jazan":            "Jazan",
    "Najran":           "Najran",
    "Al-Baha":          "Al Bahah",
    "Al-Jouf":          "Al Jawf",
}

def clean_healthcare_df(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Administrative Area"] = df["Administrative Area"].str.strip()
    df = df[~df["Administrative Area"].isin(["Total", "Other"])]
    df["Region"] = df["Administrative Area"].map(REGION_MAP)
    unmapped = df[df["Region"].isna()]["Administrative Area"].unique()
    if len(unmapped):
        print(f"  Warning - unmapped regions: {unmapped}")
    df = df.dropna(subset=["Region"])
    return df

# Beds per 1,000
b1000 = clean_healthcare_df(beds_per_1000)
b1000["Beds_per_1000"] = pd.to_numeric(b1000["Number"], errors="coerce")
b1000 = b1000[["Region", "Beds_per_1000"]]
print("\n\nBEDS PER 1,000 POPULATION (cleaned):")
print(b1000.to_string(index=False))

# Hospitals per 10,000
h10000 = clean_healthcare_df(hosp_per_10000)
h10000["Hospitals_per_10000"] = pd.to_numeric(h10000["Number"], errors="coerce")
h10000 = h10000[["Region", "Hospitals_per_10000"]]
print("\n\nHOSPITALS PER 10,000 POPULATION (cleaned):")
print(h10000.to_string(index=False))

# Beds by specialty — some cells contain "-" (zero beds)
spec = clean_healthcare_df(beds_specialty)
spec_num_cols = [
    "Internal", "surgical", "Orthopedics", "Urology", "Oral and Dental",
    "Obstetrics and gynecology", "Pediatrics", "Intensive care",
    "Otorhinolaryngology (ENT)", "Springs", "Chest/Respiratory Diseases",
    "Dermatology and Venereology", "Burns and Plastic Surgery",
    "Psychiatry and Neurology", "Isolation", "Other", "Total"
]
for col in spec_num_cols:
    spec[col] = (spec[col].astype(str)
                           .str.replace(",", "")
                           .str.strip()
                           .replace("-", "0"))
    spec[col] = pd.to_numeric(spec[col], errors="coerce").fillna(0).astype(int)

spec = spec[["Region"] + spec_num_cols].rename(columns={"Total": "Total_Beds"})
print("\n\nBEDS BY SPECIALTY (cleaned, first 5 rows):")
print(spec.head().to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – FEATURE ENGINEERING ON POPULATION
# ─────────────────────────────────────────────────────────────────────────────

pop_long = pop_long.sort_values(["Region", "Year"])

# Year-over-year absolute change
pop_long["Pop_change"] = pop_long.groupby("Region")["Population"].diff()

# Year-over-year growth rate (%)
pop_long["Growth_rate_pct"] = (
    pop_long.groupby("Region")["Population"].pct_change() * 100
)

# Cumulative growth since 2010
base = pop_long[pop_long["Year"] == 2010][["Region", "Population"]].rename(
    columns={"Population": "Pop_2010"}
)
pop_long = pop_long.merge(base, on="Region", how="left")
pop_long["Cumulative_growth_pct"] = (
    (pop_long["Population"] - pop_long["Pop_2010"]) / pop_long["Pop_2010"] * 100
)

print("\n\nPOPULATION WITH ENGINEERED FEATURES (first 15 rows):")
print(pop_long.head(15).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – EDA VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

regions = sorted(pop_long["Region"].unique())
palette = sns.color_palette("tab10", len(regions))

# 6a. Population trend per region (grid)
fig, axes = plt.subplots(4, 4, figsize=(22, 16))
axes = axes.flatten()
for i, region in enumerate(regions):
    sub = pop_long[pop_long["Region"] == region]
    axes[i].plot(sub["Year"], sub["Population"] / 1e6, marker="o",
                 linewidth=2, color=palette[i])
    axes[i].set_title(region, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("Year", fontsize=8)
    axes[i].set_ylabel("Population (M)", fontsize=8)
    axes[i].grid(True, alpha=0.3)
    axes[i].tick_params(labelsize=7)
for j in range(len(regions), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Population Trend by Region (2010-2022)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_population_trends.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved: eda_population_trends.png")

# 6b. All regions overlaid
plt.figure(figsize=(13, 6))
for i, region in enumerate(regions):
    sub = pop_long[pop_long["Region"] == region]
    plt.plot(sub["Year"], sub["Population"] / 1e6, marker="o",
             linewidth=1.8, label=region, color=palette[i])
plt.title("Population Growth - All Regions (2010-2022)", fontsize=13, fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Population (Millions)")
plt.legend(fontsize=7, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("eda_all_regions_overlay.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: eda_all_regions_overlay.png")

# 6c. 2022 population bar chart
pop_2022 = pop_long[pop_long["Year"] == 2022].sort_values("Population")
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(pop_2022["Region"], pop_2022["Population"] / 1e6,
               color=sns.color_palette("Blues_d", len(pop_2022)))
ax.set_xlabel("Population (Millions)", fontsize=11)
ax.set_title("Total Population by Region - 2022", fontsize=13, fontweight="bold")
for bar, val in zip(bars, pop_2022["Population"] / 1e6):
    ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}M", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("eda_population_2022_bar.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: eda_population_2022_bar.png")

# 6d. Growth rate heatmap
growth_pivot = pop_long.pivot(index="Region", columns="Year", values="Growth_rate_pct")
plt.figure(figsize=(14, 7))
sns.heatmap(growth_pivot, annot=True, fmt=".1f", cmap="RdYlGn",
            linewidths=0.5, center=0, cbar_kws={"label": "YoY Growth (%)"})
plt.title("Year-over-Year Population Growth Rate (%) by Region", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_growth_rate_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: eda_growth_rate_heatmap.png")

# 6e. Beds per 1000 vs hospitals per 10000 scatter
merged_health = b1000.merge(h10000, on="Region")
plt.figure(figsize=(10, 6))
for _, row in merged_health.iterrows():
    plt.scatter(row["Hospitals_per_10000"], row["Beds_per_1000"], s=100, zorder=3)
    plt.annotate(row["Region"], (row["Hospitals_per_10000"], row["Beds_per_1000"]),
                 fontsize=7, ha="left", xytext=(4, 2), textcoords="offset points")
plt.xlabel("Hospitals per 10,000 Population")
plt.ylabel("Beds per 1,000 Population")
plt.title("Healthcare Infrastructure: Hospitals vs Beds per Capita", fontsize=12, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("eda_beds_vs_hospitals.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: eda_beds_vs_hospitals.png")

# 6f. Specialty bed distribution stacked bar
spec_plot = spec.set_index("Region")
specialty_cols = [c for c in spec_num_cols if c not in ["Total", "Other", "Total_Beds"]]
spec_plot[specialty_cols].plot(kind="bar", stacked=True, figsize=(14, 7), colormap="tab20")
plt.title("Hospital Beds by Specialty per Region (Government Sector)", fontsize=12, fontweight="bold")
plt.xlabel("Region")
plt.ylabel("Number of Beds")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.legend(loc="upper right", fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig("eda_specialty_beds_stacked.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: eda_specialty_beds_stacked.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – POPULATION FORECASTING (2023–2030)
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    print("\nNote: xgboost not installed (pip install xgboost). Skipping XGBoost.")
    HAS_XGB = False

FUTURE_YEARS  = list(range(2023, 2031))
TRAIN_CUTOFF  = 2019  # train 2010-2019, test 2020-2022

results_rows  = []
all_forecasts = []

print("\n\n" + "=" * 60)
print("POPULATION FORECASTING - MODEL EVALUATION PER REGION")
print(f"  Train: 2010-{TRAIN_CUTOFF}  |  Test: {TRAIN_CUTOFF+1}-2022")
print("=" * 60)

for region in regions:
    sub     = pop_long[pop_long["Region"] == region].copy()
    X_all   = sub[["Year"]]
    y_all   = sub["Population"]
    X_train = sub[sub["Year"] <= TRAIN_CUTOFF][["Year"]]
    y_train = sub[sub["Year"] <= TRAIN_CUTOFF]["Population"]
    X_test  = sub[sub["Year"] > TRAIN_CUTOFF][["Year"]]
    y_test  = sub[sub["Year"] > TRAIN_CUTOFF]["Population"]

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=50, max_depth=2,
                                          learning_rate=0.1, random_state=42,
                                          verbosity=0)

    print(f"\n{region}:")
    best_rmse  = np.inf
    best_model = None

    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        mape  = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100
        r2    = r2_score(y_test, preds)
        print(f"  {name:<22} RMSE={rmse:>10,.0f}  MAE={mae:>10,.0f}  "
              f"MAPE={mape:>5.2f}%  R2={r2:.4f}")
        results_rows.append({"Region": region, "Model": name,
                              "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2})
        if rmse < best_rmse:
            best_rmse  = rmse
            best_model = name

    print(f"  --> Best model on test set: {best_model}")

    # Refit Linear Regression on ALL data for forecasting
    # (LinearRegression is best for smooth monotonic trends like population)
    final_model = LinearRegression()
    final_model.fit(X_all, y_all)
    future_X     = pd.DataFrame({"Year": FUTURE_YEARS})
    preds_future = final_model.predict(future_X).astype(int)

    for yr, pred in zip(FUTURE_YEARS, preds_future):
        all_forecasts.append({"Region": region, "Year": yr,
                               "Predicted_Population": max(pred, 0)})

# Summary
eval_df = pd.DataFrame(results_rows)
print("\n\nSUMMARY - Average metrics across all regions:")
print(eval_df.groupby("Model")[["RMSE", "MAE", "MAPE", "R2"]].mean().round(3).to_string())

future_pop = pd.DataFrame(all_forecasts)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – FORECAST VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 4, figsize=(22, 18))
axes = axes.flatten()

for i, region in enumerate(regions):
    hist  = pop_long[pop_long["Region"] == region]
    fcast = future_pop[future_pop["Region"] == region]

    axes[i].plot(hist["Year"], hist["Population"] / 1e6,
                 marker="o", linewidth=2, label="Historical", color="steelblue")
    axes[i].plot(fcast["Year"], fcast["Predicted_Population"] / 1e6,
                 marker="s", linestyle="--", linewidth=2, label="Forecast", color="tomato")
    axes[i].axvspan(2022.5, 2030.5, alpha=0.05, color="tomato")
    axes[i].axvline(2022.5, color="gray", linestyle=":", linewidth=1)
    axes[i].set_title(region, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("Year", fontsize=8)
    axes[i].set_ylabel("Population (M)", fontsize=8)
    axes[i].legend(fontsize=7)
    axes[i].grid(True, alpha=0.3)
    axes[i].tick_params(labelsize=7)

for j in range(len(regions), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Population Forecast per Region - Linear Regression (2023-2030)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("forecast_population_per_region.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved: forecast_population_per_region.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – BUILD MERGED DATASET (ready for bed demand model)
# ─────────────────────────────────────────────────────────────────────────────

# Combine historical + forecast population into one table
hist_pop           = pop_long[["Region", "Year", "Population"]].copy()
hist_pop["Is_forecast"] = False

fcast_pop          = future_pop.rename(columns={"Predicted_Population": "Population"}).copy()
fcast_pop["Is_forecast"] = True

full_pop = pd.concat([hist_pop, fcast_pop], ignore_index=True)

# Merge in bed / hospital metrics (snapshot data used as baseline for all years)
merged = full_pop.merge(b1000,  on="Region", how="left")
merged = merged.merge(h10000,   on="Region", how="left")
merged = merged.merge(
    spec[["Region", "Total_Beds", "Intensive care", "Pediatrics",
          "Obstetrics and gynecology", "Psychiatry and Neurology"]],
    on="Region", how="left"
)

# Derived demand estimates
merged["Beds_needed_estimate"]    = (merged["Population"] / 1000) * merged["Beds_per_1000"]
merged["Hospitals_estimate"]      = (merged["Population"] / 10000) * merged["Hospitals_per_10000"]

merged = merged.sort_values(["Region", "Year"]).reset_index(drop=True)

print("\n\nMERGED DATASET (first 10 rows):")
print(merged.head(10).to_string(index=False))
print(f"\nFull shape: {merged.shape}")
print("\nColumns:", list(merged.columns))
print("\nNull values per column:")
print(merged.isnull().sum())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – EXPORT ALL OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

pop_long.to_csv("population_cleaned.csv",            index=False, encoding="utf-8-sig")
future_pop.to_csv("population_forecast_2023_2030.csv", index=False, encoding="utf-8-sig")
merged.to_csv("merged_dataset.csv",                  index=False, encoding="utf-8-sig")
eval_df.to_csv("model_evaluation_population.csv",    index=False, encoding="utf-8-sig")

print("\n\n" + "=" * 60)
print("DONE. Output files:")
print("  population_cleaned.csv")
print("  population_forecast_2023_2030.csv")
print("  merged_dataset.csv            <-- use this for the bed demand model")
print("  model_evaluation_population.csv")
print("  eda_population_trends.png")
print("  eda_all_regions_overlay.png")
print("  eda_population_2022_bar.png")
print("  eda_growth_rate_heatmap.png")
print("  eda_beds_vs_hospitals.png")
print("  eda_specialty_beds_stacked.png")
print("  forecast_population_per_region.png")
print("=" * 60)
print("\nNext step: use merged_dataset.csv to train the bed demand prediction model.")