import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD & INSPECT ────────────────────────────────────────────────────────

df = pd.read_csv("poulation.csv", encoding="utf-8")

# Rename Arabic columns to English for easier coding
df.columns = ["Nationality", "Year", "Region", "Gender", "Population"]

print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nFirst rows:")
print(df.head())

# ── 2. BASIC CLEANING ────────────────────────────────────────────────────────

# Check for nulls
null_counts = df.isnull().sum()
print("\nNull values per column:\n", null_counts)

# Check for duplicates
dup_count = df.duplicated().sum()
print(f"\nDuplicate rows: {dup_count}")

# Ensure Population is numeric
df["Population"] = pd.to_numeric(df["Population"], errors="coerce")

# Drop any rows where Population couldn't be parsed
df.dropna(subset=["Population"], inplace=True)

print(f"\nYear range: {df['Year'].min()} – {df['Year'].max()}")
print(f"Regions ({df['Region'].nunique()}):\n", df["Region"].unique())
print(f"\nNationality categories: {df['Nationality'].unique()}")
print(f"Gender categories:      {df['Gender'].unique()}")

# ── 3. AGGREGATE: Total population per Region per Year ───────────────────────
# (sum across Nationality and Gender — we want total regional population)

region_year = (
    df.groupby(["Region", "Year"], as_index=False)["Population"]
    .sum()
    .rename(columns={"Population": "Total_Population"})
)

print("\nAggregated shape:", region_year.shape)
print(region_year.head(10))

# ── 4. EXPLORATORY DATA ANALYSIS ─────────────────────────────────────────────

regions = region_year["Region"].unique()

# 4a. Population trend per region
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, region in enumerate(regions):
    sub = region_year[region_year["Region"] == region]
    axes[i].plot(sub["Year"], sub["Total_Population"] / 1e6, marker="o", linewidth=2)
    axes[i].set_title(region, fontsize=10)
    axes[i].set_xlabel("Year")
    axes[i].set_ylabel("Population (M)")
    axes[i].grid(True, alpha=0.3)

# Hide unused subplots
for j in range(len(regions), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Population Trend by Region (2010–2022)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_population_trends.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved: eda_population_trends.png")

# 4b. Saudi vs Non-Saudi breakdown
nat_year = (
    df.groupby(["Nationality", "Year"], as_index=False)["Population"]
    .sum()
)

plt.figure(figsize=(10, 5))
for nat, grp in nat_year.groupby("Nationality"):
    plt.plot(grp["Year"], grp["Population"] / 1e6, marker="o", label=nat, linewidth=2)
plt.title("Saudi vs Non-Saudi Population (All Regions Combined)")
plt.xlabel("Year")
plt.ylabel("Population (Millions)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("eda_nationality_breakdown.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: eda_nationality_breakdown.png")

# 4c. Gender breakdown
gender_year = (
    df.groupby(["Gender", "Year"], as_index=False)["Population"]
    .sum()
)

plt.figure(figsize=(10, 5))
for gender, grp in gender_year.groupby("Gender"):
    plt.plot(grp["Year"], grp["Population"] / 1e6, marker="o", label=gender, linewidth=2)
plt.title("Male vs Female Population (All Regions Combined)")
plt.xlabel("Year")
plt.ylabel("Population (Millions)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("eda_gender_breakdown.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: eda_gender_breakdown.png")

# 4d. 2022 population bar chart by region
pop_2022 = region_year[region_year["Year"] == 2022].sort_values("Total_Population", ascending=True)
plt.figure(figsize=(12, 7))
plt.barh(pop_2022["Region"], pop_2022["Total_Population"] / 1e6, color="steelblue")
plt.xlabel("Population (Millions)")
plt.title("Total Population by Region – 2022")
plt.tight_layout()
plt.savefig("eda_population_2022.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: eda_population_2022.png")

# ── 5. FEATURE ENGINEERING ───────────────────────────────────────────────────

# Encode region as a numeric label for ML models
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
region_year["Region_encoded"] = le.fit_transform(region_year["Region"])

# Add growth-rate feature (year-over-year % change per region)
region_year = region_year.sort_values(["Region", "Year"])
region_year["Population_lag1"] = region_year.groupby("Region")["Total_Population"].shift(1)
region_year["Growth_rate"] = (
    (region_year["Total_Population"] - region_year["Population_lag1"])
    / region_year["Population_lag1"]
) * 100

# Fill NaN growth rate for first year of each region with the region's mean growth
region_year["Growth_rate"] = region_year.groupby("Region")["Growth_rate"].transform(
    lambda x: x.fillna(x.mean())
)

print("\nFeature-engineered dataset:")
print(region_year.head(15))

# ── 6. POPULATION FORECASTING (2023–2030) ────────────────────────────────────

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    print("\nNote: xgboost not installed. Run: pip install xgboost")
    HAS_XGB = False

future_years_list = [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]

# Train one model per region (using Year as the only feature for forecasting)
# For full ML pipeline (with bed data), Region_encoded + Growth_rate will be used too.

all_preds = []

print("\n── Per-region model evaluation (train 2010–2019, test 2020–2022) ──\n")

for region in regions:
    sub = region_year[region_year["Region"] == region].copy()

    X = sub[["Year"]]
    y = sub["Total_Population"]

    # Train/test split: train on 2010–2019, test on 2020–2022
    X_train = sub[sub["Year"] <= 2019][["Year"]]
    y_train = sub[sub["Year"] <= 2019]["Total_Population"]
    X_test  = sub[sub["Year"] >= 2020][["Year"]]
    y_test  = sub[sub["Year"] >= 2020]["Total_Population"]

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred_test = lr.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred_test))
    lr_r2   = r2_score(y_test, lr_pred_test)

    # --- Ridge Regression ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred_test = ridge.predict(X_test)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred_test))
    ridge_r2   = r2_score(y_test, ridge_pred_test)

    print(f"{region}:")
    print(f"  Linear Regression → RMSE: {lr_rmse:,.0f}  | R²: {lr_r2:.4f}")
    print(f"  Ridge Regression  → RMSE: {ridge_rmse:,.0f}  | R²: {ridge_r2:.4f}")

    # XGBoost (if available) — needs more data to shine, but include for comparison
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=50, max_depth=3, random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        xgb_pred_test = xgb.predict(X_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred_test))
        xgb_r2   = r2_score(y_test, xgb_pred_test)
        print(f"  XGBoost           → RMSE: {xgb_rmse:,.0f}  | R²: {xgb_r2:.4f}")

    # Use LinearRegression (best for smooth trends) to forecast future years
    lr_full = LinearRegression()
    lr_full.fit(X, y)

    future_X = pd.DataFrame({"Year": future_years_list})
    preds = lr_full.predict(future_X)

    future_df = pd.DataFrame({
        "Region":               region,
        "Year":                 future_years_list,
        "Predicted_Population": preds.astype(int)
    })
    all_preds.append(future_df)

future_population = pd.concat(all_preds, ignore_index=True)

print("\n── Future Population Predictions (Linear Regression) ──")
print(future_population.pivot(index="Region", columns="Year", values="Predicted_Population").to_string())

# ── 7. VISUALIZE FORECASTS ───────────────────────────────────────────────────

fig, axes = plt.subplots(4, 4, figsize=(22, 18))
axes = axes.flatten()

for i, region in enumerate(regions):
    hist = region_year[region_year["Region"] == region]
    fcast = future_population[future_population["Region"] == region]

    axes[i].plot(hist["Year"], hist["Total_Population"] / 1e6,
                 marker="o", label="Historical", linewidth=2, color="steelblue")
    axes[i].plot(fcast["Year"], fcast["Predicted_Population"] / 1e6,
                 marker="s", linestyle="--", label="Forecast", linewidth=2, color="tomato")
    axes[i].axvline(2022.5, color="gray", linestyle=":", alpha=0.7)
    axes[i].set_title(region, fontsize=9)
    axes[i].set_xlabel("Year", fontsize=8)
    axes[i].set_ylabel("Population (M)", fontsize=8)
    axes[i].legend(fontsize=7)
    axes[i].grid(True, alpha=0.3)

for j in range(len(regions), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Population Forecast per Region (2023–2030)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("forecast_population_per_region.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved: forecast_population_per_region.png")

# ── 8. EXPORT CLEANED + FORECAST DATA ────────────────────────────────────────

# Save cleaned historical data
region_year.to_csv("population_cleaned.csv", index=False, encoding="utf-8-sig")
print("Saved: population_cleaned.csv")

# Save forecasts
future_population.to_csv("population_forecast_2023_2030.csv", index=False, encoding="utf-8-sig")
print("Saved: population_forecast_2023_2030.csv")

print("\n✓ Preprocessing complete.")
print("Next step: merge population_forecast_2023_2030.csv with hospital bed datasets")
print("           and train the bed-demand prediction model.")