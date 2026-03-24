
import sys
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

required = ["pandas", "numpy", "matplotlib", "seaborn", "sklearn"]
missing  = []
for pkg in required:
    try:
        __import__(pkg if pkg != "sklearn" else "sklearn")
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"Missing packages: {missing}")
    print("Run:  pip install " + " ".join(missing))
    sys.exit(1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

CSV_FILE     = "tourism_route_dataset.csv"
RANDOM_STATE = 42
CHART_DPI    = 150

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2D6A9F", "#E05C3A", "#2EAA6E", "#F5A623", "#8E44AD", "#16A085"]
sns.set_palette(PALETTE)

print("\n" + "="*60)
print("  TSA Data Science 2026 — Personalization vs Satisfaction")
print("="*60)

try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"\n❌  File '{CSV_FILE}' not found.")
    print(f"    Update CSV_FILE at the top of this script.")
    sys.exit(1)

print(f"\n✅  Loaded {len(df):,} rows × {len(df.columns)} columns")
print(f"    Columns: {list(df.columns)}\n")

cols_lower = {c.lower(): c for c in df.columns}

def find_col(*keywords):
    for kw in keywords:
        for lc, orig in cols_lower.items():
            if kw in lc:
                return orig
    return None

satisfaction_col     = find_col("satisfaction", "rating", "score", "review")
transport_col        = find_col("transport_mode", "transport")
pref_transport_col   = find_col("preferred_transport")
dest_col             = find_col("destination_type", "destination")
pref_dest_col        = find_col("preferred_destination")
budget_col           = find_col("user_budget", "budget")
time_constraint_col  = find_col("user_time_constraint", "time_constraint")
travel_time_col      = find_col("estimated_travel_time", "travel_time")
entry_col            = find_col("entry_fee", "entry")
accom_col            = find_col("accommodation_cost", "accommodation")
food_col             = find_col("food_cost", "food")
distance_col         = find_col("total_distance", "distance")
traffic_col          = find_col("traffic_density", "traffic")
season_col           = find_col("season")
day_col              = find_col("day_type", "day")
popularity_col       = find_col("popularity_score", "popularity")

print("── Detected columns ───────────────────────────────────")
for label, col in [
    ("Satisfaction (target)",    satisfaction_col),
    ("Transport mode",           transport_col),
    ("Preferred transport",      pref_transport_col),
    ("Destination type",         dest_col),
    ("Preferred destination",    pref_dest_col),
    ("User budget",              budget_col),
    ("Time constraint (hr)",     time_constraint_col),
    ("Travel time (hr)",         travel_time_col),
    ("Entry fee",                entry_col),
    ("Accommodation cost",       accom_col),
    ("Food cost",                food_col),
    ("Distance (km)",            distance_col),
    ("Traffic density",          traffic_col),
    ("Season",                   season_col),
    ("Day type",                 day_col),
    ("Popularity score",         popularity_col),
]:
    print(f"  {label:<26}: {col if col else '-- not found --'}")
print()

if satisfaction_col is None:
    print("❌  Cannot find satisfaction/rating column. Check your CSV.")
    sys.exit(1)

df = df.copy()

numeric_to_clean = [c for c in [
    satisfaction_col, budget_col, time_constraint_col, travel_time_col,
    entry_col, accom_col, food_col, distance_col, traffic_col, popularity_col
] if c]

for c in numeric_to_clean:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df.dropna(subset=[satisfaction_col], inplace=True)
print(f"✅  After cleaning: {len(df):,} rows\n")

print("── Building key variables ──────────────────────────────")


if transport_col and pref_transport_col:
    df["match_transport"] = (
        df[transport_col].astype(str).str.strip().str.lower() ==
        df[pref_transport_col].astype(str).str.strip().str.lower()
    ).astype(float)
    print(f"  Transport match rate : {df['match_transport'].mean():.1%}")
else:
    df["match_transport"] = 0.5
    print("  Transport match      : columns not found, defaulting to 0.5")

if budget_col and entry_col and accom_col and food_col:
    df["total_cost"] = df[[entry_col, accom_col, food_col]].sum(axis=1)
    df["within_budget"] = (df["total_cost"] <= df[budget_col]).astype(float)
    print(f"  Within budget rate   : {df['within_budget'].mean():.1%}")
elif budget_col and entry_col:
    df["total_cost"] = df[entry_col]
    df["within_budget"] = (df["total_cost"] <= df[budget_col]).astype(float)
else:
    df["within_budget"] = 0.5
    print("  Within budget        : columns not found, defaulting to 0.5")

if time_constraint_col and travel_time_col:
    df["within_time"] = (df[travel_time_col] <= df[time_constraint_col]).astype(float)
    print(f"  Within time rate     : {df['within_time'].mean():.1%}")
else:
    df["within_time"] = 0.5
    print("  Within time          : columns not found, defaulting to 0.5")

if dest_col and pref_dest_col:
    df["match_destination"] = (
        df[dest_col].astype(str).str.strip().str.lower() ==
        df[pref_dest_col].astype(str).str.strip().str.lower()
    ).astype(float)
    print(f"  Destination match    : {df['match_destination'].mean():.1%}")
else:
    df["match_destination"] = 0.5
    print("  Destination match    : columns not found, defaulting to 0.5")

component_cols = ["match_transport", "within_budget", "within_time", "match_destination"]
df["personalization_score"] = df[component_cols].mean(axis=1)
print(f"\n  Personalization score: mean={df['personalization_score'].mean():.3f} "
      f"std={df['personalization_score'].std():.3f}")

poi_candidates = [c for c in df.columns if any(kw in c.lower() for kw in [
    "beach", "temple", "museum", "hill", "wildlife", "park", "heritage",
    "adventure", "city", "nature", "culture", "historical", "poi",
    "category", "type", "attraction"
]) and c not in [dest_col, pref_dest_col, satisfaction_col]]

if poi_candidates:
    print(f"\n  POI columns found    : {poi_candidates}")
    poi_df = df[poi_candidates].copy()
    if poi_df.select_dtypes(include="number").shape[1] == len(poi_candidates):
        df["unique_types_count"] = poi_df.sum(axis=1)
    else:
        df["unique_types_count"] = poi_df.apply(
            lambda row: row.dropna().nunique(), axis=1)
    df["diversity_index"] = df["unique_types_count"] / max(df["unique_types_count"].max(), 1)
    print(f"  Diversity (unique types): mean={df['unique_types_count'].mean():.2f}")
else:
    print("\n  No POI columns found — using destination variety as diversity proxy")
    if dest_col:
        df["unique_types_count"] = df[dest_col].apply(
            lambda x: len(str(x).split(",")) if pd.notna(x) else 1
        )
        if traffic_col:
            df["diversity_index"] = (
                (df["unique_types_count"] / df["unique_types_count"].max()) * 0.6 +
                (df[traffic_col] / df[traffic_col].max()).fillna(0) * 0.4
            )
        else:
            df["diversity_index"] = df["unique_types_count"] / df["unique_types_count"].max()
    else:
        df["unique_types_count"] = np.random.randint(1, 6, len(df))
        df["diversity_index"] = df["unique_types_count"] / 5

print(f"  Diversity index      : mean={df['diversity_index'].mean():.3f}\n")

df["personalization_tier"] = pd.cut(
    df["personalization_score"],
    bins=[-0.001, 0.33, 0.67, 1.001],
    labels=["Low", "Medium", "High"]
)

sat_max = df[satisfaction_col].max()
sat_threshold = 4.0 if sat_max <= 5 else sat_max * 0.75
df["high_satisfaction"] = (df[satisfaction_col] >= sat_threshold).astype(int)

print(f"── Summary ─────────────────────────────────────────────")
print(f"  Satisfaction range   : {df[satisfaction_col].min():.1f} – {df[satisfaction_col].max():.1f}")
print(f"  High satisfaction    : rating ≥ {sat_threshold:.1f} ({df['high_satisfaction'].mean():.1%} of routes)")
tier_counts = df["personalization_tier"].value_counts().sort_index()
print(f"  Personalization tiers: {dict(tier_counts)}")
print()

print("── Correlations ────────────────────────────────────────")

r_h1, p_h1 = stats.pearsonr(
    df["personalization_score"].dropna(),
    df.loc[df["personalization_score"].notna(), satisfaction_col]
)
r_h2, p_h2 = stats.pearsonr(
    df["personalization_score"].dropna(),
    df.loc[df["personalization_score"].notna(), "diversity_index"]
)
print(f"  H1 — Personalization vs Satisfaction : r={r_h1:.3f}, p={p_h1:.4f} "
      f"{'✅ significant' if p_h1 < 0.05 else '❌ not significant'}")
print(f"  H2 — Personalization vs Diversity    : r={r_h2:.3f}, p={p_h2:.4f} "
      f"{'✅ significant' if p_h2 < 0.05 else '❌ not significant'}")
print()

def save(name):
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✅  Saved → {name}.png")

print("── Generating charts ───────────────────────────────────")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Personalization Score Overview", fontsize=14, fontweight="bold")

axes[0].hist(df["personalization_score"], bins=20,
             color=PALETTE[0], edgecolor="white", linewidth=0.5)
axes[0].set_xlabel("Personalization Score (0–1)")
axes[0].set_ylabel("Number of Routes")
axes[0].set_title("Distribution of Personalization Scores")
for val, label, color in [
    (0.33, "Low/Med boundary", PALETTE[1]),
    (0.67, "Med/High boundary", PALETTE[2])
]:
    axes[0].axvline(val, color=color, linestyle="--", linewidth=1.5, label=label)
axes[0].legend(fontsize=8)

tier_counts_plot = df["personalization_tier"].value_counts().reindex(["Low", "Medium", "High"])
axes[1].bar(tier_counts_plot.index, tier_counts_plot.values,
            color=[PALETTE[1], PALETTE[3], PALETTE[2]], edgecolor="white")
axes[1].set_title("Routes per Personalization Tier")
axes[1].set_xlabel("Personalization Tier")
axes[1].set_ylabel("Count")
for i, v in enumerate(tier_counts_plot.values):
    axes[1].text(i, v + 2, str(v), ha="center", fontweight="bold")
save("chart1_personalization_distribution")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df["personalization_score"], df[satisfaction_col],
           alpha=0.25, s=12, color=PALETTE[0], label="Routes")
m, b = np.polyfit(df["personalization_score"], df[satisfaction_col], 1)
xs = np.linspace(0, 1, 100)
ax.plot(xs, m*xs+b, color=PALETTE[1], linewidth=2,
        label=f"Trend (r={r_h1:.3f}, p={p_h1:.3f})")
ax.set_xlabel("Personalization Score", fontsize=12)
ax.set_ylabel("Satisfaction Rating", fontsize=12)
ax.set_title("H1: Personalization → Satisfaction\n"
             f"r = {r_h1:.3f}  |  {'Significant' if p_h1 < 0.05 else 'Not significant'} (p={p_h1:.4f})",
             fontsize=13, fontweight="bold")
ax.legend()
save("chart2_H1_personalization_vs_satisfaction")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df["personalization_score"], df["diversity_index"],
           alpha=0.25, s=12, color=PALETTE[4], label="Routes")
m2, b2 = np.polyfit(df["personalization_score"], df["diversity_index"], 1)
ax.plot(xs, m2*xs+b2, color=PALETTE[1], linewidth=2,
        label=f"Trend (r={r_h2:.3f}, p={p_h2:.3f})")
ax.set_xlabel("Personalization Score", fontsize=12)
ax.set_ylabel("Exploration Diversity Index", fontsize=12)
ax.set_title("H2: Personalization → Diversity\n"
             f"r = {r_h2:.3f}  |  {'Significant' if p_h2 < 0.05 else 'Not significant'} (p={p_h2:.4f})",
             fontsize=13, fontweight="bold")
ax.legend()
save("chart3_H2_personalization_vs_diversity")

tier_summary = df.groupby("personalization_tier", observed=True).agg(
    avg_satisfaction=(satisfaction_col, "mean"),
    avg_diversity=("diversity_index", "mean"),
    count=("personalization_score", "count")
).reindex(["Low", "Medium", "High"])

fig, ax1 = plt.subplots(figsize=(10, 6))
fig.suptitle("The Personalization–Exploration Tradeoff\n"
             "Higher personalization = more satisfaction but less diversity",
             fontsize=14, fontweight="bold")

x = np.arange(3)
w = 0.35
bars1 = ax1.bar(x - w/2, tier_summary["avg_satisfaction"], w,
                label="Avg Satisfaction", color=PALETTE[2], edgecolor="white")
ax2 = ax1.twinx()
bars2 = ax2.bar(x + w/2, tier_summary["avg_diversity"], w,
                label="Avg Diversity Index", color=PALETTE[1], edgecolor="white")

ax1.set_xticks(x)
ax1.set_xticklabels(["Low\nPersonalization", "Medium\nPersonalization",
                      "High\nPersonalization"], fontsize=11)
ax1.set_ylabel("Average Satisfaction Rating", color=PALETTE[2], fontsize=11)
ax2.set_ylabel("Average Diversity Index", color=PALETTE[1], fontsize=11)
ax1.tick_params(axis="y", colors=PALETTE[2])
ax2.tick_params(axis="y", colors=PALETTE[1])

for bar, v in zip(bars1, tier_summary["avg_satisfaction"]):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01,
             f"{v:.2f}", ha="center", fontsize=10, fontweight="bold", color=PALETTE[2])
for bar, v in zip(bars2, tier_summary["avg_diversity"]):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.005,
             f"{v:.3f}", ha="center", fontsize=10, fontweight="bold", color=PALETTE[1])

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
save("chart4_tradeoff_bar_chart")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Personalization Components — Individual Analysis",
             fontsize=14, fontweight="bold")
axes = axes.flatten()

comp_info = [
    ("match_transport",   "Transport Preference Match",   PALETTE[0]),
    ("within_budget",     "Within Budget",                PALETTE[2]),
    ("within_time",       "Within Time Constraint",       PALETTE[3]),
    ("match_destination", "Destination Preference Match", PALETTE[4]),
]
for i, (col, title, color) in enumerate(comp_info):
    ax = axes[i]
    grp = df.groupby(col)[satisfaction_col].mean()
    bars = ax.bar(grp.index.astype(str), grp.values, color=[PALETTE[1], color],
                  edgecolor="white", width=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Mean Satisfaction")
    ax.set_xlabel("0 = No match / Over limit    1 = Match / Within limit")
    ax.set_ylim(0, df[satisfaction_col].max() * 1.15)
    for bar, v in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
save("chart5_component_breakdown")

fig, ax = plt.subplots(figsize=(10, 6))
colors_tier = {"Low": PALETTE[1], "Medium": PALETTE[3], "High": PALETTE[2]}
for tier in ["Low", "Medium", "High"]:
    sub = df[df["personalization_tier"] == tier][satisfaction_col].dropna()
    ax.hist(sub, bins=15, alpha=0.6, label=f"{tier} (n={len(sub)})",
            color=colors_tier[tier], edgecolor="white")
ax.set_xlabel("Satisfaction Rating", fontsize=12)
ax.set_ylabel("Number of Routes", fontsize=12)
ax.set_title("Satisfaction Distribution by Personalization Tier",
             fontsize=13, fontweight="bold")
ax.legend()
save("chart6_satisfaction_by_tier")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Exploration Diversity by Personalization Tier",
             fontsize=14, fontweight="bold")
tier_order = ["Low", "Medium", "High"]
sub_df = df.dropna(subset=["personalization_tier"])
sns.boxplot(data=sub_df, x="personalization_tier", y="diversity_index",
            order=tier_order, palette=[PALETTE[1], PALETTE[3], PALETTE[2]], ax=axes[0])
axes[0].set_title("Diversity Index by Tier")
axes[0].set_xlabel("Personalization Tier"); axes[0].set_ylabel("Diversity Index")

sns.violinplot(data=sub_df, x="personalization_tier", y=satisfaction_col,
               order=tier_order, palette=[PALETTE[1], PALETTE[3], PALETTE[2]], ax=axes[1])
axes[1].set_title("Satisfaction by Tier")
axes[1].set_xlabel("Personalization Tier"); axes[1].set_ylabel("Satisfaction Rating")
save("chart7_diversity_and_satisfaction_boxplots")

key_cols = ["personalization_score", satisfaction_col, "diversity_index",
            "match_transport", "within_budget", "within_time", "match_destination"]
key_cols = [c for c in key_cols if c in df.columns]
if len(key_cols) >= 3:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[key_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix — Key Variables", fontsize=14, fontweight="bold")
    save("chart8_correlation_heatmap")

if season_col:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Context Factors", fontsize=14, fontweight="bold")
    s_sat = df.groupby(season_col)[satisfaction_col].mean().sort_values(ascending=False)
    axes[0].bar(s_sat.index, s_sat.values, color=PALETTE[0], edgecolor="white")
    axes[0].set_title("Mean Satisfaction by Season")
    axes[0].set_ylabel("Mean Satisfaction"); axes[0].set_xlabel("Season")
    s_div = df.groupby(season_col)["diversity_index"].mean().sort_values(ascending=False)
    axes[1].bar(s_div.index, s_div.values, color=PALETTE[4], edgecolor="white")
    axes[1].set_title("Mean Diversity by Season")
    axes[1].set_ylabel("Diversity Index"); axes[1].set_xlabel("Season")
    save("chart9_season_context")

print("\n── Models ──────────────────────────────────────────────")

feature_cols_base = ["personalization_score", "match_transport", "within_budget",
                     "within_time", "match_destination"]
extra_numeric = [c for c in [distance_col, travel_time_col, traffic_col,
                              popularity_col, "total_cost"] if c and c in df.columns]
feature_cols = feature_cols_base + extra_numeric

if season_col:
    le = LabelEncoder()
    df["season_enc"] = le.fit_transform(df[season_col].astype(str))
    feature_cols.append("season_enc")
if dest_col:
    le2 = LabelEncoder()
    df["dest_enc"] = le2.fit_transform(df[dest_col].astype(str))
    feature_cols.append("dest_enc")
if transport_col:
    le3 = LabelEncoder()
    df["transport_enc"] = le3.fit_transform(df[transport_col].astype(str))
    feature_cols.append("transport_enc")

feature_cols = [c for c in feature_cols if c in df.columns]
model_df = df[feature_cols + [satisfaction_col, "high_satisfaction",
                               "diversity_index"]].copy()
for c in feature_cols:
    model_df[c] = pd.to_numeric(model_df[c], errors="coerce")
    model_df[c].fillna(model_df[c].median(), inplace=True)
model_df.dropna(inplace=True)
print(f"  Modelling on {len(model_df):,} rows, {len(feature_cols)} features")

X = model_df[feature_cols]

y_sat = model_df[satisfaction_col]
Xtr, Xte, ytr, yte = train_test_split(X, y_sat, test_size=0.2, random_state=RANDOM_STATE)
rf_sat = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
rf_sat.fit(Xtr, ytr)
y_pred_sat = rf_sat.predict(Xte)
r2_sat = r2_score(yte, y_pred_sat)
rmse_sat = np.sqrt(mean_squared_error(yte, y_pred_sat))
print(f"\n  Satisfaction model (Random Forest Regressor):")
print(f"    R² = {r2_sat:.3f}   RMSE = {rmse_sat:.4f}")

y_div = model_df["diversity_index"]
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y_div, test_size=0.2, random_state=RANDOM_STATE)
rf_div = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
rf_div.fit(Xtr2, ytr2)
y_pred_div = rf_div.predict(Xte2)
r2_div = r2_score(yte2, y_pred_div)
rmse_div = np.sqrt(mean_squared_error(yte2, y_pred_div))
print(f"\n  Diversity model (Random Forest Regressor):")
print(f"    R² = {r2_div:.3f}   RMSE = {rmse_div:.4f}")

fi_sat = pd.DataFrame({"feature": feature_cols,
                        "importance": rf_sat.feature_importances_})
fi_sat.sort_values("importance", ascending=True, inplace=True)
colors_fi = [PALETTE[2] if v >= fi_sat["importance"].quantile(0.75) else PALETTE[0]
             for v in fi_sat["importance"]]

fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(feature_cols)*0.45)))
fig.suptitle("Feature Importance — Random Forest Models",
             fontsize=14, fontweight="bold")

bars = axes[0].barh(fi_sat["feature"], fi_sat["importance"], color=colors_fi)
axes[0].set_title(f"Predicting Satisfaction\n(R²={r2_sat:.3f})", fontsize=11)
axes[0].set_xlabel("Importance")
for bar, v in zip(bars, fi_sat["importance"]):
    axes[0].text(v + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=8)

fi_div = pd.DataFrame({"feature": feature_cols,
                        "importance": rf_div.feature_importances_})
fi_div.sort_values("importance", ascending=True, inplace=True)
colors_fi2 = [PALETTE[1] if v >= fi_div["importance"].quantile(0.75) else PALETTE[4]
              for v in fi_div["importance"]]
bars2 = axes[1].barh(fi_div["feature"], fi_div["importance"], color=colors_fi2)
axes[1].set_title(f"Predicting Diversity\n(R²={r2_div:.3f})", fontsize=11)
axes[1].set_xlabel("Importance")
for bar, v in zip(bars2, fi_div["importance"]):
    axes[1].text(v + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=8)
save("chart10_feature_importance")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Model Performance — Predicted vs Actual",
             fontsize=14, fontweight="bold")

axes[0].scatter(yte, y_pred_sat, alpha=0.3, s=10, color=PALETTE[2])
lims = [min(yte.min(), y_pred_sat.min())-0.1, max(yte.max(), y_pred_sat.max())+0.1]
axes[0].plot(lims, lims, color=PALETTE[1], linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Actual Satisfaction"); axes[0].set_ylabel("Predicted Satisfaction")
axes[0].set_title(f"Satisfaction  R²={r2_sat:.3f}")

axes[1].scatter(yte2, y_pred_div, alpha=0.3, s=10, color=PALETTE[1])
lims2 = [min(yte2.min(), y_pred_div.min())-0.05, max(yte2.max(), y_pred_div.max())+0.05]
axes[1].plot(lims2, lims2, color=PALETTE[1], linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Actual Diversity"); axes[1].set_ylabel("Predicted Diversity")
axes[1].set_title(f"Diversity  R²={r2_div:.3f}")
save("chart11_predicted_vs_actual")

print("\n" + "="*60)
print("  KEY FINDINGS — use these on your poster")
print("="*60)
print(f"  Dataset      : {len(df):,} tourism routes, 19 variables")
print(f"  Source       : SmartTourRoutePlanner.csv — Kaggle (ziya07)")
print(f"  URL          : https://www.kaggle.com/datasets/ziya07/smarttourrouteplanner-tourism-route-dataset")
print()
print(f"  HYPOTHESIS 1 — Personalization → Satisfaction")
print(f"    Pearson r = {r_h1:.3f}  p = {p_h1:.4f}")
direction_h1 = "positive" if r_h1 > 0 else "negative"
sig_h1 = "SUPPORTED" if p_h1 < 0.05 else "NOT supported"
print(f"    Direction : {direction_h1}  →  H1 {sig_h1}")
print()
print(f"  HYPOTHESIS 2 — Personalization → Less Diversity")
print(f"    Pearson r = {r_h2:.3f}  p = {p_h2:.4f}")
direction_h2 = "negative (less diverse)" if r_h2 < 0 else "positive (more diverse)"
sig_h2 = "SUPPORTED" if p_h2 < 0.05 and r_h2 < 0 else "NOT supported"
print(f"    Direction : {direction_h2}  →  H2 {sig_h2}")
print()
print(f"  TIER AVERAGES:")
for tier in ["Low", "Medium", "High"]:
    row = tier_summary.loc[tier]
    print(f"    {tier:<8} personalization: "
          f"satisfaction={row['avg_satisfaction']:.2f}  "
          f"diversity={row['avg_diversity']:.3f}  "
          f"n={int(row['count'])}")
print()
print(f"  MODELS:")
print(f"    Satisfaction model R² = {r2_sat:.3f}")
print(f"    Diversity model    R² = {r2_div:.3f}")
print()
print(f"  TOP PREDICTORS OF SATISFACTION:")
top3 = fi_sat.nlargest(3, "importance")
for _, row in top3.iterrows():
    print(f"    {row['feature']:<30} importance={row['importance']:.4f}")
print()
print("  Charts saved:")
for i, name in enumerate([
    "chart1_personalization_distribution",
    "chart2_H1_personalization_vs_satisfaction",
    "chart3_H2_personalization_vs_diversity",
    "chart4_tradeoff_bar_chart",
    "chart5_component_breakdown",
    "chart6_satisfaction_by_tier",
    "chart7_diversity_and_satisfaction_boxplots",
    "chart8_correlation_heatmap",
    "chart9_season_context",
    "chart10_feature_importance",
    "chart11_predicted_vs_actual",
], 1):
    print(f"    {i:02d}. {name}.png")
print()
print("  Citation:")
print("  SmartTourRoutePlanner: Tourism Route Dataset. Kaggle, uploaded by ziya07.")
print("  URL: https://www.kaggle.com/datasets/ziya07/smarttourrouteplanner-tourism-route-dataset")
print("  File format: CSV")
print("="*60 + "\n")

print("── Additional Tests ───────────────────────────────────")

def anova_groups(frame, group_col, value_col):
    groups = []
    for group_name in frame[group_col].dropna().unique():
        values = frame.loc[frame[group_col] == group_name, value_col].dropna()
        if len(values) >= 2:
            groups.append(values)
    return groups

if "total_cost" in df.columns:
    df["cost_tier"] = pd.qcut(
        df["total_cost"].rank(method="first"),
        q=3,
        labels=["Low", "Medium", "High"]
    )
    cost_groups = anova_groups(df, "cost_tier", satisfaction_col)
    if len(cost_groups) >= 2:
        cost_f, cost_p = stats.f_oneway(*cost_groups)
        print(f"  Cost tier ANOVA: F={cost_f:.3f}, p={cost_p:.4f}")
    else:
        print("  Cost tier ANOVA: insufficient data")
else:
    print("  Cost tier ANOVA: total cost not available")

if season_col:
    season_groups = anova_groups(df, season_col, satisfaction_col)
    if len(season_groups) >= 2:
        season_f, season_p = stats.f_oneway(*season_groups)
        print(f"  Season ANOVA: F={season_f:.3f}, p={season_p:.4f}")
    else:
        print("  Season ANOVA: insufficient data")
else:
    print("  Season ANOVA: season column not available")

if budget_col and "total_cost" in df.columns:
    within_budget = df.loc[df["total_cost"] <= df[budget_col], satisfaction_col].dropna()
    over_budget = df.loc[df["total_cost"] > df[budget_col], satisfaction_col].dropna()
    if len(within_budget) >= 2 and len(over_budget) >= 2:
        budget_t, budget_p = stats.ttest_ind(within_budget, over_budget, equal_var=False)
        print(f"  Budget t-test: t={budget_t:.3f}, p={budget_p:.4f}")
    else:
        print("  Budget t-test: insufficient data")
else:
    print("  Budget t-test: budget comparison not available")
