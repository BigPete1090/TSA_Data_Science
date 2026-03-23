"""
TSA Data Science 2026 — SmartTourRoutePlanner Analysis
Dataset: https://www.kaggle.com/datasets/ziya07/smarttourrouteplanner-tourism-route-dataset
Research Question: What route characteristics best predict high popularity scores?

HOW TO RUN:
  1. Place the CSV in the same folder as this file.
  2. Update CSV_FILE below if your filename differs.
  3. Run:  python3 tsa_analysis.py
  4. All charts save automatically as PNG files in this folder.
"""

import sys
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

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.linear_model     import LinearRegression, Ridge
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree             import DecisionTreeRegressor
from sklearn.metrics          import r2_score, mean_squared_error, mean_absolute_error

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
CSV_FILE     = "tourism_route_dataset.csv"   # ← update if needed
RANDOM_STATE = 42
CHART_DPI    = 150

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2D6A9F", "#E05C3A", "#2EAA6E", "#F5A623", "#8E44AD", "#16A085"]
sns.set_palette(PALETTE)

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  TSA Data Science 2026 — Tourism Route Popularity")
print("="*60)

try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"\n❌  File '{CSV_FILE}' not found.")
    sys.exit(1)

print(f"\n✅  Loaded {len(df):,} rows × {len(df.columns)} columns")
print(f"    Columns: {list(df.columns)}\n")

# ═══════════════════════════════════════════════════════════════════════════
# 2. COLUMN DETECTION
# ═══════════════════════════════════════════════════════════════════════════
cols_lower = {c.lower(): c for c in df.columns}

def find_col(*keywords):
    for kw in keywords:
        for lc, orig in cols_lower.items():
            if kw in lc:
                return orig
    return None

target_col    = find_col("popularity")
dest_col      = find_col("destination_type", "destination", "end_location")
transport_col = find_col("transport_mode", "transport")
season_col    = find_col("season")
day_col       = find_col("day_type", "day")
distance_col  = find_col("distance")
time_col      = find_col("travel_time", "estimated_travel")
traffic_col   = find_col("traffic")
entry_col     = find_col("entry_fee", "entry")
accom_col     = find_col("accommodation")
food_col      = find_col("food")
budget_col    = find_col("user_budget", "budget")
time_constraint_col = find_col("time_constraint", "user_time")
pref_transport_col  = find_col("preferred_transport")
pref_dest_col       = find_col("preferred_destination")
satisfaction_col    = find_col("satisfaction")

print("── Detected columns ───────────────────────────────────")
for label, col in [
    ("TARGET: Popularity",      target_col),
    ("Destination type",        dest_col),
    ("Transport mode",          transport_col),
    ("Season",                  season_col),
    ("Day type",                day_col),
    ("Distance km",             distance_col),
    ("Travel time hr",          time_col),
    ("Traffic density",         traffic_col),
    ("Entry fee",               entry_col),
    ("Accommodation cost",      accom_col),
    ("Food cost",               food_col),
    ("User budget",             budget_col),
    ("Time constraint",         time_constraint_col),
    ("Preferred transport",     pref_transport_col),
    ("Preferred destination",   pref_dest_col),
    ("Satisfaction rating",     satisfaction_col),
]:
    print(f"  {label:<24}: {col if col else 'not found'}")
print()

if target_col is None:
    print("❌  Could not find popularity column.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# 3. CLEAN & ENGINEER
# ═══════════════════════════════════════════════════════════════════════════
df = df.copy()
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df.dropna(subset=[target_col], inplace=True)

# Numeric conversions
for c in [distance_col, time_col, traffic_col, entry_col,
          accom_col, food_col, budget_col, time_constraint_col, satisfaction_col]:
    if c:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Engineered features
if entry_col and accom_col and food_col:
    df["total_trip_cost"] = df[[entry_col, accom_col, food_col]].sum(axis=1)

if budget_col and "total_trip_cost" in df.columns:
    df["budget_utilisation"] = (df["total_trip_cost"] /
                                df[budget_col].replace(0, np.nan)).round(4)

if distance_col and time_col:
    df["avg_speed_kmh"] = (df[distance_col] /
                           df[time_col].replace(0, np.nan)).round(2)

if entry_col and distance_col:
    df["cost_per_km"] = (df[entry_col] /
                         df[distance_col].replace(0, np.nan)).round(4)

# Preference match flags
if transport_col and pref_transport_col:
    df["transport_match"] = (df[transport_col] == df[pref_transport_col]).astype(int)

if dest_col and pref_dest_col:
    df["destination_match"] = (df[dest_col] == df[pref_dest_col]).astype(int)

print(f"── Target: {target_col} ────────────────────────────────")
print(f"  Range  : {df[target_col].min():.3f} – {df[target_col].max():.3f}")
print(f"  Mean   : {df[target_col].mean():.3f}")
print(f"  Std    : {df[target_col].std():.3f}\n")

# ═══════════════════════════════════════════════════════════════════════════
# 4. DESCRIPTIVE STATS
# ═══════════════════════════════════════════════════════════════════════════
print("── Descriptive statistics ──────────────────────────────")
num_cols = df.select_dtypes(include="number").columns.tolist()
print(df[num_cols].describe().round(3).to_string())
print()

# ═══════════════════════════════════════════════════════════════════════════
# 5. CHARTS
# ═══════════════════════════════════════════════════════════════════════════
def save(name):
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✅  Saved → {name}.png")

print("── Generating charts ───────────────────────────────────")

# ── Chart 1: Popularity distribution ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Popularity Score Distribution", fontsize=14, fontweight="bold")

axes[0].hist(df[target_col], bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.5)
axes[0].axvline(df[target_col].mean(), color=PALETTE[1], linestyle="--",
                linewidth=1.5, label=f"Mean = {df[target_col].mean():.2f}")
axes[0].axvline(df[target_col].median(), color=PALETTE[2], linestyle=":",
                linewidth=1.5, label=f"Median = {df[target_col].median():.2f}")
axes[0].set_xlabel("Popularity Score (0–1)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Popularity Scores")
axes[0].legend()

# High vs low popularity split (top/bottom 33%)
lo_thresh = df[target_col].quantile(0.33)
hi_thresh = df[target_col].quantile(0.67)
labels_pie = ["Low (<0.33)", "Medium", "High (>0.67)"]
sizes = [
    (df[target_col] < lo_thresh).sum(),
    ((df[target_col] >= lo_thresh) & (df[target_col] <= hi_thresh)).sum(),
    (df[target_col] > hi_thresh).sum(),
]
axes[1].pie(sizes, labels=labels_pie, colors=[PALETTE[1], PALETTE[3], PALETTE[2]],
            autopct="%1.1f%%", startangle=90, wedgeprops=dict(edgecolor="white"))
axes[1].set_title("Popularity Tiers")
save("chart1_popularity_distribution")

# ── Chart 2: Mean popularity by category ────────────────────────────────
cat_cols = [c for c in [dest_col, transport_col, season_col, day_col] if c]
if cat_cols:
    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1: axes = [axes]
    fig.suptitle("Mean Popularity Score by Category", fontsize=14, fontweight="bold")
    for ax, col in zip(axes, cat_cols):
        means = df.groupby(col)[target_col].mean().sort_values(ascending=False)
        bars = ax.barh(means.index.astype(str), means.values, color=PALETTE[0])
        ax.axvline(df[target_col].mean(), color=PALETTE[1], linestyle="--",
                   linewidth=1.2, label="Overall mean")
        ax.set_xlabel("Mean Popularity"); ax.set_title(col.replace("_", " ").title())
        ax.legend(fontsize=8)
        for bar, v in zip(bars, means.values):
            ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=8)
    save("chart2_popularity_by_category")

# ── Chart 3: Popularity by season × destination heatmap ─────────────────
if season_col and dest_col:
    pivot = df.pivot_table(values=target_col, index=season_col,
                           columns=dest_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", linewidths=0.5,
                ax=ax, cbar_kws={"label": "Mean Popularity Score"})
    ax.set_title("Mean Popularity: Season × Destination Type",
                 fontsize=14, fontweight="bold")
    save("chart3_season_destination_heatmap")

# ── Chart 4: Popularity by transport × destination ──────────────────────
if transport_col and dest_col:
    pivot2 = df.pivot_table(values=target_col, index=transport_col,
                            columns=dest_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot2, annot=True, fmt=".3f", cmap="Blues", linewidths=0.5,
                ax=ax, cbar_kws={"label": "Mean Popularity Score"})
    ax.set_title("Mean Popularity: Transport Mode × Destination Type",
                 fontsize=14, fontweight="bold")
    save("chart4_transport_destination_heatmap")

# ── Chart 5: Numeric predictors vs popularity (scatter grid) ────────────
numeric_predictors = [c for c in [
    distance_col, time_col, traffic_col, entry_col,
    accom_col, food_col, budget_col, "total_trip_cost",
    "avg_speed_kmh", "cost_per_km"
] if c and c in df.columns]

if numeric_predictors:
    n_plots = min(len(numeric_predictors), 8)
    cols_per_row = 4
    rows = (n_plots + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row,
                             figsize=(5*cols_per_row, 4*rows))
    axes = axes.flatten() if rows > 1 else axes
    fig.suptitle("Numeric Predictors vs. Popularity Score",
                 fontsize=14, fontweight="bold")
    for i, col in enumerate(numeric_predictors[:n_plots]):
        ax = axes[i] if n_plots > 1 else axes
        sub = df[[col, target_col]].dropna()
        ax.scatter(sub[col], sub[target_col], alpha=0.25, s=8, color=PALETTE[i % len(PALETTE)])
        if len(sub) > 2:
            m, b = np.polyfit(sub[col], sub[target_col], 1)
            xs = np.linspace(sub[col].min(), sub[col].max(), 100)
            ax.plot(xs, m*xs+b, color=PALETTE[1], linewidth=1.5)
            corr = sub[col].corr(sub[target_col])
            ax.set_title(f"{col.replace('_',' ').title()}\n(r = {corr:.3f})", fontsize=10)
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=8)
        ax.set_ylabel("Popularity", fontsize=8)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    save("chart5_numeric_predictors_scatter")

# ── Chart 6: Correlation heatmap ─────────────────────────────────────────
drop_for_corr = ["route_id"]
num_df = df.select_dtypes(include="number").drop(columns=drop_for_corr, errors="ignore")
if len(num_df.columns) >= 3:
    fig, ax = plt.subplots(figsize=(13, 10))
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 8})
    ax.set_title("Correlation Matrix — All Numeric Variables",
                 fontsize=14, fontweight="bold")
    save("chart6_correlation_heatmap")

# ── Chart 7: Popularity by preference match ──────────────────────────────
match_cols = [c for c in ["transport_match", "destination_match"] if c in df.columns]
if match_cols:
    fig, axes = plt.subplots(1, len(match_cols), figsize=(6*len(match_cols), 5))
    if len(match_cols) == 1: axes = [axes]
    fig.suptitle("Does Matching User Preferences Affect Popularity?",
                 fontsize=14, fontweight="bold")
    labels_map = {"transport_match": "Transport Preference Match",
                  "destination_match": "Destination Preference Match"}
    for ax, col in zip(axes, match_cols):
        grp = df.groupby(col)[target_col].mean()
        bars = ax.bar(["No match", "Match"], grp.values, color=[PALETTE[1], PALETTE[2]],
                      edgecolor="white", width=0.5)
        ax.set_title(labels_map.get(col, col))
        ax.set_ylabel("Mean Popularity Score")
        ax.set_ylim(0, 1)
        for bar, v in zip(bars, grp.values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    save("chart7_preference_match_analysis")

# ── Chart 8: Top 10 & bottom 10 routes by popularity ────────────────────
if "route_id" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Highest vs Lowest Popularity Routes", fontsize=14, fontweight="bold")
    label_cols = [c for c in [dest_col, transport_col, season_col] if c]

    for ax, df_sub, title, color in [
        (axes[0], df.nlargest(10, target_col),  "Top 10 most popular",  PALETTE[2]),
        (axes[1], df.nsmallest(10, target_col), "Bottom 10 least popular", PALETTE[1]),
    ]:
        if label_cols:
            route_labels = df_sub[label_cols].astype(str).agg(" | ".join, axis=1)
        else:
            route_labels = df_sub["route_id"].astype(str)
        ax.barh(route_labels, df_sub[target_col], color=color, edgecolor="white")
        ax.set_title(title); ax.set_xlabel("Popularity Score")
        ax.set_xlim(0, 1)
    save("chart8_top_bottom_routes")

# ═══════════════════════════════════════════════════════════════════════════
# 6. PREDICTIVE MODELLING — REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Predictive modelling (regression) ──────────────────")

numeric_features = [c for c in [
    distance_col, time_col, traffic_col, entry_col,
    accom_col, food_col, budget_col, time_constraint_col, satisfaction_col,
    "total_trip_cost", "budget_utilisation", "avg_speed_kmh",
    "cost_per_km", "transport_match", "destination_match"
] if c and c in df.columns]

categorical_features = [c for c in [
    transport_col, dest_col, season_col, day_col
] if c and c in df.columns]

all_features = numeric_features + categorical_features
model_df = df[all_features + [target_col]].copy()

# Fill numeric NaNs with median
for col in numeric_features:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df[col].fillna(model_df[col].median(), inplace=True)

# Encode categoricals
le_map = {}
for col in categorical_features:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    le_map[col] = le

model_df.dropna(inplace=True)
print(f"  Modelling on {len(model_df):,} rows, {len(all_features)} features\n")

if len(model_df) < 50:
    print("  ⚠️  Not enough rows for modelling.")
else:
    X = model_df[all_features]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Linear Regression"  : LinearRegression(),
        "Ridge Regression"   : Ridge(alpha=1.0),
        "Decision Tree"      : DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest"      : RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        Xtr = X_train_s if "Regression" in name else X_train
        Xte = X_test_s  if "Regression" in name else X_test
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        results[name] = {"r2": r2, "rmse": rmse, "mae": mae, "model": model, "y_pred": y_pred}
        print(f"  {name:<22}  R²={r2:.3f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    best_name  = max(results, key=lambda k: results[k]["r2"])
    best       = results[best_name]
    best_model = best["model"]
    y_pred_best = best["y_pred"]
    Xte_final   = X_test_s if "Regression" in best_name else X_test

    print(f"\n  🏆  Best model: {best_name}")
    print(f"      R² = {best['r2']:.3f}  (explains {best['r2']*100:.1f}% of popularity variance)")
    print(f"      RMSE = {best['rmse']:.4f}  MAE = {best['mae']:.4f}")

    # ── Chart 9: Model comparison ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Regression Model Comparison", fontsize=14, fontweight="bold")
    names = list(results.keys())
    r2s   = [results[n]["r2"]   for n in names]
    rmses = [results[n]["rmse"] for n in names]
    colors_bar = [PALETTE[2] if n == best_name else PALETTE[0] for n in names]

    axes[0].bar(names, r2s, color=colors_bar, edgecolor="white")
    axes[0].set_title("R² Score (higher = better)")
    axes[0].set_ylabel("R²"); axes[0].set_ylim(0, 1)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    for i, v in enumerate(r2s):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(names, rmses, color=colors_bar, edgecolor="white")
    axes[1].set_title("RMSE (lower = better)")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    for i, v in enumerate(rmses):
        axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    save("chart9_model_comparison")

    # ── Chart 10: Predicted vs actual ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred_best, alpha=0.35, s=15, color=PALETTE[0], label="Routes")
    lims = [min(y_test.min(), y_pred_best.min()) - 0.05,
            max(y_test.max(), y_pred_best.max()) + 0.05]
    ax.plot(lims, lims, color=PALETTE[1], linewidth=1.5, linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual Popularity Score")
    ax.set_ylabel("Predicted Popularity Score")
    ax.set_title(f"Predicted vs Actual — {best_name}\n(R² = {best['r2']:.3f})",
                 fontsize=13, fontweight="bold")
    ax.legend()
    save("chart10_predicted_vs_actual")

    # ── Chart 11: Residuals ───────────────────────────────────────────────
    residuals = y_test - y_pred_best
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Residual Analysis — {best_name}", fontsize=14, fontweight="bold")
    axes[0].scatter(y_pred_best, residuals, alpha=0.35, s=12, color=PALETTE[4])
    axes[0].axhline(0, color=PALETTE[1], linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted Popularity"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")
    axes[1].hist(residuals, bins=25, color=PALETTE[4], edgecolor="white")
    axes[1].axvline(0, color=PALETTE[1], linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    save("chart11_residuals")

    # ── Chart 12: Feature importance ─────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        fi_df = pd.DataFrame({"feature": all_features,
                              "importance": best_model.feature_importances_})
        fi_df.sort_values("importance", ascending=True, inplace=True)
        colors_fi = [PALETTE[2] if v >= fi_df["importance"].quantile(0.75) else PALETTE[0]
                     for v in fi_df["importance"]]
        fig, ax = plt.subplots(figsize=(10, max(5, len(all_features)*0.5)))
        bars = ax.barh(fi_df["feature"], fi_df["importance"], color=colors_fi)
        ax.set_title(f"Feature Importance — {best_name}\n(green = top quartile)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        for bar, v in zip(bars, fi_df["importance"]):
            ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=8)
        save("chart12_feature_importance")

        top3 = fi_df.nlargest(3, "importance")["feature"].tolist()
        print(f"\n  Top 3 predictors of popularity:")
        for f in top3:
            imp = fi_df[fi_df["feature"] == f]["importance"].values[0]
            print(f"    {f:<30} importance = {imp:.4f}")

    elif hasattr(best_model, "coef_"):
        coefs = best_model.coef_
        fi_df = pd.DataFrame({"feature": all_features, "coef": coefs})
        fi_df.sort_values("coef", inplace=True)
        colors_fi = [PALETTE[2] if c > 0 else PALETTE[1] for c in fi_df["coef"]]
        fig, ax = plt.subplots(figsize=(10, max(5, len(all_features)*0.5)))
        ax.barh(fi_df["feature"], fi_df["coef"], color=colors_fi)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Regression Coefficients — {best_name}\n(positive = increases popularity)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Coefficient")
        save("chart12_feature_importance")

    # ── Chart 13: Top feature deep dives ─────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        top_features = fi_df.nlargest(4, "importance")["feature"].tolist()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Top 4 Predictors — Deep Dive", fontsize=14, fontweight="bold")
        axes = axes.flatten()
        for i, feat in enumerate(top_features):
            ax = axes[i]
            if df[feat].dtype == object or df[feat].nunique() <= 8:
                means = df.groupby(feat)[target_col].mean().sort_values(ascending=False)
                ax.bar(means.index.astype(str), means.values,
                       color=PALETTE[i % len(PALETTE)], edgecolor="white")
                ax.set_title(feat.replace("_", " ").title())
                ax.set_ylabel("Mean Popularity")
                plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            else:
                sub = df[[feat, target_col]].dropna()
                ax.scatter(sub[feat], sub[target_col], alpha=0.3, s=8,
                           color=PALETTE[i % len(PALETTE)])
                if len(sub) > 2:
                    m, b = np.polyfit(sub[feat], sub[target_col], 1)
                    xs = np.linspace(sub[feat].min(), sub[feat].max(), 100)
                    ax.plot(xs, m*xs+b, color=PALETTE[1], linewidth=1.5)
                    corr = sub[feat].corr(sub[target_col])
                    ax.set_title(f"{feat.replace('_',' ').title()} (r={corr:.3f})")
                ax.set_xlabel(feat.replace("_", " ").title())
                ax.set_ylabel("Popularity Score")
        save("chart13_top_predictors_deep_dive")

# ═══════════════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  KEY FINDINGS — use these on your poster")
print("="*60)
print(f"  Dataset      : {len(df):,} tourism route records")
print(f"  Source       : Kaggle CSV — ziya07/smarttourrouteplanner-tourism-route-dataset")
print(f"  Target       : popularity_score (continuous 0–1)")
print(f"  Score range  : {df[target_col].min():.3f} – {df[target_col].max():.3f}")
print(f"  Mean score   : {df[target_col].mean():.3f}")
if "best_name" in dir() and len(model_df) >= 50:
    print(f"  Best model   : {best_name}")
    print(f"  R²           : {best['r2']:.3f}  ({best['r2']*100:.1f}% of variance explained)")
    print(f"  RMSE         : {best['rmse']:.4f}")
if dest_col:
    top_dest = df.groupby(dest_col)[target_col].mean().idxmax()
    print(f"  Most popular destination : {top_dest}")
if transport_col:
    top_trans = df.groupby(transport_col)[target_col].mean().idxmax()
    print(f"  Most popular transport   : {top_trans}")
if season_col:
    top_season = df.groupby(season_col)[target_col].mean().idxmax()
    print(f"  Most popular season      : {top_season}")
print()
print("  Charts saved:")
for i, name in enumerate([
    "chart1_popularity_distribution",
    "chart2_popularity_by_category",
    "chart3_season_destination_heatmap",
    "chart4_transport_destination_heatmap",
    "chart5_numeric_predictors_scatter",
    "chart6_correlation_heatmap",
    "chart7_preference_match_analysis",
    "chart8_top_bottom_routes",
    "chart9_model_comparison",
    "chart10_predicted_vs_actual",
    "chart11_residuals",
    "chart12_feature_importance",
    "chart13_top_predictors_deep_dive",
], 1):
    print(f"    {i:02d}. {name}.png")
print()
print("  Citation for poster:")
print("  SmartTourRoutePlanner Tourism Route Dataset. Kaggle, uploaded by ziya07.")
print("  URL: https://www.kaggle.com/datasets/ziya07/smarttourrouteplanner-tourism-route-dataset")
print("  File format: CSV")
print("="*60 + "\n")