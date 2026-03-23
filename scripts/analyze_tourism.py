from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "tourism_dataset_5000.csv"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["high_satisfaction"] = (df["Satisfaction"] >= 4).astype(int)
    return df


def expand_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    expanded = df.copy()
    for col in ["Interests", "Sites Visited"]:
        values = expanded[col].apply(literal_eval)
        classes = sorted({item for items in values for item in items})
        for item in classes:
            expanded[f"{col}__{item}"] = values.apply(lambda items, item=item: int(item in items))
        expanded = expanded.drop(columns=[col])
    return expanded


def plot_satisfaction_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    order = sorted(df["Satisfaction"].unique())
    ax = sns.countplot(data=df, x="Satisfaction", hue="Satisfaction", order=order, palette="crest", legend=False)
    ax.set_title("Satisfaction Distribution")
    ax.set_xlabel("Satisfaction Score")
    ax.set_ylabel("Number of Tourist Records")
    for patch in ax.patches:
        ax.annotate(
            f"{int(patch.get_height())}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "satisfaction_distribution.png", dpi=200)
    plt.close()


def plot_high_satisfaction_by_site(df: pd.DataFrame) -> None:
    site_rates = (
        df.groupby("Site Name", as_index=False)["high_satisfaction"]
        .mean()
        .sort_values("high_satisfaction", ascending=False)
    )
    site_rates["high_satisfaction"] *= 100
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=site_rates,
        x="high_satisfaction",
        y="Site Name",
        hue="Site Name",
        palette="viridis",
        legend=False,
    )
    ax.set_title("High Satisfaction Rate by Site")
    ax.set_xlabel("High Satisfaction Rate (%)")
    ax.set_ylabel("Site")
    for patch in ax.patches:
        width = patch.get_width()
        ax.annotate(f"{width:.1f}%", (width, patch.get_y() + patch.get_height() / 2), va="center", xytext=(4, 0), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "high_satisfaction_by_site.png", dpi=200)
    plt.close()


def plot_high_satisfaction_by_duration(df: pd.DataFrame) -> None:
    duration_rates = df.groupby("Tour Duration", as_index=False)["high_satisfaction"].mean()
    duration_rates["high_satisfaction"] *= 100
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        data=duration_rates,
        x="Tour Duration",
        y="high_satisfaction",
        marker="o",
        linewidth=2.5,
        color="#1f6f8b",
    )
    ax.set_title("High Satisfaction Rate by Tour Duration")
    ax.set_xlabel("Tour Duration")
    ax.set_ylabel("High Satisfaction Rate (%)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "high_satisfaction_by_tour_duration.png", dpi=200)
    plt.close()


def plot_rating_vs_satisfaction(df: pd.DataFrame) -> None:
    table = pd.crosstab(df["Satisfaction"], df["Tourist Rating"])
    plt.figure(figsize=(11, 4))
    sns.heatmap(table, cmap="mako", cbar_kws={"label": "Count"})
    plt.title("Tourist Rating vs Satisfaction")
    plt.xlabel("Tourist Rating")
    plt.ylabel("Satisfaction")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rating_vs_satisfaction_heatmap.png", dpi=200)
    plt.close()


def save_descriptive_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    tables["site_summary"] = (
        df.groupby("Site Name")
        .agg(
            high_satisfaction_rate=("high_satisfaction", "mean"),
            avg_rating=("Tourist Rating", "mean"),
            records=("Tourist ID", "count"),
        )
        .sort_values("high_satisfaction_rate", ascending=False)
        .reset_index()
    )

    tables["preferred_duration_summary"] = (
        df.groupby("Preferred Tour Duration")
        .agg(
            high_satisfaction_rate=("high_satisfaction", "mean"),
            avg_rating=("Tourist Rating", "mean"),
            records=("Tourist ID", "count"),
        )
        .sort_values("Preferred Tour Duration")
        .reset_index()
    )

    tables["tour_duration_summary"] = (
        df.groupby("Tour Duration")
        .agg(
            high_satisfaction_rate=("high_satisfaction", "mean"),
            avg_rating=("Tourist Rating", "mean"),
            records=("Tourist ID", "count"),
        )
        .sort_values("Tour Duration")
        .reset_index()
    )

    numeric_cols = [
        "Age",
        "Preferred Tour Duration",
        "Tour Duration",
        "Tourist Rating",
        "System Response Time",
        "Recommendation Accuracy",
        "VR Experience Quality",
        "Satisfaction",
    ]
    tables["correlations"] = (
        df[numeric_cols]
        .corr(numeric_only=True)[["Satisfaction"]]
        .sort_values("Satisfaction", ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", "Satisfaction": "correlation_with_satisfaction"})
    )

    for name, table in tables.items():
        table.to_csv(OUTPUTS_DIR / f"{name}.csv", index=False)

    return tables


def build_features(df: pd.DataFrame, include_rating: bool) -> tuple[pd.DataFrame, pd.Series]:
    model_df = expand_list_columns(df)
    model_df["interest_count"] = model_df.filter(like="Interests__").sum(axis=1)
    model_df["sites_visited_count"] = model_df.filter(like="Sites Visited__").sum(axis=1)
    model_df["duration_gap"] = model_df["Tour Duration"] - model_df["Preferred Tour Duration"]
    drop_cols = ["Tourist ID", "Route ID", "Satisfaction", "high_satisfaction"]
    if not include_rating:
        drop_cols.append("Tourist Rating")

    X = model_df.drop(columns=drop_cols).copy()
    X["Accessibility"] = X["Accessibility"].astype(int)
    y = model_df["high_satisfaction"]
    return X, y


def train_models(df: pd.DataFrame, include_rating: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y = build_features(df, include_rating=include_rating)

    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        verbose_feature_names_out=False,
    )

    models = {
        "Majority Baseline": None,
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    metrics_rows: list[dict[str, object]] = []
    best_importances: pd.DataFrame | None = None
    best_confusion: pd.DataFrame | None = None
    best_f1 = -1.0

    for name, model in models.items():
        if model is None:
            majority_class = int(y_train.mode().iloc[0])
            predictions = pd.Series([majority_class] * len(y_test), index=y_test.index)
        else:
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        bal_acc = balanced_accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, zero_division=0)
        rec = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        metrics_rows.append(
            {
                "scenario": "with_tourist_rating" if include_rating else "without_tourist_rating",
                "model": name,
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
        )

        if model is not None and f1 > best_f1:
            best_f1 = f1
            cm = confusion_matrix(y_test, predictions)
            best_confusion = pd.DataFrame(
                cm,
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"],
            )

            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            fitted_model = pipeline.named_steps["model"]

            if hasattr(fitted_model, "feature_importances_"):
                importances = pd.Series(fitted_model.feature_importances_, index=feature_names)
                best_importances = (
                    importances.sort_values(ascending=False)
                    .head(15)
                    .rename_axis("feature")
                    .reset_index(name="importance")
                )
            elif hasattr(fitted_model, "coef_"):
                coefficients = pd.Series(fitted_model.coef_[0], index=feature_names)
                best_importances = (
                    coefficients.abs()
                    .sort_values(ascending=False)
                    .head(15)
                    .rename_axis("feature")
                    .reset_index(name="absolute_coefficient")
                )

    metrics_df = pd.DataFrame(metrics_rows)
    assert best_confusion is not None
    assert best_importances is not None
    return metrics_df, best_confusion, best_importances


def write_summary(df: pd.DataFrame, descriptive_tables: dict[str, pd.DataFrame], metrics: pd.DataFrame) -> None:
    majority_rate = df["high_satisfaction"].mean()
    site_top = descriptive_tables["site_summary"].iloc[0]
    site_bottom = descriptive_tables["site_summary"].iloc[-1]

    baseline = metrics[metrics["model"] == "Majority Baseline"].iloc[0]
    best_model = metrics[metrics["model"] != "Majority Baseline"].sort_values("f1", ascending=False).iloc[0]

    summary = f"""# Tourism Analysis Summary

## Dataset
- File: `data/raw/tourism_dataset_5000.csv`
- Records: {len(df):,}
- Original fields: {df.shape[1] - 1}
- Source type: CSV open dataset about cultural tourism experiences

## Main Question Answered
Which available variables best predict whether a tourist record ends in high satisfaction (`Satisfaction >= 4`)?

## What The Data Can Actually Support
- The dataset does not include spending, country, or group size.
- It does include age, interests, accessibility, site choice, duration, system response time, recommendation accuracy, VR experience quality, tourist rating, and satisfaction.
- Because `Tourist Rating` is almost a direct proxy for `Satisfaction`, it was excluded from the final model.
- Weather cannot be merged responsibly from this dataset alone because there is no trip date, season, or location-specific timestamp for each record.

## Descriptive Findings
- High satisfaction rate overall: {majority_rate:.2%}
- Highest high-satisfaction site: {site_top['Site Name']} at {site_top['high_satisfaction_rate']:.2%}
- Lowest high-satisfaction site: {site_bottom['Site Name']} at {site_bottom['high_satisfaction_rate']:.2%}
- Satisfaction values are concentrated at 3, with smaller groups at 4 and 5.

## Predictive Findings
- Baseline accuracy from always guessing the majority class: {baseline['accuracy']:.4f}
- Best leakage-free model: {best_model['model']} with accuracy {best_model['accuracy']:.4f}, balanced accuracy {best_model['balanced_accuracy']:.4f}, and F1 {best_model['f1']:.4f}
- Interpretation: the remaining non-leaking variables provide only modest predictive signal.

## Recommendation For Poster Claims
- Use the leakage-free model as the only reported model.
- Frame the project as an investigation into which experience factors relate to higher satisfaction, not as a claim that the dataset supports a highly accurate real-world predictor by itself.
- If you want to add weather later, you need a different tourism dataset that includes trip dates and precise locations for each record.
"""

    (OUTPUTS_DIR / "analysis_summary.md").write_text(summary)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    df = load_data()
    descriptive_tables = save_descriptive_tables(df)

    plot_satisfaction_distribution(df)
    plot_high_satisfaction_by_site(df)
    plot_high_satisfaction_by_duration(df)
    plot_rating_vs_satisfaction(df)

    metrics, cm_no_rating, fi_no_rating = train_models(df, include_rating=False)
    metrics.to_csv(OUTPUTS_DIR / "model_metrics.csv", index=False)
    cm_no_rating.to_csv(OUTPUTS_DIR / "confusion_matrix_without_tourist_rating.csv")
    fi_no_rating.to_csv(OUTPUTS_DIR / "feature_importance_without_tourist_rating.csv", index=False)

    leakage_note = """Tourist Rating was excluded from the final model because it is extremely close to Satisfaction and leaks the target.
Do not use weather as a predictor with this dataset unless you first add a reliable per-record date and location field from another source.
"""
    (OUTPUTS_DIR / "leakage_note.txt").write_text(leakage_note)

    write_summary(df, descriptive_tables, metrics)


if __name__ == "__main__":
    main()
