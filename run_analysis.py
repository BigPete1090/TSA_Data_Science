import os
import shutil
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path(".").resolve() / ".mplcache")
os.environ["XDG_CACHE_HOME"] = str(Path(".").resolve() / ".cache")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
FINAL_GRAPHS_DIR = BASE_DIR / "FINAL_GRAPHS"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "data" / "hotel_bookings.csv"


def ensure_dirs() -> None:
    for path in [
        FIGURES_DIR,
        RESULTS_DIR,
        FINAL_GRAPHS_DIR,
        ARTIFACTS_DIR,
        BASE_DIR / ".mplcache",
        BASE_DIR / ".cache",
    ]:
        path.mkdir(exist_ok=True)


def load_and_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    clean = df.copy()
    clean["children"] = clean["children"].fillna(0)
    clean["country"] = clean["country"].fillna("Unknown")
    clean["total_nights"] = clean["stays_in_weekend_nights"] + clean["stays_in_week_nights"]
    clean["family_size"] = clean["adults"] + clean["children"] + clean["babies"]
    clean["is_family"] = ((clean["children"] + clean["babies"]) > 0).astype(int)
    clean["arrival_month_num"] = (
        pd.Categorical(clean["arrival_date_month"], categories=month_order, ordered=True).codes + 1
    )

    top_countries = clean["country"].value_counts().head(15).index
    clean["country_grouped"] = clean["country"].where(clean["country"].isin(top_countries), "Other")
    return clean


def save_plot(filename: str) -> None:
    target = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(target, dpi=200)
    shutil.copy2(target, FINAL_GRAPHS_DIR / filename)
    plt.close()


def create_descriptive_outputs(clean: pd.DataFrame) -> dict[str, pd.Series | float]:
    overall_cancel_rate = clean["is_canceled"].mean()
    hotel_cancel = clean.groupby("hotel")["is_canceled"].mean().sort_values(ascending=False)
    segment_cancel = clean.groupby("market_segment")["is_canceled"].mean().sort_values(ascending=False)

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=clean, x="is_canceled", hue="is_canceled", palette="Set2", legend=False)
    ax.set_title("Booking Cancellation Distribution")
    ax.set_xlabel("Canceled (0 = No, 1 = Yes)")
    ax.set_ylabel("Bookings")
    for patch in ax.patches:
        ax.annotate(
            f"{int(patch.get_height())}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_plot("cancellation_distribution.png")

    hotel_plot = clean.groupby("hotel", as_index=False)["is_canceled"].mean().sort_values(
        "is_canceled", ascending=False
    )
    hotel_plot["is_canceled"] *= 100
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=hotel_plot, x="hotel", y="is_canceled", hue="hotel", palette="viridis", legend=False)
    ax.set_title("Cancellation Rate by Hotel Type")
    ax.set_xlabel("Hotel Type")
    ax.set_ylabel("Cancellation Rate (%)")
    for patch in ax.patches:
        ax.annotate(
            f"{patch.get_height():.1f}%",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_plot("cancellation_rate_by_hotel.png")

    segment_plot = (
        clean.groupby("market_segment", as_index=False)["is_canceled"]
        .mean()
        .sort_values("is_canceled", ascending=False)
        .head(8)
    )
    segment_plot["is_canceled"] *= 100
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=segment_plot,
        x="is_canceled",
        y="market_segment",
        hue="market_segment",
        palette="mako",
        legend=False,
    )
    ax.set_title("Top Market Segments by Cancellation Rate")
    ax.set_xlabel("Cancellation Rate (%)")
    ax.set_ylabel("Market Segment")
    save_plot("cancellation_rate_by_market_segment.png")

    return {
        "overall_cancel_rate": overall_cancel_rate,
        "hotel_cancel": hotel_cancel,
        "segment_cancel": segment_cancel,
    }


def create_correlation_outputs(clean: pd.DataFrame) -> pd.DataFrame:
    numeric_corr_cols = [
        "is_canceled",
        "lead_time",
        "arrival_month_num",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "total_nights",
        "adults",
        "children",
        "babies",
        "is_repeated_guest",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "booking_changes",
        "days_in_waiting_list",
        "adr",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "family_size",
    ]

    corr = clean[numeric_corr_cols].corr(numeric_only=True)
    corr.to_csv(RESULTS_DIR / "correlation_matrix.csv")

    top_corr = corr["is_canceled"].sort_values(ascending=False).reset_index()
    top_corr.columns = ["feature", "correlation_with_is_canceled"]
    top_corr.to_csv(RESULTS_DIR / "top_correlations_with_cancellation.csv", index=False)

    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Correlation Matrix for Key Tourism Booking Variables")
    save_plot("correlation_matrix.png")
    return top_corr


def train_models(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ml = clean.copy()
    leakage_columns = ["reservation_status", "reservation_status_date"]
    drop_columns = leakage_columns + ["is_canceled", "country", "agent", "company"]

    X = ml.drop(columns=drop_columns)
    y = ml["is_canceled"]

    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=4000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=40,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    fitted_models: dict[str, Pipeline] = {}
    model_rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        fitted_models[name] = pipe
        model_rows.append(
            {
                "model": name,
                "accuracy": round(accuracy_score(y_test, pred), 4),
                "precision": round(precision_score(y_test, pred), 4),
                "recall": round(recall_score(y_test, pred), 4),
                "f1": round(f1_score(y_test, pred), 4),
                "roc_auc": round(roc_auc_score(y_test, proba), 4),
            }
        )

    metrics = pd.DataFrame(model_rows).sort_values(["roc_auc", "f1"], ascending=False).reset_index(drop=True)
    metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    best_model_name = str(metrics.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    best_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    cm_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix: {best_model_name}")
    save_plot("confusion_matrix_best_model.png")

    rf = fitted_models["Random Forest"]
    feature_names = rf.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(rf.named_steps["model"].feature_importances_, index=feature_names).sort_values(
        ascending=False
    )
    importance_df = importances.head(20).reset_index()
    importance_df.columns = ["feature", "importance"]
    importance_df.to_csv(RESULTS_DIR / "random_forest_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df.head(12),
        x="importance",
        y="feature",
        hue="feature",
        palette="rocket",
        legend=False,
    )
    plt.title("Random Forest Feature Importance")
    save_plot("random_forest_feature_importance.png")

    joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib")
    return metrics, cm_df, importance_df


def write_project_info(
    overall_cancel_rate: float,
    metrics: pd.DataFrame,
    top_corr: pd.DataFrame,
) -> None:
    best_model_name = str(metrics.iloc[0]["model"])
    summary_lines = [
        "Tourism Data Science Project Summary",
        "",
        "Dataset:",
        "- Name: Hotel Booking Demand Dataset",
        "- File: data/hotel_bookings.csv",
        "- Format: CSV",
        "- Original article: Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41-49. DOI: 10.1016/j.dib.2018.11.126",
        "- Download mirror used in this workspace: https://raw.githubusercontent.com/salves94/hotel-exploratory-data-analysis/master/hotel_bookings.csv",
        "",
        "Research questions:",
        "1. Which hotel booking factors are most associated with cancellation?",
        "2. Can machine learning predict booking cancellation accurately from booking-time tourism variables?",
        "",
        f"Overall cancellation rate: {overall_cancel_rate:.2%}",
        f"Best model: {best_model_name}",
        f"Best model accuracy: {metrics.iloc[0]['accuracy']:.4f}",
        f"Best model F1: {metrics.iloc[0]['f1']:.4f}",
        f"Best model ROC AUC: {metrics.iloc[0]['roc_auc']:.4f}",
        "",
        "Top numeric correlations with cancellation:",
    ]

    for _, row in top_corr.head(8).iterrows():
        summary_lines.append(f"- {row['feature']}: {row['correlation_with_is_canceled']:.4f}")

    summary_lines += [
        "",
        "Leakage prevention:",
        "- reservation_status and reservation_status_date were excluded from machine learning because they directly reflect the booking outcome.",
        "",
        "Main output files:",
        "- figures/cancellation_distribution.png",
        "- figures/cancellation_rate_by_hotel.png",
        "- figures/cancellation_rate_by_market_segment.png",
        "- figures/correlation_matrix.png",
        "- figures/confusion_matrix_best_model.png",
        "- figures/random_forest_feature_importance.png",
        "- results/model_metrics.csv",
        "- results/top_correlations_with_cancellation.csv",
        "- results/random_forest_feature_importance.csv",
        "- results/confusion_matrix.csv",
        "- artifacts/best_model.joblib",
    ]
    (BASE_DIR / "project_info.txt").write_text("\n".join(summary_lines))


def main() -> None:
    sns.set_theme(style="whitegrid")
    pd.set_option("display.max_columns", None)
    ensure_dirs()

    clean = load_and_clean_data()
    descriptive = create_descriptive_outputs(clean)
    top_corr = create_correlation_outputs(clean)
    metrics, _, _ = train_models(clean)
    write_project_info(float(descriptive["overall_cancel_rate"]), metrics, top_corr)

    print("Analysis complete.")
    print(f"Best model: {metrics.iloc[0]['model']}")
    print(f"Accuracy: {metrics.iloc[0]['accuracy']:.4f}")
    print(f"F1: {metrics.iloc[0]['f1']:.4f}")
    print(f"ROC AUC: {metrics.iloc[0]['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
