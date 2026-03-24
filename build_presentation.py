from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def bullet_list(items: list[str]) -> str:
    return "".join(f"<li>{item}</li>" for item in items)


def table_html(df: pd.DataFrame, rounded_cols: list[str] | None = None) -> str:
    frame = df.copy()
    if rounded_cols:
        for col in rounded_cols:
            frame[col] = frame[col].map(lambda value: f"{value:.4f}")
    return frame.to_html(index=False, classes="data-table", border=0)


def main() -> None:
    metrics = pd.read_csv(BASE_DIR / "results" / "model_metrics.csv")
    top_corr = pd.read_csv(BASE_DIR / "results" / "top_correlations_with_cancellation.csv")
    importance = pd.read_csv(BASE_DIR / "results" / "random_forest_feature_importance.csv")
    cm = pd.read_csv(BASE_DIR / "results" / "confusion_matrix.csv", index_col=0)
    info_lines = (BASE_DIR / "project_info.txt").read_text().splitlines()

    overall_line = next(line for line in info_lines if line.startswith("Overall cancellation rate:"))
    best_model_line = next(line for line in info_lines if line.startswith("Best model:"))
    accuracy_line = next(line for line in info_lines if line.startswith("Best model accuracy:"))
    f1_line = next(line for line in info_lines if line.startswith("Best model F1:"))
    roc_line = next(line for line in info_lines if line.startswith("Best model ROC AUC:"))

    key_findings = [
        "City Hotels cancel more often than Resort Hotels.",
        "Long lead times are strongly associated with more cancellations.",
        "Bookings with more special requests and parking spaces are less likely to cancel.",
        "Random Forest performed best across the compared models.",
    ]

    feature_talking_points = [
        "lead_time is the strongest positive numeric predictor of cancellation.",
        "previous_cancellations also increases the chance of future cancellation.",
        "total_of_special_requests and required_car_parking_spaces are strong negative signals.",
        "deposit_type, market_segment, and customer_type matter heavily in the model.",
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TSA Tourism Data Science Presentation</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: #fffaf2;
      --ink: #1f2a2e;
      --muted: #5d6a6e;
      --accent: #0f766e;
      --accent-2: #b45309;
      --line: #d9cdbb;
      --shadow: 0 18px 40px rgba(31, 42, 46, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.15), transparent 25%),
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 25%),
        var(--bg);
    }}
    .wrap {{
      width: min(1200px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 64px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(15, 118, 110, 0.95), rgba(31, 42, 46, 0.96));
      color: white;
      border-radius: 28px;
      padding: 36px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.6rem);
      line-height: 1;
    }}
    .hero p {{
      max-width: 760px;
      margin: 0;
      font-size: 1.08rem;
      color: rgba(255,255,255,0.9);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin: 18px 0 28px;
    }}
    .card, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .card {{
      padding: 20px;
    }}
    .label {{
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .value {{
      font-size: 2rem;
      font-weight: 700;
    }}
    .section {{
      padding: 24px;
      margin-bottom: 20px;
    }}
    h2 {{
      margin-top: 0;
      font-size: 1.6rem;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 20px;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
      line-height: 1.6;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    figure {{
      margin: 0;
      background: white;
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
    }}
    figure img {{
      display: block;
      width: 100%;
      height: auto;
      background: white;
    }}
    figcaption {{
      padding: 12px 14px 16px;
      font-size: 0.96rem;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
    }}
    .data-table th {{
      color: var(--muted);
      font-weight: 700;
    }}
    .callout {{
      padding: 16px 18px;
      border-left: 5px solid var(--accent-2);
      background: rgba(180, 83, 9, 0.08);
      border-radius: 14px;
      margin-top: 16px;
    }}
    .footer {{
      margin-top: 22px;
      color: var(--muted);
      font-size: 0.94rem;
    }}
    code {{
      font-family: "Courier New", monospace;
      background: rgba(15, 118, 110, 0.08);
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Tourism Booking Cancellation Analysis</h1>
      <p>This presentation-ready view summarizes the dataset, the most useful features, and the machine learning results for a TSA Data Science project on hotel booking cancellations.</p>
    </section>

    <section class="grid">
      <div class="card">
        <div class="label">Dataset</div>
        <div class="value">119,390</div>
        <div>hotel bookings</div>
      </div>
      <div class="card">
        <div class="label">Cancellation Rate</div>
        <div class="value">{overall_line.split(': ', 1)[1]}</div>
        <div>overall target rate</div>
      </div>
      <div class="card">
        <div class="label">Best Model</div>
        <div class="value" style="font-size:1.4rem">{best_model_line.split(': ', 1)[1]}</div>
        <div>highest ROC AUC</div>
      </div>
      <div class="card">
        <div class="label">Model Quality</div>
        <div class="value" style="font-size:1.2rem">AUC {roc_line.split(': ', 1)[1]}</div>
        <div>{accuracy_line.split(': ', 1)[1]} accuracy, {f1_line.split(': ', 1)[1]} F1</div>
      </div>
    </section>

    <section class="section two-col">
      <div>
        <h2>Research Focus</h2>
        <ul>
          <li>Which hotel booking factors are most associated with cancellation?</li>
          <li>Can machine learning predict cancellation from booking-time variables?</li>
        </ul>
        <div class="callout">
          Leakage features <code>reservation_status</code> and <code>reservation_status_date</code> were excluded so the model does not cheat.
        </div>
      </div>
      <div>
        <h2>Key Findings</h2>
        <ul>{bullet_list(key_findings)}</ul>
      </div>
    </section>

    <section class="section two-col">
      <div>
        <h2>Most Useful Features</h2>
        <ul>{bullet_list(feature_talking_points)}</ul>
      </div>
      <div>
        <h2>Top Correlations With Cancellation</h2>
        {table_html(top_corr.head(8), ["correlation_with_is_canceled"])}
      </div>
    </section>

    <section class="section">
      <h2>Model Comparison</h2>
      {table_html(metrics, ["accuracy", "precision", "recall", "f1", "roc_auc"])}
    </section>

    <section class="section two-col">
      <div>
        <h2>Best Model Confusion Matrix</h2>
        {cm.to_html(classes="data-table", border=0)}
      </div>
      <div>
        <h2>Top Random Forest Features</h2>
        {table_html(importance.head(10), ["importance"])}
      </div>
    </section>

    <section class="section">
      <h2>Visual Evidence</h2>
      <div class="gallery">
        <figure>
          <img src="figures/cancellation_rate_by_hotel.png" alt="Cancellation rate by hotel type">
          <figcaption>City Hotels cancel more often than Resort Hotels.</figcaption>
        </figure>
        <figure>
          <img src="figures/correlation_matrix.png" alt="Correlation matrix">
          <figcaption>The correlation matrix highlights the strongest usable numeric signals.</figcaption>
        </figure>
        <figure>
          <img src="figures/confusion_matrix_best_model.png" alt="Confusion matrix">
          <figcaption>The Random Forest model balances identifying cancellations and non-cancellations well.</figcaption>
        </figure>
        <figure>
          <img src="figures/random_forest_feature_importance.png" alt="Feature importance">
          <figcaption>Feature importance confirms that booking commitment signals matter most.</figcaption>
        </figure>
      </div>
    </section>

    <section class="section">
      <h2>Presentation Script</h2>
      <ul>
        <li>Start with the business problem: hotel cancellations hurt planning and revenue management.</li>
        <li>Explain that the dataset covers 119,390 bookings and the target is whether a booking was canceled.</li>
        <li>Point to the correlation matrix and highlight <code>lead_time</code>, <code>special_requests</code>, and <code>parking_spaces</code>.</li>
        <li>Use the model table to show that Random Forest was the strongest overall performer.</li>
        <li>End with the conclusion that cancellation is predictable enough to support smarter tourism operations.</li>
      </ul>
    </section>

    <div class="footer">
      Rebuild everything with <code>.venv/bin/python run_analysis.py</code> and <code>.venv/bin/python build_presentation.py</code>.
    </div>
  </div>
</body>
</html>
"""
    (BASE_DIR / "presentation.html").write_text(html)
    print("Created presentation.html")


if __name__ == "__main__":
    main()
