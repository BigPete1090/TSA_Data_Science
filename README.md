# TSA Tourism Data Science Project

This project is now set up so you can use it without presenting from the notebook.

## What To Use

- `presentation.html`: the presentation-ready page you can open in a browser.
- `run_analysis.py`: reruns the full analysis and recreates the figures, CSVs, and saved model.
- `build_presentation.py`: rebuilds `presentation.html` from the latest project outputs.
- `artifacts/best_model.joblib`: saved best model pipeline.

## Presentation-Day Commands

Run these from the project folder:

```bash
.venv/bin/python run_analysis.py
.venv/bin/python build_presentation.py
```

Then open `presentation.html` in your browser.

If you want a local server instead of opening the file directly:

```bash
.venv/bin/python -m http.server 8000
```

Then visit `http://localhost:8000/presentation.html`

## Main Talking Points

- The project predicts hotel booking cancellations using tourism booking data.
- The strongest useful features are `lead_time`, `deposit_type`, `total_of_special_requests`, `required_car_parking_spaces`, `previous_cancellations`, `market_segment`, and `customer_type`.
- Random Forest is the best model in this project.
- Leakage variables were excluded so the model stays honest.
