# Cultural Tourism Analysis

This project analyzes an open cultural tourism dataset and builds prediction models for high tourist satisfaction. The analysis is set up so you can reuse the files later for your TSA poster and claims.

## Project Structure
- `data/raw/tourism_dataset_5000.csv`: raw CSV used for analysis
- `scripts/analyze_tourism.py`: reproducible analysis and modeling script
- `outputs/`: generated tables, metrics, summary, and confusion matrices
- `outputs/figures/`: poster-ready charts
- `PROJECT_NOTES.txt`: plain-language notes about the research questions, findings, and what you can safely claim

## How To Rerun
Use the local virtual environment that was created for this project:

```bash
.venv/bin/python scripts/analyze_tourism.py
```

## Important Method Note
`Tourist Rating` is extremely close to the target variable `Satisfaction`. If you include it as a predictor, the model is leaked. The final pipeline excludes `Tourist Rating` and only reports leakage-free results.

## About Weather Or Other External Predictors
This dataset does not contain trip dates or time-specific location data, so weather cannot be merged in a defensible way yet. If you want weather later, you need a tourism dataset with per-record date and location fields.
