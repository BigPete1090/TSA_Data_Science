# TSA Data Science 2026

This repository contains a tourism route analysis project that studies how personalization relates to tourist satisfaction and exploration diversity.

## Project Overview

The analysis is built around two questions:

1. Does higher personalization increase satisfaction?
2. Does higher personalization reduce exploration diversity?

`tsa_analysis.py` loads the dataset, detects relevant columns, engineers personalization and diversity features, tests the hypotheses, trains random forest models, and exports chart images for reporting.

## Files

- `tsa_analysis.py`: main analysis script
- `tourism_route_dataset.csv`: source dataset
- `requirements.txt`: required Python packages
- `artifacts/`: saved project artifacts
- `chart*.png`: generated output figures

## Installation

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Usage

Run the analysis with:

```bash
python tsa_analysis.py
```

The script expects `tourism_route_dataset.csv` to be in the repository root and saves generated charts to the same directory.

## Outputs

Running the script produces:

- hypothesis test results for satisfaction and diversity
- personalization tier summaries
- predictive model metrics for satisfaction and diversity
- chart exports showing distributions, correlations, and feature importance

## Data Source

SmartTourRoutePlanner: Tourism Route Dataset  
Kaggle uploader: `ziya07`  
<https://www.kaggle.com/datasets/ziya07/smarttourrouteplanner-tourism-route-dataset>
