# Tourism Analysis Summary

## Dataset
- File: `data/raw/tourism_dataset_5000.csv`
- Records: 5,000
- Original fields: 14
- Source type: CSV open dataset about cultural tourism experiences

## Main Question Answered
Which available variables best predict whether a tourist record ends in high satisfaction (`Satisfaction >= 4`)?

## What The Data Can Actually Support
- The dataset does not include spending, country, or group size.
- It does include age, interests, accessibility, site choice, duration, system response time, recommendation accuracy, VR experience quality, tourist rating, and satisfaction.
- Because `Tourist Rating` is almost a direct proxy for `Satisfaction`, it was excluded from the final model.
- Weather cannot be merged responsibly from this dataset alone because there is no trip date, season, or location-specific timestamp for each record.

## Descriptive Findings
- High satisfaction rate overall: 38.78%
- Highest high-satisfaction site: Eiffel Tower at 40.45%
- Lowest high-satisfaction site: Great Wall of China at 37.12%
- Satisfaction values are concentrated at 3, with smaller groups at 4 and 5.

## Predictive Findings
- Baseline accuracy from always guessing the majority class: 0.6120
- Best leakage-free model: Decision Tree with accuracy 0.4560, balanced accuracy 0.5207, and F1 0.5358
- Interpretation: the remaining non-leaking variables provide only modest predictive signal.

## Recommendation For Poster Claims
- Use the leakage-free model as the only reported model.
- Frame the project as an investigation into which experience factors relate to higher satisfaction, not as a claim that the dataset supports a highly accurate real-world predictor by itself.
- If you want to add weather later, you need a different tourism dataset that includes trip dates and precise locations for each record.
