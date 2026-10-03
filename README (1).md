# Week 2 Task: Exploratory Data Analysis and Visualization

## Overview
This project performs exploratory data analysis (EDA) and visualization on the
Titanic passenger dataset, using the cleaned dataset produced in the Week 1 task
as its starting point (891 rows, 0 missing values).

## Contents
| File | Description |
|---|---|
| `eda_analysis.py` | Full EDA script: summary statistics, univariate/bivariate/multivariate analysis, and 13 annotated charts |
| `titanic_cleaned.csv` | Input dataset (cleaned in Week 1) |
| `chart_01...chart_13.png` | All 13 generated figures |
| `Week2_EDA_Visualization_Report.docx` | Full written report with narrative, code snippets, and visuals |

## How to run
```bash
pip install pandas numpy matplotlib seaborn
python eda_analysis.py
```

## Key findings
- Overall survival rate: 38.4% (342 of 891 passengers)
- Sex was the strongest single factor: women survived at 74.2% vs. 18.9% for men
- Class compounded this: 1st-class women survived at 96.8%, 3rd-class men at 13.5%
- Children (0-12) had the highest survival rate (58.0%) of any age group
- Moderate family size (2-4) outperformed travelling alone or in very large groups
- Embarkation port differences are most plausibly explained by class composition
  (confounding), not a direct effect of the port itself

Full methodology, code, and critical discussion are in `Week2_EDA_Visualization_Report.docx`.
