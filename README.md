# BudgetVision
Automated Budget Variance &amp; Forecasting Tool built with Streamlit.

---

BudgetVision is an automated **budget variance analysis and forecasting tool** built with **Python and Streamlit**. It transforms a simple quarterly budget dataset into interactive insights, forecasts, and anomaly detection without requiring manual spreadsheet analysis.

---

## Features

* 📊 **Variance Analysis** – Compare Budget vs Actual with annual, YoY, and QoQ metrics
* 📈 **Forecasting Models** – Exponential Smoothing, Holt, Holt-Winters, SARIMA, and Trend–Seasonal Decomposition
* 🔍 **Anomaly Detection** – Detect unusual budget behavior using Rolling Z-Score and Isolation Forest
* 📉 **Seasonality Analysis** – Identify recurring quarterly spending patterns
* 🌐 **Interactive Dashboard** – Built with Streamlit for easy exploration of results

---

## Dataset Format

The application expects a CSV file with the following columns:

```
Fiscal Year | Quarter | Budget | Actual
```

Example dataset included: `quarterly_government_budget.csv`.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Statsmodels
* Scikit-Learn
* Matplotlib & Seaborn
* Streamlit

---

## Installation & Running the App

1. Clone the repository

```
git clone https://github.com/pranjalshrestha/BudgetVision.git
cd BudgetVision
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the Streamlit app

```
streamlit run app.py
```

The application will open in your browser.

---

## Project Structure

```
BudgetVision/
│
├── app.py                          # Main Streamlit application
├── quarterly_government_budget.csv # Example dataset
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
```

---

## Overview

BudgetVision converts a quarterly budget dataset into forecasting models, variance metrics, seasonal insights, and anomaly detection results to help analysts quickly understand financial trends and unusual patterns.
