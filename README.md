
# Retail Sales Forecasting & Demand Prediction

## Project Overview
This project analyzes historical retail sales data to identify sales trends, seasonal patterns, and product demand. It uses time-series forecasting techniques to predict future product sales and provides an interactive dashboard using Streamlit.

## Objectives
- Analyze historical retail sales data.
- Identify seasonal trends and demand patterns.
- Forecast future product demand.
- Compare actual sales with predicted sales.
- Visualize sales performance through an interactive dashboard.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Plotly
- Scikit-learn
- Statsmodels
- Streamlit

## Forecasting Model
Holt-Winters Exponential Smoothing is used to forecast future weekly product demand.

A seasonal naive forecasting baseline is included as a fallback.

## Model Evaluation
The model is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

The last 8 weeks of historical data are reserved for testing.

## Dashboard Features
- Sales performance KPIs
- Daily and monthly sales analysis
- Top-performing products
- Store-wise performance
- Seasonal demand analysis
- Future sales forecasting
- Downloadable forecast results

## Dataset
Store Item Demand Forecasting Dataset from Kaggle.

https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data

Download train.csv and place it inside the data folder.

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```text
Retail-Sales-Forecasting/
├── data/
│   └── train.csv
├── notebooks/
│   └── retail_sales_analysis.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Author
Data Science Portfolio Project

## Disclaimer
Forecasts are estimates based on historical sales patterns and should not be treated as guaranteed future demand.
