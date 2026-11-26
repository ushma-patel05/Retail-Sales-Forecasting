
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Sales Forecasting & Demand Prediction")
st.markdown(
    "Analyze historical sales, discover demand patterns "
    "and forecast future product demand."
)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    required = {"date", "store", "item", "sales"}

    if not required.issubset(df.columns):
        raise ValueError(
            "Dataset must contain date, store, item and sales columns."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["store"] = pd.to_numeric(df["store"], errors="coerce")
    df["item"] = pd.to_numeric(df["item"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    df = df.dropna(subset=list(required))
    df = df[df["sales"] >= 0]

    df["store"] = df["store"].astype(int)
    df["item"] = df["item"].astype(int)

    df = df.sort_values("date")

    if df.empty:
        raise ValueError("No valid records found in the dataset.")

    return df


uploaded_file = st.sidebar.file_uploader(
    "Upload retail sales CSV",
    type=["csv"]
)

data_path = Path("data/train.csv")

if uploaded_file is not None:
    source = uploaded_file
elif data_path.exists():
    source = data_path
else:
    st.warning(
        "Please place train.csv inside the data folder "
        "or upload your CSV using the sidebar."
    )
    st.stop()

try:
    df = load_data(source)
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.header("Dashboard Filters")

stores = sorted(df["store"].unique())
selected_store = st.sidebar.selectbox(
    "Select Store",
    stores
)

store_df = df[df["store"] == selected_store]

items = sorted(store_df["item"].unique())
selected_item = st.sidebar.selectbox(
    "Select Product",
    items
)

filtered_df = store_df[
    store_df["item"] == selected_item
].copy()

# ---------------- KPI SECTION ----------------

st.subheader("Sales Performance Overview")

col1, col2, col3, col4 = st.columns(4)

total_units = int(df["sales"].sum())
total_records = len(df)
total_stores = df["store"].nunique()
total_products = df["item"].nunique()

col1.metric("Total Units Sold", f"{total_units:,}")
col2.metric("Total Records", f"{total_records:,}")
col3.metric("Total Stores", total_stores)
col4.metric("Total Products", total_products)

st.divider()

# ---------------- SALES TREND ----------------

st.subheader("1. Historical Sales Analysis")

daily_sales = (
    filtered_df.groupby("date")["sales"]
    .sum()
    .reset_index()
)

fig = px.line(
    daily_sales,
    x="date",
    y="sales",
    title=f"Daily Sales - Store {selected_store}, "
          f"Product {selected_item}",
    labels={"sales": "Units Sold", "date": "Date"}
)

fig.update_traces(line=dict(width=2))
fig.update_layout(hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# ---------------- MONTHLY TREND ----------------

st.subheader("2. Monthly Sales Trends")

monthly_sales = (
    filtered_df.set_index("date")
    .resample("ME")["sales"]
    .sum()
    .reset_index()
)

monthly_sales["month"] = monthly_sales["date"].dt.strftime(
    "%b %Y"
)

fig = px.bar(
    monthly_sales,
    x="month",
    y="sales",
    title="Monthly Product Demand",
    labels={"sales": "Units Sold", "month": "Month"}
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- PRODUCT PERFORMANCE ----------------

st.subheader("3. Top Performing Products")

top_products = (
    store_df.groupby("item")["sales"]
    .sum()
    .nlargest(10)
    .reset_index()
)

fig = px.bar(
    top_products,
    x="item",
    y="sales",
    title=f"Top 10 Products - Store {selected_store}",
    labels={"item": "Product ID", "sales": "Units Sold"}
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- SEASONAL ANALYSIS ----------------

st.subheader("4. Seasonal Demand Analysis")

seasonal_df = filtered_df.copy()

seasonal_df["day_name"] = (
    seasonal_df["date"].dt.day_name()
)

seasonal_df["month"] = (
    seasonal_df["date"].dt.month_name()
)

weekly_demand = (
    seasonal_df.groupby("day_name")["sales"]
    .mean()
    .reindex([
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ])
    .reset_index()
)

fig = px.bar(
    weekly_demand,
    x="day_name",
    y="sales",
    title="Average Demand by Day of Week",
    labels={"sales": "Average Units Sold"}
)

st.plotly_chart(fig, use_container_width=True)

monthly_demand = (
    seasonal_df.groupby("month")["sales"]
    .mean()
    .reindex([
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ])
    .reset_index()
)

fig = px.line(
    monthly_demand,
    x="month",
    y="sales",
    markers=True,
    title="Average Demand by Month"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- FORECASTING MODEL ----------------

st.divider()
st.subheader("5. Future Sales Forecasting")

forecast_weeks = st.slider(
    "Select forecast horizon (weeks)",
    min_value=4,
    max_value=16,
    value=8
)

weekly_series = (
    filtered_df.set_index("date")["sales"]
    .resample("W") 
    .sum()
    .asfreq("W")
    .fillna(0)
)

if len(weekly_series) < 60:
    st.warning(
        "Not enough weekly observations for reliable "
        "seasonal forecasting. Select another product."
    )
    st.stop()

# Keep the final 8 weeks for testing
train = weekly_series.iloc[:-8]
test = weekly_series.iloc[-8:]

try:
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=52,
        initialization_method="estimated"
    )

    fitted_model = model.fit(optimized=True)
    test_predictions = fitted_model.forecast(8)

    # Fit on the complete dataset for future forecasting
    final_model = ExponentialSmoothing(
        weekly_series,
        trend="add",
        seasonal="add",
        seasonal_periods=52,
        initialization_method="estimated"
    ).fit(optimized=True)

    future_forecast = final_model.forecast(forecast_weeks)

    model_name = "Holt-Winters Exponential Smoothing"

except Exception:
    # Seasonal naive fallback
    test_predictions = pd.Series(
        [
            train.iloc[-52 + i]
            for i in range(8)
        ],
        index=test.index
    )

    future_forecast = pd.Series(
        [
            weekly_series.iloc[-52 + (i % 52)]
            for i in range(forecast_weeks)
        ],
        index=pd.date_range(
            weekly_series.index[-1] + pd.Timedelta(weeks=1),
            periods=forecast_weeks,
            freq="W"
        )
    )

    model_name = "Seasonal Naive Baseline"

# ---------------- MODEL EVALUATION ----------------

mae = mean_absolute_error(test, test_predictions)

rmse = np.sqrt(
    mean_squared_error(test, test_predictions)
)

nonzero = test != 0

if nonzero.any():
    mape = (
        np.abs(
            (test[nonzero] - test_predictions[nonzero])
            / test[nonzero]
        ).mean() * 100
    )
else:
    mape = np.nan

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric(
    "MAPE",
    f"{mape:.2f}%" if np.isfinite(mape) else "N/A"
)

st.caption(f"Forecasting method: {model_name}")
st.caption(
    "Evaluation uses the final 8 weeks as a chronological "
    "holdout set. Metrics are calculated before final refitting."
)

# ---------------- FORECAST VISUALIZATION ----------------

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=weekly_series.index,
    y=weekly_series.values,
    name="Historical Sales",
    mode="lines"
))

fig.add_trace(go.Scatter(
    x=test.index,
    y=test_predictions.values,
    name="Test Predictions",
    mode="lines",
    line=dict(dash="dash")
))

fig.add_trace(go.Scatter(
    x=future_forecast.index,
    y=future_forecast.values,
    name="Future Forecast",
    mode="lines",
    line=dict(width=3, dash="dot")
))

fig.update_layout(
    title="Historical Sales vs Predicted Sales",
    xaxis_title="Week",
    yaxis_title="Units Sold",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- FORECAST TABLE ----------------

st.subheader("6. Future Demand Predictions")

forecast_table = pd.DataFrame({
    "Forecast Date": future_forecast.index.strftime("%Y-%m-%d"),
    "Predicted Units": np.maximum(
        0, np.round(future_forecast.values)
    ).astype(int)
})

st.dataframe(
    forecast_table,
    use_container_width=True,
    hide_index=True
)

csv = forecast_table.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Forecast CSV",
    data=csv,
    file_name="retail_sales_forecast.csv",
    mime="text/csv"
)

# ---------------- FOOTER ----------------

st.divider()

)
