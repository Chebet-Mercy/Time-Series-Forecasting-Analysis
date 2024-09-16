# Time Series Regression for Accurate Demand Forecasting: Enhancing Inventory Management at Corporation Favorita
## Overview
Corporation Favorita, a leading retail chain in Latin America, operates a diverse range of stores including supermarkets, hypermarkets, and specialized outlets. To optimize inventory management and maintain profitability, effective forecasting of future product demand is essential. This project aims to develop a machine learning model to predict the optimal quantity of products that should be in stock across various store locations, leveraging historical sales data and advanced forecasting techniques.

## Project Goals
- Develop accurate forecasting models to predict future product demand.
- Enhance inventory management to prevent overstocking and understocking.
- Improve operational efficiency across multiple store locations.

## Introduction

This project focuses on predicting future trends using historical time series data. By applying various models, we aim to uncover patterns, trends, and seasonality in the data to make accurate forecasts. The project covers data preparation, exploratory data analysis, model training, evaluation, and visualization.

## Project Structure

```

├── data
│   ├── raw_data.csv
│   ├── processed_data.csv
├── notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_visualization.ipynb
├── src
│   ├── data_preparation.py
│   ├── eda.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── visualization.py
├── README.md
└── requirements.txt

```

## Data Preparation

Data preparation is crucial for the success of any time series analysis. The following steps were performed:

- **Loading Data**: The raw data was loaded into a DataFrame.
- **Handling Missing Values**: Missing values were imputed or removed to maintain data integrity.
- **Feature Engineering**: Additional features like lagged variables and rolling statistics were created.
- **Data Splitting**: The data was split into training and test sets for model evaluation.

## Exploratory Data Analysis (EDA)

EDA was conducted to understand the dataset better:

- **Trend Analysis**: Identified the overall trend in the data.
- **Seasonality Detection**: Detected monthly and yearly seasonality patterns.
- **Correlation Analysis**: Analyzed correlations between features.

## Modeling

Four advanced models were selected for forecasting:

### ARIMA

ARIMA (AutoRegressive Integrated Moving Average) is a classic time series model that captures the trend and autocorrelation in the data.

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit the ARIMA model
arima_model = ARIMA(y_train, order=(5,1,0))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(y_test))
```

### SARIMA

SARIMA (Seasonal ARIMA) extends ARIMA by accounting for seasonality.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit the SARIMA model
sarima_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit()
sarima_forecast = sarima_fit.forecast(steps=len(y_test))
```

### XGBOOST

XGBOOST (Extreme Gradient Boosting) is a powerful machine learning model that outperformed traditional time series models.

```python
from xgboost import XGBRegressor

# Fit the XGBOOST model
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_forecast = xgb_model.predict(X_test)
```

## Model Evaluation

The performance of each model was evaluated using metrics such as MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and RMSLE (Root Mean Squared Logarithmic Error).

```python
# Evaluation Code
data = {
    'Model': ['ARIMA', 'SARIMA', 'XGBOOST'],
    'MSE': [arima_metrics['MSE'], sarima_metrics['MSE'], xgboost_metrics['MSE']],
    'RMSE': [arima_metrics['RMSE'], sarima_metrics['RMSE'], xgboost_metrics['RMSE']],
    'RMSLE': [arima_metrics['RMSLE'], sarima_metrics['RMSLE'], xgboost_metrics['RMSLE']]
}

performance_df = pd.DataFrame(data)
```

## Visualization

Advanced visualizations were created to compare the forecasted values with the actual data and analyze error distributions.

### Forecasted vs. Actual Values

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=y_test.index, y=xgb_forecast, mode='lines', name='XGBOOST Forecast'))

fig.update_layout(title='Actual vs. Forecasted Values', xaxis_title='Date', yaxis_title='Value')
fig.show()
```

### Error Distributions

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.histplot(xgb_forecast - y_test, kde=True, color='blue', bins=30)
plt.title('Error Distribution for XGBOOST Model')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()
```

## Insights and Conclusions

- **Model Selection**: XGBOOST outperformed traditional models, demonstrating the power of machine learning in time series forecasting.
- **Data Preparation**: Rigorous preprocessing and feature engineering significantly enhanced model performance.
- **Business Implications**: Accurate forecasting can optimize inventory management, financial planning, supply chain operations, and strategic decision-making.

### Recommendations

- **Enhanced Forecasting Models**: Refine models by incorporating seasonal trends, oil price fluctuations, and holiday effects.
- **Inventory Optimization**: Apply forecasting models to optimize stock levels, reducing overstock and stockouts.
- **Promotional Strategies**: Develop targeted promotions based on sales trends and their impact.
- **Continuous Monitoring**: Regularly update models with new data and monitor key variables.
- **Data-Driven Decisions**: Utilize data insights to make informed decisions, enhancing efficiency and customer satisfaction.

## Future Work

- **Hybrid Models**: Combining strengths of different models for enhanced accuracy.
- **Deep Learning**: Exploring LSTM and RNN models for capturing complex temporal dependencies.
- **Real-time Forecasting**: Implementing real-time forecasting systems for dynamic updates.
