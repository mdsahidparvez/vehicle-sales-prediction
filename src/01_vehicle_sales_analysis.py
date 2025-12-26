# -*- coding: utf-8 -*-
"""
Vehicle Sales vs Fuel Price Analysis (Canada)

Author: Md Sahid Parvez
Course: Data Mining (MGSC-5126-23)
Institution: Cape Breton University

Objective:
Analyze the relationship between fuel prices and motor vehicle sales
and predict future sales using ML models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# -----------------------------
# Data Loading
# -----------------------------

df1 = pd.read_csv("motor_vehicle_sale.csv")
df2 = pd.read_csv("gas_price.csv")

# -----------------------------
# Data Cleaning & Merging
# -----------------------------

df2 = pd.concat([df2] * 2, ignore_index=True)

df1.sort_values(by=['Year', 'Month', 'GEO'], inplace=True)
df1.reset_index(drop=True, inplace=True)

df2.sort_values(by=['Year', 'Month', 'GEO'], inplace=True)
df2.reset_index(drop=True, inplace=True)

df1['Fuel Price(In Cents)'] = df2['Price']

# -----------------------------
# Feature Formatting
# -----------------------------

months_order = [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December'
]

df1['Month'] = pd.Categorical(df1['Month'], categories=months_order, ordered=True)
df1.sort_values(by=['Year', 'Month'], inplace=True)
df1.reset_index(drop=True, inplace=True)

df1 = df1[df1['Year'] != 2023]
final_df = df1.copy()

final_df['Date'] = pd.to_datetime(
    final_df['Year'].astype(str) + '-' + final_df['Month'].astype(str)
)

# -----------------------------
# Exploratory Data Analysis
# -----------------------------

avg_sales = final_df.groupby(['Date', 'Vehicle type'])['Number Of Sales'].mean().unstack()

plt.figure(figsize=(12,6))
for col in avg_sales.columns:
    plt.plot(avg_sales.index, avg_sales[col], label=col)
plt.legend()
plt.title("Average Monthly Sales by Vehicle Type")
plt.show()

# -----------------------------
# Correlation Analysis
# -----------------------------

oil_col = 'Fuel Price(In Cents)'
sales_col = 'Number Of Sales'

corr_passenger = final_df[final_df['Vehicle type']=='Passenger cars'][[oil_col, sales_col]].corr().iloc[0,1]
corr_truck = final_df[final_df['Vehicle type']=='Trucks'][[oil_col, sales_col]].corr().iloc[0,1]

print("Passenger Cars Correlation:", corr_passenger)
print("Trucks Correlation:", corr_truck)

# -----------------------------
# Feature Engineering
# -----------------------------

X = final_df[['Year', 'GEO', 'Vehicle type', 'Fuel Price(In Cents)', 'Month']]
y = final_df['Number Of Sales']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Evaluation Metrics
# -----------------------------

def evaluate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

# -----------------------------
# Model Training & Evaluation
# -----------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    preds = model.predict(X_test)

    rmse = evaluate_rmse(y_test, preds)
    mape = evaluate_mape(y_test, preds)

    print(f"\n{name}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}")
    print(f"Training Time: {train_time}")

