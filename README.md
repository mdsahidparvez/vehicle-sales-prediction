# Vehicle Sales vs Fuel Price Analysis (Canada)

## Overview
This project analyzes the relationship between fuel prices and new motor vehicle
sales in Canada (2001–2022). It applies exploratory data analysis and machine
learning models to understand consumer behavior and predict future sales.

## Dataset
- Vehicle sales data by province and vehicle type
- Monthly average retail gasoline prices
- Source: Statistics Canada

## Technologies Used
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter Notebook
- Power BI (data cleaning)
- Machine Learning models

## Models Implemented
- Linear Regression
- Gradient Boosting Regressor
- Random Forest Regressor

## Model Performance

| Model | RMSE | MAPE |
|------|------|------|
| Linear Regression | 4210.59 | 1.51% |
| Gradient Boosting | 2022.52 | 0.51% |
| Random Forest | **1298.60** | **0.11%** |

## Key Insights
- Fuel price increases negatively affect passenger car sales
- Truck sales are less sensitive to fuel price increases
- Random Forest produced the most accurate predictions


## Project Structure

1. Data collection from Statistics Canada
2. Data cleaning and preprocessing (Power BI + Python)
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Model training:
   - Linear Regression
   - Gradient Boosting
   - Random Forest
6. Model evaluation and comparison
7. Future sales prediction
