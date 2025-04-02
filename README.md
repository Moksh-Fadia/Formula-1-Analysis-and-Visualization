# Formula 1 Analysis, Visualization, and Prediction Project

This project is a comprehensive Formula 1 data analysis, visualization, and prediction application built using **Python**, **Streamlit**, **Pandas**, **Matplotlib**, **Seaborn**, **FastF1**, and **Scikit-Learn**.

## Overview

The project is divided into three major parts:
1. **Data Analysis** - Exploring and cleaning Formula 1 datasets for insights.
2. **Visualization** - Creating interactive charts and plots to understand trends and performance.
3. **Prediction Model** - Implementing a prediction model to forecast race outcomes using machine learning.

## Features

- **Race Analysis:** Analyze previous year's race data to predict this year's race winner.
- **Visualization:** Interactive graphs for analyzing driver performance, team performance, qualifying times, pit stops, and more.
- **Prediction Model:** Predict race results using a Gradient Boosting model trained on previous race data.
- **User Interface:** Streamlit-based UI with tabs for seamless navigation between analysis, visualization, and prediction.

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Moksh-Fadia/Formula-1-Analysis-and-Visualization.git
cd Formula-1-Analysis-and-Visualization


-> Usage:
Run the application with Streamlit:

streamlit run prediction.py
The application will be hosted on http://localhost:8501/


-> Analysis & Visualization
The Analysis and Visualization sections focus on understanding race results, driver standings, constructor standings, lap times, and more using various plots and charts.

Data Sources: Multiple CSV files like results.csv, driver_standings.csv, pitstops.csv, qualifying.csv, etc.
Visualizations: Created using Matplotlib and Seaborn.
Insights: Displayed using Streamlit with interactive charts and tables.


-> Prediction Model
The Prediction Model is built using GradientBoostingRegressor from sklearn.

Model Training:

Input Data: This year's qualifying times.
Algorithm: Gradient Boosting Regressor.
Prediction Target: Race outcome prediction for upcoming races.

Evaluation:
Metric: Mean Absolute Error (MAE)
Performance: Displayed in the application UI.




