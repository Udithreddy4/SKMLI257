# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')

import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset (Fix: Use read_excel instead of read_csv)
file_path = "C:\\Users\\udith\\Desktop\\New folder (2)\\TASK2\\Book1.xlsx"
input_data = pd.read_excel(file_path)

print("Data size before removing duplicates: ", input_data.shape)

# Remove duplicate entries
num_duplicates = input_data.duplicated().sum()
print("Number of duplicate observations: ", num_duplicates)
input_data.drop_duplicates(inplace=True)
print("Data size after removing duplicates: ", input_data.shape)

# Convert timestamp column to datetime format
input_data["ts"] = pd.to_datetime(input_data["ts"])

# Extract date and time features
input_data["year"] = input_data["ts"].dt.year
input_data["month"] = input_data["ts"].dt.month
input_data["date"] = input_data["ts"].dt.day
input_data["hour"] = input_data["ts"].dt.hour
input_data["minute"] = input_data["ts"].dt.minute
input_data["day_of_week"] = input_data["ts"].dt.weekday  # Monday=0, Sunday=6

# Drop unnecessary columns
data = input_data.drop(["number", "minute"], axis=1)

# Check for missing values
print("Are there any missing values?", data.isna().sum().sum() > 0)

# Aggregate ride requests per hour
hourly_requests = data.groupby(['hour', 'day_of_week']).size().reset_index(name='ride_requests')

# 📊 Plot: Ride requests per hour
plt.figure(figsize=(12, 6))
sns.lineplot(x='hour', y='ride_requests', data=hourly_requests, marker='o', color='blue')
plt.title('Ride Requests Per Hour', fontsize=16)
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Number of Ride Requests', fontsize=14)
plt.grid()
plt.show()

# Feature Engineering: Adding lag features
hourly_requests['prev_hour_requests'] = hourly_requests['ride_requests'].shift(1).fillna(0)
hourly_requests['next_hour_requests'] = hourly_requests['ride_requests'].shift(-1).fillna(0)

# Define features and target variable
X = hourly_requests[['hour', 'day_of_week', 'prev_hour_requests', 'next_hour_requests']]
y = hourly_requests['ride_requests']

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🚀 Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# 🚀 Train XGBoost Model
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# 📊 Evaluate Model Performance
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))

print(f"🔹 Random Forest - MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")
print(f"🔹 XGBoost - MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}")

# 📊 Visualizing Predictions vs Actual Values
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', color='blue', linestyle='dashed', marker='o')
plt.plot(rf_predictions, label='Random Forest Predictions', color='red', linestyle='dashed', marker='o')
plt.plot(xgb_predictions, label='XGBoost Predictions', color='green', linestyle='dashed', marker='o')
plt.legend()
plt.title("Model Predictions vs Actual Ride Requests", fontsize=16)
plt.xlabel("Samples", fontsize=14)
plt.ylabel("Ride Requests", fontsize=14)
plt.grid()
plt.show()

# 📊 Distribution of Ride Requests
plt.figure(figsize=(12, 6))
sns.histplot(hourly_requests['ride_requests'], kde=True, color='skyblue', bins=30)
plt.title("Distribution of Ride Requests", fontsize=16)
plt.xlabel("Number of Ride Requests", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid()
plt.show()

# 📊 Feature Importance (Random Forest)
importances_rf = rf_model.feature_importances_
feature_names = X.columns
indices_rf = np.argsort(importances_rf)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance - Random Forest", fontsize=16)
plt.bar(range(len(feature_names)), importances_rf[indices_rf], align="center", color='royalblue')
plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices_rf], rotation=45)
plt.xlabel('Feature', fontsize=14)
plt.ylabel('Importance', fontsize=14)
plt.grid()
plt.show()

# 📊 Residual Analysis - Random Forest
rf_residuals = y_test - rf_predictions
plt.figure(figsize=(12, 6))
sns.histplot(rf_residuals, kde=True, color='red', bins=30)
plt.title("Residuals - Random Forest", fontsize=16)
plt.xlabel("Residuals", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid()
plt.show()

# 📊 Residual Analysis - XGBoost
xgb_residuals = y_test - xgb_predictions
plt.figure(figsize=(12, 6))
sns.histplot(xgb_residuals, kde=True, color='green', bins=30)
plt.title("Residuals - XGBoost", fontsize=16)
plt.xlabel("Residuals", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid()
plt.show()
