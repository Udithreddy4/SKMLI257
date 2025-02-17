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
from sklearn.cluster import MiniBatchKMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from math import sin, cos, sqrt, atan2, radians
import time

# Load the data
input_data = pd.read_csv("C:\\Users\\udith\\Desktop\\rapido.csv")
print("Data size before removing: ", input_data.shape)

# Remove duplicates
df = input_data[input_data.duplicated()]
print("Number of duplicate observations: ", len(df))
del df
gc.collect()

input_data.drop_duplicates(keep='first', inplace=True)
print("Data size after removing: ", input_data.shape)
print("Number of unique customers: ", input_data["number"].nunique())

# Data manipulation for date and time
new = input_data["ts"].str.split(" ", n=1, expand=True)
input_data["raw_date"] = new[0]
input_data["raw_time"] = new[1]

new = input_data["raw_date"].str.split("-", n=2, expand=True)
input_data["year"] = new[0]
input_data["month"] = new[1]
input_data["date"] = new[2]

new = input_data["raw_time"].str.split(":", n=2, expand=True)
input_data["hour"] = new[0]
input_data["minute"] = new[1]

# Copy data and drop unnecessary columns
data = input_data.copy()
data.drop(["raw_date", "raw_time", "number", "minute"], axis=1, inplace=True)
del input_data
gc.collect()

# Check for missing values
print("Is there any missing value? ", data.isna().sum().sum() > 0)

# Aggregate the data to get the number of requests per hour
data['datetime'] = pd.to_datetime(data['ts'])
data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.weekday

# Group by hour and count the number of requests
hourly_requests = data.groupby(['hour', 'day_of_week']).size().reset_index(name='ride_requests')

# Plot the number of ride requests per hour
plt.figure(figsize=(12, 8))
sns.lineplot(x='hour', y='ride_requests', data=hourly_requests, marker='o')
plt.title('Ride Requests per Hour', fontsize=20)
plt.xlabel('Hour of Day', fontsize=16)
plt.ylabel('Number of Ride Requests', fontsize=16)
plt.show()

# Feature Engineering: Adding lag features (previous hour requests, etc.)
hourly_requests['prev_hour_requests'] = hourly_requests['ride_requests'].shift(1).fillna(0)
hourly_requests['next_hour_requests'] = hourly_requests['ride_requests'].shift(-1).fillna(0)

# Scaling the features
scaler = StandardScaler()
X = hourly_requests[['hour', 'day_of_week', 'prev_hour_requests', 'next_hour_requests']]
y = hourly_requests['ride_requests']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# XGBoost Model
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# Evaluate the models
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))

print(f"Random Forest MAE: {rf_mae}, RMSE: {rf_rmse}")
print(f"XGBoost MAE: {xgb_mae}, RMSE: {xgb_rmse}")

# Visualize the predictions vs actual

# 1. Actual vs predicted for both models
plt.figure(figsize=(12, 8))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(rf_predictions, label='Random Forest Predictions', color='red')
plt.plot(xgb_predictions, label='XGBoost Predictions', color='green')
plt.legend()
plt.title("Model Predictions vs Actual Ride Requests")
plt.xlabel("Samples")
plt.ylabel("Ride Requests")
plt.show()

# 2. Distribution of Ride Requests
plt.figure(figsize=(12, 8))
sns.histplot(hourly_requests['ride_requests'], kde=True, color='skyblue', bins=30)
plt.title("Distribution of Ride Requests", fontsize=20)
plt.xlabel("Number of Ride Requests", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.show()

# 3. Feature importance (Random Forest)
importances_rf = rf_model.feature_importances_
feature_names = X.columns
indices_rf = np.argsort(importances_rf)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Random Forest Feature Importance", fontsize=20)
plt.bar(range(X.shape[1]), importances_rf[indices_rf], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices_rf], rotation=90)
plt.xlabel('Feature', fontsize=16)
plt.ylabel('Importance', fontsize=16)
plt.show()

# 4. Model residuals (Random Forest)
rf_residuals = y_test - rf_predictions
plt.figure(figsize=(12, 8))
sns.histplot(rf_residuals, kde=True, color='red', bins=30)
plt.title("Random Forest Residuals", fontsize=20)
plt.xlabel("Residuals", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.show()

# 5. Model residuals (XGBoost)
xgb_residuals = y_test - xgb_predictions
plt.figure(figsize=(12, 8))
sns.histplot(xgb_residuals, kde=True, color='green', bins=30)
plt.title("XGBoost Residuals", fontsize=20)
plt.xlabel("Residuals", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.show()
