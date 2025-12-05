# Prodigy-ML-01
This repository contains the source code of Machine Learning task-01 given by Prodigy Infotech.
# 🏠 House Price Prediction (Linear Regression)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("house_prices.csv")   # Replace with your dataset path
print(df.head())

# Preprocessing (drop missing values)
df = df.dropna()

# Independent & dependent features
X = df.drop("Price", axis=1)     # Features
y = df["Price"]                  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Example prediction
sample = np.array([[3, 2, 1500]])   # Example: 3 rooms, 2 bathrooms, 1500 sqft
print("Predicted Price:", model.predict(sample)[0])

