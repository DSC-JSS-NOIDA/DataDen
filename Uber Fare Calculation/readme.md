# GPU Configuration and Kernel Runtime Analysis

## Overview
This project aims to understand the relationship between various GPU configurations and their impact on kernel runtime. Different models were evaluated to predict the GPU runtime based on its configurations, with a focus on both interpretability and prediction accuracy.

## Dataset Insights
- The dataset contains information about different GPU configurations and their associated runtimes.
- Features in the dataset provide insights into the GPU's architectural and operational settings.
- Preprocessing steps applied to the data include:
  - **Outlier Removal**: Outliers were removed from the `Runtime` column based on the Interquartile Range (IQR) method.
  - **Log Transformation**: The `Runtime` distribution was log-transformed to address its skewness.
  - **Feature Scaling**: Features were scaled using `MinMaxScaler` to bring them to a similar scale.

## Models Evaluated
Several regression models were trained and evaluated to predict GPU runtime. The models include:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Simple Neural Network
- Gradient Boosting Regressor
- K-Neighbors Regressor
- Decision Tree Regressor

## Performance Metrics
The models were evaluated based on two primary metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between the estimated values and the actual value.
- **R-squared (R²) Score**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Key Findings
- The Random Forest Regressor and DecisionTreeRegressor were top performers based on MSE and R² Score.
- Linear, Ridge, and Lasso Regressions, while providing a basic understanding, might not be the most suitable models for this dataset.

## Future Work
Considerations for future improvements include:
- Hyperparameter tuning for better model performance.
- Feature engineering to derive new insights.
- Exploring more complex neural network architectures.

## Getting Started
To get started with this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the main analysis script using `python main.py`.
4. Explore the results and model performance metrics.

## Usage
   import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Step 1: Data Preprocessing
data = pd.read_csv('uber.csv')

# Handle missing or incorrect data (if any)
data.dropna(inplace=True)

# Convert date-time strings to datetime objects
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

# Calculate distance between pickup and dropoff
def haversine_distance(lat1, lon1, lat2, lon2):
    # Haversine formula to calculate distance
    ...

# Ensure the distance column contains numerical values
data['distance'] = data.apply(lambda row: haversine_distance(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

# Convert the distance column to a numerical data type (e.g., float)
data['distance'] = data['distance'].astype(float)


# Step 2: Feature Selection
features = ['passenger_count', 'distance']
target = 'fare_amount'

# Step 3: Split Data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Building
model = xgb.XGBRegressor()

# Step 5: Training
model.fit(X_train, y_train)

# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7: Evaluation
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Step 8: User Input
while True:
    try:
        passenger_count = int(input("Enter the number of passengers: "))
        distance = float(input("Enter the distance of travel in miles: "))
        break
    except ValueError:
        print("Invalid input. Please enter a valid number.")

# Create a DataFrame with the user input
user_input = pd.DataFrame({'passenger_count': [passenger_count], 'distance': [distance]})

# Step 9: Make Predictions for User Input
fare_prediction = model.predict(user_input)

# Step 10: Display the Prediction
print(f"Estimated fare amount: ${fare_prediction[0]:.2f}")


## Contributors
-   nikhil

