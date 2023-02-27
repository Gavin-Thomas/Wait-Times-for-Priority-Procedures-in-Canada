import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
df = pd.read_csv("patient_wait_times.csv")

# Convert the "Days" column to numeric
df['Result'] = pd.to_numeric(df['Result'], errors='coerce')

# Encode categorical variables
df = pd.get_dummies(df, columns=["Province", "Indicator", "Metric", "Year"])

# Split the data into training and testing sets
X = df.drop(["Result"], axis=1) # drop the target variable
y = df["Result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a random forest regressor and fit it to the training data
rf = RandomForestRegressor(n_estimators=91, random_state=38)
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Reshape y_pred to match the dimensions of y_test
y_pred = np.reshape(y_pred, (len(y_pred), 1))

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)


