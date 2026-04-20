"""
Generate a sample trained model for testing
This creates a linear regression model using California housing-like features
"""

import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate synthetic training data matching California housing dataset structure
np.random.seed(42)

# Features: longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
#           population, households, median_income, ocean_proximity_INLAND,
#           ocean_proximity_ISLAND, ocean_proximity_NEAR_BAY, ocean_proximity_NEAR_OCEAN

X = np.array([
    [-122.23, 37.88, 41, 880, 129, 322, 126, 8.3, 0, 0, 0, 1],  # NEAR OCEAN
    [-122.22, 37.86, 21, 7099, 1106, 2401, 1138, 8.3, 0, 0, 1, 0],  # NEAR BAY
    [-119.04, 35.37, 52, 1467, 190, 496, 177, 7.2, 1, 0, 0, 0],  # INLAND
    [-120.45, 34.56, 15, 5215, 1155, 2054, 930, 6.5, 1, 0, 0, 0],  # INLAND
    [-121.89, 36.77, 19, 3424, 533, 1310, 490, 7.1, 1, 0, 0, 0],  # INLAND
    [-122.41, 37.67, 52, 1534, 242, 703, 289, 7.8, 0, 0, 0, 1],  # NEAR OCEAN
    [-122.36, 37.47, 18, 4547, 634, 1845, 682, 8.1, 0, 0, 1, 0],  # NEAR BAY
    [-120.20, 34.21, 48, 2109, 289, 792, 297, 5.9, 1, 0, 0, 0],  # INLAND
    [-119.50, 35.50, 35, 3876, 512, 1420, 540, 6.8, 1, 0, 0, 0],  # INLAND
    [-122.00, 37.33, 25, 5890, 845, 2210, 850, 8.2, 0, 0, 1, 0],  # NEAR BAY
    [-120.87, 34.12, 60, 1200, 180, 450, 200, 4.5, 1, 0, 0, 0],  # INLAND
    [-122.50, 37.80, 10, 6500, 950, 2500, 1000, 9.2, 0, 0, 0, 1],  # NEAR OCEAN
    [-121.10, 36.50, 30, 4320, 650, 1650, 700, 7.5, 1, 0, 0, 0],  # INLAND
    [-119.99, 35.78, 45, 2850, 380, 1100, 450, 6.2, 1, 0, 0, 0],  # INLAND
    [-122.60, 37.90, 8, 7200, 1050, 2800, 1100, 9.5, 0, 0, 0, 1],  # NEAR OCEAN
    [-120.77, 33.99, 55, 1050, 165, 400, 180, 4.8, 1, 0, 0, 0],  # INLAND
    [-122.30, 37.55, 22, 5600, 800, 2000, 900, 8.0, 0, 0, 1, 0],  # NEAR BAY
    [-121.50, 36.77, 35, 3650, 520, 1550, 630, 7.3, 1, 0, 0, 0],  # INLAND
    [-123.50, 38.50, 18, 4200, 580, 1700, 700, 7.8, 0, 1, 0, 0],  # ISLAND
    [-122.45, 37.65, 28, 6100, 900, 2300, 950, 8.4, 0, 0, 1, 0],  # NEAR BAY
])

# Generate target prices based on feature values and coefficients
# Using coefficients similar to what you provided
coefficients = np.array([
    -26838.27,  # longitude
    -25468.35,  # latitude
    1102.19,    # housing_median_age
    -6.02,      # total_rooms
    102.79,     # total_bedrooms
    -38.17,     # population
    48.25,      # households
    39473.98,   # median_income
    -39786.66,  # ocean_proximity_INLAND
    136125.07,  # ocean_proximity_ISLAND
    -5136.64,   # ocean_proximity_NEAR_BAY
    3431.14     # ocean_proximity_NEAR_OCEAN
])

y = X @ coefficients + np.random.normal(0, 50000, len(X))

# Train the model
print("Training model with California housing-like data...")
model = LinearRegression()
model.fit(X, y)

# Save the model
print("Saving model to house_price_model.pkl...")
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✓ Model trained and saved successfully!")
print(f"  Coefficients: {model.coef_}")
print(f"  Intercept: {model.intercept_}")

# Test prediction
test_input = np.array([[-122.4, 37.8, 25, 5000, 800, 2000, 900, 8.3, 0, 0, 0, 1]])
prediction = model.predict(test_input)[0]
print(f"\n✓ Test prediction: ${prediction:,.2f}")
print("  Features: -122.4 lon, 37.8 lat, 25 age, 5000 rooms, 800 bedrooms, 2000 pop, 900 households, 8.3 income, NEAR OCEAN")

