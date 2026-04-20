"""
Configuration file for House Price Prediction Flask app
"""

import os

# Flask Configuration
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# Model Configuration
MODEL_PATH = 'house_price_model.pkl'

# Feature Configuration
# Names and order of features as expected by the model
FEATURES = [
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'ocean_proximity_INLAND',
    'ocean_proximity_ISLAND',
    'ocean_proximity_NEAR_BAY',
    'ocean_proximity_NEAR_OCEAN'
]

# Feature constraints for validation
FEATURE_RANGES = {
    'longitude': {'min': -125, 'max': -114},
    'latitude': {'min': 32, 'max': 42},
    'housing_median_age': {'min': 1, 'max': 52},
    'total_rooms': {'min': 2, 'max': 40000},
    'total_bedrooms': {'min': 1, 'max': 6500},
    'population': {'min': 2, 'max': 35000},
    'households': {'min': 1, 'max': 6082},
    'median_income': {'min': 0.5, 'max': 15}
}
