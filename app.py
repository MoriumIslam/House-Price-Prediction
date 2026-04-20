import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'house_price_model.pkl'

def load_model():
    """Load the pickled model"""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Warning: Could not load model - {str(e)}")
            print("Model will not be available until fixed or replaced.")
            return None
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        return None

try:
    model = load_model()
except Exception as e:
    print(f"Error during model loading: {str(e)}")
    model = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features in the correct order for your model
        # Order: longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
        #        population, households, median_income, ocean_proximity_INLAND, 
        #        ocean_proximity_ISLAND, ocean_proximity_NEAR BAY, ocean_proximity_NEAR OCEAN
        features = [
            float(data.get('longitude', 0)),
            float(data.get('latitude', 0)),
            float(data.get('housing_median_age', 0)),
            float(data.get('total_rooms', 0)),
            float(data.get('total_bedrooms', 0)),
            float(data.get('population', 0)),
            float(data.get('households', 0)),
            float(data.get('median_income', 0)),
            float(data.get('ocean_proximity_INLAND', 0)),
            float(data.get('ocean_proximity_ISLAND', 0)),
            float(data.get('ocean_proximity_NEAR_BAY', 0)),
            float(data.get('ocean_proximity_NEAR_OCEAN', 0))
        ]
        
        # Reshape for prediction
        features_array = np.array([features])
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        return jsonify({
            'success': True,
            'predicted_price': round(float(prediction), 2)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic predictions"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Expect features as a list
        features = np.array([data.get('features', [])])
        
        if features.shape[1] == 0:
            return jsonify({'error': 'No features provided'}), 400
        
        prediction = model.predict(features)[0]
        
        return jsonify({
            'predicted_price': round(float(prediction), 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
