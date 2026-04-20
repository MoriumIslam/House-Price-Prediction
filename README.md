# House Price Prediction - Flask Application

A machine learning-powered Flask web application for predicting house prices based on property features.

## 📁 Project Structure

```
house_price_prediction_model/
├── app.py                    # Flask application
├── house_price_model.pkl     # Trained ML model
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Main web interface
├── static/
│   └── style.css            # Styling
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## 📋 Features

- **Web Interface**: User-friendly form to input house features
- **Real-time Predictions**: Get instant house price predictions
- **API Endpoint**: Programmatic access to predictions via `/api/predict`
- **Error Handling**: Robust error handling and validation
- **Responsive Design**: Works on desktop and mobile devices

## 🏠 Input Features

The model accepts the following features:

- **Square Feet**: Total area of the house
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms
- **Age**: Age of the house in years
- **Garage**: Number of garage spaces

## 🔌 API Endpoints

### Web Interface
- **GET** `/` - Main web interface

### Prediction Endpoints
- **POST** `/predict` - Web form predictions
- **POST** `/api/predict` - API for programmatic predictions

#### API Request Example

```json
{
  "features": [2500, 4, 2, 10, 2]
}
```

#### API Response Example

```json
{
  "predicted_price": 523500.00
}
```

## 🛠️ Configuration

To modify the default port or host:

Edit `app.py` and change:
```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📦 Dependencies

- **Flask**: Web framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **joblib**: Model serialization

## ✨ Features to Extend

- Add more input features
- Implement confidence intervals
- Add prediction history
- Integrate database for logging
- Add user authentication
- Create admin dashboard

## 🐛 Troubleshooting

### Model not loading
Ensure `house_price_model.pkl` exists in the same directory as `app.py`

### Port already in use
Change the port number in `app.py` or kill the process using port 5000

### Module not found errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📝 Notes

- The model features should align with how the ML model was trained
- Adjust feature names in `app.py` if your model uses different feature names
- The model is loaded once when the app starts

## 📄 License

This project is available for educational purposes.

---

**Happy predicting! 🎉**
# House-Price-Prediction
