#!/usr/bin/env python
"""
Quick run script for the House Price Prediction Flask app
Run: python run.py
"""

from app import app
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
