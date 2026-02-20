"""
app.py
======
Main API server for the Hypertension Clinical Decision Support System (CDSS).

This Flask application provides a REST API for hypertension prediction
using a trained Decision Tree classifier.

Framework Choice: Flask
- Flask is chosen over FastAPI for this project because:
  1. Simpler for a single-endpoint API
  2. Easier to integrate with existing Python virtual environments
  3. Minimal boilerplate for this use case
  4. Wide compatibility with sklearn serialization (joblib)

API Endpoints:
  - POST /api/predict  : Run hypertension prediction
  - GET  /api/health   : Server health check

Usage:
    python app.py

Prerequisites:
    1. Install dependencies: pip install -r requirements.txt
    2. Train the model:    python train_model.py
    3. Start the server:   python app.py

Author: Backend ML Engineer
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Import preprocessing utilities
from utils.preprocess import (
    validate_inputs,
    build_feature_array,
    classify_bp,
    identify_risk_factors,
    calculate_risk_score,
    calculate_confidence
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Flask app configuration
app = Flask(__name__)

# CORS configuration
# Allow only the VS Code Live Server frontend origins.
# methods: GET, POST, OPTIONS (OPTIONS is implicit for preflight).
# headers: Content-Type (for JSON), Authorization (for future auth).
CORS(app, origins=[
    "http://127.0.0.1:5500",
    "http://localhost:5500"
], methods=["GET", "POST", "OPTIONS"],
   allow_headers=["Content-Type", "Authorization"])

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "cdss_model.pkl")
API_VERSION = "1.0.0"

# Global model variable
model = None


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    """
    Load the trained model from disk.
    
    Returns:
        Loaded sklearn Pipeline object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    global model
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please run train_model.py first to generate the model."
        )
    
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    
    return model


# =============================================================================
# API ROUTES
# =============================================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response with server status
    """
    model_loaded = model is not None
    
    return jsonify({
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "version": API_VERSION
    }), 200 if model_loaded else 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict hypertension risk for given patient data.
    
    Expected JSON input:
    {
        "age": number,
        "bmi": number,
        "cholesterol": number,
        "systolic": number,
        "diastolic": number
    }
    
    Returns:
        JSON response with prediction and clinical details
    """
    # Check if model is loaded
    if model is None:
        return jsonify({
            "error": "Model not loaded. Run train_model.py first.",
            "detail": "FileNotFoundError: model/cdss_model.pkl not found"
        }), 500
    
    # Parse JSON input
    try:
        data = request.get_json()
    except Exception:
        return jsonify({
            "error": "Invalid JSON in request body"
        }), 400
    
    if data is None:
        return jsonify({
            "error": "Request body must be valid JSON"
        }), 400
    
    # Validate inputs
    is_valid, error_message = validate_inputs(data)
    if not is_valid:
        return jsonify({
            "error": error_message
        }), 400
    
    try:
        # Extract validated values
        age = int(data["age"])
        bmi = float(data["bmi"])
        cholesterol = int(data["cholesterol"])
        systolic = int(data["systolic"])
        diastolic = int(data["diastolic"])
        
        # Build feature array for model
        X = build_feature_array(data)
        
        # Run model prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Calculate clinical risk score (needed before confidence)
        risk_score = calculate_risk_score(age, bmi, cholesterol, systolic, diastolic)
        
        # Calculate confidence from risk-score distance to decision boundary
        conf_result = calculate_confidence(probabilities, risk_score=risk_score)
        confidence = conf_result["confidence"]
        confidence_band = conf_result["confidenceBand"]
        
        # Classify blood pressure (AHA/ACC 2017)
        bp_info = classify_bp(systolic, diastolic)
        
        # Identify risk factors
        risk_factors = identify_risk_factors(age, bmi, cholesterol, systolic, diastolic)
        
        # Build response
        response = {
            "prediction": int(prediction),
            "confidence": confidence,
            "confidenceBand": confidence_band,
            "riskScore": risk_score,
            "bpCategory": bp_info["category"],
            "bpStage": bp_info["stage"],
            "bpColor": bp_info["color"],
            "riskFactors": risk_factors
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({
            "error": "Invalid data type for field",
            "detail": str(e)
        }), 400
        
    except Exception as e:
        # Log the error for debugging
        print(f"Prediction error: {str(e)}", file=sys.stderr)
        return jsonify({
            "error": "Prediction failed",
            "detail": str(e) if app.debug else "Internal server error"
        }), 500


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "error": "Internal server error"
    }), 500


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Start the Flask application."""
    print("="*60)
    print("Hypertension CDSS - API Server")
    print("="*60)
    
    # Load model
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run the training script first:")
        print("  python train_model.py")
        sys.exit(1)
    
    # Start server
    print("\nStarting server...")
    print("API available at: http://localhost:5000")
    print("Endpoints:")
    print("  - POST /api/predict")
    print("  - GET  /api/health")
    print("\nPress Ctrl+C to stop the server")
    print("="*60)
    
    # Run Flask development server.
    # For production, use a WSGI server such as gunicorn:
    #   gunicorn -w 4 -b 0.0.0.0:5000 app:app
    debug = os.environ.get("FLASK_DEBUG", os.environ.get("DEBUG", "0")).lower() in ("1", "true", "yes")
    host = "127.0.0.1" if debug else "0.0.0.0"
    app.run(
        host=host,
        port=5000,
        debug=debug
    )


if __name__ == "__main__":
    main()
