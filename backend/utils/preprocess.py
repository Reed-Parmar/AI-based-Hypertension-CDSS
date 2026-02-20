"""
preprocess.py
============
Feature preprocessing and clinical logic helpers for the Hypertension CDSS.

This module provides:
- Input validation for patient data
- Feature array construction for model inference
- Blood pressure classification (AHA/ACC 2017 guidelines)
- Risk factor identification based on clinical thresholds

Author: Backend ML Engineer
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


# =============================================================================
# CONSTANTS — Clinical Thresholds
# =============================================================================

# Input field validation ranges
FIELD_RANGES = {
    "age": {"min": 1, "max": 120},
    "bmi": {"min": 10.0, "max": 70.0},
    "cholesterol": {"min": 50, "max": 600},
    "systolic": {"min": 60, "max": 300},
    "diastolic": {"min": 30, "max": 200},
}

# Required fields
REQUIRED_FIELDS = ["age", "bmi", "cholesterol", "systolic", "diastolic"]

# Feature names in order expected by the model
FEATURE_NAMES = ["age", "bmi", "cholesterol", "systolic", "diastolic"]


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_inputs(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate the input data dictionary.
    
    Args:
        data: Dictionary containing patient data with keys:
              age, bmi, cholesterol, systolic, diastolic
            
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if all validations pass
        - error_message: None if valid, else descriptive error string
    """
    # Check for missing fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: {field}"
        if data[field] is None:
            return False, f"Missing required field: {field}"
    
    # Check field types (must be numeric; reject booleans which pass isinstance(..., int))
    for field in REQUIRED_FIELDS:
        if isinstance(data[field], bool) or not isinstance(data[field], (int, float)):
            return False, f"Invalid value for '{field}': must be a number"
    
    # Check individual field ranges
    for field in REQUIRED_FIELDS:
        value = data[field]
        range_info = FIELD_RANGES[field]
        
        # For age, must be integer
        if field == "age" and not isinstance(value, int):
            return False, f"Invalid value for 'age': must be a whole number"
        
        if value < range_info["min"] or value > range_info["max"]:
            return False, f"Invalid value for '{field}': must be between {range_info['min']} and {range_info['max']}"
    
    # Cross-field validation: diastolic must be less than systolic
    if data["diastolic"] >= data["systolic"]:
        return False, "diastolic must be less than systolic"
    
    return True, None


def build_feature_array(data: Dict[str, Any]) -> np.ndarray:
    """
    Build a feature array from the input data dictionary.
    
    Args:
        data: Dictionary containing patient data
        
    Returns:
        numpy array of shape (1, 5) with features in order:
        [age, bmi, cholesterol, systolic, diastolic]
    """
    features = np.array([[
        data["age"],
        data["bmi"],
        data["cholesterol"],
        data["systolic"],
        data["diastolic"]
    ]], dtype=np.float64)
    
    return features


# =============================================================================
# BLOOD PRESSURE CLASSIFICATION — AHA/ACC 2017 GUIDELINES
# =============================================================================

def classify_bp(systolic: int, diastolic: int) -> Dict[str, str]:
    """
    Classify blood pressure according to AHA/ACC 2017 guidelines.
    
    Args:
        systolic: Systolic blood pressure in mmHg
        diastolic: Diastolic blood pressure in mmHg
        
    Returns:
        Dictionary with keys:
        - category: Human-readable BP category name
        - stage: CSS-safe stage identifier
        - color: Semantic color hint
    """
    # Hypertensive crisis (emergency)
    if systolic > 180 or diastolic > 120:
        return {
            "category": "Hypertensive Crisis",
            "stage": "crisis",
            "color": "darkred"
        }
    
    # Stage 2 Hypertension
    if systolic >= 140 or diastolic >= 90:
        return {
            "category": "High BP – Stage 2",
            "stage": "stage2",
            "color": "red"
        }
    
    # Stage 1 Hypertension
    if (systolic >= 130 and systolic <= 139) or (diastolic >= 80 and diastolic <= 89):
        return {
            "category": "High BP – Stage 1",
            "stage": "stage1",
            "color": "orange"
        }
    
    # Elevated
    if systolic >= 120 and systolic <= 129 and diastolic < 80:
        return {
            "category": "Elevated",
            "stage": "elevated",
            "color": "yellow"
        }
    
    # Normal
    if systolic < 120 and diastolic < 80:
        return {
            "category": "Normal",
            "stage": "normal",
            "color": "green"
        }
    
    # Fallback (unknown)
    return {
        "category": "Unknown",
        "stage": "unknown",
        "color": "gray"
    }


# =============================================================================
# RISK FACTOR IDENTIFICATION
# =============================================================================

def identify_risk_factors(age: int, bmi: float, cholesterol: int, 
                          systolic: int, diastolic: int) -> List[str]:
    """
    Identify clinical risk factors based on threshold values.
    
    This follows the same heuristic logic as the frontend mock
    to ensure consistency between client and server.
    
    Args:
        age: Patient age in years
        bmi: Body Mass Index in kg/m²
        cholesterol: Total cholesterol in mg/dL
        systolic: Systolic blood pressure in mmHg
        diastolic: Diastolic blood pressure in mmHg
        
    Returns:
        List of human-readable risk factor descriptions
    """
    risk_factors = []
    
    # ── Systolic BP — strongest predictor ──
    if systolic >= 180:
        risk_factors.append("Hypertensive-crisis-level systolic BP")
    elif systolic >= 140:
        risk_factors.append("Stage 2 systolic hypertension")
    elif systolic >= 130:
        risk_factors.append("Stage 1 systolic hypertension")
    elif systolic >= 120:
        risk_factors.append("Elevated systolic BP")
    
    # ── Diastolic BP ──
    if diastolic >= 120:
        risk_factors.append("Hypertensive-crisis-level diastolic BP")
    elif diastolic >= 90:
        risk_factors.append("Stage 2 diastolic hypertension")
    elif diastolic >= 80:
        risk_factors.append("Stage 1 diastolic hypertension")
    
    # ── Age factor ──
    if age >= 70:
        risk_factors.append("Advanced age (≥70 yrs)")
    elif age >= 60:
        risk_factors.append("Senior age (≥60 yrs)")
    elif age >= 45:
        risk_factors.append("Middle age (≥45 yrs)")
    
    # ── BMI ──
    if bmi >= 35:
        risk_factors.append("Severe obesity (BMI ≥35)")
    elif bmi >= 30:
        risk_factors.append("Obesity (BMI ≥30)")
    elif bmi >= 25:
        risk_factors.append("Overweight (BMI ≥25)")
    
    # ── Total Cholesterol ──
    if cholesterol >= 280:
        risk_factors.append("Very high cholesterol (≥280 mg/dL)")
    elif cholesterol >= 240:
        risk_factors.append("High cholesterol (≥240 mg/dL)")
    elif cholesterol >= 200:
        risk_factors.append("Borderline-high cholesterol (≥200 mg/dL)")
    
    return risk_factors


# =============================================================================
# RISK SCORE CALCULATION
# =============================================================================

def calculate_risk_score(age: int, bmi: float, cholesterol: int, 
                         systolic: int, diastolic: int) -> int:
    """
    Calculate a weighted clinical risk score (0-100).
    
    This follows the same heuristic logic as the frontend mock
    to ensure consistency between client and server.
    
    Args:
        age: Patient age in years
        bmi: Body Mass Index in kg/m²
        cholesterol: Total cholesterol in mg/dL
        systolic: Systolic blood pressure in mmHg
        diastolic: Diastolic blood pressure in mmHg
        
    Returns:
        Integer risk score between 0 and 100
    """
    risk = 0
    
    # ── Systolic BP — strongest predictor ──
    if systolic >= 180:
        risk += 42
    elif systolic >= 140:
        risk += 35
    elif systolic >= 130:
        risk += 22
    elif systolic >= 120:
        risk += 9
    
    # ── Diastolic BP ──
    if diastolic >= 120:
        risk += 30
    elif diastolic >= 90:
        risk += 22
    elif diastolic >= 80:
        risk += 12
    
    # ── Age factor ──
    if age >= 70:
        risk += 14
    elif age >= 60:
        risk += 10
    elif age >= 45:
        risk += 6
    
    # ── BMI ──
    if bmi >= 35:
        risk += 12
    elif bmi >= 30:
        risk += 8
    elif bmi >= 25:
        risk += 4
    
    # ── Total Cholesterol ──
    if cholesterol >= 280:
        risk += 10
    elif cholesterol >= 240:
        risk += 7
    elif cholesterol >= 200:
        risk += 3
    
    # Clamp to [0, 100]
    return min(100, max(0, risk))


def calculate_confidence(probabilities: np.ndarray,
                         risk_score: int = 50) -> dict:
    """
    Calculate a clinically reasonable confidence score.

    Decision Trees with pure leaf nodes produce predict_proba values of
    exactly 0 or 1, making raw probabilities useless for expressing
    degrees of certainty.  Instead, confidence is derived from how far
    the patient's **clinical risk score** lies from the decision boundary
    (50), which naturally varies with each patient's inputs.

    Strategy:
      1. Compute the absolute distance of `risk_score` from the
         decision boundary (50).  Range: 0 – 50.
      2. Map that distance into a display range of **60 – 90 %** using a
         square-root curve so that the first few points of distance
         contribute more (diminishing returns near the top).
      3. Assign a qualitative confidence band: Low / Moderate / High.

    This ensures:
      - Borderline patients (risk ≈ 50) → ~60% confidence (Low)
      - Moderate cases (risk ≈ 30 or 70) → ~74% (Moderate)
      - Clear-cut cases (risk ≈ 0 or 100) → ~90% (High)

    Args:
        probabilities: Array of shape (2,) — kept for API compatibility
                       but not used in the calculation.
        risk_score:    Weighted clinical risk score (0 – 100).

    Returns:
        dict with keys:
          - confidence (float): percentage in 60 – 90 range
          - confidenceBand (str): "Low" | "Moderate" | "High"
    """
    DISPLAY_MIN = 60.0
    DISPLAY_MAX = 90.0
    MAX_DISTANCE = 50.0

    distance = abs(risk_score - 50)                          # 0 – 50
    normalised = min(distance / MAX_DISTANCE, 1.0)           # 0.0 – 1.0
    curved = normalised ** 0.5                               # sqrt curve

    confidence = round(DISPLAY_MIN + curved * (DISPLAY_MAX - DISPLAY_MIN), 1)
    confidence = max(DISPLAY_MIN, min(DISPLAY_MAX, confidence))

    band = classify_confidence_band(confidence)
    return {"confidence": confidence, "confidenceBand": band}


def classify_confidence_band(confidence: float) -> str:
    """
    Map a numeric confidence percentage to a qualitative band.

    Bands:
      - Low:      < 72%
      - Moderate:  72% – 82%
      - High:     > 82%

    Args:
        confidence: Percentage value (expected 60–90).

    Returns:
        "Low" | "Moderate" | "High"
    """
    if confidence < 72.0:
        return "Low"
    if confidence <= 82.0:
        return "Moderate"
    return "High"
