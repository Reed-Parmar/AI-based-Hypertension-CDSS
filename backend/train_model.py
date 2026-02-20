"""
train_model.py
==============
Training script for the Hypertension CDSS Decision Tree model.

This script:
1. Generates synthetic clinical data (or loads existing dataset)
2. Trains a sklearn Pipeline with StandardScaler + DecisionTreeClassifier
3. Evaluates model performance
4. Saves the trained model to model/cdss_model.pkl

Usage:
    python train_model.py

Output:
    model/cdss_model.pkl - Trained sklearn Pipeline

Author: Backend ML Engineer
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Feature names
FEATURE_NAMES = ["age", "bmi", "cholesterol", "systolic_bp", "diastolic_bp"]


def generate_dataset(n_samples_per_class: int = 300) -> pd.DataFrame:
    """
    Generate synthetic cardiovascular dataset.
    
    The data is generated using clinically realistic parameters
    derived from epidemiological literature.
    
    Args:
        n_samples_per_class: Number of samples for each class
        
    Returns:
        DataFrame with features and label column
    """
    print(f"Generating synthetic dataset with {n_samples_per_class} samples per class...")
    
    # Hypertensive class (label = 1)
    hypertensive = pd.DataFrame({
        "age": np.random.normal(58, 9, n_samples_per_class).clip(30, 85).astype(int),
        "bmi": np.random.normal(30, 4, n_samples_per_class).clip(18, 45),
        "cholesterol": np.random.normal(248, 30, n_samples_per_class).clip(160, 320).astype(int),
        "systolic_bp": np.random.normal(158, 18, n_samples_per_class).clip(130, 200).astype(int),
        "diastolic_bp": np.random.normal(96, 10, n_samples_per_class).clip(80, 130).astype(int),
        "label": 1
    })
    
    # Normotensive class (label = 0)
    normotensive = pd.DataFrame({
        "age": np.random.normal(42, 12, n_samples_per_class).clip(18, 70).astype(int),
        "bmi": np.random.normal(23, 3, n_samples_per_class).clip(15, 30),
        "cholesterol": np.random.normal(185, 25, n_samples_per_class).clip(120, 230).astype(int),
        "systolic_bp": np.random.normal(112, 10, n_samples_per_class).clip(90, 130).astype(int),
        "diastolic_bp": np.random.normal(72, 8, n_samples_per_class).clip(55, 82).astype(int),
        "label": 0
    })
    
    # Combine and shuffle
    df = pd.concat([hypertensive, normotensive], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset generated: {len(df)} total samples")
    print(f"  - Hypertensive (label=1): {(df['label'] == 1).sum()}")
    print(f"  - Normotensive (label=0): {(df['label'] == 0).sum()}")
    
    return df


def train_model(df: pd.DataFrame) -> Pipeline:
    """
    Train a sklearn Pipeline with StandardScaler and DecisionTreeClassifier.
    
    Args:
        df: DataFrame with features and label column
        
    Returns:
        Trained sklearn Pipeline
    """
    print("\nPreparing features and labels...")
    
    X = df[FEATURE_NAMES].values
    y = df["label"].values
    
    # Stratified train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Build pipeline
    print("\nBuilding pipeline...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            criterion="gini",
            random_state=42
        ))
    ])
    
    # Train
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"              Neg     Pos")
    print(f"Actual Neg  {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"Actual Pos  {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normotensive", "Hypertensive"]))
    
    # Feature importances
    print(f"\nFeature Importances:")
    importances = pipeline.named_steps["classifier"].feature_importances_
    for name, imp in zip(FEATURE_NAMES, importances):
        print(f"  {name:20s}: {imp:.4f} ({imp*100:.1f}%)")
    
    return pipeline


def save_model(pipeline: Pipeline, filepath: str) -> None:
    """
    Save the trained model to a file using joblib.
    
    Args:
        pipeline: Trained sklearn Pipeline
        filepath: Path to save the model
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save
    joblib.dump(pipeline, filepath)
    print(f"\nModel saved to: {filepath}")
    print(f"File size: {os.path.getsize(filepath) / 1024:.1f} KB")


def main():
    """Main training pipeline."""
    print("="*60)
    print("Hypertension CDSS - Decision Tree Model Training")
    print("="*60)
    
    # Generate dataset
    df = generate_dataset()
    
    # Train model
    pipeline = train_model(df)
    
    # Save model
    model_path = os.path.join("model", "cdss_model.pkl")
    save_model(pipeline, model_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nTo start the API server, run:")
    print(f"  python app.py")
    print(f"\nThen open the frontend at:")
    print(f"  http://localhost:5000")


if __name__ == "__main__":
    main()
