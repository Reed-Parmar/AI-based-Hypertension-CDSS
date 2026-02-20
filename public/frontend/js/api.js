/**
 * api.js
 * --------------------------------------------------
 * Handles all backend communication for the
 * Hypertension Clinical Decision Support System.
 *
 * CURRENT MODE: Fully mocked — no network calls.
 * The mock simulates a Decision-Tree classifier
 * using weighted clinical heuristics.
 *
 * HOW TO CONNECT A REAL PYTHON ML BACKEND:
 *   Replace the body of predictHypertension() with
 *   the fetch() block provided in the comments below.
 *   The expected JSON contract is documented there.
 * --------------------------------------------------
 */

const Api = (() => {
  "use strict";

  /** Base URL for the real backend (update when deploying). */
  const BASE_URL = "http://localhost:5000";

  /** Simulated network latency in milliseconds. */
  const MOCK_DELAY_MS = 1400;

  /* ------------------------------------------------------------------ */
  /*  BP Classification — AHA / ACC 2017 Guidelines                      */
  /* ------------------------------------------------------------------ */

  /**
   * Classify blood pressure according to AHA/ACC 2017 categories.
   *
   * @param {number} systolic   mmHg
   * @param {number} diastolic  mmHg
   * @returns {{ category: string, stage: string, color: string }}
   */
  function classifyBP(systolic, diastolic) {
    if (systolic < 120 && diastolic < 80) {
      return { category: "Normal", stage: "normal", color: "green" };
    }
    if ((systolic >= 120 && systolic <= 129) && diastolic < 80) {
      return { category: "Elevated", stage: "elevated", color: "yellow" };
    }
    if ((systolic >= 130 && systolic <= 139) || (diastolic >= 80 && diastolic <= 89)) {
      return { category: "High BP – Stage 1", stage: "stage1", color: "orange" };
    }
    if (systolic >= 140 || diastolic >= 90) {
      return { category: "High BP – Stage 2", stage: "stage2", color: "red" };
    }
    /* Hypertensive crisis */
    if (systolic > 180 || diastolic > 120) {
      return { category: "Hypertensive Crisis", stage: "crisis", color: "darkred" };
    }
    return { category: "Unknown", stage: "unknown", color: "gray" };
  }

  /* ------------------------------------------------------------------ */
  /*  Mock Decision-Tree Prediction Logic                                 */
  /* ------------------------------------------------------------------ */

  /**
   * Simulate a Decision-Tree classifier with weighted clinical features.
   * This is NOT a real model — intended only for UI demonstration.
   *
   * Feature weights are loosely inspired by epidemiological literature:
   *   - Systolic BP  : strongest predictor
   *   - Diastolic BP : second strongest
   *   - Age          : independent risk factor
   *   - BMI          : modifiable risk factor
   *   - Cholesterol  : cardiovascular risk modifier
   *
   * @param {{ age: number, bmi: number, cholesterol: number,
   *           systolic: number, diastolic: number }} params
   * @returns {{ prediction: number, confidence: number,
   *             bpCategory: string, bpStage: string, bpColor: string,
   *             riskFactors: string[] }}
   */
  function _mockPredict({ age, bmi, cholesterol, systolic, diastolic }) {
    let risk = 0;
    const riskFactors = [];

    /* ── Systolic BP — strongest predictor ── */
    if (systolic >= 180) { risk += 42; riskFactors.push("Hypertensive-crisis-level systolic BP"); }
    else if (systolic >= 140) { risk += 35; riskFactors.push("Stage 2 systolic hypertension"); }
    else if (systolic >= 130) { risk += 22; riskFactors.push("Stage 1 systolic hypertension"); }
    else if (systolic >= 120) { risk += 9;  riskFactors.push("Elevated systolic BP"); }

    /* ── Diastolic BP ── */
    if (diastolic >= 120) { risk += 30; riskFactors.push("Hypertensive-crisis-level diastolic BP"); }
    else if (diastolic >= 90) { risk += 22; riskFactors.push("Stage 2 diastolic hypertension"); }
    else if (diastolic >= 80) { risk += 12; riskFactors.push("Stage 1 diastolic hypertension"); }

    /* ── Age factor ── */
    if (age >= 70)      { risk += 14; riskFactors.push("Advanced age (≥70 yrs)"); }
    else if (age >= 60) { risk += 10; riskFactors.push("Senior age (≥60 yrs)"); }
    else if (age >= 45) { risk += 6;  riskFactors.push("Middle age (≥45 yrs)"); }

    /* ── BMI ── */
    if (bmi >= 35)      { risk += 12; riskFactors.push("Severe obesity (BMI ≥35)"); }
    else if (bmi >= 30) { risk += 8;  riskFactors.push("Obesity (BMI ≥30)"); }
    else if (bmi >= 25) { risk += 4;  riskFactors.push("Overweight (BMI ≥25)"); }

    /* ── Total Cholesterol ── */
    if (cholesterol >= 280)      { risk += 10; riskFactors.push("Very high cholesterol (≥280 mg/dL)"); }
    else if (cholesterol >= 240) { risk += 7;  riskFactors.push("High cholesterol (≥240 mg/dL)"); }
    else if (cholesterol >= 200) { risk += 3;  riskFactors.push("Borderline-high cholesterol (≥200 mg/dL)"); }

    /* Clamp total risk to [0, 100] */
    risk = Math.min(100, Math.max(0, risk));

    /* Binary prediction at 50% threshold */
    const prediction = risk >= 50 ? 1 : 0;

    /*
     * Confidence: compress distance-from-boundary into [60, 92].
     * Mimics backend temperature-scaled logic for consistency.
     * 0 distance → 60% (uncertain), 50 distance → 92% (very confident).
     */
    const distance = Math.abs(risk - 50);           // 0 – 50
    const confidence = Math.round(60 + (distance / 50) * 32); // 60 – 92
    const confidenceBand = confidence < 72 ? "Low"
                         : confidence <= 82 ? "Moderate" : "High";

    /* BP classification (AHA 2017) */
    const bpInfo = classifyBP(systolic, diastolic);

    return {
      prediction,          // 0 = no hypertension, 1 = hypertension
      confidence,          // 60–92 %
      confidenceBand,      // "Low" | "Moderate" | "High"
      riskScore: risk,     // raw weighted score 0–100
      bpCategory: bpInfo.category,
      bpStage: bpInfo.stage,
      bpColor: bpInfo.color,
      riskFactors,         // human-readable list of flagged factors
    };
  }

  /* ------------------------------------------------------------------ */
  /*  Public API                                                           */
  /* ------------------------------------------------------------------ */

  /**
   * Request a hypertension prediction for the given patient parameters.
   *
   * Currently returns a mocked result after a simulated delay.
   * To connect a real Python/Flask ML backend, replace the Promise
   * body with the fetch() call shown in comments below.
   *
   * Expected backend JSON response shape:
   * {
   *   "prediction":  0 | 1,
   *   "confidence":  50-98,
   *   "riskScore":   0-100,
   *   "bpCategory":  string,
   *   "bpStage":     string,
   *   "bpColor":     string,
   *   "riskFactors": string[]
   * }
   *
   * @param {{ age: number, bmi: number, cholesterol: number,
   *           systolic: number, diastolic: number }} patientData
   * @returns {Promise<Object>}
   */
  async function predictHypertension(patientData) {
    /* ───────────────────────────────────────────────────────────────────
     * REAL BACKEND — connected to Python Flask API
     */
    const response = await fetch(`${BASE_URL}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patientData),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Server error: ${response.status}`);
    }
    return response.json();
  }

  /* Expose public methods */
  return { predictHypertension, classifyBP };
})();
