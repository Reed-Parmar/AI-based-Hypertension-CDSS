/**
 * validators.js
 * --------------------------------------------------
 * Centralised input-validation rules for the
 * Hypertension CDSS patient form.
 *
 * Each validator function returns either:
 *   null           — the value is valid
 *   { message }    — the value is invalid, with human-readable message
 *
 * Rules align with clinical reference ranges used in
 * biomedical literature (AHA / JNC 8 guidelines).
 * --------------------------------------------------
 */

const Validators = (() => {
  "use strict";

  /**
   * Validation rules keyed by input field name.
   * min / max are the hard clinical bounds accepted by the model.
   * warnHigh / warnLow are soft clinical reference ranges shown as hints.
   */
  const RULES = {
    age: {
      label: "Age",
      min: 1,
      max: 120,
      integer: true,
      hint: "Valid range: 1 – 120 years",
    },
    bmi: {
      label: "BMI",
      min: 10,
      max: 70,
      integer: false,
      hint: "Healthy range: 18.5 – 24.9 kg/m²",
    },
    cholesterol: {
      label: "Total Cholesterol",
      min: 50,
      max: 600,
      integer: true,
      hint: "Desirable: < 200 mg/dL",
    },
    systolic: {
      label: "Systolic BP",
      min: 60,
      max: 300,
      integer: true,
      hint: "Normal: < 120 mmHg (AHA 2017)",
    },
    diastolic: {
      label: "Diastolic BP",
      min: 30,
      max: 200,
      integer: true,
      hint: "Normal: < 80 mmHg | Elevated: 80–89 mmHg | Hypertension: ≥ 90 mmHg (AHA 2017)",
    },
  };

  /**
   * Validate a single named field.
   *
   * @param {string} fieldName – key matching an entry in RULES
   * @param {string} rawValue  – raw string value from the input element
   * @returns {null | { message: string }}
   */
  function validateField(fieldName, rawValue) {
    const rule = RULES[fieldName];
    if (!rule) return { message: "Unknown field." };

    /* ── Required check ── */
    if (rawValue === undefined || rawValue === null || String(rawValue).trim() === "") {
      return { message: `${rule.label} is required.` };
    }

    const num = Number(rawValue);

    /* ── Numeric check ── */
    if (Number.isNaN(num)) {
      return { message: `${rule.label} must be a valid number.` };
    }

    /* ── Integer check (where required) ── */
    if (rule.integer && !Number.isInteger(num)) {
      return { message: `${rule.label} must be a whole number (no decimal).` };
    }

    /* ── Positive check ── */
    if (num <= 0) {
      return { message: `${rule.label} must be greater than zero.` };
    }

    /* ── Range check ── */
    if (num < rule.min) {
      return { message: `${rule.label} is too low. Minimum accepted: ${rule.min}.` };
    }
    if (num > rule.max) {
      return { message: `${rule.label} is too high. Maximum accepted: ${rule.max}.` };
    }

    /* ── Cross-field: diastolic must be less than systolic (checked externally) ── */

    return null; // valid
  }

  /**
   * Cross-field validation: diastolic BP must be lower than systolic BP.
   *
   * @param {number} systolic
   * @param {number} diastolic
   * @returns {null | { field: string, message: string }}
   */
  function validateBPRelationship(systolic, diastolic) {
    if (!isNaN(systolic) && !isNaN(diastolic) && diastolic >= systolic) {
      return {
        field: "diastolic",
        message: "Diastolic BP must be less than Systolic BP (clinical requirement).",
      };
    }
    return null;
  }

  /**
   * Validate all form fields at once.
   *
   * @param {Object} data – { age, bmi, cholesterol, systolic, diastolic }
   * @returns {{ valid: boolean, errors: Object }}
   */
  function validateAll(data) {
    const errors = {};
    let valid = true;

    /* Validate individual fields */
    for (const key of Object.keys(RULES)) {
      const result = validateField(key, String(data[key] ?? ""));
      if (result) {
        errors[key] = result.message;
        valid = false;
      }
    }

    /* Cross-field BP relationship check (only if both fields passed individually) */
    if (!errors.systolic && !errors.diastolic) {
      const bpCheck = validateBPRelationship(Number(data.systolic), Number(data.diastolic));
      if (bpCheck) {
        errors[bpCheck.field] = bpCheck.message;
        valid = false;
      }
    }

    return { valid, errors };
  }

  /**
   * Return the ordered list of field names the form expects.
   * @returns {string[]}
   */
  function getFieldNames() {
    return Object.keys(RULES);
  }

  /**
   * Return the hint string for a given field (displayed below the input).
   * @param {string} fieldName
   * @returns {string}
   */
  function getHint(fieldName) {
    return RULES[fieldName]?.hint ?? "";
  }

  /* Public API */
  return { validateField, validateAll, getFieldNames, getHint };
})();
