/**
 * app.js
 * --------------------------------------------------
 * Main application controller for the
 * Hypertension Clinical Decision Support System.
 *
 * Responsibilities:
 *   - Wire up all DOM event listeners
 *   - Coordinate: Validators -> Api -> UI pipeline
 *   - Keep every other module fully decoupled
 *
 * Module order in index.html (required):
 *   validators.js -> api.js -> ui.js -> app.js
 * --------------------------------------------------
 */

const App = (() => {
  "use strict";

  /* ------------------------------------------------------------------ */
  /*  Internal Helpers                                                    */
  /* ------------------------------------------------------------------ */

  /**
   * Run a full form validation pass and sync the UI state.
   * Called on every keystroke and on submit.
   *
   * @returns {boolean} true when all fields are valid
   */
  function _validateForm() {
    const formData = UI.getFormData();
    const { valid, errors } = Validators.validateAll(formData);

    /* Reflect per-field error messages */
    Validators.getFieldNames().forEach((name) => {
      UI.setFieldError(name, errors[name] || null);
    });

    /* Gate the submit button */
    UI.setButtonEnabled(valid);

    return valid;
  }

  /**
   * Handle the form submit event:
   *   1. Final validation pass
   *   2. Show loading state
   *   3. Call (mocked/real) prediction API
   *   4. Render result via UI module
   *
   * @param {Event} event
   */
  async function _handleSubmit(event) {
    event.preventDefault();

    if (!_validateForm()) return;

    /* Assemble typed patient data object */
    const raw = UI.getFormData();
    const patientData = {
      age:         Number(raw.age),
      bmi:         Number(raw.bmi),
      cholesterol: Number(raw.cholesterol),
      systolic:    Number(raw.systolic),
      diastolic:   Number(raw.diastolic),
    };

    /* Enter loading state */
    UI.setLoading(true);
    UI.hideResult();

    try {
      /* Prediction call -- swap Api.predictHypertension for a real fetch in api.js */
      const result = await Api.predictHypertension(patientData);
      UI.showResult(result);
    } catch (error) {
      console.error("[CDSS] Prediction failed:", error);
      _showError("Prediction failed. Please try again.");
    } finally {
      UI.setLoading(false);
      _validateForm(); // restore button state after loading
    }
  }

  /**
   * Display a transient error banner.
   * @param {string} message
   */
  function _showError(message) {
    const banner = document.getElementById("error-banner");
    if (!banner) return;
    banner.textContent = message;
    banner.hidden = false;
    setTimeout(() => { banner.hidden = true; }, 5000);
  }

  /**
   * Handle the reset / clear form action.
   */
  function _handleReset() {
    const form = document.getElementById("prediction-form");
    if (form) form.reset();
    UI.clearAllErrors();
    UI.hideResult();
    UI.setButtonEnabled(false);
  }

  /* ------------------------------------------------------------------ */
  /*  Event Binding                                                       */
  /* ------------------------------------------------------------------ */

  /**
   * Attach all event listeners to the page.
   * Called once on DOMContentLoaded.
   */
  function _bindEvents() {
    const form      = document.getElementById("prediction-form");
    const resetBtn  = document.getElementById("reset-btn");

    if (!form) {
      console.error("[CDSS] #prediction-form not found.");
      return;
    }

    /* Live validation � runs on every input change */
    Validators.getFieldNames().forEach((fieldName) => {
      const input = document.getElementById(fieldName);
      if (!input) return;

      /* Input event: validate this field + re-check overall form state */
      input.addEventListener("input", () => {
        const fieldError = Validators.validateField(fieldName, input.value);
        UI.setFieldError(fieldName, fieldError ? fieldError.message : null);

        /* Re-run full validation to catch cross-field rules */
        const { valid } = Validators.validateAll(UI.getFormData());
        UI.setButtonEnabled(valid);
      });

      /* Blur event: enforce error display when user leaves a field */
      input.addEventListener("blur", () => {
        const fieldError = Validators.validateField(fieldName, input.value);
        UI.setFieldError(fieldName, fieldError ? fieldError.message : null);
      });
    });

    /* Form submission */
    form.addEventListener("submit", _handleSubmit);

    /* Reset button */
    if (resetBtn) resetBtn.addEventListener("click", _handleReset);
  }

  /* ------------------------------------------------------------------ */
  /*  Initialisation                                                      */
  /* ------------------------------------------------------------------ */

  /**
   * Boot the application.
   * Wires events and sets initial UI state.
   */
  function init() {
    _bindEvents();
    UI.setButtonEnabled(false); // disabled until all fields are valid
    console.info("[CDSS] Application initialised — v1.0");
  }

  /* Boot as soon as the DOM is ready */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  /* Expose init for external testing / extension */
  return { init };
})();