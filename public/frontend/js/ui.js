/**
 * ui.js
 * --------------------------------------------------
 * DOM updates and UI rendering logic for the
 * Hypertension CDSS dashboard.
 *
 * ALL direct DOM manipulation lives here.
 * Other modules (app.js, api.js) must never touch
 * the DOM directly — they call these functions instead.
 * --------------------------------------------------
 */

const UI = (() => {
  "use strict";

  /*  Cached element getter helpers (queried lazily)  */
  const el = (id) => document.getElementById(id);

  /**
   * Toggle the enabled/disabled state of the predict button.
   * @param {boolean} enabled
   */
  function setButtonEnabled(enabled) {
    const btn = el("predict-btn");
    if (btn) btn.disabled = !enabled;
  }

  /**
   * Show/hide the loading state on the predict button.
   * @param {boolean} loading
   */
  function setLoading(loading) {
    const btn     = el("predict-btn");
    const btnText = el("predict-btn-text");
    const loader  = el("predict-btn-loader");
    const icon    = btn ? btn.querySelector(".btn-icon") : null;

    if (!btn || !btnText || !loader) return;

    if (loading) {
      btnText.hidden = true;
      loader.hidden  = false;
      btn.disabled   = true;
      if (icon) icon.hidden = true;
    } else {
      btnText.hidden = false;
      loader.hidden  = true;
      btn.disabled   = false;
      if (icon) icon.hidden = false;
    }
  }

  /**
   * Set or clear an inline validation error for a field.
   * Also toggles .input--error / .input--valid CSS classes.
   *
   * @param {string}      fieldName
   * @param {string|null} message  — null clears the error
   */
  function setFieldError(fieldName, message) {
    const errorEl = el(`${fieldName}-error`);
    const input   = el(fieldName);

    if (errorEl) errorEl.textContent = message || "";

    if (input) {
      const hasValue = input.value.trim() !== "";
      input.classList.toggle("input--error", !!message);
      input.classList.toggle("input--valid", !message && hasValue);
    }
  }

  /**
   * Clear all field errors and reset input appearance.
   */
  function clearAllErrors() {
    Validators.getFieldNames().forEach((name) => setFieldError(name, null));
  }

  /**
   * Render the prediction result card.
   *
   * @param {{ prediction: number, confidence: number, riskScore: number,
   *           bpCategory: string, bpStage: string, bpColor: string,
   *           riskFactors: string[] }} result
   */
  function showResult(result) {
    const section     = el("results-section");
    const card        = el("result-card");
    const resultIcon  = el("result-icon");
    const resultText  = el("result-text");
    const barFill     = el("confidence-bar-fill");
    const confValue   = el("confidence-value");
    const bpBadge     = el("bp-category-badge");
    const riskList    = el("risk-factors-list");
    const scoreNum    = el("risk-score-value");

    if (!section || !card) return;

    const isHypertensive = result.prediction === 1;

    /*  Result card colour  */
    card.classList.remove("result--danger", "result--success");
    card.classList.add(isHypertensive ? "result--danger" : "result--success");

    /*  Icon  */
    if (resultIcon) {
      resultIcon.innerHTML = isHypertensive
        ? `<svg width="32" height="32" viewBox="0 0 24 24" fill="none"
              stroke="#dc2626" stroke-width="2" stroke-linecap="round"
              stroke-linejoin="round" aria-hidden="true">
             <circle cx="12" cy="12" r="10"/>
             <line x1="12" y1="8" x2="12" y2="12"/>
             <line x1="12" y1="16" x2="12.01" y2="16"/>
           </svg>`
        : `<svg width="32" height="32" viewBox="0 0 24 24" fill="none"
              stroke="#16a34a" stroke-width="2" stroke-linecap="round"
              stroke-linejoin="round" aria-hidden="true">
             <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
             <polyline points="22 4 12 14.01 9 11.01"/>
           </svg>`;
    }

    /*  Label  */
    if (resultText) {
      resultText.textContent = isHypertensive
        ? "Hypertension Detected"
        : "No Hypertension Detected";
    }

    /*  Confidence bar (animated)  */
    if (barFill) {
      barFill.style.width = "0%";
      void barFill.offsetWidth; // trigger reflow for CSS transition
      barFill.style.width = `${result.confidence}%`;
    }
    if (confValue) confValue.textContent = `${result.confidence}%`;

    /*  Confidence band badge  */
    const confBand = el("confidence-band");
    if (confBand && result.confidenceBand) {
      confBand.textContent = result.confidenceBand;
      confBand.className = `confidence-band confidence-band--${result.confidenceBand.toLowerCase()}`;
    }

    /*  BP category badge  */
    if (bpBadge) {
      bpBadge.textContent = result.bpCategory;
      bpBadge.className   = `bp-badge bp-badge--${result.bpStage}`;
    }

    /*  Risk score number  */
    if (scoreNum) scoreNum.textContent = result.riskScore ?? "--";

    /*  Risk factors list  */
    if (riskList) {
      riskList.innerHTML = "";
      if (result.riskFactors && result.riskFactors.length > 0) {
        result.riskFactors.forEach((factor) => {
          const li = document.createElement("li");
          li.textContent = factor;
          riskList.appendChild(li);
        });
      } else {
        const li = document.createElement("li");
        li.textContent = "No significant risk factors flagged.";
        li.className = "risk-factor--none";
        riskList.appendChild(li);
      }
    }

    /*  Show section & scroll into view  */
    section.hidden = false;
    section.scrollIntoView({ behavior: "smooth", block: "nearest" });

    /*  Render clinical interpretation panel  */
    renderClinicalInterpretation(result);

    /*  Add to session history table  */
    _appendHistory(result);
  }

  /**
   * Hide the results section.
   */
  function hideResult() {
    const section = el("results-section");
    if (section) section.hidden = true;
    hideClinicalInterpretation();
  }

  /**
   * Gather current form values keyed by field name.
   * @returns {Object}
   */
  function getFormData() {
    const data = {};
    Validators.getFieldNames().forEach((name) => {
      const input = el(name);
      data[name] = input ? input.value : "";
    });
    return data;
  }

  /**
   * Show or hide a full-page loading overlay.
   * @param {boolean} visible
   */
  function setOverlay(visible) {
    const overlay = el("loading-overlay");
    if (overlay) overlay.hidden = !visible;
  }

  /* ------------------------------------------------------------------ */
  /*  Session History Table                                               */
  /* ------------------------------------------------------------------ */

  let _historyCount = 0;

  /**
   * Append a new row to the history table.
   * @param {Object} result
   */
  function _appendHistory(result) {
    const tbody     = el("history-tbody");
    const emptyRow  = el("history-empty-row");
    const histSect  = el("history-section");

    if (!tbody) return;

    /* Remove placeholder row on first entry */
    if (emptyRow) emptyRow.remove();

    _historyCount += 1;

    /* Gather form values for the row */
    const data = getFormData();

    const VALID_BP_STAGES = ["normal", "elevated", "stage1", "stage2", "crisis", "unknown"];

    const tr = document.createElement("tr");
    tr.className = result.prediction === 1 ? "history-row--danger" : "history-row--success";

    /* Build cells safely using textContent to prevent XSS */
    const cells = [
      String(_historyCount),
      data.age ?? "--",
      data.bmi ?? "--",
      data.cholesterol ?? "--",
      `${data.systolic ?? "--"} / ${data.diastolic ?? "--"}`,
    ];

    cells.forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    });

    /* BP badge cell */
    const bpTd = document.createElement("td");
    const bpSpan = document.createElement("span");
    bpSpan.className = "bp-badge";
    const stage = VALID_BP_STAGES.includes(result.bpStage) ? result.bpStage : "unknown";
    bpSpan.classList.add(`bp-badge--${stage}`);
    bpSpan.textContent = result.bpCategory;
    bpTd.appendChild(bpSpan);
    tr.appendChild(bpTd);

    /* Prediction cell */
    const predTd = document.createElement("td");
    predTd.classList.add(result.prediction === 1 ? "text-danger" : "text-success");
    predTd.textContent = result.prediction === 1 ? "\u274C Hypertensive" : "\u2705 Normal";
    tr.appendChild(predTd);

    /* Confidence cell */
    const confTd = document.createElement("td");
    confTd.textContent = `${result.confidence}%`;
    tr.appendChild(confTd);

    tbody.prepend(tr); // newest first

    /* Show history section */
    if (histSect) histSect.hidden = false;
  }

  /**
   * Clear the session history table.
   */
  function clearHistory() {
    const tbody    = el("history-tbody");
    const histSect = el("history-section");
    if (!tbody) return;

    tbody.innerHTML = `
      <tr id="history-empty-row">
        <td colspan="8" class="history-empty">No predictions yet in this session.</td>
      </tr>`;
    _historyCount = 0;
    if (histSect) histSect.hidden = true;
  }

  /**
   * Render the Clinical Interpretation panel after a prediction.
   * Shows primary decision driver, secondary factors, and model note.
   *
   * @param {{ prediction: number, confidence: number, riskScore: number,
   *           bpCategory: string, bpStage: string, riskFactors: string[] }} result
   */
  function renderClinicalInterpretation(result) {
    const panel = el("clinical-interpretation");
    if (!panel) return;

    const isHypertensive = result.prediction === 1;

    /* Primary decision driver — highest-weight risk factor or BP status */
    const primary = (result.riskFactors && result.riskFactors.length > 0)
      ? result.riskFactors[0]
      : (isHypertensive ? "Elevated Blood Pressure" : "Blood pressure within normal range");

    /* Secondary factors — remaining risk factors */
    const secondary = (result.riskFactors && result.riskFactors.length > 1)
      ? result.riskFactors.slice(1)
      : [];

    /* Primary driver */
    const driverEl = el("ci-primary-driver");
    if (driverEl) driverEl.textContent = primary;

    /* Secondary list */
    const secList = el("ci-secondary-list");
    if (secList) {
      secList.innerHTML = "";
      if (secondary.length > 0) {
        secondary.forEach((f) => {
          const li = document.createElement("li");
          li.textContent = f;
          secList.appendChild(li);
        });
      } else {
        const li = document.createElement("li");
        li.textContent = "No additional risk factors identified.";
        li.className = "ci-factor--none";
        secList.appendChild(li);
      }
    }

    /* Confidence */
    const confEl = el("ci-confidence");
    if (confEl) confEl.textContent = `${result.confidence}%`;

    /* Confidence band in CI panel */
    const ciBand = el("ci-confidence-band");
    if (ciBand && result.confidenceBand) {
      ciBand.textContent = result.confidenceBand;
      ciBand.className = `confidence-band confidence-band--${result.confidenceBand.toLowerCase()}`;
    }

    /* Colour accent */
    const card = el("ci-card");
    if (card) {
      card.classList.remove("ci-card--danger", "ci-card--success");
      card.classList.add(isHypertensive ? "ci-card--danger" : "ci-card--success");
    }

    panel.hidden = false;
  }

  /**
   * Hide the clinical interpretation panel.
   */
  function hideClinicalInterpretation() {
    const panel = el("clinical-interpretation");
    if (panel) panel.hidden = true;
  }

  /* Public API */
  return {
    setButtonEnabled,
    setLoading,
    setFieldError,
    clearAllErrors,
    showResult,
    hideResult,
    getFormData,
    setOverlay,
    clearHistory,
    renderClinicalInterpretation,
    hideClinicalInterpretation,
  };

})();