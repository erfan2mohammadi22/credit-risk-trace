/* ============================================
   CreditTrace — Demo Logic (Web Worker Edition)
   Two modes: Quick Presets (instant) + Custom Analysis (Worker)
   ============================================ */

const LGD = 0.45;
const COST_OF_FUNDS = 0.02;
const OPERATING_COST = 0.01;
const PROFIT_MARGIN = 0.03;

const PRESETS_URL = 'onnx/presets.json';
const WORKER_URL = 'js/onnx-worker.js';

const SUBGRADES = {
  A: ['A1','A2','A3','A4','A5'], B: ['B1','B2','B3','B4','B5'],
  C: ['C1','C2','C3','C4','C5'], D: ['D1','D2','D3','D4','D5'],
  E: ['E1','E2','E3','E4','E5'], F: ['F1','F2','F3','F4','F5'],
  G: ['G1','G2','G3','G4','G5'],
};

const DEFAULT_FORM_VALUES = {
  loan_amnt: 15000, int_rate: 14.5, annual_inc: 60000, dti: 20,
  fico_range_low: 690, revol_util: 55, emp_length: 4,
  term: 36, grade: 'C', sub_grade: 'C3',
  home_ownership: 'RENT', purpose: 'debt_consolidation',
};

let presetsData = null;
let currentPresetKey = null;
let customModelReady = false;
let onnxWorker = null;
let pendingPrediction = null;

function $(id) { return document.getElementById(id); }

function riskClass(pd) {
  if (pd < 0.10) return 'risk-low';
  if (pd < 0.30) return 'risk-medium';
  return 'risk-high';
}

function riskLabel(pd) {
  if (pd < 0.10) return 'Low Risk';
  if (pd < 0.30) return 'Moderate Risk';
  return 'High Risk';
}

function formatMoney(x) {
  return '$' + Math.round(x).toLocaleString('en-US');
}

function pdToScore(pd) {
  const logOdds = Math.log(pd / (1 - pd) + 0.001);
  let score = 850 - 80 * (logOdds + 4);
  score = Math.max(300, Math.min(850, score));
  return Math.round(score);
}

// ============================================
// WEB WORKER
// ============================================
function initWorker() {
  if (onnxWorker) return;

  onnxWorker = new Worker(WORKER_URL);

  onnxWorker.onmessage = (event) => {
    const msg = event.data;
    const type = msg.type;

    if (type === 'metadata-loaded') {
      console.log('[Worker] Metadata loaded:', msg.metadata.n_features, 'features');
      return;
    }

    if (type === 'download-progress') {
      const pct = msg.percent || 0;
      const loadedMB = (msg.loaded / 1024 / 1024).toFixed(1);
      const totalMB = msg.total ? (msg.total / 1024 / 1024).toFixed(1) : '~30';

      const bar = $('download-bar');
      const pctEl = $('download-percent');
      const detEl = $('download-detail');

      if (bar) bar.style.width = Math.min(100, pct) + '%';
      if (pctEl) pctEl.textContent = Math.round(pct) + '%';
      if (detEl) detEl.textContent = `${loadedMB} MB / ${totalMB} MB`;

      if (pct >= 99) switchToInitPhase();
      return;
    }

    if (type === 'initializing') return;

    if (type === 'ready') {
      markInitComplete();
      customModelReady = true;
      setTimeout(() => showCustomStage('form'), 400);
      return;
    }

    if (type === 'prediction') {
      if (pendingPrediction) {
        const cb = pendingPrediction;
        pendingPrediction = null;
        cb.resolve(msg.result);
      }
      return;
    }

    if (type === 'error') {
      console.error('[Worker] Error:', msg.error);
      const detEl = $('init-detail');
      if (detEl) {
        detEl.textContent = 'Error: ' + msg.error;
        detEl.style.color = 'var(--danger)';
      }
      if (pendingPrediction) {
        const cb = pendingPrediction;
        pendingPrediction = null;
        cb.reject(new Error(msg.error));
      }
      return;
    }
  };

  onnxWorker.onerror = (e) => {
    console.error('[Worker] Uncaught:', e.message);
  };
}

let currentPhase = 'idle';

function switchToInitPhase() {
  if (currentPhase !== 'download') return;
  currentPhase = 'init';

  const stepDownload = $('step-download');
  const stepInit = $('step-init');
  const downloadBar = $('download-bar');
  const downloadPct = $('download-percent');
  const initBar = $('init-bar');
  const initPct = $('init-percent');
  const initDet = $('init-detail');

  if (stepDownload) {
    stepDownload.classList.remove('active');
    stepDownload.classList.add('done');
  }
  if (downloadBar) downloadBar.classList.add('complete');
  if (downloadPct) {
    downloadPct.textContent = '100%';
    downloadPct.style.color = 'var(--success)';
  }

  if (stepInit) stepInit.classList.add('active');
  if (initPct) initPct.textContent = '…';
  if (initDet) initDet.textContent = 'Compiling graph...';
  if (initBar) {
    initBar.style.width = '';
    initBar.classList.add('indeterminate');
  }
}

function markInitComplete() {
  currentPhase = 'done';

  const stepInit = $('step-init');
  const initBar = $('init-bar');
  const initPct = $('init-percent');
  const initDet = $('init-detail');

  if (stepInit) {
    stepInit.classList.remove('active');
    stepInit.classList.add('done');
  }
  if (initBar) {
    initBar.classList.remove('indeterminate');
    initBar.classList.add('complete');
    initBar.style.width = '100%';
  }
  if (initPct) {
    initPct.textContent = '✓';
    initPct.style.color = 'var(--success)';
  }
  if (initDet) {
    initDet.textContent = 'Ready';
    initDet.style.color = 'var(--success)';
  }
}

function sendLoadCommand() {
  if (!onnxWorker) return;
  onnxWorker.postMessage({
    type: 'load',
    payload: {
      modelUrl: '../onnx/model.onnx',
      metadataUrl: '../onnx/metadata.json',
    },
  });
}

function predictInWorker(input) {
  return new Promise((resolve, reject) => {
    if (!onnxWorker) {
      reject(new Error('Worker not initialized'));
      return;
    }
    if (pendingPrediction) {
      reject(new Error('Another prediction is in progress'));
      return;
    }
    pendingPrediction = { resolve, reject };
    onnxWorker.postMessage({ type: 'predict', payload: { input } });

    setTimeout(() => {
      if (pendingPrediction) {
        const cb = pendingPrediction;
        pendingPrediction = null;
        cb.reject(new Error('Prediction timed out'));
      }
    }, 30000);
  });
}

// ============================================
// PRESETS
// ============================================
async function loadPresets() {
  if (presetsData) return presetsData;
  try {
    const resp = await fetch(PRESETS_URL);
    if (!resp.ok) throw new Error('Failed to load presets.json');
    presetsData = await resp.json();
    return presetsData;
  } catch (e) {
    console.warn('[Demo] Presets load failed:', e);
    return null;
  }
}

// ============================================
// TABS
// ============================================
function switchTab(tabName) {
  document.querySelectorAll('.demo-tab').forEach((b) => {
    const isActive = b.dataset.tab === tabName;
    b.classList.toggle('active', isActive);
    b.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });

  $('panel-quick').style.display = tabName === 'quick' ? '' : 'none';
  $('panel-custom').style.display = tabName === 'custom' ? '' : 'none';

  if (tabName === 'custom') {
    if (customModelReady) {
      showCustomStage('form');
    } else {
      showCustomStage('warning');
    }
  }
}

function showCustomStage(stage) {
  $('custom-stage-warning').style.display = stage === 'warning' ? '' : 'none';
  $('custom-stage-loading').style.display = stage === 'loading' ? '' : 'none';
  $('custom-stage-form').style.display = stage === 'form' ? '' : 'none';
}

// ============================================
// DOWNLOAD FLOW
// ============================================
function downloadAndContinue() {
  showCustomStage('loading');

  currentPhase = 'download';

  const stepDownload = $('step-download');
  const stepInit = $('step-init');
  const downloadBar = $('download-bar');
  const downloadPct = $('download-percent');
  const downloadDet = $('download-detail');
  const initBar = $('init-bar');
  const initPct = $('init-percent');
  const initDet = $('init-detail');

  if (stepDownload) {
    stepDownload.classList.add('active');
    stepDownload.classList.remove('done');
  }
  if (stepInit) stepInit.classList.remove('active', 'done');

  if (downloadBar) {
    downloadBar.style.width = '0%';
    downloadBar.classList.remove('complete');
  }
  if (downloadPct) {
    downloadPct.textContent = '0%';
    downloadPct.style.color = '';
  }
  if (downloadDet) downloadDet.textContent = '0 MB / ~30 MB';

  if (initBar) {
    initBar.classList.remove('indeterminate', 'complete');
    initBar.style.width = '';
    initBar.style.background = '';
  }
  if (initPct) {
    initPct.textContent = '—';
    initPct.style.color = '';
  }
  if (initDet) {
    initDet.textContent = 'Waiting for download...';
    initDet.style.color = '';
  }

  initWorker();
  sendLoadCommand();
}

function cancelCustomFlow() {
  switchTab('quick');
}

// ============================================
// INPUT LABELS
// ============================================
const INPUT_LABELS = {
  loan_amnt: 'Loan Amount',
  int_rate: 'Interest Rate',
  annual_inc: 'Annual Income',
  dti: 'Debt-to-Income',
  fico_range_low: 'FICO Score',
  revol_util: 'Revolving Util.',
  emp_length: 'Employment Length',
  term: 'Term',
  grade: 'Grade',
  sub_grade: 'Sub-grade',
  home_ownership: 'Home Ownership',
  purpose: 'Purpose',
};

function formatInputValue(key, v) {
  if (v === undefined || v === null) return '—';

  switch (key) {
    case 'loan_amnt':
    case 'annual_inc':
      return formatMoney(v);
    case 'int_rate':
    case 'dti':
      return Number(v).toFixed(1) + '%';
    case 'revol_util':
      return v + '%';
    case 'emp_length':
      return v + ' yrs';
    case 'term':
      return v + ' months';
    case 'home_ownership':
      return { MORTGAGE: 'Mortgage', OWN: 'Own', RENT: 'Rent' }[v] || v;
    case 'purpose':
      return {
        debt_consolidation: 'Debt Consolidation',
        credit_card: 'Credit Card',
        home_improvement: 'Home Improvement',
        major_purchase: 'Major Purchase',
        small_business: 'Small Business',
        car: 'Car',
        medical: 'Medical',
      }[v] || v;
    default:
      return v;
  }
}

function buildInputsHtml(inputs) {
  if (!inputs) return '';

  const rows = Object.keys(INPUT_LABELS)
    .filter((key) => inputs[key] !== undefined)
    .map((key) => `
      <div class="input-row">
        <span class="input-label">${INPUT_LABELS[key]}</span>
        <span class="input-value">${formatInputValue(key, inputs[key])}</span>
      </div>
    `)
    .join('');

  if (!rows) return '';

  return `
    <details class="inputs-details" open>
      <summary>Input Parameters Used</summary>
      <div class="inputs-grid">
        ${rows}
      </div>
    </details>
  `;
}

// ============================================
// RENDER RESULTS
// ============================================
function renderResults(data, options = {}) {
  const { isCustom = false } = options;

  const pd = data.pd;
  const cls = riskClass(pd);
  const label = riskLabel(pd);
  const score = data.credit_score ?? pdToScore(pd);
  const el = data.el;
  const minRate = data.min_rate;
  const loanAmnt = data.inputs?.loan_amnt ?? 0;
  const term = data.inputs?.term ?? '—';
  const pdPct = (pd * 100).toFixed(2);
  const gaugeWidth = Math.min(100, pd * 100);

  const shapItems = data.shap_top || [];
  const maxImpact = Math.max(...shapItems.map((i) => Math.abs(i.impact)), 0.01);
  const shapHtml = shapItems.map((item) => {
    const pct = Math.min(50, (Math.abs(item.impact) / maxImpact) * 50);
    const sign = item.impact > 0 ? 'positive' : 'negative';
    const signChar = item.impact > 0 ? '+' : '−';
    return `
      <div class="shap-row">
        <div class="shap-name" title="${item.feature}">${item.feature}</div>
        <div class="shap-bar-wrap">
          <div class="shap-bar-center"></div>
          <div class="shap-bar ${sign}" style="width: ${pct}%;"></div>
        </div>
        <div class="shap-value ${sign}">${signChar}${Math.abs(item.impact).toFixed(3)}</div>
      </div>
    `;
  }).join('');

  const presetName = data.name || 'Custom Input';
  const presetEmoji = data.emoji || '🔬';

  const badgeHtml = `
    <div class="result-preset-badge">
      <span>${presetEmoji}</span>
      <strong>${presetName}</strong>
    </div>
  `;

  const timeHtml = isCustom
    ? `<div class="financial-item">
        <div class="financial-label">Prediction Time</div>
        <div class="financial-value">${data.time ? data.time.toFixed(0) : '—'}ms</div>
        <div class="financial-sub">In your browser</div>
      </div>`
    : `<div class="financial-item">
        <div class="financial-label">Model</div>
        <div class="financial-value">v5</div>
        <div class="financial-sub">LightGBM · AUC 0.7287</div>
      </div>`;

  const inputsHtml = !isCustom ? buildInputsHtml(data.inputs) : '';

  const ctaHtml = !isCustom ? `
    <div class="result-cta">
      <p class="result-cta-text">Want to try different values?</p>
      <button class="btn btn-secondary" id="go-custom-btn">
        🔬 Open Custom Analysis
      </button>
    </div>
  ` : '';

  const html = `
    <div class="result-header">
      ${badgeHtml}
      <div class="text-dim" style="font-size: 0.72rem; font-family: var(--font-mono);">
        ${formatMoney(loanAmnt)} · ${term}mo
      </div>
    </div>

    <div class="pd-display">
      <div class="pd-label">Probability of Default</div>
      <div class="pd-value ${cls}">${pdPct}%</div>
      <div class="pd-risk-badge ${cls}">${label}</div>
      <div class="pd-gauge">
        <div class="pd-gauge-fill" style="width: ${gaugeWidth}%;"></div>
      </div>
    </div>

    <div class="financial-grid">
      <div class="financial-item">
        <div class="financial-label">Credit Score</div>
        <div class="financial-value">${score}</div>
        <div class="financial-sub">300 – 850</div>
      </div>
      <div class="financial-item">
        <div class="financial-label">Expected Loss</div>
        <div class="financial-value">${formatMoney(el)}</div>
        <div class="financial-sub">PD × LGD × EAD</div>
      </div>
      <div class="financial-item">
        <div class="financial-label">Suggested Rate</div>
        <div class="financial-value">${minRate.toFixed(1)}%</div>
        <div class="financial-sub">Risk-based</div>
      </div>
      ${timeHtml}
    </div>

    <div class="shap-section">
      <div class="shap-title">Why this prediction?</div>
      <div class="shap-sub">Feature impact vs. dataset baseline</div>
      <div class="shap-container">
        ${shapHtml}
      </div>
    </div>

    ${inputsHtml}
    ${ctaHtml}
  `;

  if (isCustom) {
    $('results').innerHTML = html;
  } else {
    $('quick-results').innerHTML = `<div class="card result-card">${html}</div>`;
  }

  const goCustomBtn = $('go-custom-btn');
  if (goCustomBtn) {
    goCustomBtn.addEventListener('click', () => {
      if (data.inputs) fillCustomFormFromPreset(data.inputs);
      switchTab('custom');
    });
  }
}

// ============================================
// PRESET SELECTION
// ============================================
async function selectPreset(presetKey) {
  document.querySelectorAll('.preset-card').forEach((c) => {
    c.classList.toggle('active', c.dataset.preset === presetKey);
  });

  currentPresetKey = presetKey;

  const url = new URL(window.location);
  url.searchParams.set('preset', presetKey);
  window.history.replaceState({}, '', url);

  const data = await loadPresets();
  if (!data || !data.presets[presetKey]) {
    $('quick-results').innerHTML = `
      <div class="status status-error">Failed to load preset data.</div>
    `;
    return;
  }

  renderResults(data.presets[presetKey], {});
}

// ============================================
// CUSTOM FORM
// ============================================
function buildForm() {
  const formEl = $('demo-form');
  if (!formEl) return;

  const fields = [
    { id: 'loan_amnt', label: 'Loan Amount', type: 'slider', min: 500, max: 40000, step: 500, format: (v) => formatMoney(v) },
    { id: 'int_rate', label: 'Interest Rate', type: 'slider', min: 5, max: 30, step: 0.1, format: (v) => v.toFixed(1) + '%' },
    { id: 'annual_inc', label: 'Annual Income', type: 'slider', min: 10000, max: 300000, step: 1000, format: (v) => formatMoney(v) },
    { id: 'dti', label: 'Debt-to-Income', type: 'slider', min: 0, max: 40, step: 0.5, format: (v) => v.toFixed(1) + '%' },
    { id: 'fico_range_low', label: 'FICO Score', type: 'slider', min: 610, max: 845, step: 1, format: (v) => v },
    { id: 'revol_util', label: 'Revolving Utilization', type: 'slider', min: 0, max: 120, step: 1, format: (v) => v + '%' },
    { id: 'emp_length', label: 'Employment Length', type: 'slider', min: 0, max: 10, step: 1, format: (v) => v + ' yrs' },
    { id: 'term', label: 'Loan Term', type: 'select', options: [{ v: 36, l: '36 months' }, { v: 60, l: '60 months' }] },
    { id: 'grade', label: 'Grade', type: 'select', options: ['A','B','C','D','E','F','G'].map((g) => ({ v: g, l: 'Grade ' + g })) },
    { id: 'sub_grade', label: 'Sub-grade', type: 'select', options: [] },
    { id: 'home_ownership', label: 'Home Ownership', type: 'select', options: [
      { v: 'MORTGAGE', l: 'Mortgage' }, { v: 'OWN', l: 'Own' }, { v: 'RENT', l: 'Rent' },
    ] },
    { id: 'purpose', label: 'Purpose', type: 'select', options: [
      { v: 'debt_consolidation', l: 'Debt Consolidation' },
      { v: 'credit_card', l: 'Credit Card' },
      { v: 'home_improvement', l: 'Home Improvement' },
      { v: 'major_purchase', l: 'Major Purchase' },
      { v: 'small_business', l: 'Small Business' },
      { v: 'car', l: 'Car' },
      { v: 'medical', l: 'Medical' },
    ] },
  ];

  formEl.innerHTML = fields.map((f) => {
    if (f.type === 'slider') {
      return `
        <div class="form-group">
          <label class="form-label">
            <span>${f.label}</span>
            <span class="form-value" id="val-${f.id}">—</span>
          </label>
          <input type="range" id="in-${f.id}" min="${f.min}" max="${f.max}" step="${f.step}">
        </div>
      `;
    } else {
      const opts = f.options.map((o) => `<option value="${o.v}">${o.l}</option>`).join('');
      return `
        <div class="form-group">
          <label class="form-label"><span>${f.label}</span></label>
          <select id="in-${f.id}">${opts}</select>
        </div>
      `;
    }
  }).join('');

  fields.forEach((f) => {
    const el = $('in-' + f.id);
    if (f.type === 'slider') {
      el.value = DEFAULT_FORM_VALUES[f.id];
      $('val-' + f.id).textContent = f.format(parseFloat(el.value));
      el.addEventListener('input', () => {
        $('val-' + f.id).textContent = f.format(parseFloat(el.value));
      });
    }
    if (f.id === 'grade') {
      el.addEventListener('change', updateSubgrade);
    }
  });

  $('in-term').value = DEFAULT_FORM_VALUES.term;
  $('in-grade').value = DEFAULT_FORM_VALUES.grade;
  $('in-home_ownership').value = DEFAULT_FORM_VALUES.home_ownership;
  $('in-purpose').value = DEFAULT_FORM_VALUES.purpose;
  updateSubgrade();
  $('in-sub_grade').value = DEFAULT_FORM_VALUES.sub_grade;
}

function updateSubgrade() {
  const grade = $('in-grade').value;
  const subEl = $('in-sub_grade');
  const options = SUBGRADES[grade] || [];
  subEl.innerHTML = options.map((s) => `<option value="${s}">${s}</option>`).join('');
}

function fillCustomFormFromPreset(inputs) {
  Object.keys(inputs).forEach((id) => {
    const el = $('in-' + id);
    if (el) {
      el.value = inputs[id];
      if (el.type === 'range') el.dispatchEvent(new Event('input'));
    }
  });
  updateSubgrade();
  if (inputs.sub_grade) $('in-sub_grade').value = inputs.sub_grade;
}

function getFormValues() {
  const numeric = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_low', 'revol_util', 'emp_length'];
  const categorical = ['term', 'grade', 'sub_grade', 'home_ownership', 'purpose'];

  const values = {};
  numeric.forEach((id) => { values[id] = parseFloat($('in-' + id).value); });
  categorical.forEach((id) => {
    let v = $('in-' + id).value;
    if (id === 'term') v = parseInt(v);
    values[id] = v;
  });

  values.fico_range_high = values.fico_range_low + 4;
  values.delinq_2yrs = 0;
  values.inq_last_6mths = 0;
  values.mths_since_last_delinq = 60;
  values.open_acc = 10;
  values.pub_rec = 0;
  values.revol_bal = Math.round((values.revol_util / 100) * values.annual_inc * 0.3);
  values.total_acc = 20;
  values.application_type = 'Individual';
  values.mort_acc = values.home_ownership === 'MORTGAGE' ? 2 : 0;
  values.pub_rec_bankruptcies = 0;
  values.credit_history_months = 120;
  values.verification_status = 'Verified';
  values.installment = Math.round(
    values.loan_amnt * (values.int_rate / 100 / 12) /
    (1 - Math.pow(1 + values.int_rate / 100 / 12, -values.term))
  );

  return values;
}

// ============================================
// FEATURE IMPACT (lightweight heuristic)
// ============================================
function computeFeatureImpact(baseValues, basePd) {
  const topFeatures = ['sub_grade', 'term', 'int_rate'];
  const impacts = [];

  for (const feat of topFeatures) {
    let delta = 0;

    if (feat === 'sub_grade') {
      const gradeIdx = 'ABCDEFG'.indexOf(baseValues.sub_grade[0]);
      const num = parseInt(baseValues.sub_grade[1]);
      delta = -(((gradeIdx - 3) * 0.05) + ((num - 3) * 0.01));
    } else if (feat === 'term') {
      delta = baseValues.term === 60 ? -0.08 : 0.03;
    } else if (feat === 'int_rate') {
      delta = (baseValues.int_rate - 13.24) * 0.015;
    }

    impacts.push({ feature: feat, impact: delta });
  }

  impacts.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  return impacts;
}

// ============================================
// ANALYZE
// ============================================
async function analyze() {
  const btn = $('analyze-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-sm"></span>Predicting...';

  try {
    const values = getFormValues();

    const t0 = performance.now();
    const result = await predictInWorker(values);
    const t1 = performance.now();
    const time = t1 - t0;

    btn.innerHTML = '<span class="spinner-sm"></span>Computing impact...';
    const impacts = computeFeatureImpact(values, result.pd);

    const data = {
      name: 'Custom Input',
      emoji: '🔬',
      inputs: values,
      pd: result.pd,
      credit_score: pdToScore(result.pd),
      el: result.pd * LGD * values.loan_amnt,
      min_rate: (COST_OF_FUNDS + result.pd * LGD + OPERATING_COST + PROFIT_MARGIN) * 100,
      shap_top: impacts.map((i) => ({
        feature: i.feature,
        value: '—',
        impact: i.impact,
      })),
      time,
    };

    renderResults(data, { isCustom: true });
    btn.innerHTML = 'Analyze';
  } catch (err) {
    console.error('[Demo] Analyze failed:', err);
    $('results').innerHTML = `
      <div class="status status-error">Error: ${err.message}</div>
    `;
    btn.innerHTML = 'Analyze';
  } finally {
    btn.disabled = false;
  }
}

// ============================================
// INIT
// ============================================
document.addEventListener('DOMContentLoaded', async () => {
  buildForm();

  document.querySelectorAll('.demo-tab').forEach((btn) => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  document.querySelectorAll('.preset-card').forEach((card) => {
    card.addEventListener('click', () => selectPreset(card.dataset.preset));
  });

  const cancelBtn = $('custom-cancel');
  const startBtn = $('custom-start');
  const analyzeBtn = $('analyze-btn');

  if (cancelBtn) cancelBtn.addEventListener('click', cancelCustomFlow);
  if (startBtn) startBtn.addEventListener('click', downloadAndContinue);
  if (analyzeBtn) analyzeBtn.addEventListener('click', analyze);

  // Streamlit link — placeholder URL, will be updated after deploy
  const streamlitBtn = $('custom-streamlit');
  if (streamlitBtn) {
    // Currently href="#"; replace with real URL after Streamlit deploy
    // Example: streamlitBtn.href = 'https://credittrace.streamlit.app';
  }

  const urlPreset = new URL(window.location).searchParams.get('preset');
  if (urlPreset && document.querySelector(`.preset-card[data-preset="${urlPreset}"]`)) {
    await selectPreset(urlPreset);
  }
});