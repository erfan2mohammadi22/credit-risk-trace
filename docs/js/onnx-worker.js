// docs/js/onnx-worker.js
// Web Worker for ONNX model inference. Runs off the main thread to keep UI responsive.

// ============================================
// Load ONNX Runtime from local file
// (COEP require-corp blocks CDN loads)
// ============================================
importScripts('../onnx/ort/ort.min.js');

// ============================================
// Runtime configuration
// ============================================
// Local WASM files
ort.env.wasm.wasmPaths = new URL('../onnx/wasm/', self.location.href).href;

// Device-aware thread allocation
const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent || '');
const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 1;

let threads;
if (isMobile) {
  // Mobile: max 2 threads (battery + thermal balance)
  threads = Math.min(2, cores);
} else {
  // Desktop: 2-6 threads (leave one for UI)
  threads = Math.min(6, Math.max(2, cores - 1));
}

ort.env.wasm.numThreads = threads;

console.log(
  `[Worker] ORT loaded. Threads: ${threads} (cores: ${cores}, mobile: ${isMobile})`
);

// ============================================
// State
// ============================================
let session = null;
let metadata = null;

// ============================================
// Message handling
// ============================================
self.onmessage = async (event) => {
  const { type, payload } = event.data;

  try {
    if (type === 'load') {
      const { modelUrl, metadataUrl } = payload;

      // --- 1. Metadata ---
      console.log('[Worker] Fetching metadata:', metadataUrl);
      const metaResponse = await fetch(metadataUrl);
      if (!metaResponse.ok) {
        throw new Error(`Metadata fetch failed: ${metaResponse.status}`);
      }
      metadata = await metaResponse.json();
      console.log('[Worker] Metadata loaded:', metadata.n_features, 'features');
      self.postMessage({ type: 'metadata-loaded', metadata });

      // --- 2. Model with progress ---
      console.log('[Worker] Downloading model:', modelUrl);
      const modelResponse = await fetch(modelUrl);
      if (!modelResponse.ok) {
        throw new Error(`Model fetch failed: ${modelResponse.status}`);
      }

      const contentLength = modelResponse.headers.get('content-length');
      const total = contentLength ? parseInt(contentLength, 10) : 0;

      if (!modelResponse.body) {
        throw new Error('ReadableStream not supported in this browser.');
      }

      const reader = modelResponse.body.getReader();
      const chunks = [];
      let loaded = 0;

      // GitHub Pages sends files gzipped, so content-length is the
      // compressed size, not the actual download size.
      const reportedTotal = total || 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;

        let percent = 0;
        if (reportedTotal > 0) {
          if (loaded <= reportedTotal * 1.05) {
            percent = (loaded / reportedTotal) * 100;
          } else {
            const estimated = Math.max(loaded, reportedTotal * 6);
            percent = Math.min(99, (loaded / estimated) * 100);
          }
        }

        self.postMessage({
          type: 'download-progress',
          loaded,
          total: loaded,
          percent: Math.min(100, percent),
        });
      }

      const allChunks = new Uint8Array(loaded);
      let position = 0;
      for (const chunk of chunks) {
        allChunks.set(chunk, position);
        position += chunk.length;
      }
      console.log('[Worker] Model downloaded:', loaded, 'bytes');

      // --- 3. Create ONNX session ---
      self.postMessage({ type: 'initializing' });
      console.log('[Worker] Creating ONNX session...');

      session = await ort.InferenceSession.create(allChunks.buffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
      });

      console.log('[Worker] Session ready.');
      self.postMessage({ type: 'ready' });
    }

    if (type === 'predict') {
      if (!session) throw new Error('Session not initialized');
      const { input } = payload;

      const vector = buildFeatureVector(input);
      const tensor = new ort.Tensor('float32', vector, [1, vector.length]);

      const inputName = session.inputNames[0];
      const outputs = await session.run({ [inputName]: tensor });

      const labelName = session.outputNames[0];
      const probName = session.outputNames[1];
      const label = outputs[labelName].data[0];
      const probabilities = outputs[probName].data;
      const pd = probabilities[1];

      self.postMessage({
        type: 'prediction',
        result: { pd, prediction: label },
      });
    }
  } catch (error) {
    console.error('[Worker] Error:', error);
    self.postMessage({ type: 'error', error: error.message || String(error) });
  }
};

// ============================================
// Feature engineering (mirrors Python pipeline)
// ============================================
function computeDerivedFeatures(input) {
  const safe = (x) => (x === null || x === undefined || isNaN(x)) ? 0 : x;

  const loan_amnt = safe(input.loan_amnt);
  const annual_inc = safe(input.annual_inc);
  const revol_bal = safe(input.revol_bal);
  const open_acc = safe(input.open_acc);
  const total_acc = safe(input.total_acc);
  const delinq_2yrs = safe(input.delinq_2yrs);
  const pub_rec_bankruptcies = safe(input.pub_rec_bankruptcies);
  const mort_acc = safe(input.mort_acc);
  const fico_range_low = safe(input.fico_range_low);
  const fico_range_high = safe(input.fico_range_high);
  const installment = safe(input.installment);

  const monthly_income = annual_inc / 12;

  return {
    loan_income_ratio: loan_amnt / (annual_inc + 1),
    monthly_income: monthly_income,
    installment_to_income: installment / (monthly_income + 1),
    balance_per_account: revol_bal / (open_acc + 1),
    delinq_per_account: delinq_2yrs / (total_acc + 1),
    log_annual_inc: Math.log1p(annual_inc),
    log_revol_bal: Math.log1p(revol_bal),
    log_loan_amnt: Math.log1p(loan_amnt),
    fico_avg: (fico_range_low + fico_range_high) / 2,
    fico_range: fico_range_high - fico_range_low,
    total_credit_lines: open_acc + mort_acc,
    has_delinq: delinq_2yrs > 0 ? 1 : 0,
    has_bankruptcy: pub_rec_bankruptcies > 0 ? 1 : 0,
    mths_since_last_delinq_missing: 0,
    emp_length_missing: 0,
    mort_acc_missing: 0,
  };
}

function buildFeatureVector(input) {
  const featureNames = metadata.feature_names;
  const mappings = metadata.category_mappings;
  const derived = computeDerivedFeatures(input);
  const merged = { ...input, ...derived };

  const vector = new Float32Array(featureNames.length);

  for (let i = 0; i < featureNames.length; i++) {
    const name = featureNames[i];
    let value = merged[name];

    if (mappings[name]) {
      const key = String(value);
      value = mappings[name][key] !== undefined ? mappings[name][key] : -1;
    }

    if (value === null || value === undefined || Number.isNaN(value)) {
      value = -1;
    }

    vector[i] = value;
  }

  return vector;
}