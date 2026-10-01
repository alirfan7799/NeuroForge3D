/**
 * NeuroForge-3D Main Application Controller
 * Coordinates Three.js 3D views, Web Worker compute engine, Chart.js telemetry,
 * 2D Decision Boundary canvas, and user interface state.
 */

import { Viewport } from './scene/viewport.js';
import { MLPView } from './scene/mlpView.js';
import { CNNView } from './scene/cnnView.js';
import { LSTMView } from './scene/lstmView.js';
import { MetricsChart } from './charts/metricsChart.js';
import { BoundaryCanvas } from './charts/boundaryCanvas.js';
import { generateDataset } from './data/datasets.js';

class NeuroForgeApp {
  constructor() {
    this.currentArch = 'mlp';
    this.currentDatasetType = 'spiral';
    this.isTraining = false;
    this.topology = [2, 8, 8, 2];
    this.learningRate = 0.03;
    this.optimizer = 'adam';
    this.activation = 'relu';
    this.batchSize = 32;
    this.targetEpochs = 0; // 0 = continuous

    this.worker = null;
    this.viewport = null;
    this.mlpView = null;
    this.cnnView = null;
    this.lstmView = null;
    this.metricsChart = null;
    this.boundaryCanvas = null;

    this.currentDataset = null;
    this.init();
  }

  init() {
    const container = document.getElementById('canvas-container');
    this.viewport = new Viewport(container, this.handle3DHover.bind(this));

    this.mlpView = new MLPView(this.viewport);
    this.cnnView = new CNNView(this.viewport);
    this.lstmView = new LSTMView(this.viewport);

    this.viewport.onBeforeRender = (delta) => {
      this.mlpView.update(delta);
      this.cnnView.update(delta);
      this.lstmView.update(delta);
    };

    const metricsCanvas = document.getElementById('metrics-chart');
    if (metricsCanvas) {
      this.metricsChart = new MetricsChart(metricsCanvas);
    }

    const boundaryCanvasEl = document.getElementById('boundary-canvas');
    if (boundaryCanvasEl) {
      this.boundaryCanvas = new BoundaryCanvas(boundaryCanvasEl, (coords) => {
        this.runInference(coords);
      });
    }

    this.initWorker();
    this.loadDataset(this.currentDatasetType);
    this.bindEvents();
    this.setupInferencePanelForArch(this.currentArch);
  }

  initWorker() {
    this.worker = new Worker('./js/worker/engine.worker.js');

    this.worker.onmessage = (e) => {
      const { type, data } = e.data;

      switch (type) {
        case 'MODEL_READY':
          this.handleModelReady(data);
          break;
        case 'EPOCH_UPDATE':
          this.handleEpochUpdate(data);
          break;
        case 'BOUNDARY_UPDATE':
          if (this.boundaryCanvas) {
            this.boundaryCanvas.updateBoundary(data);
          }
          break;
        case 'INFERENCE_RESULT':
          this.handleInferenceResult(data);
          break;
        case 'TARGET_REACHED':
          this.handleTargetReached(data);
          break;
        default:
          break;
      }
    };

    this.worker.postMessage({
      type: 'INIT_ARCH',
      data: {
        architecture: this.currentArch,
        topology: this.topology,
        activation: this.activation,
        learningRate: this.learningRate,
        optimizer: this.optimizer,
        batchSize: this.batchSize,
        targetEpochs: this.targetEpochs
      }
    });
  }

  loadDataset(type) {
    this.currentDatasetType = type;
    this.currentDataset = generateDataset(type);

    if (this.boundaryCanvas && (type === 'spiral' || type === 'moons' || type === 'circles' || type === 'xor')) {
      this.boundaryCanvas.setDataset(this.currentDataset);
    }

    if (this.worker) {
      this.worker.postMessage({
        type: 'SET_DATASET',
        data: this.currentDataset
      });
    }

    if (type === 'digits' && this.cnnView && this.currentDataset.samples) {
      this.cnnView.setInputImage(this.currentDataset.samples[0].grid);
    }
  }

  handleModelReady(data) {
    const paramBadge = document.getElementById('param-count-badge');
    if (paramBadge && data.totalParams) {
      paramBadge.textContent = `${data.totalParams} params`;
    }

    if (data.weights && this.mlpView && this.currentArch === 'mlp') {
      this.mlpView.updateWeights(data.weights);
    }
  }

  handleEpochUpdate(data) {
    const epochEl = document.getElementById('stat-epoch');
    const lossEl = document.getElementById('stat-loss');
    const accEl = document.getElementById('stat-acc');

    if (epochEl) epochEl.textContent = data.epoch;
    if (lossEl) lossEl.textContent = data.loss.toFixed(4);
    if (accEl) accEl.textContent = `${Math.round(data.accuracy * 100)}%`;

    if (this.metricsChart) {
      this.metricsChart.addPoint(data.epoch, data.loss, data.accuracy);
    }

    if (this.currentArch === 'mlp') {
      if (this.mlpView && data.weights) {
        this.mlpView.updateWeights(data.weights);
      }
      if (this.boundaryCanvas && data.boundary) {
        this.boundaryCanvas.updateBoundary(data.boundary);
      }
      if (this.isTraining && data.epoch % 4 === 0 && this.mlpView) {
        this.mlpView.triggerSignalWave('forward');
        if (data.epoch % 8 === 0) {
          this.mlpView.triggerSignalWave('backward');
        }
      }
    } else if (this.currentArch === 'cnn') {
      if (this.cnnView) {
        this.cnnView.updateActivations(data);
      }
    }

    // Update gradient magnitude spectrum bars
    if (data.gradientMagnitudes && data.gradientMagnitudes.length > 0) {
      this.renderGradientSpectrum(data.gradientMagnitudes);
    }

    // Update dynamic inference readout as training evolves
    if (data.samplePred !== undefined && data.samplePred !== null) {
      this.updateInferenceReadout(data.samplePred);
    }
  }

  handleInferenceResult(data) {
    if (this.mlpView && this.currentArch === 'mlp') {
      this.mlpView.updateActivations(data.layerActivations);
      this.mlpView.triggerSignalWave('forward');
    }

    if (data.prediction) {
      this.updateInferenceReadout(data.prediction);
      if (this.boundaryCanvas && this.currentArch === 'mlp') {
        this.boundaryCanvas.setProbePrediction(data.prediction);
      }
    }
  }

  handleTargetReached(data) {
    this.isTraining = false;
    this.updatePlayButtonState();
    const badge = document.getElementById('inference-status-badge');
    if (badge) {
      badge.textContent = `Epoch ${data.epoch} Reached`;
      badge.style.color = '#10b981';
      badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    }
  }

  setupInferencePanelForArch(arch) {
    const container = document.getElementById('inference-bars-container');
    const title = document.getElementById('inference-title');
    const badge = document.getElementById('inference-status-badge');
    if (!container) return;

    if (badge) {
      badge.textContent = 'Untrained (0%)';
      badge.style.color = '#94a3b8';
      badge.style.borderColor = 'rgba(255, 255, 255, 0.15)';
    }

    if (arch === 'mlp') {
      title.innerHTML = 'Classification <span class="accent">Confidence</span>';
      container.innerHTML = `
        <div class="pred-row">
          <span style="color: #ff2a85; font-weight: 600;">Class 0</span>
          <div class="pred-track"><div id="pred-bar-0" class="pred-fill class-0" style="width: 0%;"></div></div>
          <span id="pred-val-0" class="font-mono" style="font-size: 0.72rem;">0%</span>
        </div>
        <div class="pred-row">
          <span style="color: #00f0ff; font-weight: 600;">Class 1</span>
          <div class="pred-track"><div id="pred-bar-1" class="pred-fill class-1" style="width: 0%;"></div></div>
          <span id="pred-val-1" class="font-mono" style="font-size: 0.72rem;">0%</span>
        </div>
      `;
    } else if (arch === 'cnn') {
      title.innerHTML = 'CNN 3-Class <span class="accent">Softmax</span>';
      container.innerHTML = `
        <div class="pred-row">
          <span style="color: #00f0ff; font-weight: 600;">Ring / Circle</span>
          <div class="pred-track"><div id="pred-bar-0" class="pred-fill" style="width: 0%; background: #00f0ff;"></div></div>
          <span id="pred-val-0" class="font-mono" style="font-size: 0.72rem;">0%</span>
        </div>
        <div class="pred-row">
          <span style="color: #8b5cf6; font-weight: 600;">Vertical Bar</span>
          <div class="pred-track"><div id="pred-bar-1" class="pred-fill" style="width: 0%; background: #8b5cf6;"></div></div>
          <span id="pred-val-1" class="font-mono" style="font-size: 0.72rem;">0%</span>
        </div>
        <div class="pred-row">
          <span style="color: #ff2a85; font-weight: 600;">Cross / Diag</span>
          <div class="pred-track"><div id="pred-bar-2" class="pred-fill" style="width: 0%; background: #ff2a85;"></div></div>
          <span id="pred-val-2" class="font-mono" style="font-size: 0.72rem;">0%</span>
        </div>
      `;
    } else if (arch === 'lstm') {
      title.innerHTML = 'Temporal <span class="accent">Sequence Target</span>';
      container.innerHTML = `
        <div class="pred-row">
          <span style="color: #f59e0b; font-weight: 600;">Target (y)</span>
          <div class="pred-track"><div id="lstm-target-bar" class="pred-fill" style="width: 50%; background: #f59e0b;"></div></div>
          <span id="lstm-target-val" class="font-mono" style="font-size: 0.72rem;">+0.720</span>
        </div>
        <div class="pred-row">
          <span style="color: #00f0ff; font-weight: 600;">Pred (ŷ)</span>
          <div class="pred-track"><div id="lstm-pred-bar" class="pred-fill" style="width: 0%; background: #00f0ff;"></div></div>
          <span id="lstm-pred-val" class="font-mono" style="font-size: 0.72rem;">0.000</span>
        </div>
        <div style="font-size: 0.72rem; color: var(--text-dim); text-align: right; margin-top: 4px;">
          Residual Error: <strong id="lstm-err-val" class="text-cyan">0.000</strong>
        </div>
      `;
    }
  }

  updateInferenceReadout(pred) {
    const badge = document.getElementById('inference-status-badge');
    if (badge) {
      badge.textContent = 'Active (Live)';
      badge.style.color = '#00f0ff';
      badge.style.borderColor = 'rgba(0, 240, 255, 0.4)';
    }

    if (this.currentArch === 'mlp' && Array.isArray(pred) && pred.length >= 2) {
      const b0 = document.getElementById('pred-bar-0');
      const b1 = document.getElementById('pred-bar-1');
      const v0 = document.getElementById('pred-val-0');
      const v1 = document.getElementById('pred-val-1');
      const p0 = Math.round(pred[0] * 100);
      const p1 = Math.round(pred[1] * 100);
      if (b0) b0.style.width = `${p0}%`;
      if (b1) b1.style.width = `${p1}%`;
      if (v0) v0.textContent = `${p0}%`;
      if (v1) v1.textContent = `${p1}%`;
    } else if (this.currentArch === 'cnn' && Array.isArray(pred) && pred.length >= 3) {
      for (let i = 0; i < 3; i++) {
        const bar = document.getElementById(`pred-bar-${i}`);
        const val = document.getElementById(`pred-val-${i}`);
        const pct = Math.round(pred[i] * 100);
        if (bar) bar.style.width = `${pct}%`;
        if (val) val.textContent = `${pct}%`;
      }
    } else if (this.currentArch === 'lstm') {
      const predVal = typeof pred === 'number' ? pred : (pred[0] || 0);
      const targetVal = 0.72; // Representative synthetic sequence target
      const bar = document.getElementById('lstm-pred-bar');
      const valEl = document.getElementById('lstm-pred-val');
      const errEl = document.getElementById('lstm-err-val');

      const normPct = Math.max(0, Math.min(100, Math.round(((predVal + 1) / 2) * 100)));
      if (bar) bar.style.width = `${normPct}%`;
      if (valEl) valEl.textContent = (predVal >= 0 ? '+' : '') + predVal.toFixed(3);
      if (errEl) errEl.textContent = Math.abs(predVal - targetVal).toFixed(4);
    }
  }

  renderGradientSpectrum(grads) {
    const container = document.getElementById('gradient-spectrum-bars');
    if (!container) return;

    container.innerHTML = '';
    if (!grads || grads.length === 0) {
      container.innerHTML = '<span class="text-xs text-slate-500">Grad tape active...</span>';
      return;
    }

    const maxMag = Math.max(0.001, ...grads.map(g => g.magnitude));

    grads.slice(0, 6).forEach((g, idx) => {
      const row = document.createElement('div');
      row.className = 'spectrum-row';

      const label = document.createElement('span');
      label.className = 'spectrum-label';
      label.textContent = `L${idx + 1}`;

      const barTrack = document.createElement('div');
      barTrack.className = 'spectrum-track';

      const barFill = document.createElement('div');
      barFill.className = 'spectrum-fill';
      const pct = Math.min(100, Math.round((g.magnitude / maxMag) * 100));
      barFill.style.width = `${Math.max(4, pct)}%`;

      const val = document.createElement('span');
      val.className = 'spectrum-val';
      val.textContent = g.magnitude.toFixed(4);

      barTrack.appendChild(barFill);
      row.appendChild(label);
      row.appendChild(barTrack);
      row.appendChild(val);
      container.appendChild(row);
    });
  }

  runInference(input) {
    if (this.worker) {
      this.worker.postMessage({
        type: 'RUN_INFERENCE',
        data: { input }
      });
    }
  }

  switchArchitecture(arch) {
    this.currentArch = arch;
    this.isTraining = false;
    this.updatePlayButtonState();

    if (this.metricsChart) this.metricsChart.reset();
    const epochEl = document.getElementById('stat-epoch');
    const lossEl = document.getElementById('stat-loss');
    const accEl = document.getElementById('stat-acc');
    if (epochEl) epochEl.textContent = '0';
    if (lossEl) lossEl.textContent = '0.000';
    if (accEl) accEl.textContent = '0%';

    // Toggle 3D Views
    this.mlpView.setVisible(arch === 'mlp');
    this.cnnView.setVisible(arch === 'cnn');
    this.lstmView.setVisible(arch === 'lstm');

    // Toggle DOM panels
    const mlpPanel = document.getElementById('panel-mlp-config');
    const cnnPanel = document.getElementById('panel-cnn-config');
    const lstmPanel = document.getElementById('panel-lstm-config');
    const boundaryBox = document.getElementById('boundary-box');

    if (mlpPanel) mlpPanel.style.display = arch === 'mlp' ? 'block' : 'none';
    if (cnnPanel) cnnPanel.style.display = arch === 'cnn' ? 'block' : 'none';
    if (lstmPanel) lstmPanel.style.display = arch === 'lstm' ? 'block' : 'none';
    if (boundaryBox) boundaryBox.style.display = arch === 'mlp' ? 'block' : 'none';

    // Adjust optimal learning rate per architecture
    if (arch === 'cnn') {
      this.learningRate = 0.002;
    } else if (arch === 'lstm') {
      this.learningRate = 0.008;
    } else {
      this.learningRate = 0.030;
    }
    const lrSlider = document.getElementById('slider-lr');
    const lrVal = document.getElementById('val-lr');
    if (lrSlider) lrSlider.value = this.learningRate;
    if (lrVal) lrVal.textContent = this.learningRate.toFixed(3);

    // 1. Tell worker immediately to initialize target architecture
    this.worker.postMessage({
      type: 'INIT_ARCH',
      data: {
        architecture: arch,
        topology: this.topology,
        activation: this.activation,
        learningRate: this.learningRate,
        optimizer: this.optimizer,
        batchSize: this.batchSize,
        targetEpochs: this.targetEpochs
      }
    });

    // 2. Update Dataset dropdown options
    this.updateDatasetDropdownForArch(arch);

    // 3. Load corresponding dataset
    if (arch === 'mlp') {
      const type = (this.currentDatasetType === 'digits' || this.currentDatasetType === 'sequence') ? 'spiral' : this.currentDatasetType;
      this.loadDataset(type);
    } else if (arch === 'cnn') {
      this.loadDataset('digits');
    } else if (arch === 'lstm') {
      this.loadDataset('sequence');
    }

    // 4. Update the bottom Inference panel layout
    this.setupInferencePanelForArch(arch);
  }

  updateDatasetDropdownForArch(arch) {
    const datasetSelect = document.getElementById('select-dataset');
    if (!datasetSelect) return;

    datasetSelect.innerHTML = '';
    if (arch === 'mlp') {
      datasetSelect.innerHTML = `
        <option value="spiral">Spiral (Non-Linear)</option>
        <option value="moons">Two Moons</option>
        <option value="circles">Concentric Circles</option>
        <option value="xor">XOR Quadrants</option>
      `;
      datasetSelect.value = (this.currentDatasetType === 'digits' || this.currentDatasetType === 'sequence') ? 'spiral' : this.currentDatasetType;
    } else if (arch === 'cnn') {
      datasetSelect.innerHTML = `
        <option value="digits">Geometric Shapes (Ring, Bar, Cross)</option>
      `;
      datasetSelect.value = 'digits';
    } else if (arch === 'lstm') {
      datasetSelect.innerHTML = `
        <option value="sequence">Synthetic Temporal Sequences (Sine)</option>
      `;
      datasetSelect.value = 'sequence';
    }
  }

  handle3DHover(hit) {
    const tooltip = document.getElementById('neuron-tooltip');
    if (!tooltip) return;

    if (!hit) {
      tooltip.style.opacity = '0';
      tooltip.style.pointerEvents = 'none';
      return;
    }

    const { type, data, screenX, screenY } = hit;
    let content = '';

    if (type === 'neuron') {
      const layerName = data.isInput ? 'Input Layer' : (data.isOutput ? 'Output (Softmax)' : `Hidden Layer #${data.layerIndex}`);
      content = `
        <div class="tooltip-header"><span class="badge-tag">Neuron</span> ${layerName} [Node #${data.neuronIndex + 1}]</div>
        <div class="tooltip-row"><span>Activation a:</span> <strong class="text-cyan">${(data.activation || 0).toFixed(4)}</strong></div>
        <div class="tooltip-row"><span>Bias b:</span> <strong class="text-magenta">${(data.bias || 0).toFixed(4)}</strong></div>
      `;
    } else if (type === 'synapse') {
      const sign = (data.weight >= 0) ? '+ Positive' : '- Negative';
      const colorClass = data.weight >= 0 ? 'text-cyan' : 'text-magenta';
      content = `
        <div class="tooltip-header"><span class="badge-tag">Synapse</span> W (L${data.fromLayer + 1} #${data.fromNeuron + 1} → L${data.toLayer + 1} #${data.toNeuron + 1})</div>
        <div class="tooltip-row"><span>Weight w:</span> <strong class="${colorClass}">${(data.weight || 0).toFixed(4)}</strong></div>
        <div class="tooltip-row"><span>Polarity:</span> <span class="${colorClass}">${sign}</span></div>
      `;
    } else if (type === 'cnn_pixel' || type === 'cnn_feature' || type === 'cnn_pool') {
      content = `
        <div class="tooltip-header"><span class="badge-tag">CNN Cell</span> ${data.layer}</div>
        <div class="tooltip-row"><span>Coordinate:</span> [R:${data.row}, C:${data.col}]</div>
        <div class="tooltip-row"><span>Activation:</span> <strong class="text-cyan">${(data.value || data.activation || 0).toFixed(3)}</strong></div>
      `;
    } else if (type === 'cnn_output') {
      content = `
        <div class="tooltip-header"><span class="badge-tag">CNN Class</span> ${data.className}</div>
        <div class="tooltip-row"><span>Confidence:</span> <strong class="text-cyan">${Math.round((data.probability || 0.33) * 100)}%</strong></div>
      `;
    } else if (type === 'lstm_gate') {
      content = `
        <div class="tooltip-header"><span class="badge-tag">LSTM Gate</span> ${data.gate}</div>
        <div class="tooltip-row"><span>Activation:</span> ${data.activation}</div>
        <div class="tooltip-formula">${data.formula}</div>
        <div class="tooltip-desc">${data.description}</div>
      `;
    } else if (type === 'lstm_component') {
      content = `
        <div class="tooltip-header"><span class="badge-tag">LSTM Highway</span> ${data.name}</div>
        <div class="tooltip-formula">${data.formula}</div>
        <div class="tooltip-desc">${data.description}</div>
      `;
    }

    if (content) {
      tooltip.innerHTML = content;
      tooltip.style.left = `${screenX + 16}px`;
      tooltip.style.top = `${screenY - 20}px`;
      tooltip.style.opacity = '1';
    }
  }

  toggleTraining() {
    this.isTraining = !this.isTraining;
    this.updatePlayButtonState();

    this.worker.postMessage({
      type: this.isTraining ? 'START_TRAINING' : 'PAUSE_TRAINING'
    });
  }

  updatePlayButtonState() {
    const playBtn = document.getElementById('btn-train-toggle');
    if (!playBtn) return;
    if (this.isTraining) {
      playBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
        <span>Pause</span>
      `;
      playBtn.classList.add('btn-active-train');
    } else {
      playBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
        <span>Train Model</span>
      `;
      playBtn.classList.remove('btn-active-train');
    }
  }

  bindEvents() {
    // Architecture Switcher
    const archButtons = document.querySelectorAll('.btn-arch-tab');
    archButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        archButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.switchArchitecture(btn.dataset.arch);
      });
    });

    // Dataset Select
    const datasetSelect = document.getElementById('select-dataset');
    if (datasetSelect) {
      datasetSelect.addEventListener('change', (e) => {
        this.loadDataset(e.target.value);
      });
    }

    // Play/Pause Training
    const trainBtn = document.getElementById('btn-train-toggle');
    if (trainBtn) {
      trainBtn.addEventListener('click', () => this.toggleTraining());
    }

    // Step 1 Epoch
    const stepBtn = document.getElementById('btn-step-epoch');
    if (stepBtn) {
      stepBtn.addEventListener('click', () => {
        this.isTraining = false;
        this.updatePlayButtonState();
        this.worker.postMessage({ type: 'STEP_EPOCH' });
      });
    }

    // Reset Weights
    const resetBtn = document.getElementById('btn-reset-weights');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        this.isTraining = false;
        this.updatePlayButtonState();
        this.worker.postMessage({ type: 'RESET_MODEL' });
        if (this.metricsChart) this.metricsChart.reset();
        document.getElementById('stat-epoch').textContent = '0';
        document.getElementById('stat-loss').textContent = '0.000';
        document.getElementById('stat-acc').textContent = '0%';
        this.setupInferencePanelForArch(this.currentArch);
      });
    }

    // Surge Signal Pulse Button
    const surgeBtn = document.getElementById('btn-surge-signal');
    if (surgeBtn) {
      surgeBtn.addEventListener('click', () => {
        if (this.mlpView && this.currentArch === 'mlp') {
          this.mlpView.triggerSignalWave('forward');
          setTimeout(() => this.mlpView.triggerSignalWave('backward'), 350);
        } else if (this.cnnView && this.currentArch === 'cnn') {
          this.cnnView.updateActivations({});
        }
      });
    }

    // Explode Layers Slider
    const explodeSlider = document.getElementById('slider-explode');
    if (explodeSlider) {
      explodeSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        if (this.mlpView) this.mlpView.setExplodeFactor(val);
      });
    }

    // Camera Presets
    const cameraBtns = document.querySelectorAll('.btn-cam-preset');
    cameraBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        cameraBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.viewport.setCameraPreset(btn.dataset.preset);
      });
    });

    // Learning Rate Slider
    const lrSlider = document.getElementById('slider-lr');
    const lrVal = document.getElementById('val-lr');
    if (lrSlider) {
      lrSlider.addEventListener('input', (e) => {
        this.learningRate = parseFloat(e.target.value);
        if (lrVal) lrVal.textContent = this.learningRate.toFixed(3);
        this.worker.postMessage({
          type: 'UPDATE_HYPERPARAMS',
          data: { learningRate: this.learningRate }
        });
      });
    }

    // Target Epochs Select
    const targetEpochSelect = document.getElementById('select-target-epochs');
    if (targetEpochSelect) {
      targetEpochSelect.addEventListener('change', (e) => {
        this.targetEpochs = parseInt(e.target.value, 10);
        this.worker.postMessage({
          type: 'UPDATE_HYPERPARAMS',
          data: { targetEpochs: this.targetEpochs }
        });
      });
    }

    // Batch Size Select
    const batchSelect = document.getElementById('select-batch-size');
    if (batchSelect) {
      batchSelect.addEventListener('change', (e) => {
        this.batchSize = parseInt(e.target.value, 10);
        this.worker.postMessage({
          type: 'UPDATE_HYPERPARAMS',
          data: { batchSize: this.batchSize }
        });
      });
    }

    // Optimizer Select
    const optSelect = document.getElementById('select-optimizer');
    if (optSelect) {
      optSelect.addEventListener('change', (e) => {
        this.optimizer = e.target.value;
        this.worker.postMessage({
          type: 'UPDATE_HYPERPARAMS',
          data: { optimizer: this.optimizer }
        });
      });
    }

    // Activation Select
    const actSelect = document.getElementById('select-activation');
    if (actSelect) {
      actSelect.addEventListener('change', (e) => {
        this.activation = e.target.value;
        this.worker.postMessage({
          type: 'INIT_ARCH',
          data: {
            architecture: this.currentArch,
            topology: this.topology,
            activation: this.activation,
            learningRate: this.learningRate,
            optimizer: this.optimizer,
            batchSize: this.batchSize,
            targetEpochs: this.targetEpochs
          }
        });
      });
    }

    // Add / Remove Hidden Layer
    const btnAddLayer = document.getElementById('btn-add-layer');
    const btnRemoveLayer = document.getElementById('btn-remove-layer');

    if (btnAddLayer) {
      btnAddLayer.addEventListener('click', () => {
        if (this.topology.length < 6) {
          this.topology.splice(this.topology.length - 1, 0, 8);
          this.updateMLPTopology();
        }
      });
    }

    if (btnRemoveLayer) {
      btnRemoveLayer.addEventListener('click', () => {
        if (this.topology.length > 3) {
          this.topology.splice(this.topology.length - 2, 1);
          this.updateMLPTopology();
        }
      });
    }
  }

  updateMLPTopology() {
    const topoBadge = document.getElementById('topology-badge');
    if (topoBadge) {
      topoBadge.textContent = this.topology.join(' → ');
    }

    this.mlpView.buildArchitecture(this.topology);
    this.worker.postMessage({
      type: 'INIT_ARCH',
      data: {
        architecture: 'mlp',
        topology: this.topology,
        activation: this.activation,
        learningRate: this.learningRate,
        optimizer: this.optimizer,
        batchSize: this.batchSize,
        targetEpochs: this.targetEpochs
      }
    });

    if (this.metricsChart) this.metricsChart.reset();
  }
}

// Instantiate on DOM load
window.addEventListener('DOMContentLoaded', () => {
  new NeuroForgeApp();
});
