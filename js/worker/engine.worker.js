/**
 * NeuroForge-3D Compute Engine Web Worker
 * Runs TensorFlow.js machine learning loops, gradient computations,
 * and decision boundary evaluations completely off the main UI thread.
 */

/* global importScripts, tf */
try {
  importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
} catch (e) {
  console.error('[Worker] Failed to load TensorFlow.js from CDN:', e);
}

// Internal Worker State
let model = null;
let currentArch = 'mlp'; // 'mlp' | 'cnn' | 'lstm'
let mlpTopology = [2, 8, 8, 2];
let activation = 'relu';
let optimizerName = 'adam';
let learningRate = 0.03;
let batchSize = 32;
let targetEpochs = 0; // 0 = continuous

let dataset = null;
let xTensor = null;
let yTensor = null;

let isTraining = false;
let currentEpoch = 0;
let trainingLoopId = null;

// Helper to extract weights snapshot
function getWeightsSnapshot() {
  if (!model) return [];
  const layersWeights = [];
  try {
    model.layers.forEach((layer, idx) => {
      const weights = layer.getWeights();
      if (weights && weights.length >= 1) {
        const kernel = weights[0].arraySync();
        const bias = weights[1] ? weights[1].arraySync() : [];
        layersWeights.push({
          layerIndex: idx,
          name: layer.name,
          kernel,
          bias
        });
      }
    });
  } catch (err) {
    console.warn('[Worker] Could not snapshot weights:', err);
  }
  return layersWeights;
}

// Build MLP Model
function buildMLP(topology, act = 'relu') {
  if (model) {
    try { model.dispose(); } catch (e) {}
  }

  const newModel = tf.sequential();
  
  newModel.add(tf.layers.dense({
    units: topology[1],
    inputShape: [topology[0]],
    activation: act,
    kernelInitializer: 'glorotNormal',
    biasInitializer: 'zeros'
  }));

  for (let i = 2; i < topology.length - 1; i++) {
    newModel.add(tf.layers.dense({
      units: topology[i],
      activation: act,
      kernelInitializer: 'glorotNormal',
      biasInitializer: 'zeros'
    }));
  }

  newModel.add(tf.layers.dense({
    units: topology[topology.length - 1],
    activation: 'softmax',
    kernelInitializer: 'glorotNormal',
    biasInitializer: 'zeros'
  }));

  const opt = createOptimizer(optimizerName, learningRate);
  newModel.compile({
    optimizer: opt,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  model = newModel;
  currentArch = 'mlp';
  currentEpoch = 0;

  postModelSummary();
}

// Build CNN Model for 8x8 synthetic feature patterns
function buildCNN() {
  if (model) {
    try { model.dispose(); } catch (e) {}
  }

  const newModel = tf.sequential();
  newModel.add(tf.layers.conv2d({
    inputShape: [8, 8, 1],
    filters: 4,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  newModel.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  newModel.add(tf.layers.flatten());
  newModel.add(tf.layers.dropout({ rate: 0.25 })); // Standard industry dropout to prevent rapid memorization
  newModel.add(tf.layers.dense({ units: 12, activation: 'relu' }));
  newModel.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

  const opt = createOptimizer(optimizerName, learningRate || 0.002);
  newModel.compile({
    optimizer: opt,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  model = newModel;
  currentArch = 'cnn';
  currentEpoch = 0;

  postModelSummary();
}

// Build LSTM Model for Sequence Prediction
function buildLSTM() {
  if (model) {
    try { model.dispose(); } catch (e) {}
  }

  const newModel = tf.sequential();
  newModel.add(tf.layers.lstm({
    units: 6,
    inputShape: [5, 1],
    returnSequences: false,
    activation: 'tanh',
    recurrentActivation: 'sigmoid'
  }));
  newModel.add(tf.layers.dense({ units: 4, activation: 'relu' }));
  newModel.add(tf.layers.dense({ units: 1 }));

  const opt = createOptimizer(optimizerName, learningRate);
  newModel.compile({
    optimizer: opt,
    loss: 'meanSquaredError',
    metrics: ['mae']
  });

  model = newModel;
  currentArch = 'lstm';
  currentEpoch = 0;

  postModelSummary();
}

function createOptimizer(name, lr) {
  switch (name ? name.toLowerCase() : 'adam') {
    case 'sgd':
      return tf.train.sgd(lr);
    case 'rmsprop':
      return tf.train.rmsprop(lr);
    case 'adam':
    default:
      return tf.train.adam(lr);
  }
}

function postModelSummary() {
  if (!model) return;
  const layers = model.layers.map(l => ({
    name: l.name,
    outputShape: l.outputShape,
    trainable: l.trainable
  }));
  const totalParams = model.countParams();

  self.postMessage({
    type: 'MODEL_READY',
    data: {
      architecture: currentArch,
      topology: currentArch === 'mlp' ? mlpTopology : null,
      layers,
      totalParams,
      weights: getWeightsSnapshot()
    }
  });
}

// Universal dataset tensor constructor with zero rank-mismatch
function setDataset(data) {
  if (!data) return;

  if (xTensor) {
    try { xTensor.dispose(); } catch (e) {}
    xTensor = null;
  }
  if (yTensor) {
    try { yTensor.dispose(); } catch (e) {}
    yTensor = null;
  }

  dataset = data;

  try {
    if (data.samples || currentArch === 'cnn' || data.type === 'digits') {
      // 8x8 CNN samples
      const samples = data.samples || [];
      if (samples.length > 0) {
        const X_arr = samples.map(s => s.grid);
        const y_arr = samples.map(s => s.label);
        // Reshape cleanly from 2D [N, 64] -> 4D [N, 8, 8, 1]
        xTensor = tf.tensor2d(X_arr).reshape([X_arr.length, 8, 8, 1]);
        yTensor = tf.oneHot(tf.tensor1d(y_arr, 'int32'), 3);
      }
    } else if (data.sequences || currentArch === 'lstm' || data.type === 'sequence') {
      // LSTM temporal sequences
      const seqs = data.sequences || [];
      if (seqs.length > 0) {
        const X_flat = seqs.map(s => s.input.slice(0, 5));
        const y_flat = seqs.map(s => [s.target]);
        // Reshape cleanly from 2D [N, 5] -> 3D [N, 5, 1]
        xTensor = tf.tensor2d(X_flat).reshape([X_flat.length, 5, 1]);
        yTensor = tf.tensor2d(y_flat);
      }
    } else if (data.X && data.y) {
      // MLP 2D points
      xTensor = tf.tensor2d(data.X);
      yTensor = tf.oneHot(tf.tensor1d(data.y, 'int32'), 2);
    }
  } catch (err) {
    console.error('[Worker] Failed to construct tensors from dataset:', err);
  }
}

// Universal rank-safe slice along the first dimension (batch)
function sliceFirstDim(tensor, batchCount) {
  const b = Math.min(batchCount, tensor.shape[0]);
  const start = new Array(tensor.rank).fill(0);
  const size = [...tensor.shape];
  size[0] = b;
  return tensor.slice(start, size);
}

// Perform 1 Training Epoch and calculate Gradients
async function trainOneEpoch() {
  if (!model || !xTensor || !yTensor) return;

  try {
    currentEpoch++;

    const history = await model.fit(xTensor, yTensor, {
      epochs: 1,
      batchSize: batchSize,
      shuffle: true,
      validationSplit: currentArch === 'cnn' ? 0.25 : 0,
      verbose: 0
    });

    const loss = history.history.val_loss ? history.history.val_loss[0] : (history.history.loss ? history.history.loss[0] : 0.05);
    let acc = 0;
    if (history.history.val_accuracy) {
      acc = history.history.val_accuracy[0];
    } else if (history.history.val_acc) {
      acc = history.history.val_acc[0];
    } else if (history.history.acc) {
      acc = history.history.acc[0];
    } else if (history.history.accuracy) {
      acc = history.history.accuracy[0];
    } else if (history.history.mae) {
      acc = Math.max(0, 1 - Math.min(1, history.history.mae[0]));
    } else {
      acc = Math.max(0, 1 - Math.min(1, loss));
    }

    // Compute live gradients
    const gradientInfo = computeGradientMagnitudes();

    // Periodic decision boundary computation for MLP
    let boundary = null;
    if (currentArch === 'mlp' && (currentEpoch % 2 === 0 || !isTraining)) {
      boundary = computeDecisionBoundaryGrid(32);
    }

    // Compute live sample prediction for HUD readout
    let samplePred = null;
    if (currentArch === 'cnn') {
      const firstSample = xTensor.slice([0, 0, 0, 0], [1, 8, 8, 1]);
      const pred = model.predict(firstSample);
      samplePred = pred.arraySync()[0];
      firstSample.dispose();
      pred.dispose();
    } else if (currentArch === 'lstm') {
      const firstSample = xTensor.slice([0, 0, 0], [1, 5, 1]);
      const pred = model.predict(firstSample);
      samplePred = pred.arraySync()[0][0];
      firstSample.dispose();
      pred.dispose();
    }

    self.postMessage({
      type: 'EPOCH_UPDATE',
      data: {
        architecture: currentArch,
        epoch: currentEpoch,
        loss: parseFloat(loss.toFixed(5)),
        accuracy: parseFloat(acc.toFixed(4)),
        gradientMagnitudes: gradientInfo,
        weights: getWeightsSnapshot(),
        boundary: boundary,
        samplePred: samplePred
      }
    });

    // Check if target epoch reached
    if (targetEpochs > 0 && currentEpoch >= targetEpochs) {
      isTraining = false;
      if (trainingLoopId) clearTimeout(trainingLoopId);
      self.postMessage({ type: 'TARGET_REACHED', data: { epoch: currentEpoch } });
    }
  } catch (err) {
    console.error('[Worker] Error during trainOneEpoch:', err);
  }
}

// Compute layer-by-layer gradient magnitudes safely for any tensor rank
function computeGradientMagnitudes() {
  try {
    if (!model || !xTensor || !yTensor) return [];

    const bCount = Math.min(16, xTensor.shape[0]);
    const batchX = sliceFirstDim(xTensor, bCount);
    const batchY = sliceFirstDim(yTensor, bCount);

    const { grads } = tf.variableGrads(() => {
      const preds = model.predict(batchX);
      if (currentArch === 'lstm') {
        return tf.losses.meanSquaredError(batchY, preds);
      } else {
        return tf.losses.categoricalCrossentropy(batchY, preds);
      }
    });

    batchX.dispose();
    batchY.dispose();

    const layerGradients = [];
    for (const [varName, gradTensor] of Object.entries(grads)) {
      const norm = gradTensor.norm().arraySync();
      layerGradients.push({
        variable: varName,
        magnitude: parseFloat(norm.toFixed(5))
      });
      gradTensor.dispose();
    }
    return layerGradients;
  } catch (err) {
    return [];
  }
}

// Compute 2D Decision Boundary Grid for classification visualization
function computeDecisionBoundaryGrid(resolution = 32) {
  if (!model || currentArch !== 'mlp') return null;

  try {
    const min = -2.2;
    const max = 2.2;
    const step = (max - min) / (resolution - 1);
    const gridPoints = [];

    for (let r = 0; r < resolution; r++) {
      const y = max - r * step;
      for (let c = 0; c < resolution; c++) {
        const x = min + c * step;
        gridPoints.push([x, y]);
      }
    }

    const gridTensor = tf.tensor2d(gridPoints);
    const predictions = model.predict(gridTensor);
    const probs = predictions.arraySync();

    gridTensor.dispose();
    predictions.dispose();

    const boundaryGrid = probs.map(p => p[1]);

    return {
      grid: boundaryGrid,
      resolution: resolution,
      bounds: [min, max]
    };
  } catch (err) {
    return null;
  }
}

// Run single sample inference
function runInference(inputSample) {
  if (!model) return;

  tf.tidy(() => {
    try {
      let inputTensor;
      if (currentArch === 'mlp') {
        inputTensor = tf.tensor2d([inputSample]);
      } else if (currentArch === 'cnn') {
        inputTensor = tf.tensor2d([inputSample]).reshape([1, 8, 8, 1]);
      } else if (currentArch === 'lstm') {
        inputTensor = tf.tensor2d([inputSample]).reshape([1, 5, 1]);
      }

      const layerActivations = [];
      let currentActivation = inputTensor;

      layerActivations.push({
        layerIndex: 0,
        name: 'Input Layer',
        activations: inputTensor.arraySync()[0]
      });

      for (let i = 0; i < model.layers.length; i++) {
        const layer = model.layers[i];
        currentActivation = layer.apply(currentActivation);
        const acts = currentActivation.arraySync()[0];
        layerActivations.push({
          layerIndex: i + 1,
          name: layer.name,
          activations: acts
        });
      }

      const finalPred = layerActivations[layerActivations.length - 1].activations;

      self.postMessage({
        type: 'INFERENCE_RESULT',
        data: {
          input: inputSample,
          layerActivations: layerActivations,
          prediction: finalPred
        }
      });
    } catch (err) {
      console.error('[Worker] Inference error:', err);
    }
  });
}

// Continuous training loop
async function runTrainingLoop() {
  if (!isTraining) return;
  await trainOneEpoch();
  if (isTraining) {
    trainingLoopId = setTimeout(runTrainingLoop, 20);
  }
}

// Worker Message Dispatcher
self.onmessage = async function (e) {
  const { type, data } = e.data;

  switch (type) {
    case 'INIT_ARCH':
      currentArch = data.architecture || 'mlp';
      learningRate = data.learningRate || 0.03;
      optimizerName = data.optimizer || 'adam';
      batchSize = data.batchSize || 32;
      targetEpochs = data.targetEpochs !== undefined ? data.targetEpochs : targetEpochs;

      if (currentArch === 'mlp') {
        mlpTopology = data.topology || [2, 8, 8, 2];
        activation = data.activation || 'relu';
        buildMLP(mlpTopology, activation);
      } else if (currentArch === 'cnn') {
        buildCNN();
      } else if (currentArch === 'lstm') {
        buildLSTM();
      }

      if (dataset) {
        setDataset(dataset);
      }
      break;

    case 'SET_DATASET':
      setDataset(data);
      if (currentArch === 'mlp') {
        const boundary = computeDecisionBoundaryGrid(32);
        self.postMessage({
          type: 'BOUNDARY_UPDATE',
          data: boundary
        });
      }
      break;

    case 'UPDATE_HYPERPARAMS':
      if (data.learningRate) learningRate = data.learningRate;
      if (data.optimizer) optimizerName = data.optimizer;
      if (data.batchSize) batchSize = data.batchSize;
      if (data.targetEpochs !== undefined) targetEpochs = data.targetEpochs;

      if (model) {
        const opt = createOptimizer(optimizerName, learningRate);
        const lossFn = currentArch === 'lstm' ? 'meanSquaredError' : 'categoricalCrossentropy';
        const metricName = currentArch === 'lstm' ? ['mae'] : ['accuracy'];
        model.compile({
          optimizer: opt,
          loss: lossFn,
          metrics: metricName
        });
      }
      break;

    case 'START_TRAINING':
      if (!isTraining) {
        isTraining = true;
        runTrainingLoop();
      }
      break;

    case 'PAUSE_TRAINING':
      isTraining = false;
      if (trainingLoopId) clearTimeout(trainingLoopId);
      break;

    case 'STEP_EPOCH':
      isTraining = false;
      if (trainingLoopId) clearTimeout(trainingLoopId);
      await trainOneEpoch();
      break;

    case 'RESET_MODEL':
      isTraining = false;
      if (trainingLoopId) clearTimeout(trainingLoopId);
      currentEpoch = 0;
      if (currentArch === 'mlp') {
        buildMLP(mlpTopology, activation);
      } else if (currentArch === 'cnn') {
        buildCNN();
      } else if (currentArch === 'lstm') {
        buildLSTM();
      }
      if (dataset) {
        setDataset(dataset);
      }
      break;

    case 'RUN_INFERENCE':
      runInference(data.input);
      break;

    default:
      console.warn('[Worker] Unknown message type:', type);
  }
};
