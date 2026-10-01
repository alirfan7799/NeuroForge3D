/**
 * NeuroForge-3D Synthetic Benchmark Datasets
 * Provides mathematical generators for classic non-linear machine learning benchmarks.
 */

export function generateDataset(type = 'spiral', numPoints = 250, noise = 0.15) {
  switch (type) {
    case 'spiral':
      return generateSpiral(numPoints, noise);
    case 'moons':
      return generateMoons(numPoints, noise);
    case 'circles':
      return generateCircles(numPoints, noise);
    case 'xor':
      return generateXOR(numPoints, noise);
    case 'digits':
      return generateDigits(240);
    case 'sequence':
      return generateSequence(200);
    default:
      return generateSpiral(numPoints, noise);
  }
}

// 2-Class Archimedean Spiral
export function generateSpiral(numPoints = 250, noise = 0.12) {
  const pointsPerClass = Math.floor(numPoints / 2);
  const X = [];
  const y = [];

  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < pointsPerClass; i++) {
      const r = (i / pointsPerClass) * 1.8 + 0.15;
      const t = 1.75 * (i / pointsPerClass) * 2 * Math.PI + (c * Math.PI);
      const nx = (Math.random() - 0.5) * noise * 2;
      const ny = (Math.random() - 0.5) * noise * 2;

      const px = r * Math.sin(t) + nx;
      const py = r * Math.cos(t) + ny;

      X.push([px, py]);
      y.push(c);
    }
  }

  return { X, y, type: 'spiral', bounds: [-2.2, 2.2] };
}

// Two Interleaving Half Moons
export function generateMoons(numPoints = 250, noise = 0.1) {
  const pointsPerClass = Math.floor(numPoints / 2);
  const X = [];
  const y = [];

  for (let i = 0; i < pointsPerClass; i++) {
    const angle = (Math.PI * i) / pointsPerClass;
    const nx = (Math.random() - 0.5) * noise * 2;
    const ny = (Math.random() - 0.5) * noise * 2;
    const px = Math.cos(angle) * 1.2 - 0.5 + nx;
    const py = Math.sin(angle) * 1.2 - 0.2 + ny;
    X.push([px, py]);
    y.push(0);
  }

  for (let i = 0; i < pointsPerClass; i++) {
    const angle = (Math.PI * i) / pointsPerClass;
    const nx = (Math.random() - 0.5) * noise * 2;
    const ny = (Math.random() - 0.5) * noise * 2;
    const px = 1 - Math.cos(angle) * 1.2 - 0.5 + nx;
    const py = 1 - Math.sin(angle) * 1.2 - 0.7 + ny;
    X.push([px, py]);
    y.push(1);
  }

  return { X, y, type: 'moons', bounds: [-2.0, 2.0] };
}

// Concentric Circles
export function generateCircles(numPoints = 250, noise = 0.08) {
  const pointsPerClass = Math.floor(numPoints / 2);
  const X = [];
  const y = [];

  for (let i = 0; i < pointsPerClass; i++) {
    const angle = Math.random() * 2 * Math.PI;
    const r = Math.random() * 0.55 + 0.1;
    const nx = (Math.random() - 0.5) * noise;
    const ny = (Math.random() - 0.5) * noise;
    X.push([r * Math.cos(angle) + nx, r * Math.sin(angle) + ny]);
    y.push(0);
  }

  for (let i = 0; i < pointsPerClass; i++) {
    const angle = Math.random() * 2 * Math.PI;
    const r = 1.15 + (Math.random() - 0.5) * 0.35;
    const nx = (Math.random() - 0.5) * noise;
    const ny = (Math.random() - 0.5) * noise;
    X.push([r * Math.cos(angle) + nx, r * Math.sin(angle) + ny]);
    y.push(1);
  }

  return { X, y, type: 'circles', bounds: [-2.0, 2.0] };
}

// XOR Quadrants
export function generateXOR(numPoints = 250, noise = 0.15) {
  const X = [];
  const y = [];

  for (let i = 0; i < numPoints; i++) {
    const px = (Math.random() - 0.5) * 3.0;
    const py = (Math.random() - 0.5) * 3.0;
    const label = (px * py > 0) ? 1 : 0;
    const nx = (Math.random() - 0.5) * noise;
    const ny = (Math.random() - 0.5) * noise;
    X.push([px + nx, py + ny]);
    y.push(label);
  }

  return { X, y, type: 'xor', bounds: [-2.0, 2.0] };
}

/**
 * 8x8 Synthetic Geometric Digit Patterns for CNN
 * Includes realistic spatial jitter, thickness variation, and background noise
 * to demonstrate true translation invariance and smooth convergence.
 */
export function generateDigits(numSamples = 240) {
  const samples = [];

  for (let i = 0; i < numSamples; i++) {
    const label = i % 3; // 0: Ring / Circle, 1: Vertical Bar, 2: Cross / Diagonal
    const grid = new Float32Array(8 * 8);

    // Random spatial shift between -1 and +1 pixels to require CNN translation invariance
    const shiftR = Math.floor(Math.random() * 3) - 1; // -1, 0, or 1
    const shiftC = Math.floor(Math.random() * 3) - 1;

    const setPixel = (r, c, val) => {
      const nr = r + shiftR;
      const nc = c + shiftC;
      if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
        grid[nr * 8 + nc] = Math.max(grid[nr * 8 + nc], val);
      }
    };

    if (label === 0) {
      // Circle / Ring
      const ringCoords = [
        [1, 3], [1, 4],
        [2, 2], [2, 5],
        [3, 1], [3, 6],
        [4, 1], [4, 6],
        [5, 2], [5, 5],
        [6, 3], [6, 4]
      ];
      for (const [r, c] of ringCoords) {
        setPixel(r, c, 0.75 + Math.random() * 0.25);
      }
    } else if (label === 1) {
      // Vertical Bar with variable thickness
      const barCol = 3 + (Math.random() > 0.5 ? 1 : 0);
      for (let r = 1; r < 7; r++) {
        setPixel(r, barCol, 0.8 + Math.random() * 0.2);
        if (Math.random() > 0.4) {
          setPixel(r, barCol + (barCol === 3 ? 1 : -1), 0.55 + Math.random() * 0.25);
        }
      }
    } else {
      // Diagonal Cross (X)
      for (let d = 1; d < 7; d++) {
        setPixel(d, d, 0.75 + Math.random() * 0.25);
        setPixel(d, 7 - d, 0.75 + Math.random() * 0.25);
      }
    }

    // Realistic Gaussian background sensory noise
    for (let j = 0; j < 64; j++) {
      const noise = (Math.random() - 0.5) * 0.18;
      grid[j] = Math.max(0, Math.min(1.0, grid[j] + noise));
    }

    samples.push({ grid: Array.from(grid), label });
  }

  return { samples, type: 'digits', imageSize: 8, numClasses: 3 };
}

/**
 * Synthetic Temporal Sequences for LSTM
 * Multi-harmonic non-stationary sine wave sequence testing temporal dependency learning.
 */
export function generateSequence(numSequences = 200, seqLength = 6) {
  const sequences = [];

  for (let i = 0; i < numSequences; i++) {
    const f1 = 0.35 + Math.random() * 0.4;
    const f2 = 0.7 + Math.random() * 0.5;
    const p1 = Math.random() * Math.PI * 2;
    const p2 = Math.random() * Math.PI * 2;

    const seq = [];
    for (let t = 0; t < seqLength; t++) {
      const val = 0.6 * Math.sin(t * f1 + p1) + 0.4 * Math.cos(t * f2 + p2) + (Math.random() - 0.5) * 0.08;
      seq.push(val);
    }

    const targetVal = 0.6 * Math.sin(seqLength * f1 + p1) + 0.4 * Math.cos(seqLength * f2 + p2);
    sequences.push({ input: seq, target: targetVal });
  }

  return { sequences, type: 'sequence', seqLength };
}
