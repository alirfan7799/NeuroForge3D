/**
 * NeuroForge-3D Live Decision Boundary Canvas
 * Renders real-time classification probability contours and data point scatter,
 * with click-to-probe interactive inference trigger.
 */

export class BoundaryCanvas {
  constructor(canvasElement, onProbePoint) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
    this.onProbePoint = onProbePoint;

    this.dataset = null;
    this.boundaryData = null; // { grid, resolution, bounds }
    this.probeMarker = null;  // [x, y]

    // Off-screen canvas for smooth pixel rendering of boundary
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCtx = this.offscreenCanvas.getContext('2d');

    this.initEvents();
  }

  initEvents() {
    this.canvas.addEventListener('click', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      const bounds = this.boundaryData ? this.boundaryData.bounds : [-2.2, 2.2];
      const normX = clickX / this.canvas.width;
      const normY = clickY / this.canvas.height;

      const worldX = bounds[0] + normX * (bounds[1] - bounds[0]);
      const worldY = bounds[1] - normY * (bounds[1] - bounds[0]); // Inverted Y

      this.probeMarker = [worldX, worldY];
      this.draw();

      if (this.onProbePoint) {
        this.onProbePoint([worldX, worldY]);
      }
    });
  }

  setDataset(dataset) {
    this.dataset = dataset;
    this.draw();
  }

  updateBoundary(boundaryData) {
    this.boundaryData = boundaryData;
    this.draw();
  }

  draw() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    this.ctx.clearRect(0, 0, width, height);

    const bounds = this.boundaryData ? this.boundaryData.bounds : [-2.2, 2.2];

    // 1. Draw Decision Boundary Contour Heatmap
    if (this.boundaryData && this.boundaryData.grid) {
      const res = this.boundaryData.resolution;
      this.offscreenCanvas.width = res;
      this.offscreenCanvas.height = res;

      const imgData = this.offscreenCtx.createImageData(res, res);
      const data = imgData.data;

      for (let i = 0; i < this.boundaryData.grid.length; i++) {
        const prob = this.boundaryData.grid[i]; // 0.0 (Class 0) to 1.0 (Class 1)
        const pIdx = i * 4;

        // Class 0: Deep Violet (#4c1d95 / #db2777), Class 1: Cyber Cyan (#06b6d4 / #00f0ff)
        // Interpolate smoothly
        const r = Math.round((1 - prob) * 200 + prob * 0);
        const g = Math.round((1 - prob) * 30 + prob * 220);
        const b = Math.round((1 - prob) * 150 + prob * 255);

        data[pIdx] = r;
        data[pIdx + 1] = g;
        data[pIdx + 2] = b;
        data[pIdx + 3] = 160; // Semitransparent
      }

      this.offscreenCtx.putImageData(imgData, 0, 0);

      // Smoothly scale offscreen image onto main canvas
      this.ctx.imageSmoothingEnabled = true;
      this.ctx.imageSmoothingQuality = 'high';
      this.ctx.drawImage(this.offscreenCanvas, 0, 0, width, height);

      // Subtle cyber grid overlay
      this.drawGridOverlay(width, height);
    } else {
      // Dark placeholder background
      this.ctx.fillStyle = '#0b0f19';
      this.ctx.fillRect(0, 0, width, height);
      this.drawGridOverlay(width, height);
    }

    // 2. Draw Dataset Scatter Points
    if (this.dataset && this.dataset.X) {
      const spanX = bounds[1] - bounds[0];
      const spanY = bounds[1] - bounds[0];

      for (let i = 0; i < this.dataset.X.length; i++) {
        const [px, py] = this.dataset.X[i];
        const label = this.dataset.y[i];

        const cx = ((px - bounds[0]) / spanX) * width;
        const cy = ((bounds[1] - py) / spanY) * height;

        this.ctx.beginPath();
        this.ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);

        if (label === 0) {
          this.ctx.fillStyle = '#ff2a85';
          this.ctx.strokeStyle = '#ffe4e6';
        } else {
          this.ctx.fillStyle = '#00f0ff';
          this.ctx.strokeStyle = '#cffafe';
        }

        this.ctx.lineWidth = 1;
        this.ctx.fill();
        this.ctx.stroke();
      }
    }

    // 3. Draw Probe Marker if user clicked to probe
    if (this.probeMarker) {
      const spanX = bounds[1] - bounds[0];
      const spanY = bounds[1] - bounds[0];
      const cx = ((this.probeMarker[0] - bounds[0]) / spanX) * width;
      const cy = ((bounds[1] - this.probeMarker[1]) / spanY) * height;

      // Pulsing target crosshair
      this.ctx.beginPath();
      this.ctx.arc(cx, cy, 7, 0, Math.PI * 2);
      this.ctx.strokeStyle = '#ffea00';
      this.ctx.lineWidth = 2;
      this.ctx.stroke();

      this.ctx.beginPath();
      this.ctx.arc(cx, cy, 2, 0, Math.PI * 2);
      this.ctx.fillStyle = '#ffffff';
      this.ctx.fill();
    }
  }

  drawGridOverlay(width, height) {
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    this.ctx.lineWidth = 1;
    const step = width / 6;

    for (let x = step; x < width; x += step) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, height);
      this.ctx.stroke();
    }
    for (let y = step; y < height; y += step) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(width, y);
      this.ctx.stroke();
    }
  }
}
