# NeuroForge-3D — Interactive 3D Neural Architecture Simulator

<p align="center">
  <img src="https://img.shields.io/badge/Three.js-WebGL-00f0ff?style=for-the-badge&logo=three.js&logoColor=black" alt="Three.js" />
  <img src="https://img.shields.io/badge/TensorFlow.js-4.x-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow.js" />
  <img src="https://img.shields.io/badge/Web_Workers-Multithreaded-8b5cf6?style=for-the-badge&logo=w3c&logoColor=white" alt="Web Workers" />
  <img src="https://img.shields.io/badge/Chart.js-Telemetry-f43f5e?style=for-the-badge&logo=chartdotjs&logoColor=white" alt="Chart.js" />
  <img src="https://img.shields.io/badge/Performance-60_FPS_Zero--Lag-10b981?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Performance" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License MIT" />
</p>

---

## 🌟 Overview

**NeuroForge-3D** is an industry-grade, browser-based **3D neural network workbench** that transforms deep learning from an opaque "black box" into an interactive, living WebGL simulation.

Powered by **Three.js**, **TensorFlow.js**, **Web Workers**, and **Chart.js**, NeuroForge-3D visualizes forward and backward passes, weight and bias adjustments, receptive field convolutions, and recurrent temporal gates in real-time 3D space—maintaining an uninterrupted **60+ FPS** rendering loop by offloading all tensor matrix calculus to dedicated background threads.

---

## 🚀 Key Engineering Highlights

### 1. Asynchronous Web Worker Compute Pipeline (Zero UI Freezing)
In standard browser ML applications, running iterative backpropagation on the main thread locks up the UI and drops frames. NeuroForge-3D decouples rendering from computation:
- **Background Worker (`engine.worker.js`):** Executes mini-batch training (`model.fit()`), tensor gradient tracking (`tf.variableGrads`), validation evaluation, and $32 \times 32$ decision boundary grid inference.
- **Main Thread (`viewport.js` & `app.js`):** Focuses solely on 60 FPS Three.js WebGL scene graph rendering, raycasting hover detection, and Chart.js telemetry updates.

### 2. Multi-Architecture Deep Learning Support
- 🌐 **Multi-Layer Perceptron (MLP):** Configurable fully connected layers. Visualizes forward signal surges in **Electric Cyan** and backpropagation gradient waves in **Neon Magenta**. Synapses dynamically reflect weight values through sign color-coding and opacity scaling ($|w|$).
- 📦 **Convolutional Neural Network (CNN):** 3D feature map volumes showing 2D input convolution, an animated $3 \times 3$ scanning receptive-field kernel prism, Max-Pooling spatial reduction, 25% Dropout regularization, and dense projections.
- 🔁 **Long Short-Term Memory (LSTM):** 3D Recurrent Memory Cell detailing the **Cell State Highway ($C_t$)**, **Hidden State Highway ($h_t$)**, and the 4 fundamental gating units:
  - **Forget Gate ($f_t = \sigma(W_f \cdot [h, x] + b_f)$)**
  - **Input Gate ($i_t = \sigma(W_i \cdot [h, x] + b_i)$)**
  - **Candidate State ($\tilde{C}_t = \tanh(W_c \cdot [h, x] + b_c)$)**
  - **Output Gate ($o_t = \sigma(W_o \cdot [h, x] + b_o)$)**

### 3. Live 2D Decision Boundary & Click-to-Probe Inference
- Real-time contour surface updated as training progresses across classic machine learning benchmarks (**Archimedean Spiral**, **Two Moons**, **Concentric Circles**, and **XOR Quadrants**).
- **Click-to-Probe:** Click anywhere on the 2D canvas to drop an interactive targeting reticle; the network immediately runs single-sample inference, surges the signal through the 3D network, and outputs softmax class confidence percentages.

### 4. Mathematical Gradient Diagnostics
- Tracks layer-by-layer Euclidean gradient norms ($\|\partial L / \partial W\|$) in real time via `tf.variableGrads` to visibly demonstrate detection of **vanishing** and **exploding** gradients.

---

## 🏗️ Architecture & Project Structure

```
NeuroForge3D/
├── index.html              # Main application shell with WebGL canvas, HUD, & controls
├── README.md               # Flagship documentation & architecture guide
├── .gitignore              # Standard version control exclusions
├── serve.py                # Lightweight local development server with CORS/MIME handling
├── css/
│   └── style.css           # Cyber-scientific glassmorphic dark theme & layout system
└── js/
    ├── app.js              # Central application coordinator & state bus
    ├── charts/
    │   ├── boundaryCanvas.js # Live 2D decision boundary contour & probe reticle
    │   └── metricsChart.js   # Dual-axis real-time Chart.js loss & accuracy telemetry
    ├── data/
    │   └── datasets.js     # Synthetic benchmark generators (Spiral, Moons, Circles, Digits)
    ├── scene/
    │   ├── viewport.js     # Three.js scene, camera presets, raycaster, starfield
    │   ├── mlpView.js      # 3D Multi-Layer Perceptron mesh, nodes, synapses, wave pulses
    │   ├── cnnView.js      # 3D CNN feature map slices, scanning kernel box, pooling
    │   └── lstmView.js     # 3D LSTM recurrent cell gates & memory highway conduits
    └── worker/
        └── engine.worker.js # TensorFlow.js worker: non-blocking training, gradients, inference
```

---

## ⚡ Quick Start

### Option A: Local Development (Python 3)
NeuroForge-3D is built using standard ES modules and CDNs with zero npm build configuration required:

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/NeuroForge-3D.git
cd NeuroForge-3D

# 2. Launch local server
python serve.py

# 3. Open in browser
# Navigate to http://localhost:8080/
```

### Option B: One-Click Deployment (GitHub Pages)
1. Push this repository to GitHub.
2. Go to **Settings** $\to$ **Pages**.
3. Under **Branch**, select `main` and `/ (root)`, then click **Save**.
4. Your live 3D simulator is instantly available online to recruiters!

---

## 🛠️ Tech Stack & Dependencies

| Layer | Technologies Used | Description |
| :--- | :--- | :--- |
| **3D Graphics Engine** | [Three.js](https://threejs.org/) (r160), WebGL | 3D scene graphs, custom shader materials, particle systems, raycasting |
| **Deep Learning Engine** | [TensorFlow.js](https://www.tensorflow.org/js) (v4.22) | Layer models, backpropagation calculus, gradient tapes, tensor operations |
| **Concurrency** | HTML5 Web Workers API | Off-thread asynchronous matrix computations (Zero main-thread blocking) |
| **Data Telemetry** | [Chart.js](https://www.chartjs.org/) (v4.4), Canvas API | Dual-axis real-time convergence charts & 2D decision contour rasterization |
| **Styling & UI/UX** | Modern Vanilla CSS3, Glassmorphism | Custom design tokens, backdrop filters, responsive flex/grid layouts |

---

## 💼 Resume / CV Bullet Points

Feel free to adapt these bullet points for your resume or LinkedIn portfolio:

> **NeuroForge-3D — Interactive 3D Neural Architecture Simulator**  
> *Tech Stack: Three.js (WebGL), TensorFlow.js, Web Workers, Chart.js, Vanilla ES6+*  
> • Engineered a browser-based 3D neural network visualizer modeling forward/backward propagation and weight/bias gradients across MLP, CNN, and LSTM architectures.  
> • Implemented an asynchronous Web Worker computing pipeline for TensorFlow.js training, maintaining a rock-solid 60+ FPS Three.js rendering loop without main-thread blocking.  
> • Developed a real-time analytics suite featuring live 2D decision boundary contour mapping and synchronized Chart.js loss/accuracy convergence telemetry.  
> • Integrated real-time gradient tracking ($\|\partial L / \partial W\|$) via `tf.variableGrads` to dynamically visualize and detect vanishing/exploding gradients across deep layers.

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute for educational and commercial purposes.
