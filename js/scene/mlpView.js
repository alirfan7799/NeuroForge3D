/**
 * NeuroForge-3D Multi-Layer Perceptron (MLP) 3D View
 * Renders dense layer nodes, synaptic weights, dynamic forward pulse particles,
 * and backward backpropagation gradient waves in WebGL.
 */

import * as THREE from 'three';

export class MLPView {
  constructor(viewport) {
    this.viewport = viewport;
    this.group = new THREE.Group();
    this.viewport.scene.add(this.group);

    this.topology = [2, 8, 8, 2];
    this.layerSpacing = 7.0;
    this.nodeSpacing = 1.8;
    this.explodeFactor = 1.0;

    // Node & Synapse storage
    this.neuronMeshes = []; // 2D array [layer][neuron]
    this.synapseLines = [];  // Array of synapse line objects
    this.interactiveObjects = [];

    // Signal Particle Pulses
    this.pulseParticles = []; // { mesh, startPos, endPos, progress, speed, direction, color }

    // Color definitions
    this.COLOR_POS_WEIGHT = new THREE.Color(0x00f0ff);
    this.COLOR_NEG_WEIGHT = new THREE.Color(0xff2a85);
    this.COLOR_NEUTRAL = new THREE.Color(0x334155);

    this.buildArchitecture(this.topology);
  }

  buildArchitecture(topology) {
    this.topology = topology;
    this.clear();

    const numLayers = topology.length;
    const startX = -((numLayers - 1) * this.layerSpacing * this.explodeFactor) / 2;

    this.neuronMeshes = [];
    this.interactiveObjects = [];

    // Shared geometries & base materials
    const sphereGeo = new THREE.SphereGeometry(0.55, 32, 32);

    // 1. Create Neuron Spheres
    for (let l = 0; l < numLayers; l++) {
      const layerNodes = [];
      const count = topology[l];
      const x = startX + l * this.layerSpacing * this.explodeFactor;
      const startY = ((count - 1) * this.nodeSpacing) / 2;

      for (let n = 0; n < count; n++) {
        const y = startY - n * this.nodeSpacing;
        const z = (Math.sin(n * 0.8) * 0.4); // Subtle 3D depth wave

        const mat = new THREE.MeshStandardMaterial({
          color: 0x0f172a,
          emissive: 0x00f0ff,
          emissiveIntensity: 0.2,
          roughness: 0.2,
          metalness: 0.8
        });

        const mesh = new THREE.Mesh(sphereGeo, mat);
        mesh.position.set(x, y, z);
        mesh.userData = {
          type: 'neuron',
          layerIndex: l,
          neuronIndex: n,
          totalInLayer: count,
          activation: 0,
          bias: 0,
          isInput: l === 0,
          isOutput: l === numLayers - 1
        };

        // Outer glow halo ring
        const haloGeo = new THREE.RingGeometry(0.65, 0.8, 24);
        const haloMat = new THREE.MeshBasicMaterial({
          color: 0x00f0ff,
          transparent: true,
          opacity: 0.25,
          side: THREE.DoubleSide
        });
        const haloMesh = new THREE.Mesh(haloGeo, haloMat);
        haloMesh.rotation.y = Math.PI / 4;
        mesh.add(haloMesh);

        this.group.add(mesh);
        layerNodes.push(mesh);
        this.interactiveObjects.push(mesh);
      }
      this.neuronMeshes.push(layerNodes);
    }

    // 2. Create Synaptic Connections
    this.buildSynapses();

    // 3. Register interactive objects for raycaster
    this.viewport.setInteractiveObjects(this.interactiveObjects);
  }

  buildSynapses() {
    this.synapseLines = [];
    const numLayers = this.neuronMeshes.length;

    for (let l = 0; l < numLayers - 1; l++) {
      const currLayer = this.neuronMeshes[l];
      const nextLayer = this.neuronMeshes[l + 1];

      for (let i = 0; i < currLayer.length; i++) {
        for (let j = 0; j < nextLayer.length; j++) {
          const n1 = currLayer[i];
          const n2 = nextLayer[j];

          const lineGeo = new THREE.BufferGeometry().setFromPoints([
            n1.position,
            n2.position
          ]);

          const lineMat = new THREE.LineBasicMaterial({
            color: 0x1e293b,
            transparent: true,
            opacity: 0.25,
            linewidth: 1
          });

          const line = new THREE.Line(lineGeo, lineMat);
          line.userData = {
            type: 'synapse',
            fromLayer: l,
            fromNeuron: i,
            toLayer: l + 1,
            toNeuron: j,
            weight: 0.1
          };

          this.group.add(line);
          this.synapseLines.push({
            line,
            from: n1,
            to: n2,
            mat: lineMat,
            userData: line.userData
          });
        }
      }
    }
  }

  updateWeights(weightsData) {
    if (!weightsData || weightsData.length === 0) return;

    weightsData.forEach(layerWeight => {
      const l = layerWeight.layerIndex;
      const kernel = layerWeight.kernel; // [inUnits][outUnits]
      const bias = layerWeight.bias;     // [outUnits]

      // Update biases on target neurons
      if (this.neuronMeshes[l + 1] && bias) {
        for (let j = 0; j < bias.length; j++) {
          if (this.neuronMeshes[l + 1][j]) {
            this.neuronMeshes[l + 1][j].userData.bias = bias[j];
          }
        }
      }

      // Update synapse colors and opacities
      this.synapseLines.forEach(item => {
        if (item.userData.fromLayer === l) {
          const i = item.userData.fromNeuron;
          const j = item.userData.toNeuron;
          if (kernel && kernel[i] && kernel[i][j] !== undefined) {
            const w = kernel[i][j];
            item.userData.weight = w;

            // Positive = Cyan, Negative = Magenta
            const absW = Math.min(1.0, Math.abs(w) / 1.5);
            const targetColor = w >= 0 ? this.COLOR_POS_WEIGHT : this.COLOR_NEG_WEIGHT;
            item.mat.color.copy(targetColor);
            item.mat.opacity = Math.max(0.12, absW * 0.75);
          }
        }
      });
    });
  }

  updateActivations(layerActivations) {
    if (!layerActivations) return;

    layerActivations.forEach(layer => {
      const l = layer.layerIndex;
      const acts = layer.activations;
      if (this.neuronMeshes[l] && acts) {
        for (let i = 0; i < acts.length; i++) {
          const neuron = this.neuronMeshes[l][i];
          if (neuron) {
            const val = acts[i];
            neuron.userData.activation = val;

            // Map activation to glow: 0 = dim slate, 1 = blazing electric cyan
            const clamped = Math.max(0, Math.min(1, val));
            neuron.material.emissiveIntensity = 0.2 + clamped * 1.5;
            neuron.material.emissive.setHSL(0.5 + clamped * 0.05, 1.0, 0.3 + clamped * 0.4);
          }
        }
      }
    });
  }

  triggerSignalWave(direction = 'forward') {
    // Generate pulse energy packets along active synapses
    const isForward = direction === 'forward';
    const numLayers = this.neuronMeshes.length;
    const pulseCount = Math.min(100, this.synapseLines.length);

    // Pick random subset of synapses for dynamic aesthetic
    const shuffled = [...this.synapseLines].sort(() => 0.5 - Math.random());
    const selected = shuffled.slice(0, pulseCount);

    const sphereGeo = new THREE.SphereGeometry(0.12, 12, 12);
    const pulseColor = isForward ? 0x00f0ff : 0xff2a85;

    selected.forEach(syn => {
      const mat = new THREE.MeshBasicMaterial({
        color: pulseColor,
        transparent: true,
        opacity: 0.9
      });
      const mesh = new THREE.Mesh(sphereGeo, mat);

      const start = isForward ? syn.from.position : syn.to.position;
      const end = isForward ? syn.to.position : syn.from.position;

      mesh.position.copy(start);
      this.group.add(mesh);

      // Stagger start time based on layer
      const layerDelay = (isForward ? syn.userData.fromLayer : (numLayers - syn.userData.toLayer)) * 0.15;

      this.pulseParticles.push({
        mesh,
        startPos: start.clone(),
        endPos: end.clone(),
        progress: -layerDelay,
        speed: 1.8 + Math.random() * 0.4,
        mat
      });
    });
  }

  setExplodeFactor(factor) {
    this.explodeFactor = factor;
    const numLayers = this.topology.length;
    const startX = -((numLayers - 1) * this.layerSpacing * this.explodeFactor) / 2;

    for (let l = 0; l < numLayers; l++) {
      const x = startX + l * this.layerSpacing * this.explodeFactor;
      for (let n = 0; n < this.neuronMeshes[l].length; n++) {
        const neuron = this.neuronMeshes[l][n];
        neuron.position.x = x;
      }
    }

    // Refresh synapse line positions
    this.synapseLines.forEach(item => {
      item.line.geometry.setFromPoints([item.from.position, item.to.position]);
    });
  }

  update(delta) {
    // Animate signal pulses
    for (let i = this.pulseParticles.length - 1; i >= 0; i--) {
      const p = this.pulseParticles[i];
      p.progress += delta * p.speed;

      if (p.progress >= 0 && p.progress <= 1) {
        p.mesh.visible = true;
        p.mesh.position.lerpVectors(p.startPos, p.endPos, p.progress);
        p.mat.opacity = 1 - Math.abs(p.progress - 0.5) * 0.8;
      } else if (p.progress < 0) {
        p.mesh.visible = false;
      } else {
        // Pulse finished: cleanup
        this.group.remove(p.mesh);
        p.mesh.geometry.dispose();
        p.mat.dispose();
        this.pulseParticles.splice(i, 1);
      }
    }
  }

  setVisible(visible) {
    this.group.visible = visible;
    if (visible) {
      this.viewport.setInteractiveObjects(this.interactiveObjects);
    }
  }

  clear() {
    while (this.group.children.length > 0) {
      const child = this.group.children[0];
      this.group.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
        else child.material.dispose();
      }
    }
    this.pulseParticles = [];
    this.synapseLines = [];
    this.neuronMeshes = [];
  }
}
