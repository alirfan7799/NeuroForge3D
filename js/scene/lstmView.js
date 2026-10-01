/**
 * NeuroForge-3D Long Short-Term Memory (LSTM) 3D View
 * Visualizes the 3D Recurrent Memory Cell: Cell State Highway, Forget Gate,
 * Input Gate, Candidate State, Output Gate, and temporal recurrence flows.
 */

import * as THREE from 'three';

export class LSTMView {
  constructor(viewport) {
    this.viewport = viewport;
    this.group = new THREE.Group();
    this.viewport.scene.add(this.group);

    this.interactiveObjects = [];
    this.circulatingParticles = [];
    this.currentTimeStep = 1; // 0: t-1, 1: t, 2: t+1

    this.buildLSTMCell();
    this.setVisible(false);
  }

  buildLSTMCell() {
    this.clear();
    this.interactiveObjects = [];

    // Base materials and geometries
    const gateGeo = new THREE.CylinderGeometry(0.8, 0.8, 0.4, 24);
    const opGeo = new THREE.SphereGeometry(0.5, 20, 20);

    // Color definitions for LSTM Components
    const COLOR_FORGET = 0xf59e0b;    // Amber
    const COLOR_INPUT = 0x10b981;     // Emerald
    const COLOR_CANDIDATE = 0x00f0ff; // Cyan
    const COLOR_OUTPUT = 0xa855f7;    // Purple
    const COLOR_HIGHWAY = 0x38bdf8;   // Sky blue

    // 1. Top Cell State Highway (Ct-1 -> Ct)
    const cellHighwayGeo = new THREE.CylinderGeometry(0.18, 0.18, 24, 16);
    cellHighwayGeo.rotateZ(Math.PI / 2);
    const cellHighwayMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7,
      emissive: COLOR_HIGHWAY,
      emissiveIntensity: 0.6,
      metalness: 0.8,
      roughness: 0.2
    });
    const cellHighway = new THREE.Mesh(cellHighwayGeo, cellHighwayMat);
    cellHighway.position.set(0, 4.5, 0);
    cellHighway.userData = {
      type: 'lstm_component',
      name: 'Cell State Highway (C_t)',
      formula: 'C_t = f_t ⊙ C_{t-1} + i_t ⊙ C~_t',
      description: 'Long-term memory conduit running uninterrupted through time.'
    };
    this.group.add(cellHighway);
    this.interactiveObjects.push(cellHighway);

    // 2. Bottom Hidden State Highway (ht-1 -> ht)
    const hiddenHighwayGeo = new THREE.CylinderGeometry(0.18, 0.18, 24, 16);
    hiddenHighwayGeo.rotateZ(Math.PI / 2);
    const hiddenHighwayMat = new THREE.MeshStandardMaterial({
      color: 0x4f46e5,
      emissive: 0x6366f1,
      emissiveIntensity: 0.5,
      metalness: 0.8,
      roughness: 0.2
    });
    const hiddenHighway = new THREE.Mesh(hiddenHighwayGeo, hiddenHighwayMat);
    hiddenHighway.position.set(0, -4.5, 0);
    hiddenHighway.userData = {
      type: 'lstm_component',
      name: 'Hidden State Highway (h_t)',
      formula: 'h_t = o_t ⊙ tanh(C_t)',
      description: 'Short-term working memory emitted to the output layer.'
    };
    this.group.add(hiddenHighway);
    this.interactiveObjects.push(hiddenHighway);

    // 3. Gate Modules:
    // (a) Forget Gate [f_t] at X = -6, Y = 0
    const forgetGateMat = new THREE.MeshStandardMaterial({
      color: 0x78350f,
      emissive: COLOR_FORGET,
      emissiveIntensity: 0.7
    });
    const forgetMesh = new THREE.Mesh(gateGeo, forgetGateMat);
    forgetMesh.position.set(-6, 0, 0);
    forgetMesh.userData = {
      type: 'lstm_gate',
      gate: 'Forget Gate (f_t)',
      activation: 'σ (Sigmoid)',
      formula: 'f_t = σ(W_f · [h_{t-1}, x_t] + b_f)',
      description: 'Decides what percentage of previous cell memory C_{t-1} to forget or retain.'
    };
    this.group.add(forgetMesh);
    this.interactiveObjects.push(forgetMesh);

    // (b) Input Gate [i_t] at X = -2, Y = 0
    const inputGateMat = new THREE.MeshStandardMaterial({
      color: 0x064e3b,
      emissive: COLOR_INPUT,
      emissiveIntensity: 0.7
    });
    const inputMesh = new THREE.Mesh(gateGeo, inputGateMat);
    inputMesh.position.set(-2, 0, 0);
    inputMesh.userData = {
      type: 'lstm_gate',
      gate: 'Input Gate (i_t)',
      activation: 'σ (Sigmoid)',
      formula: 'i_t = σ(W_i · [h_{t-1}, x_t] + b_i)',
      description: 'Determines which candidate updates should be committed to cell state.'
    };
    this.group.add(inputMesh);
    this.interactiveObjects.push(inputMesh);

    // (c) Candidate State [C~_t] at X = 2, Y = 0
    const candGateMat = new THREE.MeshStandardMaterial({
      color: 0x083344,
      emissive: COLOR_CANDIDATE,
      emissiveIntensity: 0.7
    });
    const candMesh = new THREE.Mesh(gateGeo, candGateMat);
    candMesh.position.set(2, 0, 0);
    candMesh.userData = {
      type: 'lstm_gate',
      gate: 'Candidate State (~C_t)',
      activation: 'tanh',
      formula: '~C_t = tanh(W_c · [h_{t-1}, x_t] + b_c)',
      description: 'Creates a vector of new candidate values that could be added to the state.'
    };
    this.group.add(candMesh);
    this.interactiveObjects.push(candMesh);

    // (d) Output Gate [o_t] at X = 6, Y = 0
    const outGateMat = new THREE.MeshStandardMaterial({
      color: 0x3b0764,
      emissive: COLOR_OUTPUT,
      emissiveIntensity: 0.7
    });
    const outMesh = new THREE.Mesh(gateGeo, outGateMat);
    outMesh.position.set(6, 0, 0);
    outMesh.userData = {
      type: 'lstm_gate',
      gate: 'Output Gate (o_t)',
      activation: 'σ (Sigmoid)',
      formula: 'o_t = σ(W_o · [h_{t-1}, x_t] + b_o)',
      description: 'Filters the current cell state through tanh to form the new hidden state h_t.'
    };
    this.group.add(outMesh);
    this.interactiveObjects.push(outMesh);

    // 4. Pointwise Operator Spheres on Highway
    // Multiply junction at X = -6, Y = 4.5 (Forget * Ct-1)
    const multMat = new THREE.MeshStandardMaterial({ color: 0x334155, emissive: 0xffea00, emissiveIntensity: 0.4 });
    const mult1 = new THREE.Mesh(opGeo, multMat);
    mult1.position.set(-6, 4.5, 0);
    mult1.userData = {
      type: 'lstm_op',
      name: 'Pointwise Multiply (⊗)',
      description: 'Multiplies old cell state by forget gate: f_t ⊙ C_{t-1}'
    };
    this.group.add(mult1);
    this.interactiveObjects.push(mult1);

    // Add junction at X = 0, Y = 4.5 (Old + New Candidate)
    const addMat = new THREE.MeshStandardMaterial({ color: 0x334155, emissive: 0x10b981, emissiveIntensity: 0.4 });
    const add1 = new THREE.Mesh(opGeo, addMat);
    add1.position.set(0, 4.5, 0);
    add1.userData = {
      type: 'lstm_op',
      name: 'Pointwise Add (⊕)',
      description: 'Adds candidate memory scaled by input gate: + (i_t ⊙ ~C_t)'
    };
    this.group.add(add1);
    this.interactiveObjects.push(add1);

    // 5. Connective Conduit Lines (vertical conduits from gates to highways)
    this.createConduit([-6, -4.5, 0], [-6, 0, 0], COLOR_FORGET);
    this.createConduit([-6, 0, 0], [-6, 4.5, 0], COLOR_FORGET);

    this.createConduit([-2, -4.5, 0], [-2, 0, 0], COLOR_INPUT);
    this.createConduit([2, -4.5, 0], [2, 0, 0], COLOR_CANDIDATE);
    this.createConduit([0, 0, 0], [0, 4.5, 0], COLOR_CANDIDATE);

    this.createConduit([6, -4.5, 0], [6, 0, 0], COLOR_OUTPUT);
    this.createConduit([6, 4.5, 0], [6, 0, 0], COLOR_OUTPUT);

    // 6. Circulating energy particles along memory loops
    this.initCirculatingParticles();
  }

  createConduit(start, end, colorHex) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...start),
      new THREE.Vector3(...end)
    ]);
    const mat = new THREE.LineBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.45,
      linewidth: 2
    });
    const line = new THREE.Line(geo, mat);
    this.group.add(line);
  }

  initCirculatingParticles() {
    this.circulatingParticles = [];
    const count = 30;
    const sphereGeo = new THREE.SphereGeometry(0.12, 12, 12);
    const mat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });

    for (let i = 0; i < count; i++) {
      const mesh = new THREE.Mesh(sphereGeo, mat);
      this.group.add(mesh);
      this.circulatingParticles.push({
        mesh,
        pathType: i % 2 === 0 ? 'cellHighway' : 'hiddenHighway',
        progress: Math.random(),
        speed: 0.3 + Math.random() * 0.2
      });
    }
  }

  update(delta) {
    if (!this.group.visible) return;

    // Move memory highway flow particles
    this.circulatingParticles.forEach(p => {
      p.progress = (p.progress + delta * p.speed) % 1.0;
      const x = (p.progress - 0.5) * 24; // -12 to +12
      const y = p.pathType === 'cellHighway' ? 4.5 : -4.5;
      const z = Math.sin(p.progress * Math.PI * 4) * 0.2;
      p.mesh.position.set(x, y, z);
    });
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
    this.circulatingParticles = [];
  }
}
