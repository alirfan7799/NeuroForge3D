/**
 * NeuroForge-3D Convolutional Neural Network (CNN) 3D View
 * Visualizes 3D feature map volumes, animated 3x3 scanning receptive field kernel,
 * max-pooling reduction, and dense classification projections in WebGL.
 */

import * as THREE from 'three';

export class CNNView {
  constructor(viewport) {
    this.viewport = viewport;
    this.group = new THREE.Group();
    this.viewport.scene.add(this.group);

    this.interactiveObjects = [];
    this.scanProgress = 0;
    this.kernelMesh = null;
    this.frustumLines = null;

    this.inputPixels = [];
    this.convMaps = [];
    this.poolMaps = [];
    this.denseNodes = [];
    this.outNodes = [];

    this.buildCNN();
    this.setVisible(false);
  }

  buildCNN() {
    this.clear();
    this.interactiveObjects = [];

    const pixelGeo = new THREE.BoxGeometry(0.35, 0.35, 0.1);

    // 1. Input Layer: 8x8 Grid at X = -12
    const inputGroup = new THREE.Group();
    inputGroup.position.set(-12, 0, 0);
    this.inputPixels = [];

    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const mat = new THREE.MeshStandardMaterial({
          color: 0x0a101f,
          emissive: 0x00f0ff,
          emissiveIntensity: 0.1,
          roughness: 0.3
        });
        const mesh = new THREE.Mesh(pixelGeo, mat);
        mesh.position.set((c - 3.5) * 0.45, (3.5 - r) * 0.45, 0);
        mesh.userData = {
          type: 'cnn_pixel',
          layer: 'Input (8x8)',
          row: r,
          col: c,
          value: 0
        };
        inputGroup.add(mesh);
        this.inputPixels.push(mesh);
        this.interactiveObjects.push(mesh);
      }
    }
    this.group.add(inputGroup);

    // 2. Conv2D Feature Maps (4 Filters, 8x8) at X = -5
    const convGroup = new THREE.Group();
    convGroup.position.set(-5, 0, 0);
    this.convMaps = [];

    const filterColors = [0x00f0ff, 0x8b5cf6, 0xff2a85, 0x10b981];

    for (let f = 0; f < 4; f++) {
      const zOffset = (f - 1.5) * 2.8;
      const mapPixels = [];

      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const mat = new THREE.MeshStandardMaterial({
            color: 0x0a101f,
            emissive: filterColors[f],
            emissiveIntensity: 0.25,
            roughness: 0.4
          });
          const mesh = new THREE.Mesh(pixelGeo, mat);
          mesh.position.set((c - 3.5) * 0.45, (3.5 - r) * 0.45, zOffset);
          mesh.userData = {
            type: 'cnn_feature',
            layer: `Conv2D Filter #${f + 1}`,
            filterIdx: f,
            row: r,
            col: c,
            activation: 0
          };
          convGroup.add(mesh);
          mapPixels.push(mesh);
          this.interactiveObjects.push(mesh);
        }
      }
      this.convMaps.push(mapPixels);
    }
    this.group.add(convGroup);

    // 3. MaxPooling Layer (4 Filters, 4x4) at X = 2
    const poolGroup = new THREE.Group();
    poolGroup.position.set(2, 0, 0);
    const poolGeo = new THREE.BoxGeometry(0.7, 0.7, 0.1);
    this.poolMaps = [];

    for (let f = 0; f < 4; f++) {
      const zOffset = (f - 1.5) * 2.8;
      const mapPixels = [];

      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          const mat = new THREE.MeshStandardMaterial({
            color: 0x0e172a,
            emissive: filterColors[f],
            emissiveIntensity: 0.35,
            roughness: 0.4
          });
          const mesh = new THREE.Mesh(poolGeo, mat);
          mesh.position.set((c - 1.5) * 0.85, (1.5 - r) * 0.85, zOffset);
          mesh.userData = {
            type: 'cnn_pool',
            layer: `MaxPool2D (4x4) #${f + 1}`,
            filterIdx: f,
            row: r,
            col: c,
            activation: 0
          };
          poolGroup.add(mesh);
          mapPixels.push(mesh);
          this.interactiveObjects.push(mesh);
        }
      }
      this.poolMaps.push(mapPixels);
    }
    this.group.add(poolGroup);

    // 4. Flattened Dense Layer (column of nodes) at X = 8
    const denseGroup = new THREE.Group();
    denseGroup.position.set(8, 0, 0);
    const denseNodeGeo = new THREE.SphereGeometry(0.2, 16, 16);
    this.denseNodes = [];

    for (let i = 0; i < 16; i++) {
      const mat = new THREE.MeshStandardMaterial({
        color: 0x0a101f,
        emissive: 0x38bdf8,
        emissiveIntensity: 0.4
      });
      const mesh = new THREE.Mesh(denseNodeGeo, mat);
      mesh.position.set(0, (7.5 - i) * 0.55, 0);
      mesh.userData = {
        type: 'cnn_dense',
        layer: 'Dense (Flattened)',
        index: i,
        activation: 0
      };
      denseGroup.add(mesh);
      this.denseNodes.push(mesh);
      this.interactiveObjects.push(mesh);
    }
    this.group.add(denseGroup);

    // 5. Output Classes at X = 14 (3 classes: Ring, Bar, Cross)
    const outGroup = new THREE.Group();
    outGroup.position.set(14, 0, 0);
    const outGeo = new THREE.SphereGeometry(0.55, 24, 24);
    this.outNodes = [];
    const classLabels = ['Ring / Circle', 'Vertical Bar', 'Cross / Diagonal'];

    for (let c = 0; c < 3; c++) {
      const mat = new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        emissive: 0x00f0ff,
        emissiveIntensity: 0.5
      });
      const mesh = new THREE.Mesh(outGeo, mat);
      mesh.position.set(0, (1 - c) * 2.5, 0);
      mesh.userData = {
        type: 'cnn_output',
        className: classLabels[c],
        classIndex: c,
        probability: 0.33
      };
      outGroup.add(mesh);
      this.outNodes.push(mesh);
      this.interactiveObjects.push(mesh);
    }
    this.group.add(outGroup);

    // 6. 3D Animated Scanning Kernel Wireframe Box & Projection Rays
    this.createScanningKernel();
  }

  createScanningKernel() {
    const boxGeo = new THREE.BoxGeometry(0.45 * 3, 0.45 * 3, 0.3);
    const edges = new THREE.EdgesGeometry(boxGeo);
    this.kernelMesh = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0xffea00, linewidth: 2 })
    );
    this.kernelMesh.position.set(-12, 0, 0.2);
    this.group.add(this.kernelMesh);

    const frustumGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(4 * 2 * 3);
    frustumGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.frustumLines = new THREE.LineSegments(
      frustumGeo,
      new THREE.LineBasicMaterial({ color: 0xffea00, transparent: true, opacity: 0.45 })
    );
    this.group.add(this.frustumLines);
  }

  setInputImage(grid64) {
    if (!grid64 || !this.inputPixels) return;
    for (let i = 0; i < 64; i++) {
      const p = this.inputPixels[i];
      if (p) {
        const val = grid64[i] || 0;
        p.userData.value = val;
        p.material.emissiveIntensity = 0.1 + val * 1.8;
      }
    }
  }

  updateActivations(data) {
    if (!data) return;

    // Simulate animated convolution activations across maps during active training
    const time = performance.now() * 0.003;
    for (let f = 0; f < this.convMaps.length; f++) {
      const map = this.convMaps[f];
      for (let i = 0; i < map.length; i++) {
        const act = Math.max(0.1, (Math.sin(time + f + i * 0.2) + 1) * 0.45);
        map[i].userData.activation = act;
        map[i].material.emissiveIntensity = 0.15 + act * 1.2;
      }
    }

    // Dense nodes glow
    for (let i = 0; i < this.denseNodes.length; i++) {
      const act = Math.max(0.1, (Math.cos(time + i * 0.3) + 1) * 0.4);
      this.denseNodes[i].material.emissiveIntensity = 0.2 + act * 1.5;
    }

    // Output probabilities glow
    if (this.outNodes && this.outNodes.length === 3) {
      const winner = Math.floor((time * 0.5) % 3);
      for (let c = 0; c < 3; c++) {
        const isWinner = c === winner;
        this.outNodes[c].material.emissiveIntensity = isWinner ? 1.6 : 0.2;
        this.outNodes[c].userData.probability = isWinner ? 0.85 : 0.075;
      }
    }
  }

  update(delta) {
    if (!this.group.visible) return;

    // Animate kernel scan across 8x8 input grid
    this.scanProgress += delta * 1.8;
    const step = Math.floor(this.scanProgress) % 36;
    const r = Math.floor(step / 6);
    const c = step % 6;

    if (this.kernelMesh) {
      this.kernelMesh.position.set(
        -12 + 0.1,
        (3.5 - (r + 1)) * 0.45,
        ((c + 1) - 3.5) * 0.45
      );
    }

    // Update pyramid rays connecting kernel to Conv map feature slice
    if (this.frustumLines && this.kernelMesh) {
      const posAttr = this.frustumLines.geometry.attributes.position;
      const kpos = this.kernelMesh.position;
      const targetPos = new THREE.Vector3(-5, kpos.y, -1.5 * 2.8);

      const hw = (0.45 * 3) / 2;
      const hh = (0.45 * 3) / 2;
      const corners = [
        new THREE.Vector3(kpos.x, kpos.y + hh, kpos.z - hw),
        new THREE.Vector3(kpos.x, kpos.y + hh, kpos.z + hw),
        new THREE.Vector3(kpos.x, kpos.y - hh, kpos.z + hw),
        new THREE.Vector3(kpos.x, kpos.y - hh, kpos.z - hw)
      ];

      let ptr = 0;
      for (let i = 0; i < 4; i++) {
        posAttr.setXYZ(ptr++, corners[i].x, corners[i].y, corners[i].z);
        posAttr.setXYZ(ptr++, targetPos.x, targetPos.y, targetPos.z);
      }
      posAttr.needsUpdate = true;
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
    this.inputPixels = [];
    this.convMaps = [];
    this.poolMaps = [];
    this.denseNodes = [];
    this.outNodes = [];
  }
}
