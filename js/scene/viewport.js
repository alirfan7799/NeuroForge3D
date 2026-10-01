/**
 * NeuroForge-3D Viewport
 * Manages Three.js WebGL scene, lighting, camera controls, raycasting hover detection,
 * background starfield particles, and camera animation presets.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class Viewport {
  constructor(canvasContainer, onHoverEntity) {
    this.container = canvasContainer;
    this.onHoverEntity = onHoverEntity;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2(-1000, -1000);
    this.interactiveObjects = []; // Mesh objects for raycasting

    this.particleField = null;
    this.gridHelper = null;
    this.clock = new THREE.Clock();

    // Camera preset animation target
    this.targetCameraPos = null;
    this.targetControlsTarget = null;

    this.init();
  }

  init() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x070911);
    this.scene.fog = new THREE.FogExp2(0x070911, 0.015);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    this.camera.position.set(0, 12, 28);

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    this.container.appendChild(this.renderer.domElement);

    // 4. Orbit Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.maxDistance = 80;
    this.controls.minDistance = 5;
    this.controls.target.set(0, 0, 0);

    // 5. Lighting Setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x00f0ff, 1.8);
    dirLight1.position.set(15, 20, 20);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xff2a85, 1.4);
    dirLight2.position.set(-15, -10, -15);
    this.scene.add(dirLight2);

    const pointLight = new THREE.PointLight(0x8b5cf6, 2, 40);
    pointLight.position.set(0, 5, 0);
    this.scene.add(pointLight);

    // 6. Cyber Grid Ground
    this.gridHelper = new THREE.GridHelper(50, 40, 0x1f293d, 0x0d1322);
    this.gridHelper.position.y = -8;
    this.scene.add(this.gridHelper);

    // 7. Background Nebula Starfield
    this.createBackgroundParticles();

    // 8. Event Listeners
    window.addEventListener('resize', this.onWindowResize.bind(this));
    this.renderer.domElement.addEventListener('pointermove', this.onPointerMove.bind(this));
    this.renderer.domElement.addEventListener('pointerleave', () => {
      this.mouse.set(-1000, -1000);
      if (this.onHoverEntity) this.onHoverEntity(null);
    });

    // Start render loop
    this.animate();
  }

  createBackgroundParticles() {
    const count = 700;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    const color1 = new THREE.Color(0x00f0ff);
    const color2 = new THREE.Color(0x8b5cf6);

    for (let i = 0; i < count; i++) {
      const idx = i * 3;
      positions[idx] = (Math.random() - 0.5) * 90;
      positions[idx + 1] = (Math.random() - 0.5) * 60;
      positions[idx + 2] = (Math.random() - 0.5) * 90;

      const mixed = color1.clone().lerp(color2, Math.random());
      colors[idx] = mixed.r;
      colors[idx + 1] = mixed.g;
      colors[idx + 2] = mixed.b;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.25,
      vertexColors: true,
      transparent: true,
      opacity: 0.65,
      blending: THREE.AdditiveBlending
    });

    this.particleField = new THREE.Points(geometry, material);
    this.scene.add(this.particleField);
  }

  onWindowResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  onPointerMove(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Check hover
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(this.interactiveObjects, false);

    if (intersects.length > 0) {
      const hit = intersects[0].object;
      if (this.onHoverEntity) {
        this.onHoverEntity({
          type: hit.userData.type,
          data: hit.userData,
          screenX: event.clientX,
          screenY: event.clientY
        });
      }
    } else {
      if (this.onHoverEntity) this.onHoverEntity(null);
    }
  }

  setInteractiveObjects(objects) {
    this.interactiveObjects = objects;
  }

  setCameraPreset(mode) {
    switch (mode) {
      case 'front':
        this.smoothCameraTo(new THREE.Vector3(0, 0, 26), new THREE.Vector3(0, 0, 0));
        break;
      case 'iso':
        this.smoothCameraTo(new THREE.Vector3(18, 14, 22), new THREE.Vector3(0, 0, 0));
        break;
      case 'top':
        this.smoothCameraTo(new THREE.Vector3(0, 30, 0.1), new THREE.Vector3(0, 0, 0));
        break;
      case 'side':
        this.smoothCameraTo(new THREE.Vector3(28, 0, 0), new THREE.Vector3(0, 0, 0));
        break;
      default:
        this.smoothCameraTo(new THREE.Vector3(0, 12, 28), new THREE.Vector3(0, 0, 0));
    }
  }

  smoothCameraTo(pos, target) {
    this.targetCameraPos = pos;
    this.targetControlsTarget = target;
  }

  animate() {
    requestAnimationFrame(this.animate.bind(this));

    const delta = this.clock.getDelta();
    const elapsed = this.clock.getElapsedTime();

    // Rotate starfield slowly
    if (this.particleField) {
      this.particleField.rotation.y = elapsed * 0.02;
      this.particleField.rotation.x = Math.sin(elapsed * 0.01) * 0.05;
    }

    // Camera interpolation if preset requested
    if (this.targetCameraPos) {
      this.camera.position.lerp(this.targetCameraPos, 0.08);
      this.controls.target.lerp(this.targetControlsTarget, 0.08);
      if (this.camera.position.distanceTo(this.targetCameraPos) < 0.1) {
        this.targetCameraPos = null;
        this.targetControlsTarget = null;
      }
    }

    this.controls.update();

    // Notify registered sub-views (MLP, CNN, LSTM)
    if (this.onBeforeRender) {
      this.onBeforeRender(delta, elapsed);
    }

    this.renderer.render(this.scene, this.camera);
  }
}
