/**
 * Cornell scenes for pt-webgpu gap-closure captures (mirrors cornell-box scenario tweaks).
 */
import * as THREE from 'three';
import { VITRUM_USER_DATA_KEYS as K } from '@vitrum/three-bindings';

export type GapClosureCornellScenarioId =
  | 'ptwgpu-parity-material-fields'
  | 'rfe03-layered-front-back'
  | 'rfe07-11-sss-mixed-panels'
  | 'rfe08-13-spectral-payload'
  | 'rfe14-thinfilm-angle-shift'
  | 'rfe09-bridge-global-cmf'
  | 'cornell-box'
  | 'cornell-parity';

function applyScenarioMaterialTweaks(
  material: THREE.MeshPhysicalMaterial,
  scenarioId: string,
): void {
  if (scenarioId.includes('caustic')) {
    material.transmission = 0.5;
    material.ior = 1.5;
    material.thickness = 0.25;
  }

  if (
    scenarioId.includes('spectral') ||
    scenarioId.includes('thinfilm') ||
    scenarioId.includes('cmf')
  ) {
    material.transmission = 0.75;
    material.ior = 1.52;
    material.thickness = 0.4;
    material.attenuationDistance = 1.5;
    material.attenuationColor.setRGB(0.72, 0.9, 1.0);
    material.userData[K.SPECTRAL_ATTEN] = {
      wavelengthStart: 380,
      wavelengthEnd: 780,
      values: new Float32Array([
        0.08, 0.1, 0.12, 0.15, 0.2, 0.28, 0.36, 0.44, 0.52, 0.58, 0.64, 0.68, 0.7, 0.68,
        0.62, 0.54, 0.46, 0.38, 0.31, 0.25, 0.2, 0.16, 0.13, 0.11, 0.1, 0.09, 0.085, 0.08,
        0.078, 0.076, 0.074, 0.072,
      ]),
    };
    material.userData[K.THIN_FILM_STACK] = {
      incidentIor: 1.0,
      angleDependent: true,
      layers: [
        { ior: 2.1, thicknessNm: 72, extinctionCoefficient: 0.015 },
        { ior: 1.46, thicknessNm: 118, extinctionCoefficient: 0.0 },
      ],
    };
  }

  if (scenarioId.includes('layered')) {
    material.userData[K.FRONT_LAYER] = { transmission: [0.95, 0.8, 0.65], roughness: 0.18 };
    material.userData[K.BACK_LAYER] = { transmission: [0.65, 0.8, 0.95], roughness: 0.28 };
  }

  if (scenarioId.includes('sss')) {
    material.transmission = 0.55;
    material.userData[K.SCATTERING_COEFF] = 0.18;
    material.userData[K.SCATTERING_RGB] = [0.16, 0.2, 0.24];
    material.userData[K.SCATTERING_ANISO] = 0.35;
  }
}

export function buildGapClosureCornellThreeScene(
  scenarioId: GapClosureCornellScenarioId | string,
): THREE.Scene {
  const scene = new THREE.Scene();
  const white = new THREE.MeshPhysicalMaterial({ color: 0xe8e8e8, roughness: 1, metalness: 0 });
  const red = new THREE.MeshPhysicalMaterial({ color: 0xab3a2f, roughness: 1, metalness: 0 });
  const green = new THREE.MeshPhysicalMaterial({ color: 0x2d7a3e, roughness: 1, metalness: 0 });
  const glass = new THREE.MeshPhysicalMaterial({
    color: 0x9aa5ad,
    roughness: 0.75,
    metalness: 0,
    transmission: 0,
    ior: 1.5,
    thickness: 0.25,
  });
  applyScenarioMaterialTweaks(glass, scenarioId);

  const mk = (
    geo: THREE.BufferGeometry,
    mat: THREE.MeshPhysicalMaterial,
    pos: [number, number, number],
    scale: [number, number, number],
  ) => {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pos[0], pos[1], pos[2]);
    mesh.scale.set(scale[0], scale[1], scale[2]);
    scene.add(mesh);
  };

  const t = 0.02;
  mk(new THREE.BoxGeometry(2, t, 2), white, [0, -1, 0], [1, 1, 1]);
  mk(new THREE.BoxGeometry(2, t, 2), white, [0, 1, 0], [1, 1, 1]);
  mk(new THREE.BoxGeometry(t, 2, 2), green, [1, 0, 0], [1, 1, 1]);
  mk(new THREE.BoxGeometry(t, 2, 2), red, [-1, 0, 0], [1, 1, 1]);
  mk(new THREE.BoxGeometry(2, 2, t), white, [0, 0, -1], [1, 1, 1]);
  mk(new THREE.BoxGeometry(0.6, 0.6, 0.6), white, [-0.35, -0.65, 0.2], [1, 1, 1]);
  mk(new THREE.BoxGeometry(0.6, 1.2, 0.6), glass, [0.3, -0.35, -0.3], [1, 1, 1]);

  const light = new THREE.RectAreaLight(0xffffff, 12, 1.0, 1.0);
  light.position.set(0, 0.98, 0);
  light.rotation.x = -Math.PI / 2;
  scene.add(light);

  if (scenarioId.includes('parity') || scenarioId.includes('ptwgpu')) {
    const point = new THREE.PointLight(0x99bbff, 2.5);
    point.position.set(-0.7, 0.2, 0.65);
    scene.add(point);
  }

  return scene;
}

export function ptWebgpuExtensionsForGapScenario(
  scenarioId: string,
): Record<string, boolean> | undefined {
  if (scenarioId.includes('spectral') || scenarioId === 'rfe08-13-spectral-payload') {
    return { 'vitrum.ptWebgpu.spectralHeroWavelength': true };
  }
  return undefined;
}

export function defaultCausticForGapScenario(
  scenarioId: string,
  variantCaustic: string | null | undefined,
): 'none' | 'manifold-nee' | 'photon-map' {
  if (variantCaustic === 'manifold-nee' || variantCaustic === 'photon-map') {
    return variantCaustic;
  }
  if (scenarioId.includes('caustic')) return 'manifold-nee';
  return 'none';
}
