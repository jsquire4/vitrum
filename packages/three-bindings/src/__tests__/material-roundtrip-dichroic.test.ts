/**
 * Dichroic LUT + full advanced-field round-trip tests (split from material-vitrum-roundtrip).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { convertMaterial } from '../material.js';
import { vitrumSceneToThree } from '../vitrumSceneToThree.js';
import type {
  MaterialSpec as VitrumMaterial,
  SpectralCurve,
  ThinFilmStack,
  SurfaceAbsorptionLayer,
} from '@vitrum/core';

function vitrumMatToThreeMat(mat: VitrumMaterial): THREE.MeshPhysicalMaterial {
  const pos = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const nor = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const scene = vitrumSceneToThree({
    primitives: [{ kind: 'mesh', id: 'test', positions: pos, normals: nor, material: mat }],
    emitters: [],
    environment: { kind: 'none' },
  });
  const mesh = scene.children[0] as THREE.Mesh;
  return mesh.material as THREE.MeshPhysicalMaterial;
}

describe('Dichroic LUT round-trip (Fix 3)', () => {
  it('reads vitrumDichroicReflectanceLUT + vitrumDichroicTransmittanceLUT into Material.extensions.dichroicLUTs', () => {
    const reflectanceLut = { __marker: 'reflectance' };
    const transmittanceLut = { __marker: 'transmittance' };
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData['vitrumDichroicReflectanceLUT'] = reflectanceLut;
    m.userData['vitrumDichroicTransmittanceLUT'] = transmittanceLut;
    const v = convertMaterial(m);
    const dichroic = v.extensions?.['dichroicLUTs'] as
      | { reflectance?: unknown; transmittance?: unknown }
      | undefined;
    expect(dichroic).toBeDefined();
    expect(dichroic?.reflectance).toBe(reflectanceLut);
    expect(dichroic?.transmittance).toBe(transmittanceLut);
  });

  it('stamps userData.vitrumDichroic*LUT from Material.extensions.dichroicLUTs (vitrum → THREE)', () => {
    const reflectanceLut = { __marker: 'reflectance' };
    const transmittanceLut = { __marker: 'transmittance' };
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0, metallic: 0,
      extensions: {
        dichroicLUTs: { reflectance: reflectanceLut, transmittance: transmittanceLut },
      },
    });
    expect(threeMat.userData['vitrumDichroicReflectanceLUT']).toBe(reflectanceLut);
    expect(threeMat.userData['vitrumDichroicTransmittanceLUT']).toBe(transmittanceLut);
  });

  it('does NOT stamp dichroic keys when extensions.dichroicLUTs is absent (no phantom keys)', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
    });
    expect(threeMat.userData['vitrumDichroicReflectanceLUT']).toBeUndefined();
    expect(threeMat.userData['vitrumDichroicTransmittanceLUT']).toBeUndefined();
  });

  it('full round-trip: THREE userData.vitrumDichroic*LUT → vitrum → THREE userData.vitrumDichroic*LUT', () => {
    const reflectanceLut = { __marker: 'reflectance-rt' };
    const transmittanceLut = { __marker: 'transmittance-rt' };
    const original = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    original.userData['vitrumDichroicReflectanceLUT'] = reflectanceLut;
    original.userData['vitrumDichroicTransmittanceLUT'] = transmittanceLut;

    const vitrumMat = convertMaterial(original);
    const dichroic = vitrumMat.extensions?.['dichroicLUTs'] as
      | { reflectance?: unknown; transmittance?: unknown }
      | undefined;
    expect(dichroic?.reflectance).toBe(reflectanceLut);
    expect(dichroic?.transmittance).toBe(transmittanceLut);

    const backToThree = vitrumMatToThreeMat(vitrumMat);
    expect(backToThree.userData['vitrumDichroicReflectanceLUT']).toBe(reflectanceLut);
    expect(backToThree.userData['vitrumDichroicTransmittanceLUT']).toBe(transmittanceLut);
  });

  it('survives partial dichroic LUT (only reflectance present)', () => {
    const reflectanceLut = { __marker: 'reflectance-only' };
    const original = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    original.userData['vitrumDichroicReflectanceLUT'] = reflectanceLut;
    const vitrumMat = convertMaterial(original);
    const dichroic = vitrumMat.extensions?.['dichroicLUTs'] as
      | { reflectance?: unknown; transmittance?: unknown }
      | undefined;
    expect(dichroic?.reflectance).toBe(reflectanceLut);
    expect(dichroic?.transmittance).toBeUndefined();
    const backToThree = vitrumMatToThreeMat(vitrumMat);
    expect(backToThree.userData['vitrumDichroicReflectanceLUT']).toBe(reflectanceLut);
    expect(backToThree.userData['vitrumDichroicTransmittanceLUT']).toBeUndefined();
  });
});

describe('Full round-trip: THREE userData.vitrum* → vitrum.Material → THREE userData.vitrum*', () => {
  it('preserves all RFE-06..08 + RFE-03 fields end-to-end', () => {
    const original = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    const spectralValues = new Float32Array(81).fill(0.3);
    const thinFilm: ThinFilmStack = { layers: [{ ior: 2.5, thicknessNm: 80 }] };
    original.userData['vitrumDispersionAbbeNumber'] = 30;
    original.userData['vitrumScatteringCoefficient'] = 2.5;
    original.userData['vitrumScatteringAnisotropy'] = 0.75;
    original.userData['vitrumScatteringCoefficientRGB'] = [0.1, 0.2, 0.3];
    original.userData['vitrumSpectralAttenuation'] = {
      wavelengthStart: 380, wavelengthEnd: 780, values: spectralValues,
    } satisfies SpectralCurve;
    original.userData['vitrumThinFilmStack'] = thinFilm;
    original.userData['vitrumFrontLayer'] = { transmission: [0.9, 0.85, 0.8] } satisfies SurfaceAbsorptionLayer;
    original.userData['vitrumBackLayer'] = { transmission: [0.7, 0.7, 0.7] } satisfies SurfaceAbsorptionLayer;

    const vitrumMat = convertMaterial(original);
    expect(vitrumMat.dispersionAbbeNumber).toBe(30);
    expect(vitrumMat.scatteringCoefficient).toBe(2.5);
    expect(vitrumMat.scatteringAnisotropy).toBe(0.75);
    expect(vitrumMat.scatteringCoefficientRGB).toEqual([0.1, 0.2, 0.3]);
    expect(vitrumMat.spectralAttenuation?.wavelengthStart).toBe(380);
    expect(vitrumMat.thinFilmStack?.layers.length).toBe(1);
    expect(vitrumMat.frontLayer).toBeDefined();
    expect(vitrumMat.backLayer).toBeDefined();

    const backToThree = vitrumMatToThreeMat(vitrumMat);
    expect(backToThree.userData['vitrumDispersionAbbeNumber']).toBe(30);
    expect(backToThree.userData['vitrumScatteringCoefficient']).toBe(2.5);
    expect(backToThree.userData['vitrumScatteringAnisotropy']).toBe(0.75);
    expect(backToThree.userData['vitrumScatteringCoefficientRGB']).toEqual([0.1, 0.2, 0.3]);
    const sa = backToThree.userData['vitrumSpectralAttenuation'] as SpectralCurve;
    expect(sa.wavelengthStart).toBe(380);
    expect(sa.wavelengthEnd).toBe(780);
    expect(sa.values).toBe(spectralValues);
    expect(backToThree.userData['vitrumThinFilmStack']).toBeDefined();
    expect(backToThree.userData['vitrumFrontLayer']).toBeDefined();
    expect(backToThree.userData['vitrumBackLayer']).toBeDefined();
  });
});
