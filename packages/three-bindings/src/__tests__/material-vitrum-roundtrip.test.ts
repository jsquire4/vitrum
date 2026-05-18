/**
 * material-vitrum-roundtrip.test.ts
 *
 * Verifies that the vitrum* userData stamps for core Material fields are
 * correctly propagated in both directions of the data flow:
 *
 *   THREE.MeshPhysicalMaterial (userData.vitrum*)
 *     → convertMaterial()
 *   vitrum.Material (core fields)
 *     → vitrumSceneToThree() / vitrumMaterialToThree()
 *   THREE.MeshPhysicalMaterial (userData.vitrum* re-stamped)
 *
 * Coverage: per-face surface absorption layers (frontLayer/backLayer),
 *           chromatic dispersion (Abbe number), volume scattering
 *           (scatteringCoefficient/RGB/anisotropy), and spectral attenuation
 *           + multi-layer thin-film stack.
 *
 * Host-app extensions (dichroic LUTs etc.) live in
 * @vitrum/stained-glass-extensions and carry their own round-trip tests.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { convertMaterial } from '../material.js';
import { vitrumSceneToThree } from '../vitrumSceneToThree.js';
import type { Material as VitrumMaterial, SpectralCurve, ThinFilmStack, SurfaceAbsorptionLayer } from '@vitrum/core';

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/**
 * Build a minimal vitrum.Scene from a vitrum.Material and pass it through
 * vitrumSceneToThree, then return the THREE material on the first mesh.
 */
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

// ────────────────────────────────────────────────────────────────────────────
// THREE → vitrum direction (convertMaterial reads userData.vitrum*)
// ────────────────────────────────────────────────────────────────────────────

describe('convertMaterial: THREE userData.vitrum* → vitrum.Material', () => {
  it('reads vitrumDispersionAbbeNumber (RFE-06)', () => {
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData['vitrumDispersionAbbeNumber'] = 30;
    const v = convertMaterial(m);
    expect(v.dispersionAbbeNumber).toBe(30);
  });

  it('reads vitrumScatteringCoefficient and vitrumScatteringAnisotropy (RFE-07)', () => {
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData['vitrumScatteringCoefficient'] = 2.5;
    m.userData['vitrumScatteringAnisotropy'] = 0.75;
    const v = convertMaterial(m);
    expect(v.scatteringCoefficient).toBe(2.5);
    expect(v.scatteringAnisotropy).toBe(0.75);
  });

  it('reads vitrumScatteringCoefficientRGB (RFE-07)', () => {
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData['vitrumScatteringCoefficientRGB'] = [0.1, 0.2, 0.3];
    const v = convertMaterial(m);
    expect(v.scatteringCoefficientRGB).toEqual([0.1, 0.2, 0.3]);
  });

  it('reads vitrumSpectralAttenuation as SpectralCurve object (RFE-08)', () => {
    const curve: SpectralCurve = {
      wavelengthStart: 380,
      wavelengthEnd: 780,
      values: new Float32Array(81).fill(0.5),
    };
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData['vitrumSpectralAttenuation'] = curve;
    const v = convertMaterial(m);
    expect(v.spectralAttenuation).toBeDefined();
    expect(v.spectralAttenuation?.wavelengthStart).toBe(380);
    expect(v.spectralAttenuation?.wavelengthEnd).toBe(780);
    expect(v.spectralAttenuation?.values.length).toBe(81);
  });

  it('rejects bare Float32Array for vitrumSpectralAttenuation (deprecated path removed, D11)', () => {
    // Pre-alpha breaking change: the bare Float32Array back-compat branch was
    // deleted in the 2026-05-11 sweep. Callers must use the full SpectralCurve
    // shape: { wavelengthStart, wavelengthEnd, values: Float32Array }.
    const raw = new Float32Array(81).fill(0.1);
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData['vitrumSpectralAttenuation'] = raw;
    const v = convertMaterial(m);
    expect(v.spectralAttenuation).toBeUndefined();
  });

  it('reads vitrumThinFilmStack (RFE-08)', () => {
    const stack: ThinFilmStack = { layers: [{ ior: 2.5, thicknessNm: 80 }] };
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData['vitrumThinFilmStack'] = stack;
    const v = convertMaterial(m);
    expect(v.thinFilmStack).toBeDefined();
    expect(v.thinFilmStack?.layers.length).toBe(1);
  });

  it('reads vitrumFrontLayer and vitrumBackLayer (RFE-03)', () => {
    const layer: SurfaceAbsorptionLayer = { transmission: [0.9, 0.85, 0.8] };
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData['vitrumFrontLayer'] = { transmission: [0.9, 0.85, 0.8] };
    m.userData['vitrumBackLayer'] = { transmission: [0.7, 0.7, 0.7] };
    const v = convertMaterial(m);
    expect(v.frontLayer).toBeDefined();
    expect(v.backLayer).toBeDefined();
    expect(v.frontLayer?.transmission).toEqual([0.9, 0.85, 0.8]);
    void layer;
  });

  it('leaves RFE fields undefined when no userData.vitrum* stamps present', () => {
    const m = new THREE.MeshPhysicalMaterial({ color: 0xff0000 });
    const v = convertMaterial(m);
    expect(v.dispersionAbbeNumber).toBeUndefined();
    expect(v.scatteringCoefficient).toBeUndefined();
    expect(v.scatteringAnisotropy).toBeUndefined();
    expect(v.scatteringCoefficientRGB).toBeUndefined();
    expect(v.spectralAttenuation).toBeUndefined();
    expect(v.thinFilmStack).toBeUndefined();
    expect(v.frontLayer).toBeUndefined();
    expect(v.backLayer).toBeUndefined();
  });

  it('ignores non-numeric vitrumDispersionAbbeNumber (type guard)', () => {
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData['vitrumDispersionAbbeNumber'] = 'not-a-number';
    const v = convertMaterial(m);
    expect(v.dispersionAbbeNumber).toBeUndefined();
  });
});

// ────────────────────────────────────────────────────────────────────────────
// vitrum → THREE direction (vitrumSceneToThree stamps userData.vitrum*)
// ────────────────────────────────────────────────────────────────────────────

describe('vitrumSceneToThree: vitrum.Material → THREE userData.vitrum*', () => {
  it('stamps vitrumDispersionAbbeNumber (RFE-06)', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0, metallic: 0,
      dispersionAbbeNumber: 30,
    });
    expect(threeMat.userData['vitrumDispersionAbbeNumber']).toBe(30);
  });

  it('stamps vitrumScatteringCoefficient and vitrumScatteringAnisotropy (RFE-07)', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0, metallic: 0,
      scatteringCoefficient: 2.5,
      scatteringAnisotropy: 0.75,
    });
    expect(threeMat.userData['vitrumScatteringCoefficient']).toBe(2.5);
    expect(threeMat.userData['vitrumScatteringAnisotropy']).toBe(0.75);
  });

  it('stamps vitrumScatteringCoefficientRGB (RFE-07)', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0, metallic: 0,
      scatteringCoefficientRGB: [0.1, 0.2, 0.3],
    });
    expect(threeMat.userData['vitrumScatteringCoefficientRGB']).toEqual([0.1, 0.2, 0.3]);
  });

  it('stamps vitrumSpectralAttenuation (RFE-08)', () => {
    const curve: SpectralCurve = {
      wavelengthStart: 380,
      wavelengthEnd: 780,
      values: new Float32Array(81).fill(0.5),
    };
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0, metallic: 0,
      spectralAttenuation: curve,
    });
    const stamped = threeMat.userData['vitrumSpectralAttenuation'] as SpectralCurve | undefined;
    expect(stamped).toBeDefined();
    expect(stamped?.wavelengthStart).toBe(380);
    expect(stamped?.wavelengthEnd).toBe(780);
  });

  it('stamps vitrumThinFilmStack (RFE-08)', () => {
    const stack: ThinFilmStack = {
      layers: [{ ior: 2.5, thicknessNm: 80 }, { ior: 1.46, thicknessNm: 120 }],
      incidentIor: 1.0,
    };
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0, metallic: 0,
      thinFilmStack: stack,
    });
    const stamped = threeMat.userData['vitrumThinFilmStack'] as ThinFilmStack | undefined;
    expect(stamped).toBeDefined();
    expect(stamped?.layers.length).toBe(2);
  });

  it('stamps vitrumFrontLayer and vitrumBackLayer (RFE-03)', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0, metallic: 0,
      frontLayer: { transmission: [0.9, 0.85, 0.8] },
      backLayer: { transmission: [0.7, 0.7, 0.7] },
    });
    expect(threeMat.userData['vitrumFrontLayer']).toBeDefined();
    expect(threeMat.userData['vitrumBackLayer']).toBeDefined();
  });

  it('does NOT stamp absent RFE fields (no phantom keys)', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
    });
    expect(threeMat.userData['vitrumDispersionAbbeNumber']).toBeUndefined();
    expect(threeMat.userData['vitrumScatteringCoefficient']).toBeUndefined();
    expect(threeMat.userData['vitrumSpectralAttenuation']).toBeUndefined();
    expect(threeMat.userData['vitrumThinFilmStack']).toBeUndefined();
    expect(threeMat.userData['vitrumFrontLayer']).toBeUndefined();
    expect(threeMat.userData['vitrumBackLayer']).toBeUndefined();
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Gap 5 — Anisotropy round-trip (stainedGlass audit 2026-05-12)
// ────────────────────────────────────────────────────────────────────────────

describe('convertMaterial: anisotropy + anisotropyRotation (Gap 5)', () => {
  it('reads anisotropy and anisotropyRotation directly from THREE material', () => {
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.anisotropy = 0.7;
    m.anisotropyRotation = 0.3;
    const v = convertMaterial(m);
    expect(v.anisotropy).toBeCloseTo(0.7);
    expect(v.anisotropyRotation).toBeCloseTo(0.3);
  });

  it('leaves anisotropy undefined when THREE material has default anisotropy=0', () => {
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    // THREE defaults anisotropy to 0; vitrum should not populate the field.
    const v = convertMaterial(m);
    expect(v.anisotropy).toBeUndefined();
    expect(v.anisotropyRotation).toBeUndefined();
  });
});

describe('vitrumSceneToThree: anisotropy + anisotropyRotation written to THREE material (Gap 5)', () => {
  it('writes anisotropy and anisotropyRotation onto the THREE material', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0, metallic: 0,
      anisotropy: 0.7,
      anisotropyRotation: 0.3,
    });
    expect(threeMat.anisotropy).toBeCloseTo(0.7);
    expect(threeMat.anisotropyRotation).toBeCloseTo(0.3);
  });

  it('preserves anisotropy=0 when explicitly set in vitrum Material', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0, metallic: 0,
      anisotropy: 0,
    });
    expect(threeMat.anisotropy).toBe(0);
    expect(threeMat.anisotropyRotation).toBe(0); // defaults to 0 when anisotropy defined
  });

  it('does NOT stamp anisotropy when absent from vitrum Material (no phantom field)', () => {
    // When anisotropy is absent the THREE material must have its own default (0).
    // We verify the field was not artificially overwritten by the converter.
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
    });
    // THREE default is 0; as long as we haven't broken the default this passes.
    expect(threeMat.anisotropy).toBe(0);
  });
});

describe('Full round-trip: anisotropy THREE → vitrum → THREE (Gap 5)', () => {
  it('preserves anisotropy=0.7 and anisotropyRotation=0.3 through the full round-trip', () => {
    const original = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    original.anisotropy = 0.7;
    original.anisotropyRotation = 0.3;

    // THREE → vitrum
    const vitrumMat = convertMaterial(original);
    expect(vitrumMat.anisotropy).toBeCloseTo(0.7);
    expect(vitrumMat.anisotropyRotation).toBeCloseTo(0.3);

    // vitrum → THREE
    const backToThree = vitrumMatToThreeMat(vitrumMat);
    expect(backToThree.anisotropy).toBeCloseTo(0.7);
    expect(backToThree.anisotropyRotation).toBeCloseTo(0.3);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Full round-trip: THREE → vitrum → THREE
// ────────────────────────────────────────────────────────────────────────────
//
// Note: dichroic LUT round-trip coverage has moved to
// @vitrum/stained-glass-extensions (W3-D3 — host-app extensions extracted into
// a dedicated opt-in package; library defaults to host-agnostic behavior).
// See packages/stained-glass-extensions/src/__tests__/dichroic-luts.test.ts.

describe('library default: no host-app extension behavior unless converter wired', () => {
  it('convertMaterial does NOT auto-populate Material.extensions for unknown userData keys', () => {
    // Library-default convertMaterial is host-agnostic — random userData keys
    // are NOT mirrored into Material.extensions. Hosts that need a specific
    // extension wire a MaterialExtensionConverter explicitly.
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData['someHostSpecificKey'] = { foo: 'bar' };
    const v = convertMaterial(m);
    expect(v.extensions).toBeUndefined();
  });
});

describe('Full round-trip: THREE userData.vitrum* → vitrum.Material → THREE userData.vitrum*', () => {
  it('preserves all RFE-06..08 + RFE-03 fields end-to-end', () => {
    // Step 1: build a THREE material with all vitrum stamps
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

    // Step 2: THREE → vitrum
    const vitrumMat = convertMaterial(original);
    expect(vitrumMat.dispersionAbbeNumber).toBe(30);
    expect(vitrumMat.scatteringCoefficient).toBe(2.5);
    expect(vitrumMat.scatteringAnisotropy).toBe(0.75);
    expect(vitrumMat.scatteringCoefficientRGB).toEqual([0.1, 0.2, 0.3]);
    expect(vitrumMat.spectralAttenuation?.wavelengthStart).toBe(380);
    expect(vitrumMat.thinFilmStack?.layers.length).toBe(1);
    expect(vitrumMat.frontLayer).toBeDefined();
    expect(vitrumMat.backLayer).toBeDefined();

    // Step 3: vitrum → THREE
    const backToThree = vitrumMatToThreeMat(vitrumMat);
    expect(backToThree.userData['vitrumDispersionAbbeNumber']).toBe(30);
    expect(backToThree.userData['vitrumScatteringCoefficient']).toBe(2.5);
    expect(backToThree.userData['vitrumScatteringAnisotropy']).toBe(0.75);
    expect(backToThree.userData['vitrumScatteringCoefficientRGB']).toEqual([0.1, 0.2, 0.3]);
    const sa = backToThree.userData['vitrumSpectralAttenuation'] as SpectralCurve;
    expect(sa.wavelengthStart).toBe(380);
    expect(sa.wavelengthEnd).toBe(780);
    expect(sa.values).toBe(spectralValues); // same Float32Array reference — no copy
    expect(backToThree.userData['vitrumThinFilmStack']).toBeDefined();
    expect(backToThree.userData['vitrumFrontLayer']).toBeDefined();
    expect(backToThree.userData['vitrumBackLayer']).toBeDefined();
  });
});
