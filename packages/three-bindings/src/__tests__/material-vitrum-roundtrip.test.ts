/**
 * material-vitrum-roundtrip.test.ts
 *
 * Verifies that the vitrum* userData stamps are correctly propagated in both
 * directions of the data flow:
 *
 *   THREE.MeshPhysicalMaterial (userData.vitrum*)
 *     → convertMaterial()
 *   vitrum.Material (RFE fields)
 *     → vitrumSceneToThree() / vitrumMaterialToThree()
 *   THREE.MeshPhysicalMaterial (userData.vitrum* re-stamped)
 *
 * Coverage: RFE-03 (frontLayer/backLayer), RFE-06 (dispersionAbbeNumber),
 *           RFE-07 (scatteringCoefficient/RGB/anisotropy), RFE-08 (spectralAttenuation/thinFilmStack).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { convertMaterial } from '../material.js';
import { vitrumSceneToThree } from '../vitrumSceneToThree.js';
import type { MaterialSpec as VitrumMaterial, SpectralCurve, ThinFilmStack, SurfaceAbsorptionLayer } from '@vitrum/core';
import { asTextureRef } from '@vitrum/core';

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

  it('rejects non-contract vitrumSpectralAttenuation shapes', () => {
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData['vitrumSpectralAttenuation'] = {
      wavelengthStart: 380,
      wavelengthEnd: 780,
      values: [0.1, 0.2, 0.3],
    };
    expect(convertMaterial(m).spectralAttenuation).toBeUndefined();

    m.userData['vitrumSpectralAttenuation'] = {
      wavelengthStart: 780,
      wavelengthEnd: 380,
      values: new Float32Array([0.1, 0.2, 0.3]),
    };
    expect(convertMaterial(m).spectralAttenuation).toBeUndefined();
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

// ────────────────────────────────────────────────────────────────────────────
// Disney-BSDF lobes — sheen / clearcoat / iridescence (vitrum → THREE)
//
// These are first-class THREE.MeshPhysicalMaterial properties the pt-webgl fork
// reads verbatim in MaterialsTexture.js. The reverse direction (convertMaterial)
// already extracted them; this pins that vitrumMaterialToThree re-applies them
// so a core-authored MaterialSpec carrying these lobes reaches the renderer.
// ────────────────────────────────────────────────────────────────────────────

describe('vitrumSceneToThree: Disney-BSDF lobes written onto THREE material', () => {
  it('applies all 8 lobe fields with the correct THREE property names + shapes', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1],
      roughness: 0.4,
      metallic: 0,
      sheen: 0.8,
      sheenColor: [0.2, 0.5, 0.9],
      sheenRoughness: 0.6,
      clearcoat: 0.7,
      clearcoatRoughness: 0.25,
      iridescence: 0.9,
      iridescenceIor: 1.8,
      iridescenceThicknessRange: [120, 480],
    });

    // sheen (float), sheenColor (THREE.Color), sheenRoughness (float)
    expect(threeMat.sheen).toBeCloseTo(0.8);
    expect(threeMat.sheenColor).toBeInstanceOf(THREE.Color);
    expect(threeMat.sheenColor.r).toBeCloseTo(0.2);
    expect(threeMat.sheenColor.g).toBeCloseTo(0.5);
    expect(threeMat.sheenColor.b).toBeCloseTo(0.9);
    expect(threeMat.sheenRoughness).toBeCloseTo(0.6);

    // clearcoat (float), clearcoatRoughness (float)
    expect(threeMat.clearcoat).toBeCloseTo(0.7);
    expect(threeMat.clearcoatRoughness).toBeCloseTo(0.25);

    // iridescence (float); core iridescenceIor → THREE iridescenceIOR (caps);
    // iridescenceThicknessRange is a [min,max] array.
    expect(threeMat.iridescence).toBeCloseTo(0.9);
    expect(threeMat.iridescenceIOR).toBeCloseTo(1.8);
    expect(Array.isArray(threeMat.iridescenceThicknessRange)).toBe(true);
    expect(threeMat.iridescenceThicknessRange[0]).toBeCloseTo(120);
    expect(threeMat.iridescenceThicknessRange[1]).toBeCloseTo(480);
  });

  it('preserves explicit 0 lobe values (round-trip clean), not clobbered to THREE defaults', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1],
      roughness: 0,
      metallic: 0,
      sheen: 0,
      clearcoat: 0,
      iridescence: 0,
    });
    expect(threeMat.sheen).toBe(0);
    expect(threeMat.clearcoat).toBe(0);
    expect(threeMat.iridescence).toBe(0);
  });

  it('does NOT clobber THREE defaults when lobes are absent from the spec', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
    });
    // THREE.MeshPhysicalMaterial defaults: sheen=0, clearcoat=0, iridescence=0,
    // iridescenceIOR=1.3, iridescenceThicknessRange=[100,400], sheenColor=black.
    expect(threeMat.sheen).toBe(0);
    expect(threeMat.clearcoat).toBe(0);
    expect(threeMat.iridescence).toBe(0);
    expect(threeMat.iridescenceIOR).toBeCloseTo(1.3);
    expect(threeMat.iridescenceThicknessRange).toEqual([100, 400]);
  });
});

describe('Full round-trip: Disney lobes core → THREE → convertMaterial → core', () => {
  it('preserves all 8 lobes (incl. the iridescenceIor↔iridescenceIOR rename)', () => {
    const original: VitrumMaterial = {
      baseColor: [1, 1, 1],
      roughness: 0.4,
      metallic: 0,
      sheen: 0.8,
      sheenColor: [0.2, 0.5, 0.9],
      sheenRoughness: 0.6,
      clearcoat: 0.7,
      clearcoatRoughness: 0.25,
      iridescence: 0.9,
      iridescenceIor: 1.8,
      iridescenceThicknessRange: [120, 480],
    };

    // core → THREE
    const threeMat = vitrumMatToThreeMat(original);
    // THREE → core
    const back = convertMaterial(threeMat);

    expect(back.sheen).toBeCloseTo(0.8);
    expect(back.sheenColor?.[0]).toBeCloseTo(0.2);
    expect(back.sheenColor?.[1]).toBeCloseTo(0.5);
    expect(back.sheenColor?.[2]).toBeCloseTo(0.9);
    expect(back.sheenRoughness).toBeCloseTo(0.6);
    expect(back.clearcoat).toBeCloseTo(0.7);
    expect(back.clearcoatRoughness).toBeCloseTo(0.25);
    expect(back.iridescence).toBeCloseTo(0.9);
    expect(back.iridescenceIor).toBeCloseTo(1.8);
    expect(back.iridescenceThicknessRange?.[0]).toBeCloseTo(120);
    expect(back.iridescenceThicknessRange?.[1]).toBeCloseTo(480);
  });
});

describe('vitrumSceneToThree: emissive / alphaMap mapping', () => {
  it('treats vitrum emissive as final radiance color (no double intensity scaling)', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      emissive: [2, 3, 4],
      emissiveIntensity: 5,
    });
    expect(threeMat.emissive.r).toBeCloseTo(2);
    expect(threeMat.emissive.g).toBeCloseTo(3);
    expect(threeMat.emissive.b).toBeCloseTo(4);
    expect(threeMat.emissiveIntensity).toBe(1);
  });

  it('forwards alphaMap from vitrum material', () => {
    const alphaMap = new THREE.Texture();
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      alphaMap: asTextureRef(alphaMap),
    });
    expect(threeMat.alphaMap).toBe(alphaMap);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// G-P0.4 (d) — reverse path (vitrumSceneToThree) must NOT silently drop the
// opacity / alphaMode / AO / lobe-texture fields the forward path produces.
//
// Before the d72d39b "round-trip restores" fix, vitrumMaterialToThree forwarded
// only the 7 base maps + Disney lobe SCALARS, so a core-authored alpha-blend /
// AO'd / clearcoat-mapped material lost those when the walkaround engine
// synthesized its THREE scene (BVH/DDGI source) or when fed to pt-webgl —
// transparency survived only via transmission>0. Each test below asserts a field
// that the FORWARD path (convertMaterial) already extracts now survives the
// REVERSE conversion too. These would have caught the silent-drop asymmetry.
// ────────────────────────────────────────────────────────────────────────────

describe('vitrumSceneToThree: G-P0.4(d) alpha / opacity reverse path', () => {
  it('forwards opacity onto the THREE material', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      opacity: 0.4,
    });
    expect(threeMat.opacity).toBeCloseTo(0.4);
  });

  it('alphaMode=blend → transparent=true, alphaTest=0', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      alphaMode: 'blend', opacity: 0.5,
    });
    expect(threeMat.transparent).toBe(true);
    expect(threeMat.alphaTest).toBe(0);
  });

  it('alphaMode=mask → alphaTest = alphaCutoff, transparent=false', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      alphaMode: 'mask', alphaCutoff: 0.33,
    });
    expect(threeMat.transparent).toBe(false);
    expect(threeMat.alphaTest).toBeCloseTo(0.33);
  });

  it('alphaMode=mask without explicit cutoff defaults alphaTest to 0.5 (glTF default)', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      alphaMode: 'mask',
    });
    expect(threeMat.alphaTest).toBeCloseTo(0.5);
  });

  it('bare opacity < 1 (no alphaMode) implies transparent (so blend survives without transmission)', () => {
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      opacity: 0.6,
    });
    expect(threeMat.transparent).toBe(true);
    expect(threeMat.opacity).toBeCloseTo(0.6);
  });
});

describe('vitrumSceneToThree: G-P0.4(d) AO map reverse path', () => {
  it('forwards aoMap and aoMapIntensity', () => {
    const aoMap = new THREE.Texture();
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      aoMap: asTextureRef(aoMap),
      aoMapIntensity: 0.7,
    });
    expect(threeMat.aoMap).toBe(aoMap);
    expect(threeMat.aoMapIntensity).toBeCloseTo(0.7);
  });

  it('aoMap with default intensity round-trips as intensity 1', () => {
    const aoMap = new THREE.Texture();
    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      aoMap: asTextureRef(aoMap),
    });
    expect(threeMat.aoMap).toBe(aoMap);
    expect(threeMat.aoMapIntensity).toBeCloseTo(1);
  });
});

describe('vitrumSceneToThree: G-P0.4(d) lobe + transmission texture maps reverse path', () => {
  it('forwards transmissionMap, clearcoat/sheen/iridescence/anisotropy maps (not just the scalar lobes)', () => {
    const transmissionMap = new THREE.Texture();
    const clearcoatMap = new THREE.Texture();
    const clearcoatRoughnessMap = new THREE.Texture();
    const clearcoatNormalMap = new THREE.Texture();
    const sheenColorMap = new THREE.Texture();
    const sheenRoughnessMap = new THREE.Texture();
    const iridescenceMap = new THREE.Texture();
    const iridescenceThicknessMap = new THREE.Texture();
    const anisotropyMap = new THREE.Texture();

    const threeMat = vitrumMatToThreeMat({
      baseColor: [1, 1, 1], roughness: 0.4, metallic: 0,
      transmission: 0.5,
      transmissionMap: asTextureRef(transmissionMap),
      clearcoatMap: asTextureRef(clearcoatMap),
      clearcoatRoughnessMap: asTextureRef(clearcoatRoughnessMap),
      clearcoatNormalMap: asTextureRef(clearcoatNormalMap),
      sheenColorMap: asTextureRef(sheenColorMap),
      sheenRoughnessMap: asTextureRef(sheenRoughnessMap),
      iridescenceMap: asTextureRef(iridescenceMap),
      iridescenceThicknessMap: asTextureRef(iridescenceThicknessMap),
      anisotropyMap: asTextureRef(anisotropyMap),
    });

    expect(threeMat.transmissionMap).toBe(transmissionMap);
    expect(threeMat.clearcoatMap).toBe(clearcoatMap);
    expect(threeMat.clearcoatRoughnessMap).toBe(clearcoatRoughnessMap);
    expect(threeMat.clearcoatNormalMap).toBe(clearcoatNormalMap);
    expect(threeMat.sheenColorMap).toBe(sheenColorMap);
    expect(threeMat.sheenRoughnessMap).toBe(sheenRoughnessMap);
    expect(threeMat.iridescenceMap).toBe(iridescenceMap);
    expect(threeMat.iridescenceThicknessMap).toBe(iridescenceThicknessMap);
    expect(threeMat.anisotropyMap).toBe(anisotropyMap);
  });
});

describe('Full round-trip: G-P0.4(d) alpha + AO + maps THREE → vitrum → THREE', () => {
  it('preserves alphaMode/alphaCutoff, opacity, aoMap(+intensity) and the lobe maps', () => {
    const aoMap = new THREE.Texture();
    const transmissionMap = new THREE.Texture();
    const clearcoatMap = new THREE.Texture();

    const original = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    // alphaMode=mask path: alphaTest>0 → convertMaterial reports 'mask'.
    original.alphaTest = 0.4;
    original.opacity = 0.85;
    original.aoMap = aoMap;
    original.aoMapIntensity = 0.6;
    original.transmission = 0.5;
    original.transmissionMap = transmissionMap;
    original.clearcoat = 0.8;
    original.clearcoatMap = clearcoatMap;

    // THREE → core
    const vitrumMat = convertMaterial(original);
    expect(vitrumMat.alphaMode).toBe('mask');
    expect(vitrumMat.alphaCutoff).toBeCloseTo(0.4);
    expect(vitrumMat.opacity).toBeCloseTo(0.85);
    expect(vitrumMat.aoMap).toBeDefined();
    expect(vitrumMat.aoMapIntensity).toBeCloseTo(0.6);
    expect(vitrumMat.transmissionMap).toBeDefined();
    expect(vitrumMat.clearcoatMap).toBeDefined();

    // core → THREE (the previously-dropping direction)
    const back = vitrumMatToThreeMat(vitrumMat);
    expect(back.alphaTest).toBeCloseTo(0.4);
    expect(back.transparent).toBe(false); // mask is not transparent
    expect(back.opacity).toBeCloseTo(0.85);
    expect(back.aoMap).toBe(aoMap);
    expect(back.aoMapIntensity).toBeCloseTo(0.6);
    expect(back.transmissionMap).toBe(transmissionMap);
    expect(back.clearcoatMap).toBe(clearcoatMap);
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
