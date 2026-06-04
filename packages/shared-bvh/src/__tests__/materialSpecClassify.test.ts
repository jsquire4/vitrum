/**
 * Equivalence tests — THREE-free `MaterialSpec` emitter / Beer-Lambert /
 * surface-texture classifiers vs. their THREE-material siblings.
 *
 * Validates the THREE-decoupling increment (`plan/three-decouple-analysis-
 * 2026-06-03.md` §7 next-increment 1): each `*Core` / `materialSpec*` function
 * in `materialEntry.ts` must produce the SAME result as the THREE-material
 * reader it replaces, for an equivalent material. The downstream bytes these
 * feed (emitter CDF, `bvhIndex.w`, `bvh_beer`, `bvh_emissive_le`) are
 * golden-pinned, so byte-for-byte equivalence here is the no-GPU validation
 * that the producer can be swapped THREE → core with no radiometric drift.
 *
 * **Oracle.** The THREE-side functions are file-local (not exported) in
 * `@vitrum/walkaround-hybrid`, and that package is DOWNSTREAM of `shared-bvh`
 * (importing it would invert the dependency). So the THREE variants are
 * reproduced here as verbatim oracles (copied from
 * `walkaround-hybrid/src/restir/{packingHelpers,emitterList}.ts` as of
 * 2026-06-03). They run on real `THREE.MeshPhysicalMaterial` objects; the
 * `*Core` functions run on the equivalent `MaterialSpec`; the two outputs are
 * asserted equal. If the THREE source changes, this visible oracle is the
 * review surface that keeps the pairing honest.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import type { MaterialSpec } from '@vitrum/core';

import {
  MATERIAL_DEFAULT_TRI_COLOR,
  applyBeerLambertColor,
  classifyTriangleEmitterCore,
  materialSpecEmissiveLe,
  materialSpecSkipEmitter,
  materialSpecSurfaceTextureId,
  materialSpecTriColor,
} from '../materialEntry.js';

// ──────────────────────────────────────────────────────────────────────────
// THREE-side oracles — verbatim copies of the file-local functions in
// walkaround-hybrid (packingHelpers.ts / emitterList.ts), 2026-06-03.
// ──────────────────────────────────────────────────────────────────────────

const WARM_GRAY_DEFAULT_R = 153;
const WARM_GRAY_DEFAULT_G = 148;
const WARM_GRAY_DEFAULT_B = 140;

function oracleApplyBeerLambert(
  attCol: THREE.Color,
  thickness: number | undefined,
  attDist: number | undefined,
): THREE.Color {
  if (thickness === undefined || attDist === undefined) return attCol;
  if (!Number.isFinite(thickness) || !Number.isFinite(attDist)) return attCol;
  if (thickness <= 0 || attDist <= 0) return attCol;
  const k = thickness / attDist;
  return new THREE.Color(
    Math.pow(Math.max(1e-6, attCol.r), k),
    Math.pow(Math.max(1e-6, attCol.g), k),
    Math.pow(Math.max(1e-6, attCol.b), k),
  );
}

function oracleResolveTriColor(mat: THREE.Material, applyBeer: boolean): THREE.Color {
  const physMat = mat as THREE.MeshPhysicalMaterial;
  const stdMat = mat as THREE.MeshStandardMaterial;
  const transmission = physMat.transmission ?? 0;
  const isTransmissive = transmission > 0.01;
  const attenColor = (physMat as { attenuationColor?: THREE.Color }).attenuationColor;
  if (isTransmissive && attenColor) {
    if (applyBeer) {
      return oracleApplyBeerLambert(
        attenColor,
        (physMat as { thickness?: number }).thickness,
        (physMat as { attenuationDistance?: number }).attenuationDistance,
      );
    }
    return attenColor;
  }
  return physMat.color ?? stdMat?.color ?? new THREE.Color(0.6, 0.58, 0.55);
}

function oracleMaterialEmissiveLe(mat: THREE.Material): [number, number, number] | null {
  const meshMat = mat as THREE.MeshStandardMaterial;
  const em = meshMat.emissive;
  if (!em) return null;
  const ei = meshMat.emissiveIntensity;
  if (!(ei && ei > 0)) return null;
  if (em.r <= 0 && em.g <= 0 && em.b <= 0) return null;
  return [em.r * ei, em.g * ei, em.b * ei];
}

function oracleClassifyTriangleEmitter(
  mat: THREE.Material,
  normal: { x: number; y: number; z: number },
  lightDir: THREE.Vector3,
  primaryIntensity: number,
): { color: [number, number, number]; intensity: number } | null {
  const meshMat = mat as THREE.MeshStandardMaterial;
  const emissiveLe = oracleMaterialEmissiveLe(mat);
  if (emissiveLe != null) {
    return { color: emissiveLe, intensity: meshMat.emissiveIntensity ?? 1 };
  }
  const physMat = mat as THREE.MeshPhysicalMaterial;
  if (!physMat.transmission || physMat.transmission <= 0.1) return null;
  const skipEmitter =
    (mat.userData as { skipEmitter?: boolean } | undefined)?.skipEmitter === true;
  if (skipEmitter) return null;
  const sunDot = Math.abs(
    lightDir.x * normal.x + lightDir.y * normal.y + lightDir.z * normal.z,
  );
  if (sunDot <= 0.05) return null;
  const baseColor = physMat.color ?? new THREE.Color(1, 1, 1);
  const attenColor = physMat.attenuationColor ?? new THREE.Color(1, 1, 1);
  const trans = physMat.transmission;
  return {
    color: [
      baseColor.r * attenColor.r * trans * primaryIntensity * sunDot,
      baseColor.g * attenColor.g * trans * primaryIntensity * sunDot,
      baseColor.b * attenColor.b * trans * primaryIntensity * sunDot,
    ],
    intensity: primaryIntensity * trans * sunDot,
  };
}

/** Same RGBA8 byte pack the THREE `packBVHIndexWTri` applies to a resolved
 *  color (raw attenuation color, no Beer), so we can compare the byte the GPU
 *  actually sees, not just the float. */
function rgb8(c: { r: number; g: number; b: number } | [number, number, number]): [number, number, number] {
  const r = Array.isArray(c) ? c[0] : c.r;
  const g = Array.isArray(c) ? c[1] : c.g;
  const b = Array.isArray(c) ? c[2] : c.b;
  return [Math.round(r * 255) & 0xff, Math.round(g * 255) & 0xff, Math.round(b * 255) & 0xff];
}

/** The Beer pack clamps to [0,1] before the byte round (THREE `packBVHBeerColorTri`). */
function rgb8Clamped(c: [number, number, number]): [number, number, number] {
  return [
    Math.round(Math.min(1, c[0]) * 255) & 0xff,
    Math.round(Math.min(1, c[1]) * 255) & 0xff,
    Math.round(Math.min(1, c[2]) * 255) & 0xff,
  ];
}

// ──────────────────────────────────────────────────────────────────────────
// Matched material pairs — one THREE.MeshPhysicalMaterial + the equivalent
// MaterialSpec (fields mapped exactly as `convertMaterial` / `extractThree-
// PbrScalars` would: THREE.color → baseColor, etc.).
// ──────────────────────────────────────────────────────────────────────────

interface Pair {
  name: string;
  three: THREE.MeshPhysicalMaterial;
  core: MaterialSpec;
}

function makePair(name: string, p: {
  baseColor?: [number, number, number];
  emissive?: [number, number, number];
  emissiveIntensity?: number;
  roughness?: number;
  metallic?: number;
  transmission?: number;
  ior?: number;
  attenuationColor?: [number, number, number];
  attenuationDistance?: number;
  thickness?: number;
  surfaceTextureId?: number;
  skipEmitter?: boolean;
}): Pair {
  const three = new THREE.MeshPhysicalMaterial();
  if (p.baseColor) three.color = new THREE.Color(...p.baseColor);
  if (p.emissive) three.emissive = new THREE.Color(...p.emissive);
  if (p.emissiveIntensity !== undefined) three.emissiveIntensity = p.emissiveIntensity;
  if (p.roughness !== undefined) three.roughness = p.roughness;
  if (p.metallic !== undefined) three.metalness = p.metallic;
  if (p.transmission !== undefined) three.transmission = p.transmission;
  if (p.ior !== undefined) three.ior = p.ior;
  if (p.attenuationColor) three.attenuationColor = new THREE.Color(...p.attenuationColor);
  if (p.attenuationDistance !== undefined) three.attenuationDistance = p.attenuationDistance;
  if (p.thickness !== undefined) three.thickness = p.thickness;
  if (p.surfaceTextureId !== undefined) three.userData['surfaceTextureId'] = p.surfaceTextureId;
  if (p.skipEmitter !== undefined) three.userData['skipEmitter'] = p.skipEmitter;

  const core: MaterialSpec = {
    baseColor: p.baseColor ?? [1, 1, 1],
    roughness: p.roughness ?? 0.5,
    metallic: p.metallic ?? 0,
    ...(p.emissive ? { emissive: p.emissive } : {}),
    ...(p.emissiveIntensity !== undefined ? { emissiveIntensity: p.emissiveIntensity } : {}),
    ...(p.transmission !== undefined ? { transmission: p.transmission } : {}),
    ...(p.ior !== undefined ? { ior: p.ior } : {}),
    ...(p.attenuationColor ? { attenuationColor: p.attenuationColor } : {}),
    ...(p.attenuationDistance !== undefined ? { attenuationDistance: p.attenuationDistance } : {}),
    ...(p.thickness !== undefined ? { thickness: p.thickness } : {}),
    ...(p.surfaceTextureId !== undefined || p.skipEmitter !== undefined
      ? {
          extensions: {
            ...(p.surfaceTextureId !== undefined ? { surfaceTextureId: p.surfaceTextureId } : {}),
            ...(p.skipEmitter !== undefined ? { skipEmitter: p.skipEmitter } : {}),
          },
        }
      : {}),
  };
  return { name, three, core };
}

const PAIRS: Pair[] = [
  makePair('plain opaque diffuse', { baseColor: [0.5, 0.6, 0.7], roughness: 0.4 }),
  makePair('emissive lamp', {
    baseColor: [0.1, 0.1, 0.1],
    emissive: [0.8, 0.4, 0.2],
    emissiveIntensity: 5,
  }),
  makePair('emissive, intensity 0 (not an emitter)', {
    emissive: [0.8, 0.4, 0.2],
    emissiveIntensity: 0,
  }),
  makePair('emissive black (not an emitter)', {
    emissive: [0, 0, 0],
    emissiveIntensity: 3,
  }),
  makePair('faintly transmissive (tints, below emitter threshold)', {
    baseColor: [0.9, 0.95, 1.0],
    transmission: 0.05,
    attenuationColor: [0.7, 0.85, 0.95],
  }),
  makePair('clear glass (transmissive emitter eligible)', {
    baseColor: [0.95, 0.97, 1.0],
    transmission: 0.9,
    attenuationColor: [0.6, 0.8, 0.95],
    attenuationDistance: 2.5,
    thickness: 0.3,
  }),
  makePair('transmissive but no attenuation color', {
    baseColor: [0.4, 0.7, 0.9],
    transmission: 0.8,
  }),
  makePair('transmissive, skipEmitter set', {
    baseColor: [1, 1, 1],
    transmission: 0.6,
    attenuationColor: [0.5, 0.5, 0.5],
    skipEmitter: true,
  }),
  makePair('transmissive with surfaceTextureId', {
    baseColor: [0.8, 0.8, 0.8],
    transmission: 0.7,
    attenuationColor: [0.9, 0.7, 0.6],
    surfaceTextureId: 3,
  }),
  makePair('thickness set but attenuationDistance absent (no Beer)', {
    transmission: 0.5,
    attenuationColor: [0.5, 0.6, 0.7],
    thickness: 0.4,
  }),
  makePair('attenuationDistance Infinity (no Beer)', {
    transmission: 0.5,
    attenuationColor: [0.5, 0.6, 0.7],
    thickness: 0.4,
    attenuationDistance: Infinity,
  }),
];

describe('materialSpecEmissiveLe ≡ materialEmissiveLe (THREE)', () => {
  for (const { name, three, core } of PAIRS) {
    it(name, () => {
      const oracle = oracleMaterialEmissiveLe(three);
      const got = materialSpecEmissiveLe(core);
      if (oracle == null) {
        expect(got).toBeNull();
      } else {
        expect(got).not.toBeNull();
        expect(got![0]).toBeCloseTo(oracle[0], 6);
        expect(got![1]).toBeCloseTo(oracle[1], 6);
        expect(got![2]).toBeCloseTo(oracle[2], 6);
      }
    });
  }
});

describe('materialSpecTriColor ≡ resolveTriColor (THREE)', () => {
  for (const { name, three, core } of PAIRS) {
    it(`${name} — raw (applyBeer=false)`, () => {
      const oracle = oracleResolveTriColor(three, false);
      const got = materialSpecTriColor(core, false);
      expect(got[0]).toBeCloseTo(oracle.r, 6);
      expect(got[1]).toBeCloseTo(oracle.g, 6);
      expect(got[2]).toBeCloseTo(oracle.b, 6);
      // The byte the bvhIndex.w lane actually carries must match too.
      expect(rgb8(got)).toEqual(rgb8(oracle));
    });
    it(`${name} — Beer (applyBeer=true)`, () => {
      const oracle = oracleResolveTriColor(three, true);
      const got = materialSpecTriColor(core, true);
      expect(got[0]).toBeCloseTo(oracle.r, 6);
      expect(got[1]).toBeCloseTo(oracle.g, 6);
      expect(got[2]).toBeCloseTo(oracle.b, 6);
      // The byte the bvh_beer lane actually carries (clamped) must match too.
      expect(rgb8Clamped(got)).toEqual(rgb8Clamped([oracle.r, oracle.g, oracle.b]));
    });
  }
});

describe('classifyTriangleEmitterCore ≡ classifyTriangleEmitter (THREE)', () => {
  // A spread of normals + light directions so the sunDot branch is exercised
  // both above and below the 0.05 reject threshold.
  const normals = [
    { x: 0, y: 1, z: 0 },
    { x: 1, y: 0, z: 0 },
    { x: 0.577, y: 0.577, z: 0.577 },
  ];
  const lightDirs = [
    new THREE.Vector3(0, 1, 0),      // parallel to first normal → sunDot 1
    new THREE.Vector3(1, 0, 0),      // orthogonal to first normal → sunDot 0 (reject)
    new THREE.Vector3(0.267, 0.535, 0.802), // oblique
  ];
  const intensities = [3.0, 1.0, 7.5];

  for (const { name, three, core } of PAIRS) {
    for (let ni = 0; ni < normals.length; ni++) {
      for (let li = 0; li < lightDirs.length; li++) {
        const intensity = intensities[li]!;
        it(`${name} — normal#${ni} light#${li}`, () => {
          const oracle = oracleClassifyTriangleEmitter(three, normals[ni]!, lightDirs[li]!, intensity);
          const got = classifyTriangleEmitterCore(
            core,
            normals[ni]!,
            { x: lightDirs[li]!.x, y: lightDirs[li]!.y, z: lightDirs[li]!.z },
            intensity,
          );
          if (oracle == null) {
            expect(got).toBeNull();
          } else {
            expect(got).not.toBeNull();
            expect(got!.color[0]).toBeCloseTo(oracle.color[0], 6);
            expect(got!.color[1]).toBeCloseTo(oracle.color[1], 6);
            expect(got!.color[2]).toBeCloseTo(oracle.color[2], 6);
            expect(got!.intensity).toBeCloseTo(oracle.intensity, 6);
          }
        });
      }
    }
  }
});

describe('materialSpecSurfaceTextureId ≡ userData.surfaceTextureId & 0x7', () => {
  for (const { name, three, core } of PAIRS) {
    it(name, () => {
      const raw = (three.userData as { surfaceTextureId?: number }).surfaceTextureId;
      const oracle = (typeof raw === 'number' ? raw : 0) & 0x7;
      expect(materialSpecSurfaceTextureId(core)).toBe(oracle);
    });
  }
  it('masks to 3 bits (id 13 → 5)', () => {
    const core: MaterialSpec = {
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      extensions: { surfaceTextureId: 13 },
    };
    expect(materialSpecSurfaceTextureId(core)).toBe(13 & 0x7);
  });
});

describe('materialSpecSkipEmitter ≡ userData.skipEmitter === true', () => {
  for (const { name, three, core } of PAIRS) {
    it(name, () => {
      const oracle = (three.userData as { skipEmitter?: boolean }).skipEmitter === true;
      expect(materialSpecSkipEmitter(core)).toBe(oracle);
    });
  }
  it('only the literal true skips (truthy non-true does not)', () => {
    const core: MaterialSpec = {
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      extensions: { skipEmitter: 1 }, // truthy non-true → must NOT skip
    };
    expect(materialSpecSkipEmitter(core)).toBe(false);
  });
});

describe('applyBeerLambertColor ≡ applyBeerLambert (THREE), unit-level', () => {
  const col: [number, number, number] = [0.6, 0.8, 0.95];
  const cases: Array<{ t: number | undefined; d: number | undefined; label: string }> = [
    { t: 0.3, d: 2.5, label: 'finite t/d → power law' },
    { t: undefined, d: 2.5, label: 'thickness undefined → passthrough' },
    { t: 0.3, d: undefined, label: 'attDist undefined → passthrough' },
    { t: 0, d: 2.5, label: 'thickness 0 → passthrough' },
    { t: 0.3, d: 0, label: 'attDist 0 → passthrough' },
    { t: Infinity, d: 2.5, label: 'thickness Infinity → passthrough' },
    { t: 0.3, d: Infinity, label: 'attDist Infinity → passthrough' },
    { t: 1.0, d: 0.5, label: 'k=2 strong absorption' },
  ];
  for (const { t, d, label } of cases) {
    it(label, () => {
      const oracle = oracleApplyBeerLambert(new THREE.Color(...col), t, d);
      const got = applyBeerLambertColor(col, t, d);
      expect(got[0]).toBeCloseTo(oracle.r, 6);
      expect(got[1]).toBeCloseTo(oracle.g, 6);
      expect(got[2]).toBeCloseTo(oracle.b, 6);
    });
  }
  it('floors each channel at 1e-6 before the power (matches THREE)', () => {
    const got = applyBeerLambertColor([0, 0, 0], 1, 1);
    const oracle = oracleApplyBeerLambert(new THREE.Color(0, 0, 0), 1, 1);
    expect(got[0]).toBeCloseTo(oracle.r, 6);
  });
});

describe('MATERIAL_DEFAULT_TRI_COLOR matches the THREE warm-gray fallback', () => {
  it('equals (0.6, 0.58, 0.55)', () => {
    const oracle = new THREE.Color(0.6, 0.58, 0.55);
    expect(MATERIAL_DEFAULT_TRI_COLOR[0]).toBeCloseTo(oracle.r, 6);
    expect(MATERIAL_DEFAULT_TRI_COLOR[1]).toBeCloseTo(oracle.g, 6);
    expect(MATERIAL_DEFAULT_TRI_COLOR[2]).toBeCloseTo(oracle.b, 6);
  });
  // The warm-gray default constants the THREE byte-pack uses (153,148,140) are
  // the rgb8 of (0.6,0.58,0.55) — confirm the core default round-trips to them.
  it('rgb8 of the default equals the THREE warm-gray byte constants', () => {
    expect(rgb8([...MATERIAL_DEFAULT_TRI_COLOR])).toEqual([
      WARM_GRAY_DEFAULT_R, WARM_GRAY_DEFAULT_G, WARM_GRAY_DEFAULT_B,
    ]);
  });
});
