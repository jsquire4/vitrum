/**
 * material.ts — THREE.js material → @vitrum/core Material converter.
 *
 * Handles MeshStandardMaterial and MeshPhysicalMaterial, including all
 * physical-material extensions (transmission, clearcoat, sheen, iridescence).
 * Other material types (ShaderMaterial, etc.) are rejected at the mesh level
 * in mesh.ts before this converter is called.
 */

import type * as THREE from 'three';
import type { Material, Vec3, SpectralCurve, ThinFilmStack, SurfaceAbsorptionLayer } from '@vitrum/core';
import { VITRUM_USER_DATA_KEYS as K } from './userDataKeys.js';

/** THREE.MeshPhysicalMaterial default index of refraction. Used as the
 *  "no-op" guard so callers that did not customize IOR don't generate
 *  noisy override stamps on the vitrum side. */
const THREE_PHYSICAL_DEFAULT_IOR = 1.5;

// ────────────────────────────────────────────────────────────────────────────
// Internal type aliases
// ────────────────────────────────────────────────────────────────────────────

export type ThreeStdMat = THREE.MeshStandardMaterial;
export type ThreePhysMat = THREE.MeshPhysicalMaterial;

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

export function colorToVec3(c: THREE.Color): Vec3 {
  return [c.r, c.g, c.b];
}

function isPhysical(m: ThreeStdMat): m is ThreePhysMat {
  return (m as ThreePhysMat).isMeshPhysicalMaterial === true;
}

// ────────────────────────────────────────────────────────────────────────────
// Material converter
// ────────────────────────────────────────────────────────────────────────────

export function convertMaterial(m: ThreeStdMat): Material {
  const base: Material = {
    baseColor: colorToVec3(m.color),
    roughness: m.roughness,
    metallic: m.metalness,
  };

  const emR = m.emissive.r * m.emissiveIntensity;
  const emG = m.emissive.g * m.emissiveIntensity;
  const emB = m.emissive.b * m.emissiveIntensity;
  if (emR !== 0 || emG !== 0 || emB !== 0) {
    base.emissive = [emR, emG, emB];
    base.emissiveIntensity = m.emissiveIntensity;
  }

  if (m.map != null) base.baseColorMap = m.map;
  if (m.normalMap != null) {
    base.normalMap = m.normalMap;
    base.normalScale = m.normalScale.x;
  }
  if (m.roughnessMap != null) base.roughnessMap = m.roughnessMap;
  if (m.metalnessMap != null) base.metallicMap = m.metalnessMap;
  if (m.emissiveMap != null) base.emissiveMap = m.emissiveMap;
  if (m.alphaMap != null) base.alphaMap = m.alphaMap;

  if (!isPhysical(m)) return base;

  const p = m;
  if (p.transmission !== 0) base.transmission = p.transmission;
  if (p.ior !== THREE_PHYSICAL_DEFAULT_IOR) base.ior = p.ior;

  if (p.attenuationDistance !== Infinity) {
    base.attenuationColor = colorToVec3(p.attenuationColor);
    base.attenuationDistance = p.attenuationDistance;
  }

  if (p.thickness !== 0) base.thickness = p.thickness;
  if (p.transmissionMap != null) base.transmissionMap = p.transmissionMap;

  if (p.sheen !== 0) {
    base.sheen = p.sheen;
    base.sheenColor = colorToVec3(p.sheenColor);
    base.sheenRoughness = p.sheenRoughness;
  }

  if (p.clearcoat !== 0) {
    base.clearcoat = p.clearcoat;
    base.clearcoatRoughness = p.clearcoatRoughness;
  }

  if (p.iridescence !== 0) {
    base.iridescence = p.iridescence;
    // THREE uses iridescenceIOR (caps); core uses iridescenceIor (camelCase).
    base.iridescenceIor = p.iridescenceIOR;
    base.iridescenceThicknessRange = p.iridescenceThicknessRange;
  }

  // ── userData.vitrum* stamps (RFE-06..08 / RFE-03) ──────────────────────────
  // The host stamps these on THREE materials so backends can read them via the
  // vitrum.Material contract. We project them unconditionally here; each guard
  // preserves exactOptionalPropertyTypes (no field set = field absent).
  const ud = (p.userData ?? {}) as Record<string, unknown>;

  // RFE-06 (Sprint 8 — chromatic dispersion, Abbe number)
  const rawDispersion = ud[K.DISPERSION_ABBE];
  if (typeof rawDispersion === 'number') {
    base.dispersionAbbeNumber = rawDispersion;
  }

  // RFE-07 (Sprint 7 — volume scattering + HG anisotropy)
  const rawScatCoeff = ud[K.SCATTERING_COEFF];
  if (typeof rawScatCoeff === 'number') {
    base.scatteringCoefficient = rawScatCoeff;
  }
  const rawScatRgb = ud[K.SCATTERING_RGB];
  if (
    Array.isArray(rawScatRgb) &&
    (rawScatRgb as unknown[]).length === 3
  ) {
    base.scatteringCoefficientRGB = rawScatRgb as unknown as Vec3;
  }
  const rawScatAniso = ud[K.SCATTERING_ANISO];
  if (typeof rawScatAniso === 'number') {
    base.scatteringAnisotropy = rawScatAniso;
  }

  // RFE-08 (Sprint 12 — hero-wavelength spectral attenuation curve)
  // The host stamp is a SpectralCurve object: { wavelengthStart, wavelengthEnd, values }.
  // We accept both a full SpectralCurve object and, for back-compat, a plain
  // Float32Array treated as 380–780 nm / 81-sample curve.
  const rawSpectral = ud[K.SPECTRAL_ATTEN];
  if (rawSpectral != null) {
    if (
      typeof rawSpectral === 'object' &&
      !Array.isArray(rawSpectral) &&
      'wavelengthStart' in (rawSpectral as object) &&
      'wavelengthEnd' in (rawSpectral as object) &&
      'values' in (rawSpectral as object)
    ) {
      base.spectralAttenuation = rawSpectral as SpectralCurve;
    } else if (rawSpectral instanceof Float32Array && rawSpectral.length >= 3) {
      // Raw Float32Array fallback: assume 380–780 nm range.
      // Deprecated path — kept for back-compat with hosts that stamp the
      // userData curve as a bare Float32Array. New code should write
      // `{ wavelengthStart, wavelengthEnd, values: Float32Array }`.
      console.warn(
        '[vitrum/three-bindings] SpectralCurve as Float32Array is deprecated. ' +
          'Use { wavelengthStart, wavelengthEnd, values: Float32Array } instead. ' +
          'Scheduled for removal in Phase 7 / Sprint 1.',
      );
      base.spectralAttenuation = {
        wavelengthStart: 380,
        wavelengthEnd: 780,
        values: rawSpectral,
      };
    }
  }

  // RFE-08 (Sprint 12 — multi-layer thin-film stack)
  const rawThinFilm = ud[K.THIN_FILM_STACK];
  if (rawThinFilm != null && typeof rawThinFilm === 'object' && !Array.isArray(rawThinFilm)) {
    base.thinFilmStack = rawThinFilm as ThinFilmStack;
  }

  // RFE-03 (Sprint X — per-face surface absorption layers)
  const rawFront = ud[K.FRONT_LAYER];
  if (rawFront != null && typeof rawFront === 'object' && !Array.isArray(rawFront)) {
    base.frontLayer = rawFront as SurfaceAbsorptionLayer;
  }
  const rawBack = ud[K.BACK_LAYER];
  if (rawBack != null && typeof rawBack === 'object' && !Array.isArray(rawBack)) {
    base.backLayer = rawBack as SurfaceAbsorptionLayer;
  }

  return base;
}
