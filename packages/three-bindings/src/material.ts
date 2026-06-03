/**
 * material.ts — THREE.js material → @vitrum/core Material converter.
 *
 * Handles MeshStandardMaterial and MeshPhysicalMaterial, including all
 * physical-material extensions (transmission, clearcoat, sheen, iridescence).
 * MeshBasicMaterial is accepted via convertBasicMaterial — it is rendered
 * as a flat self-lit surface (emissive = color, roughness = 1) so unlit
 * overlay meshes appear consistent under PT/walkaround.
 * Other material types (ShaderMaterial, etc.) are rejected at the mesh level
 * in mesh.ts before this converter is called.
 */

import type * as THREE from 'three';
import type { MaterialSpec, Vec2, Vec3, SpectralCurve, ThinFilmStack, SurfaceAbsorptionLayer, TextureRef, UvTransform } from '@vitrum/core';
import { VITRUM_USER_DATA_KEYS as K } from './userDataKeys.js';

/**
 * Mutable staging shape used internally by the converters to assemble the
 * field bag before freezing into a `MaterialSpec`. `MaterialSpec` itself is
 * readonly per W3-D1, but the builders construct via conditional assignment
 * and then narrow on return.
 */
type MaterialDraft = {
  -readonly [P in keyof MaterialSpec]: MaterialSpec[P];
};

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

/**
 * Wrap a THREE.Texture as a structured `TextureRef`, projecting THREE's
 * `offset`/`repeat`/`rotation` into a `KHR_texture_transform` `UvTransform` and
 * `texture.channel` into `texCoord` (which mesh UV set). Identity transform and
 * channel 0 are omitted (exactOptionalPropertyTypes-friendly).
 */
export function toTextureRef(tex: THREE.Texture): TextureRef {
  const t: { offset?: Vec2; scale?: Vec2; rotation?: number } = {};
  if (tex.offset.x !== 0 || tex.offset.y !== 0) t.offset = [tex.offset.x, tex.offset.y];
  if (tex.repeat.x !== 1 || tex.repeat.y !== 1) t.scale = [tex.repeat.x, tex.repeat.y];
  if (tex.rotation !== 0) t.rotation = tex.rotation;
  const ref: { handle: unknown; texCoord?: number; transform?: UvTransform } = { handle: tex };
  const ch = (tex as { channel?: number }).channel;
  if (typeof ch === 'number' && ch !== 0) ref.texCoord = ch;
  if (t.offset !== undefined || t.scale !== undefined || t.rotation !== undefined) ref.transform = t;
  return ref;
}

// ────────────────────────────────────────────────────────────────────────────
// Material converter
// ────────────────────────────────────────────────────────────────────────────

export function convertMaterial(m: ThreeStdMat): MaterialSpec {
  const base: MaterialDraft = {
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

  if (m.map != null) base.baseColorMap = toTextureRef(m.map);
  if (m.normalMap != null) {
    base.normalMap = toTextureRef(m.normalMap);
    base.normalScale = m.normalScale.x;
  }
  if (m.roughnessMap != null) base.roughnessMap = toTextureRef(m.roughnessMap);
  if (m.metalnessMap != null) base.metallicMap = toTextureRef(m.metalnessMap);
  if (m.emissiveMap != null) base.emissiveMap = toTextureRef(m.emissiveMap);
  if (m.alphaMap != null) base.alphaMap = toTextureRef(m.alphaMap);
  if (m.aoMap != null) {
    base.aoMap = toTextureRef(m.aoMap);
    if (m.aoMapIntensity !== 1) base.aoMapIntensity = m.aoMapIntensity;
  }

  // Alpha mode (glTF semantics): alphaTest > 0 → mask; else transparent → blend.
  if (m.alphaTest > 0) {
    base.alphaMode = 'mask';
    base.alphaCutoff = m.alphaTest;
  } else if (m.transparent) {
    base.alphaMode = 'blend';
  }
  if (m.opacity !== 1) base.opacity = m.opacity;

  if (!isPhysical(m)) return base;

  const p = m;
  if (p.transmission !== 0) base.transmission = p.transmission;
  if (p.ior !== THREE_PHYSICAL_DEFAULT_IOR) base.ior = p.ior;

  if (p.attenuationDistance !== Infinity) {
    base.attenuationColor = colorToVec3(p.attenuationColor);
    base.attenuationDistance = p.attenuationDistance;
  }

  if (p.thickness !== 0) base.thickness = p.thickness;
  if (p.transmissionMap != null) base.transmissionMap = toTextureRef(p.transmissionMap);

  if (p.sheen !== 0) {
    base.sheen = p.sheen;
    base.sheenColor = colorToVec3(p.sheenColor);
    base.sheenRoughness = p.sheenRoughness;
    if (p.sheenColorMap != null) base.sheenColorMap = toTextureRef(p.sheenColorMap);
    if (p.sheenRoughnessMap != null) base.sheenRoughnessMap = toTextureRef(p.sheenRoughnessMap);
  }

  if (p.clearcoat !== 0) {
    base.clearcoat = p.clearcoat;
    base.clearcoatRoughness = p.clearcoatRoughness;
    if (p.clearcoatMap != null) base.clearcoatMap = toTextureRef(p.clearcoatMap);
    if (p.clearcoatRoughnessMap != null) base.clearcoatRoughnessMap = toTextureRef(p.clearcoatRoughnessMap);
    if (p.clearcoatNormalMap != null) {
      base.clearcoatNormalMap = toTextureRef(p.clearcoatNormalMap);
      base.clearcoatNormalScale = p.clearcoatNormalScale.x;
    }
  }

  if (p.iridescence !== 0) {
    base.iridescence = p.iridescence;
    // THREE uses iridescenceIOR (caps); core uses iridescenceIor (camelCase).
    base.iridescenceIor = p.iridescenceIOR;
    base.iridescenceThicknessRange = p.iridescenceThicknessRange;
    if (p.iridescenceMap != null) base.iridescenceMap = toTextureRef(p.iridescenceMap);
    if (p.iridescenceThicknessMap != null) base.iridescenceThicknessMap = toTextureRef(p.iridescenceThicknessMap);
  }

  // Gap 5 (stainedGlass audit 2026-05-12) — anisotropy is set DIRECTLY on the
  // THREE MeshPhysicalMaterial (not via userData) by the baking pipeline for
  // ripple/waterglass cells. Read both fields off the live material object.
  // Mirror the iridescence pattern: only capture when non-zero so default-zero
  // THREE materials don't populate the vitrum Material with phantom fields.
  if (p.anisotropy !== 0) {
    base.anisotropy = p.anisotropy;
    // Always capture rotation alongside anisotropy; 0 rotation is meaningful.
    base.anisotropyRotation = p.anisotropyRotation;
    if (p.anisotropyMap != null) base.anisotropyMap = toTextureRef(p.anisotropyMap);
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
  // Only the full SpectralCurve shape is accepted:
  //   { wavelengthStart: number, wavelengthEnd: number, values: Float32Array }
  // The deprecated bare Float32Array path was removed in the 2026-05-11 sweep
  // (pre-alpha; no external consumers). See Foundations Item #35 / D11.
  const rawSpectral = ud[K.SPECTRAL_ATTEN];
  if (
    rawSpectral != null &&
    typeof rawSpectral === 'object' &&
    !Array.isArray(rawSpectral) &&
    'wavelengthStart' in (rawSpectral) &&
    'wavelengthEnd' in (rawSpectral) &&
    'values' in (rawSpectral)
  ) {
    base.spectralAttenuation = rawSpectral as SpectralCurve;
  }

  // RFE-08 (Sprint 12 — multi-layer thin-film stack)
  const rawThinFilm = ud[K.THIN_FILM_STACK];
  if (rawThinFilm != null && typeof rawThinFilm === 'object' && !Array.isArray(rawThinFilm)) {
    base.thinFilmStack = rawThinFilm as ThinFilmStack;
  }

  // RFE-03 (Sprint 14 — per-face surface absorption layers)
  const rawFront = ud[K.FRONT_LAYER];
  if (rawFront != null && typeof rawFront === 'object' && !Array.isArray(rawFront)) {
    base.frontLayer = rawFront as SurfaceAbsorptionLayer;
  }
  const rawBack = ud[K.BACK_LAYER];
  if (rawBack != null && typeof rawBack === 'object' && !Array.isArray(rawBack)) {
    base.backLayer = rawBack as SurfaceAbsorptionLayer;
  }

  // RFE-10 dichroic addendum (PHY.1 — 2026-05-12). The stainedGlass dichroic
  // body baker emits two 256×1 RGBA-float DataTextures pre-convolving the
  // TMM × CIE 1931 standard observer. Forward both through
  // `Material.extensions.dichroicLUTs` for raster backends that bind them
  // directly; PT backends may continue to evaluate the TMM in-shader from
  // `thinFilmStack` and ignore the LUT. Core never inspects `extensions`,
  // matching the existing escape-hatch contract.
  const rawDichroicR = ud[K.DICHROIC_REFLECTANCE_LUT];
  const rawDichroicT = ud[K.DICHROIC_TRANSMITTANCE_LUT];
  if (rawDichroicR != null || rawDichroicT != null) {
    const extensions = { ...(base.extensions ?? {}) } as Record<string, unknown>;
    extensions['dichroicLUTs'] = {
      reflectance: rawDichroicR,
      transmittance: rawDichroicT,
    };
    base.extensions = extensions;
  }

  return base;
}

// ────────────────────────────────────────────────────────────────────────────
// MeshBasicMaterial converter
// ────────────────────────────────────────────────────────────────────────────

/**
 * Convert a THREE.MeshBasicMaterial into a flat self-lit vitrum Material.
 * MeshBasicMaterial is unlit in three.js; under PT/walkaround we approximate
 * the same look by setting `emissive = color, emissiveIntensity = 1,
 * roughness = 1, metallic = 0`. This keeps overlay/preview meshes
 * (panel-mount preview, debug grids) readable regardless of scene lighting
 * without requiring a separate unlit pass.
 *
 * Maps that the basic material carries (`map`, `alphaMap`) are forwarded;
 * other PBR maps are not relevant for the basic material type.
 */
export function convertBasicMaterial(m: THREE.MeshBasicMaterial): MaterialSpec {
  const c = colorToVec3(m.color);
  const base: MaterialDraft = {
    baseColor: c,
    roughness: 1.0,
    metallic: 0.0,
    emissive: c,
    emissiveIntensity: 1.0,
  };
  if (m.map != null) base.baseColorMap = toTextureRef(m.map);
  if (m.alphaMap != null) base.alphaMap = toTextureRef(m.alphaMap);
  return base;
}

// ────────────────────────────────────────────────────────────────────────────
// Cross-engine PBR-scalar extraction (P2-6.1)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Minimal PBR scalar bag that every walkaround/DDGI/path-tracer engine ends up
 * needing on the host side when building GPU material buffers. Captures the
 * "extract THREE PBR fields with `?? default` semantics" shape that was
 * previously duplicated across engines.
 *
 * Use {@link extractThreePbrScalars} to fill this from a THREE material with
 * caller-supplied defaults. Each consumer maps from the struct into its own
 * GPU layout — there is no shared packing format, only a shared extraction.
 *
 * Fields without a meaningful default on the THREE side are wrapped optional
 * (`attenuationColor`, `attenuationDistance`, `thickness`) — engines that want
 * a fallback should `??` at the use site.
 */
export interface PbrScalars {
  readonly baseColor: Vec3;
  readonly emissive: Vec3;
  readonly emissiveIntensity: number;
  readonly roughness: number;
  readonly metallic: number;
  readonly transmission: number;
  readonly ior: number;
  readonly attenuationColor: Vec3;
  readonly attenuationDistance: number;
  readonly thickness: number;
}

export interface PbrDefaults {
  readonly baseColor: Vec3;
  readonly emissive: Vec3;
  readonly emissiveIntensity: number;
  readonly roughness: number;
  readonly metallic: number;
  readonly transmission: number;
  readonly ior: number;
  readonly attenuationColor: Vec3;
  readonly attenuationDistance: number;
  readonly thickness: number;
}

const PBR_DEFAULTS: PbrDefaults = {
  baseColor: [1, 1, 1],
  emissive: [0, 0, 0],
  emissiveIntensity: 1,
  roughness: 0.5,
  metallic: 0,
  transmission: 0,
  ior: THREE_PHYSICAL_DEFAULT_IOR,
  attenuationColor: [1, 1, 1],
  attenuationDistance: Infinity,
  thickness: 0,
};

/**
 * Extract PBR scalars from a THREE material with caller-supplied defaults
 * for any missing fields. Handles MeshStandardMaterial vs
 * MeshPhysicalMaterial type narrowing and the various optional fields that
 * only physical materials carry (transmission, ior, attenuation*, thickness).
 *
 * Pass {@link PBR_DEFAULTS_DEFAULT} (or omit) to use the
 * library-standard defaults; pass a custom defaults bag to keep
 * byte-equivalent behavior with a legacy engine that diverged from the
 * standard defaults.
 */
export function extractThreePbrScalars(
  mat: THREE.Material,
  defaults: Partial<PbrDefaults> = {},
): PbrScalars {
  const d = { ...PBR_DEFAULTS, ...defaults };
  const stdMat = mat as ThreeStdMat;
  const physMat = mat as ThreePhysMat;
  const color = stdMat.color;
  const emissive = stdMat.emissive;
  return {
    baseColor: color ? colorToVec3(color) : d.baseColor,
    emissive: emissive ? colorToVec3(emissive) : d.emissive,
    emissiveIntensity: stdMat.emissiveIntensity ?? d.emissiveIntensity,
    roughness: stdMat.roughness ?? d.roughness,
    metallic: stdMat.metalness ?? d.metallic,
    transmission: physMat.transmission ?? d.transmission,
    ior: physMat.ior ?? d.ior,
    attenuationColor: physMat.attenuationColor
      ? colorToVec3(physMat.attenuationColor)
      : d.attenuationColor,
    attenuationDistance: physMat.attenuationDistance ?? d.attenuationDistance,
    thickness: physMat.thickness ?? d.thickness,
  };
}

/** Library-standard PBR defaults — exposed for engines that want to opt
 *  into the canonical set rather than maintain their own. */
export const PBR_DEFAULTS_DEFAULT: PbrDefaults = PBR_DEFAULTS;
