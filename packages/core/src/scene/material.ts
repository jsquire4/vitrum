// Scene description — backend-agnostic.
//
// Material types: the core PBR `MaterialSpec`, spectral curves (RFE-01),
// layered BSDF surface absorption (RFE-03), and multi-layer thin-film stacks
// (RFE-04), plus the opaque `TextureRef` handle.

import type { Vec2, Vec3 } from './math.js';

// ────────────────────────────────────────────────────────────────────────────
// Spectral rendering types (RFE-01)
// ────────────────────────────────────────────────────────────────────────────

/**
 * A spectral reflectance or attenuation curve sampled at uniformly spaced
 * wavelengths across the visible range (or a portion of it).
 *
 * `wavelengthStart` / `wavelengthEnd` are in nanometers (e.g. 380 / 700).
 * `values` are per-sample coefficients (units depend on usage context —
 * for spectralAttenuation: μ(λ) in inverse scene-length-units matching
 * attenuationDistance). Must have ≥ 3 entries; the engine linearly
 * interpolates between samples.
 *
 * Reference: Wilkie et al., "Hero Wavelength Spectral Sampling," EGSR 2014.
 */
export interface SpectralCurve {
  readonly wavelengthStart: number;  // nm, e.g. 380
  readonly wavelengthEnd: number;    // nm, e.g. 700
  readonly values: Float32Array;     // μ(λ) in units matching attenuationDistance; length ≥ 3
}

// ────────────────────────────────────────────────────────────────────────────
// Layered BSDF types (RFE-03)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Describes a thin absorbing layer applied to one face of a dielectric surface.
 * Models a surface-fused coating or diffusion-tinted region that modulates
 * transmitted and reflected radiance independently from the bulk material.
 *
 * The layer is treated as an infinitesimally thin absorber applied BEFORE
 * the bulk BSDF is evaluated — incoming radiance is multiplied by
 * `transmission` before entering the bulk, and outgoing radiance is multiplied
 * by `transmission` before exiting. This is the simplified (non-multiple-
 * scattering) form of Belcour 2018's atomic decomposition.
 *
 * Reference: Belcour, "Efficient Rendering of Layered Materials using an
 * Atomic Decomposition with Statistical Operators," ACM TOG (SIGGRAPH 2018).
 */
export interface SurfaceAbsorptionLayer {
  /**
   * Per-channel (RGB) transmission of the thin absorbing layer.
   * [1, 1, 1] = no absorption (identity layer).
   * [0, 0, 0] = fully opaque mask.
   * Each component ∈ [0, 1].
   */
  readonly transmission: Vec3;

  /**
   * Optional roughness override for this face's surface.
   * When set, replaces Material.roughness for this face only.
   * Range: [0, 1].
   */
  readonly roughness?: number;

  /**
   * Optional normal map texture for this face only.
   * Useful when the two faces have different surface treatments.
   */
  readonly normalMap?: TextureRef;
  readonly normalScale?: number;
}

// ────────────────────────────────────────────────────────────────────────────
// Multi-layer thin-film types (RFE-04)
// ────────────────────────────────────────────────────────────────────────────

/**
 * A single layer in a thin-film optical stack.
 * The stack is ordered from topmost (air-adjacent) layer to
 * bottom (substrate-adjacent) layer.
 *
 * Reference: Born & Wolf, "Principles of Optics" (1959/1999);
 * Macleod, "Thin-Film Optical Filters," 4th ed. (2010) — Abeles TMM formalism.
 */
export interface ThinFilmLayer {
  /** Refractive index of this layer (real part). */
  readonly ior: number;
  /**
   * Extinction coefficient (imaginary part of complex IOR).
   * 0 for dielectric layers; > 0 for lossy or metallic layers.
   */
  readonly extinctionCoefficient?: number;
  /** Physical layer thickness in nanometers. */
  readonly thicknessNm: number;
}

/**
 * A multi-layer thin-film stack evaluated via the Transfer Matrix Method
 * (Abeles formalism). The engine computes per-wavelength reflectance and
 * transmittance by multiplying the 2×2 characteristic matrices of each layer.
 * The resulting spectral R(λ) and T(λ) are convolved with CIE CMFs to produce
 * the RGB BSDF weight used at each shade point.
 *
 * When `thinFilmStack` is present on a Material, it overrides the single-layer
 * iridescence model (iridescence / iridescenceIor / iridescenceThicknessRange).
 */
export interface ThinFilmStack {
  /**
   * Ordered array of thin-film layers, topmost first (closest to incident
   * medium, usually air). The substrate IOR is taken from Material.ior.
   * Minimum: 1 layer. Backends may cap for performance (e.g., 64 layers).
   */
  readonly layers: ReadonlyArray<ThinFilmLayer>;

  /**
   * IOR of the incident medium (the medium the ray arrives from).
   * Typically 1.0 (air). Default: 1.0.
   */
  readonly incidentIor?: number;

  /**
   * If true, evaluate TMM per viewing angle (cos θ from Snell's law through
   * each layer). This produces the correct viewing-angle-dependent spectral
   * shift. If false, evaluate at normal incidence only (faster, less accurate
   * for grazing angles). Default: true.
   */
  readonly angleDependent?: boolean;
}

// ────────────────────────────────────────────────────────────────────────────
// Material
// ────────────────────────────────────────────────────────────────────────────

/** Generic PBR material — superset of the standard PBR fields with optional
 *  Disney-BSDF lobes for backends that support them. The `extensions` field
 *  is the escape hatch: backends can read backend-specific data from it
 *  without polluting the core type (e.g., normalMap-perturbed shadow ray
 *  parameters, the Phase 4 contribution).
 *
 *  **The contract type is fully immutable.** Every field is `readonly` and
 *  primitives expose `readonly material: MaterialSpec`. To change any material
 *  field after `setScene`, hosts MUST construct a fresh `MaterialSpec` (clone
 *  + override via spread) and call `engine.updatePrimitive` (or
 *  `updateEmitter`) with the new value. Backends do not detect in-place
 *  mutations — this is enforced by the type system, not just by convention
 *  (W3-D1, 2026-05-17).
 *
 *  Texture handles are opaque to core. The scene-binding layer (e.g.,
 *  @vitrum/three-bindings) is responsible for converting host textures to
 *  whatever format the backend expects (typed arrays for upload, GPU texture
 *  handles, etc.). Core just routes them through. */
export interface MaterialSpec {
  // ── Base PBR ────────────────────────────────────────────────────────────
  readonly baseColor: Vec3;
  readonly roughness: number;            // 0 = mirror, 1 = matte
  readonly metallic: number;             // 0 = dielectric, 1 = pure metal
  readonly emissive?: Vec3;
  readonly emissiveIntensity?: number;

  // ── Transmission / refraction ───────────────────────────────────────────
  readonly transmission?: number;        // 0 = opaque, 1 = fully transparent
  readonly ior?: number;                  // index of refraction
  readonly attenuationColor?: Vec3;       // Beer-Lambert: color the medium absorbs to
  readonly attenuationDistance?: number;  // Beer-Lambert: depth at which attenuationColor reached
  readonly thickness?: number;            // Beer-Lambert: actual slab thickness

  // ── Texture maps (opaque handles, see TextureRef) ───────────────────────
  readonly baseColorMap?: TextureRef;
  readonly normalMap?: TextureRef;
  readonly normalScale?: number;
  readonly roughnessMap?: TextureRef;
  readonly metallicMap?: TextureRef;
  readonly transmissionMap?: TextureRef;
  readonly emissiveMap?: TextureRef;
  readonly alphaMap?: TextureRef;

  // ── Disney BSDF extensions (optional) ───────────────────────────────────
  readonly sheen?: number;
  readonly sheenColor?: Vec3;
  readonly sheenRoughness?: number;
  readonly clearcoat?: number;
  readonly clearcoatRoughness?: number;
  readonly iridescence?: number;
  readonly iridescenceIor?: number;
  readonly iridescenceThicknessRange?: Vec2;

  // ── Spectral attenuation (RFE-01) ──────────────────────────────────────
  /**
   * Spectral attenuation coefficient table, sampled at uniformly spaced
   * wavelengths. Each entry is μ(λ) in inverse scene-length-units (matching
   * attenuationDistance units).
   *
   * When present, the engine uses per-wavelength Beer-Lambert instead of the
   * RGB approximation. Length must be >= 3; typical use is 8–32 samples
   * spanning 380–700 nm. The engine linearly interpolates between samples.
   *
   * When absent and attenuationColor is present, engine falls back to the
   * existing RGB Beer-Lambert approximation.
   *
   * Reference: Wilkie et al., "Hero Wavelength Spectral Sampling," EGSR 2014.
   */
  readonly spectralAttenuation?: SpectralCurve;

  /**
   * Abbe number V_d = (n_d − 1) / (n_F − n_C) for wavelength-dependent IOR.
   * Higher values = lower dispersion. When set, the engine computes
   * wavelength-dependent IOR via the two-term Cauchy approximation and uses
   * hero-wavelength sampling so each path traces a single wavelength.
   * Range: 20 (dense flint, high dispersion) to 90 (crown glass, low).
   * Default: undefined (no dispersion).
   *
   * Reference: OpenPBR Surface v1.1.1 `transmission_dispersion_abbe_number`.
   */
  readonly dispersionAbbeNumber?: number;

  // ── Volume scattering (RFE-02) ──────────────────────────────────────────
  /**
   * Scattering coefficient σ_s, in inverse scene-length-units (matching
   * attenuationDistance units). The total extinction coefficient is
   * σ_t = σ_a + σ_s, where σ_a is derived from attenuationDistance.
   *
   * When scatteringCoefficient > 0 and transmission > 0, the backend
   * activates volumetric path tracing (delta tracking) for rays passing
   * through this medium.
   *
   * Reference: Novák et al., "Monte Carlo Methods for Volumetric Light
   * Transport Simulation," CGF 2018 (delta tracking / null-collision).
   */
  readonly scatteringCoefficient?: number;

  /**
   * Henyey-Greenstein phase function asymmetry parameter g ∈ (−1, 1).
   *   g = 0:   isotropic scatter.
   *   g > 0:   forward-biased scatter (biological tissue ~0.9).
   *   g < 0:   backward-biased scatter (retroreflective powders).
   * Default: 0 (isotropic) when scatteringCoefficient is set.
   *
   * Reference: Henyey & Greenstein, "Diffuse Radiation in the Galaxy," 1941.
   */
  readonly scatteringAnisotropy?: number;

  /**
   * Per-channel (RGB) scattering coefficients. When present, overrides the
   * scalar scatteringCoefficient with wavelength-dependent values, producing
   * chromatic scattering (e.g., sky-blue tint in forward-scattered white light).
   * Units and defaults same as scatteringCoefficient.
   *
   * Reference: OpenPBR Surface v1.1.1 `transmission_scatter`,
   * Standard Surface `transmission_scatter`.
   */
  readonly scatteringCoefficientRGB?: Vec3;

  // ── Per-face BSDF asymmetry / layered BSDF (RFE-03) ────────────────────
  /**
   * Thin absorbing layer applied to the front (outward-normal) face.
   * "Front" is the face whose normal points in the direction used to define
   * the mesh's vertex normals.
   *
   * When present, the BSDF for rays hitting this face is:
   *   L_front = SurfaceAbsorptionLayer.transmission ⊙ L_bulk
   * where L_bulk is the standard Material BSDF evaluation.
   *
   * Reference: Belcour, "Efficient Rendering of Layered Materials using an
   * Atomic Decomposition with Statistical Operators," ACM TOG (SIGGRAPH 2018).
   */
  readonly frontLayer?: SurfaceAbsorptionLayer;

  /**
   * Thin absorbing layer applied to the back (inward-normal) face.
   * Symmetric semantics to frontLayer.
   */
  readonly backLayer?: SurfaceAbsorptionLayer;

  // ── Multi-layer thin-film interference / TMM (RFE-04) ──────────────────
  /**
   * Multi-layer thin-film stack evaluated via the Transfer Matrix Method.
   * When present, overrides the single-layer iridescence model
   * (iridescence / iridescenceIor / iridescenceThicknessRange).
   * The TMM result is applied as a wavelength-dependent BSDF weight
   * on the specular lobe.
   *
   * Reference: Born & Wolf, "Principles of Optics" (Abeles TMM);
   * Belcour & Barla, "A Practical Extension to Microfacet Theory for the
   * Modeling of Varying Iridescence," ACM TOG (SIGGRAPH 2017).
   */
  readonly thinFilmStack?: ThinFilmStack;

  // ── Anisotropic specular (Gap 5 — stainedGlass audit 2026-05-12) ───────
  /**
   * Anisotropic specular highlight strength ∈ [0, 1].
   * 0 = isotropic (default); 1 = fully anisotropic.
   *
   * Mirrors `THREE.MeshPhysicalMaterial.anisotropy`. The field is set
   * directly on the THREE material (not via userData) for ripple and
   * waterglass cells in the stainedGlass baking pipeline.
   *
   * Reference: Three.js MeshPhysicalMaterial.anisotropy
   * (https://threejs.org/docs/#api/en/materials/MeshPhysicalMaterial.anisotropy).
   */
  readonly anisotropy?: number;

  /**
   * Rotation of the anisotropic highlight in radians ∈ [0, π].
   * Only meaningful when `anisotropy` > 0.
   *
   * Mirrors `THREE.MeshPhysicalMaterial.anisotropyRotation`.
   *
   * Reference: Three.js MeshPhysicalMaterial.anisotropyRotation
   * (https://threejs.org/docs/#api/en/materials/MeshPhysicalMaterial.anisotropyRotation).
   */
  readonly anisotropyRotation?: number;

  // ── Backend escape hatch ────────────────────────────────────────────────
  /** Backends may read keyed fields from here for backend-specific features.
   *  Core never inspects this map. */
  readonly extensions?: Readonly<Record<string, unknown>>;
}

/** Feature slices for adapters and partial material updates (W3 contract hygiene). */
export type BasePbrMaterialFields = Pick<
  MaterialSpec,
  'baseColor' | 'roughness' | 'metallic' | 'emissive' | 'emissiveIntensity'
>;
export type TransmissionMaterialFields = Pick<
  MaterialSpec,
  'transmission' | 'ior' | 'attenuationColor' | 'attenuationDistance' | 'thickness'
>;
export type MaterialMapFields = Pick<
  MaterialSpec,
  | 'baseColorMap'
  | 'normalMap'
  | 'normalScale'
  | 'roughnessMap'
  | 'metallicMap'
  | 'transmissionMap'
  | 'emissiveMap'
  | 'alphaMap'
>;
export type DisneyBsdMaterialFields = Pick<
  MaterialSpec,
  | 'sheen'
  | 'sheenColor'
  | 'sheenRoughness'
  | 'clearcoat'
  | 'clearcoatRoughness'
  | 'iridescence'
  | 'iridescenceIor'
  | 'iridescenceThicknessRange'
>;
export type VolumeMaterialFields = Pick<
  MaterialSpec,
  'scatteringCoefficient' | 'scatteringAnisotropy' | 'scatteringCoefficientRGB'
>;
export type LayeredBsdMaterialFields = Pick<MaterialSpec, 'frontLayer' | 'backLayer' | 'thinFilmStack'>;

/** Opaque texture reference. The scene-binding layer creates these; backends
 *  consume them. The shape varies — for WebGL2 backends it might be a
 *  `WebGLTexture` plus metadata, for WebGPU it might be a `GPUTexture`, for
 *  in-memory uploads it might be a `Uint8Array` + descriptor. Core doesn't
 *  care. */
export type TextureRef = unknown;
