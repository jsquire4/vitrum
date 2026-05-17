// Scene description — backend-agnostic.
//
// Design principle: every scene the engine renders is composed of three things —
// PRIMITIVES (what occupies space), EMITTERS (what gives off light), and an
// ENVIRONMENT (the world's hemispheric light source). The camera lives in
// FrameInput because it changes per-frame; the scene itself is camera-free.
//
// The general case is the union of every shape we might support. The current
// concrete needs are triangle meshes (panels, walls, floors) and analytic
// primitives (architectural-pattern shapes such as H-channel came rails;
// Phase 6 sprint 5 lands them). Future kinds extend the discriminated union
// without breaking older backends — backends pattern-match on `kind` and
// ignore unknown kinds with a warning, not a crash.

// ────────────────────────────────────────────────────────────────────────────
// Math primitives (these are exported for hosts to construct against)
// ────────────────────────────────────────────────────────────────────────────

export type Vec2 = readonly [number, number];
export type Vec3 = readonly [number, number, number];
export type Vec4 = readonly [number, number, number, number];

/** Column-major 4×4 matrix, 16 elements. Matches Three.js + WebGPU/WebGL convention. */
export type Mat4 = Float32Array;

/** A monotonic, host-supplied identifier. Stable across `setScene` calls so
 *  backends can do incremental updates. Hosts should use whatever their scene
 *  graph uses — three.js `Object3D.uuid`, integer counters, etc. */
export type SceneNodeId = string;

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
 *  **Mutability contract:** primitive and emitter types expose `readonly material:
 *  Material` — the *slot* is not reassigned through the contract API. To change
 *  any material field after `setScene`, hosts MUST call `engine.updatePrimitive`
 *  (or `updateEmitter`) with the patched material. Backends are not required to
 *  detect in-place mutations and will not generally rescan unchanged slots,
 *  matching the existing incremental-update contract on `Engine`.
 *
 *  Texture handles are opaque to core. The scene-binding layer (e.g.,
 *  @vitrum/three-bindings) is responsible for converting host textures to
 *  whatever format the backend expects (typed arrays for upload, GPU texture
 *  handles, etc.). Core just routes them through. */
export interface Material {
  // ── Base PBR ────────────────────────────────────────────────────────────
  baseColor: Vec3;
  roughness: number;            // 0 = mirror, 1 = matte
  metallic: number;             // 0 = dielectric, 1 = pure metal
  emissive?: Vec3;
  emissiveIntensity?: number;

  // ── Transmission / refraction ───────────────────────────────────────────
  transmission?: number;        // 0 = opaque, 1 = fully transparent
  ior?: number;                  // index of refraction
  attenuationColor?: Vec3;       // Beer-Lambert: color the medium absorbs to
  attenuationDistance?: number;  // Beer-Lambert: depth at which attenuationColor reached
  thickness?: number;            // Beer-Lambert: actual slab thickness

  // ── Texture maps (opaque handles, see TextureRef) ───────────────────────
  baseColorMap?: TextureRef;
  normalMap?: TextureRef;
  normalScale?: number;
  roughnessMap?: TextureRef;
  metallicMap?: TextureRef;
  transmissionMap?: TextureRef;
  emissiveMap?: TextureRef;
  alphaMap?: TextureRef;

  // ── Disney BSDF extensions (optional) ───────────────────────────────────
  sheen?: number;
  sheenColor?: Vec3;
  sheenRoughness?: number;
  clearcoat?: number;
  clearcoatRoughness?: number;
  iridescence?: number;
  iridescenceIor?: number;
  iridescenceThicknessRange?: Vec2;

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
  spectralAttenuation?: SpectralCurve;

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
  dispersionAbbeNumber?: number;

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
  scatteringCoefficient?: number;

  /**
   * Henyey-Greenstein phase function asymmetry parameter g ∈ (−1, 1).
   *   g = 0:   isotropic scatter.
   *   g > 0:   forward-biased scatter (biological tissue ~0.9).
   *   g < 0:   backward-biased scatter (retroreflective powders).
   * Default: 0 (isotropic) when scatteringCoefficient is set.
   *
   * Reference: Henyey & Greenstein, "Diffuse Radiation in the Galaxy," 1941.
   */
  scatteringAnisotropy?: number;

  /**
   * Per-channel (RGB) scattering coefficients. When present, overrides the
   * scalar scatteringCoefficient with wavelength-dependent values, producing
   * chromatic scattering (e.g., sky-blue tint in forward-scattered white light).
   * Units and defaults same as scatteringCoefficient.
   *
   * Reference: OpenPBR Surface v1.1.1 `transmission_scatter`,
   * Standard Surface `transmission_scatter`.
   */
  scatteringCoefficientRGB?: Vec3;

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
  frontLayer?: SurfaceAbsorptionLayer;

  /**
   * Thin absorbing layer applied to the back (inward-normal) face.
   * Symmetric semantics to frontLayer.
   */
  backLayer?: SurfaceAbsorptionLayer;

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
  thinFilmStack?: ThinFilmStack;

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
  anisotropy?: number;

  /**
   * Rotation of the anisotropic highlight in radians ∈ [0, π].
   * Only meaningful when `anisotropy` > 0.
   *
   * Mirrors `THREE.MeshPhysicalMaterial.anisotropyRotation`.
   *
   * Reference: Three.js MeshPhysicalMaterial.anisotropyRotation
   * (https://threejs.org/docs/#api/en/materials/MeshPhysicalMaterial.anisotropyRotation).
   */
  anisotropyRotation?: number;

  // ── Backend escape hatch ────────────────────────────────────────────────
  /** Backends may read keyed fields from here for backend-specific features.
   *  Core never inspects this map.
   *
   *  **Verified consumers (W3-D17 audit, 2026-05-17):**
   *  - `@vitrum/three-bindings` reads `extensions.dichroicLUTs` in
   *    `vitrumSceneToThree.ts` (re-stamps the pre-convolved angle-indexed
   *    LUTs into THREE userData so they survive the THREE → vitrum → THREE
   *    round trip) and writes the same key in `material.ts::convertMaterial`
   *    from inbound THREE userData. See RFE-10 dichroic addendum (PHY.1).
   *
   *  Not yet read by `pt-webgl`, `pt-webgpu`, or `walkaround-hybrid`; those
   *  may bind keyed entries here as new backend-specific features land. */
  extensions?: Readonly<Record<string, unknown>>;
}

/** Opaque texture reference. The scene-binding layer creates these; backends
 *  consume them. The shape varies — for WebGL2 backends it might be a
 *  `WebGLTexture` plus metadata, for WebGPU it might be a `GPUTexture`, for
 *  in-memory uploads it might be a `Uint8Array` + descriptor. Core doesn't
 *  care. */
export type TextureRef = unknown;

// ────────────────────────────────────────────────────────────────────────────
// Primitives — geometry that occupies space
// ────────────────────────────────────────────────────────────────────────────

/** Triangle mesh. Position/normal/uv arrays follow three.js convention:
 *  flat Float32Arrays where consecutive triples (or pairs for uv) describe
 *  one vertex. `indices` is optional; without it, vertices are interpreted
 *  as triangle-list. */
export interface MeshPrimitive {
  readonly kind: 'mesh';
  readonly id: SceneNodeId;
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs?: Float32Array;
  readonly tangents?: Float32Array;       // xyzw per vertex; w = bitangent sign
  readonly indices?: Uint32Array | Uint16Array;
  readonly material: Material;
  readonly transform?: Mat4;              // identity if absent
  readonly castShadow?: boolean;          // default true
  readonly receiveShadow?: boolean;       // default true
}

/** Same geometry repeated at many transforms. Backend may build a single BVH
 *  once and traverse via instance transforms. */
export interface InstancedMeshPrimitive {
  readonly kind: 'instanced-mesh';
  readonly id: SceneNodeId;
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs?: Float32Array;
  readonly tangents?: Float32Array;
  readonly indices?: Uint32Array | Uint16Array;
  readonly material: Material;
  readonly instances: ReadonlyArray<Mat4>;
}

/** Closed-form ray-primitive intersection. Backend-supported shapes only;
 *  unsupported shapes log a warning and degrade to skip (or to mesh
 *  tessellation if a fallback geometry is provided).
 *
 *  Phase 6 sprint 5 introduces 'h-channel-came' for our analytic came/solder
 *  geometry. Future shapes (gemstones via 'ellipsoid', pillars via 'capsule',
 *  etc.) extend this discriminated union without breaking existing scenes.
 */
export interface AnalyticPrimitive {
  readonly kind: 'analytic';
  readonly id: SceneNodeId;
  readonly shape: AnalyticShape;
  readonly params: Float32Array;          // shape-specific layout, see AnalyticShape
  readonly material: Material;
  readonly transform?: Mat4;
  readonly fallbackMesh?: Omit<MeshPrimitive, 'kind' | 'id' | 'material' | 'transform'>;
}

export type AnalyticShape =
  | 'sphere'           // params: [cx, cy, cz, radius]
  | 'box'              // params: [cx, cy, cz, hx, hy, hz]
  | 'capsule'          // params: [ax, ay, az, bx, by, bz, radius]
  | 'cylinder'         // params: [cx, cy, cz, radius, halfHeight]
  | 'h-channel-came';  // params: [length, railWidth, blockHeight, webThickness] — H-channel rail primitive, Phase 6 sprint 5

export type ScenePrimitive =
  | MeshPrimitive
  | InstancedMeshPrimitive
  | AnalyticPrimitive;

// ────────────────────────────────────────────────────────────────────────────
// Emitters — anything that gives off light
// ────────────────────────────────────────────────────────────────────────────

export type SceneEmitter =
  | DirectionalEmitter
  | DiscAreaEmitter
  | RectAreaEmitter
  | PointEmitter
  | SpotEmitter
  | MeshAreaEmitter;

export interface EmitterBase {
  readonly id: SceneNodeId;
  readonly color: Vec3;
  readonly intensity: number;
  readonly castShadow?: boolean;          // default true
}

export interface DirectionalEmitter extends EmitterBase {
  readonly kind: 'directional';
  readonly direction: Vec3;               // unit vector pointing AT the light
  /** Optional: angular subtense for soft shadows. 0 = perfectly directional. */
  readonly angularDiameter?: number;
}

export interface DiscAreaEmitter extends EmitterBase {
  readonly kind: 'disc-area';
  readonly position: Vec3;
  readonly normal: Vec3;
  readonly radius: number;
}

export interface RectAreaEmitter extends EmitterBase {
  readonly kind: 'rect-area';
  readonly position: Vec3;
  readonly uAxis: Vec3;                   // half-width vector
  readonly vAxis: Vec3;                   // half-height vector (uAxis × vAxis = normal)
}

export interface PointEmitter extends EmitterBase {
  readonly kind: 'point';
  readonly position: Vec3;
  readonly distance?: number;             // attenuation falloff distance
  readonly decay?: number;                // 0 = no decay, 2 = physical inverse-square
}

export interface SpotEmitter extends EmitterBase {
  readonly kind: 'spot';
  readonly position: Vec3;
  readonly direction: Vec3;
  readonly angle: number;                 // half-cone angle in radians
  readonly penumbra?: number;             // 0–1; 0 = hard edge, 1 = full penumbra
  readonly distance?: number;
  readonly decay?: number;
}

export interface MeshAreaEmitter extends EmitterBase {
  readonly kind: 'mesh-area';
  /** References a `MeshPrimitive` in the scene by id. The emitter samples
   *  surface points on that mesh; the mesh's material's emissive contributes
   *  to the radiance. Used for textured panel cells (e.g., stained-glass
   *  cells where each cell contributes its baked emissive). */
  readonly meshId: SceneNodeId;
}

// ────────────────────────────────────────────────────────────────────────────
// Environment — hemispheric / global light source
// ────────────────────────────────────────────────────────────────────────────

export type SceneEnvironment =
  | HdriEnvironment
  | ProceduralSkyEnvironment
  | NoneEnvironment;

export interface HdriEnvironment {
  readonly kind: 'hdri';
  readonly hdri: TextureRef;
  readonly intensity?: number;            // default 1
  readonly rotationY?: number;            // radians, default 0
}

export interface ProceduralSkyEnvironment {
  readonly kind: 'procedural-sky';
  readonly sunDirection: Vec3;
  readonly turbidity: number;
  readonly rayleigh: number;
  readonly mieCoefficient: number;
  readonly mieDirectionalG: number;
  readonly intensity?: number;
}

export interface NoneEnvironment {
  readonly kind: 'none';
}

// ────────────────────────────────────────────────────────────────────────────
// The Scene
// ────────────────────────────────────────────────────────────────────────────

/** A complete, immutable scene description. Hosts call `engine.setScene(scene)`
 *  with a new Scene whenever the geometry, materials, or lighting topology
 *  changes. For frequent property edits (color sliders, intensity scrubs),
 *  prefer `engine.updatePrimitive` / `engine.updateEmitter` if the backend
 *  reports `capabilities.supportsIncrementalScene = true`. */
export interface Scene {
  readonly primitives: ReadonlyArray<ScenePrimitive>;
  readonly emitters: ReadonlyArray<SceneEmitter>;
  readonly environment: SceneEnvironment;
}
