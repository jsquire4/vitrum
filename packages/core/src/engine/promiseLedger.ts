import type { AnalyticShape, ScenePrimitive } from '../scene/primitives.js';
import type { SceneEmitter } from '../scene/emitters.js';
import type { SceneEnvironment } from '../scene/environment.js';
import type { MaterialSpec } from '../scene/material.js';
import type {
  BackendSupportDetails,
  BackendSupportMode,
  EngineDenoiserMode,
  EngineCapabilities,
  FramePresentationMode,
  IncrementalPatchSupport,
} from './capabilities.js';

export type BackendId = 'walkaround-hybrid' | 'pt-webgl2' | 'pt-webgpu';

export interface BackendMethodPromises {
  readonly updatePrimitive: boolean;
  readonly updateEmitter: boolean;
  readonly updateEnvironment: boolean;
  readonly addPrimitive: boolean;
  readonly removePrimitive: boolean;
  readonly setSize: boolean;
  readonly updateLighting: boolean;
  readonly onFrame: boolean;
  readonly onProgress: boolean;
  readonly debug: boolean;
  /** Whether the backend implements the optional `Engine.getScene()` scene
   *  read-back (returns the retained canonical core {@link Scene}). All three
   *  shipping backends retain the core Scene and implement it. */
  readonly getScene: boolean;
  /** Whether the backend implements `Engine.onError()` — the GPU/runtime
   *  error subscription.  All three shipping backends wire this. */
  readonly onError: boolean;
  /** Whether the backend implements `Engine.onWarning()` — the structured
   *  non-fatal warning subscription. All three shipping backends wire this. */
  readonly onWarning: boolean;
  /**
   * Whether the backend implements `Engine.captureFrame()` — the GPU→CPU
   * pixel readback that returns a {@link CapturedFrame} (linear HDR RGBA
   * Float32, top-left origin).  All three shipping backends implement this.
   */
  readonly captureFrame: boolean;
  /**
   * Whether the backend implements `Engine.createInverseSession()` — the
   * differentiable ray-tracing inverse session opener. pt-webgpu only;
   * walkaround-hybrid omits this method. pt-webgl2 exposes the finite-difference
   * session only; pt-webgpu also has the scoped path-replay adjoint fast path.
   */
  readonly createInverseSession: boolean;
  /**
   * Whether the backend implements `Engine.getRestirPtResultBuffer()` —
   * the ReSTIR-PT resolve-pass output buffer accessor. pt-webgpu only (and only
   * when `pt-webgpu-one-edge-gris-reconnection` is active); other backends omit it.
   */
  readonly getRestirPtResultBuffer: boolean;
  /**
   * Whether the backend implements `Engine.getPresentationSource()` for
   * host-side presentation of an engine-owned offscreen texture.
   */
  readonly getPresentationSource: boolean;
  /**
   * Whether the backend implements `Engine.getProgressiveSeedTexture()` —
   * the progressive walkaround→PT seed SOURCE (post-denoise HDR output exposed
   * for cross-engine seeding). walkaround-hybrid only; PT backends omit it.
   */
  readonly getProgressiveSeedTexture: boolean;
  /**
   * Whether the backend implements `Engine.seedAccumulator()` — the
   * progressive walkaround→PT seed SINK (injects a decaying prior into the
   * accumulator). pt-webgpu only; walkaround-hybrid and pt-webgl2 omit it.
   */
  readonly seedAccumulator: boolean;
  /**
   * Whether the backend implements `exportGIState()` / `importGIState()` —
   * the GI-state persistence surface (DDGI probe-atlas + ReSTIR-GI reservoir
   * export/import). walkaround-hybrid only; PT backends omit this surface.
   */
  readonly giStatePersistence: boolean;
}

export interface FrameInputPromises {
  readonly honorsViewportPerFrame: boolean;
  readonly requiresSwapChainView: boolean;
  readonly honorsPerFrameBounces: boolean;
}

export interface BackendPromiseRecord {
  readonly supportsIncrementalScene: boolean;
  readonly incrementalPatchSupport: IncrementalPatchSupport;
  /** Whether the backend implements the explicit whole-primitive add/remove API
   *  ({@link Engine.addPrimitive} / {@link Engine.removePrimitive}). Distinct
   *  from `incrementalPatchSupport.topology` (count-change patches on an
   *  EXISTING primitive). */
  readonly supportsAddRemovePrimitive: boolean;
  readonly supportsAuxBuffers: boolean;
  readonly accumulates: boolean;
  readonly supportedPrimitiveKinds: readonly ScenePrimitive['kind'][];
  readonly supportedEmitterKinds: readonly SceneEmitter['kind'][];
  readonly supportedEnvironmentKinds: readonly SceneEnvironment['kind'][];
  readonly supportedAnalyticShapes: readonly AnalyticShape[];
  readonly presentationMode: FramePresentationMode;
  /** Static backend promise surface for the backend's audited full-capability
   *  profile. Runtime-selected compatibility tiers may return a narrower
   *  `EngineCapabilities.supportDetails` object while keeping the same backend
   *  id; hosts must use the live engine capabilities for per-instance routing. */
  readonly supportDetails: BackendSupportDetails;
  readonly methodPromises: BackendMethodPromises;
  readonly frameInputPromises: FrameInputPromises;
}

// ── Compile-time drift guard ─────────────────────────────────────────────────
//
// `BackendPromiseRecord` mirrors a subset of `EngineCapabilities`. The
// collection fields (supportedPrimitiveKinds / supportedEmitterKinds /
// supportedEnvironmentKinds / supportedAnalyticShapes) intentionally diverge in
// container type: the ledger uses readonly arrays (serialisable, easy to assert
// in tests) while `EngineCapabilities` uses `ReadonlySet` (O(1) has-check).
// That divergence is structural-by-design and cannot be bridged with a blanket
// `satisfies Partial<EngineCapabilities>`.
//
// Instead we assert that the SCALAR / STRUCT fields that mirror cap keys are
// structurally compatible. Uses the `AssertExtends<TExpected, TActual>` idiom:
// the generic `U extends T` constraint fires a real TS error (not a silent
// `never`) if `BackendPromiseRecord` drops or renames a guarded key.
//
// `_LedgerCapabilitySlice` picks exactly the cap keys whose types ARE compatible
// across both shapes. When a new scalar/struct cap is added to `EngineCapabilities`
// and it belongs in the ledger, add it here AND to `BackendPromiseRecord`.
type _LedgerCapabilitySlice = Pick<
  EngineCapabilities,
  | 'supportsIncrementalScene'
  | 'incrementalPatchSupport'
  | 'supportsAddRemovePrimitive'
  | 'supportsAuxBuffers'
  | 'accumulates'
  | 'presentationMode'
  | 'supportDetails'
>;
// `U extends T` in the type parameter position emits TS2344 if U is not
// assignable to T — unlike `declare const x: never` which is always valid.
type _AssertExtends<T, U extends T> = U;
// This line errors if BackendPromiseRecord drops or incompatibly changes any
// of the capability keys listed in _LedgerCapabilitySlice.
type _LedgerCoversCapabilities = _AssertExtends<_LedgerCapabilitySlice, BackendPromiseRecord>;
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Every incremental-patch facet supported. Shared frozen value referenced by
 * all three backend records below — they are byte-identical on this field, so
 * a single shared const prevents drift between them.
 *
 * Per-backend `incrementalPatchSupport.topology === true` rationale (what
 * "count-change patches on an EXISTING primitive are absorbed without a full
 * setScene" means for each backend):
 *
 *   • walkaround-hybrid — vertex/index-COUNT changes (positions/normals/uvs/
 *     tangents/indices) ARE absorbed: a geometry change invalidates every cached
 *     GI signal on this realtime stack, so the engine re-runs its BVH rebuild +
 *     temporal reset (reusing the packing path, not a targeted in-place edit).
 *     `instances`/`params`/`shape`/`fallbackMesh`/`kind` patches are ALSO absorbed
 *     (P5, 2026-06-03): `HybridEngine.updatePrimitive` intercepts these
 *     wholesale-replacement fields and routes them through a full setScene rebuild
 *     (mutate-Scene → setScene spine, like addPrimitive) — no longer a throw. So
 *     instance-COUNT changes work here too; the rebuild is the cost (GI is
 *     invalidated either way on a realtime stack), matching pt-webgl2/pt-webgpu's
 *     contract surface.
 *   • pt-webgl2 — mesh/skinned vertex/index-COUNT changes and instanced-mesh
 *     instance-COUNT changes route through the retained core scene and rebuild
 *     the backend's scene textures/BVH pack. Co-present `material` routes through
 *     the same setScene repack, so material/light indices cannot drift.
 *   • pt-webgpu — instanced-mesh instance-COUNT change → TLAS-only rebuild, BLAS
 *     reused (rebuildTlasReuseBlas + uploadScenePackTlasRealloc); mesh/skinned
 *     vertex/index-COUNT change → rebuild only the changed primitive's BLAS,
 *     splice into concat buffers, rebase offsets + TLAS roots
 *     (rebuildPrimitiveBlas + uploadScenePackGeometryRealloc).
 *
 * In all three, `id`/`kind` morphs throw in patchPrimitiveInScene, and
 * whole-primitive ADD/REMOVE is setScene (see supportsAddRemovePrimitive), not a
 * patch. So `topology` means "vertex/index-count patches on an existing primitive
 * are absorbed" — fully on all three backends, incl. the instance-count case
 * (walkaround does it via a full setScene rebuild, the PT backends via targeted
 * TLAS/BLAS realloc).
 *
 * Per-backend `supportsAddRemovePrimitive === true` rationale: addPrimitive
 * appends a new primitive and removePrimitive evicts one, each by routing a
 * fresh mutated `Scene` copy through the engine's existing setScene packing path
 * (convert→expand→repack). A new primitive almost always brings a NEW material,
 * and the targeted geometry-only regen SKIPS material re-pack; reusing the
 * shared setScene path re-packs the material + light arrays correctly by
 * construction — no fragile per-array index remap. Distinct from
 * incrementalPatchSupport.topology (count-change patches on an EXISTING primitive).
 */
const ALL_PATCHES_SUPPORTED: IncrementalPatchSupport = Object.freeze({
  transform: true,
  positions: true,
  material: true,
  emitter: true,
  topology: true,
});

const ALL_EMITTERS_NATIVE: BackendSupportDetails['emitters'] = Object.freeze({
  directional: 'native',
  'rect-area': 'native',
  'disc-area': 'native',
  point: 'native',
  spot: 'native',
  'mesh-area': 'native',
});

/**
 * Walkaround-hybrid emitter support: point/spot now have native analytic
 * direct-light paths in shade / transparent OIT and are also projected into
 * DDGI / RC fixture-light uploads for probe transport. ReSTIR-DI reservoirs
 * remain area-emitter based, but point/spot are no longer DDGI-only rows.
 */
const WALKAROUND_EMITTERS: BackendSupportDetails['emitters'] = Object.freeze({
  directional: 'native',
  'rect-area': 'native',
  'disc-area': 'native',
  // H41 — additive analytic NEE loop in shade.wgsl (binding 13, separate from
  // the RIS area-emitter pool). Inverse-square + spot cone smoothstep falloff
  // with deterministic shadow rays. Grade promoted from 'approximate' to 'native'.
  point: 'native',
  spot: 'native',
  'mesh-area': 'native',
});

/**
 * pt-webgpu emitter support: all emitter kinds are now native.
 * disc-area was previously approximate (32-triangle fan); it is now packed natively
 * into the rect-area stream with a shape discriminator (RECT_DISC_SHAPE_DISC = 1.0
 * in emission.w) and sampled via the concentric-disc map (Shirley & Chiu 1997),
 * exactly matching pt-webgl2's CIRC_AREA_LIGHT handling.
 * Promoted approximate → native, 2026-06-10 (emitterPacking.ts `packDiscAsRect`).
 * A/B radiometric validation in R9-B.
 */
const PT_WEBGPU_EMITTERS: BackendSupportDetails['emitters'] = ALL_EMITTERS_NATIVE;

const PT_WEBGPU_ANALYTIC_SHAPES_NATIVE: BackendSupportDetails['analyticShapes'] = Object.freeze({
  sphere: 'native',
  box: 'native',
  capsule: 'native',
  cylinder: 'native',
  'h-channel-came': 'native',
});

const ANALYTIC_SHAPES_FALLBACK_GENERATED_MESH: BackendSupportDetails['analyticShapes'] = Object.freeze({
  sphere: 'fallback-generated-mesh',
  box: 'fallback-generated-mesh',
  capsule: 'fallback-generated-mesh',
  cylinder: 'fallback-generated-mesh',
  'h-channel-came': 'fallback-generated-mesh',
});

// ── CAP-01 — per-field material support matrix (2026-06-11) ──────────────────
//
// Every public `MaterialSpec` field has an explicit row for each shipping
// backend, graded by READING the actual packer + shader consumption path:
//   'native'      — the authored value demonstrably reaches a shader
//                   contribution path with the field's documented semantics.
//   'approximate' — consumed, but with materially different semantics
//                   (quantization, reinterpretation, partial sub-field
//                   consumption, authored-tangent substitution, …). The
//                   divergence is documented on the row.
//   'unsupported' — silently dropped by the backend's scene ingestion
//                   (structured `*.unsupported-…` / `*.unconsumed-…` warnings
//                   fire on setScene where render-affecting).
//
// Evidence trail (code-read 2026-06-11):
//   pt-webgl2  — scene/materialsTexture.ts (93-texel packer) +
//                scene/texturesArray.ts (SAMPLED_MAP_KEYS atlas) +
//                glsl/render/get_surface_record_function.glsl.js +
//                glsl/shader/bsdf/* + glsl/render/attenuate_hit_function.glsl.js.
//   pt-webgpu  — scene/materialPacking.ts (27-vec4 packer) +
//                scene/materialTextures.ts (6-vec4 texture descriptors) +
//                wgsl/pathTrace/material.wgsl.ts + bsdf.wgsl.ts + kernel.wgsl.ts.
//   walkaround — restir/packingHelpers.ts (quantized per-tri lanes) +
//                restir/consumedMaterialFields.ts (allowlist; everything not in
//                it is warned via `walkaround-hybrid.unconsumed-material-fields`)
//                + shaders/shade.wgsl.ts / ris.wgsl.ts consumption.

/**
 * Runtime list of every public `MaterialSpec` key, in declaration order.
 * The two compile-time asserts below force this array AND each backend's
 * `materials` matrix to stay exhaustive: adding a field to `MaterialSpec`
 * without extending both is a TypeScript error, not a silent omission.
 */
export const MATERIAL_SPEC_FIELDS = [
  // Base PBR
  'baseColor', 'roughness', 'metallic', 'emissive', 'emissiveIntensity', 'shadingModel',
  // Alpha / coverage
  'alphaMode', 'alphaCutoff', 'opacity', 'doubleSided',
  // Transmission / refraction
  'transmission', 'ior', 'attenuationColor', 'attenuationDistance', 'thickness',
  // Texture maps + their scalars
  'baseColorMap', 'normalMap', 'normalScale', 'roughnessMap', 'metallicMap',
  'transmissionMap', 'thicknessMap', 'emissiveMap', 'alphaMap', 'aoMap', 'aoMapIntensity',
  'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap', 'clearcoatNormalScale',
  'sheenColorMap', 'sheenRoughnessMap', 'iridescenceMap', 'iridescenceThicknessMap',
  'anisotropyMap', 'specularColorMap', 'specularIntensityMap',
  'bumpMap', 'bumpScale', 'displacementMap', 'displacementScale', 'displacementBias', 'displacementSubdivisions',
  'lightMap', 'lightMapIntensity',
  // Disney BSDF extensions
  'sheen', 'sheenColor', 'sheenRoughness', 'clearcoat', 'clearcoatRoughness',
  'iridescence', 'iridescenceIor', 'iridescenceThicknessRange',
  // Dielectric specular (KHR_materials_specular) + IBL
  'specularIntensity', 'specularColor', 'envMapIntensity',
  // Spectral (RFE-01)
  'spectralAttenuation', 'dispersionAbbeNumber',
  // Volume scattering (RFE-02)
  'scatteringCoefficient', 'scatteringAnisotropy', 'scatteringCoefficientRGB',
  // Layered BSDF (RFE-03) + thin-film stack (RFE-04)
  'frontLayer', 'backLayer', 'thinFilmStack',
  // Anisotropic specular
  'anisotropy', 'anisotropyRotation',
  // Backend escape hatch
  'extensions',
] as const satisfies readonly (keyof MaterialSpec)[];

/**
 * Material fields unavailable in the shipping `pt-webgpu` lite trace profile
 * in addition to fields the full profile already rejects.
 *
 * This is executable contract data rather than adapter documentation: the
 * pt-webgpu scene gate and the glTF compatibility evaluator both consume this
 * exact frozen tuple. Keeping the profile restriction beside the exhaustive
 * MaterialSpec key list prevents either consumer from independently retyping
 * (and drifting from) the renderer's accepted domain.
 */
export const PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS = Object.freeze([
  'baseColorMap',
  'normalMap',
  'normalScale',
  'roughnessMap',
  'metallicMap',
  'transmissionMap',
  'thicknessMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'aoMapIntensity',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'clearcoatNormalScale',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'specularColorMap',
  'specularIntensityMap',
  'bumpMap',
  'bumpScale',
  'lightMap',
  'lightMapIntensity',
  'alphaMode',
  'alphaCutoff',
  'opacity',
  'envMapIntensity',
  'anisotropy',
  'anisotropyRotation',
] as const satisfies readonly (keyof MaterialSpec)[]);

// Compile-time exhaustiveness (reverse direction): a `MaterialSpec` key MISSING
// from MATERIAL_SPEC_FIELDS makes this type non-never → TS2344 below.
type _MissingMaterialSpecFields = Exclude<keyof MaterialSpec, (typeof MATERIAL_SPEC_FIELDS)[number]>;
type _AssertMaterialFieldsExhaustive = _AssertExtends<never, _MissingMaterialSpecFields>;

/** A full (non-partial) per-field material support matrix. Assignable to
 *  `BackendSupportDetails['materials']` (which stays Partial for back-compat).
 *  Internal — the public surface is the (Partial-typed) `supportDetails.materials`
 *  plus the exported `MATERIAL_SPEC_FIELDS` key list. */
type MaterialSupportMatrix = Readonly<
  Record<(typeof MATERIAL_SPEC_FIELDS)[number], BackendSupportMode>
>;

/**
 * walkaround-hybrid — the realtime GI stack's material model is QUANTIZED
 * per-triangle lanes (RGBA8 baseColor in bvhIndex.w, u8 rough/metal/ior lanes,
 * 4-bit transmission, scalar-alpha cutout bit, pre-baked Beer-Lambert tint) +
 * f32 HDR emissive Le. The texture-atlas path samples readable uv0/uv1
 * baseColorMap, normal/ORM/AO/alpha/emissive/transmission/thickness/light
 * maps plus specular/clearcoat/sheen/anisotropy/iridescence/bump map
 * families; the approximate rows below reflect compact GI/reservoir semantics,
 * not dropped TextureRefs.
 * Everything not consumed is warned once per setScene via
 * `walkaround-hybrid.unconsumed-material-fields`
 * (restir/consumedMaterialFields.ts allowlist — this matrix mirrors it exactly:
 * row !== 'unsupported' ⇔ field ∈ CONSUMED_MATERIAL_FIELDS).
 */
const WALKAROUND_MATERIALS: MaterialSupportMatrix = Object.freeze({
  // packBVHIndexWFromCore: RGBA8-quantized; replaced by attenuationColor on
  // transmissive surfaces (packingHelpers.ts resolveTriColor mirror).
  baseColor: 'approximate',
  // packBVHRoughMetalFromCore: u8 lane + B1 default invariants (0.85 / glass 0.05).
  // Roughness is supported for opaque reflection. When combined with physical
  // transmission, only numerical-zero delta interfaces are accepted; see the
  // machine-readable materialProfiles.roughTransmission row below.
  roughness: 'approximate',
  // u8 lane + a separate BINARY isMetal classification bit in bvhIndex.w.
  metallic: 'approximate',
  // packBVHEmissiveLeFromCore: full f32 HDR Le lane + ReSTIR-DI emitter
  // classification (shared-bvh emitterClassify.ts).
  emissive: 'native',
  // Folded into emitter Le at classification (emitterClassify.ts Le = emissive·ei).
  emissiveIntensity: 'native',
  shadingModel: 'approximate',
  // Cutout coverage: scalar mask uses opacity < alphaCutoff; atlas-backed
  // alpha maps are sampled in the primary/RIS/GI traversal path and
  // in atlas-backed shade/ReSTIR-DI/ReSTIR-GI/NRC/GRIS shadow visibility.
  // Fractional blend is camera-composited by walkaround's transparent-OIT pass,
  // including direct sun plus analytic point/spot lighting with alpha-aware
  // direct shadow transmittance and finite-emitter direct light. Secondary
  // ReSTIR-GI/NRC/DDGI/RC traversal uses independent seeded stochastic
  // coverage, so blend layers participate without deterministic correlation.
  alphaMode: 'native',
  alphaCutoff: 'native',
  opacity: 'native',
  // Dedicated per-triangle side metadata filters opaque back faces in hybrid,
  // DDGI, RC, and shadow traversal. Mirrored TLAS instances parity-correct the
  // hit side; transmissive exits remain admissible for closed volumes.
  doubleSided: 'native',
  // 4-bit trans4 lane in bvhIndex.w (16 steps).
  transmission: 'approximate',
  // u8-quantized [1,3] lane (B1-ior-per-tri) + DDGI material entry.
  ior: 'approximate',
  // Beer-Lambert tint PRE-BAKED per triangle at pack time (bvh_beer lane) with
  // `thickness` as a fixed slab; atlas-backed thicknessMap multiplies that slab
  // approximately by exponentiating the tint with thicknessTexture.g at hit UV.
  attenuationColor: 'approximate',
  attenuationDistance: 'approximate',
  thickness: 'approximate',
  // CPU pixel payloads and nominal WalkaroundWebGpuTextureSource descriptors
  // share a full-mip RGBA32F atlas. Authored sampler policy, texture transforms,
  // and compact arbitrary texCoord affine charts are honored. Primary and
  // secondary hits use a bounded camera-projected triangle footprint rather
  // than propagated ray differentials. Approximate because glass Beer/
  // transmission tint still uses the scalar packed color path, and compact
  // GI/reservoir paths retain mapped-emitter and transparent-transport limits.
  baseColorMap: 'approximate',
  // Phase-3D map slice: normalMap perturbs the camera-visible smooth normal
  // through authored/generated tangent.xyzw when present, falling back to a
  // derived per-triangle tangent frame. roughness/metallic sample glTF G/B
  // channels in shade for visible BRDF terms; AO samples the glTF R channel and
  // multiplies the runtime GTAO factor; transmissionMap modulates
  // shade/RIS/GI glass gating. Alpha maps feed ordered primary OIT and seeded
  // secondary coverage. Light maps are receiver-local irradiance converted to
  // outgoing albedo/π radiance in ReSTIR-GI, DDGI, and RC.
  roughnessMap: 'approximate',
  metallicMap: 'approximate',
  normalMap: 'approximate',
  normalScale: 'approximate',
  transmissionMap: 'approximate',
  thicknessMap: 'approximate',
  // Accepted emissive maps upload to a dedicated HDR atlas and every radiating
  // triangle is split into exact constant-radiance texel-cell emitters. The CDF,
  // alias table, light tree, p-hat, and final payload therefore share one
  // density. GPU-only/unreadable sources and missing UV sets are synchronously
  // rejected before scene publication rather than falling back to scalar Le.
  emissiveMap: 'native',
  alphaMap: 'native',
  aoMap: 'approximate',
  aoMapIntensity: 'approximate',
  // Rich-lobe maps drive the authored BRDF evaluation in direct, specular
  // indirect, DI, and GI receiver targets. Reservoir proposal PDFs are evaluated
  // for the distribution actually sampled, so they need not be lobe-matched.
  clearcoatMap: 'native',
  clearcoatRoughnessMap: 'native',
  // Clearcoat normal maps reuse walkaround's authored-tangent-aware atlas
  // normal path and feed shade-owned/DI/GI suffix plus receiver clearcoat
  // evaluations.
  clearcoatNormalMap: 'native',
  clearcoatNormalScale: 'native',
  // Atlas-backed sheen maps modulate the shade-owned Charlie sheen lobe plus DI/GI
  // suffix and receiver material payloads.
  sheenColorMap: 'native',
  sheenRoughnessMap: 'native',
  // Atlas-backed KHR_materials_iridescence maps modulate the shade-owned thin-film
  // F0 model plus DI/GI receiver-lobe material payloads.
  iridescenceMap: 'native',
  iridescenceThicknessMap: 'native',
  // Atlas-backed KHR_materials_anisotropy maps multiply the shade-owned scalar
  // anisotropic GGX branch (B = strength, RG = direction) and DI/GI suffix plus
  // receiver-lobe payloads. Proposal sampling remains unbiased because the
  // sampled source density, not an anisotropic surrogate, is used in weights.
  anisotropyMap: 'native',
  // Atlas-backed specular maps ride the material atlas and modulate shade-owned,
  // DI/GI receiver-lobe scalar specular controls.
  specularColorMap: 'native',
  specularIntensityMap: 'native',
  // Atlas-backed bump maps ride the walkaround material atlas and finite-difference a
  // visible normal perturbation for shade-owned, DI, GI suffix, and receiver
  // material payloads. The compact reservoir path keeps this approximate rather
  // than native.
  bumpMap: 'approximate',
  bumpScale: 'approximate',
  // CPU-readable displacement maps are applied before the shared merged BVH is
  // built. Authored vertices move by default; opt-in uniform dicing via
  // displacementSubdivisions adds bounded uniform CPU microdisplacement. The
  // absence of adaptive/error-bounded dicing is the intentional approximation.
  displacementMap: 'approximate',
  displacementScale: 'approximate',
  displacementBias: 'approximate',
  displacementSubdivisions: 'approximate',
  lightMap: 'native',
  lightMapIntensity: 'native',
  // Scalar sheen rides material atlas metadata and adds a Charlie/Neubelt-
  // Pettineo lobe in shade-owned direct/analytic/sun/specular-indirect paths
  // plus DI/GI receiver material payloads.
  sheen: 'native',
  sheenColor: 'native',
  sheenRoughness: 'native',
  // Scalar clearcoat rides material atlas metadata and adds a fixed-F0 GGX top
  // coat in shade-owned direct/analytic/sun/specular-indirect paths plus DI/GI
  // receiver material payloads.
  clearcoat: 'native',
  clearcoatRoughness: 'native',
  // Scalar KHR_materials_iridescence rides material atlas metadata and modifies
  // shade-owned GGX F0 in direct/analytic/sun/specular-indirect paths plus DI/GI
  // receiver material payloads.
  iridescence: 'native',
  iridescenceIor: 'native',
  iridescenceThicknessRange: 'native',
  // Scalar KHR_materials_specular controls ride the material atlas metadata and
  // modulate dielectric F0 in shade-owned direct/analytic/sun/specular-indirect
  // paths plus DI/GI receiver material payloads.
  specularIntensity: 'native',
  specularColor: 'native',
  // Material-atlas scalar consumed by the HDRI environment-light path:
  // RIS candidate scoring, canonical temporal/spatial p-hat reuse, and shade
  // resolve all apply the same per-surface scale. Finite emitters/sun/analytic
  // lights are intentionally unaffected.
  envMapIntensity: 'native',
  // Realtime optical reduction shared by shade/OIT/ReSTIR/DDGI/RC: a
  // 32-sample 380–780 nm CIE/D65 attenuation integral and three-channel
  // Cauchy/Abbe refraction. Deliberately approximate rather than unsupported.
  spectralAttenuation: 'approximate',
  dispersionAbbeNumber: 'approximate',
  // Bounded homogeneous single scattering uses per-channel sigma_s, RGB Beer
  // transmittance, and a normalized HG phase in shade/ReSTIR/DDGI/RC. It is an
  // intentional realtime single-scatter approximation, not a rejected field.
  scatteringCoefficient: 'approximate',
  scatteringAnisotropy: 'approximate',
  scatteringCoefficientRGB: 'approximate',
  // Face-selected transmission/roughness/normal controls are consumed at both
  // dielectric interfaces and in the camera/ReSTIR/DDGI/RC material payloads.
  frontLayer: 'approximate',
  backLayer: 'approximate',
  // Full spectral CPU TMM (stable Redheffer composition) preintegrated into
  // eight incidence-angle bins for forward/reverse realtime consumption.
  thinFilmStack: 'approximate',
  // Scalar anisotropy rides material atlas metadata and swaps shade-owned GGX
  // evals to an anisotropic branch. RIS source PDFs match their actual proposal
  // distributions, so isotropic proposals affect variance rather than bias.
  anisotropy: 'native',
  anisotropyRotation: 'native',
  // extensions.surfaceTextureId → texType3 procedural-pattern lane;
  // extensions.skipEmitter → emitter classification. Consumed as defined.
  extensions: 'native',
});

/**
 * pt-webgl2 — the 111-texel RGBA32F materials texture (materialsTexture.ts)
 * carries near-the-full MaterialSpec; the GLSL surface-record/BSDF chain
 * consumes it (get_surface_record_function.glsl.js, shader/bsdf/*). Texture
 * maps resolve through the SAMPLED_MAP_KEYS atlas (texturesArray.ts) with
 * per-map KHR_texture_transform + uv-set selection.
 */
const PT_WEBGL2_MATERIALS: MaterialSupportMatrix = Object.freeze({
  baseColor: 'native',
  roughness: 'native',
  metallic: 'native',
  emissive: 'native',
  emissiveIntensity: 'native',
  // Terminal base-color visibility branch in pt-webgl2 (`material.unlit`).
  // Approximate because it is not sampled as an emissive light source.
  shadingModel: 'approximate',
  alphaMode: 'native',
  alphaCutoff: 'native',
  opacity: 'native',
  // Packed into material.side (0 = both orientations, 1 = front only).
  doubleSided: 'native',
  transmission: 'native',
  ior: 'native',
  // attenuate_hit_function.glsl.js — per-ray Beer-Lambert.
  attenuationColor: 'native',
  attenuationDistance: 'native',
  // Path-length attenuation clamps to authored thickness/thicknessMap when present,
  // but still depends on closed-surface traversal rather than a true thin-shell volume.
  thickness: 'approximate',
  baseColorMap: 'native',
  normalMap: 'native',
  normalScale: 'native',
  roughnessMap: 'native',  // glTF G channel
  metallicMap: 'native',   // glTF B channel
  transmissionMap: 'native',
  thicknessMap: 'approximate',
  emissiveMap: 'native',
  // Sampled with uv-set selection, KHR_texture_transform, and wrap modes in
  // both surface-record and attenuation paths.
  alphaMap: 'native',
  aoMap: 'native',
  aoMapIntensity: 'native',
  clearcoatMap: 'native',
  clearcoatRoughnessMap: 'native',
  clearcoatNormalMap: 'native',
  clearcoatNormalScale: 'native',
  sheenColorMap: 'native',
  sheenRoughnessMap: 'native',
  iridescenceMap: 'native',
  iridescenceThicknessMap: 'native',
  anisotropyMap: 'native',
  specularColorMap: 'native',
  specularIntensityMap: 'native',
  bumpMap: 'native',
  bumpScale: 'native',
  // CPU-readable displacement maps run through shared-bvh geometry expansion
  // before the WebGL2 merged BVH/material textures are uploaded. Authored
  // vertices move by default; displacementSubdivisions adds bounded uniform CPU
  // microdisplacement, but not adaptive/error-bounded microgeometry.
  displacementMap: 'approximate',
  displacementScale: 'approximate',
  displacementBias: 'approximate',
  displacementSubdivisions: 'approximate',
  lightMap: 'native',
  lightMapIntensity: 'native',
  sheen: 'native',
  sheenColor: 'native',
  sheenRoughness: 'native',
  clearcoat: 'native',
  clearcoatRoughness: 'native',
  iridescence: 'native',
  iridescenceIor: 'native',
  iridescenceThicknessRange: 'native',
  specularIntensity: 'native',
  specularColor: 'native',
  envMapIntensity: 'native',
  // 32-sample uniform μ(λ) grid (s20..27), consumed in spectral mode.
  spectralAttenuation: 'native',
  dispersionAbbeNumber: 'native',
  scatteringCoefficient: 'native',
  scatteringAnisotropy: 'native',
  // Consumed as authored per-channel σ_s (s16 sssSigmaS) by the implemented
  // scalar-majorant WebGL single-scatter model. The model itself is the bounded
  // approximation; this grade is not conditional on a future evidence gate.
  scatteringCoefficientRGB: 'approximate',
  // transmission tint + per-face roughness override + per-face normal map/scale
  // are packed, face-selected, and sampled by surface + attenuation shaders.
  frontLayer: 'native',
  backLayer: 'native',
  // Full TMM evaluation; 35-layer cap (contract sanctions backend caps).
  thinFilmStack: 'native',
  // Scalar KHR_materials_anisotropy and its texture map are packed into
  // reserved material lanes and drive anisotropic GGX sampling/eval/PDF in the
  // pt-webgl2 BSDF, including the extension's RG strength/rotation convention.
  anisotropy: 'native',
  anisotropyRotation: 'native',
  // Contract-sanctioned escape hatch this backend deliberately does not read
  // (no warning — `extensions` is host-discretionary by design).
  extensions: 'unsupported',
});

/**
 * pt-webgpu — the 29-vec4 material buffer (materialPacking.ts) + the 82-vec4
 * texture-descriptor buffer (materialTextures.ts) feed material.wgsl /
 * bsdf.wgsl / kernel.wgsl. Full-tier material samplers now consume
 * TextureRef.texCoord, KHR_texture_transform, wrapS/T, and heterogeneous-layer
 * UV-fit scales per map for every texture field this backend samples.
 */
const PT_WEBGPU_MATERIALS: MaterialSupportMatrix = Object.freeze({
  baseColor: 'native',
  roughness: 'native',
  metallic: 'native',
  emissive: 'native',
  emissiveIntensity: 'native',
  // Terminal base-color visibility branch in pt-webgpu (`mat.isUnlit`).
  // Approximate because it is not sampled as an emissive light source.
  shadingModel: 'approximate',
  alphaMode: 'native',
  alphaCutoff: 'native',
  opacity: 'native',
  // Opaque one-sided back faces are skipped by closest/visibility traversal;
  // transmissive exits remain admissible for closed dielectric volumes.
  doubleSided: 'native',
  transmission: 'native',
  ior: 'native',
  // σ_a = −ln(attenuationColor)/attenuationDistance (WS4 vec4 #22) drives the
  // volumetric walk's per-channel extinction — the glTF-correct derivation.
  attenuationColor: 'native',
  attenuationDistance: 'native',
  // VOL-THICKNESS: pt-webgpu clamps Beer-Lambert attenuation distance to authored
  // KHR volume thickness when present. Approximate because the closed-surface
  // tracer still uses geometric boundaries rather than exact thin-shell volume
  // integration.
  thickness: 'approximate',
  baseColorMap: 'native',
  // Sampled + TBN-applied with glTF normalTexture.scale, per-map UV metadata,
  // and authored/generated tangent.xyzw handedness when available. Legacy or
  // tangentless scenes fall back to UV-gradient frame derivation.
  normalMap: 'native',
  normalScale: 'native',
  // Distinct slots with independent UV/wrap metadata; combined glTF
  // metallicRoughness maps still work by pointing both slots at one layer.
  roughnessMap: 'native',
  metallicMap: 'native',
  transmissionMap: 'native',
  thicknessMap: 'approximate',
  // Forward paths sample emissive maps natively. Inverse path-replay now also
  // replays the camera-direct emissive texel multiplier for emissive /
  // emissiveIntensity, while non-primary/indirect emission remains FD.
  emissiveMap: 'native',
  alphaMap: 'native',
  // Forward paths apply AO as a deliberate glTF local albedo multiplier. The
  // path-replay adjoint now mirrors that local factor for scoped baseColor fits.
  aoMap: 'native',
  aoMapIntensity: 'native',
  // Full-tier megakernel, inverse replay, BSDF area/environment connections,
  // ReSTIR-PT, BDPT, MNEE, and SPPM all consume the same texture-modulated
  // extension-lobe values. Clearcoat-normal evaluation/PDFs carry the authored
  // tangent-space normal through every finite-connection estimator; discrete
  // TMM thin-film vertices are excluded from those finite-density families.
  clearcoatMap: 'native',
  clearcoatRoughnessMap: 'native',
  clearcoatNormalMap: 'native',
  clearcoatNormalScale: 'native',
  sheenColorMap: 'native',
  sheenRoughnessMap: 'native',
  iridescenceMap: 'native',
  iridescenceThicknessMap: 'native',
  // Full-tier samplers consume KHR_materials_anisotropy RG/B map data. The
  // anisotropic lobe uses a conservative projected-roughness Kulla-Conty
  // approximation over the isotropic E LUT. That bounded energy model is the
  // final reason for the approximate grade.
  anisotropyMap: 'approximate',
  specularColorMap: 'native',
  specularIntensityMap: 'native',
  bumpMap: 'native',
  bumpScale: 'native',
  // Full-tier packSceneFromCore and merged-mode paths apply CPU-readable
  // displacement maps before BLAS/BVH construction. Authored vertices move by
  // default; displacementSubdivisions adds bounded uniform CPU microdisplacement,
  // but not adaptive/error-bounded microgeometry.
  displacementMap: 'approximate',
  displacementScale: 'approximate',
  displacementBias: 'approximate',
  displacementSubdivisions: 'approximate',
  lightMap: 'native',
  lightMapIntensity: 'native',
  sheen: 'native',
  sheenColor: 'native',
  sheenRoughness: 'native',
  clearcoat: 'native',
  clearcoatRoughness: 'native',
  iridescence: 'native',
  iridescenceIor: 'native',
  iridescenceThicknessRange: 'native',
  // SPEC-01 — scalar factors and local maps are consumed coherently by ordinary
  // PT, inverse replay, MNEE/SPPM receivers, ReSTIR-PT, and both BDPT subpaths.
  specularIntensity: 'native',
  specularColor: 'native',
  envMapIntensity: 'native',
  spectralAttenuation: 'native',
  dispersionAbbeNumber: 'native',
  scatteringCoefficient: 'native',
  scatteringAnisotropy: 'native',
  // Genuine per-channel σ_s (vec4 #3 → kernel.wgsl sigmaS) — native here,
  // unlike pt-webgl2's albedo reinterpretation.
  scatteringCoefficientRGB: 'native',
  // transmission tint + per-face roughness override + per-face normal map/scale
  // are packed, face-selected, and sampled by the full-tier material path.
  frontLayer: 'native',
  backLayer: 'native',
  // Full TMM evaluation; 8-layer cap (contract sanctions backend caps).
  thinFilmStack: 'native',
  // Anisotropic GGX sampler/eval/PDF and map rotation are live, including inverse
  // direct-light replay. Energy compensation uses the same conservative
  // projected-roughness Kulla-Conty approximation as anisotropyMap; the bounded
  // energy-compensation model is the final reason for the approximate grade.
  anisotropy: 'approximate',
  anisotropyRotation: 'approximate',
  // Contract-sanctioned escape hatch this backend deliberately does not read
  // (no warning — `extensions` is host-discretionary by design).
  extensions: 'unsupported',
});

// ── SHADOW-01 — per-backend shadow-flag support rows (2026-06-11) ─────────────
//
// Graded by code-read of the actual shadow-ray predicates:
//   pt-webgl2  — glsl/composeTraceGlsl.ts (`!material.castShadow && state.isShadowRay`
//                continuation gate; materialsTexture.ts s14 castShadow lane, fed from
//                the primitive flag via shared-bvh `splitMaterialsByCastShadow`) +
//                lightsTexture.ts s5.g castShadowDisabled lane consumed in
//                direct_light_contribution_function.glsl.js.
//   pt-webgpu  — materialPacking.ts vec4 #25 .w castShadowDisabled lane consumed by
//                intersectionCore.wgsl.ts `triShadowCastDisabled` in every any-hit
//                (occlusion) traversal (both tiers); emitterPacking.ts per-light
//                castShadowDisabled lanes consumed by the kernel/kernelLite NEE loops
//                + connect.wgsl BSDF-MIS area connections.
  //   walkaround — packingHelpers.ts bvh_material flag bit 0 + shared MaterialEntry
  //                flag bit 1 are consumed by cast-shadow-masked any-hit variants
  //                in DI, ReSTIR-GI, DDGI, GRIS reuse, and RC shadow visibility.
  //                Emitter castShadow:false rides
  //                analytic-lights, shared EmitterTri .w, DDGI/RC high-bit light
  //                kind flags, RC sunCastShadowDisabled, and the main direct-sun
  //                shade flag.

type ShadowSupportMatrix = Readonly<
  Record<'primitiveCastShadow' | 'emitterCastShadow', BackendSupportMode>
>;

/** walkaround-hybrid — primitive castShadow is honored by DI shadow predicates,
 *  ReSTIR-GI reservoir visibility, DDGI probe direct-light visibility, RC probe
 *  direct-light visibility, and GRIS reconnection visibility. The main pipeline
 *  uses traceSceneAlphaTintTransmittanceTextured for atlas-backed direct/ReSTIR/GI
 *  shadow visibility; DDGI/RC read shared MaterialEntry flag bit 1 through
 *  predicate-backed shared-BVH traversal.
 *  Emitter castShadow is honored across direct analytic/area NEE,
 *  DDGI fixture/sun probe lights, RC fixture/sun probe lights, and the main
 *  direct-sun shade path → 'native'. */
const WALKAROUND_SHADOWS: ShadowSupportMatrix = Object.freeze({
  primitiveCastShadow: 'native',
  emitterCastShadow: 'native',
});

/** pt-webgl2 — primitive castShadow rides the fork integrator's shadow-ray
 *  continuation gate (castShadow:false surfaces are transparent to shadow
 *  rays) → 'native'. Emitter castShadow is honored for analytic NEE lights
 *  (rect/disc/spot/point/directional via the s5.g lane) and mesh-area
 *  triangle-light NEE (uMeshLights s5.g), plus BDPT direct connections to
 *  the emitter endpoint (uBdptLightPathTex metadata row 3). Folded mesh-area
 *  emitter materials carry a dedicated meshEmitterCastShadowDisabled flag so
 *  the forward emissive-hit MIS side is skipped for ordinary non-specular
 *  BSDF hits while camera/specular-visible emission remains visible. */
const PT_WEBGL2_SHADOWS: ShadowSupportMatrix = Object.freeze({
  primitiveCastShadow: 'native',
  emitterCastShadow: 'native',
});

/** pt-webgpu — primitive castShadow is enforced in `traceMeshBvh`'s any-hit
 *  (occlusion) mode, which underlies EVERY traceAny call site on both tiers
 *  (NEE shadow rays, BSDF-MIS connections, ReSTIR-PT reconnection visibility,
 *  MNEE/caustic legs) → 'native'. `AnalyticPrimitive.castShadow` is a
 *  first-class validated field and the analytic intersection path honors it.
 *  Emitter castShadow is honored by the default kernel/kernelLite NEE loops + the connect.wgsl
 *  BSDF-MIS area-light connections for all 6 emitter kinds. Lite-tier
 *  directional NEE decodes the signed `cameraPos.w` mirror for the first
 *  directional flag. ReSTIR-PT suffix direct lighting also consumes the packed
 *  point/spot/rect/disc/mesh lanes and the packed N-directional records. MNEE
 *  caustic source connections honor the flag for point-light light-leg
 *  visibility plus finite-area rect/disc/mesh leg visibility. BDPT
 *  light-subpath bounce-0 records mirror the same emitter flag into row 4.x and
 *  the eye↔light connection skips the visibility ray for that emitter endpoint.
 *  SPPM photon-map source selection skips castShadow:false emitters and
 *  renormalizes the photon-source PDF over the remaining shadow-casting sources,
 *  so no-shadow emitters remain direct/camera/specular-visible without seeding
 *  caustic/shadow transport → 'native'. */
const PT_WEBGPU_SHADOWS: ShadowSupportMatrix = Object.freeze({
  primitiveCastShadow: 'native',
  emitterCastShadow: 'native',
});

type DenoiserSupportMatrix = Readonly<Record<EngineDenoiserMode, BackendSupportMode>>;

const WALKAROUND_DENOISERS: DenoiserSupportMatrix = Object.freeze({
  none: 'native',
  // Creation-time policy resolver: native on walkaround-hybrid. It chooses the
  // best concrete denoiser from host-supplied neural/OIDN assets, otherwise warns
  // and falls back to the existing atrous-variance path. It does NOT imply bundled
  // production neural weights.
  auto: 'native',
  atrous: 'native',
  'atrous-variance': 'native',
  'svgf-real': 'native',
  bmfr: 'native',
  'oidn-final': 'native',
  // Native dispatch path, but opt-in only: hosts must supply validated weights.
  // Repo checkpoint assets are validation-only and are never selected as a
  // production default.
  neural: 'native',
});

const WALKAROUND_DENOISER_SPATIAL_SHAPE_REQUIREMENTS: NonNullable<
  BackendSupportDetails['denoiserSpatialShapeRequirements']
> = Object.freeze({
  neural: Object.freeze({
    minWidth: 1,
    minHeight: 1,
    widthMultiple: 1,
    heightMultiple: 1,
  }),
});

const PT_WEBGL2_DENOISERS: DenoiserSupportMatrix = Object.freeze({
  none: 'native',
  auto: 'native',
  atrous: 'unsupported',
  'atrous-variance': 'unsupported',
  'svgf-real': 'unsupported',
  bmfr: 'unsupported',
  'oidn-final': 'native',
  neural: 'unsupported',
});

const PT_WEBGPU_DENOISERS: DenoiserSupportMatrix = Object.freeze({
  none: 'native',
  auto: 'native',
  atrous: 'unsupported',
  'atrous-variance': 'unsupported',
  'svgf-real': 'unsupported',
  bmfr: 'unsupported',
  'oidn-final': 'native',
  neural: 'unsupported',
});

// ── Shared mutation/method constants (D1.4) ──────────────────────────────────
//
// Extracted to eliminate copy-paste drift between the three backend records.
// Deep-equal to the prior inline literals (byte-identical ledger output).

/** pt-webgl2 mutations — scalar material edits, emitter edits, and environment
 *  swaps update only the affected GL scene textures. Mesh-area emitter edits
 *  refresh both the folded emissive material texture and the mesh-light NEE
 *  texture without rebuilding the merged BVH.
 *  Same-topology transform/positions/normal/uv/tangent/color edits refit BVH
 *  bounds on the retained node topology and use texSubImage* writes into existing
 *  BVH position/bounds, material-flag, and attribute textures. Primitive layout
 *  changes that alter topology, material slots, or analytic fallback shape still
 *  use a full scene-texture repack. Topology add/remove rebuilds the WebGL2
 *  geometry/material/atlas/BVH texture pack for list edits after applying the
 *  same analytic-to-mesh fallback tessellation used by setScene(). Resize is
 *  native: it reallocates render targets and resets accumulation without
 *  scene/BVH work. Atomic multi-domain lighting replacement is implemented,
 *  but currently validates and rebuilds the complete scene-texture pack. */
const PT_WEBGL2_MUTATIONS: BackendPromiseRecord['supportDetails']['mutations'] = Object.freeze({
  transform: 'native',
  positions: 'native',
  material: 'native',
  emitter: 'native',
  topology: 'fallback-rebuild',
  addPrimitive: 'fallback-rebuild',
  removePrimitive: 'fallback-rebuild',
  environment: 'native',
  resize: 'native',
  lighting: 'fallback-rebuild',
});

/** pt-webgpu mutations — geometry/transform/topology/emitter/env patches have
 *  targeted native paths. Material scalar edits that live in `materialsBuffer`
 *  update one slot in-place. Full-tier scalar descriptor edits
 *  (`alphaMode`/cutoff/opacity, AO/light/env intensity, normal/bump/
 *  clearcoat-normal scale, anisotropy) also rewrite the matching
 *  `materialTexDescriptorsBuffer` slice. Material texture handles, layer
 *  normal descriptors, displacement geometry, and lite-tier descriptor scalars
 *  still fall back to full scene repack so texture arrays and unsupported
 *  descriptors stay coherent. The coarse row is therefore `fallback-rebuild`
 *  rather than `native`. Add/remove are fallback-rebuild (insert/evict forces
 *  a full BLAS/TLAS repack). Resize eagerly replaces only render-target state;
 *  lighting replaces only emitter/environment/light-tree buffers. */
const PT_WEBGPU_MUTATIONS: BackendPromiseRecord['supportDetails']['mutations'] = Object.freeze({
  transform: 'native',
  positions: 'native',
  material: 'fallback-rebuild',
  emitter: 'native',
  topology: 'native',
  addPrimitive: 'fallback-rebuild',
  removePrimitive: 'fallback-rebuild',
  environment: 'native',
  resize: 'native',
  lighting: 'native',
});

/** Method-promise fields that are true (or false) identically across all three
 *  shipping backends.  Spread into each record; backend-specific fields
 *  (setSize, updateLighting, debug, createInverseSession, getRestirPtResultBuffer,
 *  getProgressiveSeedTexture, seedAccumulator, giStatePersistence) are added
 *  after the spread. */
const COMMON_METHOD_PROMISES: Pick<
  BackendMethodPromises,
  | 'updatePrimitive'
  | 'updateEmitter'
  | 'updateEnvironment'
  | 'addPrimitive'
  | 'removePrimitive'
  | 'onFrame'
  | 'onProgress'
  | 'getScene'
  | 'onError'
  | 'onWarning'
  | 'captureFrame'
> = Object.freeze({
  updatePrimitive: true,
  updateEmitter: true,
  updateEnvironment: true,
  addPrimitive: true,
  removePrimitive: true,
  onFrame: true,
  onProgress: true,
  getScene: true,
  onError: true,
  onWarning: true,
  captureFrame: true,
});
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Machine-checkable backend contract truth table.
 *
 * Tests in backend packages assert runtime behavior against this ledger so
 * capability/method drift fails mechanically.
 */
export const BACKEND_PROMISE_LEDGER: Readonly<Record<BackendId, BackendPromiseRecord>> = {
  'walkaround-hybrid': {
    supportsIncrementalScene: true,
    incrementalPatchSupport: ALL_PATCHES_SUPPORTED,
    supportsAddRemovePrimitive: true,
    // The pipeline exposes normal/depth, albedo, motion, and the freshest side
    // of its full-resolution Welford variance ping-pong on every rendered frame.
    supportsAuxBuffers: true,
    accumulates: false,
    // The render-scene path ingests mesh / skinned-mesh / instanced-mesh;
    // analytic primitives are accepted in the authored scene and converted to
    // generated MeshPrimitive fallbacks before BVH/GI ingestion consumes them.
    // instanced-mesh IS genuine here — walkaround renders instances via the
    // TLAS per-instance traversal path.
    supportedPrimitiveKinds: ['mesh', 'skinned-mesh', 'instanced-mesh', 'analytic'],
    // rect-area/disc-area → ReSTIR-DI emitter tris + DDGI fixtures; mesh-area →
    // mesh emissive material; directional → DDGI `sun` light. point/spot →
    // analytic shade/OIT direct-light terms plus DDGI/RC fixture lights. Spots
    // carry real cone data (spotAxis + cosInner/cosOuter) and evalPointLight in
    // the probe shader applies smoothstep cone falloff. See
    // coreEmittersToDDGILights and shade.wgsl's H41 analytic NEE loop.
    // `directional` → coreEmittersToDDGILights maps it to a `sun` DDGILight
    // carrying its real direction + intensity + colour; the host single-counts it
    // by setting the DDGI sun-intensity multiplier to 1. ReSTIR-DI harvests no
    // directional emitter, so there is no DI double-count.
    supportedEmitterKinds: ['directional', 'rect-area', 'disc-area', 'point', 'spot', 'mesh-area'],
    // procedural-sky bakes through resolveHybridEnvironment into a finite
    // Preetham equirect + CDF. Approximate means model/resolution limits, not
    // scalar-only data loss.
    supportedEnvironmentKinds: ['none', 'hdri', 'procedural-sky'],
    supportedAnalyticShapes: ['sphere', 'box', 'capsule', 'cylinder', 'h-channel-came'],
    presentationMode: 'swapchain-required',
    supportDetails: {
      primitives: {
        mesh: 'native',
        'skinned-mesh': 'native',
        'instanced-mesh': 'native',
        analytic: 'fallback-generated-mesh',
      },
      emitters: WALKAROUND_EMITTERS,
      environments: {
        none: 'native',
        // B3 + Wave 4 (2026-06-10) — directional IBL COMPLETE: sinθ-weighted
        // equirect inverse-CDFs at scene-group bindings 15-19; envImportanceSample
        // is a DI NEE candidate in the RIS loop (`ris.wgsl`); GI-escape rays read
        // the real map (`risGi.wgsl` envRadiance) and the NRC variant matches;
        // DDGI probe misses sample the HDRI (procedural fallback when none); RC
        // last-cascade env is bound; `updateEnvironment` rebuilds the directional
        // CDFs at runtime via resolveHybridEnvironment → updateDirectionalEnvironment.
        // Native by source semantics; runtime evidence is tracked separately.
        hdri: 'native',
        // resolveHybridEnvironment handles procedural-sky by baking the shared
        // Preetham model to an equirect/CDF and preserving scalar sky as fallback.
        // 'approximate' is the honest grade for a finite baked model.
        'procedural-sky': 'approximate',
      },
      analyticShapes: ANALYTIC_SHAPES_FALLBACK_GENERATED_MESH,
      materials: WALKAROUND_MATERIALS,
      materialProfiles: {
        // Delta interfaces use bounded per-interface Snell/Fresnel transport.
        deltaTransmission: 'approximate',
        // Positive roughness uses an unfloored Heitz VNDF proposal paired with
        // the Walter rough-BTDF PDF/weight; zero roughness is a discrete event.
        roughTransmission: 'approximate',
        // Face layers and optical top-layer transmittance participate in the
        // bounded dielectric walk; rich reflection lobes remain realtime-reduced.
        layeredTransmission: 'approximate',
        // Normal/bump maps are sampled at every supported dielectric boundary;
        // geometric normals continue to own medium enter/exit classification.
        normalMappedTransmission: 'approximate',
        participatingMedia: 'approximate',
        faceLayers: 'approximate',
      },
      shadows: WALKAROUND_SHADOWS,
      denoisers: WALKAROUND_DENOISERS,
      denoiserSpatialShapeRequirements:
        WALKAROUND_DENOISER_SPATIAL_SHAPE_REQUIREMENTS,
      motionVectors: {
        units: 'pixels',
        direction: 'previous-minus-current',
        geometry: 'camera-only',
        sceneMutationPolicy: 'reset-history',
      },
      causticStrategies: {
        'refractive-trace': {
          mode: 'approximate',
          estimatorScope:
            'camera-visible diffuse receiver <- up-to-four specular transmission interfaces <- directional emitter; two bounded stratified RGB candidates, not Newton manifold NEE',
          emitterKinds: {
            directional: 'native',
            point: 'unsupported',
            spot: 'unsupported',
            'rect-area': 'unsupported',
            'disc-area': 'unsupported',
            'mesh-area': 'unsupported',
            environment: 'unsupported',
          },
          volumeScattering: 'unsupported',
          incompatibleFeatures: [
            'manifold-nee',
            'photon-map',
          ],
        },
        'manifold-nee': {
          mode: 'approximate',
          estimatorScope:
            'camera-visible diffuse receiver <- one-to-eight mapped rough/delta transmission events <- sampled explicit or environment endpoint; bounded SMS inverse-basin correction',
          emitterKinds: {
            directional: 'native',
            point: 'native',
            spot: 'native',
            'rect-area': 'native',
            'disc-area': 'native',
            'mesh-area': 'native',
            environment: 'native',
          },
          volumeScattering: 'unsupported',
          incompatibleFeatures: [
            'refractive-trace',
            'photon-map',
          ],
        },
      },
      mutations: {
        transform: 'native',
        positions: 'native',
        material: 'native',
        emitter: 'native',
        topology: 'fallback-rebuild',
        addPrimitive: 'fallback-rebuild',
        removePrimitive: 'fallback-rebuild',
        environment: 'approximate',
        resize: 'native',
        lighting: 'native',
      },
    },
    methodPromises: {
      ...COMMON_METHOD_PROMISES,
      // updateEnvironment note: env-only runtime update IS implemented
      // (HybridEngine.updateEnvironment): maps SceneEnvironment onto sky fallback
      // scalars plus directional equirect/CDF data when CPU-readable HDRI or
      // procedural-sky payloads are available, invalidates the DDGI probe cache
      // + resets the temporal accumulator, NO BVH rebuild. ✓ via spread.
      //
      // addPrimitive/removePrimitive note: implemented via full setScene-rebuild;
      // see supportsAddRemovePrimitive above. ✓ via spread.
      //
      // GPU error surface: device.uncapturederror (throttled) + device.lost. ✓ via spread.
      //
      // GPU→CPU pixel readback: resolvedTexture for 'linear'; 'output' rejects
      // (swap-chain write, no engine-owned display buffer to read back). ✓ via spread.
      setSize: true,
      updateLighting: true,
      debug: true,
      // walkaround-hybrid does NOT implement createInverseSession (differentiable RT
      // is pt-webgpu only in the current slice).
      createInverseSession: false,
      // getRestirPtResultBuffer is pt-webgpu only (ReSTIR-PT reuse).
      getRestirPtResultBuffer: false,
      // Walkaround presents directly into the host-provided swap-chain view.
      getPresentationSource: false,
      // walkaround-hybrid exposes the post-denoise resolvedTexture as a seed SOURCE for
      // the progressive walkaround→PT handoff (see getProgressiveSeedTexture).
      getProgressiveSeedTexture: true,
      // seedAccumulator is the SINK side of the handoff — pt-webgpu only. Walkaround
      // resamples every frame (no persistent accumulator to seed).
      seedAccumulator: false,
      // DDGI probe-atlas + ReSTIR-GI reservoir export/import — walkaround-hybrid only.
      giStatePersistence: true,
    },
    frameInputPromises: {
      honorsViewportPerFrame: true,
      requiresSwapChainView: true,
      honorsPerFrameBounces: false,
    },
  },
  'pt-webgl2': {
    supportsIncrementalScene: true,
    incrementalPatchSupport: ALL_PATCHES_SUPPORTED,
    supportsAddRemovePrimitive: true,
    // WebGL2 full-tier exposes useful normalDepth + albedo MRT products, but the
    // core `supportsAuxBuffers` flag promises BOTH FrameOutput.variance and
    // FrameOutput.motionVectors. Those are not emitted by pt-webgl2 today, so the
    // contract-level flag stays false even when OIDN-style albedo/normal inputs
    // are available.
    supportsAuxBuffers: false,
    accumulates: true,
    // The native WebGL2 packer ingests triangle geometry. Authored analytic
    // primitives are accepted at the contract boundary and tessellated to
    // generated MeshPrimitive fallbacks before scene texture/BVH upload.
    // instanced-mesh IS supported: the backend scene pack preserves each
    // instance at its real per-instance world transform.
    supportedPrimitiveKinds: ['mesh', 'skinned-mesh', 'instanced-mesh', 'analytic'],
    supportedEmitterKinds: ['directional', 'rect-area', 'disc-area', 'point', 'spot', 'mesh-area'],
    supportedEnvironmentKinds: ['none', 'hdri', 'procedural-sky'],
    supportedAnalyticShapes: ['sphere', 'box', 'capsule', 'cylinder', 'h-channel-came'],
    presentationMode: 'offscreen-texture',
    supportDetails: {
      primitives: {
        mesh: 'native',
        // 2026-06-10 (Wave 2): pt-webgl2 solves the pose at ingestion via core
        // `solveSkin` (bones + bindMatrix + morph targets; see
        // scene/solveSkinPrimitives.ts); `updatePrimitive({bones})` re-solves
        // through the full-rebuild path. Promoted approximate→native.
        'skinned-mesh': 'native',
        'instanced-mesh': 'native',
        // Authored analytic primitives are tessellated via core's
        // analyticPrimitiveToMesh() before the triangle-only WebGL2 packer.
        analytic: 'fallback-generated-mesh',
      },
      emitters: ALL_EMITTERS_NATIVE,
      environments: {
        none: 'native',
        hdri: 'native',
        // Preetham 1999 analytic daylight model baked to a 256x128 equirect
        // map and routed through the HDRI importance-sampling path. All scene
        // fields are consumed; 'approximate' reflects finite bake resolution.
        'procedural-sky': 'approximate',
      },
      analyticShapes: ANALYTIC_SHAPES_FALLBACK_GENERATED_MESH,
      materials: PT_WEBGL2_MATERIALS,
      shadows: PT_WEBGL2_SHADOWS,
      denoisers: PT_WEBGL2_DENOISERS,
      causticStrategies: {
        bdpt: {
          mode: 'native',
          estimatorScope:
            'bounded general BDPT: power-weighted light subpaths store 1..8 vertices (default 4); finite-emitter c=0 plus c>=1 light/eye surface and participating-medium vertices carry an exact eight-entry nested homogeneous-medium stack, RGB or hero-wavelength Beer visibility, authored sigma_a/sigma_s, and Henyey-Greenstein phase transport through Veach power-heuristic MIS; thicknessMap modulates the core contract\'s authored surface-volume attenuation; distant paths are disjointly owned by primary c=0 NEE, camera/delta forward escape, or c>=1 BDPT',
          emitterKinds: {
            directional: 'native',
            point: 'native',
            spot: 'native',
            'rect-area': 'native',
            'disc-area': 'native',
            'mesh-area': 'native',
            environment: 'native',
          },
          volumeScattering: 'native',
          incompatibleFeatures: [],
        },
      },
      // D1 (2026-07-20) — pt-webgl2 packs up to 35 thin-film layers (GLSL stride);
      // MUST match materialsTexture.ts THIN_FILM_LAYER_LIMIT.
      thinFilmLayerLimit: 35,
      // buildCapabilities() mirrors this matrix: same-topology transform and
      // position/attribute edits are native subimage updates; topology/list
      // edits still fallback-rebuild. Resize is a native render-target
      // realloc/reset path.
      mutations: PT_WEBGL2_MUTATIONS,
    },
    methodPromises: {
      ...COMMON_METHOD_PROMISES,
      // Context-lost surface: webglcontextlost canvas event. ✓ via spread.
      // GPU→CPU pixel readback: accum FBO (RGBA32F, rows flipped to top-left)
      // for 'linear'; present FBO for 'output'. ✓ via spread.
      setSize: true,
      // Atomic `{ emitters?, environment? }` candidate validation + scene
      // texture replacement. A failed validation/upload preserves the prior
      // retained scene and GL resources.
      updateLighting: true,
      // T3.G #30 — pt-webgl2 exposes debug.pickPrimitive and advertises
      // capabilities.debugSurface=true.
      debug: true,
      // pt-webgl2 implements the inverse-rendering API surface with the safe
      // backend-agnostic finite-difference method. It intentionally does not
      // expose pt-webgpu's scoped path-replay adjoint hook.
      createInverseSession: true,
      getRestirPtResultBuffer: false,
      getPresentationSource: false,
      getProgressiveSeedTexture: false,
      seedAccumulator: false,
      giStatePersistence: false,
    },
    frameInputPromises: {
      honorsViewportPerFrame: true,
      requiresSwapChainView: false,
      honorsPerFrameBounces: true,
    },
  },
  'pt-webgpu': {
    supportsIncrementalScene: true,
    incrementalPatchSupport: ALL_PATCHES_SUPPORTED,
    supportsAddRemovePrimitive: true,
    supportsAuxBuffers: true,
    accumulates: true,
    supportedPrimitiveKinds: ['mesh', 'instanced-mesh', 'analytic', 'skinned-mesh'],
    supportedEmitterKinds: ['directional', 'point', 'spot', 'rect-area', 'disc-area', 'mesh-area'],
    supportedEnvironmentKinds: ['none', 'hdri', 'procedural-sky'],
    supportedAnalyticShapes: ['sphere', 'box', 'capsule', 'cylinder', 'h-channel-came'],
    presentationMode: 'offscreen-texture',
    supportDetails: {
      primitives: {
        mesh: 'native',
        // 2026-06-10 (Wave 2): pt-webgpu solves the pose at ingestion via core
        // `solveSkin` (bones + bindMatrix + morph targets; see
        // scene/uploadSceneBuffers.ts applySolveSkinToScene) and re-solves on
        // `updatePrimitive({bones})` via the mutation router's bones fast path.
        // Promoted approximate→native.
        'skinned-mesh': 'native',
        'instanced-mesh': 'native',
        analytic: 'native',
      },
      emitters: PT_WEBGPU_EMITTERS,
      environments: {
        none: 'native',
        hdri: 'native',
        // Preetham 1999 analytic daylight model (baked to 256×128 equirect,
        // routed through the HDRI importance-sampling path).  All scene fields
        // (turbidity, rayleigh, mieCoefficient, mieDirectionalG, sunDirection,
        // intensity) are consumed.  'approximate' reflects the bake resolution
        // (256×128) vs a continuous analytical eval in the kernel.
        'procedural-sky': 'approximate',
      },
      analyticShapes: PT_WEBGPU_ANALYTIC_SHAPES_NATIVE,
      materials: PT_WEBGPU_MATERIALS,
      shadows: PT_WEBGPU_SHADOWS,
      denoisers: PT_WEBGPU_DENOISERS,
      motionVectors: {
        units: 'pixels',
        direction: 'current-minus-previous',
        geometry: 'camera-only',
        sceneMutationPolicy: 'reset-history',
      },
      mutations: PT_WEBGPU_MUTATIONS,
      causticStrategies: {
        'manifold-nee': {
          mode: 'native',
          estimatorScope:
            'camera-visible finite surface receiver <- one-to-eight planar geometric-normal mesh/instanced/skinned delta interfaces <- sampled explicit or environment endpoint; bounded Newton/SMS solve; analytic delta interfaces, varying interface normals, and normal/bump/layer-normal mapped interfaces fail closed before upload',
          emitterKinds: {
            directional: 'native',
            point: 'native',
            spot: 'native',
            'rect-area': 'native',
            'disc-area': 'native',
            'mesh-area': 'native',
            environment: 'native',
          },
          volumeScattering: 'unsupported',
          incompatibleFeatures: [],
        },
        'photon-map': {
          mode: 'native',
          estimatorScope:
            'camera-visible finite surface or homogeneous-medium collision receiver <- one-or-more production delta surface events <- sampled light; surface uses a disk density and volume uses a medium-identity-filtered HG sphere density with independent progressive state',
          emitterKinds: {
            directional: 'native',
            point: 'native',
            spot: 'native',
            'rect-area': 'native',
            'disc-area': 'native',
            'mesh-area': 'native',
            environment: 'native',
          },
          volumeScattering: 'native',
          incompatibleFeatures: [],
        },
      },
      // pt-webgpu's independently enabled `bdpt:true` mode is deliberately a
      // bounded explicit-connection family. It does not allocate a camera-We
      // film-splat pass for t=1 light tracing, and its MIS denominator masks
      // every unallocated strategy rather than pretending it was sampled.
      bidirectionalPathTracing: {
        mode: 'bounded-explicit-connections',
        maxLightVertices: 8,
        maxEyeVertices: 8,
        pureEyeStrategy: 'partitioned-eye-estimator',
        cameraSplatStrategy: 'native',
        misDenominator: 'sampled-strategies-only',
      },
      samplingSequences: {
        default: 'pcg',
        modes: { pcg: 'native', sobol: 'native' },
        sobol: {
          lowDiscrepancyDimensions: 512,
          continuation: 'independent-pcg',
          sampleBlockSize: 65536,
          frameIndexPeriod: 4294967296,
        },
      },
      // D1 (2026-07-20) — pt-webgpu packs up to 8 thin-film layers (WGSL loop
      // stride); MUST match materialPacking.ts THIN_FILM_LAYER_LIMIT + WGSL 8u.
      thinFilmLayerLimit: 8,
    },
    methodPromises: {
      ...COMMON_METHOD_PROMISES,
      // GPU error surface: device.uncapturederror (throttled) + device.lost. ✓ via spread.
      // GPU→CPU pixel readback: accumTexture (rgba16float decoded to f32) for
      // 'linear'; presentTexture (rgba16float) for 'output'. ✓ via spread.
      setSize: true,
      updateLighting: true,
      debug: true,
      // pt-webgpu implements the inverse-rendering API surface. Explicit
      // finite-difference is available across both tiers. Full-tier path replay
      // is fail-closed and currently certified only for single-bounce material
      // emissive; every out-of-domain request throws before applying initial
      // parameter values instead of silently changing the selected method.
      createInverseSession: true,
      // getRestirPtResultBuffer is gated on active feature
      // 'pt-webgpu-one-edge-gris-reconnection'. The method IS wired on the engine class (returns
      // null when the feature is off); the ledger records it as implemented
      // because the method EXISTS on every pt-webgpu instance, not just those
      // with ReSTIR-PT enabled.
      getRestirPtResultBuffer: true,
      // Exposes the live WebGPU device/texture pair used for host presentation.
      getPresentationSource: true,
      // pt-webgpu does not expose a seed SOURCE (the resolvedTexture/progressive-seed
      // source path is walkaround-hybrid only).
      getProgressiveSeedTexture: false,
      // pt-webgpu implements seedAccumulator — the SINK side of the
      // progressive walkaround→PT handoff (decaying-prior injection into the
      // Welford running-mean accumulator).
      seedAccumulator: true,
      // GI-state persistence is walkaround-hybrid only (DDGI/ReSTIR-GI).
      // pt-webgpu's accumulator is a different convergence structure (Welford
      // running mean, no probe atlas) and has no export/import surface.
      giStatePersistence: false,
    },
    frameInputPromises: {
      honorsViewportPerFrame: true,
      requiresSwapChainView: false,
      honorsPerFrameBounces: true,
    },
  },
};
