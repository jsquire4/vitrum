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
// Material
// ────────────────────────────────────────────────────────────────────────────

/** Generic PBR material — superset of the standard PBR fields with optional
 *  Disney-BSDF lobes for backends that support them. The `extensions` field
 *  is the escape hatch: backends can read backend-specific data from it
 *  without polluting the core type (e.g., normalMap-perturbed shadow ray
 *  parameters, the Phase 4 contribution).
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

  // ── Backend escape hatch ────────────────────────────────────────────────
  /** Backends may read keyed fields from here for backend-specific features.
   *  Core never inspects this map. */
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
