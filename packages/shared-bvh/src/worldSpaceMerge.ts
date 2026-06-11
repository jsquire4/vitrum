/**
 * worldSpaceMerge.ts — world-space merged tri-stream / BVH builder
 * from a `@vitrum/core` `Scene`.
 *
 * This is the canonical ingestion path for consumers that need WORLD-space
 * geometry instead of the local-space BLAS + separate TLAS matrices emitted by
 * `packSceneFromCore`:
 *
 *   1. the ReSTIR emitter list (triangle area / world face-normal / centroid /
 *      AABB — `walkaround-hybrid/restir/bvhCore.ts`),
 *   2. the DDGI merged BVH, and
 *   3. the RC merged BVH.
 *
 * `mergeWorldSpaceFromCore` iterates `scene.primitives`, transforms each
 * primitive's local `positions`/`normals` into world space by the core
 * `transform`, concatenates the streams, and builds one merged BVH via
 * `buildArrayBvh`.
 *
 * ── Parity with `buildSceneBVH` (the migration gate) ──────────────────────────
 *
 * The merged vertex stream (positions + normals, pre-BVH-reorder) is built from
 * core transforms:
 *   • positions  — `worldPos = transform · localPos` (raw 4×4).
 *   • normals    — `worldN = normalize(normalMatrix · localN)` where
 *     `normalMatrix` is the inverse-transpose of the upper-left 3×3.
 *   • winding    — when `det(transform) < 0`, the builder swaps v0↔v2 of every
 *     triangle so mirrored transforms keep front-facing winding.
 *   • order      — primitives are concatenated in `scene.primitives` order.
 *
 * To make both the "stream parity" and the "set / AABB / ray-query equivalence"
 * checkable, the result carries the pre-BVH-reorder merged stream
 * ({@link WorldSpaceMergeResult.mergedIndices} / `mergedTriMaterialId`) ALONGSIDE
 * the post-build BVH-ordered `indices` / `triMaterialId`.
 *
 */

import type { Mat4, MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import type { PlainAabb } from './aabb.js';
import { buildArrayBvh } from './buildArrayBvh.js';

const IDENTITY_MAT4: readonly number[] = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];

// ──────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────

/** Per-source-primitive (or per-instance) vertex range in the merged buffer.
 *  Mirrors `SceneBVHCommonResult.meshVertexRanges` (the THREE path) so the
 *  transform-only refit fast path can address one primitive's world vertices in
 *  place. `name` is the primitive id (for instanced primitives every instance
 *  shares the same id — the per-instance split is implicit in the contiguous
 *  ranges). */
export interface MergedMeshVertexRange {
  readonly name: string;
  readonly vertexStart: number;
  readonly vertexCount: number;
  readonly triStart: number;
  readonly triCount: number;
}

export interface WorldSpaceMergeResult {
  /**
   * Packed 32-byte BVH-node buffer (single root), byte-layout-identical to
   * `buildSceneBVH`'s `bvhNodes` (8 × u32 / node, relative interior offsets,
   * leaf `0xFFFF0000 | count`). NOTE: the tree TOPOLOGY differs from
   * `buildSceneBVH` (different SAH builder — see module docstring R1); only the
   * per-node byte LAYOUT is shared.
   */
  readonly bvhNodes: Float32Array;

  /** Merged world-space positions. Stride = {@link positionStrideFloats}. */
  readonly positions: Float32Array;
  readonly positionStrideFloats: 3 | 4;

  /**
   * BVH-traversal-ordered triangle indices (3 u32 / triangle — stride 3, the
   * `array<vec3u>` form, same as `buildSceneBVH`). Permuted by the SAH build;
   * NOT the same permutation as `buildSceneBVH` (R1). For the unpermuted merge
   * order use {@link mergedIndices}.
   */
  readonly indices: Uint32Array;
  readonly bvhIndexStride: 3;

  /** Per-triangle materialId, BVH-reordered to match {@link indices}. */
  readonly triMaterialId: Uint32Array;

  /** Merged world-space normals (THREE normal-matrix + normalize). Stride =
   *  {@link positionStrideFloats}. */
  readonly normals: Float32Array;

  /** Merged per-vertex texture coords (stride 2, merge/vertex order — same order as
   *  {@link positions}/{@link normals}; the BVH reorders triangles not vertices).
   *  (0,0) for vertices whose source primitive carries no UVs. */
  readonly uvs: Float32Array;

  /**
   * Pre-BVH-reorder merged triangle indices (3 u32 / triangle, MERGE order —
   * the order `StaticGeometryGenerator` concatenates, before any SAH permute).
   * This is the order-deterministic stream the golden-parity test pins against
   * `buildSceneBVH` (whose `positions`/`normals` are likewise never reordered by
   * the BVH build). Emitter-list consumers that want the merge order (not the
   * BVH order) read this.
   */
  readonly mergedIndices: Uint32Array;

  /** Per-triangle materialId in {@link mergedIndices} (merge) order. */
  readonly mergedTriMaterialId: Uint32Array;

  /**
   * Deduped source materials in first-seen primitive order — the core
   * `MaterialSpec[]` counterpart to `buildSceneBVH`'s `THREE.Material[]`.
   * `triMaterialId` / `mergedTriMaterialId` index into this list. Dedup is by
   * the same structural signature `snapshotPreBuildMaterials` uses (so React/R3F
   * material churn collapses to the structural minimum — see {@link materialSig}).
   */
  readonly materials: readonly MaterialSpec[];

  /** World-space AABB of the merged geometry (THREE-free `{min,max}` in place of
   *  `THREE.Box3`). Float-identical to `buildSceneBVH`'s `boundingBox` for an
   *  equivalent scene. */
  readonly boundingBox: PlainAabb;

  /** Per-source-primitive vertex ranges (mirrors the THREE path). */
  readonly meshVertexRanges: readonly MergedMeshVertexRange[];

  /** Vertex count in the merged buffer (positions.length / stride). */
  readonly vertexCount: number;
  /** Triangle count in the merged buffer. */
  readonly triangleCount: number;
}

export interface WorldSpaceMergeOptions {
  /**
   * Position / normal buffer element stride in floats:
   *   4 → 16-byte vec3f-aligned layout (default; the WGSL `array<vec3f>` form
   *       the DDGI/RC merged-BVH shaders read, matching
   *       `buildSceneBVH({positionStride:4})`).
   *   3 → 12-byte packed layout (raster/TSL; matches `buildSceneBVH`'s default).
   */
  readonly positionStride?: 3 | 4;

  /**
   * Caller-supplied primitive filter — returns true for primitives that should
   * contribute to the merged BVH. Defaults to {@link DEFAULT_MERGE_FILTER}
   * (every mesh-like primitive: mesh / instanced-mesh / skinned-mesh; analytic
   * primitives are skipped — they have no triangle stream). Mirrors
   * `buildSceneBVH`'s mesh filter ("any mesh with a position attribute").
   */
  readonly filter?: (primitive: ScenePrimitive) => boolean;
}

// ──────────────────────────────────────────────────────────────────────────
// Mesh-like predicate + default filter
// ──────────────────────────────────────────────────────────────────────────

type MeshLikePrimitive = Extract<
  ScenePrimitive,
  { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }
>;

function isMeshLike(primitive: ScenePrimitive): primitive is MeshLikePrimitive {
  return (
    primitive.kind === 'mesh' ||
    primitive.kind === 'skinned-mesh' ||
    primitive.kind === 'instanced-mesh'
  );
}

/** Default merge filter — every mesh-like primitive contributes (analytic
 *  primitives have no triangle stream and are skipped). Mirrors the DDGI mesh
 *  filter ("any visible mesh with a position attribute"). */
export const DEFAULT_MERGE_FILTER = (primitive: ScenePrimitive): boolean =>
  isMeshLike(primitive);

// ──────────────────────────────────────────────────────────────────────────
// World transforms — bit-for-bit mirrors of THREE / StaticGeometryGenerator
// ──────────────────────────────────────────────────────────────────────────

/**
 * `worldPos = m · localPos` for a column-major 4×4 `m` (= THREE's
 * `Vector3.applyMatrix4`, including the perspective divide). Identical
 * arithmetic to `scenePack.ts:transformPoint` BUT keeps the THREE `w`-divide
 * convention (`w = 1/(…); xyz *= w`) so the f32 round-off matches SGG exactly.
 */
function applyMatrix4(m: ArrayLike<number>, x: number, y: number, z: number): [number, number, number] {
  const e0 = m[0] ?? 0, e4 = m[4] ?? 0, e8 = m[8] ?? 0, e12 = m[12] ?? 0;
  const e1 = m[1] ?? 0, e5 = m[5] ?? 0, e9 = m[9] ?? 0, e13 = m[13] ?? 0;
  const e2 = m[2] ?? 0, e6 = m[6] ?? 0, e10 = m[10] ?? 0, e14 = m[14] ?? 0;
  const e3 = m[3] ?? 0, e7 = m[7] ?? 0, e11 = m[11] ?? 0, e15 = m[15] ?? 0;
  const w = 1 / (e3 * x + e7 * y + e11 * z + e15);
  return [
    (e0 * x + e4 * y + e8 * z + e12) * w,
    (e1 * x + e5 * y + e9 * z + e13) * w,
    (e2 * x + e6 * y + e10 * z + e14) * w,
  ];
}

/**
 * Normal matrix (upper-left 3×3 inverse-transpose) of a column-major 4×4, as a
 * length-9 row-major-by-THREE-convention array — bit-for-bit THREE's
 * `Matrix3.getNormalMatrix(m4) = setFromMatrix4(m4).invert().transpose()`.
 *
 * `setFromMatrix4` lays out the upper-left 3×3 of the column-major m4 into the
 * Matrix3 as: n11=m[0], n21=m[1], n31=m[2], n12=m[4], n22=m[5], n32=m[6],
 * n13=m[8], n23=m[9], n33=m[10] (so te = [m0,m1,m2, m4,m5,m6, m8,m9,m10]).
 * `invert()` then writes te' via THREE's exact cofactor expressions, and
 * `transpose()` swaps the off-diagonal pairs. We fold invert+transpose into the
 * final element assignment so there's a single, mirror-able expression set.
 */
function getNormalMatrix3(m: ArrayLike<number>): Float64Array {
  // setFromMatrix4(m4) → Matrix3 elements (te):
  const n11 = m[0] ?? 0, n21 = m[1] ?? 0, n31 = m[2] ?? 0;
  const n12 = m[4] ?? 0, n22 = m[5] ?? 0, n32 = m[6] ?? 0;
  const n13 = m[8] ?? 0, n23 = m[9] ?? 0, n33 = m[10] ?? 0;

  // invert(): THREE's exact cofactor formula (Matrix3.invert).
  const t11 = n33 * n22 - n32 * n23;
  const t12 = n32 * n13 - n33 * n12;
  const t13 = n23 * n12 - n22 * n13;
  const det = n11 * t11 + n21 * t12 + n31 * t13;

  const inv = new Float64Array(9);
  if (det === 0) {
    // THREE sets all nine elements to 0 on a singular matrix.
    return inv;
  }
  const detInv = 1 / det;
  // inv = te' (the inverted Matrix3), THREE's exact element assignments.
  inv[0] = t11 * detInv;
  inv[1] = (n31 * n23 - n33 * n21) * detInv;
  inv[2] = (n32 * n21 - n31 * n22) * detInv;
  inv[3] = t12 * detInv;
  inv[4] = (n33 * n11 - n31 * n13) * detInv;
  inv[5] = (n31 * n12 - n32 * n11) * detInv;
  inv[6] = t13 * detInv;
  inv[7] = (n21 * n13 - n23 * n11) * detInv;
  inv[8] = (n22 * n11 - n21 * n12) * detInv;

  // transpose() in place: swap (1,3),(2,6),(5,7).
  const out = new Float64Array(9);
  out[0] = inv[0]!; out[4] = inv[4]!; out[8] = inv[8]!;
  out[1] = inv[3]!; out[3] = inv[1]!;
  out[2] = inv[6]!; out[6] = inv[2]!;
  out[5] = inv[7]!; out[7] = inv[5]!;
  return out;
}

/**
 * `worldN = normalize( normalMatrix · localN )` — bit-for-bit THREE's
 * `Vector3.applyNormalMatrix(m3) = applyMatrix3(m3).normalize()`. `nm` is the
 * length-9 Matrix3 from {@link getNormalMatrix3}; `applyMatrix3` reads it as
 * `x' = nm0·x + nm3·y + nm6·z`, etc. (THREE's `Vector3.applyMatrix3`).
 */
function applyNormalMatrix(
  nm: ArrayLike<number>,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  const nx = (nm[0] ?? 0) * x + (nm[3] ?? 0) * y + (nm[6] ?? 0) * z;
  const ny = (nm[1] ?? 0) * x + (nm[4] ?? 0) * y + (nm[7] ?? 0) * z;
  const nz = (nm[2] ?? 0) * x + (nm[5] ?? 0) * y + (nm[8] ?? 0) * z;
  // THREE's Vector3.normalize: divide by length, or by 1 when length 0.
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
  return [nx / len, ny / len, nz / len];
}

/** Full 4×4 determinant of a column-major matrix — = THREE's
 *  `Matrix4.determinant()` (used to detect the winding flip). */
function determinant4(m: ArrayLike<number>): number {
  const n11 = m[0] ?? 0, n12 = m[4] ?? 0, n13 = m[8] ?? 0, n14 = m[12] ?? 0;
  const n21 = m[1] ?? 0, n22 = m[5] ?? 0, n23 = m[9] ?? 0, n24 = m[13] ?? 0;
  const n31 = m[2] ?? 0, n32 = m[6] ?? 0, n33 = m[10] ?? 0, n34 = m[14] ?? 0;
  const n41 = m[3] ?? 0, n42 = m[7] ?? 0, n43 = m[11] ?? 0, n44 = m[15] ?? 0;
  return (
    n41 * (
      +n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 +
      n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34
    ) +
    n42 * (
      +n11 * n23 * n34 - n11 * n24 * n33 + n14 * n21 * n33 -
      n13 * n21 * n34 + n13 * n24 * n31 - n14 * n23 * n31
    ) +
    n43 * (
      +n11 * n24 * n32 - n11 * n22 * n34 - n14 * n21 * n32 +
      n12 * n21 * n34 + n14 * n22 * n31 - n12 * n24 * n31
    ) +
    n44 * (
      -n13 * n22 * n31 - n11 * n23 * n32 + n11 * n22 * n33 +
      n13 * n21 * n32 - n12 * n21 * n33 + n12 * n23 * n31
    )
  );
}

// ──────────────────────────────────────────────────────────────────────────
// Material dedup — mirrors snapshotPreBuildMaterials' value-dedup signature
// ──────────────────────────────────────────────────────────────────────────

/**
 * Structural material signature — the core `MaterialSpec` counterpart to
 * `legacy/bvhCommon.ts:snapshotPreBuildMaterials`'s `matSig`. Hashes only the fields
 * the GI/PT consumers read (baseColor, emissive, emissiveIntensity, roughness,
 * metallic, transmission, ior, Beer-Lambert attenuation fields, and base/normal
 * map handle identity), with the SAME `toFixed(4)` quantisation, so two primitives
 * carrying structurally-equal materials collapse to one LUT slot — exactly as the
 * THREE value-dedup does for React/R3F material churn. Map identity uses the opaque
 * `TextureRef.handle` (the core analogue of THREE's `texture.uuid`).
 *
 * Beer-Lambert fields (`attenuationColor`, `attenuationDistance`, `thickness`) are
 * included to match `materialSetHashFloats` in `sceneBvh.ts`. Infinity
 * `attenuationDistance` is normalised to the token `'Inf'` so the signature remains
 * a stable string (JSON.stringify of Infinity produces `null`).
 */
export function materialSig(m: MaterialSpec): string {
  const col = m.baseColor;
  const colS = col
    ? `${(col[0] ?? 0).toFixed(4)},${(col[1] ?? 0).toFixed(4)},${(col[2] ?? 0).toFixed(4)}`
    : '';
  const em = m.emissive;
  const emS = em
    ? `${(em[0] ?? 0).toFixed(4)},${(em[1] ?? 0).toFixed(4)},${(em[2] ?? 0).toFixed(4)}`
    : '';
  const r = (m.roughness ?? 0.5).toFixed(4);
  const mt = (m.metallic ?? 0).toFixed(4);
  const ei = (m.emissiveIntensity ?? 1).toFixed(4);
  const tr = (m.transmission ?? 0).toFixed(4);
  const ior = (m.ior ?? 1.5).toFixed(4);
  const mapU = handleId(m.baseColorMap?.handle);
  const nmU = handleId(m.normalMap?.handle);
  // Beer-Lambert fields — must match materialSetHashFloats in sceneBvh.ts.
  const ac = m.attenuationColor;
  const acS = ac
    ? `${(ac[0] ?? 1).toFixed(4)},${(ac[1] ?? 1).toFixed(4)},${(ac[2] ?? 1).toFixed(4)}`
    : '1.0000,1.0000,1.0000';
  const adRaw = m.attenuationDistance;
  const adS = adRaw == null
    ? 'Inf'
    : !isFinite(adRaw)
      ? 'Inf'
      : adRaw.toFixed(4);
  const thS = (m.thickness ?? 0).toFixed(4);
  return `${colS}|${emS}|${ei}|${r}|${mt}|${tr}|${ior}|${mapU}|${nmU}|${acS}|${adS}|${thS}`;
}

/**
 * Stable per-object identity registry for the material dedup signature.
 *
 * The module-level WeakMap is LOAD-BEARING: handle identity must persist
 * ACROSS merge calls so the same object (e.g. the same decoded ImageBitmap)
 * always maps to the same signature token. Wrapping it in an exported object
 * enables test-time reset without exposing the raw module globals.
 *
 * `reset()` is intentionally not called in production — it would invalidate
 * cached signatures and break dedup continuity. Call it only in test teardown
 * to prevent cross-test object-identity bleed.
 */
export const HandleIdRegistry = {
  _ids: new WeakMap<object, string>(),
  _seq: 0,
  /** Return the stable id for `handle`. Assigns a new one on first encounter. */
  get(handle: object): string {
    let id = this._ids.get(handle);
    if (id === undefined) {
      id = `h${this._seq++}`;
      this._ids.set(handle, id);
    }
    return id;
  },
  /**
   * Reset the registry — for TEST USE ONLY.
   * Clears all assigned ids and resets the sequence counter.
   * Do NOT call in production: existing handle ids become stale.
   */
  reset(): void {
    this._ids = new WeakMap();
    this._seq = 0;
  },
};

/** A stable per-handle identity string for the dedup signature. Objects use
 *  {@link HandleIdRegistry}; primitives stringify directly; absent handles
 *  contribute the empty string. */
function handleId(handle: unknown): string {
  if (handle == null) return '';
  if (typeof handle === 'object' || typeof handle === 'function') {
    return HandleIdRegistry.get(handle as object);
  }
  // eslint-disable-next-line @typescript-eslint/no-base-to-string -- at this point handle is a primitive (guarded: not object/function), String() is safe
  return String(handle);
}

// ──────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────

/**
 * Build a single merged WORLD-space tri stream + BVH from a `@vitrum/core`
 * `Scene` — the THREE-free analogue of `buildSceneBVH` for the merged-BVH
 * consumers (DDGI / RC / ReSTIR emitter list). See the module docstring for the
 * exact parity contract with `buildSceneBVH` (stream float-parity; BVH-topology
 * divergence by R1).
 *
 * @param scene a `@vitrum/core` `Scene`.
 * @param opts  see {@link WorldSpaceMergeOptions}.
 */
export function mergeWorldSpaceFromCore(
  scene: Scene,
  opts: WorldSpaceMergeOptions = {},
): WorldSpaceMergeResult {
  const stride = opts.positionStride ?? 4;
  const filter = opts.filter ?? DEFAULT_MERGE_FILTER;

  // Accumulators for the merged world-space stream (merge order — NOT yet
  // BVH-reordered). `mergedIndices` is stride-3 (3 u32 / triangle).
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = []; // per-vertex texture coords (stride 2, merge order, untransformed)
  const mergedIndices: number[] = [];
  const mergedTriMaterialId: number[] = [];
  const meshVertexRanges: MergedMeshVertexRange[] = [];

  // Material value-dedup LUT (mirrors snapshotPreBuildMaterials).
  const materials: MaterialSpec[] = [];
  const sigToSlot = new Map<string, number>();

  const resolveMaterialSlot = (mat: MaterialSpec): number => {
    const sig = materialSig(mat);
    const existing = sigToSlot.get(sig);
    if (existing !== undefined) return existing;
    const slot = materials.length;
    sigToSlot.set(sig, slot);
    materials.push(mat);
    return slot;
  };

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (const primitive of scene.primitives) {
    if (!filter(primitive) || !isMeshLike(primitive)) continue;

    const matSlot = resolveMaterialSlot(primitive.material);

    // For an instanced-mesh, world-merge every instance (one baked copy per
    // transform — the faithful world-space expansion). mesh / skinned-mesh have
    // exactly one transform (`transform` or identity). This matches what a
    // world-baked merge of the equivalent THREE scene produces: N baked
    // InstancedMesh copies (the ReSTIR emitter path deliberately EXCLUDES
    // instanced meshes from its world merge — callers that need that exclusion
    // pass a `filter` that rejects `kind === 'instanced-mesh'`).
    const transforms: ReadonlyArray<Mat4 | undefined> =
      primitive.kind === 'instanced-mesh'
        ? primitive.instances
        : [primitive.transform];

    const basePositions = primitive.positions;
    const baseNormals = primitive.normals;
    const baseUvs = primitive.uvs; // optional; (0,0) per vertex when absent
    const localVertexCount = Math.floor(basePositions.length / 3);
    if (localVertexCount < 3) continue;

    // Sequential index when the primitive carries none (triangle-list), matching
    // `packOneMeshLikePrimitive` / SGG's index synthesis.
    const baseIndices: ArrayLike<number> =
      primitive.indices ??
      (() => {
        const gen = new Uint32Array(localVertexCount);
        for (let i = 0; i < localVertexCount; i += 1) gen[i] = i;
        return gen;
      })();
    const localTriCount = Math.floor(baseIndices.length / 3);
    if (localTriCount === 0) continue;

    for (const transform of transforms) {
      const m = transform ?? IDENTITY_MAT4;
      const normalMatrix = getNormalMatrix3(m);
      const flip = determinant4(m) < 0;

      const vertexStart = Math.floor(positions.length / stride);
      const triStart = Math.floor(mergedTriMaterialId.length);

      // Transform + append this instance's vertices (world space).
      for (let i = 0; i < localVertexCount; i += 1) {
        const lx = basePositions[i * 3] ?? 0;
        const ly = basePositions[i * 3 + 1] ?? 0;
        const lz = basePositions[i * 3 + 2] ?? 0;
        const [wx, wy, wz] = applyMatrix4(m, lx, ly, lz);
        positions.push(wx, wy, wz);
        if (stride === 4) positions.push(0);

        const lnx = baseNormals[i * 3] ?? 0;
        const lny = baseNormals[i * 3 + 1] ?? 1;
        const lnz = baseNormals[i * 3 + 2] ?? 0;
        const [nx, ny, nz] = applyNormalMatrix(normalMatrix, lnx, lny, lnz);
        normals.push(nx, ny, nz);
        if (stride === 4) normals.push(0);

        // UVs are 2D texture coords — transform-invariant (no world-matrix applied).
        uvs.push(baseUvs?.[i * 2] ?? 0, baseUvs?.[i * 2 + 1] ?? 0);

        if (wx < minX) minX = wx; if (wx > maxX) maxX = wx;
        if (wy < minY) minY = wy; if (wy > maxY) maxY = wy;
        if (wz < minZ) minZ = wz; if (wz > maxZ) maxZ = wz;
      }

      // Append this instance's triangles (global vertex refs = local + start),
      // applying the negative-determinant winding flip (v0↔v2) when needed —
      // mirroring StaticGeometryGenerator's `invertGeometry`.
      for (let t = 0; t < localTriCount; t += 1) {
        const a = (baseIndices[t * 3] ?? 0) + vertexStart;
        const b = (baseIndices[t * 3 + 1] ?? 0) + vertexStart;
        const c = (baseIndices[t * 3 + 2] ?? 0) + vertexStart;
        if (flip) {
          mergedIndices.push(c, b, a);
        } else {
          mergedIndices.push(a, b, c);
        }
        mergedTriMaterialId.push(matSlot);
      }

      meshVertexRanges.push({
        name: primitive.id,
        vertexStart,
        vertexCount: localVertexCount,
        triStart,
        triCount: localTriCount,
      });
    }
  }

  const vertexCount = Math.floor(positions.length / stride);
  const triangleCount = mergedTriMaterialId.length;

  // Empty scene → an empty-but-valid result (one zeroed leaf node), so DDGI/RC
  // don't have to special-case a primitive-less frame (mirrors
  // `emptyBVHResult`'s intent, THREE-free).
  if (triangleCount === 0) {
    return {
      bvhNodes: new Float32Array(8),
      positions: new Float32Array(stride === 4 ? 12 : 9),
      positionStrideFloats: stride,
      indices: new Uint32Array([0, 1, 2]),
      bvhIndexStride: 3,
      triMaterialId: new Uint32Array(1),
      normals: new Float32Array(stride === 4 ? 12 : 9),
      uvs: new Float32Array(6),
      mergedIndices: new Uint32Array([0, 1, 2]),
      mergedTriMaterialId: new Uint32Array(1),
      materials,
      boundingBox: { min: [-1, -1, -1], max: [1, 1, 1] },
      meshVertexRanges,
      vertexCount: 3,
      triangleCount: 0,
    };
  }

  const packedPositions = new Float32Array(positions);
  const packedNormals = new Float32Array(normals);
  const packedUvs = new Float32Array(uvs);
  const packedMergedIndices = new Uint32Array(mergedIndices);
  const packedMergedTriMaterialId = new Uint32Array(mergedTriMaterialId);

  // Build the merged single-root BVH over the world-space stream. stride-3 index
  // in / out (the `array<vec3u>` form, matching `buildSceneBVH`'s stride-3
  // `indices`). The reorder permutes triangles but NOT vertices.
  const bvh = buildArrayBvh(packedPositions, packedMergedIndices, packedMergedTriMaterialId, {
    positionStride: stride,
    indexStride: 3,
  });

  return {
    bvhNodes: bvh.bvhNodes,
    positions: packedPositions,
    positionStrideFloats: stride,
    indices: bvh.reorderedIndices,
    bvhIndexStride: 3,
    triMaterialId: bvh.reorderedTriMaterialIds,
    normals: packedNormals,
    uvs: packedUvs,
    mergedIndices: packedMergedIndices,
    mergedTriMaterialId: packedMergedTriMaterialId,
    materials,
    boundingBox: { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
    meshVertexRanges,
    vertexCount,
    triangleCount,
  };
}
