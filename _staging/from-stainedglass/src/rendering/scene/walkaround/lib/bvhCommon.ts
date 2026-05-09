/**
 * Tier 2 shared GI primitive — single-root BVH builder.
 *
 * Canonical core consolidating the three GI engines' (DDGI / RC / ReSTIR)
 * structurally-identical BVH build logic into one module.
 *
 * What this module guarantees:
 *  • ensureIndexed normalisation across mixed indexed/non-indexed source
 *    geometries.
 *  • Pre-build per-vertex matId snapshot (vertices are NOT reordered by
 *    MeshBVH; consulting them post-build is BVH-invariant).
 *  • Group-collapse single-root MeshBVH build (else the WGSL traversal
 *    walks only the first source mesh's triangles).
 *  • Packed positions / indices / normals / triMaterialId extraction.
 *  • Optional 12-byte (stride-3, raster/TSL) vs 16-byte (stride-4, WGSL
 *    vec3f-aligned) layout — caller picks via `positionStride`.
 *  • Optional proxy-mesh substitution for densely-tessellated flat
 *    surfaces — opt-in via `proxyMeshNames`.
 *
 * What this module deliberately does NOT do (technique-specific; stays
 * per-engine):
 *  • Emitter list construction (ReSTIR only).
 *  • UV-pack-into-position-w (ReSTIR's compute-cast trick to fit inside
 *    the 8-storage-buffer-per-stage device limit).
 *  • bvhIndex.w RGBA8 + texType packing (ReSTIR-only).
 *  • MaterialEntry flat-struct packing (RC's cascade compute only).
 *
 * Per-engine consumers wrap this module's output with their own packing:
 *   - DDGI:   wraps `SceneBvh` class around `buildSceneBVH({positionStride: 3})`
 *             for dirty-tracking semantics.
 *   - RC:     calls `buildSceneBVH({positionStride: 4})` then packs
 *             `materials` array → `MaterialEntry` flat-struct sibling fn.
 *   - ReSTIR: calls `buildSceneBVH({positionStride: 4, proxyMeshNames: ...})`
 *             then packs UV into position[*].w + builds emitter list +
 *             packs color into bvhIndex[*].w sibling fns.
 */

import * as THREE from 'three';
import { MeshBVH, StaticGeometryGenerator } from 'three-mesh-bvh';

// MeshBVH exposes `_roots` at runtime but it is not in the published
// `.d.ts`. We need the field to read the packed BVHNode buffer.
//
// Pin: tested against three-mesh-bvh ^0.9.x (current package.json
// constraint). The `_roots` field name is not part of the published
// public API — re-verify on every minor version bump. If the field is
// ever renamed by upstream, the buildSceneBVH below will throw with a
// clear runtime error rather than silently misread the BVH bytes.
interface BvhWithRoots extends MeshBVH {
  _roots: ArrayBuffer[];
}

// ──────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────

export interface SceneBVHCommonResult {
  /** The constructed BVH instance (caller may keep for debug / re-use). */
  bvh: MeshBVH;

  /**
   * Raw BVH-node buffer — 32 bytes/node, single root, layout matches
   * three-mesh-bvh's internal 8 × u32 pack:
   *   f32[0..2]  bounds.min xyz
   *   f32[3..5]  bounds.max xyz
   *   u32[6]     rightChildOrTriOffset (RELATIVE to current node index for
   *              internal nodes; absolute for leaves on three-mesh-bvh
   *              0.9.x — see §3.3 / Q8)
   *   u32[7]     splitAxisOrTriCount  (LEAFNODE_MASK_32 | count for leaves)
   *
   * Returned as Float32Array for direct DMA to a WebGPU storage buffer.
   */
  bvhNodes: Float32Array;

  /**
   * Vertex positions, world-space. Stride matches `positionStrideFloats`:
   *   3 → 12-byte raw layout (TSL / raster path);
   *   4 → 16-byte vec3f-aligned layout for WGSL `array<vec3f>` reads;
   *        .w slot is zero-filled (callers that need to pack UV/color in
   *        .w do so as a post-process — see ReSTIR's bvhCompute.ts).
   */
  positions: Float32Array;
  positionStrideFloats: 3 | 4;

  /**
   * Triangle indices (3 × u32 per triangle). Read AFTER the BVH build —
   * MeshBVH's SAH partition reorders the index buffer in place.
   */
  indices: Uint32Array;

  /**
   * Per-triangle materialId (one u32 per triangle). Derived via the
   * BVH-invariant per-vertex matId snapshot taken pre-build, so it
   * survives the SAH reorder. Indexes into `materials`.
   */
  triMaterialId: Uint32Array;

  /**
   * Vertex normals — same stride as positions. Empty zero-filled buffer
   * if the source geometry had no normal attribute (defensive; merged
   * geometry always has them when source meshes do).
   */
  normals: Float32Array;

  /**
   * Source materials in mesh-traversal order. Length = number of unique
   * materials encountered across the filtered scene roots, in first-seen
   * order. Caller packs per-pipeline (RC's MaterialEntry flat-struct,
   * ReSTIR's per-tri color packing, etc.).
   */
  materials: THREE.Material[];

  /** World-space bounding box of the merged geometry. */
  boundingBox: THREE.Box3;
}

export interface SceneBVHCommonOpts {
  /**
   * Caller-supplied filter — returns true for Object3D's whose geometry
   * should contribute to the BVH. Defaults to MeshStandard + MeshPhysical
   * meshes (the convention 2-of-3 walkaround branches use today).
   */
  filter?: (obj: THREE.Object3D) => boolean;

  /**
   * Position-buffer element stride in floats:
   *   3 → 12 bytes/vertex (raster / TSL path; default).
   *   4 → 16 bytes/vertex (WGSL vec3f-aligned; .w zero-filled).
   * Per WGSL spec, `array<vec3<f32>>` has alignment 16 and size 12 →
   * stride = roundUp(16, 12) = 16. Reading a tightly-packed 12-byte
   * positions array as `array<vec3f>` from WGSL garbles every vertex
   * past index 0 (the visible "scrambled geometry" symptom — see §3.2).
   */
  positionStride?: 3 | 4;

  /**
   * Object names whose geometry should be substituted with a 1×1-segment
   * bounding-box plane during BVH build, then restored after. Use for
   * densely-tessellated visual-only flat surfaces (32×32 floor /
   * ceiling) that would otherwise consume thousands of BVH leaves and
   * trip Windows TDR on per-pixel ray budgets. The visual mesh is
   * untouched — only the merged BVH walks the proxy.
   *
   * Names match `Object3D.name`. The substituted geometry must be
   * roughly planar (XY-extent in local space) — the proxy is a
   * `THREE.PlaneGeometry(w, h, 1, 1)` matched to the local-space
   * bounding-box dimensions; the original `matrixWorld` carries
   * orientation + position. ReSTIR uses `{floor, ceiling}`; DDGI and
   * RC don't need this today (their per-frame BVH-walk budgets are
   * smaller) but the option is technique-agnostic.
   */
  proxyMeshNames?: Set<string>;

  /**
   * Hide environment / backdrop meshes (drei <Sky>, <Environment>, etc.)
   * with bounding-sphere world-radius > this threshold during BVH build.
   * Default 500 world units. Set to `Infinity` to keep them in.
   */
  skyHideRadiusThreshold?: number;
}

// ──────────────────────────────────────────────────────────────────────────
// Default filter: MeshStandard + MeshPhysical visible meshes.
// ──────────────────────────────────────────────────────────────────────────

const DEFAULT_FILTER = (obj: THREE.Object3D): boolean => {
  if (!(obj instanceof THREE.Mesh)) return false;
  const mat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
  if (!mat) return false;
  return (
    mat instanceof THREE.MeshStandardMaterial ||
    mat instanceof THREE.MeshPhysicalMaterial
  );
};

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

/**
 * Normalise a BufferGeometry so it is always indexed. Three-mesh-bvh's
 * StaticGeometryGenerator requires all geometries to be uniformly indexed
 * or uniformly non-indexed. Mixing them throws at merge time.
 *
 * Converting non-indexed to indexed is safe and lossless — we add a
 * sequential 0,1,2,… index over the existing vertices.
 */
function ensureIndexed(geo: THREE.BufferGeometry): void {
  if (geo.index !== null) return;
  const posCount =
    (geo.attributes['position'] as THREE.BufferAttribute | undefined)?.count ?? 0;
  if (posCount === 0) return;
  const idx = new Uint32Array(posCount);
  for (let i = 0; i < posCount; i++) idx[i] = i;
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
}

/** Build the per-vertex → matId map BEFORE the BVH build (vertices are
 *  BVH-invariant). Also returns the mesh-traversal-ordered materials list. */
function snapshotPreBuildMaterials(
  meshes: THREE.Mesh[],
  merged: THREE.BufferGeometry,
): { vertexMatId: Uint32Array; materials: THREE.Material[] } {
  const vertexCount = (merged.attributes['position'] as THREE.BufferAttribute).count;
  const vertexMatId = new Uint32Array(vertexCount);

  // One entry per unique THREE.Material across all source meshes, in
  // mesh-traversal order. Multi-material meshes (e.g. GlassMesh's
  // [front, front, back] array) collapse to their primary (index-0)
  // material — this matches the conventions 2-of-3 walkaround branches
  // use today.
  const matLut: THREE.Material[] = [];
  const matMap = new Map<THREE.Material, number>();
  for (const mesh of meshes) {
    const meshMat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
    if (meshMat && !matMap.has(meshMat)) {
      matMap.set(meshMat, matLut.length);
      matLut.push(meshMat);
    }
  }

  const origGroups = merged.groups;
  const origIdxArr = merged.index!.array;

  if (origGroups.length > 0) {
    // StaticGeometryGenerator emits one group per source mesh in input
    // order, so `origGroups[gi]` corresponds to `meshes[gi]`. Stamp every
    // vertex referenced by each group's pre-build index range with the
    // primary material's index in `matLut`.
    for (let gi = 0; gi < origGroups.length; gi++) {
      const group = origGroups[gi]!;
      const mesh = meshes[gi] ?? meshes[meshes.length - 1]!;
      const meshMat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      const matIdx = meshMat ? (matMap.get(meshMat) ?? 0) : 0;
      const idxStart = group.start;
      const idxEnd = idxStart + group.count;
      for (let i = idxStart; i < idxEnd; i++) {
        vertexMatId[origIdxArr[i]!] = matIdx;
      }
    }
  }
  // origGroups empty → vertexMatId zero-init is correct (single material).

  return { vertexMatId, materials: matLut };
}

// ──────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────

/**
 * Build a single-root BVH over the filtered geometry under `sceneOrRoots`.
 * Cost is O(n log n) on triangle count; for the canonical post-roommesh
 * honeycomb scene (~30k triangles) the build runs in ~50 ms. Caller
 * should debounce rebuilds — see `useSceneBVH` for the canonical pattern.
 *
 * @throws {Error} when MeshBVH's StaticGeometryGenerator fails to merge
 *         the filtered geometry — typically degenerate triangles
 *         (zero-area), mismatched attribute strides across source meshes,
 *         or interleaved buffer layouts that can't be normalized to
 *         indexed form. Callers should treat this as a scene-graph
 *         problem (bad mesh data), not a transient runtime issue —
 *         retrying will not succeed without changing the input.
 *
 * @throws {Error} (transitively) if the underlying three-mesh-bvh
 *         private `_roots` field has been renamed by an upstream
 *         version bump (see BvhWithRoots note above the implementation).
 */
export function buildSceneBVH(
  sceneOrRoots: THREE.Scene | THREE.Object3D[],
  opts: SceneBVHCommonOpts = {},
): SceneBVHCommonResult {
  const filter = opts.filter ?? DEFAULT_FILTER;
  const positionStride = opts.positionStride ?? 3;
  const proxyMeshNames = opts.proxyMeshNames ?? new Set<string>();
  const skyHideRadius = opts.skyHideRadiusThreshold ?? 500;

  const roots: THREE.Object3D[] = Array.isArray(sceneOrRoots)
    ? sceneOrRoots
    : [sceneOrRoots];

  // ── 0. Force-update world matrices ─────────────────────────────────────
  // StaticGeometryGenerator applies mesh.matrixWorld to every vertex
  // position. If this runs in a useEffect async callback (outside the
  // RAF render loop), Three has not yet called scene.updateMatrixWorld()
  // for newly mounted meshes — the merged geometry would land in
  // undefined (often origin) world positions. Force the update here
  // so the merge is correct.
  for (const root of roots) {
    root.updateMatrixWorld(true);
  }

  // ── 0a. Hide environment / backdrop meshes from BVH traversal ──────────
  // Sky domes (drei <Sky>), Environment probes, and other backdrop
  // objects have very large bounding spheres (radius > 1000 world units).
  // They are visual-only and must NOT enter the BVH — their huge extents
  // break the BVH bounds, and ray-BVH traversal almost never intersects
  // sky geometry from inside the room anyway. Temporarily set them
  // invisible so traverseVisible skips them.
  const tempHidden: THREE.Object3D[] = [];
  if (Number.isFinite(skyHideRadius)) {
    for (const root of roots) {
      root.traverseVisible((obj) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh) return;
        const geo = mesh.geometry;
        if (!geo) return;
        if (!geo.boundingSphere) geo.computeBoundingSphere();
        const radius = geo.boundingSphere?.radius ?? 0;
        const worldScale = new THREE.Vector3();
        mesh.matrixWorld.decompose(new THREE.Vector3(), new THREE.Quaternion(), worldScale);
        const worldRadius = radius * Math.max(worldScale.x, worldScale.y, worldScale.z);
        if (worldRadius > skyHideRadius) {
          obj.visible = false;
          tempHidden.push(obj);
        }
      });
    }
  }

  // ── 0b. BVH proxy substitution for densely-tessellated flat surfaces ─
  // Visual mesh ≠ ray-trace mesh — see §3.4 inventory entry. Per-branch
  // opt-in via `proxyMeshNames`; defaults to empty set (no substitution).
  const proxySwaps: {
    mesh: THREE.Mesh;
    original: THREE.BufferGeometry;
    proxy: THREE.BufferGeometry;
  }[] = [];
  if (proxyMeshNames.size > 0) {
    for (const root of roots) {
      root.traverseVisible((obj) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh) return;
        if (!proxyMeshNames.has(obj.name)) return;
        const original = mesh.geometry;
        if (!original) return;
        if (!original.boundingBox) original.computeBoundingBox();
        const bb = original.boundingBox;
        if (!bb) return;
        const w = bb.max.x - bb.min.x;
        const h = bb.max.y - bb.min.y;
        if (w <= 0 || h <= 0) return; // not a recognisable plane; skip
        const proxy = new THREE.PlaneGeometry(w, h, 1, 1);
        mesh.geometry = proxy;
        proxySwaps.push({ mesh, original, proxy });
      });
    }
  }

  // ── 1. Walk roots, collect filtered meshes, normalise to indexed ───────
  const meshes: THREE.Mesh[] = [];
  for (const root of roots) {
    root.traverseVisible((obj) => {
      if (filter(obj)) meshes.push(obj as THREE.Mesh);
    });
  }

  if (meshes.length === 0) {
    // Restore swaps + visibility before returning empty.
    for (const swap of proxySwaps) {
      swap.mesh.geometry = swap.original;
      swap.proxy.dispose();
    }
    for (const obj of tempHidden) obj.visible = true;
    return emptyBVHResult(positionStride);
  }

  for (const mesh of meshes) {
    ensureIndexed(mesh.geometry);
  }

  // ── 2. Generate merged geometry ────────────────────────────────────────
  const sgg = new StaticGeometryGenerator(meshes);
  // Always include normals; UVs are optional but cheap to carry for
  // callers that want them downstream (the ReSTIR branch packs UV into
  // position.w as a sibling step).
  sgg.attributes = ['position', 'normal', 'uv'];
  // applyWorldTransforms is true by default in three-mesh-bvh's
  // StaticGeometryGenerator; setting it explicitly is defensive.
  (sgg as unknown as { applyWorldTransforms?: boolean }).applyWorldTransforms = true;

  const merged = sgg.generate();

  // Restore visibility / proxy swaps now that the merge has captured the
  // proxy geometry into the merged buffer.
  for (const swap of proxySwaps) {
    swap.mesh.geometry = swap.original;
    swap.proxy.dispose();
  }
  for (const obj of tempHidden) obj.visible = true;

  if (!merged.index) {
    // Defensive: every source mesh was indexed by the normalisation
    // pass above, but if StaticGeometryGenerator ever returns a
    // non-indexed merge, build a sequential index so the BVH build +
    // downstream reads of merged.index don't crash.
    merged.computeVertexNormals();
    const posAttr = merged.attributes['position'] as THREE.BufferAttribute;
    const idx = new Uint32Array(posAttr.count);
    for (let i = 0; i < posAttr.count; i++) idx[i] = i;
    merged.setIndex(new THREE.BufferAttribute(idx, 1));
  }

  // ── 3. Pre-build per-vertex matId snapshot ────────────────────────────
  // MUST run before `new MeshBVH(merged)` because the BVH build reorders
  // the index buffer in place via SAH. Vertices are NOT reordered — see
  // top-of-file comment for the full rationale.
  const { vertexMatId, materials } = snapshotPreBuildMaterials(meshes, merged);

  // ── 4. Collapse groups + build BVH ────────────────────────────────────
  // Replace per-mesh groups with a single group covering all triangles
  // so MeshBVH builds ONE unified root. We deliberately do NOT restore
  // the original groups afterwards — they would be stale (their
  // start/count ranges no longer match the reordered index buffer).
  // The vertexMatId map is the sole source of truth for per-triangle
  // material lookups from here on.
  merged.clearGroups();
  merged.addGroup(0, merged.index!.count, 0);
  const bvh = new MeshBVH(merged, { strategy: 0 /* SAH */ }) as BvhWithRoots;

  // ── 5. Pack BVH node array from _roots[0] ─────────────────────────────
  // After the group collapse, _roots is guaranteed length 1. Copy the
  // raw bytes into a fresh Float32Array so the caller can DMA directly
  // (and the original BVH object's internal buffer may be mutated by
  // subsequent edits).
  //
  // three-mesh-bvh's _roots stores ArrayBuffer (newer) or a typed-array
  // view (older). Normalise to a fresh ArrayBuffer-backed Float32Array
  // by routing through the typed-array constructor — it accepts either
  // shape and copies in both branches.
  const bvhNodes = packRootBuffer(bvh._roots[0]!);

  // ── 6. Extract positions / normals / indices ──────────────────────────
  const posAttr = merged.attributes['position'] as THREE.BufferAttribute;
  const normAttr = merged.attributes['normal'] as THREE.BufferAttribute | undefined;
  const indexAttr = merged.index!;

  const vertexCount = posAttr.count;
  const indexCount = indexAttr.count;
  const triCount = indexCount / 3;

  // Indices: read AFTER the BVH build (SAH reorders in place).
  const srcIdx = indexAttr.array;
  const indices = new Uint32Array(srcIdx.length);
  for (let i = 0; i < srcIdx.length; i++) indices[i] = srcIdx[i]!;

  // Positions: stride 3 (raw) or 4 (vec3f-aligned, .w zero-filled).
  let positions: Float32Array;
  if (positionStride === 4) {
    positions = new Float32Array(vertexCount * 4);
    const srcPos = posAttr.array;
    for (let i = 0; i < vertexCount; i++) {
      positions[i * 4 + 0] = srcPos[i * 3 + 0]!;
      positions[i * 4 + 1] = srcPos[i * 3 + 1]!;
      positions[i * 4 + 2] = srcPos[i * 3 + 2]!;
      // .w left at 0 — caller packs UV / color / etc. as a post-process.
    }
  } else {
    // Stride 3: copy raw to detach from the merged geometry's lifetime.
    positions =
      posAttr.array instanceof Float32Array
        ? new Float32Array(posAttr.array)
        : Float32Array.from(posAttr.array);
  }

  // Normals: same stride as positions for layout symmetry.
  let normals: Float32Array;
  if (normAttr) {
    if (positionStride === 4) {
      normals = new Float32Array(vertexCount * 4);
      const srcNorm = normAttr.array;
      for (let i = 0; i < vertexCount; i++) {
        normals[i * 4 + 0] = srcNorm[i * 3 + 0]!;
        normals[i * 4 + 1] = srcNorm[i * 3 + 1]!;
        normals[i * 4 + 2] = srcNorm[i * 3 + 2]!;
      }
    } else {
      normals =
        normAttr.array instanceof Float32Array
          ? new Float32Array(normAttr.array)
          : Float32Array.from(normAttr.array);
    }
  } else {
    normals = new Float32Array(vertexCount * positionStride);
  }

  // ── 7. Per-triangle materialId via BVH-invariant vertex snapshot ──────
  // Any vertex of triangle t works (StaticGeometryGenerator never splits
  // a triangle across source meshes), so reading vertex 0 is sufficient.
  // Stale `merged.groups` ranges are NOT consulted.
  const triMaterialId = new Uint32Array(triCount);
  for (let t = 0; t < triCount; t++) {
    triMaterialId[t] = vertexMatId[indices[t * 3]!]!;
  }

  // ── 8. Compute bounds ─────────────────────────────────────────────────
  merged.computeBoundingBox();
  const boundingBox = merged.boundingBox!.clone();

  return {
    bvh,
    bvhNodes,
    positions,
    positionStrideFloats: positionStride,
    indices,
    triMaterialId,
    normals,
    materials,
    boundingBox,
  };
}

/**
 * Empty-but-valid result so callers don't have to special-case
 * "no meshes matched the filter yet" (e.g. early-mount paint).
 */
function emptyBVHResult(positionStride: 3 | 4): SceneBVHCommonResult {
  const emptyGeo = new THREE.BufferGeometry();
  emptyGeo.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(9), 3),
  );
  emptyGeo.setIndex(new THREE.BufferAttribute(new Uint32Array([0, 1, 2]), 1));
  emptyGeo.addGroup(0, 3, 0);
  const bvh = new MeshBVH(emptyGeo) as BvhWithRoots;
  return {
    bvh,
    bvhNodes: packRootBuffer(bvh._roots[0]!),
    positions: new Float32Array(positionStride === 4 ? 12 : 9),
    positionStrideFloats: positionStride,
    indices: new Uint32Array([0, 1, 2]),
    triMaterialId: new Uint32Array(1),
    normals: new Float32Array(positionStride === 4 ? 12 : 9),
    materials: [],
    boundingBox: new THREE.Box3(
      new THREE.Vector3(-1, -1, -1),
      new THREE.Vector3(1, 1, 1),
    ),
  };
}

/**
 * Copy three-mesh-bvh's `_roots[0]` (which may be an ArrayBuffer or a
 * typed-array view depending on library version) into a fresh
 * Float32Array. Handles both the ArrayBuffer-shape and the view-shape
 * by routing through `Float32Array.from` for the latter.
 */
function packRootBuffer(root0: ArrayBuffer | ArrayBufferView): Float32Array {
  if (root0 instanceof ArrayBuffer) {
    // Newer three-mesh-bvh: `_roots[i]` is a raw ArrayBuffer. Slice() to
    // detach from the BVH's internal lifetime, then create a view.
    return new Float32Array(root0.slice(0));
  }
  // Older three-mesh-bvh: `_roots[i]` is a Float32Array view. Copy via
  // the typed-array constructor, which works regardless of the
  // underlying buffer kind (ArrayBuffer or SharedArrayBuffer).
  const view = root0 as Float32Array;
  return new Float32Array(view);
}
