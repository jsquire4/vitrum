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
 *   - DDGI:   wraps `SceneBvh` class around `buildSceneBVH({positionStride: 4})`
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

function assertMeshBvhRootsLayout(bvh: MeshBVH): void {
  const roots = (bvh as BvhWithRoots)._roots;
  if (!roots || roots.length === 0 || roots[0] == null) {
    throw new Error(
      '[@vitrum/shared-bvh] MeshBVH._roots is missing or empty — three-mesh-bvh internal layout likely changed. Pin three-mesh-bvh to the tested ^0.9.x range.',
    );
  }
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
   *   u32[6]     rightChildOrTriOffset (RELATIVE offset to right child for
   *              interior nodes; absolute triangle offset for leaf nodes —
   *              identical leaf semantics in three-mesh-bvh 0.7.x and 0.9.x)
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
   * Triangle indices — 3 u32 per triangle (`BvhIndexStride = 3`, the
   * `array<vec3u>` WGSL form used by RC and DDGI shaders).  Read AFTER
   * the BVH build — MeshBVH's SAH partition reorders the index buffer in
   * place.
   *
   * Callers that require stride 4 (pt-webgpu, ReSTIR) must post-process:
   *   - pt-webgpu: expand to vec4u, zeroing `.w` (zero-fill contract).
   *   - ReSTIR:    expand and pack RGBA material color + texType into `.w`.
   *
   * Upload-time assertion (recommended):
   * ```ts
   * const stride = bvhIndexStride;  // 3 or 4
   * if (indexData.byteLength % (stride * 4) !== 0)
   *   throw new Error(`BVH index buffer not aligned to stride ${stride}`);
   * ```
   */
  indices: Uint32Array;

  /**
   * Index-buffer stride — always 3 for `buildSceneBVH` output (3 u32 per
   * triangle, no padding).  Callers that post-process to stride 4 should
   * document that separately.
   */
  bvhIndexStride: 3;

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

  /** Merged-geometry UV attribute when one was present on the source meshes,
   *  otherwise `undefined`. Exposed here so callers (e.g. ReSTIR's bvhCompute)
   *  don't reach into `bvh.geometry.attributes['uv']` directly — that private
   *  field can change shape across three-mesh-bvh releases. */
  uvAttribute?: THREE.BufferAttribute;

  /** World-space bounding box of the merged geometry. */
  boundingBox: THREE.Box3;

  /**
   * Per-source-mesh vertex ranges in the merged buffer, in the same order
   * as the meshes the filter accepted (matching `traverseVisible` order).
   * Each entry's `name` is the source `Object3D.name` — for synthesized
   * scenes (`vitrumSceneToThree`) the name equals the primitive id, so a
   * primitive id round-trips back to its vertex range without a fresh
   * traversal.
   *
   * `StaticGeometryGenerator` concatenates each source mesh's vertices
   * contiguously, and the BVH build reorders the *index* buffer but NOT
   * the vertex buffer — so these ranges remain valid for the lifetime of
   * the returned buffers. Used by `HybridEngine.updatePrimitive`'s
   * transform-only refit fast path.
   */
  meshVertexRanges: ReadonlyArray<{
    name: string;
    vertexStart: number;
    vertexCount: number;
    triStart: number;
    triCount: number;
  }>;
}

export interface SceneBVHCommonOpts {
  /**
   * Caller-supplied filter — returns true for Object3D's whose geometry
   * should contribute to the BVH. Defaults to MeshStandard + MeshPhysical
   * meshes (the convention 2-of-3 walkaround engines use today).
   */
  filter?: (obj: THREE.Object3D) => boolean;

  /**
   * Position-buffer element stride in floats:
   *   3 → 12 bytes/vertex (raster / TSL path; default).
   *   4 → 16 bytes/vertex (WGSL vec3f-aligned; .w zero-filled).
   * Per WGSL spec, `array<vec3<f32>>` has alignment 16 and size 12 →
   * stride = roundUp(16, 12) = 16. Reading a tightly-packed 12-byte
   * positions array as `array<vec3f>` from WGSL garbles every vertex
   * past index 0 (the visible "scrambled geometry" symptom).
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
//
// B6 KNOWN LIMITATION: MeshStandardMaterial is a sentinel for "any
// opaque surface" in the original scene (stained-glass studio). Edge
// materials (lead came beads, solder beads) are created as
// MeshStandardMaterial, so they pass this filter and enter the BVH,
// bloating leaf count and risking shadow-ray self-intersection at
// panel edges. The fix is to stamp `userData.excludeFromBVH = true`
// in the edge material factory and reject it here, but that requires
// a host-app callsite change. Documented as a known rider; do not fix
// inside this library without coordinating the host callsite change.
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
 *  BVH-invariant). Also returns the mesh-traversal-ordered materials list.
 *
 *  **Multi-material:** if `mesh.material` is an array, only index `0` is used
 *  for the BVH material snapshot (matches historical walkaround convention).
 */
function snapshotPreBuildMaterials(
  meshes: THREE.Mesh[],
  merged: THREE.BufferGeometry,
): { vertexMatId: Uint32Array; materials: THREE.Material[] } {
  const vertexCount = (merged.attributes['position'] as THREE.BufferAttribute).count;
  const vertexMatId = new Uint32Array(vertexCount);

  // One entry per *value-unique* material across all source meshes.
  // Multi-material meshes (e.g. GlassMesh's [front, front, back] array)
  // collapse to their primary (index-0) material.
  //
  // Dedup-by-value (not by identity) because hosts that use React/R3F
  // construct fresh THREE.Material instances on re-render even when the
  // PBR field values are identical. Without dedup, redundant instances
  // overflow DDGI's 64-material slot cap and force per-frame BVH rebuilds.
  // Value-dedup collapses this back to the structural minimum. — fix 2026-05-12.
  //
  // The signature hashes only the fields the consumers (DDGI / ReSTIR
  // shading / PT path) actually read: baseColor, emissive, emissiveIntensity,
  // roughness, metalness, transmission, ior, and texture-map identity.
  // Two materials with identical signatures are interchangeable for
  // path-traced and probe-updated illumination.
  const matSig = (m: THREE.Material): string => {
    const s = m as THREE.MeshStandardMaterial;
    const p = m as THREE.MeshPhysicalMaterial;
    const col = s.color ? `${s.color.r.toFixed(4)},${s.color.g.toFixed(4)},${s.color.b.toFixed(4)}` : '';
    const em = s.emissive ? `${s.emissive.r.toFixed(4)},${s.emissive.g.toFixed(4)},${s.emissive.b.toFixed(4)}` : '';
    const r = (s.roughness ?? 0.5).toFixed(4);
    const mt = (s.metalness ?? 0).toFixed(4);
    const ei = (s.emissiveIntensity ?? 1).toFixed(4);
    const tr = (p.transmission ?? 0).toFixed(4);
    const ior = (p.ior ?? 1.5).toFixed(4);
    // Map identity (uuid is stable across React renders for the same source).
    const mapU = s.map ? s.map.uuid : '';
    const nmU = s.normalMap ? s.normalMap.uuid : '';
    return `${col}|${em}|${ei}|${r}|${mt}|${tr}|${ior}|${mapU}|${nmU}`;
  };
  const matLut: THREE.Material[] = [];
  const matMap = new Map<THREE.Material, number>();
  const sigMap = new Map<string, number>();
  for (const mesh of meshes) {
    const meshMat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
    if (!meshMat || matMap.has(meshMat)) continue;
    const sig = matSig(meshMat);
    const existing = sigMap.get(sig);
    if (existing !== undefined) {
      // Different THREE.Material instance, same PBR signature — alias the
      // identity to the existing canonical slot.
      matMap.set(meshMat, existing);
    } else {
      const idx = matLut.length;
      sigMap.set(sig, idx);
      matMap.set(meshMat, idx);
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
  // Sky domes, Environment probes, and other backdrop objects have very
  // large bounding spheres (radius > 1000 world units). They are
  // visual-only and must NOT enter the BVH — their huge extents break
  // the BVH bounds, and ray-BVH traversal almost never intersects sky
  // geometry from inside the scene anyway. Temporarily set them
  // invisible so traverseVisible skips them.
  const tempHidden: THREE.Object3D[] = [];
  if (Number.isFinite(skyHideRadius)) {
    // Hoist scratch vectors outside the traversal — was allocating
    // three THREE.Vector3 + one Quaternion per visible mesh per build.
    const _decompPos = new THREE.Vector3();
    const _decompQuat = new THREE.Quaternion();
    const _decompScale = new THREE.Vector3();
    for (const root of roots) {
      root.traverseVisible((obj) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh) return;
        const geo = mesh.geometry;
        if (!geo) return;
        if (!geo.boundingSphere) geo.computeBoundingSphere();
        const radius = geo.boundingSphere?.radius ?? 0;
        mesh.matrixWorld.decompose(_decompPos, _decompQuat, _decompScale);
        const worldRadius = radius * Math.max(_decompScale.x, _decompScale.y, _decompScale.z);
        if (worldRadius > skyHideRadius) {
          obj.visible = false;
          tempHidden.push(obj);
        }
      });
    }
  }

  // ── 0b. BVH proxy substitution for densely-tessellated flat surfaces ─
  // Visual mesh ≠ ray-trace mesh. Per-engine opt-in via `proxyMeshNames`;
  // defaults to empty set (no substitution).
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
  // callers that want them downstream (the ReSTIR engine packs UV into
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

  // ── 3b. Per-source-mesh vertex ranges ─────────────────────────────────
  // Snapshot now (pre-clearGroups, pre-BVH-build). The BVH build reorders
  // INDICES via SAH but never moves vertices, so these ranges remain
  // valid for the lifetime of the returned buffers. Consumers (e.g.
  // HybridEngine.updatePrimitive's transform-only refit) use them to
  // refresh one mesh's worldspace positions in place without touching
  // the rest of the merged vertex buffer.
  const meshVertexRanges: Array<{
    name: string;
    vertexStart: number;
    vertexCount: number;
    triStart: number;
    triCount: number;
  }> = [];
  {
    const origGroups2 = merged.groups;
    const idxArr2 = merged.index!.array;
    if (origGroups2.length === meshes.length && origGroups2.length > 0) {
      // One group per source mesh — scan each group's index slice for
      // min/max vertex referenced. Vertex ranges are contiguous per
      // mesh thanks to StaticGeometryGenerator's concatenation order.
      for (let gi = 0; gi < origGroups2.length; gi++) {
        const g = origGroups2[gi]!;
        const mesh = meshes[gi]!;
        let mn = Number.POSITIVE_INFINITY;
        let mx = Number.NEGATIVE_INFINITY;
        const end = g.start + g.count;
        for (let i = g.start; i < end; i++) {
          const v = idxArr2[i]!;
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
        const vertexStart = Number.isFinite(mn) ? mn : 0;
        const vertexCount = Number.isFinite(mx) ? mx - vertexStart + 1 : 0;
        const triStart = Math.floor(g.start / 3);
        const triCount = Math.floor(g.count / 3);
        meshVertexRanges.push({ name: mesh.name, vertexStart, vertexCount, triStart, triCount });
      }
    } else if (meshes.length === 1) {
      // Single-material merge (no groups) — one mesh covers all vertices.
      const total = (merged.attributes['position'] as THREE.BufferAttribute).count;
      const triCount = Math.floor(merged.index!.count / 3);
      meshVertexRanges.push({
        name: meshes[0]!.name,
        vertexStart: 0,
        vertexCount: total,
        triStart: 0,
        triCount,
      });
    } else {
      // Defensive fallback: group metadata absent / mismatched. Emit
      // empty ranges so `meshVertexRanges.length === meshes.length`
      // — the caller's id→range lookup will return vertexCount=0 and
      // skip the refit, preferring the topology-rebuild path.
      for (const mesh of meshes) {
        meshVertexRanges.push({
          name: mesh.name,
          vertexStart: 0,
          vertexCount: 0,
          triStart: 0,
          triCount: 0,
        });
      }
    }
  }

  // ── 4. Collapse groups + build BVH ────────────────────────────────────
  // Replace per-mesh groups with a single group covering all triangles
  // so MeshBVH builds ONE unified root.
  merged.clearGroups();
  merged.addGroup(0, merged.index!.count, 0);
  const bvh = new MeshBVH(merged, { strategy: 0 /* SAH */ }) as BvhWithRoots;
  assertMeshBvhRootsLayout(bvh);

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
  normalizeBvhInteriorOffsets(bvhNodes);

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

  const uvAttribute = merged.attributes['uv'] as THREE.BufferAttribute | undefined;

  return {
    bvh,
    bvhNodes,
    positions,
    positionStrideFloats: positionStride,
    indices,
    // buildSceneBVH always returns stride-3 indices (3 u32 per triangle,
    // no padding). Callers needing stride 4 must post-process — see the
    // BvhIndexStride type in @vitrum/shared-bvh for the contract.
    bvhIndexStride: 3,
    triMaterialId,
    normals,
    materials,
    ...(uvAttribute != null ? { uvAttribute } : {}),
    boundingBox,
    meshVertexRanges,
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
  assertMeshBvhRootsLayout(bvh);
  return {
    bvh,
    bvhNodes: packRootBuffer(bvh._roots[0]!),
    positions: new Float32Array(positionStride === 4 ? 12 : 9),
    positionStrideFloats: positionStride,
    indices: new Uint32Array([0, 1, 2]),
    bvhIndexStride: 3,
    triMaterialId: new Uint32Array(1),
    normals: new Float32Array(positionStride === 4 ? 12 : 9),
    materials: [],
    boundingBox: new THREE.Box3(
      new THREE.Vector3(-1, -1, -1),
      new THREE.Vector3(1, 1, 1),
    ),
    meshVertexRanges: [],
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

/**
 * Verify a relative-offset BVH produced by `buildCpuBvh` (pt-webgpu) or
 * `normalizeBvhInteriorOffsets` (shared-bvh).
 *
 * For every interior node the right-child offset must satisfy:
 *   1 ≤ rightChildOrTriOffset < totalNodes
 * (offset 0 would alias the node itself; offset ≥ totalNodes would be out
 * of bounds.) Leaf nodes are skipped — their `rightChildOrTriOffset` field
 * holds a triangle array offset, not a node index.
 *
 * Cost: O(totalNodes) — call only in dev / test mode (the function is always
 * exported so callers control the gate). Throws on the first violation found.
 */
export function validateBvhEncoding(
  nodeBytes: Float32Array | Uint32Array,
  totalNodes: number,
): void {
  const UINT32_PER_NODE = 8;
  const LEAFNODE_FLAG = 0xffff;
  const u32 =
    nodeBytes instanceof Uint32Array
      ? nodeBytes
      : new Uint32Array(nodeBytes.buffer, nodeBytes.byteOffset, nodeBytes.length);

  for (let i = 0; i < totalNodes; i++) {
    const base = i * UINT32_PER_NODE;
    const splitOrCount = u32[base + 7]!;
    const isLeaf = (splitOrCount >>> 16) === LEAFNODE_FLAG;
    if (isLeaf) continue;
    const offset = u32[base + 6]!;
    if (offset < 1 || offset >= totalNodes) {
      throw new Error(
        `[@vitrum/shared-bvh] validateBvhEncoding: interior node ${i} has invalid ` +
          `relative right-child offset ${offset} (must be in [1, ${totalNodes - 1}]). ` +
          `Check that the BVH was built with relative-offset encoding.`,
      );
    }
  }
}

/**
 * Normalise the interior-node `rightChildOrTriOffset` field to the
 * canonical "relative node index" encoding that all downstream WGSL
 * shaders assume (`right_child = nodeIdx + offset`).
 *
 * three-mesh-bvh changed the packed-buffer layout between 0.7.x and
 * 0.9.x:
 *  - 0.7.x stores the **absolute u32 index** of the right child
 *    (`uint32Array[stride4 + 6] = nextUnusedPointer / 4`). Values are
 *    multiples of 8 (= UINT32_PER_NODE) and grow with tree size.
 *  - 0.9.x stores the **relative node index**
 *    (`uint32Array[node32 + 6] = rightNodeIdx - currentNodeIdx`). Values
 *    are always strictly less than `totalNodes`.
 *
 * The host application (and `three-gpu-pathtracer`) pin 0.7.x; Vite's
 * `dedupe` collapses shared-bvh's `^0.9.9` declaration down to that
 * 0.7.x at bundle time. Rather than dictating which version the host
 * picks, this helper detects the source layout from the data itself
 * and rewrites in-place so the GPU layout is invariant.
 *
 * Detection rule: under 0.9.x the stored relative offset MUST be less
 * than `totalNodes` (definitionally — it is a node-index delta inside
 * the same buffer). Under 0.7.x the stored absolute u32 index is the
 * right child's node index × UINT32_PER_NODE, so it ranges up to
 * `(totalNodes - 1) × 8`. So: `firstInteriorOffset >= totalNodes`
 * unambiguously identifies the 0.7.x layout.
 *
 * Leaf nodes are untouched — both versions store the triangle offset
 * in the same field with identical semantics.
 */
function normalizeBvhInteriorOffsets(bvhNodes: Float32Array): void {
  const UINT32_PER_NODE = 8;
  const LEAFNODE_FLAG = 0xFFFF;
  const u32 = new Uint32Array(bvhNodes.buffer, bvhNodes.byteOffset, bvhNodes.length);
  const totalNodes = bvhNodes.length / UINT32_PER_NODE;
  if (totalNodes <= 1) return;

  let needsConversion = false;
  for (let i = 0; i < totalNodes; i++) {
    const base = i * UINT32_PER_NODE;
    const splitOrCount = u32[base + 7]!;
    const isLeaf = (splitOrCount >>> 16) === LEAFNODE_FLAG;
    if (isLeaf) continue;
    const value = u32[base + 6]!;
    if (value >= totalNodes) { needsConversion = true; }
    break;
  }
  if (!needsConversion) return;

  for (let i = 0; i < totalNodes; i++) {
    const base = i * UINT32_PER_NODE;
    const splitOrCount = u32[base + 7]!;
    const isLeaf = (splitOrCount >>> 16) === LEAFNODE_FLAG;
    if (isLeaf) continue;
    const absoluteU32Idx = u32[base + 6]!;
    const absoluteNodeIdx = absoluteU32Idx / UINT32_PER_NODE;
    u32[base + 6] = absoluteNodeIdx - i;
  }
}

/**
 * In-place BVH **refit** — recompute every node's AABB bounds without
 * rebuilding tree topology. Used by `HybridEngine.updatePrimitive` when a
 * transform-only patch arrives: tree structure (split planes, parent/
 * child links, triangle ordering, leaf membership) is preserved; only the
 * `bounds.min/max` fields are refreshed from the (already updated) vertex
 * positions.
 *
 * Cost: O(totalNodes + sum(leafTriCounts)). For a 30 k-tri scene this
 * runs in well under 1 ms vs. ~50 ms for a full SAH rebuild — the whole
 * point of the fast path.
 *
 * **Algorithm**: iterative post-order DFS. Each node is "first-visited"
 * (push children + emit-marker) and then "emit-visited" (compute AABB
 * from children's already-refit bounds or from leaf triangles).
 *
 * **Inputs**:
 *  - `bvhNodes`: the 32-byte-per-node packed Float32Array buffer. Its
 *    `bounds` fields (slots 0–5) are overwritten in place; the topology
 *    fields (slots 6, 7) are untouched.
 *  - `indices`: stride-3 triangle index buffer (3 u32 per triangle) as
 *    returned by `buildSceneBVH`. **Stride-4 callers** (ReSTIR / pt-webgpu)
 *    must pass a stride-3 view since refit reads `indices[t*3 + (0|1|2)]`.
 *  - `positions`: vertex-position buffer. Stride may be 3 or 4 floats per
 *    vertex; the .w lane (if present) is ignored.
 *  - `positionStrideFloats`: 3 or 4 — matches the build-time layout.
 *
 * **Invariant preserved**: leaves keep their `triOffset` + `triCount`;
 * interior nodes keep their relative right-child offset. The GPU
 * traversal continues to work unchanged.
 */
export function refitBvhBounds(
  bvhNodes: Float32Array,
  indices: Uint32Array,
  positions: Float32Array,
  positionStrideFloats: 3 | 4,
): void {
  const UINT32_PER_NODE = 8;
  const LEAFNODE_FLAG = 0xffff;
  const totalNodes = bvhNodes.length / UINT32_PER_NODE;
  if (totalNodes === 0) return;
  const u32 = new Uint32Array(bvhNodes.buffer, bvhNodes.byteOffset, bvhNodes.length);
  const f32 = bvhNodes;

  // Build a post-order traversal order via an iterative DFS so we don't
  // risk a stack overflow on deep trees. Each stack entry packs
  // (nodeIdx, second-visit-flag) into one 32-bit slot:
  //   bit 31 set ⇒ second visit (emit AABB)
  //   low 31 bits ⇒ node index
  const order = new Int32Array(totalNodes);
  let orderLen = 0;
  const stack = new Int32Array(totalNodes * 2);
  let sp = 0;
  stack[sp++] = 0; // root, first visit
  while (sp > 0) {
    const entry = stack[--sp]!;
    const isSecondVisit = (entry & 0x80000000) !== 0;
    const nodeIdx = entry & 0x7fffffff;
    if (isSecondVisit) {
      order[orderLen++] = nodeIdx;
      continue;
    }
    const splitOrCount = u32[nodeIdx * UINT32_PER_NODE + 7]!;
    const isLeaf = (splitOrCount >>> 16) === LEAFNODE_FLAG;
    if (isLeaf) {
      order[orderLen++] = nodeIdx;
      continue;
    }
    // Internal node: schedule emit-now after both children, then push
    // children (right first so left pops first — stable visitation order
    // for debugging).
    stack[sp++] = nodeIdx | 0x80000000;
    const leftChild = nodeIdx + 1;
    const rightChild = nodeIdx + u32[nodeIdx * UINT32_PER_NODE + 6]!;
    stack[sp++] = rightChild;
    stack[sp++] = leftChild;
  }

  // Walk the post-order list and refit each node's AABB.
  for (let oi = 0; oi < orderLen; oi++) {
    const nodeIdx = order[oi]!;
    const base = nodeIdx * UINT32_PER_NODE;
    const splitOrCount = u32[base + 7]!;
    const isLeaf = (splitOrCount >>> 16) === LEAFNODE_FLAG;

    let mnX = Infinity, mnY = Infinity, mnZ = Infinity;
    let mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;

    if (isLeaf) {
      const triCount = splitOrCount & 0xffff;
      const triOffset = u32[base + 6]!;
      for (let t = 0; t < triCount; t++) {
        const triIdx = triOffset + t;
        const i0 = indices[triIdx * 3 + 0]!;
        const i1 = indices[triIdx * 3 + 1]!;
        const i2 = indices[triIdx * 3 + 2]!;
        // Three vertices per triangle — inline the unrolled cmin/cmax
        // rather than allocate an array per iteration.
        let off = i0 * positionStrideFloats;
        let x = positions[off + 0]!, y = positions[off + 1]!, z = positions[off + 2]!;
        if (x < mnX) mnX = x; if (x > mxX) mxX = x;
        if (y < mnY) mnY = y; if (y > mxY) mxY = y;
        if (z < mnZ) mnZ = z; if (z > mxZ) mxZ = z;
        off = i1 * positionStrideFloats;
        x = positions[off + 0]!; y = positions[off + 1]!; z = positions[off + 2]!;
        if (x < mnX) mnX = x; if (x > mxX) mxX = x;
        if (y < mnY) mnY = y; if (y > mxY) mxY = y;
        if (z < mnZ) mnZ = z; if (z > mxZ) mxZ = z;
        off = i2 * positionStrideFloats;
        x = positions[off + 0]!; y = positions[off + 1]!; z = positions[off + 2]!;
        if (x < mnX) mnX = x; if (x > mxX) mxX = x;
        if (y < mnY) mnY = y; if (y > mxY) mxY = y;
        if (z < mnZ) mnZ = z; if (z > mxZ) mxZ = z;
      }
    } else {
      const leftChild = nodeIdx + 1;
      const rightChild = nodeIdx + u32[base + 6]!;
      // Inline both children's union to avoid an iterator allocation.
      let cBase = leftChild * UINT32_PER_NODE;
      let cMnX = f32[cBase + 0]!, cMnY = f32[cBase + 1]!, cMnZ = f32[cBase + 2]!;
      let cMxX = f32[cBase + 3]!, cMxY = f32[cBase + 4]!, cMxZ = f32[cBase + 5]!;
      if (cMnX < mnX) mnX = cMnX; if (cMxX > mxX) mxX = cMxX;
      if (cMnY < mnY) mnY = cMnY; if (cMxY > mxY) mxY = cMxY;
      if (cMnZ < mnZ) mnZ = cMnZ; if (cMxZ > mxZ) mxZ = cMxZ;
      cBase = rightChild * UINT32_PER_NODE;
      cMnX = f32[cBase + 0]!; cMnY = f32[cBase + 1]!; cMnZ = f32[cBase + 2]!;
      cMxX = f32[cBase + 3]!; cMxY = f32[cBase + 4]!; cMxZ = f32[cBase + 5]!;
      if (cMnX < mnX) mnX = cMnX; if (cMxX > mxX) mxX = cMxX;
      if (cMnY < mnY) mnY = cMnY; if (cMxY > mxY) mxY = cMxY;
      if (cMnZ < mnZ) mnZ = cMnZ; if (cMxZ > mxZ) mxZ = cMxZ;
    }

    f32[base + 0] = mnX; f32[base + 1] = mnY; f32[base + 2] = mnZ;
    f32[base + 3] = mxX; f32[base + 4] = mxY; f32[base + 5] = mxZ;
  }
}
