// Pure patch-eligibility predicates and index resolvers for the pt-webgpu
// incremental-update fast paths. These were extracted verbatim from
// `PTEngineWebGPU` (theme T14 decomposition): every function here is a pure
// function over (scene, patch, supportedAnalyticShapes) with no GPU-device
// dependency, so the engine keeps the GPU-touching fast-path orchestration in
// `updatePrimitive`/`updateEmitter` and delegates the eligibility decisions
// here.
import type { ScenePrimitive } from '@vitrum/core';
import type { Scene } from '@vitrum/core';
import type { UploadedSceneBuffers } from './uploadSceneBuffers.js';

/**
 * Resolve the dense material-array index for a primitive id, matching the
 * material-packing order in `buildPackedScene` (analytic primitives only
 * contribute a material when their shape is in `supportedAnalyticShapes`;
 * mesh-like primitives always contribute). Returns `null` when the primitive
 * is absent or contributes no material slot.
 */
export function materialIndexForPrimitive(
  scene: Scene,
  primitiveId: string,
  supportedAnalyticShapes: ReadonlySet<string>,
): number | null {
  let materialIndex = 0;
  for (const primitive of scene.primitives) {
    let contributesMaterial = false;
    if (primitive.kind === 'analytic') {
      contributesMaterial = supportedAnalyticShapes.has(primitive.shape);
    } else {
      contributesMaterial = true;
    }
    if (primitive.id === primitiveId) {
      return contributesMaterial ? materialIndex : null;
    }
    if (contributesMaterial) materialIndex += 1;
  }
  return null;
}

/**
 * Resolve the dense analytic-array index for an analytic primitive id, matching
 * the analytic-packing order in `buildPackedScene`. Returns `null` when the
 * primitive is absent or is not a supported analytic shape.
 */
export function analyticIndexForPrimitive(
  scene: Scene,
  primitiveId: string,
  supportedAnalyticShapes: ReadonlySet<string>,
): number | null {
  let analyticIndex = 0;
  for (const primitive of scene.primitives) {
    if (primitive.kind !== 'analytic') continue;
    if (!supportedAnalyticShapes.has(primitive.shape)) continue;
    if (primitive.id === primitiveId) {
      return analyticIndex;
    }
    analyticIndex += 1;
  }
  return null;
}

/**
 * The material fields that contain TextureRef handles. A patch that changes any
 * of these fields must go through a full repack because the GPU texture arrays
 * (`materialTexDescriptorsBuffer`, sRGB/linear texture_2d_arrays) are not
 * incrementally writeable — only the packed float scalars are.
 *
 * Keep this list in sync with `MaterialMapFields` in `@vitrum/core`.
 */
const TEXTURE_MAP_FIELDS: ReadonlySet<string> = new Set([
  'baseColorMap',
  'normalMap',
  'roughnessMap',
  'metallicMap',
  'transmissionMap',
  'thicknessMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'specularColorMap',
  'specularIntensityMap',
  'bumpMap',
  'displacementMap',
  'lightMap',
]);

/**
 * Material-only patch: `material` present and no other facet keys touched.
 *
 * Item 2a — texture-map fields (TextureRef: baseColorMap, normalMap, etc.) are
 * NOT eligible for the material fast path because the fast path only rewrites
 * the packed float scalars in `materialsBuffer`. Texture-map changes require a
 * full repack so `materialTexDescriptorsBuffer` and the GPU texture_2d_arrays
 * are rebuilt. Any patch whose `material` contains a TextureRef field falls
 * through to `setScene`.
 */
export function canFastPathMaterialPatch(
  patch: Partial<ScenePrimitive>,
): patch is Partial<ScenePrimitive> & { material: ScenePrimitive['material'] } {
  if (patch.material == null) return false;
  for (const key of Object.keys(patch)) {
    if (key !== 'material' && key !== 'id' && key !== 'kind') return false;
  }
  // Reject if any TextureRef field is present in the material patch.
  const mat = patch.material as unknown as Record<string, unknown>;
  for (const field of Object.keys(mat)) {
    if (TEXTURE_MAP_FIELDS.has(field)) return false;
  }
  return true;
}

/**
 * Geometry (positions/normals) patch eligible for in-place BLAS refit: only
 * positions/normals touched, vertex counts unchanged, and the primitive is a
 * (skinned-)mesh.
 */
export function canFastPathGeometryPatch(
  primitive: ScenePrimitive,
  patch: Partial<ScenePrimitive>,
): boolean {
  if (primitive.kind !== 'mesh' && primitive.kind !== 'skinned-mesh') {
    return false;
  }
  const keys = Object.keys(patch).filter((k) => k !== 'id' && k !== 'kind');
  if (!keys.every((k) => k === 'positions' || k === 'normals')) {
    return false;
  }
  if (!('positions' in patch) && !('normals' in patch)) {
    return false;
  }
  if ('positions' in patch && patch.positions != null) {
    const cur = primitive.kind === 'mesh' || primitive.kind === 'skinned-mesh'
      ? primitive.positions
      : null;
    if (cur != null && patch.positions.length !== cur.length) {
      return false;
    }
  }
  if ('normals' in patch && patch.normals != null) {
    const cur = primitive.kind === 'mesh' || primitive.kind === 'skinned-mesh'
      ? primitive.normals
      : null;
    if (cur != null && patch.normals.length !== cur.length) {
      return false;
    }
  }
  return true;
}

/**
 * Topology-RESIZE patch on a (skinned-)mesh: positions/normals/indices are the
 * only facets touched AND at least one of them changes the vertex or triangle
 * count (slice-2 BLAS-resize splice). Distinct from
 * {@link canFastPathGeometryPatch}, which is the SAME-count in-place refit. The
 * splice in `rebuildPrimitiveBlas` grows/shrinks the concat buffers around the
 * changed primitive and rebases every downstream offset, so this kind no longer
 * forces a full `setScene`. Returns false for instanced-mesh / analytic (their
 * count changes are handled elsewhere or unsupported) and when nothing resized.
 */
export function canFastPathTopologyResizePatch(
  primitive: ScenePrimitive,
  patch: Partial<ScenePrimitive>,
): boolean {
  if (primitive.kind !== 'mesh' && primitive.kind !== 'skinned-mesh') {
    return false;
  }
  const keys = Object.keys(patch).filter((k) => k !== 'id' && k !== 'kind');
  if (keys.length === 0) return false;
  if (!keys.every((k) => k === 'positions' || k === 'normals' || k === 'indices')) {
    return false;
  }
  // The vertex count is derived from positions; the triangle count from indices
  // (or the implicit 0,1,2,… index when indices is absent → vertexCount/3 tris).
  const curVertexCount = Math.floor(primitive.positions.length / 3);
  const nextPositions =
    'positions' in patch && patch.positions != null
      ? (patch.positions)
      : primitive.positions;
  const nextVertexCount = Math.floor(nextPositions.length / 3);

  const curIndices = primitive.indices;
  const curTriCount = Math.floor((curIndices?.length ?? primitive.positions.length / 3) / 3);
  let nextTriCount = curTriCount;
  if ('indices' in patch) {
    const nextIndices = patch.indices as Uint32Array | Uint16Array | undefined;
    nextTriCount = Math.floor((nextIndices?.length ?? nextPositions.length / 3) / 3);
  } else if ('positions' in patch && curIndices == null) {
    // No explicit indices: triangle count tracks vertex count.
    nextTriCount = Math.floor(nextVertexCount / 3);
  }

  return nextVertexCount !== curVertexCount || nextTriCount !== curTriCount;
}

/**
 * Transform-only patch eligible for TLAS-only refit: `transform` (or
 * `instances` for instanced meshes) is the only facet touched.
 */
export function canFastPathTransformPatch(
  primitive: ScenePrimitive,
  patch: Partial<ScenePrimitive>,
): boolean {
  const keys = Object.keys(patch).filter((k) => k !== 'id' && k !== 'kind');
  if (primitive.kind === 'instanced-mesh') {
    if (!keys.every((k) => k === 'instances')) return false;
    return 'instances' in patch;
  }
  if (primitive.kind === 'mesh' || primitive.kind === 'skinned-mesh') {
    if (!keys.every((k) => k === 'transform')) return false;
    return 'transform' in patch;
  }
  if (primitive.kind === 'analytic') {
    if (!keys.every((k) => k === 'transform')) return false;
    return 'transform' in patch;
  }
  return false;
}

/**
 * Instances-only patch on an `instanced-mesh` whose instance COUNT changed
 * (slice-1 TLAS-only rebuild). Distinct from {@link canFastPathTransformPatch},
 * which handles the SAME-count case (in-place TLAS write). Returns false unless
 * the only mutated facet is `instances`, the primitive is an instanced-mesh, and
 * the patch's instance count differs from the current primitive's.
 */
export function canFastPathInstancedTopologyPatch(
  primitive: ScenePrimitive,
  patch: Partial<ScenePrimitive>,
): boolean {
  if (primitive.kind !== 'instanced-mesh') return false;
  const keys = Object.keys(patch).filter((k) => k !== 'id' && k !== 'kind');
  if (!keys.every((k) => k === 'instances')) return false;
  const nextInstances = (patch as { instances?: readonly unknown[] }).instances;
  if (nextInstances == null) return false;
  return nextInstances.length !== primitive.instances.length;
}

/**
 * True when a freshly refit TLAS has byte-for-byte identical buffer lengths to
 * the currently-uploaded buffers, so an in-place `writeBuffer` is safe without
 * reallocating GPU storage.
 */
export function canReuseTlasBufferLengths(
  sb: UploadedSceneBuffers,
  next: {
    readonly tlasNodes: Uint32Array;
    readonly tlasInstanceIndices: Uint32Array;
    readonly tlasBlasRoots: Uint32Array;
    readonly tlasInstanceWorldToLocal: Float32Array;
    readonly tlasInstanceLocalToWorld: Float32Array;
  },
): boolean {
  return (
    next.tlasNodes.byteLength === sb.tlasNodes.byteLength &&
    next.tlasInstanceIndices.byteLength === sb.tlasInstanceIndices.byteLength &&
    next.tlasBlasRoots.byteLength === sb.tlasBlasRoots.byteLength &&
    next.tlasInstanceWorldToLocal.byteLength === sb.tlasInstanceWorldToLocal.byteLength &&
    next.tlasInstanceLocalToWorld.byteLength === sb.tlasInstanceLocalToWorld.byteLength
  );
}
