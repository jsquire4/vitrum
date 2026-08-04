/**
 * sceneAabbFromBvh.ts — derive a padded world-space AABB by scanning the
 * uploaded BVH vertex-position buffer (vec4f / stride-4 layout: xyz + packed
 * payload in `.w`).
 *
 * The walkaround pipeline does not surface its scene bounds as a first-class
 * field, so three subsystems (the NRC hash-grid in `WalkaroundGPUPipeline`, the
 * PPG sTree in `PPGCoordinator`, and the ReGIR grid in `ReGIRCoordinator`) each
 * recover them by scanning the BVH position buffer the host always uploads.
 * This helper is the single canonical copy of that scan.
 *
 * Behaviour (kept byte-identical to the three former inline copies):
 *   - Empty / sub-vec4 buffer (`view.length < 4`) → ±10 fallback.
 *   - Non-finite extent (e.g. all-NaN positions) → ±10 fallback.
 *   - Otherwise: min/max over each vec4f's xyz, padded by `(max-min)*0.01 + 1e-3`
 *     per axis so points on the scene boundary still map inside the volume.
 */

/** Padded world-space AABB. Tuple-array form consumed by all three call sites
 *  (the PPG `AABB` type is structurally identical to this). */
export interface SceneAabb {
  min: [number, number, number];
  max: [number, number, number];
}

/** Stride of one TLAS node in f32 words (8 x 32-bit: bounds + 2 payload words). */
const TLAS_NODE_F32_STRIDE = 8;

/**
 * Optional TLAS-aware shape. When the walkaround BVH is in `tlas` mode the
 * position buffer holds per-BLAS LOCAL-space vertices (traversal transforms the
 * ray into instance space), so scanning it yields bounds in the wrong coordinate
 * space. TLAS node 0 already stores the world-space AABB of the whole scene.
 */
interface TlasAwareBvhLike {
  readonly bvhMode?: 'merged' | 'tlas';
  readonly bvhPositions: { cpuData: ArrayBuffer };
  readonly tlas?: { readonly nodes: { cpuData: ArrayBuffer }; readonly nodeCount: number } | undefined;
}

function padded(
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
): SceneAabb {
  const padX = (maxX - minX) * 0.01 + 1e-3;
  const padY = (maxY - minY) * 0.01 + 1e-3;
  const padZ = (maxZ - minZ) * 0.01 + 1e-3;
  return {
    min: [minX - padX, minY - padY, minZ - padZ],
    max: [maxX + padX, maxY + padY, maxZ + padZ],
  };
}

/**
 * Scan a stride-4 (vec4f) BVH position buffer into a padded {@link SceneAabb}.
 *
 * Accepts the raw position data directly (`ArrayBuffer` or `Float32Array`) —
 * the shared package's API is backend-neutral. The legacy
 * `{ bvhPositions: { cpuData } }` shape (walkaround-hybrid's GPU-resource
 * object) is still accepted for the existing call sites.
 *
 * When the caller passes the full walkaround BVH object AND it is in `tlas`
 * mode, the world-space bounds are taken from TLAS node 0 instead of the
 * position scan. `resolveReSTIRBvhMode` selects `tlas` for any scene with more
 * than one mesh-like primitive (or any instanced-mesh), so the position scan
 * alone reported LOCAL-space bounds for essentially every multi-mesh scene —
 * which the NRC hash grid, the PPG sTree, and the ReGIR grid then used as their
 * world-space spatial subdivision.
 */
export function deriveSceneAABBFromBvhPositions(
  bvh:
    | ArrayBuffer
    | Float32Array
    | { bvhPositions: { cpuData: ArrayBuffer } }
    | TlasAwareBvhLike,
): SceneAabb {
  if (
    !(bvh instanceof Float32Array) &&
    !(bvh instanceof ArrayBuffer) &&
    (bvh as TlasAwareBvhLike).bvhMode === 'tlas'
  ) {
    const tlas = (bvh as TlasAwareBvhLike).tlas;
    if (tlas != null && tlas.nodeCount > 0) {
      const nodes = new Float32Array(tlas.nodes.cpuData);
      if (nodes.length >= TLAS_NODE_F32_STRIDE) {
        const rootMinX = nodes[0]!, rootMinY = nodes[1]!, rootMinZ = nodes[2]!;
        const rootMaxX = nodes[3]!, rootMaxY = nodes[4]!, rootMaxZ = nodes[5]!;
        if (
          Number.isFinite(rootMinX) && Number.isFinite(rootMaxX) &&
          Number.isFinite(rootMinY) && Number.isFinite(rootMaxY) &&
          Number.isFinite(rootMinZ) && Number.isFinite(rootMaxZ) &&
          rootMaxX >= rootMinX && rootMaxY >= rootMinY && rootMaxZ >= rootMinZ
        ) {
          return padded(rootMinX, rootMinY, rootMinZ, rootMaxX, rootMaxY, rootMaxZ);
        }
      }
    }
  }
  const view =
    bvh instanceof Float32Array
      ? bvh
      : new Float32Array(bvh instanceof ArrayBuffer ? bvh : bvh.bvhPositions.cpuData);
  if (view.length < 4) {
    return { min: [-10, -10, -10], max: [10, 10, 10] };
  }
  // BVH position layout: vec4f per vertex (xyz + packed UV in w). Stride 4.
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i + 3 <= view.length; i += 4) {
    const x = view[i]!, y = view[i + 1]!, z = view[i + 2]!;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
    return { min: [-10, -10, -10], max: [10, 10, 10] };
  }
  return padded(minX, minY, minZ, maxX, maxY, maxZ);
}
