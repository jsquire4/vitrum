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

/**
 * Scan a stride-4 (vec4f) BVH position buffer into a padded {@link SceneAabb}.
 *
 * Accepts the raw position data directly (`ArrayBuffer` or `Float32Array`) —
 * the shared package's API is backend-neutral. The legacy
 * `{ bvhPositions: { cpuData } }` shape (walkaround-hybrid's GPU-resource
 * object) is still accepted for the existing call sites.
 */
export function deriveSceneAABBFromBvhPositions(
  bvh: ArrayBuffer | Float32Array | { bvhPositions: { cpuData: ArrayBuffer } },
): SceneAabb {
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
  const padX = (maxX - minX) * 0.01 + 1e-3;
  const padY = (maxY - minY) * 0.01 + 1e-3;
  const padZ = (maxZ - minZ) * 0.01 + 1e-3;
  return {
    min: [minX - padX, minY - padY, minZ - padZ],
    max: [maxX + padX, maxY + padY, maxZ + padZ],
  };
}
