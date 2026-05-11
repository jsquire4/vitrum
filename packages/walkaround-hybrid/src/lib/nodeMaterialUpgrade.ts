/**
 * NodeMaterial upgrade pattern — replicates the WebGPU renderer's internal
 * `NodeLibrary.fromMaterial()` pass so callers can attach `outputNode` /
 * `lightingNode` / `emissiveNode` (TSL injection points) to existing
 * MeshPhysical / MeshStandard materials.
 *
 * The `for...in` reflective property copy mirrors three.js's internal
 * NodeLibrary, which means transmission, ior, clearcoat, sheen, normalScale,
 * emissive, color, map, etc. all transfer without per-property hand-rolling.
 *
 * Tier 2 shared GI primitive. Consumed by DDGI (probe shading injection)
 * and RC (GI receiver wrapping). ReSTIR bypasses raster materials entirely
 * (compute-cast primary).
 *
 * Cross-subsystem usage note: this utility is consumed by applyDDGIShading.ts
 * (DDGI path, Step 4 of walkaround-hybrid extraction) and giReceiver.ts
 * (RC path, Step 4). Extracting it here makes it available to both once
 * those files are extracted.
 */

import * as THREE from 'three';
import {
  MeshPhysicalNodeMaterial,
  MeshStandardNodeMaterial,
} from 'three/webgpu';

/**
 * Upgrade a vanilla MeshPhysicalMaterial / MeshStandardMaterial to its
 * NodeMaterial sibling so callers can attach TSL injection points.
 *
 * This mirrors `NodeLibrary.fromMaterial()` in three/src/renderers/common/
 * nodes/NodeLibrary.js, which the WebGPU renderer uses internally during
 * its node-builder pass. The renderer creates an internal NodeMaterial each
 * frame, but does NOT mutate `mesh.material` — so a user-set `outputNode`
 * never reaches the GPU unless `mesh.material` is replaced explicitly.
 *
 * @param mat   - Source material to upgrade.
 * @returns       The upgraded NodeMaterial, or `null` when the input is not
 *                a MeshPhysical / MeshStandard variant. If the input is
 *                already a NodeMaterial it is returned unchanged.
 */
export function upgradeToNodeMaterial(
  mat: THREE.Material,
): MeshPhysicalNodeMaterial | MeshStandardNodeMaterial | null {
  // Already upgraded — return unchanged so callers can call this idempotently.
  if ((mat as { isNodeMaterial?: boolean }).isNodeMaterial === true) {
    return mat as unknown as MeshPhysicalNodeMaterial | MeshStandardNodeMaterial;
  }

  let nodeMat: MeshPhysicalNodeMaterial | MeshStandardNodeMaterial | null = null;
  if ((mat as THREE.MeshPhysicalMaterial).isMeshPhysicalMaterial === true) {
    nodeMat = new MeshPhysicalNodeMaterial();
  } else if ((mat as THREE.MeshStandardMaterial).isMeshStandardMaterial === true) {
    nodeMat = new MeshStandardNodeMaterial();
  }
  if (!nodeMat) return null;

  // Three's NodeLibrary.fromMaterial uses `for (const key in material)` —
  // walks own + inherited enumerable properties. Replicate exactly so
  // transmission, ior, clearcoat, sheen, normalScale, emissive, color,
  // map, etc. all transfer.
  const src = mat as unknown as Record<string, unknown>;
  const dst = nodeMat as unknown as Record<string, unknown>;
  for (const key in src) {
    dst[key] = src[key];
  }
  return nodeMat;
}
