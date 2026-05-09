/**
 * Walkaround GI lighting node (§8).
 *
 * Builds a TSL node that samples the C0 cascade at a fragment's world
 * position and folds radiance against the surface BRDF. Injected into
 * MeshPhysicalNodeMaterial.lightingNode.
 *
 * The cascade is stored as a flat StorageBufferAttribute:
 *   index = (px + py*PX + pz*PX*PY) * raysPerProbe + rayBinIdx
 *
 * Sampling is 8-corner trilinear over probe grid × exact ray bin.
 * 16 oct-decode directions × cosine weight gives diffuse GI.
 */

import {
  Fn, vec3, float, uniform,
  positionWorld, normalWorld,
  dot, max, clamp,
  storage,
} from 'three/tsl';
import type { CascadeBuffers } from './cascadePyramid';
import { CASCADE_DIMS } from './cascadePyramid';

/** Octahedron decode: 2D unit-square uv → unit direction. */
function octDirForIndex(idx: number, gridSize: number): [number, number, number] {
  const gx = (idx % gridSize) + 0.5;
  const gy = (Math.floor(idx / gridSize)) + 0.5;
  const px = gx / gridSize * 2 - 1;
  const py = gy / gridSize * 2 - 1;
  let nx = px, ny = py;
  const nz = 1.0 - Math.abs(px) - Math.abs(py);
  if (nz < 0) {
    const tx = nx, ty = ny;
    nx = (1.0 - Math.abs(ty)) * (tx >= 0 ? 1 : -1);
    ny = (1.0 - Math.abs(tx)) * (ty >= 0 ? 1 : -1);
  }
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
  return [nx / len, ny / len, nz / len];
}

/** TSL scalar float uniform helper that supports .value mutation. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyNode = any;

export interface WalkaroundLightingNodes {
  lightingNode: AnyNode;
}

/**
 * Build the walkaround GI lighting contribution node.
 * Returns a TSL node that adds indirect GI to any receiver material.
 *
 * The cascade contains ray-traced radiance written by `cascadeDispatch.ts`
 * (probe ray-cast → bottom-up merge). When WebGPU is not the active backend
 * `cascadeDispatch` falls back to a constant debug fill so the receiver
 * materials still render.
 */
export function buildWalkaroundLightingNode(
  cascadeBuffers: CascadeBuffers,
): WalkaroundLightingNodes {
  const c0Dim = CASCADE_DIMS[0];
  const RAYS   = c0Dim.rays;                         // 16
  const GRID   = Math.round(Math.sqrt(RAYS));        // 4
  const [PX, PY, PZ] = c0Dim.probes;                 // see CASCADE_DIMS in cascadePyramid.ts

  // Precompute oct-decoded directions for all 16 bins as constants.
  const DIRS: [number, number, number][] = [];
  for (let i = 0; i < RAYS; i++) {
    DIRS.push(octDirForIndex(i, GRID));
  }

  // Uniform block for cascade geometry.
  const uOriginX = uniform(cascadeBuffers.probeOriginWorld.x, 'float');
  const uOriginY = uniform(cascadeBuffers.probeOriginWorld.y, 'float');
  const uOriginZ = uniform(cascadeBuffers.probeOriginWorld.z, 'float');
  const uSizeX   = uniform(cascadeBuffers.roomSize.x, 'float');
  const uSizeY   = uniform(cascadeBuffers.roomSize.y, 'float');
  const uSizeZ   = uniform(cascadeBuffers.roomSize.z, 'float');

  // C0 storage reference. Use cascadeBuffers.gpuCascades[0] — the same
  // StorageBufferAttribute instance that cascadeDispatch writes into.
  // Both paths share a single GPU buffer; the compute mutates the
  // Float32Array contents in place each frame, so the bound storage
  // node sees fresh radiance without needing a node-graph swap.
  // If the StorageBufferAttribute INSTANCE itself changes (e.g. room
  // bounds resize → useCascadeBuffers reallocates), the caller must
  // call buildWalkaroundLightingNode again to rebuild the node graph
  // — the storage reference is captured by the Fn closure at build
  // time and cannot be patched in place.
  const c0Attr  = cascadeBuffers.gpuCascades[0]!;
  const c0Storage = storage(c0Attr, 'vec4f', c0Attr.count).toReadOnly();

  // Build the diffuse GI term: sum over 16 C0 directions weighted by cosine.
  // Trilinear probe interpolation via 8-corner loop (unrolled outside for TSL).
  //
  // We use the TSL Fn() helper to construct a node function. The implementation
  // does trilinear sampling of the C0 buffer at the fragment's world position.
  const giDiffuseNode = Fn(() => {
    const diffuse = vec3(0, 0, 0).toVar('rcDiffuse');

    // Fragment probe UV coords (normalized probe grid position).
    const wPosX = positionWorld.x.sub(uOriginX).div(uSizeX);
    const wPosY = positionWorld.y.sub(uOriginY).div(uSizeY);
    const wPosZ = positionWorld.z.sub(uOriginZ).div(uSizeZ);

    // Probe grid float coords (center at 0.5 per cell).
    const gridFx = wPosX.mul(float(PX)).sub(0.5);
    const gridFy = wPosY.mul(float(PY)).sub(0.5);
    const gridFz = wPosZ.mul(float(PZ)).sub(0.5);
    const gridIx = gridFx.floor();
    const gridIy = gridFy.floor();
    const gridIz = gridFz.floor();
    const fx = gridFx.sub(gridFx.floor());
    const fy = gridFy.sub(gridFy.floor());
    const fz = gridFz.sub(gridFz.floor());

    // Sum over 16 direction bins.
    for (let d = 0; d < RAYS; d++) {
      const dir = DIRS[d]!;
      // Cosine-weighted contribution.
      const nDotL = max(float(0), dot(normalWorld, vec3(dir[0], dir[1], dir[2])));

      // Trilinear interpolation over 8 corners.
      const sample = vec3(0, 0, 0).toVar(`rcSample${d}`);

      for (let dz = 0; dz < 2; dz++) {
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            // Clamped probe index for corner.
            const cx = clamp(gridIx.add(float(dx)).toInt(), 0, PX - 1);
            const cy = clamp(gridIy.add(float(dy)).toInt(), 0, PY - 1);
            const cz = clamp(gridIz.add(float(dz)).toInt(), 0, PZ - 1);

            // probeIdx = cx + cy*PX + cz*PX*PY
            const probeIdx = cx
              .add(cy.mul(PX))
              .add(cz.mul(PX * PY));
            // outIdx = probeIdx * RAYS + d
            const outIdx = probeIdx.mul(RAYS).add(d);

            // Trilinear weight.
            const wx = dx === 0 ? float(1).sub(fx) : fx;
            const wy = dy === 0 ? float(1).sub(fy) : fy;
            const wz = dz === 0 ? float(1).sub(fz) : fz;
            const w  = wx.mul(wy).mul(wz);

            // Sample from C0 buffer.
            const rad = c0Storage.element(outIdx).xyz;
            sample.addAssign(rad.mul(w));
          }
        }
      }

      // Accumulate cosine-weighted radiance.
      diffuse.addAssign(sample.mul(nDotL));
    }

    // Monte-Carlo normalization for uniform full-sphere sampling with N directions:
    // PDF = 1/(4π) per sample, so estimator = (4π/N) × Σ L_i × max(0, cos θ_i).
    // The max(0, ...) clamp zero-weights the lower hemisphere; directions on the
    // full sphere still contribute the correct 4π solid angle factor.
    return diffuse.mul(4.0 * Math.PI / RAYS);
  })();

  return {
    lightingNode: giDiffuseNode,
  };
}

