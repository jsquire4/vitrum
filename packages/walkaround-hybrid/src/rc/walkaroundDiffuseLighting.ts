/**
 * Walkaround GI lighting node.
 *
 * Builds a TSL node that samples the C0 cascade at a fragment's world
 * position and folds radiance against the surface BRDF. Injected into
 * MeshPhysicalNodeMaterial.lightingNode (via giReceiver.ts).
 *
 * The cascade is stored as a flat StorageBufferAttribute:
 *   index = (px + py*PX + pz*PX*PY) * raysPerProbe + rayBinIdx
 *
 * Sampling is 8-corner trilinear over probe grid × exact ray bin.
 * 16 oct-decode directions × cosine weight gives diffuse GI.
 *
 * Extracted from `_staging/legacy-source/src/rendering/scene/walkaround/walkaroundDiffuseLighting.ts`.
 * TSL imports preserved per extraction plan Option (i) — this is a Three.js NodeMaterial
 * customization hook built with TSL's Fn() node system.
 * Requires `three/tsl` as peer dep.
 */

import {
  Fn, vec3, float, uniform,
  positionWorld, normalWorld,
  dot, max, clamp,
  storage,
} from 'three/tsl';
import type { CascadeBuffers } from './cascadePyramid.js';
import { CASCADE_DIMS } from './cascadePyramid.js';
import { computeOctahedralSolidAngles } from './octahedralSolidAngles.js';

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
  const [PX, PY, PZ] = c0Dim.probes;

  // Precompute oct-decoded directions for all 16 bins as constants.
  const DIRS: [number, number, number][] = [];
  for (let i = 0; i < RAYS; i++) {
    DIRS.push(octDirForIndex(i, GRID));
  }

  // Per-bin solid-angle weights (Ω_i) for the N×N octahedral grid.
  // Replaces the uniform 4π/N assumption.  Sum of all Ω_i ≈ 4π.
  //
  // Reference: Cigolle et al. 2014, "A Survey of Efficient Representations
  // for Independent Unit Vectors", JCGT §2 / Appendix A.2.
  // The octahedral grid is NOT solid-angle-uniform — texels near the fold
  // edges subtend a smaller solid angle than central texels.
  const solidAngles = computeOctahedralSolidAngles(GRID);

  // Uniform block for cascade geometry.
  const uOriginX = uniform(cascadeBuffers.probeOriginWorld.x, 'float');
  const uOriginY = uniform(cascadeBuffers.probeOriginWorld.y, 'float');
  const uOriginZ = uniform(cascadeBuffers.probeOriginWorld.z, 'float');
  const uSizeX   = uniform(cascadeBuffers.roomSize.x, 'float');
  const uSizeY   = uniform(cascadeBuffers.roomSize.y, 'float');
  const uSizeZ   = uniform(cascadeBuffers.roomSize.z, 'float');

  // C0 storage reference. Use cascadeBuffers.gpuCascades[0] — the same
  // StorageBufferAttribute instance that cascadeDispatch writes into.
  // Both paths share a single GPU buffer; the compute mutates contents
  // in place each frame.
  // If the StorageBufferAttribute INSTANCE itself changes (e.g. room
  // bounds resize → CascadeBufferManager reallocates), the caller must
  // call buildWalkaroundLightingNode again to rebuild the node graph.
  const c0Attr  = cascadeBuffers.gpuCascades[0]!;
  // AnyNode cast: storage() from three/tsl has conservative typings that don't
  // accept the StorageBufferAttribute + string-type combo at strict TSC level.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const c0Storage = (storage as any)(c0Attr, 'vec4f', c0Attr.count).toReadOnly() as AnyNode;

  // Build the diffuse GI term: sum over 16 C0 directions weighted by cosine.
  // Trilinear probe interpolation via 8-corner loop (unrolled outside for TSL).
  const giDiffuseNode = Fn(() => {
    const diffuse = vec3(0, 0, 0).toVar('rcDiffuse');

    const wPosX = positionWorld.x.sub(uOriginX).div(uSizeX);
    const wPosY = positionWorld.y.sub(uOriginY).div(uSizeY);
    const wPosZ = positionWorld.z.sub(uOriginZ).div(uSizeZ);

    const gridFx = wPosX.mul(float(PX)).sub(0.5);
    const gridFy = wPosY.mul(float(PY)).sub(0.5);
    const gridFz = wPosZ.mul(float(PZ)).sub(0.5);
    const gridIx = gridFx.floor();
    const gridIy = gridFy.floor();
    const gridIz = gridFz.floor();
    const fx = gridFx.sub(gridFx.floor());
    const fy = gridFy.sub(gridFy.floor());
    const fz = gridFz.sub(gridFz.floor());

    for (let d = 0; d < RAYS; d++) {
      const dir = DIRS[d]!;
      const nDotL = max(float(0), dot(normalWorld, vec3(dir[0], dir[1], dir[2])));

      const sample = vec3(0, 0, 0).toVar(`rcSample${d}`);

      for (let dz = 0; dz < 2; dz++) {
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            // AnyNode casts: TSL clamp() / toInt() / mul() have strict typed overloads
            // that don't model mixed-int-float chains well at TSC level.
            const cx = clamp(gridIx.add(float(dx)).toInt() as AnyNode, 0 as AnyNode, (PX - 1) as AnyNode) as AnyNode;
            const cy = clamp(gridIy.add(float(dy)).toInt() as AnyNode, 0 as AnyNode, (PY - 1) as AnyNode) as AnyNode;
            const cz = clamp(gridIz.add(float(dz)).toInt() as AnyNode, 0 as AnyNode, (PZ - 1) as AnyNode) as AnyNode;

            const probeIdx = (cx as AnyNode)
              .add((cy as AnyNode).mul(PX))
              .add((cz as AnyNode).mul(PX * PY));
            const outIdx = (probeIdx as AnyNode).mul(RAYS).add(d);

            const wx = dx === 0 ? float(1).sub(fx) : fx;
            const wy = dy === 0 ? float(1).sub(fy) : fy;
            const wz = dz === 0 ? float(1).sub(fz) : fz;
            const w  = (wx as AnyNode).mul(wy).mul(wz);

            const rad = (c0Storage as AnyNode).element(outIdx).xyz;
            (sample as AnyNode).addAssign(rad.mul(w));
          }
        }
      }

      // Receiver irradiance integral: E_i = L_i · cos(θ_i) · Ω_i
      // where Ω_i is the solid angle of direction bin i in the octahedral grid.
      // Using per-bin solid angle instead of uniform 4π/N corrects the
      // non-uniform solid-angle distribution near the octahedral fold edges.
      // Reference: Cigolle et al. 2014, JCGT §A.2.
      const omega = solidAngles[d]!;
      (diffuse as AnyNode).addAssign((sample as AnyNode).mul(nDotL).mul(omega));
    }

    // No additional normalization factor: the per-bin Ω_i already encodes
    // the solid-angle weight for the irradiance integral E = Σ L_i·cos(θ_i)·Ω_i.
    return diffuse as AnyNode;
  })();

  return {
    lightingNode: giDiffuseNode,
  };
}
