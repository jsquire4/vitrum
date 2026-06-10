/**
 * createCornellScene — shared Cornell-box scene factory.
 *
 * Builds a minimal physically-correct Cornell box using the @vitrum/core
 * Scene contract: five quad-panels (floor, ceiling, back wall, left/right
 * walls) plus one rect-area emitter centred on the ceiling.
 *
 * Geometry uses triangle-list (no index buffer). All coordinates are in the
 * range [-1, 1] in X/Z and [0, 2] in Y (Y-up, ground-level floor).
 *
 * The scene is camera-free (cameras live in FrameInput per the vitrum contract).
 * Hosts control quality via vitrumSpp / vitrumBounces URL params and the
 * engine's FrameInput quality field.
 */

import type { Scene, MeshPrimitive, RectAreaEmitter, MaterialSpec } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';

// ── Types ─────────────────────────────────────────────────────────────────────

type Corner = readonly [number, number, number];
type Quad   = readonly [Corner, Corner, Corner, Corner];

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Build a flat quad mesh from four corners (ccw winding, Y-up). */
function quad(
  id: string,
  corners: Quad,
  material: MaterialSpec,
): MeshPrimitive {
  const a = corners[0];
  const b = corners[1];
  const c = corners[2];
  const d = corners[3];

  // Two triangles: abc + acd (counter-clockwise from front face)
  const positions = new Float32Array([
    a[0], a[1], a[2],
    b[0], b[1], b[2],
    c[0], c[1], c[2],
    a[0], a[1], a[2],
    c[0], c[1], c[2],
    d[0], d[1], d[2],
  ]);

  // Compute flat normal from first triangle edges ab × ac
  const abx = b[0]-a[0], aby = b[1]-a[1], abz = b[2]-a[2];
  const acx = c[0]-a[0], acy = c[1]-a[1], acz = c[2]-a[2];
  const nx = aby*acz - abz*acy;
  const ny = abz*acx - abx*acz;
  const nz = abx*acy - aby*acx;
  const len = Math.sqrt(nx*nx + ny*ny + nz*nz) || 1;
  const nxn = nx/len, nyn = ny/len, nzn = nz/len;

  const normals = new Float32Array([
    nxn, nyn, nzn,
    nxn, nyn, nzn,
    nxn, nyn, nzn,
    nxn, nyn, nzn,
    nxn, nyn, nzn,
    nxn, nyn, nzn,
  ]);

  const uvs = new Float32Array([0,0, 1,0, 1,1, 0,0, 1,1, 0,1]);

  const identity16 = new Float32Array([
    1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1,
  ]);

  return {
    kind: 'mesh',
    id,
    positions,
    normals,
    uvs,
    material,
    transform: asMat4(identity16),
  };
}

// ── Materials ─────────────────────────────────────────────────────────────────

const white: MaterialSpec  = { baseColor: [0.73, 0.73, 0.73], roughness: 1.0, metallic: 0.0 };
const red: MaterialSpec    = { baseColor: [0.65, 0.05, 0.05], roughness: 1.0, metallic: 0.0 };
const green: MaterialSpec  = { baseColor: [0.12, 0.45, 0.12], roughness: 1.0, metallic: 0.0 };

// ── Geometry ──────────────────────────────────────────────────────────────────

// Cornell box spans [-1, 1] in X and Z, [0, 2] in Y.

const floor    = quad('floor',  [[-1,0,-1],[ 1,0,-1],[ 1,0, 1],[-1,0, 1]] as const, white);
const ceiling  = quad('ceil',   [[-1,2,-1],[-1,2, 1],[ 1,2, 1],[ 1,2,-1]] as const, white);
const backWall = quad('back',   [[-1,0,-1],[-1,2,-1],[ 1,2,-1],[ 1,0,-1]] as const, white);
const leftWall = quad('left',   [[-1,0,-1],[-1,0, 1],[-1,2, 1],[-1,2,-1]] as const, red);
const rightWall= quad('right',  [[ 1,0,-1],[ 1,2,-1],[ 1,2, 1],[ 1,0, 1]] as const, green);

// ── Emitter ───────────────────────────────────────────────────────────────────

// Rect-area light centred on the ceiling (Y=2, slightly toward back wall).
const ceilingEmitter: RectAreaEmitter = {
  kind: 'rect-area',
  id: 'ceiling-light',
  color: [1, 1, 1],
  intensity: 10,
  position: [0, 1.99, -0.25],
  uAxis:    [0.25, 0, 0],   // half-width 0.25 in X
  vAxis:    [0,    0, 0.25], // half-height 0.25 in Z
};

// ── Scene factory ─────────────────────────────────────────────────────────────

/**
 * Create a minimal Cornell box scene built on the @vitrum/core contract.
 *
 * Five Lambertian panels (floor / ceiling / back / left-red / right-green)
 * and one rect-area emitter on the ceiling. Environment is `'none'` — all
 * illumination comes from the ceiling emitter.
 *
 * The scene is static and camera-free. Hosts supply FrameInput camera matrices
 * each frame via `attachVitrum` / `createEngine`.
 *
 * @returns A @vitrum/core {@link Scene} ready for `engine.setScene()` or
 *   `createEngine({ scene })`.
 */
export function createCornellScene(): Scene {
  return {
    primitives: [floor, ceiling, backWall, leftWall, rightWall],
    emitters:   [ceilingEmitter],
    environment: { kind: 'none' },
  };
}
