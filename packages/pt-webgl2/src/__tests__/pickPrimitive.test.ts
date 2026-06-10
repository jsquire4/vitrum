/**
 * Tests for PTEngineWebGL2.debug.pickPrimitive (T3.G #30).
 *
 * The full pick path: pickPrimitive(x,y) → unproject through last-frame camera
 * → ray-cast against the retained core Scene (CPU, no GPU readback).
 *
 * The mock-GL device lets us call renderFrame() to deposit a camera without
 * a real WebGL2 context; the pick is then entirely CPU-side.
 */
import { describe, it, expect } from 'vitest';
import type { FrameInput, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';

// ── helpers ──────────────────────────────────────────────────────────────────

const MAT: MaterialSpec = { baseColor: [0.8, 0.8, 0.8], roughness: 1, metallic: 0 };

/** Axis-aligned quad (half-size 1) centred at (cx,cy,z). */
function quad(id: string, cx: number, cy: number, z: number): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([cx - 1, cy - 1, z, cx + 1, cy - 1, z, cx + 1, cy + 1, z, cx - 1, cy + 1, z]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: MAT,
  };
}

// three.js-style symmetric perspective (column-major, NDC z ∈ [-1,1]).
function makePerspective(fovDeg: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan((fovDeg * Math.PI) / 360);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

// Camera at (0,0,5) looking down −Z (view = translate -5 in Z, column-major).
const VIEW_MAT = asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]));
const PROJ_MAT = asMat4(makePerspective(60, 1, 0.1, 100));
const W = 200;
const H = 200;

function makeFrame(): FrameInput {
  return {
    viewMatrix: VIEW_MAT,
    projMatrix: PROJ_MAT,
    cameraPosition: [0, 0, 5],
    viewport: { width: W, height: H, devicePixelRatio: 1 },
    frameSeed: 1,
    frameIndex: 0,
  };
}

function makeScene(...prims: MeshPrimitive[]): Scene {
  return { primitives: prims, emitters: [], environment: { kind: 'none' } };
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe('PTEngineWebGL2.debug.pickPrimitive (T3.G #30)', () => {
  it('capabilities.debugSurface is true', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    expect(engine.capabilities.debugSurface).toBe(true);
    engine.dispose();
  });

  it('debug.pickPrimitive is a function', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    expect(typeof engine.debug?.pickPrimitive).toBe('function');
    engine.dispose();
  });

  it('returns null before the first renderFrame (no camera yet)', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    engine.setScene(makeScene(quad('Q', 0, 0, 0)));
    // No renderFrame yet → no camera → null.
    expect(engine.debug?.pickPrimitive?.(W / 2, H / 2)).toBeNull();
    engine.dispose();
  });

  it('picks the centred quad after renderFrame deposits the camera', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    engine.setScene(makeScene(quad('CentreQuad', 0, 0, 0)));
    engine.renderFrame(makeFrame());
    expect(engine.debug?.pickPrimitive?.(W / 2, H / 2)).toBe('CentreQuad');
    engine.dispose();
  });

  it('returns null when the centre ray misses all geometry', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    // A quad offset far to the right — the centre ray doesn't hit it.
    engine.setScene(makeScene(quad('OffCenter', 50, 0, 0)));
    engine.renderFrame(makeFrame());
    expect(engine.debug?.pickPrimitive?.(W / 2, H / 2)).toBeNull();
    engine.dispose();
  });

  it('picks the NEARER of two overlapping quads', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    engine.setScene(makeScene(quad('Far', 0, 0, 0), quad('Near', 0, 0, 2)));
    engine.renderFrame(makeFrame());
    // 'Near' is at z=2, closer to camera at z=5 than 'Far' at z=0.
    expect(engine.debug?.pickPrimitive?.(W / 2, H / 2)).toBe('Near');
    engine.dispose();
  });
});
