import { describe, it, expect } from 'vitest';
import type { FrameInput } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import {
  packFrameParams,
  multiplyMat4,
  invertMat4,
  FRAME_PARAMS_SLOTS,
  FRAME_PARAMS_BYTE_SIZE,
  type FrameParamsConfig,
  type FrameParamsScene,
} from './frameParamsPacker.js';

// A representative perspective projection (fov ~50°, aspect 1, near 0.1, far 100),
// column-major. Invertible.
const PROJ = asMat4(
  new Float32Array([
    2.144_507, 0, 0, 0,
    0, 2.144_507, 0, 0,
    0, 0, -1.002_002, -1,
    0, 0, -0.200_200, 0,
  ]),
);

// A non-trivial view matrix: a camera at (3, 4, 5) with a small rotation.
// Built as the inverse of a known camera-world transform so we can sanity-check
// the recovered cameraWorldMatrix.
const CAMERA_WORLD = asMat4(
  new Float32Array([
    0.866_025, 0, -0.5, 0, // column 0
    0, 1, 0, 0, // column 1
    0.5, 0, 0.866_025, 0, // column 2
    3, 4, 5, 1, // column 3 (translation)
  ]),
);
const VIEW = asMat4(invertMat4(CAMERA_WORLD)!);

function makeInput(over: Partial<FrameInput> = {}): FrameInput {
  return {
    frameIndex: 7,
    frameSeed: 0x1234_5678,
    viewMatrix: VIEW,
    projMatrix: PROJ,
    cameraPosition: [3, 4, 5],
    viewport: { width: 800, height: 600, devicePixelRatio: 1 },
    ...over,
  } as FrameInput;
}

const CFG: FrameParamsConfig = {
  maxBounces: 12,
  spectral: true,
  causticStrategy: 'manifold-nee',
  mneeMaxIterations: 32,
  mneeMaxChainLength: 4,
  bdpt: true,
};

const SCENE: FrameParamsScene = {
  triangleCount: 4242,
  bvhNodeCount: 1717,
  lightCount: 3,
  hasEnvironmentMap: true,
  environmentMapWidth: 2048,
  environmentMapHeight: 1024,
};

describe('packFrameParams', () => {
  it('produces a std140-sized buffer', () => {
    const buf = packFrameParams(CFG, SCENE, makeInput(), 400, 300);
    expect(buf.byteLength).toBe(FRAME_PARAMS_BYTE_SIZE);
    expect(buf.byteLength % 16).toBe(0);
  });

  it('writes resolution into the resolution slots', () => {
    const buf = packFrameParams(CFG, SCENE, makeInput(), 400, 300);
    const u = new Uint32Array(buf);
    expect(u[FRAME_PARAMS_SLOTS.resolutionX]).toBe(400);
    expect(u[FRAME_PARAMS_SLOTS.resolutionY]).toBe(300);
  });

  it('packs the scalar scene counts + frame seed + bounces', () => {
    const buf = packFrameParams(CFG, SCENE, makeInput(), 400, 300);
    const u = new Uint32Array(buf);
    expect(u[FRAME_PARAMS_SLOTS.frameSeed]).toBe(0x1234_5678);
    expect(u[FRAME_PARAMS_SLOTS.bounces]).toBe(12);
    expect(u[FRAME_PARAMS_SLOTS.triangleCount]).toBe(4242);
    expect(u[FRAME_PARAMS_SLOTS.bvhNodeCount]).toBe(1717);
    expect(u[FRAME_PARAMS_SLOTS.lightCount]).toBe(3);
    expect(u[FRAME_PARAMS_SLOTS.hasEnvironmentMap]).toBe(1);
    expect(u[FRAME_PARAMS_SLOTS.environmentMapWidth]).toBe(2048);
    expect(u[FRAME_PARAMS_SLOTS.environmentMapHeight]).toBe(1024);
  });

  it('packs the spectral scalars + CMF integrals', () => {
    const buf = packFrameParams(CFG, SCENE, makeInput(), 400, 300);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    expect(u[FRAME_PARAMS_SLOTS.uSpectralRendering]).toBe(1);
    expect(f[FRAME_PARAMS_SLOTS.heroLambdaNm]).toBeCloseTo(550, 5);
    expect(f[FRAME_PARAMS_SLOTS.heroPdf]).toBeCloseTo(1, 6);
    // CMF integrals are positive, finite constants.
    expect(f[FRAME_PARAMS_SLOTS.cmfIntegralX]).toBeGreaterThan(0);
    expect(f[FRAME_PARAMS_SLOTS.cmfIntegralY]).toBeGreaterThan(0);
    expect(f[FRAME_PARAMS_SLOTS.cmfIntegralZ]).toBeGreaterThan(0);
  });

  it('maps the caustic-strategy enum: none→0, manifold-nee→1, photon-map→2', () => {
    const slot = FRAME_PARAMS_SLOTS.uCausticStrategy;
    const none = new Uint32Array(
      packFrameParams({ ...CFG, causticStrategy: 'none' }, SCENE, makeInput(), 1, 1),
    );
    const manifold = new Uint32Array(
      packFrameParams({ ...CFG, causticStrategy: 'manifold-nee' }, SCENE, makeInput(), 1, 1),
    );
    const photon = new Uint32Array(
      packFrameParams({ ...CFG, causticStrategy: 'photon-map' }, SCENE, makeInput(), 1, 1),
    );
    expect(none[slot]).toBe(0);
    expect(manifold[slot]).toBe(1);
    expect(photon[slot]).toBe(2);
  });

  it('packs the MNEE + BDPT scalars', () => {
    const buf = packFrameParams(CFG, SCENE, makeInput(), 1, 1);
    const u = new Uint32Array(buf);
    expect(u[FRAME_PARAMS_SLOTS.uMneeMaxIterations]).toBe(32);
    expect(u[FRAME_PARAMS_SLOTS.uMneeMaxChainLength]).toBe(4);
    expect(u[FRAME_PARAMS_SLOTS.uBdptEnabled]).toBe(1);
    expect(u[FRAME_PARAMS_SLOTS.uBdptMaxEyeDepth]).toBe(12);
    // bdpt:false → 0
    const off = new Uint32Array(
      packFrameParams({ ...CFG, bdpt: false }, SCENE, makeInput(), 1, 1),
    );
    expect(off[FRAME_PARAMS_SLOTS.uBdptEnabled]).toBe(0);
  });

  it('mat4 slots are 16-byte aligned', () => {
    // float-slot × 4 bytes must be a multiple of 16.
    expect((FRAME_PARAMS_SLOTS.cameraWorldMatrix * 4) % 16).toBe(0);
    expect((FRAME_PARAMS_SLOTS.invProjectionMatrix * 4) % 16).toBe(0);
  });

  it('cameraWorldMatrix · viewMatrix ≈ identity (proves the inverse)', () => {
    const buf = packFrameParams(CFG, SCENE, makeInput(), 1, 1);
    const f = new Float32Array(buf);
    const cameraWorld = f.slice(
      FRAME_PARAMS_SLOTS.cameraWorldMatrix,
      FRAME_PARAMS_SLOTS.cameraWorldMatrix + 16,
    );
    const product = multiplyMat4(cameraWorld, VIEW);
    const identity = [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ];
    for (let i = 0; i < 16; i += 1) {
      expect(product[i]).toBeCloseTo(identity[i]!, 4);
    }
  });

  it('invProjectionMatrix · projMatrix ≈ identity', () => {
    const buf = packFrameParams(CFG, SCENE, makeInput(), 1, 1);
    const f = new Float32Array(buf);
    const invProj = f.slice(
      FRAME_PARAMS_SLOTS.invProjectionMatrix,
      FRAME_PARAMS_SLOTS.invProjectionMatrix + 16,
    );
    const product = multiplyMat4(invProj, PROJ);
    const identity = [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ];
    for (let i = 0; i < 16; i += 1) {
      expect(product[i]).toBeCloseTo(identity[i]!, 4);
    }
  });

  it('throws on a singular viewMatrix', () => {
    expect(() =>
      packFrameParams(CFG, SCENE, makeInput({ viewMatrix: asMat4(new Float32Array(16)) }), 1, 1),
    ).toThrow(/non-invertible viewMatrix/);
  });

  it('throws on a singular projMatrix', () => {
    expect(() =>
      packFrameParams(CFG, SCENE, makeInput({ projMatrix: asMat4(new Float32Array(16)) }), 1, 1),
    ).toThrow(/non-invertible projMatrix/);
  });
});

describe('invertMat4', () => {
  it('returns null for a singular matrix', () => {
    expect(invertMat4(new Float32Array(16))).toBeNull();
  });

  it('inverts a known transform round-trip', () => {
    const inv = invertMat4(CAMERA_WORLD)!;
    expect(inv).not.toBeNull();
    const product = multiplyMat4(inv, CAMERA_WORLD);
    for (let i = 0; i < 16; i += 1) {
      const expected = i % 5 === 0 ? 1 : 0; // diagonal of column-major 4×4
      expect(product[i]).toBeCloseTo(expected, 4);
    }
  });
});
