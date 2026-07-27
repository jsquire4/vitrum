import { describe, expect, it } from 'vitest';
import type { FrameInput } from '@vitrum/core';
import { asMat4, resolveFrameCameraPosition } from '@vitrum/core';
import {
  FRAME_PARAMS_BUFFER_ALLOC_BYTES,
  packFrameParams,
  type FrameParamsEngineConfig,
  type FrameParamsSceneInputs,
} from '../frameParamsPacker.js';
import { FrameParamsSlot } from '../scene/frameParamsLayout.js';
import { invertMat4, multiplyMat4 } from '../math/mat4.js';

/**
 * GOLDEN / byte-identity test for the extracted FrameParamsPacker (Task 4.3).
 *
 * The packed UBO feeds the GPU at pinned offsets, so byte-identity is mandatory.
 * Two layers of proof here:
 *
 *  1. `reconstructExpected` re-derives the EXACT bytes the pre-extraction
 *     `#buildParamsBuffer` produced (same FrameParamsSlot offsets, same value
 *     expressions), independently of `packFrameParams`. We assert the full
 *     `Uint8Array` matches across a matrix of representative inputs (camera,
 *     scene config, BDPT on/off, tier, light-tree, spectral, caustics). If the
 *     extraction drifted by a single byte this fails.
 *
 *  2. Frozen literal goldens for a canonical input pin the absolute bytes so a
 *     future refactor of BOTH the packer and the reconstruction can't silently
 *     co-drift.
 */

// ── Independent reconstruction of the legacy #buildParamsBuffer body ──────────
function reconstructExpected(
  config: FrameParamsEngineConfig,
  sb: FrameParamsSceneInputs,
  input: FrameInput,
  width: number,
  height: number,
): ArrayBuffer {
  const vp = multiplyMat4(input.projMatrix, input.viewMatrix);
  const invVp = invertMat4(asMat4(vp));
  if (invVp == null) throw new Error('non-invertible vp');
  const ab = new ArrayBuffer(FRAME_PARAMS_BUFFER_ALLOC_BYTES);
  const u = new Uint32Array(ab);
  const f = new Float32Array(ab);
  u[FrameParamsSlot.width] = width;
  u[FrameParamsSlot.height] = height;
  u[FrameParamsSlot.frameIndex] = input.frameIndex >>> 0;
  u[FrameParamsSlot.frameSeed] = input.frameSeed >>> 0;
  u[FrameParamsSlot.triangleCount] = sb.triangleCount >>> 0;
  u[FrameParamsSlot.maxBounces] = config.activeBounces >>> 0;
  u[FrameParamsSlot.bvhNodeCount] = sb.bvhNodeCount >>> 0;
  // Item 24 — zero analyticCount on lite tier (mirrors frameParamsPacker.ts).
  u[FrameParamsSlot.analyticCount] =
    config.traceTier === 'lite' ? 0 : sb.analyticCount >>> 0;
  u[FrameParamsSlot.pointLightCount] = sb.pointLightCount >>> 0;
  u[FrameParamsSlot.spotLightCount] = sb.spotLightCount >>> 0;
  u[FrameParamsSlot.rectAreaLightCount] = sb.rectAreaLightCount >>> 0;
  u[FrameParamsSlot.meshAreaLightCount] = sb.meshAreaLightCount >>> 0;
  u[FrameParamsSlot.mneeMaxIterations] = config.mneeMaxIterations >>> 0;
  u[FrameParamsSlot.mneeMaxChainLength] = config.mneeMaxChainLength >>> 0;
  u[FrameParamsSlot.hasEnvironmentMap] = sb.hasEnvironmentMap ? 1 : 0;
  u[FrameParamsSlot.causticStrategy] =
    config.causticStrategy === 'manifold-nee'
      ? 1
      : config.causticStrategy === 'photon-map'
        ? 2
        : 0;
  u[FrameParamsSlot.environmentMapWidth] = sb.environmentMapWidth >>> 0;
  u[FrameParamsSlot.environmentMapHeight] = sb.environmentMapHeight >>> 0;
  f[FrameParamsSlot.triIntersectEpsilon] = 1e-5;
  u[FrameParamsSlot.tlasNodeCount] = sb.tlasNodeCount >>> 0;
  u[FrameParamsSlot.spectralEnabled] = config.spectralEnabled ? 1 : 0;
  f[FrameParamsSlot.heroLambdaNm] = 550.0;
  f[FrameParamsSlot.heroPdf] = 1.0;
  const bdptActive = config.bdpt && config.traceTier === 'full';
  u[FrameParamsSlot.bdptEnabled] = bdptActive ? 1 : 0;
  u[FrameParamsSlot.bdptMaxLightBounces] = config.bdptMaxLightBounces >>> 0;
  u[FrameParamsSlot.bdptMaxEyeDepth] = config.activeBounces >>> 0;
  const lightTreeOn =
    config.traceTier === 'full' && sb.lightTreeEnabled && config.lightTreeImportanceSampling;
  u[FrameParamsSlot.lightTreeEnabled] = lightTreeOn ? 1 : 0;
  u[FrameParamsSlot.lightTreeNodeCount] = lightTreeOn ? sb.lightTreeNodeCount >>> 0 : 0;
  // N-directional: kernel loops this many records from the directionalLights storage buffer.
  u[FrameParamsSlot.directionalLightCount] = sb.directionalLightCount >>> 0;
  f[FrameParamsSlot.sceneCenterX] = sb.sceneCenter[0];
  f[FrameParamsSlot.sceneCenterY] = sb.sceneCenter[1];
  f[FrameParamsSlot.sceneCenterZ] = sb.sceneCenter[2];
  f[FrameParamsSlot.sceneRadius] = Math.max(1e-3, sb.sceneRadius);
  u[FrameParamsSlot.directLightingMode] =
    config.directLightingMode === 'summed-expectation' ? 1 : 0;
  // H14-E: HDRI intensity in its own slot (slot 31), separate from environmentSun.w.
  f[FrameParamsSlot.environmentHdriIntensity] = sb.environmentHdriIntensity;
  const cameraPosition = resolveFrameCameraPosition(input);
  f[FrameParamsSlot.cameraPos] = cameraPosition[0];
  f[FrameParamsSlot.cameraPos + 1] = cameraPosition[1];
  f[FrameParamsSlot.cameraPos + 2] = cameraPosition[2];
  f[FrameParamsSlot.environmentTint] = sb.environmentTint[0];
  f[FrameParamsSlot.environmentTint + 1] = sb.environmentTint[1];
  f[FrameParamsSlot.environmentTint + 2] = sb.environmentTint[2];
  // H6: environmentTint.w now carries environmentHdriRotationY (was hardcoded 0).
  f[FrameParamsSlot.environmentTint + 3] = sb.environmentHdriRotationY;
  f[FrameParamsSlot.environmentSun] = sb.environmentSunDirection[0];
  f[FrameParamsSlot.environmentSun + 1] = sb.environmentSunDirection[1];
  f[FrameParamsSlot.environmentSun + 2] = sb.environmentSunDirection[2];
  f[FrameParamsSlot.environmentSun + 3] = sb.environmentSunStrength;
  f.set(invVp, FrameParamsSlot.invViewProj);
  f.set(vp, FrameParamsSlot.viewProj);
  const prevVp = multiplyMat4(
    input.prevProjMatrix ?? input.projMatrix,
    input.prevViewMatrix ?? input.viewMatrix,
  );
  f.set(prevVp, FrameParamsSlot.prevViewProj);
  return ab;
}

function makeSceneInputs(over: Partial<FrameParamsSceneInputs> = {}): FrameParamsSceneInputs {
  return {
    triangleCount: 12,
    bvhNodeCount: 23,
    analyticCount: 1,
    directionalLightCount: 1,
    pointLightCount: 2,
    spotLightCount: 1,
    rectAreaLightCount: 1,
    meshAreaLightCount: 3,
    hasEnvironmentMap: true,
    environmentMapWidth: 1024,
    environmentMapHeight: 512,
    tlasNodeCount: 7,
    lightTreeEnabled: true,
    lightTreeNodeCount: 9,
    sceneCenter: [4, 5, 6],
    sceneRadius: 7,
    environmentTint: [0.95, 0.97, 1.0],
    environmentSunDirection: [0.0, 1.0, 0.0],
    environmentSunStrength: 3.5,
    environmentHdriIntensity: 1.0,
    environmentHdriRotationY: 0,
    ...over,
  };
}

function makeConfig(over: Partial<FrameParamsEngineConfig> = {}): FrameParamsEngineConfig {
  return {
    activeBounces: 4,
    mneeMaxIterations: 8,
    mneeMaxChainLength: 3,
    causticStrategy: 'none',
    spectralEnabled: false,
    traceTier: 'full',
    bdpt: false,
    bdptMaxLightBounces: 3,
    lightTreeImportanceSampling: true,
    directLightingMode: 'sampled-selection',
    ...over,
  };
}

// A non-trivial perspective-ish projection × view so invVp/vp/prevVp are
// non-identity and exercise the matrix lanes.
const PROJ = asMat4(
  new Float32Array([
    1.5, 0, 0, 0,
    0, 2.0, 0, 0,
    0, 0, -1.002, -1,
    0, 0, -0.2, 0,
  ]),
);
const VIEW = asMat4(
  new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    -2, -3, -8, 1,
  ]),
);

function makeInput(over: Partial<FrameInput> = {}): FrameInput {
  return {
    frameIndex: 5,
    frameSeed: 0xabcd1234,
    cameraPosition: [2, 3, 8],
    viewMatrix: VIEW,
    projMatrix: PROJ,
    viewport: { width: 800, height: 600, devicePixelRatio: 1 },
    ...over,
  };
}

const MATRIX: ReadonlyArray<{
  readonly name: string;
  readonly config: FrameParamsEngineConfig;
  readonly sb: FrameParamsSceneInputs;
  readonly input: FrameInput;
  readonly width: number;
  readonly height: number;
}> = [
  {
    name: 'baseline full tier, lighttree on, no bdpt',
    config: makeConfig(),
    sb: makeSceneInputs(),
    input: makeInput(),
    width: 800,
    height: 600,
  },
  {
    name: 'bdpt on (full tier)',
    config: makeConfig({ bdpt: true, bdptMaxLightBounces: 2, activeBounces: 6 }),
    sb: makeSceneInputs(),
    input: makeInput({ frameIndex: 0, frameSeed: 1 }),
    width: 1280,
    height: 720,
  },
  {
    name: 'lite tier disables bdpt + lighttree',
    config: makeConfig({ traceTier: 'lite', bdpt: true }),
    sb: makeSceneInputs(),
    input: makeInput(),
    width: 640,
    height: 360,
  },
  {
    name: 'spectral + photon-map caustics',
    config: makeConfig({ spectralEnabled: true, causticStrategy: 'photon-map' }),
    sb: makeSceneInputs({ hasEnvironmentMap: false, environmentMapWidth: 0, environmentMapHeight: 0 }),
    input: makeInput(),
    width: 256,
    height: 256,
  },
  {
    // Item 24 — lite tier must zero analyticCount even when sb.analyticCount > 0.
    // The lite kernel has no analytic-primitive path; a phantom count would be an
    // unused-but-wrong value in the UBO. Full-tier path is unaffected.
    name: 'lite tier zeros analyticCount (item 24)',
    config: makeConfig({ traceTier: 'lite' }),
    sb: makeSceneInputs({ analyticCount: 5 }),
    input: makeInput(),
    width: 320,
    height: 240,
  },
  {
    name: 'manifold-nee caustics + lighttree forced off via flag',
    config: makeConfig({ causticStrategy: 'manifold-nee', lightTreeImportanceSampling: false }),
    sb: makeSceneInputs(),
    input: makeInput(),
    width: 100,
    height: 100,
  },
  {
    name: 'lighttree disabled by scene (lightTreeEnabled=false)',
    config: makeConfig(),
    sb: makeSceneInputs({ lightTreeEnabled: false }),
    input: makeInput(),
    width: 200,
    height: 200,
  },
  {
    name: 'inverse summed direct-light mode',
    config: makeConfig({ directLightingMode: 'summed-expectation' }),
    sb: makeSceneInputs(),
    input: makeInput(),
    width: 128,
    height: 128,
  },
  {
    name: 'with prev matrices for motion vectors',
    config: makeConfig(),
    sb: makeSceneInputs(),
    input: makeInput({
      prevProjMatrix: PROJ,
      prevViewMatrix: asMat4(
        new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          -1.5, -3, -7.5, 1,
        ]),
      ),
    }),
    width: 800,
    height: 600,
  },
];

describe('FrameParamsPacker — byte-identity golden (pt-webgpu Task 4.3)', () => {
  it('produces an exact generated-size buffer', () => {
    const out = packFrameParams(makeConfig(), makeSceneInputs(), makeInput(), 800, 600);
    expect(out.byteLength).toBe(FRAME_PARAMS_BUFFER_ALLOC_BYTES);
  });

  for (const tc of MATRIX) {
    it(`byte-identical to legacy reconstruction — ${tc.name}`, () => {
      const got = new Uint8Array(packFrameParams(tc.config, tc.sb, tc.input, tc.width, tc.height));
      const want = new Uint8Array(reconstructExpected(tc.config, tc.sb, tc.input, tc.width, tc.height));
      expect(got.length).toBe(FRAME_PARAMS_BUFFER_ALLOC_BYTES);
      expect(Array.from(got)).toEqual(Array.from(want));
    });
  }

  it('throws on a non-invertible view-projection matrix', () => {
    const singularInput = makeInput({
      viewMatrix: asMat4(new Float32Array(16)), // all-zero → singular
      projMatrix: asMat4(new Float32Array(16)),
    });
    expect(() => packFrameParams(makeConfig(), makeSceneInputs(), singularInput, 800, 600)).toThrow(
      /non-invertible/,
    );
  });

  it('H6 zero-rotation invariant: environmentTint.w = 0 when environmentHdriRotationY = 0', () => {
    // rotationY = 0 must write 0.0 to environmentTint.w — byte-identical to the
    // pre-H6 hardcoded 0 that was there before.
    const ab = packFrameParams(makeConfig(), makeSceneInputs({ environmentHdriRotationY: 0 }), makeInput(), 800, 600);
    const f = new Float32Array(ab);
    expect(f[FrameParamsSlot.environmentTint + 3]).toBe(0);
  });

  it('H6 packer writes non-zero environmentHdriRotationY to environmentTint.w', () => {
    const rotY = Math.PI / 4; // 45°
    const ab = packFrameParams(makeConfig(), makeSceneInputs({ environmentHdriRotationY: rotY }), makeInput(), 800, 600);
    const f = new Float32Array(ab);
    expect(f[FrameParamsSlot.environmentTint + 3]).toBeCloseTo(rotY, 6);
  });

  it('item 24 — lite tier writes analyticCount=0 regardless of sb.analyticCount', () => {
    // The lite kernel has no analytic-primitive path; writing the real analyticCount
    // would leave a phantom count that a future lite-kernel change could misread.
    // Verify that packFrameParams zeros it on lite tier and that full tier is unaffected.
    const liteAb = packFrameParams(
      makeConfig({ traceTier: 'lite' }),
      makeSceneInputs({ analyticCount: 7 }),
      makeInput(),
      400,
      300,
    );
    const liteU = new Uint32Array(liteAb);
    expect(liteU[FrameParamsSlot.analyticCount]).toBe(0);

    // Full tier must still write the real count.
    const fullAb = packFrameParams(
      makeConfig({ traceTier: 'full' }),
      makeSceneInputs({ analyticCount: 7 }),
      makeInput(),
      400,
      300,
    );
    const fullU = new Uint32Array(fullAb);
    expect(fullU[FrameParamsSlot.analyticCount]).toBe(7);
  });

  it('packs inverse direct-light summed-expectation mode into the scalar tail', () => {
    const sampled = new Uint32Array(packFrameParams(
      makeConfig({ directLightingMode: 'sampled-selection' }),
      makeSceneInputs(),
      makeInput(),
      400,
      300,
    ));
    const summed = new Uint32Array(packFrameParams(
      makeConfig({ directLightingMode: 'summed-expectation' }),
      makeSceneInputs(),
      makeInput(),
      400,
      300,
    ));
    expect(FrameParamsSlot.directLightingMode).toBe(93);
    expect(sampled[FrameParamsSlot.directLightingMode]).toBe(0);
    expect(summed[FrameParamsSlot.directLightingMode]).toBe(1);
  });

  it('keeps neutral hero lanes because spectral sampling is invocation-local', () => {
    const packed = new Float32Array(packFrameParams(
      makeConfig({ spectralEnabled: true, bdpt: true }),
      makeSceneInputs(),
      makeInput(),
      800,
      600,
    ));
    expect(packed[FrameParamsSlot.heroLambdaNm]).toBe(550);
    expect(packed[FrameParamsSlot.heroPdf]).toBe(1);
  });

  it('frozen literal golden for the canonical baseline input', () => {
    // Pins the absolute u32/f32 slot values so the packer + the reconstruction
    // cannot silently co-drift. Spot-checks the scalar header + camera/environment
    // lanes (matrix lanes are covered by the byte-identity matrix above).
    const ab = packFrameParams(makeConfig(), makeSceneInputs(), makeInput(), 800, 600);
    const u = new Uint32Array(ab);
    const f = new Float32Array(ab);
    expect(u[FrameParamsSlot.width]).toBe(800);
    expect(u[FrameParamsSlot.height]).toBe(600);
    expect(u[FrameParamsSlot.frameIndex]).toBe(5);
    expect(u[FrameParamsSlot.frameSeed]).toBe(0xabcd1234);
    expect(u[FrameParamsSlot.triangleCount]).toBe(12);
    expect(u[FrameParamsSlot.maxBounces]).toBe(4);
    expect(u[FrameParamsSlot.bvhNodeCount]).toBe(23);
    expect(u[FrameParamsSlot.analyticCount]).toBe(1);
    expect(u[FrameParamsSlot.pointLightCount]).toBe(2);
    expect(u[FrameParamsSlot.spotLightCount]).toBe(1);
    expect(u[FrameParamsSlot.rectAreaLightCount]).toBe(1);
    expect(u[FrameParamsSlot.meshAreaLightCount]).toBe(3);
    expect(u[FrameParamsSlot.mneeMaxIterations]).toBe(8);
    expect(u[FrameParamsSlot.mneeMaxChainLength]).toBe(3);
    expect(u[FrameParamsSlot.hasEnvironmentMap]).toBe(1);
    expect(u[FrameParamsSlot.causticStrategy]).toBe(0);
    expect(u[FrameParamsSlot.environmentMapWidth]).toBe(1024);
    expect(u[FrameParamsSlot.environmentMapHeight]).toBe(512);
    expect(f[FrameParamsSlot.triIntersectEpsilon]).toBeCloseTo(1e-5, 10);
    expect(u[FrameParamsSlot.tlasNodeCount]).toBe(7);
    expect(u[FrameParamsSlot.spectralEnabled]).toBe(0);
    expect(f[FrameParamsSlot.heroLambdaNm]).toBe(550);
    expect(f[FrameParamsSlot.heroPdf]).toBe(1);
    expect(u[FrameParamsSlot.bdptEnabled]).toBe(0);
    expect(u[FrameParamsSlot.bdptMaxLightBounces]).toBe(3);
    expect(u[FrameParamsSlot.bdptMaxEyeDepth]).toBe(4);
    expect(u[FrameParamsSlot.lightTreeEnabled]).toBe(1);
    expect(u[FrameParamsSlot.lightTreeNodeCount]).toBe(9);
    // N-directional: directionalLightCount was set to 1 in makeSceneInputs.
    expect(u[FrameParamsSlot.directionalLightCount]).toBe(1);
    expect(f[FrameParamsSlot.sceneCenterX]).toBe(4);
    expect(f[FrameParamsSlot.sceneCenterY]).toBe(5);
    expect(f[FrameParamsSlot.sceneCenterZ]).toBe(6);
    expect(f[FrameParamsSlot.sceneRadius]).toBe(7);
    expect(u[FrameParamsSlot.directLightingMode]).toBe(0);
    expect(f[FrameParamsSlot.cameraPos]).toBe(2);
    expect(f[FrameParamsSlot.cameraPos + 1]).toBe(3);
    expect(f[FrameParamsSlot.cameraPos + 2]).toBe(8);
    expect(f[FrameParamsSlot.environmentHdriIntensity]).toBe(1);
    expect(f[FrameParamsSlot.environmentSun + 3]).toBe(3.5);
  });
});
