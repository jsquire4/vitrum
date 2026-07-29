/**
 * Phase-0 productization — TRUE quarter-resolution GTAO path.
 *
 * `gtaoMode: 'quarter'` was a declared option that the pipeline silently
 * treated as `'on'` (half-res): no quarter-res AO target existed, so selecting
 * it gave half-res cost while promising the cheaper quarter-res. This suite
 * pins the real quarter-res path:
 *
 *   - `createGtaoFrameResources(..., downscale)` sizes the AO compute target at
 *     `W/downscale × H/downscale` — downscale 2 ⇒ half-res ('on'); downscale 4
 *     ⇒ quarter-res ('quarter'). The full-res AO target shade reads stays
 *     W × H at every downscale.
 *   - `GTAOPass.dispatch` dispatches `ceil((W/ds)/8) × ceil((H/ds)/8)`
 *     workgroups and packs `gtaoDownscale` into the GTAO UBO (6th f32, offset
 *     20) so both gtao + gtaoUpsample shaders map between the AO grid and
 *     full-res coords.
 *   - Both shaders read `gtaoDownscale` instead of the prior hardcoded `÷2`.
 *   - End-to-end: a pipeline initialized with `gtaoMode:'quarter'` allocates a
 *     quarter-res AO target distinct from `'on'`; `'off'` / `'on'` unchanged.
 */

import { describe, it, expect, vi } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import { createGtaoFrameResources } from '../src/pipeline/frameResources/createGtaoFrameResources.js';
import { createFrameResources } from '../src/pipeline/resourceManager.js';
import { GTAOPass } from '../src/pipeline/passes/GTAOPass.js';
import { GTAO_COMMON_WGSL } from '../src/shaders/gtaoCommon.wgsl.js';
import { GTAO_WGSL } from '../src/shaders/gtao.wgsl.js';
import { GTAO_UPSAMPLE_WGSL } from '../src/shaders/gtaoUpsample.wgsl.js';
import type { PassDispatchContext } from '../src/pipeline/Pass.js';

installWebGPUPolyfills();

type Vec3 = readonly [number, number, number];

interface NormalDepthSample {
  normal: Vec3;
  depth: number;
}

function add3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scale3(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function clamp01Vec3(v: Vec3): Vec3 {
  return [
    Math.min(1, Math.max(0, v[0])),
    Math.min(1, Math.max(0, v[1])),
    Math.min(1, Math.max(0, v[2])),
  ];
}

function normalDepthAt(grid: readonly NormalDepthSample[][], x: number, y: number): NormalDepthSample {
  return grid[y]![x]!;
}

function aoAt(grid: readonly Vec3[][], x: number, y: number): Vec3 {
  return grid[y]![x]!;
}

function gtaoSimilarityWeight(
  centerDepth: number,
  centerNormal: Vec3,
  sampleDepth: number,
  sampleNormal: Vec3,
  bilateralDepthSigma: number,
): number {
  const depthDelta = Math.abs(centerDepth - sampleDepth);
  const sigma = Math.max(1e-6, bilateralDepthSigma);
  const depthW = Math.exp(-depthDelta / (2 * sigma * sigma));
  const nDot = Math.max(0, dot3(centerNormal, sampleNormal));
  return depthW * nDot ** 16;
}

function gtaoUpsampleReference(params: {
  fullX: number;
  fullY: number;
  fullWidth: number;
  fullHeight: number;
  downscale: number;
  normalDepth: readonly NormalDepthSample[][];
  aoHalf: readonly Vec3[][];
  bilateralDepthSigma: number;
}): Vec3 {
  const { fullX, fullY, fullWidth, fullHeight, downscale, normalDepth, aoHalf, bilateralDepthSigma } = params;
  const ds = Math.max(1, Math.trunc(downscale));
  const halfWidth = Math.trunc(fullWidth / ds);
  const halfHeight = Math.trunc(fullHeight / ds);
  const center = normalDepthAt(normalDepth, fullX, fullY);

  if (center.depth < 1e-4) return [1, 1, 1];

  const halfX = Math.trunc(fullX / ds);
  const halfY = Math.trunc(fullY / ds);
  let sumAO: Vec3 = [0, 0, 0];
  let sumW = 0;

  for (let dy = 0; dy < 2; dy += 1) {
    for (let dx = 0; dx < 2; dx += 1) {
      const sampleHalfX = Math.min(halfX + dx, halfWidth - 1);
      const sampleHalfY = Math.min(halfY + dy, halfHeight - 1);
      const sampleFullX = Math.min(sampleHalfX * ds + Math.trunc(ds / 2), fullWidth - 1);
      const sampleFullY = Math.min(sampleHalfY * ds + Math.trunc(ds / 2), fullHeight - 1);
      const sample = normalDepthAt(normalDepth, sampleFullX, sampleFullY);
      const w = gtaoSimilarityWeight(center.depth, center.normal, sample.depth, sample.normal, bilateralDepthSigma);
      sumAO = add3(sumAO, scale3(aoAt(aoHalf, sampleHalfX, sampleHalfY), w));
      sumW += w;
    }
  }

  if (sumW > 1e-4) return clamp01Vec3(scale3(sumAO, 1 / sumW));

  const sx1 = Math.min(halfX + 1, halfWidth - 1);
  const sy1 = Math.min(halfY + 1, halfHeight - 1);
  return clamp01Vec3(scale3(add3(
    add3(aoAt(aoHalf, halfX, halfY), aoAt(aoHalf, sx1, halfY)),
    add3(aoAt(aoHalf, halfX, sy1), aoAt(aoHalf, sx1, sy1)),
  ), 0.25));
}

// ── Size-capturing mock device ──────────────────────────────────────────────
//
// Returns texture mocks that reflect the createTexture descriptor's
// size/format/label so tests can assert the AO target dimensions. `size` is
// the WebGPU [width, height(, depth)] tuple form used by the factories.

interface CapturedTexture {
  width: number;
  height: number;
  format: string;
  label?: string;
}

function makeSizingDevice(): GPUDevice {
  return {
    createTexture: vi.fn((desc: { size: number[]; format: string; label?: string }) => {
      const tex = {
        width: desc.size[0]!,
        height: desc.size[1]!,
        format: desc.format,
        label: desc.label,
        destroy: () => {},
        createView: () => ({}),
      };
      return tex as unknown as GPUTexture;
    }),
    createBuffer: vi.fn((desc: { size: number; usage: number; label?: string }) =>
      ({ size: desc.size, usage: desc.usage, label: desc.label, destroy: () => {} }) as unknown as GPUBuffer),
    createSampler: vi.fn(() => ({}) as unknown as GPUSampler),
    queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
  } as unknown as GPUDevice;
}

// ── createGtaoFrameResources — true downscale sizing ─────────────────────────

describe('createGtaoFrameResources — downscale sizes the AO compute target', () => {
  it('downscale 2 ("on") allocates a HALF-res AO target (W/2 × H/2)', () => {
    const dev = makeSizingDevice();
    const r = createGtaoFrameResources(dev, 256, 128, 2);
    const half = r.aoHalfTexture as unknown as CapturedTexture;
    expect([half.width, half.height]).toEqual([128, 64]);
  });

  it('downscale 4 ("quarter") allocates a QUARTER-res AO target (W/4 × H/4)', () => {
    const dev = makeSizingDevice();
    const r = createGtaoFrameResources(dev, 256, 128, 4);
    const quarter = r.aoHalfTexture as unknown as CapturedTexture;
    expect([quarter.width, quarter.height]).toEqual([64, 32]);
  });

  it('the quarter-res target is a real step BELOW the half-res ("on") target', () => {
    const dev = makeSizingDevice();
    const onTex = createGtaoFrameResources(dev, 256, 256, 2).aoHalfTexture as unknown as CapturedTexture;
    const quarterTex = createGtaoFrameResources(dev, 256, 256, 4).aoHalfTexture as unknown as CapturedTexture;
    // Quarter is half of half per axis (1/4 the area = 1/16 of full-res).
    expect(quarterTex.width).toBe(onTex.width / 2);
    expect(quarterTex.height).toBe(onTex.height / 2);
    expect(quarterTex.width).toBe(256 / 4);
  });

  it('the full-res AO target (shade reads this) stays W × H at every downscale', () => {
    const dev = makeSizingDevice();
    const r2 = createGtaoFrameResources(dev, 256, 128, 2);
    const r4 = createGtaoFrameResources(dev, 256, 128, 4);
    const full2 = r2.aoFullTexture as unknown as CapturedTexture;
    const full4 = r4.aoFullTexture as unknown as CapturedTexture;
    expect([full2.width, full2.height]).toEqual([256, 128]);
    expect([full4.width, full4.height]).toEqual([256, 128]);
  });

  it('defaults to half-res (downscale 2) when the factor is omitted — "on" unchanged', () => {
    const dev = makeSizingDevice();
    const r = createGtaoFrameResources(dev, 256, 256);
    const half = r.aoHalfTexture as unknown as CapturedTexture;
    expect([half.width, half.height]).toEqual([128, 128]);
  });

  it('clamps a bad downscale (0) to 1 so it never produces a zero-sized target', () => {
    const dev = makeSizingDevice();
    const r = createGtaoFrameResources(dev, 64, 64, 0);
    const tex = r.aoHalfTexture as unknown as CapturedTexture;
    expect(tex.width).toBeGreaterThanOrEqual(1);
    expect(tex.height).toBeGreaterThanOrEqual(1);
  });
});

// ── createFrameResources threads the downscale through ───────────────────────

describe('createFrameResources — gtaoDownscale option threads to the AO target', () => {
  it('gtaoDownscale: 4 sizes gtao.aoHalfTexture at quarter-res', () => {
    const dev = makeSizingDevice();
    const res = createFrameResources(dev, 256, 256, { gtaoDownscale: 4 });
    const tex = res.gtao.aoHalfTexture as unknown as CapturedTexture;
    expect([tex.width, tex.height]).toEqual([64, 64]);
  });

  it('omitting gtaoDownscale keeps the half-res target ("on" byte-compatible)', () => {
    const dev = makeSizingDevice();
    const res = createFrameResources(dev, 256, 256);
    const tex = res.gtao.aoHalfTexture as unknown as CapturedTexture;
    expect([tex.width, tex.height]).toEqual([128, 128]);
  });
});

// ── GTAOPass — dispatch resolution + UBO packing ─────────────────────────────

interface CapturedDispatch { label: string; wgX: number; wgY: number; wgZ: number; }

function makeDispatchCtx(
  width: number,
  height: number,
  gtaoDownscale: number,
  captured: CapturedDispatch[],
  ubo: { bytes?: ArrayBuffer },
): PassDispatchContext {
  const device = {
    createBindGroup: () => ({}),
    queue: {
      writeBuffer: (_buf: unknown, _off: number, data: ArrayBuffer) => { ubo.bytes = data; },
    },
  } as unknown as GPUDevice;
  const encoder = {
    beginComputePass: (desc: { label: string }) => ({
      setPipeline: () => {},
      setBindGroup: () => {},
      dispatchWorkgroups: (x: number, y: number, z: number) =>
        captured.push({ label: desc.label, wgX: x, wgY: y, wgZ: z }),
      end: () => {},
    }),
  } as unknown as GPUCommandEncoder;

  // Prime the BGL cache so getGTAOBindGroupLayout short-circuits (no real
  // createBindGroupLayout needed); buildBindGroupFromTable then just calls the
  // mock device.createBindGroup which returns {}.
  const bglCache = { gtao: {} } as unknown as PassDispatchContext['bglCache'];

  const view = () => ({}) as unknown as GPUTextureView;
  const resources = {
    gtao: {
      aoHalfTexture: { createView: view },
      gtaoUboBuffer: { size: 96, usage: 0 },
    },
    common: {
      gNormalDepthTexture: { createView: view },
      albedoTexture: { createView: view },
    },
  } as unknown as PassDispatchContext['resources'];

  // Minimal camera projMatrix (only [5] is read for tanFovHalf) + gtao params.
  const inputs = {
    camera: {
      projMatrix: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      viewMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    },
    gtao: {
      gtaoRadiusPx: 32,
      gtaoIntensity: 1,
      gtaoDepthThreshold: 1,
      gtaoBilateralDepthSigma: 1,
    },
  } as unknown as PassDispatchContext['inputs'];

  return {
    device, encoder, width, height,
    frameIndex: 0, frameCount: 0,
    bglCache, resources, inputs,
    frameBindGroup: {} as unknown as GPUBindGroup,
    diSpatialReverseFrameBindGroup: {} as unknown as GPUBindGroup,
    diTerminalFrameBindGroup: {} as unknown as GPUBindGroup,
    diTerminalReservoirBuffer: { size: 256 } as unknown as GPUBuffer,
    risGiFrameBindGroup: {} as unknown as GPUBindGroup,
    sceneBindGroup: {} as unknown as GPUBindGroup,
    uboBindGroup: {} as unknown as GPUBindGroup,
    hybridLayersBindGroup: {} as unknown as GPUBindGroup,
    shadeHybridLayersBindGroup: {} as unknown as GPUBindGroup,
    buildTransparentOitBindGroup: () => ({} as unknown as GPUBindGroup),
    lightTreeBindGroup: {} as unknown as GPUBindGroup,
    wgX: Math.ceil(width / 8), wgY: Math.ceil(height / 8),
    wgX16: 0, wgY16: 0, halfWgX: 0, halfWgY: 0,
    checkerboardWgX: 0, checkerboardWgY: 0,
    checkerboardOn: false, frameParity: 0,
    welfordPing: 0,
    gtaoDownscale,
    gNormalDepthView: {} as unknown as GPUTextureView,
    computeDesc: (label) => ({ label }),
    renderTimestampWrites: () => undefined,
    frameState: {} as unknown as PassDispatchContext['frameState'],
  };
}

describe('GTAOPass.dispatch — resolution + UBO downscale', () => {
  it('"on" (downscale 2) dispatches at HALF-res: ceil((W/2)/8) × ceil((H/2)/8)', () => {
    const captured: CapturedDispatch[] = [];
    const ubo: { bytes?: ArrayBuffer } = {};
    const pass = new GTAOPass({} as unknown as GPUComputePipeline);
    pass.dispatch(makeDispatchCtx(256, 128, 2, captured, ubo));
    expect(captured).toHaveLength(1);
    expect(captured[0]!.label).toBe('gtao');
    expect(captured[0]!.wgX).toBe(Math.ceil(128 / 8)); // (256/2)/8 = 16
    expect(captured[0]!.wgY).toBe(Math.ceil(64 / 8));   // (128/2)/8 = 8
  });

  it('"quarter" (downscale 4) dispatches at QUARTER-res: ceil((W/4)/8) × ceil((H/4)/8)', () => {
    const captured: CapturedDispatch[] = [];
    const ubo: { bytes?: ArrayBuffer } = {};
    const pass = new GTAOPass({} as unknown as GPUComputePipeline);
    pass.dispatch(makeDispatchCtx(256, 128, 4, captured, ubo));
    expect(captured).toHaveLength(1);
    expect(captured[0]!.wgX).toBe(Math.ceil(64 / 8)); // (256/4)/8 = 8
    expect(captured[0]!.wgY).toBe(Math.ceil(32 / 8)); // (128/4)/8 = 4
  });

  it('quarter dispatches FEWER workgroups than on (genuinely cheaper, not a no-op)', () => {
    const onCap: CapturedDispatch[] = [];
    const qCap: CapturedDispatch[] = [];
    new GTAOPass({} as unknown as GPUComputePipeline).dispatch(makeDispatchCtx(512, 512, 2, onCap, {}));
    new GTAOPass({} as unknown as GPUComputePipeline).dispatch(makeDispatchCtx(512, 512, 4, qCap, {}));
    const onWG = onCap[0]!.wgX * onCap[0]!.wgY;
    const qWG = qCap[0]!.wgX * qCap[0]!.wgY;
    // 512: on = (32×32)=1024, quarter = (16×16)=256 → exactly 1/4 the workgroups.
    expect(qWG).toBeLessThan(onWG);
    expect(qWG).toBe(onWG / 4);
  });

  it('packs gtaoDownscale into the GTAO UBO 6th f32 (byte offset 20)', () => {
    const captured: CapturedDispatch[] = [];
    const ubo: { bytes?: ArrayBuffer } = {};
    new GTAOPass({} as unknown as GPUComputePipeline).dispatch(makeDispatchCtx(256, 256, 4, captured, ubo));
    expect(ubo.bytes).toBeDefined();
    const dv = new DataView(ubo.bytes!);
    // [0]=tanFovHalf [4]=radiusPx [8]=intensity [12]=depthThresh
    // [16]=bilateralDepthSigma [20]=gtaoDownscale [24]=_pad1 [28]=_pad2,
    // followed by the 64-byte world-to-view matrix.
    expect(dv.getFloat32(20, true)).toBe(4);
    expect(dv.byteLength).toBe(96);
    expect(dv.getFloat32(32, true)).toBe(1);
    expect(dv.getFloat32(52, true)).toBe(1);
    expect(dv.getFloat32(72, true)).toBe(1);
    expect(dv.getFloat32(92, true)).toBe(1);
  });

  it('"on" packs gtaoDownscale = 2 (UBO field is load-bearing in both modes)', () => {
    const captured: CapturedDispatch[] = [];
    const ubo: { bytes?: ArrayBuffer } = {};
    new GTAOPass({} as unknown as GPUComputePipeline).dispatch(makeDispatchCtx(256, 256, 2, captured, ubo));
    const dv = new DataView(ubo.bytes!);
    expect(dv.getFloat32(20, true)).toBe(2);
  });
});

// ── GTAO upsample — independent per-channel bilateral oracle ────────────────

describe('gtaoUpsample reference behavior — per-channel bilateral reconstruction', () => {
  const aoHalf: Vec3[][] = [
    [[0.2, 0.4, 0.8], [0.8, 0.4, 0.2]],
    [[0.1, 0.7, 0.3], [0.9, 0.2, 0.6]],
  ];

  function flatNormalDepth(width: number, height: number): NormalDepthSample[][] {
    return Array.from({ length: height }, () =>
      Array.from({ length: width }, () => ({ normal: [0, 0, 1] as Vec3, depth: 1 })),
    );
  }

  it('reduces to an unweighted per-channel 2x2 average when all taps match the surface', () => {
    const normalDepth = flatNormalDepth(4, 4);
    const ao = gtaoUpsampleReference({
      fullX: 0,
      fullY: 0,
      fullWidth: 4,
      fullHeight: 4,
      downscale: 2,
      normalDepth,
      aoHalf,
      bilateralDepthSigma: 1,
    });

    expect(ao[0]).toBeCloseTo((0.2 + 0.8 + 0.1 + 0.9) / 4, 12);
    expect(ao[1]).toBeCloseTo((0.4 + 0.4 + 0.7 + 0.2) / 4, 12);
    expect(ao[2]).toBeCloseTo((0.8 + 0.2 + 0.3 + 0.6) / 4, 12);
  });

  it('preserves a depth edge by weighting the matching low-res cell over distant neighbors', () => {
    const normalDepth = flatNormalDepth(4, 4);
    normalDepth[1]![3] = { normal: [0, 0, 1], depth: 50 };
    normalDepth[3]![1] = { normal: [0, 0, 1], depth: 50 };
    normalDepth[3]![3] = { normal: [0, 0, 1], depth: 50 };

    const ao = gtaoUpsampleReference({
      fullX: 1,
      fullY: 1,
      fullWidth: 4,
      fullHeight: 4,
      downscale: 2,
      normalDepth,
      aoHalf,
      bilateralDepthSigma: 0.01,
    });

    expect(ao[0]).toBeCloseTo(0.2, 12);
    expect(ao[1]).toBeCloseTo(0.4, 12);
    expect(ao[2]).toBeCloseTo(0.8, 12);
  });

  it('falls back to the per-channel 2x2 average when every bilateral weight is rejected', () => {
    const normalDepth = flatNormalDepth(4, 4);
    normalDepth[1]![1] = { normal: [1, 0, 0], depth: 1 };
    normalDepth[1]![3] = { normal: [1, 0, 0], depth: 1 };
    normalDepth[3]![1] = { normal: [1, 0, 0], depth: 1 };
    normalDepth[3]![3] = { normal: [1, 0, 0], depth: 1 };

    const ao = gtaoUpsampleReference({
      fullX: 0,
      fullY: 0,
      fullWidth: 4,
      fullHeight: 4,
      downscale: 2,
      normalDepth,
      aoHalf,
      bilateralDepthSigma: 1,
    });

    expect(ao[0]).toBeCloseTo((0.2 + 0.8 + 0.1 + 0.9) / 4, 12);
    expect(ao[1]).toBeCloseTo((0.4 + 0.4 + 0.7 + 0.2) / 4, 12);
    expect(ao[2]).toBeCloseTo((0.8 + 0.2 + 0.3 + 0.6) / 4, 12);
  });
});

// ── Shaders read the downscale (no hardcoded ÷2) ─────────────────────────────

describe('GTAO shaders — downscale-driven (no hardcoded /2u)', () => {
  it('gtao.wgsl declares gtaoDownscale in the shared UBO struct and uses it to size the AO grid', () => {
    // GTAOUniforms struct lives in gtaoCommon (required by gtao); check the struct there.
    expect(GTAO_COMMON_WGSL).toContain('gtaoDownscale: f32');
    expect(GTAO_WGSL).toContain('u32(gtao_ubo.gtaoDownscale)');
    expect(GTAO_WGSL).toContain('fullDims / ds');
    // The former hardcoded half-res mapping must be gone.
    expect(GTAO_WGSL).not.toContain('fullDims / 2u');
    expect(GTAO_WGSL).not.toContain('gid.xy * 2u + 1u');
  });

  it('gtaoUpsample.wgsl uses gtaoDownscale from the shared UBO struct and maps taps with it', () => {
    // GTAOUniforms struct lives in gtaoCommon (required by gtaoUpsample); check the struct there.
    expect(GTAO_COMMON_WGSL).toContain('gtaoDownscale: f32');
    expect(GTAO_UPSAMPLE_WGSL).toContain('u32(up_gtao.gtaoDownscale)');
    expect(GTAO_UPSAMPLE_WGSL).toContain('gid.xy / ds');
    expect(GTAO_UPSAMPLE_WGSL).toContain('sampleHalf * ds + ds / 2u');
    expect(GTAO_UPSAMPLE_WGSL).not.toContain('fullDims / 2u');
    expect(GTAO_UPSAMPLE_WGSL).not.toContain('gid.xy / 2u');
    expect(GTAO_UPSAMPLE_WGSL).not.toContain('sampleHalf * 2u + 1u');
  });
});

// ── End-to-end: pipeline init resolves gtaoMode → AO target resolution ───────
//
// WalkaroundGPUPipeline.initialize() needs a full GPU device + shader compile;
// that's GPU-only. Instead we pin the resolved-downscale mapping that
// initialize() applies (gtaoMode → downscale → createFrameResources) at the
// resource-factory boundary, which is the load-bearing seam this fix adds.

describe('gtaoMode → AO target resolution mapping (the fix)', () => {
  const cases: ReadonlyArray<[mode: string, downscale: number, expectAxis: number]> = [
    ['on', 2, 128],       // half-res
    ['quarter', 4, 64],   // quarter-res — distinct from 'on'
  ];
  for (const [mode, downscale, expectAxis] of cases) {
    it(`gtaoMode '${mode}' → downscale ${downscale} → AO target ${expectAxis}px (W=256)`, () => {
      const dev = makeSizingDevice();
      const res = createFrameResources(dev, 256, 256, { gtaoDownscale: downscale });
      const tex = res.gtao.aoHalfTexture as unknown as CapturedTexture;
      expect(tex.width).toBe(expectAxis);
      expect(tex.height).toBe(expectAxis);
    });
  }

  it("quarter's AO target is distinct from on's (silent no-op regression guard)", () => {
    const dev = makeSizingDevice();
    const onTex = createFrameResources(dev, 256, 256, { gtaoDownscale: 2 }).gtao.aoHalfTexture as unknown as CapturedTexture;
    const qTex = createFrameResources(dev, 256, 256, { gtaoDownscale: 4 }).gtao.aoHalfTexture as unknown as CapturedTexture;
    expect(qTex.width).not.toBe(onTex.width);
    expect(qTex.width).toBe(onTex.width / 2);
  });
});
