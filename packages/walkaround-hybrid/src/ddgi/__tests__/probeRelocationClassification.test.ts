import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';
import {
  IRR_PROBE_STATE_LOCAL_X,
  IRR_PROBE_STATE_LOCAL_Y,
  IRR_STRIDE,
} from '../ddgiAtlasLayout.js';
import { DDGI_SAMPLE_WGSL, DDGI_GRID_UBO_WGSL } from '../ddgiSampleWgsl.js';
import { ProbeGrid } from '../probeGrid.js';
import {
  buildInitialProbeStateData,
  clampProbeRelocationOffset,
  classifyAndRelocateProbe,
  DDGI_EXPLICIT_PROBE_STATE_BYTES,
  DDGI_PACKED_PROBE_STATE_BYTES,
  DDGI_PROBE_MAX_OFFSET_NORMALIZED,
  DDGI_PROBE_STATE_F32_RADIUS_TOLERANCE_NORMALIZED,
  DDGI_PROBE_STATE_MAX_QUANTIZATION_ERROR_NORMALIZED,
  isValidProbeStateData,
  packedProbeStateElementOffset,
  readPackedProbeStateFromIrradianceAtlas,
  writePackedProbeStateToIrradianceAtlas,
  type PackedProbeStateAtlasLayout,
} from '../probeState.js';
import {
  copyProbeIrradianceAndPackedStateForward,
  dispatchProbeClassifyRelocatePass,
} from '../probeUpdateDispatcher.js';
import { submitProbeUpdateCommand } from '../probeUpdatePass.js';
import type { ProbeUpdateGpuState } from '../probeUpdateGpuState.js';
import {
  makeProbeUpdateBlendIrrWGSL,
  makeProbeUpdateBlendVisWGSL,
} from '../wgsl/probeUpdateBlend.wgsl.js';
import { PROBE_CLASSIFY_RELOCATE_WGSL } from '../wgsl/probeClassifyRelocate.wgsl.js';
import { makeProbeUpdateRaysWGSL } from '../wgsl/probeUpdateRays.wgsl.js';

installWebGPUPolyfills();

function packedLayout(
  dimsX = 2,
  dimsY = 2,
  dimsZ = 1,
  spacing = 8,
): PackedProbeStateAtlasLayout {
  return {
    dimsX,
    dimsY,
    dimsZ,
    irradianceWidth: dimsX * IRR_STRIDE,
    irradianceHeight: dimsY * dimsZ * IRR_STRIDE,
    spacing,
  };
}

describe('DDGI probe relocation/classification CPU oracle and ABI', () => {
  it('keeps the v6 snapshot ABI explicit while packing one rgba16float atlas texel per probe', () => {
    expect(DDGI_EXPLICIT_PROBE_STATE_BYTES).toBe(16);
    expect(DDGI_PACKED_PROBE_STATE_BYTES).toBe(8);
    const inactive = buildInitialProbeStateData(2);
    expect(inactive.byteLength).toBe(2 * DDGI_EXPLICIT_PROBE_STATE_BYTES);
    expect(Array.from(inactive)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);

    const active = buildInitialProbeStateData(2, true);
    expect(Array.from(active)).toEqual([0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('reserves local (4,4), outside every SH coefficient texel and without an irradiance border writer', () => {
    expect([IRR_PROBE_STATE_LOCAL_X, IRR_PROBE_STATE_LOCAL_Y]).toEqual([4, 4]);
    const layout = packedLayout();
    for (let probe = 0; probe < 4; probe += 1) {
      const element = packedProbeStateElementOffset(probe, layout);
      const texel = element / 4;
      const x = texel % layout.irradianceWidth;
      const y = Math.floor(texel / layout.irradianceWidth);
      expect(x % IRR_STRIDE).toBe(4);
      expect(y % IRR_STRIDE).toBe(4);
      for (let shY = 1; shY <= 3; shY += 1) {
        for (let shX = 1; shX <= 3; shX += 1) {
          expect([x % IRR_STRIDE, y % IRR_STRIDE]).not.toEqual([shX, shY]);
        }
      }
    }
    const passSource = readFileSync(new URL('../probeUpdatePass.ts', import.meta.url), 'utf8');
    expect(passSource).not.toContain('dispatchProbeUpdateBorderIrrPass');
  });

  it('round-trips normalized f16 offsets within the declared bound, including subnormal and multi-axis boundary cases', () => {
    const layout = packedLayout(2, 1, 1, 7.5);
    const nearZeroNormalized = 1.6740297621090525e-7;
    // Derived from the exact norm-0.45 vector whose independently rounded f16
    // components have radius 0.450204514. The 1e-8 inward guard keeps the
    // Float32 snapshot itself strictly valid; decode must still clamp f16.
    const boundary = [
      0.2603768642052341 * 0.99999999,
      0.2672182895282315 * 0.99999999,
      -0.25159148302010415 * 0.99999999,
    ] as const;
    const explicit = new Float32Array([
      nearZeroNormalized * layout.spacing,
      -nearZeroNormalized * layout.spacing,
      0,
      1,
      boundary[0] * layout.spacing,
      boundary[1] * layout.spacing,
      boundary[2] * layout.spacing,
      1,
    ]);
    const atlas = new Uint16Array(
      layout.irradianceWidth * layout.irradianceHeight * 4,
    );
    writePackedProbeStateToIrradianceAtlas(atlas, explicit, layout);
    const decoded = readPackedProbeStateFromIrradianceAtlas(atlas, layout);

    expect(decoded[0]).toBeGreaterThan(0);
    expect(decoded[0]).toBeLessThan(layout.spacing * 1e-5);
    for (let lane = 0; lane < explicit.length; lane += 1) {
      if (lane % 4 === 3) {
        expect(decoded[lane]).toBe(explicit[lane]);
      } else {
        expect(Math.abs(decoded[lane]! - explicit[lane]!)).toBeLessThanOrEqual(
          layout.spacing *
            DDGI_PROBE_STATE_MAX_QUANTIZATION_ERROR_NORMALIZED,
        );
      }
    }
    expect(Math.hypot(decoded[4]!, decoded[5]!, decoded[6]!)).toBeLessThanOrEqual(
      layout.spacing * DDGI_PROBE_MAX_OFFSET_NORMALIZED,
    );
    expect(isValidProbeStateData(decoded, layout.spacing)).toBe(true);
  });

  it('keeps thousands of f16-quantized boundary directions strictly inside after Float32 publication', () => {
    const side = 64;
    const count = side * side;
    const layout = packedLayout(side, side, 1, 3.25);
    const explicit = new Float32Array(count * 4);
    for (let index = 0; index < count; index += 1) {
      const y = 1 - (2 * (index + 0.5)) / count;
      const radius = Math.sqrt(Math.max(0, 1 - y * y));
      const phi = index * Math.PI * (3 - Math.sqrt(5));
      const base = index * 4;
      const scale =
        layout.spacing * DDGI_PROBE_MAX_OFFSET_NORMALIZED * 0.9999999;
      explicit[base] = Math.cos(phi) * radius * scale;
      explicit[base + 1] = y * scale;
      explicit[base + 2] = Math.sin(phi) * radius * scale;
      explicit[base + 3] = 1;
    }
    const atlas = new Uint16Array(
      layout.irradianceWidth * layout.irradianceHeight * 4,
    );
    writePackedProbeStateToIrradianceAtlas(atlas, explicit, layout);
    const decoded = readPackedProbeStateFromIrradianceAtlas(atlas, layout);
    const maxRadius =
      layout.spacing * DDGI_PROBE_MAX_OFFSET_NORMALIZED;
    for (let index = 0; index < count; index += 1) {
      const base = index * 4;
      expect(
        Math.hypot(
          decoded[base]!,
          decoded[base + 1]!,
          decoded[base + 2]!,
        ),
      ).toBeLessThanOrEqual(maxRadius);
    }
  });

  it('rejects unsafe probe-count, atlas-element, and byte-length products before indexing', () => {
    expect(() =>
      buildInitialProbeStateData(Number.MAX_SAFE_INTEGER),
    ).toThrow('non-negative safe integer');
    expect(() =>
      packedProbeStateElementOffset(0, {
        dimsX: Number.MAX_SAFE_INTEGER,
        dimsY: 1,
        dimsZ: 1,
        irradianceWidth: Number.MAX_SAFE_INTEGER,
        irradianceHeight: IRR_STRIDE,
        spacing: 1,
      }),
    ).toThrow('layout is invalid');
    expect(() =>
      packedProbeStateElementOffset(0, {
        dimsX: 100_000_000,
        dimsY: 100_000_000,
        dimsZ: 1,
        irradianceWidth: 500_000_000,
        irradianceHeight: 500_000_000,
        spacing: 1,
      }),
    ).toThrow('layout is invalid');
  });

  it('hard-clamps malformed/oversized offsets to the 0.45-cell sphere', () => {
    const clamped = clampProbeRelocationOffset([8, 0, 0], 2);
    expect(clamped).toEqual([0.9, 0, 0]);
    expect(Math.hypot(...clamped)).toBeCloseTo(
      2 * DDGI_PROBE_MAX_OFFSET_NORMALIZED,
      7,
    );
    expect(clampProbeRelocationOffset([Number.NaN, 0, 0], 2)).toEqual([0, 0, 0]);
  });

  it('rejects non-finite, out-of-cell, and non-binary persisted state lanes', () => {
    expect(isValidProbeStateData(new Float32Array([0.1, 0, 0, 1]), 1)).toBe(true);
    expect(isValidProbeStateData(new Float32Array([
      -0.19071006774902344,
      -0.3849506974220276,
      0.1339501440525055,
      1,
    ]), 1)).toBe(true);
    expect(isValidProbeStateData(new Float32Array([
      DDGI_PROBE_MAX_OFFSET_NORMALIZED +
        DDGI_PROBE_STATE_F32_RADIUS_TOLERANCE_NORMALIZED * 2,
      0,
      0,
      1,
    ]), 1)).toBe(false);
    expect(isValidProbeStateData(new Float32Array([Number.NaN, 0, 0, 1]), 1)).toBe(false);
    expect(isValidProbeStateData(new Float32Array([0.5, 0, 0, 1]), 1)).toBe(false);
    expect(isValidProbeStateData(new Float32Array([0, 0, 0, 0.5]), 1)).toBe(false);
  });

  it('classifies a backface-dominated probe inactive while moving it by a bounded step', () => {
    const result = classifyAndRelocateProbe(
      [0, 0, 0],
      [
        { direction: [1, 0, 0], hitDistance: -0.4 },
        { direction: [0, 1, 0], hitDistance: -0.6 },
        { direction: [-1, 0, 0], hitDistance: 0.4 },
        { direction: [0, -1, 0], hitDistance: 1.0e20 },
      ],
      1,
    );
    expect(result.active).toBe(false);
    expect(result.backfaceCount).toBe(2);
    expect(result.offset[0]).toBeGreaterThan(0);
    expect(Math.hypot(...result.offset)).toBeLessThanOrEqual(0.45);
  });

  it('activates a probe with front-face geometry inside its voxel and restores offsets toward zero', () => {
    const result = classifyAndRelocateProbe(
      [0.2, 0, 0],
      [
        { direction: [1, 0, 0], hitDistance: 0.5 },
        { direction: [0, 1, 0], hitDistance: 1.0e20 },
      ],
      1,
    );
    expect(result.active).toBe(true);
    expect(result.validRayCount).toBe(1);
    expect(result.offset[0]).toBeCloseTo(0, 7);
  });

  it('recognizes a sub-clearance front hit and relocates toward the opposite open ray', () => {
    const result = classifyAndRelocateProbe(
      [0, 0, 0],
      [
        { direction: [1, 0, 0], hitDistance: 0.001 },
        { direction: [-1, 0, 0], hitDistance: 1.0e20 },
      ],
      1,
    );
    expect(result.active).toBe(true);
    expect(result.validRayCount).toBe(0);
    expect(result.offset[0]).toBeCloseTo(-0.2, 7);
    expect(Math.hypot(...result.offset)).toBeLessThanOrEqual(0.45);
  });

  it('keeps an all-miss probe inactive instead of publishing open visibility', () => {
    const result = classifyAndRelocateProbe(
      [0, 0, 0],
      [
        { direction: [1, 0, 0], hitDistance: 1.0e20 },
        { direction: [-1, 0, 0], hitDistance: 1.0e20 },
      ],
      1,
    );
    expect(result.active).toBe(false);
    expect(result.validRayCount).toBe(0);
  });
});

describe('DDGI packed probe-state shader contract', () => {
  it('writes normalized state into the next rgba16float irradiance atlas and publishes active state through ray padding', () => {
    expect(PROBE_CLASSIFY_RELOCATE_WGSL).toContain('probeClassifyRelocate');
    expect(PROBE_CLASSIFY_RELOCATE_WGSL).toContain('MAX_OFFSET_NORMALIZED');
    expect(PROBE_CLASSIFY_RELOCATE_WGSL).toContain('irradiancePrev');
    expect(PROBE_CLASSIFY_RELOCATE_WGSL).toContain('irradianceOut');
    expect(PROBE_CLASSIFY_RELOCATE_WGSL).toContain(
      'texture_storage_2d<rgba16float, write>',
    );
    expect(PROBE_CLASSIFY_RELOCATE_WGSL).toContain(
      'const MISS_DISTANCE: f32 = 1.0e19;',
    );
    expect(PROBE_CLASSIFY_RELOCATE_WGSL).toContain(
      'rayResults[baseIdx]._pad0 = activeLane',
    );
    expect(PROBE_CLASSIFY_RELOCATE_WGSL).not.toMatch(/\blet active\b/);
  });

  it('loads bounded packed relocation from the irradiance atlas without suppressing inactive recovery rays', () => {
    const wgsl = makeProbeUpdateRaysWGSL(8);
    expect(wgsl).not.toContain('@group(2) @binding(8)');
    expect(wgsl).toContain('textureLoad(irradiancePrev');
    expect(wgsl).toContain('normalizedOffset * gridParams.spacing');
    expect(wgsl).toContain('out._pad0 = 0.0');
    expect(wgsl).not.toMatch(/if\s*\(\s*state\.w\s*<\s*0\.5\s*\)\s*\{\s*return/);
  });

  it('normalizes irradiance over valid rays and writes conservative zero for inactive/all-invalid visibility', () => {
    const irr = makeProbeUpdateBlendIrrWGSL();
    const vis = makeProbeUpdateBlendVisWGSL();
    expect(irr).toContain('validRayCount = validRayCount + 1u');
    expect(irr).toContain('f32(max(validRayCount, 1u))');
    expect(irr).not.toContain('/ f32(RAYS_PER_PROBE)');
    expect(irr).toContain('rayResults[baseIdx]._pad0 < 0.5');
    expect(vis).toContain('rayResults[baseIdx]._pad0 < 0.5');
    expect(vis).toMatch(/validRayCount > 0u[\s\S]*65504\.0/);
    expect(vis).toMatch(/Every ray was invalid[\s\S]*newDepth = 0\.0/);
  });

  it('receiver loads packed state from irradiance, excludes inactive probes, clamps offsets, and renormalizes', () => {
    expect(DDGI_GRID_UBO_WGSL).not.toContain('@group(3) @binding(10)');
    expect(DDGI_GRID_UBO_WGSL).not.toContain('ddgiProbeState');
    expect(DDGI_SAMPLE_WGSL).toContain('textureLoad(irradianceAtlas, stateCoord, 0)');
    expect(DDGI_SAMPLE_WGSL).toContain('if (state.w < 0.5) { continue; }');
    expect(DDGI_SAMPLE_WGSL).toContain('normalizedOffset * gridSpacing');
    expect(DDGI_SAMPLE_WGSL).toContain('offsetLength2 > maxOffset * maxOffset');
    expect(DDGI_SAMPLE_WGSL).toContain('probeToSurfaceLen2 > 1.0e-12');
    expect(DDGI_SAMPLE_WGSL).not.toContain('normalize(biasedPos - probeWorld)');
    expect(DDGI_SAMPLE_WGSL).toContain('return sum / totalWeight');
    expect(DDGI_SAMPLE_WGSL).toContain('fallbackIrr * fallbackVisibility');
  });
});

describe('DDGI irradiance/state ping-pong publication', () => {
  function texture(label: string, width: number, height: number): GPUTexture {
    return { label, width, height } as unknown as GPUTexture;
  }

  it('carries the complete irradiance/state atlas forward for consecutive strata and swaps once', () => {
    const grid = new ProbeGrid();
    grid.computeFromBounds({ min: [0, 0, 0], max: [2, 2, 2] }, 1, 4);
    grid.allocateAtlases();
    const irradianceA = texture(
      'irradiance-a',
      grid.irradianceReadTex.width,
      grid.irradianceReadTex.height,
    );
    const irradianceB = texture(
      'irradiance-b',
      grid.irradianceWriteTex.width,
      grid.irradianceWriteTex.height,
    );
    const encoder = {
      copyTextureToTexture: vi.fn(),
      finish: vi.fn(() => ({}) as GPUCommandBuffer),
    };
    const queue = { submit: vi.fn() };

    copyProbeIrradianceAndPackedStateForward(
      encoder,
      irradianceA,
      irradianceB,
    );
    const firstIrr = grid.irradianceReadTex;
    submitProbeUpdateCommand(encoder, queue, () => grid.swap());
    expect(grid.irradianceReadTex).not.toBe(firstIrr);

    copyProbeIrradianceAndPackedStateForward(
      encoder,
      irradianceB,
      irradianceA,
    );
    submitProbeUpdateCommand(encoder, queue, () => grid.swap());
    expect(grid.irradianceReadTex).toBe(firstIrr);
    expect(encoder.copyTextureToTexture).toHaveBeenNthCalledWith(
      1,
      { texture: irradianceA },
      { texture: irradianceB },
      {
        width: irradianceA.width,
        height: irradianceA.height,
        depthOrArrayLayers: 1,
      },
    );
    expect(encoder.copyTextureToTexture).toHaveBeenNthCalledWith(
      2,
      { texture: irradianceB },
      { texture: irradianceA },
      {
        width: irradianceB.width,
        height: irradianceB.height,
        depthOrArrayLayers: 1,
      },
    );
  });

  it('pins copy -> ray producer -> classifier -> SH/visibility blend ordering', () => {
    const source = readFileSync(new URL('../probeUpdatePass.ts', import.meta.url), 'utf8');
    const sequence = source.slice(source.indexOf('// Run compute passes.'));
    const copy = sequence.indexOf('copyProbeIrradianceAndPackedStateForward(');
    const rays = sequence.indexOf('dispatchProbeUpdateRaysPass(');
    const classify = sequence.indexOf('dispatchProbeClassifyRelocatePass(');
    const blendIrr = sequence.indexOf('dispatchProbeUpdateBlendIrrPass(');
    const blendVis = sequence.indexOf('dispatchProbeUpdateBlendVisPass(');
    expect(copy).toBeGreaterThanOrEqual(0);
    expect(copy).toBeLessThan(rays);
    expect(rays).toBeLessThan(classify);
    expect(classify).toBeLessThan(blendIrr);
    expect(blendIrr).toBeLessThan(blendVis);
  });

  it('binds the exact active prefix when rounded classifier dispatch follows a larger stratum', () => {
    const bindGroupEntries: GPUBindGroupEntry[][] = [];
    const device = {
      createBindGroup: vi.fn((descriptor: GPUBindGroupDescriptor) => {
        bindGroupEntries.push([...descriptor.entries]);
        return {} as GPUBindGroup;
      }),
    } as unknown as GPUDevice;
    const classifyRelocatePipeline = {
      getBindGroupLayout: vi.fn(() => ({} as GPUBindGroupLayout)),
    } as unknown as GPUComputePipeline;
    const gpu = {
      device,
      classifyRelocatePipeline,
      rayResultsBuf: { size: 65 * 192 * 32 } as GPUBuffer,
      activeProbesBuf: { size: 65 * 4 } as GPUBuffer,
      gridParamsBuf: { size: 64 } as GPUBuffer,
      bgCache: null,
    } as unknown as ProbeUpdateGpuState;
    const dispatchWorkgroups = vi.fn();
    const encoder = {
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(),
        setBindGroup: vi.fn(),
        dispatchWorkgroups,
        end: vi.fn(),
      })),
    } as unknown as GPUCommandEncoder;
    const irrRead = {
      createView: vi.fn(() => ({} as GPUTextureView)),
    } as unknown as GPUTexture;
    const irrWrite = {
      createView: vi.fn(() => ({} as GPUTextureView)),
    } as unknown as GPUTexture;

    dispatchProbeClassifyRelocatePass(
      encoder,
      gpu,
      65,
      irrRead,
      irrWrite,
    );
    dispatchProbeClassifyRelocatePass(
      encoder,
      gpu,
      63,
      irrRead,
      irrWrite,
    );

    expect(device.createBindGroup).toHaveBeenCalledTimes(2);
    const activeResources = bindGroupEntries.map((entries) =>
      entries.find(({ binding }) => binding === 1)?.resource as
        | GPUBufferBinding
        | undefined,
    );
    expect(activeResources.map((resource) => resource?.size)).toEqual([
      65 * 4,
      63 * 4,
    ]);
    expect(dispatchWorkgroups).toHaveBeenNthCalledWith(1, 2, 1, 1);
    expect(dispatchWorkgroups).toHaveBeenNthCalledWith(2, 1, 1, 1);
  });

  it('does not publish the atlas swap when queue submission fails', () => {
    const grid = new ProbeGrid();
    grid.allocateAtlases();
    const before = {
      irr: grid.irradianceReadTex,
      vis: grid.visibilityReadTex,
    };
    const encoder = { finish: vi.fn(() => ({}) as GPUCommandBuffer) };
    const queue = {
      submit: vi.fn(() => {
        throw new Error('submit failed');
      }),
    };

    expect(() =>
      submitProbeUpdateCommand(encoder, queue, () => grid.swap()),
    ).toThrow('submit failed');
    expect(grid.irradianceReadTex).toBe(before.irr);
    expect(grid.visibilityReadTex).toBe(before.vis);
  });
});
