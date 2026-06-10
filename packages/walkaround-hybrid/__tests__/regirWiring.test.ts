/**
 * regirWiring.test.ts — verifies ReGIR (Boksansky 2021) is wired into ReSTIR-DI
 * light selection correctly + UNBIASEDLY, and that ReGIR-OFF falls back to the
 * light-tree path BIT-IDENTICALLY.
 *
 * Concerns:
 *   1. UBO contract: WalkaroundUBO declares the ReGIR fields at the documented
 *      byte offsets; `updateUBO` packs them and a ReGIR-OFF state writes zeros
 *      (so the kernel's `regirEnabled == 0` gate keeps RIS on the tree path).
 *   2. RIS WGSL branches on `ubo.regirEnabled`, draws from the cell reservoir,
 *      and feeds the cell pmf into the SAME unbiased weight divisor the
 *      light-tree path uses (`emitterSelPmf * ls.pdfArea`).
 *   3. ReGIR-OFF bit-identity: with `regirEnabled` left 0, the UBO bytes 0..364
 *      (everything the light-tree path reads) are byte-identical whether or not
 *      a ReGIR-OFF state is passed — so RIS reproduces the light-tree path
 *      exactly.
 *   4. ReGIRCoordinator: grid geometry derivation, grid-region byte sizing
 *      (0 when off), the live gate (off unless tree live + pipeline ready), and
 *      the UBO-state gate.
 */

import { describe, expect, it } from 'vitest';
import { REGIR_FLOATS_PER_SURVIVOR } from '@vitrum/shared-samplers';
import { RIS_WGSL } from '../src/shaders/ris.wgsl.js';
import { REGIR_WGSL, REGIR_BUILD_WGSL } from '../src/shaders/regir.wgsl.js';
import { WALKAROUND_UBO_WGSL } from '../src/shaders/walkaroundUbo.wgsl.js';
import { updateUBO, type RegirUboState } from '../src/pipeline/uboUpdater.js';
import {
  ReGIRCoordinator,
  resolveReGIRConfig,
} from '../src/pipeline/ReGIRCoordinator.js';
import type { SceneBVHBuffers } from '../src/restir/bvhTypes.js';
import type { PipelineFrameInputs } from '../src/pipeline/WalkaroundGPUPipeline.js';

// ── Minimal frame inputs (only the fields updateUBO reads) ────────────────────
function fakeInputs(): PipelineFrameInputs {
  const m = new Float32Array(16);
  return {
    camera: { viewMatrix: m, projMatrix: m, prevViewProjMatrix: m, cameraPos: [0, 0, 0] },
    screen: { screenWidth: 64, screenHeight: 64, frameSeed: 7, swapChainView: {} as GPUTextureView, swapChainFormat: 'bgra8unorm' },
    lighting: {
      totalEmissivePower: 1, emitterCount: 4,
      primaryLightDir: [0, 1, 0], primaryLightIntensity: 1,
      skyTint: [0, 0, 0], skyIrradiance: 0,
      emitterDist2Floor: 0.01, directFireflyClamp: 4, causticBoost: 1, causticVisClamp: 1,
      lightTreeEnabled: 1, lightTreeNodeCount: 7,
    },
    restirDI: { temporalMClampDI: 20, spatialReuseRadiusPx: 30, spatialDepthTolFloor: 0.05 },
    restirGI: {
      restirGiWCap: 16, restirGiIrrClamp: 5, restirGiMClamp: 50,
      restirGiSpatialRadiusPx: 12, restirGiSpatialNormalDotMin: 0.9,
      restirGiSpatialCoplanarTol: 0.05,
    },
    gtao: { gtaoRadiusPx: 32, gtaoIntensity: 2, gtaoDepthThreshold: 2, gtaoBilateralDepthSigma: 0.25, adaptiveSamplingThresholdLow: 0.01, adaptiveSamplingThresholdHigh: 0.1 },
    filter: {
      triIntersectEpsilon: 1e-5, glassMixScale: 0.7, indirectFireflyClamp: [1, 1, 1],
      atrousDirectSigmas: [128, 5, 0.05], atrousIndirectSigmas: [32, 20, 0.5],
      stainedGlassFlags: 0,
    },
    bvh: { bvhMode: 0, tlasNodeCount: 0 },
    nrc: {},
  } as PipelineFrameInputs;
}

/** Capture-only device: writeBuffer copies bytes into a backing buffer we read. */
function capturingDevice(backing: Uint8Array): GPUDevice {
  return {
    queue: {
      writeBuffer: (_buf: GPUBuffer, offset: number, data: ArrayBuffer) => {
        backing.set(new Uint8Array(data), offset);
      },
    },
  } as unknown as GPUDevice;
}

describe('WalkaroundUBO — ReGIR field contract', () => {
  it('declares the ReGIR fields with the documented offsets', () => {
    for (const sub of [
      'regirOrigin:', 'regirInvCellSize:', 'regirDims:', 'regirEnabled:',
      'regirCandidatesPerCell:', 'regirSurvivorsPerCell:', 'regirGridFloatOffset:',
    ]) {
      expect(WALKAROUND_UBO_WGSL).toContain(sub);
    }
    // The struct grew to 416 bytes (documented end pad).
    expect(WALKAROUND_UBO_WGSL).toContain('struct size 416 bytes');
  });
});

describe('updateUBO — ReGIR packing + offsets', () => {
  it('packs an enabled ReGIR state at the documented float/u32 indices', () => {
    const backing = new Uint8Array(416);
    const dev = capturingDevice(backing);
    const regir: RegirUboState = {
      enabled: true,
      origin: [1.5, -2.5, 3.5],
      invCellSize: 0.25,
      dims: [16, 8, 12],
      candidatesPerCell: 32,
      survivorsPerCell: 8,
      gridFloatOffset: 84, // arbitrary passthrough value (UBO field round-trip test)
    };
    updateUBO(dev, {} as GPUBuffer, fakeInputs(), { enabled: false, mixAlpha: 0 }, regir);
    const f32 = new Float32Array(backing.buffer);
    const u32 = new Uint32Array(backing.buffer);
    // offset 368 = float index 92 (regirOrigin.xyz)
    expect(f32[92]).toBeCloseTo(1.5);
    expect(f32[93]).toBeCloseTo(-2.5);
    expect(f32[94]).toBeCloseTo(3.5);
    expect(f32[95]).toBeCloseTo(0.25); // offset 380 — regirInvCellSize
    expect(u32[96]).toBe(16);          // offset 384 — regirDims.x
    expect(u32[97]).toBe(8);
    expect(u32[98]).toBe(12);
    expect(u32[99]).toBe(1);           // offset 396 — regirEnabled
    expect(u32[100]).toBe(32);         // offset 400 — M
    expect(u32[101]).toBe(8);          // offset 404 — K
    expect(u32[102]).toBe(84);         // offset 408 — gridFloatOffset
  });

  it('ReGIR-OFF default writes a zeroed gate (regirEnabled = 0)', () => {
    const backing = new Uint8Array(416);
    updateUBO(capturingDevice(backing), {} as GPUBuffer, fakeInputs(),
      { enabled: false, mixAlpha: 0 }); // regir defaults to OFF
    const u32 = new Uint32Array(backing.buffer);
    expect(u32[99]).toBe(0); // regirEnabled gate off
    expect(u32[100]).toBe(0);
    expect(u32[101]).toBe(0);
    expect(u32[102]).toBe(0);
  });

  it('ReGIR-OFF bit-identity: bytes 0..364 (the light-tree path inputs) are identical with vs without a ReGIR-OFF state', () => {
    const a = new Uint8Array(416);
    const b = new Uint8Array(416);
    updateUBO(capturingDevice(a), {} as GPUBuffer, fakeInputs(), { enabled: false, mixAlpha: 0 });
    updateUBO(capturingDevice(b), {} as GPUBuffer, fakeInputs(), { enabled: false, mixAlpha: 0 }, {
      enabled: false, origin: [9, 9, 9], invCellSize: 99, dims: [9, 9, 9],
      candidatesPerCell: 9, survivorsPerCell: 9, gridFloatOffset: 9,
    });
    // Everything the light-tree path reads (bytes 0..364 inclusive) must be
    // byte-identical: a disabled ReGIR state never perturbs the tree path.
    expect(Array.from(a.slice(0, 368))).toEqual(Array.from(b.slice(0, 368)));
    // ReGIR region is zeroed in BOTH (the disabled state collapses to OFF).
    expect(new Uint32Array(b.buffer)[99]).toBe(0);
  });
});

describe('RIS WGSL — ReGIR cell selection enters the unbiased weight', () => {
  it('the M_LIGHT loop branches on ubo.regirEnabled and samples the cell reservoir', () => {
    expect(RIS_WGSL).toContain('ubo.regirEnabled == 1u');
    expect(RIS_WGSL).toContain('regir_sample_cell(pos, &rng)');
    // The cell pmf (q̂_c/Ŝ) is captured into emitterSelPmf — the EXACT source
    // pmf the WRS weight divides by.
    expect(RIS_WGSL).toContain('emitterSelPmf = rs.pSel');
  });

  it('the cell pmf flows through the SAME unbiased divisor as the tree path', () => {
    // pX = emitterSelPmf × ls.pdfArea, then w = p̂ / pX. The regir branch sets
    // emitterSelPmf = the cell pmf, so the divisor is exact ⇒ unbiased.
    expect(RIS_WGSL).toContain('emitterSelPmf * ls.pdfArea');
    expect(RIS_WGSL).toContain('let w = select(0.0, pHat / pX, pHat > 0.0)');
  });

  it('the tree + flat-CDF fallback paths are preserved (regir branch is else-if)', () => {
    expect(RIS_WGSL).toContain('sampleLightTree(pos, ubo.emitterDist2Floor');
    expect(RIS_WGSL).toContain('sampleEmitterIdx(&emitterCdf, emCount');
  });
});

describe('ReGIR WGSL — grid build stores the exact per-cell pmf', () => {
  it('the read path returns the survivor emitter + its pSel', () => {
    expect(REGIR_WGSL).toContain('fn regir_sample_cell(');
    expect(REGIR_WGSL).toContain('out.pSel = pSel');
  });

  it('the build kernel computes pSel = q̂(e*) · M / wSum (the unbiased effective pmf)', () => {
    expect(REGIR_BUILD_WGSL).toContain('fn regirBuildMain(');
    expect(REGIR_BUILD_WGSL).toContain('let pSel = (chosenQHat * f32(M)) / wSum;');
    // The build pass gates on the same UBO flag.
    expect(REGIR_BUILD_WGSL).toContain('if (ubo.regirEnabled == 0u) { return; }');
    // WRS source weight = target / tree pdf (light-tree-seeded).
    expect(REGIR_BUILD_WGSL).toContain('let w = qHat / draw.pdf;');
  });
});

describe('ReGIRCoordinator', () => {
  function bvh(nodeCount: number, treeEnabled: boolean): SceneBVHBuffers {
    // 8 verts in a 4-unit cube, stride-4 (xyz + packed-uv w).
    const positions = new Float32Array([
      0, 0, 0, 0,  4, 0, 0, 0,  0, 4, 0, 0,  4, 4, 0, 0,
      0, 0, 4, 0,  4, 0, 4, 0,  0, 4, 4, 0,  4, 4, 4, 0,
    ]);
    return {
      bvhPositions: { cpuData: positions.buffer as ArrayBuffer, byteLength: positions.byteLength, count: 8 },
      lightTreeNodeCount: nodeCount,
      lightTreeEnabled: treeEnabled,
    } as unknown as SceneBVHBuffers;
  }

  it('off by default → no grid bytes, not live, OFF UBO state', () => {
    const c = new ReGIRCoordinator(resolveReGIRConfig());
    expect(c.config.enabled).toBe(false);
    expect(c.gridRegionBytes()).toBe(0);
    c.initialize(bvh(7, true), true);
    expect(c.live).toBe(false);
    expect(c.uboState().enabled).toBe(false);
  });

  it('enabled + tree live + pipeline ready → live, grid geometry derived from BVH bounds', () => {
    const c = new ReGIRCoordinator(resolveReGIRConfig({
      enabled: true, cellsPerAxis: 8, candidatesPerCell: 16, survivorsPerCell: 4,
    }));
    // Grid byte count = 8³ cells × 4 survivors × 2 floats × 4 bytes.
    expect(c.gridRegionBytes()).toBe(8 ** 3 * 4 * REGIR_FLOATS_PER_SURVIVOR * 4);
    c.initialize(bvh(7, true), true);
    expect(c.live).toBe(true);
    const u = c.uboState();
    expect(u.enabled).toBe(true);
    // Origin = padded AABB min (slightly below 0 for the 4-unit cube).
    expect(u.origin[0]).toBeLessThanOrEqual(0);
    // Cell size = maxSpan / cellsPerAxis ≈ 4.x / 8 ⇒ invCellSize ≈ 8/4.x.
    expect(u.invCellSize).toBeGreaterThan(0);
    // Dims clamped to [1, cellsPerAxis]; a ~4-unit cube at cellSize ~0.5 ⇒ 8.
    expect(u.dims[0]).toBeGreaterThanOrEqual(1);
    expect(u.dims[0]).toBeLessThanOrEqual(8);
    expect(u.gridFloatOffset).toBe(7 * 16); // nodeCount × LIGHT_TREE_FLOATS_PER_NODE (B8: 12→16)
    expect(u.candidatesPerCell).toBe(16);
    expect(u.survivorsPerCell).toBe(4);
  });

  it('enabled but tree NOT live (< 2 emitters) → not live, falls back to light-tree path', () => {
    const c = new ReGIRCoordinator(resolveReGIRConfig({ enabled: true }));
    c.initialize(bvh(0, false), true);
    expect(c.live).toBe(false);
    expect(c.uboState().enabled).toBe(false);
  });

  it('enabled but grid-build pipeline NOT ready → not live', () => {
    const c = new ReGIRCoordinator(resolveReGIRConfig({ enabled: true }));
    c.initialize(bvh(7, true), false);
    expect(c.live).toBe(false);
  });

  it('refreshAfterEmitterRebuild moves the grid float offset and drops ReGIR if the tree goes degenerate', () => {
    const c = new ReGIRCoordinator(resolveReGIRConfig({ enabled: true }));
    c.initialize(bvh(7, true), true);
    expect(c.live).toBe(true);
    // Emitter rebuild grew the tree (more nodes) → offset shifts.
    c.refreshAfterEmitterRebuild({ lightTreeNodeCount: 15, lightTreeEnabled: true });
    expect(c.uboState().gridFloatOffset).toBe(15 * 16); // B8: stride 12→16
    // Tree went degenerate → ReGIR drops to the tree/flat path.
    c.refreshAfterEmitterRebuild({ lightTreeNodeCount: 0, lightTreeEnabled: false });
    expect(c.live).toBe(false);
  });
});
