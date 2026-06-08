/**
 * checkerboardUbo.test.ts — checkerboard half-res-shading gate wiring + OFF
 * bit-identity.
 *
 * Checkerboard rendering ships OFF by default and inert (the standard
 * "ship off-default, validate, then enable" pattern shared by
 * RC/PPG/ReGIR/NRC/restirPtReuse). This test pins:
 *
 *   1. UBO contract — WalkaroundUBO repurposes the two trailing pad slots
 *      (`_padPreVec3` → `frameParity` @ offset 316, `_padEnd` → `checkerboardOn`
 *      @ offset 332) WITHOUT growing the struct (still 416 bytes).
 *   2. `updateUBO` packs `frameParity` at u32[79] and `checkerboardOn` at
 *      u32[83], both defaulting to 0 (OFF) when the checkerboard arg is omitted.
 *   3. OFF-is-BIT-IDENTICAL — with checkerboard absent / disabled, EVERY UBO
 *      byte is identical to the pre-checkerboard packing (both repurposed pad
 *      slots stay 0), so the shade kernel shades every pixel byte-for-byte and
 *      the ResolvePass passes through.
 *   4. Turning the gate ON flips ONLY the two slots (checkerboardOn=1 and
 *      frameParity=frameCount&1); all other bytes are unchanged.
 *   5. Shade + resolve consume the SAME parity source — when ON, shade.wgsl's
 *      dispatch is COMPACTED to ~half the threads (one per active-parity pixel)
 *      and it decodes the compacted global-invocation id back to the true pixel
 *      `px = gid.x*2 + ((gid.y + frameParity) & 1u)`, which lands EXACTLY on the
 *      `(px+py)&1u == frameParity` set, while resolve.wgsl's `isShadedPixel`
 *      copies through that same `(px+py)&1u == frameParity` set and reprojects
 *      its complement — so the pixels shade writes are exactly the pixels
 *      resolve copies through (and the ones shade skips are exactly the ones
 *      resolve gap-fills). The OLD mechanism (full-res dispatch + per-thread gap
 *      early-out) shaded the identical set but wasted the gap threads; the
 *      compaction is a pure GPU-time optimisation with the same output.
 */

import { describe, expect, it } from 'vitest';
import { updateUBO } from '../src/pipeline/uboUpdater.js';
import { WALKAROUND_UBO_WGSL } from '../src/shaders/walkaroundUbo.wgsl.js';
import { SHADE_WGSL } from '../src/shaders/shade.wgsl.js';
import { SPATIAL_WGSL } from '../src/shaders/spatial.wgsl.js';
import { RIS_WGSL } from '../src/shaders/ris.wgsl.js';
import { RESOLVE_WGSL } from '../src/shaders/resolve.wgsl.js';
import type { PipelineFrameInputs } from '../src/pipeline/WalkaroundGPUPipeline.js';

// Minimal frame inputs (only the fields updateUBO reads).
function fakeInputs(): PipelineFrameInputs {
  const m = new Float32Array(16);
  return {
    camera: { viewMatrix: m, projMatrix: m, prevViewMatrix: m, cameraPos: [0, 0, 0] },
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

// updateUBO arg order: (device, buffer, inputs, ppg?, regir?, checkerboard?).
const PPG_OFF = { enabled: false, mixAlpha: 0 } as const;
const REGIR_ARG = undefined; // falls through to the REGIR_OFF default.

describe('WalkaroundUBO — checkerboard gate contract', () => {
  it('repurposes the trailing pad slots into frameParity/checkerboardOn (struct stays 416 bytes)', () => {
    expect(WALKAROUND_UBO_WGSL).toContain('frameParity:                u32,');
    expect(WALKAROUND_UBO_WGSL).toContain('offset 316 - checkerboard frame phase');
    expect(WALKAROUND_UBO_WGSL).toContain('checkerboardOn:             u32,');
    expect(WALKAROUND_UBO_WGSL).toContain('offset 332 - checkerboard sparse-shade gate');
    // No pad name survives.
    expect(WALKAROUND_UBO_WGSL).not.toContain('_padPreVec3:');
    expect(WALKAROUND_UBO_WGSL).not.toContain('_padEnd:');
    // The struct still ends at 416 bytes (the gate reused the pads — no growth).
    expect(WALKAROUND_UBO_WGSL).toContain('struct size 416 bytes');
  });
});

describe('updateUBO — checkerboard packing at u32[79]/u32[83]', () => {
  it('defaults to 0/0 (OFF) when the checkerboard arg is omitted', () => {
    const backing = new Uint8Array(416);
    updateUBO(capturingDevice(backing), {} as GPUBuffer, fakeInputs());
    const u = new Uint32Array(backing.buffer);
    expect(u[79]).toBe(0); // frameParity
    expect(u[83]).toBe(0); // checkerboardOn
  });

  it('packs checkerboardOn=1 + frameParity=frameCount&1 when enabled', () => {
    const even = new Uint8Array(416);
    const odd = new Uint8Array(416);
    updateUBO(capturingDevice(even), {} as GPUBuffer, fakeInputs(), PPG_OFF, REGIR_ARG, { enabled: true, frameParity: 0 });
    updateUBO(capturingDevice(odd), {} as GPUBuffer, fakeInputs(), PPG_OFF, REGIR_ARG, { enabled: true, frameParity: 1 });
    expect(new Uint32Array(even.buffer)[83]).toBe(1);
    expect(new Uint32Array(even.buffer)[79]).toBe(0);
    expect(new Uint32Array(odd.buffer)[83]).toBe(1);
    expect(new Uint32Array(odd.buffer)[79]).toBe(1);
  });

  it('masks frameParity to a single bit (frameCount&1) even for large frame counts', () => {
    const backing = new Uint8Array(416);
    updateUBO(capturingDevice(backing), {} as GPUBuffer, fakeInputs(), PPG_OFF, REGIR_ARG, { enabled: true, frameParity: 1234567 });
    expect(new Uint32Array(backing.buffer)[79]).toBe(1234567 & 1); // == 1
  });
});

describe('checkerboard-OFF bit-identity', () => {
  it('every UBO byte is identical whether checkerboard is omitted or explicitly disabled', () => {
    const a = new Uint8Array(416);
    const b = new Uint8Array(416);
    updateUBO(capturingDevice(a), {} as GPUBuffer, fakeInputs()); // omitted ⇒ OFF
    // Explicitly OFF — even with frameParity:1, disabled forces both slots to 0.
    updateUBO(capturingDevice(b), {} as GPUBuffer, fakeInputs(), PPG_OFF, REGIR_ARG, { enabled: false, frameParity: 1 });
    expect(a).toEqual(b);
    const u = new Uint32Array(a.buffer);
    expect(u[79]).toBe(0);
    expect(u[83]).toBe(0);
  });

  it('turning the gate ON changes ONLY u32[79] + u32[83]; all other bytes identical', () => {
    const off = new Uint8Array(416);
    const on = new Uint8Array(416);
    updateUBO(capturingDevice(off), {} as GPUBuffer, fakeInputs());
    updateUBO(capturingDevice(on), {} as GPUBuffer, fakeInputs(), PPG_OFF, REGIR_ARG, { enabled: true, frameParity: 1 });
    const offU = new Uint32Array(off.buffer);
    const onU = new Uint32Array(on.buffer);
    for (let i = 0; i < offU.length; i += 1) {
      if (i === 79) {
        expect(onU[i]).toBe(1); // frameParity = 1
        expect(offU[i]).toBe(0);
      } else if (i === 83) {
        expect(onU[i]).toBe(1); // checkerboardOn = 1
        expect(offU[i]).toBe(0);
      } else {
        expect(onU[i]).toBe(offU[i]);
      }
    }
  });
});

describe('shade + resolve consume the SAME parity source', () => {
  it('shade.wgsl decodes the compacted dispatch to the active-parity pixel when checkerboardOn==1', () => {
    // The OFF default (checkerboardOn==0) keeps pix == gid.xy ⇒ full-res
    // dispatch, bit-identity with the pre-checkerboard kernel.
    expect(SHADE_WGSL).toContain('ubo.checkerboardOn == 1u');
    // ON ⇒ the compacted gid.x is doubled and offset by the per-row start
    // column so the decoded pixel lands on the active parity:
    //   startCol = (gid.y + frameParity) & 1u; px = gid.x*2 + startCol.
    expect(SHADE_WGSL).toContain('(gid.y + ubo.frameParity) & 1u');
    expect(SHADE_WGSL).toContain('gid.x * 2u + startCol');
    // The active-parity invariant: every decoded pixel satisfies
    // (px+py)&1u == frameParity (proven exhaustively in dispatchEquivalence's
    // "compacted-gid decode covers exactly the active-parity pixel set" test).
  });

  it('spatial.wgsl decodes the compacted dispatch with the SAME parity decode as shade', () => {
    // The two DI spatial passes (spatial-1/spatial-2) dominate the walkaround
    // frame — each thread does castPrimary(center) + 5× castPrimary(neighbor) =
    // 6 BVH traversals. Compacting the dispatch to the active-parity pixels
    // genuinely skips the gap-parity 6-cast work (vs a shader early-return that
    // still occupies the warp). spatial reuses shade's EXACT decode so the
    // refined spatialReservoir slots are precisely the ones shade reads:
    //   startCol = (gid.y + frameParity) & 1u; px = gid.x*2 + startCol.
    expect(SPATIAL_WGSL).toContain('ubo.checkerboardOn == 1u');
    expect(SPATIAL_WGSL).toContain('(gid.y + ubo.frameParity) & 1u');
    expect(SPATIAL_WGSL).toContain('gid.x * 2u + startCol');
    // OFF default keeps pix == gid.xy ⇒ full-res dispatch, bit-identity.
    expect(SPATIAL_WGSL).toContain('var pix = gid.xy;');
  });

  it('ris.wgsl decodes the compacted dispatch with the SAME parity decode as shade/spatial', () => {
    // RIS SEEDS the per-pixel reservoir (primary BVH cast + the M_LIGHT=64
    // emitter-candidate loop) — the most expensive initial-candidate stage.
    // Compacting it to the active-parity pixels genuinely skips the gap-parity
    // candidate generation; the gap slots keep the carried-forward reservoir the
    // FULL-RATE temporal pass refines. RIS reuses shade's EXACT decode so it
    // re-seeds precisely the reservoirs shade reads this frame:
    //   startCol = (gid.y + frameParity) & 1u; px = gid.x*2 + startCol.
    expect(RIS_WGSL).toContain('ubo.checkerboardOn == 1u');
    expect(RIS_WGSL).toContain('(gid.y + ubo.frameParity) & 1u');
    expect(RIS_WGSL).toContain('gid.x * 2u + startCol');
    // OFF default keeps pix == gid.xy ⇒ full-res dispatch, bit-identity.
    expect(RIS_WGSL).toContain('var pix = gid.xy;');
  });

  it('resolve.wgsl gap-fills the COMPLEMENT — (px+py)&1 == frameParity is the SHADED half', () => {
    // resolve treats == frameParity as shaded (copy through) and != as gap
    // (reproject) — the exact complement of shade's skip set, so shade skips
    // == resolve gap-fills.
    expect(RESOLVE_WGSL).toContain('((px + py) & 1u) == frameParity');
    // OFF (checkerboardOn==0) keeps resolve in passthrough.
    expect(RESOLVE_WGSL).toContain('if (checkerboardOn == 0u) { return true; }');
  });
});
