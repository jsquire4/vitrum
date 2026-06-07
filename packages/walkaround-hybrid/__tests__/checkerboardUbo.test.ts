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
 *   5. Shade + resolve consume the SAME parity source — shade.wgsl's gap
 *      early-out compares `(gid.x+gid.y)&1u != ubo.frameParity`, and
 *      resolve.wgsl's `isShadedPixel` compares `(px+py)&1u == frameParity`, so
 *      the pixels shade skips are exactly the pixels resolve gap-fills.
 */

import { describe, expect, it } from 'vitest';
import { updateUBO } from '../src/pipeline/uboUpdater.js';
import { WALKAROUND_UBO_WGSL } from '../src/shaders/walkaroundUbo.wgsl.js';
import { SHADE_WGSL } from '../src/shaders/shade.wgsl.js';
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
  it('shade.wgsl gates the gap early-out on ubo.checkerboardOn==1 + parity != frameParity', () => {
    // The OFF default (checkerboardOn==0) never takes the early-out ⇒ bit-identity.
    expect(SHADE_WGSL).toContain('ubo.checkerboardOn == 1u');
    // Gap pixel = parity NOT equal to frameParity (shade SKIPS those when ON).
    expect(SHADE_WGSL).toContain('((gid.x + gid.y) & 1u) != (ubo.frameParity & 1u)');
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
