/**
 * grisReuseUbo.test.ts — GRIS / ReSTIR-PT reconnection-shift reuse gate wiring.
 *
 * Verifies:
 *   1. UBO contract — WalkaroundUBO declares `restirPtReuse` at the documented
 *      offset 412 (the repurposed `_regirPad` slot) and the struct still ends at
 *      416 bytes.
 *   2. `updateUBO` packs the gate at u32 index 103, defaulting to 0 (OFF).
 *   3. OFF-is-BIT-IDENTICAL — with `restirPtReuse` absent / 0, EVERY UBO byte is
 *      identical to the pre-GRIS packing (the gate byte stays 0), so the GI
 *      spatial/temporal reuse the shader runs is byte-for-byte the legacy path.
 *   4. COMPILE-TIME GRIS gating — the GRIS reconnection shift lives in
 *      SEPARATE shader variants (SPATIAL_GI_GRIS_WGSL / TEMPORAL_GI_GRIS_WGSL),
 *      composed only when the host opts into restirPtReuse. The DEFAULT (OFF)
 *      variants now still bind @group(1), because receiver-lobe GI p-hat recasts
 *      need scene/material payloads, but they contain no GRIS branch.
 *
 * The UBO `restirPtReuse` field is kept (harmless telemetry/consistency) so the
 * offset / packing / OFF-bit-identity contracts below still hold.
 */

import { describe, expect, it } from 'vitest';
import { updateUBO } from '../src/pipeline/uboUpdater.js';
import { WALKAROUND_UBO_WGSL } from '../src/shaders/walkaroundUbo.wgsl.js';
import { SPATIAL_GI_WGSL, SPATIAL_GI_GRIS_WGSL } from '../src/shaders/spatialGi.wgsl.js';
import { TEMPORAL_GI_WGSL, TEMPORAL_GI_GRIS_WGSL } from '../src/shaders/temporalGi.wgsl.js';
import { GRIS_REUSE_WGSL } from '../src/shaders/grisReuse.wgsl.js';
import type { PipelineFrameInputs } from '../src/pipeline/WalkaroundGPUPipeline.js';

// Minimal frame inputs (only the fields updateUBO reads). `restirPtReuse` is
// deliberately omitted so the default-OFF path is exercised.
function fakeInputs(restirPtReuseOverride?: number): PipelineFrameInputs {
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
      ...(restirPtReuseOverride !== undefined ? { restirPtReuse: restirPtReuseOverride } : {}),
    },
    gtao: { gtaoRadiusPx: 32, gtaoIntensity: 2, gtaoDepthThreshold: 2, gtaoBilateralDepthSigma: 0.25, adaptiveSamplingThresholdLow: 0.01, adaptiveSamplingThresholdHigh: 0.1 },
    filter: {
      triIntersectEpsilon: 1e-5, glassMixScale: 0.7, indirectFireflyClamp: [1, 1, 1],
      atrousDirectSigmas: [128, 5, 0.05], atrousIndirectSigmas: [32, 20, 0.5],
      stainedGlassFlags: 0,
    },
    bvh: { bvhMode: 0, tlasNodeCount: 0 },
    nrc: {},
    composite: { tonemapMode: 0, exposure: 1.0, outputColorSpace: 0 },
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

describe('WalkaroundUBO — GRIS reuse gate contract', () => {
  it('declares restirPtReuse at offset 412 (the former _regirPad slot)', () => {
    expect(WALKAROUND_UBO_WGSL).toContain('restirPtReuse:              u32,');
    expect(WALKAROUND_UBO_WGSL).toContain('offset 412 — GRIS reuse gate');
    // No _regirPad name survives.
    expect(WALKAROUND_UBO_WGSL).not.toContain('_regirPad:');
    // The struct still ends at 416 bytes (the gate reused the pad — no growth).
    expect(WALKAROUND_UBO_WGSL).toContain('struct size 416 bytes');
  });
});

describe('updateUBO — restirPtReuse packing at u32[103]', () => {
  it('defaults to 0 (OFF) when the input omits restirPtReuse', () => {
    const backing = new Uint8Array(416);
    updateUBO(capturingDevice(backing), {} as GPUBuffer, fakeInputs());
    expect(new Uint32Array(backing.buffer)[103]).toBe(0);
  });

  it('packs 1 when restirPtReuse is set', () => {
    const backing = new Uint8Array(416);
    updateUBO(capturingDevice(backing), {} as GPUBuffer, fakeInputs(1));
    expect(new Uint32Array(backing.buffer)[103]).toBe(1);
  });
});

describe('GRIS-OFF bit-identity', () => {
  it('every UBO byte is identical whether restirPtReuse is omitted or explicitly 0', () => {
    const a = new Uint8Array(416);
    const b = new Uint8Array(416);
    updateUBO(capturingDevice(a), {} as GPUBuffer, fakeInputs()); // omitted ⇒ 0
    updateUBO(capturingDevice(b), {} as GPUBuffer, fakeInputs(0));
    expect(a).toEqual(b);
    // And the gate byte is 0 in both — the reuse passes read the OFF path.
    expect(new Uint32Array(a.buffer)[103]).toBe(0);
  });

  it('turning the gate ON changes ONLY u32[103] (offset 412); all other bytes identical', () => {
    const off = new Uint8Array(416);
    const on = new Uint8Array(416);
    updateUBO(capturingDevice(off), {} as GPUBuffer, fakeInputs());
    updateUBO(capturingDevice(on), {} as GPUBuffer, fakeInputs(1));
    const offU = new Uint32Array(off.buffer);
    const onU = new Uint32Array(on.buffer);
    for (let i = 0; i < offU.length; i += 1) {
      if (i === 103) {
        expect(onU[i]).toBe(1);
        expect(offU[i]).toBe(0);
      } else {
        // Every OTHER u32 word is byte-identical — the gate is the only delta,
        // so OFF is bit-identical and ON is a single-field flip.
        expect(onU[i]).toBe(offU[i]);
      }
    }
  });
});

describe('GI reuse shaders — GRIS gated at COMPILE time (separate variants)', () => {
  it('spatialGi OFF (default) is standard reuse with receiver-lobe scene recast and NO GRIS branch', () => {
    // Legacy clamped-Jacobian reuse only.
    expect(SPATIAL_GI_WGSL).toContain('jacobianReconnectionShift(');
    // The runtime gate is GONE — gating is compile-time-shader-variant now.
    expect(SPATIAL_GI_WGSL).not.toContain('ubo.restirPtReuse == 1u');
    // No GRIS symbols or reconnection-visibility ray, but group(1) is now
    // required for receiver-material p-hat recasts.
    expect(SPATIAL_GI_WGSL).not.toContain('grisShiftJacobian(');
    expect(SPATIAL_GI_WGSL).not.toContain('grisReconnectionVisible(');
    expect(SPATIAL_GI_WGSL).not.toContain('grisPairwiseDenomNeighbor(');
    expect(SPATIAL_GI_WGSL).toContain('@group(1)');
    expect(SPATIAL_GI_WGSL).toContain('restir_gi_receiver_phat_from_surface_or_geometry(');
  });

  it('spatialGi ON (GRIS) carries the shift + reconnection-visibility + full-GBH MIS', () => {
    expect(SPATIAL_GI_GRIS_WGSL).toContain('grisShiftJacobian(');
    expect(SPATIAL_GI_GRIS_WGSL).toContain('grisReconnectionVisible(');
    // The spatial pass combines with the EXACT generalized balance heuristic
    // (Σ m_i = 1) rather than the streaming pairwise approximation: it gathers
    // the accepted neighbours and builds the full GBH denominator inline from
    // the per-domain receiver-lobe target, so the streaming pairwise helper is
    // no longer called here (the pairwise denom drifts off unity for K>1 and
    // over-energised the reservoir — the V19 spatial divergence).
    expect(SPATIAL_GI_GRIS_WGSL).toContain('restir_gi_receiver_phat_from_surface_or_geometry(');
    expect(SPATIAL_GI_GRIS_WGSL).not.toContain('grisPairwiseDenomNeighbor(');
    // The GRIS variant declares the @group(1) scene BVH/TLAS group.
    expect(SPATIAL_GI_GRIS_WGSL).toContain('@group(1)');
    // It is GRIS-only: no legacy clamped-Jacobian helper, no runtime gate.
    expect(SPATIAL_GI_GRIS_WGSL).not.toContain('jacobianReconnectionShift(');
    expect(SPATIAL_GI_GRIS_WGSL).not.toContain('ubo.restirPtReuse == 1u');
  });

  it('temporalGi OFF (default) is standard reuse with receiver-lobe scene recast and NO GRIS branch', () => {
    expect(TEMPORAL_GI_WGSL).toContain('jacobianReconnectionShift(');
    expect(TEMPORAL_GI_WGSL).not.toContain('ubo.restirPtReuse == 1u');
    expect(TEMPORAL_GI_WGSL).not.toContain('grisShiftJacobian(');
    expect(TEMPORAL_GI_WGSL).not.toContain('tgiReconnectionVisible(');
    expect(TEMPORAL_GI_WGSL).not.toContain('grisPairwiseDenomNeighbor(');
    expect(TEMPORAL_GI_WGSL).toContain('@group(1)');
    expect(TEMPORAL_GI_WGSL).toContain('restir_gi_receiver_phat_from_surface_or_geometry(');
  });

  it('temporalGi ON (GRIS) carries the shift + reconnection-visibility + pairwise MIS', () => {
    expect(TEMPORAL_GI_GRIS_WGSL).toContain('grisShiftJacobian(');
    expect(TEMPORAL_GI_GRIS_WGSL).toContain('tgiReconnectionVisible(');
    expect(TEMPORAL_GI_GRIS_WGSL).toContain('grisPairwiseDenomNeighbor(');
    expect(TEMPORAL_GI_GRIS_WGSL).toContain('@group(1)');
    expect(TEMPORAL_GI_GRIS_WGSL).not.toContain('jacobianReconnectionShift(');
    expect(TEMPORAL_GI_GRIS_WGSL).not.toContain('ubo.restirPtReuse == 1u');
  });

  it('grisReuse module mirrors the oracle geometry-term + Jacobian-from-cached-half-G', () => {
    // The destination-cosine half-G and the Jacobian-as-ratio are the WGSL
    // counterparts the TS mirror (grisReuseMis.ts) pins against the oracle.
    expect(GRIS_REUSE_WGSL).toContain('fn grisReconnectionGeometryTerm(');
    expect(GRIS_REUSE_WGSL).toContain('fn grisShiftJacobian(');
    expect(GRIS_REUSE_WGSL).toContain('fn grisTargetAt(');
  });
});
