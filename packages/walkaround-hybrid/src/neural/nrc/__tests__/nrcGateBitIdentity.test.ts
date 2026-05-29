// nrcGateBitIdentity.test.ts — the LOAD-BEARING gate test for the NRC opt-in.
//
// Proves the honest OFF-bit-identity acceptance criterion at the UBO-byte level:
// when nrcEnabled is 0/absent, EVERY byte of the WalkaroundUBO is unchanged from
// the pre-NRC layout, and ONLY u32[91] (offset 364, the former _ppgPad2 slot)
// flips to 1 when nrcEnabled is on. Same discipline as the V19 GRIS gate proof.
//
// The NRC gate lands in the previously-zero `_ppgPad2` pad slot, so an OFF gate
// is byte-for-byte identical to a build with no NRC field at all — the default
// GI path is provably unchanged.

import { describe, it, expect } from 'vitest';
import { updateUBO } from '../../../pipeline/uboUpdater.ts';
import type { PipelineFrameInputs } from '../../../pipeline/WalkaroundGPUPipeline.ts';

// A fake GPUDevice whose queue.writeBuffer captures the bytes updateUBO writes.
function makeCapturingDevice(): { device: GPUDevice; captured: () => Uint8Array } {
  let last: Uint8Array | null = null;
  const queue = {
    writeBuffer: (_buf: unknown, _off: number, data: BufferSource) => {
      const ab = (data as ArrayBufferView).buffer ?? (data as ArrayBuffer);
      last = new Uint8Array(ab.slice(0));
    },
  };
  const device = { queue } as unknown as GPUDevice;
  return { device, captured: () => last ?? new Uint8Array() };
}

// Minimal-but-complete inputs for the UBO write. updateUBO only reads the
// UBO-relevant numeric/array fields (never swapChainView), so we cast a partial.
function baseInputs(): PipelineFrameInputs {
  const m = new Float32Array(16);
  for (let i = 0; i < 16; i++) m[i] = i * 0.5 + 1;
  return {
    viewMatrix: m, projMatrix: m, prevViewMatrix: m, prevProjMatrix: m,
    cameraPos: [1, 2, 3],
    screenWidth: 1920, screenHeight: 1080,
    frameSeed: 42,
    totalEmissivePower: 12.5, emitterCount: 7,
    primaryLightDir: [0, -1, 0], primaryLightIntensity: 3,
    skyTint: [0.6, 0.7, 0.9], skyIrradiance: 1.5,
    emitterDist2Floor: 0.01, directFireflyClamp: 4,
    causticBoost: 1, causticVisClamp: 1,
    temporalMClampDI: 20, spatialReuseRadiusPx: 30, spatialDepthTolFloor: 0.05,
    triIntersectEpsilon: 1e-5, glassMixScale: 0.7,
    restirGiWCap: 16, restirGiIrrClamp: 5, restirGiMClamp: 50,
    restirGiSpatialRadiusPx: 12, restirGiSpatialNormalDotMin: 0.906,
    restirGiSpatialCoplanarTol: 0.05,
    indirectFireflyClamp: [1, 1, 1],
    bvhMode: 0, tlasNodeCount: 0,
    lightTreeEnabled: 0, lightTreeNodeCount: 0,
    stainedGlassFlags: 0,
    restirPtReuse: 0,
    // nrcEnabled deliberately omitted in the OFF case → must default to 0.
  } as unknown as PipelineFrameInputs;
}

const NRC_GATE_U32_INDEX = 91; // offset 364 / 4

describe('NRC gate — OFF bit-identity (the honest acceptance criterion)', () => {
  it('omitting nrcEnabled is byte-identical to nrcEnabled: 0', () => {
    const a = makeCapturingDevice();
    updateUBO(a.device, {} as GPUBuffer, baseInputs());
    const off = a.captured();

    const b = makeCapturingDevice();
    updateUBO(b.device, {} as GPUBuffer, { ...baseInputs(), nrcEnabled: 0 });
    const zero = b.captured();

    expect(off.length).toBe(416);
    expect(Array.from(off)).toEqual(Array.from(zero));
  });

  it('turning NRC on flips ONLY u32[91] (offset 364) — every other byte unchanged', () => {
    const off = makeCapturingDevice();
    updateUBO(off.device, {} as GPUBuffer, { ...baseInputs(), nrcEnabled: 0 });
    const offU32 = new Uint32Array(off.captured().buffer.slice(0));

    const on = makeCapturingDevice();
    updateUBO(on.device, {} as GPUBuffer, { ...baseInputs(), nrcEnabled: 1 });
    const onU32 = new Uint32Array(on.captured().buffer.slice(0));

    expect(offU32.length).toBe(104); // 416 / 4
    for (let i = 0; i < offU32.length; i++) {
      if (i === NRC_GATE_U32_INDEX) {
        expect(offU32[i]).toBe(0);
        expect(onU32[i]).toBe(1);
      } else {
        expect(onU32[i]).toBe(offU32[i]);
      }
    }
  });

  it('the gate uses the former _ppgPad2 slot (offset 364) — no UBO size change', () => {
    const on = makeCapturingDevice();
    updateUBO(on.device, {} as GPUBuffer, { ...baseInputs(), nrcEnabled: 1 });
    const bytes = on.captured();
    // Size unchanged at 416 (416 % 16 == 0) — the gate repurposed a pad slot.
    expect(bytes.length).toBe(416);
    const u32 = new Uint32Array(bytes.buffer.slice(0));
    // restirPtReuse (offset 412 / u32[103]) is still the LAST field and unaffected.
    expect(u32[103]).toBe(0);
    expect(u32[NRC_GATE_U32_INDEX]).toBe(1);
  });
});
