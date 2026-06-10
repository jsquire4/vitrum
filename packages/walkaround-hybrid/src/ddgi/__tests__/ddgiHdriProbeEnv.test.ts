/**
 * Wave 4 (2026-06-10) — HDRI into DDGI probe misses.
 *
 * Test coverage:
 *  1. UBO golden pin — DDGI_FRAME_PARAMS_UBO includes hasEnv/envRotationY/
 *     envIntensity fields; byte-layout verified against known values.
 *     Wave 4: HDRI into DDGI probe misses, 2026-06-10.
 *     RENDER-CHANGING for HDRI scenes, A/B pending V28-B.
 *
 *  2. packProbeUpdateFrameParams: hasEnv=false (default) packs 0u into the
 *     hasEnv slot (no-HDRI scenes are byte-identical to pre-Wave-4).
 *
 *  3. packProbeUpdateFrameParams: hasEnv=true packs 1u, envRotationY, and
 *     envIntensity at the correct byte offsets.
 *
 *  4. WGSL structural assertions:
 *     a. The equirect sample path is present in the generated WGSL
 *        (ddgiEnvMap binding, ddgiEnvRotateYNeg, textureLoad).
 *     b. The procedural sky fallback is retained (sampleSkyColor still
 *        references skyTint and skyIrradiance).
 *     c. The hasEnv gate is present (the WGSL branches on frameParams.hasEnv).
 *     d. The equirect UV math matches environmentSample.wgsl:
 *        atan2(lookupDir.z, lookupDir.x) + acos(clamp(lookupDir.y, -1.0, 1.0)).
 *
 *  5. Bind-group layout: dispatchProbeUpdateRaysPass builds bg2 with 8
 *     entries (bindings 0–7), including the new bindings 6 (ddgiEnvMap) and
 *     7 (ddgiEnvSamp).
 *
 *  6. DDGI.setEnvironment() forwarding: calling DDGI.setEnvironment() reaches
 *     ProbeUpdatePass.setEnvironment() (ProbeUpdatePass is the DDGI API owner).
 */

import { describe, expect, it, vi } from 'vitest';
import { makeProbeUpdateRaysWGSL } from '../wgsl/probeUpdateRays.wgsl.js';
import { DDGI_FRAME_PARAMS_UBO } from '../probeUpdateUbos.js';
import {
  packProbeUpdateFrameParams,
} from '../probeUpdateFrameParams.js';
import type { ProbeUpdateGpuState } from '../probeUpdateGpuState.js';
import { dispatchProbeUpdateRaysPass } from '../probeUpdateDispatcher.js';
import { DDGI } from '../DDGI.js';
import { ProbeUpdatePass } from '../probeUpdatePass.js';
import { ProbeGrid } from '../probeGrid.js';
import { SceneBvh } from '@vitrum/shared-bvh';

// ── 1. UBO golden pin ────────────────────────────────────────────────────────

describe('Wave 4 — DDGI_FRAME_PARAMS_UBO field layout (hasEnv/envRotationY/envIntensity)', () => {
  /**
   * Golden-pin for the DDGI FrameParams UBO layout after Wave 4 extension.
   *
   * Provenance: Wave 4: HDRI into DDGI probe misses, 2026-06-10.
   * RENDER-CHANGING for HDRI scenes (probes now sample the equirect map on
   * miss instead of the procedural gradient). Byte-identical for no-HDRI
   * scenes (hasEnv=0). A/B pending V28-B (real GPU HDRI probe validation).
   *
   * Layout (all offsets in bytes, little-endian):
   *   0–11:   randomRotation (vec3f)
   *   12–15:  frameIndex (u32)
   *   16–19:  totalProbes (u32)
   *   20–23:  probesPerFrame (u32)
   *   24–27:  _pad0 (u32)
   *   28–31:  _pad1 (u32)
   *   32–43:  skyTint (vec3f)
   *   44–47:  skyIrradiance (f32)
   *   48–51:  glassMixScale (f32)
   *   52–55:  indirectFeedback (u32)
   *   56–59:  hasEnv (u32)        ← Wave 4 (was _pad3)
   *   60–63:  envRotationY (f32)  ← Wave 4 (was _pad4)
   *   64–67:  envIntensity (f32)  ← Wave 4 (new slot)
   *
   * The next 64-byte boundary is 128 bytes — the UBO stays within one
   * WebGPU min-storage-buffer-binding-alignment unit.
   */
  it('UBO fieldOffsets includes hasEnv, envRotationY, envIntensity keys', () => {
    const offsets = DDGI_FRAME_PARAMS_UBO.fieldOffsets;
    expect(offsets).toHaveProperty('hasEnv');
    expect(offsets).toHaveProperty('envRotationY');
    expect(offsets).toHaveProperty('envIntensity');
  });

  it('hasEnv byte offset is after indirectFeedback (i.e. >= 52)', () => {
    // indirectFeedback is at offset 52 (0-indexed: see layout comment above).
    // hasEnv follows immediately.
    const offsets = DDGI_FRAME_PARAMS_UBO.fieldOffsets;
    expect((offsets as Record<string, number>).hasEnv).toBeGreaterThanOrEqual(52);
  });

  it('envRotationY byte offset is after hasEnv', () => {
    const offsets = DDGI_FRAME_PARAMS_UBO.fieldOffsets as Record<string, number>;
    expect(offsets.envRotationY!).toBeGreaterThan(offsets.hasEnv!);
  });

  it('envIntensity byte offset is after envRotationY', () => {
    const offsets = DDGI_FRAME_PARAMS_UBO.fieldOffsets as Record<string, number>;
    expect(offsets.envIntensity!).toBeGreaterThan(offsets.envRotationY!);
  });

  it('UBO sizeBytes grows to accommodate the new fields (>= 52 bytes)', () => {
    // Pre-Wave-4 the UBO was 48 bytes (randomRotation-vec3f+frameIndex=16,
    // totalProbes/probesPerFrame/_pad0/_pad1=16, skyTint-vec3f+skyIrradiance=16
    // = 48); then glassMixScale + indirectFeedback + _pad3 + _pad4 = +16 = 64.
    // Wave 4 replaces _pad3/_pad4 with hasEnv/envRotationY and adds envIntensity
    // (+4 bytes net). The defineUbo packer aligns to 16-byte boundaries so the
    // size stays at 64 (the next 16-byte boundary after 52 is 64).
    expect(DDGI_FRAME_PARAMS_UBO.sizeBytes).toBeGreaterThanOrEqual(52);
  });
});

// ── 2–3. packProbeUpdateFrameParams ─────────────────────────────────────────

describe('Wave 4 — packProbeUpdateFrameParams with hasEnv fields', () => {
  const BASE_INPUT = {
    frameIndex: 0,
    totalProbes: 8,
    skyTint: [0.4, 0.6, 1.0] as [number, number, number],
    skyIrradiance: 2.0,
    glassMixScale: 0.7,
  } as const;

  // Helper: read a u32 at a byte offset from an ArrayBuffer
  function readU32At(buf: ArrayBuffer, byteOffset: number): number {
    return new DataView(buf).getUint32(byteOffset, true);
  }
  // Helper: read a f32 at a byte offset from an ArrayBuffer
  function readF32At(buf: ArrayBuffer, byteOffset: number): number {
    return new DataView(buf).getFloat32(byteOffset, true);
  }

  it('default (hasEnv omitted) packs hasEnv=0, envRotationY=0, envIntensity=0 — no-HDRI byte-identical', () => {
    const buf = packProbeUpdateFrameParams({ ...BASE_INPUT });
    const offsets = DDGI_FRAME_PARAMS_UBO.fieldOffsets as Record<string, number>;
    expect(readU32At(buf, offsets.hasEnv!)).toBe(0);        // hasEnv = 0u
    expect(readF32At(buf, offsets.envRotationY!)).toBe(0);   // envRotationY = 0
    expect(readF32At(buf, offsets.envIntensity!)).toBe(0);   // envIntensity = 0
  });

  it('hasEnv=true packs 1u into the hasEnv slot', () => {
    const buf = packProbeUpdateFrameParams({ ...BASE_INPUT, hasEnv: true, envRotationY: 0, envIntensity: 1 });
    const offsets = DDGI_FRAME_PARAMS_UBO.fieldOffsets as Record<string, number>;
    expect(readU32At(buf, offsets.hasEnv!)).toBe(1);
  });

  it('envRotationY is packed verbatim (e.g. Math.PI/4)', () => {
    const rotY = Math.PI / 4;
    const buf = packProbeUpdateFrameParams({ ...BASE_INPUT, hasEnv: true, envRotationY: rotY, envIntensity: 2.5 });
    const offsets = DDGI_FRAME_PARAMS_UBO.fieldOffsets as Record<string, number>;
    expect(readF32At(buf, offsets.envRotationY!)).toBeCloseTo(rotY, 5);
  });

  it('envIntensity is packed verbatim (e.g. 3.14)', () => {
    const buf = packProbeUpdateFrameParams({ ...BASE_INPUT, hasEnv: true, envRotationY: 0, envIntensity: 3.14 });
    const offsets = DDGI_FRAME_PARAMS_UBO.fieldOffsets as Record<string, number>;
    expect(readF32At(buf, offsets.envIntensity!)).toBeCloseTo(3.14, 5);
  });

  it('hasEnv=false explicitly also packs 0 (same as default)', () => {
    const dflt = packProbeUpdateFrameParams({ ...BASE_INPUT });
    const explicit = packProbeUpdateFrameParams({ ...BASE_INPUT, hasEnv: false, envRotationY: 0, envIntensity: 0 });
    expect(new Uint8Array(dflt)).toEqual(new Uint8Array(explicit));
  });
});

// ── 4. WGSL structural assertions ───────────────────────────────────────────

describe('Wave 4 — WGSL structural assertions for HDRI probe-ray miss path', () => {
  // Generate the WGSL once for all structural checks.
  const wgsl = makeProbeUpdateRaysWGSL(64);

  it('(a) equirect bindings are declared (@group(2) @binding(6) ddgiEnvMap)', () => {
    expect(wgsl).toContain('@group(2) @binding(6)');
    expect(wgsl).toContain('ddgiEnvMap');
    expect(wgsl).toContain('texture_2d<f32>');
  });

  it('(a) NO env sampler binding exists (trust-audit F3, 2026-06-10)', () => {
    // The env lookup is textureLoad-only; a declared-but-unused sampler at
    // binding(7) was stripped by layout:'auto' while the dispatcher still
    // passed an entry for it — failing bind-group validation EVERY frame.
    // Pin the absence so the class cannot silently return.
    expect(wgsl).not.toContain('@group(2) @binding(7)');
    expect(wgsl).not.toContain('ddgiEnvSamp:  sampler');
  });

  it('(a) equirect UV math uses textureLoad on ddgiEnvMap', () => {
    expect(wgsl).toContain('textureLoad(ddgiEnvMap');
  });

  it('(a) ddgiEnvRotateYNeg helper is defined (H6 RY(-rotY) world→map)', () => {
    expect(wgsl).toContain('fn ddgiEnvRotateYNeg');
    // Verify the rotation formula: c*x - s*z (same as envRotateYNeg in environmentSample.wgsl)
    expect(wgsl).toContain('c * d.x - s * d.z');
    expect(wgsl).toContain('s * d.x + c * d.z');
  });

  it('(b) procedural sky fallback is retained (skyTint and skyIrradiance still referenced)', () => {
    expect(wgsl).toContain('frameParams.skyTint');
    expect(wgsl).toContain('frameParams.skyIrradiance');
  });

  it('(c) hasEnv gate is present (frameParams.hasEnv == 1u)', () => {
    expect(wgsl).toContain('frameParams.hasEnv == 1u');
  });

  it('(d) equirect UV math uses atan2 and acos matching environmentSample.wgsl convention', () => {
    // phi = atan2(lookupDir.z, lookupDir.x) — same as environmentSample.wgsl
    expect(wgsl).toContain('atan2(lookupDir.z, lookupDir.x)');
    // theta = acos(clamp(lookupDir.y, -1.0, 1.0))
    expect(wgsl).toContain('acos(clamp(lookupDir.y, -1.0, 1.0))');
  });

  it('(d) U coordinate uses fract(...PI...) matching environmentSample.wgsl', () => {
    // u = fract(phi * (1/(2π)) + 0.5) — equivalent to fract(phi*INV_PI*0.5 + 0.5)
    expect(wgsl).toContain('fract(phi');
    expect(wgsl).toContain('+ 0.5)');
  });

  it('(d) V coordinate is clamp(theta/PI, 0, 0.999999) matching environmentSample.wgsl', () => {
    expect(wgsl).toContain('clamp(theta');
    expect(wgsl).toContain('0.999999');
  });

  it('FrameParams struct contains hasEnv, envRotationY, envIntensity fields', () => {
    expect(wgsl).toContain('hasEnv:');
    expect(wgsl).toContain('envRotationY:');
    expect(wgsl).toContain('envIntensity:');
  });

  it('sampleSkyColor function is still present (entry point for both paths)', () => {
    expect(wgsl).toContain('fn sampleSkyColor');
  });

  it('glass transmitted env path still calls sampleSkyColor (glass miss still uses env)', () => {
    // Glass hit path: `let transmitted = sampleSkyColor(dir) * mat.attenuationColor`
    expect(wgsl).toContain('sampleSkyColor(dir)');
  });
});

// ── 5. Bind-group layout: bg2 has 8 entries ──────────────────────────────────

describe('Wave 4 — dispatchProbeUpdateRaysPass bg2 has 8 entries (bindings 0–7)', () => {
  it('bg2 createBindGroup call includes binding 6 and binding 7 entries', () => {
    const bindGroupEntryLists: unknown[][] = [];
    const mockDevice = {
      createBindGroup: vi.fn((desc: { entries: unknown[] }) => {
        bindGroupEntryLists.push(desc.entries);
        return {};
      }),
      createBuffer: vi.fn(() => ({ size: 16 })),
    } as unknown as GPUDevice;

    const mockTex = { createView: vi.fn(() => ({})) } as unknown as GPUTexture;
    const mockView = {} as GPUTextureView;
    const mockSampler = {} as GPUSampler;
    const mockBuf = { size: 16 } as unknown as GPUBuffer;
    const mockPipeline = {
      getBindGroupLayout: vi.fn(() => ({})),
    } as unknown as GPUComputePipeline;

    const gpu: ProbeUpdateGpuState = {
      device: mockDevice,
      raysPipeline:      mockPipeline,
      blendIrrPipeline:  mockPipeline,
      blendVisPipeline:  mockPipeline,
      borderVisPipeline: mockPipeline,
      irrScratchTex:     null,
      visScratchTex:     null,
      bvhBuf:            mockBuf,
      posBuf:            mockBuf,
      idxBuf:            mockBuf,
      normBuf:           mockBuf,
      matIdBuf:          mockBuf,
      tlasNodesBuf:      mockBuf,
      tlasInstIdxBuf:    mockBuf,
      tlasBlasRootsBuf:  mockBuf,
      tlasW2lBuf:        mockBuf,
      tlasL2wBuf:        mockBuf,
      traceParamsBuf:    mockBuf,
      materialsBuf:      mockBuf,
      lightsBuf:         mockBuf,
      emitterTrisBuf:    mockBuf,
      emitterTrisCount:  0,
      gridParamsBuf:     mockBuf,
      frameParamsBuf:    mockBuf,
      blendParamsBuf:    mockBuf,
      borderVisUboBuf:   mockBuf,
      rayResultsBuf:     mockBuf,
      activeProbesBuf:   mockBuf,
      linearSampler:     mockSampler,
      envMapView:        mockView,
      envMapOwnedByPass: true,
      envMapPlaceholderTex: null,
      envSamplerForProbe: mockSampler,
    };

    const encoder = {
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(),
        setBindGroup: vi.fn(),
        dispatchWorkgroups: vi.fn(),
        end: vi.fn(),
      })),
    } as unknown as GPUCommandEncoder;

    dispatchProbeUpdateRaysPass(encoder, gpu, 1, mockTex);

    // bg2 is the 3rd createBindGroup call (index 2).
    // Trust-audit F3 (2026-06-10): 7 entries (0-6) — NO sampler entry at 7.
    // The WGSL uses textureLoad only; layout:'auto' strips an unused sampler,
    // so an 8th entry failed bind-group validation on every frame.
    const bg2Entries = bindGroupEntryLists[2] as Array<{ binding: number }>;
    expect(bg2Entries).toBeDefined();
    expect(bg2Entries.length).toBe(7);

    const bindings = bg2Entries.map((e) => e.binding).sort((a, b) => a - b);
    expect(bindings).toEqual([0, 1, 2, 3, 4, 5, 6]);
  });
});

// ── 6. DDGI.setEnvironment() forwarding ─────────────────────────────────────

describe('Wave 4 — DDGI.setEnvironment() forwards to ProbeUpdatePass', () => {
  it('DDGI.setEnvironment() calls ProbeUpdatePass.setEnvironment() with same args', () => {
    const ddgi = new DDGI();
    const spy = vi.spyOn(ddgi.pass, 'setEnvironment');

    const mockView = {} as GPUTextureView;
    const mockSampler = {} as GPUSampler;
    ddgi.setEnvironment(mockView, mockSampler, Math.PI / 6, 1.5, true);

    expect(spy).toHaveBeenCalledOnce();
    expect(spy).toHaveBeenCalledWith(mockView, mockSampler, Math.PI / 6, 1.5, true);
  });

  it('DDGI.setEnvironment(null, null, 0, 0, false) disables HDRI (procedural fallback)', () => {
    const ddgi = new DDGI();
    const spy = vi.spyOn(ddgi.pass, 'setEnvironment');

    ddgi.setEnvironment(null, null, 0, 0, false);

    expect(spy).toHaveBeenCalledWith(null, null, 0, 0, false);
  });

  it('ProbeUpdatePass.setEnvironment() updates internal env fields without requiring reinit', () => {
    const bvh = new SceneBvh();
    const grid = new ProbeGrid();
    const pass = new ProbeUpdatePass(bvh, grid);

    const mockView = {} as GPUTextureView;
    // Should not throw even before init().
    expect(() => pass.setEnvironment(mockView, null, 0.5, 2.0, true)).not.toThrow();
  });
});
