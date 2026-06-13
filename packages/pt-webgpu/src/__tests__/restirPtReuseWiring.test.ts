/**
 * restirPtReuseWiring.test.ts — wiring contract for the EXPERIMENTAL ReSTIR-PT
 * reservoir/reuse pre-passes (gpuResources.ts + index.ts).
 *
 * The load-bearing invariant of this increment is OFF-BYTE-IDENTITY: with the
 * `restirPtReuse` flag OFF (the default), NONE of the reuse resources/pipelines
 * are created and the default megakernel render is unchanged. ON allocates the
 * reservoir ping-pong + result + params and creates the three reuse pipelines.
 *
 * NO real GPU here (mock device, like seedAccumulator.test.ts). The naga-compile
 * gate that the composed reuse modules actually COMPILE on a device is the
 * separate hardware step `wsl-gpu/scripts/restir-pt-compile-gate.ts` (string
 * goldens / mock pipelines do NOT catch symbol-scope / binding mismatches); the
 * unbiasedness A/B is `wsl-gpu/scripts/restir-pt-unbiased-ab.ts`.
 */

import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import {
  composeRestirPtProducerWgsl,
  composeRestirPtTemporalWgsl,
  composeRestirPtResolveWgsl,
  RPT_GROUP0_BINDING_BASE,
} from '../wgsl/pathTrace/restirPtCompose.wgsl.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

// ── A mock device rich enough to run a full-tier renderFrame ──────────────────
// Records every createShaderModule code + createComputePipeline entryPoint +
// createBuffer label so the test can assert which resources the reuse path
// created. Reports full-tier limits so resolvePtWebgpuTraceTier picks 'full'.
interface Recorder {
  shaderCodes: string[];
  pipelineEntryPoints: string[];
  bufferLabels: string[];
  bindGroupLabels: string[];
  computePassLabels: string[];
}

function makeFullTierDevice(rec: Recorder): GPUDevice {
  installGpuConstStubs();
  const pass = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
    end: vi.fn(),
  };
  const encoder = {
    beginComputePass: vi.fn((desc?: { label?: string }) => {
      rec.computePassLabels.push(desc?.label ?? '');
      return pass;
    }),
    clearBuffer: vi.fn(),
    copyBufferToBuffer: vi.fn(),
    finish: vi.fn(() => ({})),
  };
  return {
    queue: { writeBuffer: vi.fn(), writeTexture: vi.fn(), submit: vi.fn() },
    createBuffer: vi.fn((desc?: { label?: string }) => {
      rec.bufferLabels.push(desc?.label ?? '');
      return { label: desc?.label ?? '', destroy: vi.fn() };
    }),
    ...textureStubMethods(),
    createShaderModule: vi.fn((desc?: { code?: string }) => {
      rec.shaderCodes.push(desc?.code ?? '');
      return { getCompilationInfo: vi.fn(async () => ({ messages: [] })) };
    }),
    createBindGroupLayout: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    createComputePipeline: vi.fn((desc?: { compute?: { entryPoint?: string } }) => {
      rec.pipelineEntryPoints.push(desc?.compute?.entryPoint ?? '');
      return { getBindGroupLayout: vi.fn(() => ({})) };
    }),
    createBindGroup: vi.fn((desc?: { label?: string }) => {
      rec.bindGroupLabels.push(desc?.label ?? '');
      return {};
    }),
    createCommandEncoder: vi.fn(() => encoder),
    limits: { maxStorageBuffersPerShaderStage: 64, maxStorageTexturesPerShaderStage: 8 },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function emptyRecorder(): Recorder {
  return { shaderCodes: [], pipelineEntryPoints: [], bufferLabels: [], bindGroupLabels: [], computePassLabels: [] };
}

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0.1 },
      },
    ],
    emitters: [
      { kind: 'directional', id: 'sun', direction: [0, -1, 0], color: [1, 1, 1], intensity: 1 },
    ],
    environment: { kind: 'none' },
  };
}

/** Identity 4×4 (column-major) — invertible (det = 1), so packFrameParams's
 *  invViewProj computation succeeds. The actual camera values are irrelevant to
 *  the wiring assertions (which only inspect resource creation, not pixels). */
function identityMat(): Float32Array {
  const m = new Float32Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
}

function frameInput(size: number) {
  return {
    viewMatrix: asMat4(identityMat()),
    projMatrix: asMat4(identityMat()),
    cameraPosition: [0, 0, 1] as [number, number, number],
    viewport: { width: size, height: size, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 1,
    quality: { samplesTarget: 4, bounces: 2, resolutionFactor: 1 },
  };
}

describe('ReSTIR-PT reuse wiring — OFF by default (byte-identity)', () => {
  it('default-trace WGSL is untouched (the composed string is NOT modified by the reuse wiring)', () => {
    // The reuse path composes SEPARATE per-pass modules; it must never mutate the
    // default megakernel string. This is a cheap guard alongside the SHA pin in
    // wgslContract.test.ts (which is the authoritative byte-identity check).
    expect(PT_WEBGPU_TRACE_WGSL.length).toBe(326468); // re-pinned 2026-06-12 (environment:none black fallback). See wgslContract.test.ts for the authoritative SHA.
    // The default trace must NOT contain any restir-pt reuse entry point, nor the
    // A1 composite megakernel's rpt_result_in binding (that is a SEPARATE pipeline).
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('fn restirPtProduce');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('fn restirPtTemporal');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('fn restirPtSpatial');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('fn restirPtResolve');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('rpt_result_in');
  });

  it('OFF: a full-tier render creates NO reservoir buffers and NO reuse pipelines', async () => {
    const rec = emptyRecorder();
    const engine = await createPTEngine_WebGPU({ device: makeFullTierDevice(rec) });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(16));

    // No reuse pipeline entry points were created.
    expect(rec.pipelineEntryPoints).not.toContain('restirPtProduce');
    expect(rec.pipelineEntryPoints).not.toContain('restirPtTemporal');
    expect(rec.pipelineEntryPoints).not.toContain('restirPtSpatial');
    expect(rec.pipelineEntryPoints).not.toContain('restirPtResolve');
    // A1: no composite megakernel — the default megakernel runs (byte-identical).
    expect(rec.computePassLabels).not.toContain('vitrum.pt-webgpu.restirPt.spatial');
    expect(rec.shaderCodes.some((c) => c.includes('rpt_result_in'))).toBe(false);
    // No reservoir / reuse buffers were created.
    expect(rec.bufferLabels.some((l) => l.includes('restirPt'))).toBe(false);
    // The experimental capability flag is absent.
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-restir-pt-reuse')).toBe(false);
    engine.dispose();
  });

  it('OFF: getRestirPtResultBuffer returns null', async () => {
    const rec = emptyRecorder();
    const engine = await createPTEngine_WebGPU({ device: makeFullTierDevice(rec) });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(16));
    // The accessor is engine-internal (not on the Engine contract); narrow to it.
    const buf = (engine as unknown as { getRestirPtResultBuffer(): unknown }).getRestirPtResultBuffer();
    expect(buf).toBeNull();
    engine.dispose();
  });
});

describe('ReSTIR-PT reuse wiring — ON (full tier)', () => {
  it('ON: a full-tier render creates the 3 reuse pipelines + the reservoir/result/params buffers', async () => {
    const rec = emptyRecorder();
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      restirPtReuse: true,
    });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(16));

    // All four reuse compute entry points were created exactly once.
    expect(rec.pipelineEntryPoints.filter((e) => e === 'restirPtProduce').length).toBe(1);
    expect(rec.pipelineEntryPoints.filter((e) => e === 'restirPtTemporal').length).toBe(1);
    expect(rec.pipelineEntryPoints.filter((e) => e === 'restirPtSpatial').length).toBe(1);
    expect(rec.pipelineEntryPoints.filter((e) => e === 'restirPtResolve').length).toBe(1);
    // A1: the COMPOSITE megakernel was compiled (E0-direct-only + reads rpt_result_in).
    const compositeModules = rec.shaderCodes.filter((c) => c.includes('rpt_result_in'));
    expect(compositeModules.length).toBe(1);
    expect(compositeModules[0]).toContain('@group(0) @binding(23)');
    // The default megakernel (full path) must NOT be the composite (no rpt_result_in).
    expect(rec.shaderCodes.some((c) => c.includes('fn main') && !c.includes('rpt_result_in'))).toBe(true);

    // The reservoir ping-pong + spatial + result + params buffers were allocated.
    expect(rec.bufferLabels).toContain('vitrum.pt-webgpu.restirPt.reservoir.cur');
    expect(rec.bufferLabels).toContain('vitrum.pt-webgpu.restirPt.reservoir.prev');
    expect(rec.bufferLabels).toContain('vitrum.pt-webgpu.restirPt.reservoir.spatial');
    expect(rec.bufferLabels).toContain('vitrum.pt-webgpu.restirPt.result');
    expect(rec.bufferLabels).toContain('vitrum.pt-webgpu.restirPt.params');

    // The reuse group-0 bind group was built.
    expect(rec.bindGroupLabels).toContain('vitrum.pt-webgpu.restirPt.bindgroup0');

    // The capability flag is advertised.
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-restir-pt-reuse')).toBe(true);

    // The debug result buffer is exposed.
    const buf = (engine as unknown as { getRestirPtResultBuffer(): unknown }).getRestirPtResultBuffer();
    expect(buf).not.toBeNull();
    engine.dispose();
  });

  it('ON: cached same-size reservoirs still dispatch reuse on later frames', async () => {
    const rec = emptyRecorder();
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      restirPtReuse: true,
    });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(16));
    engine.renderFrame(frameInput(16));

    expect(rec.computePassLabels.filter((l) => l === 'vitrum.pt-webgpu.restirPt.produce').length).toBe(2);
    expect(rec.computePassLabels.filter((l) => l === 'vitrum.pt-webgpu.restirPt.temporal').length).toBe(2);
    expect(rec.computePassLabels.filter((l) => l === 'vitrum.pt-webgpu.restirPt.spatial').length).toBe(2);
    expect(rec.computePassLabels.filter((l) => l === 'vitrum.pt-webgpu.restirPt.resolve').length).toBe(2);
    engine.dispose();
  });

  it('ON: over-budget reservoirs skip the reuse setup for that frame', async () => {
    const rec = emptyRecorder();
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      restirPtReuse: true,
    });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(2048));

    expect(rec.bufferLabels).not.toContain('vitrum.pt-webgpu.restirPt.reservoir.cur');
    expect(rec.bufferLabels).not.toContain('vitrum.pt-webgpu.restirPt.reservoir.prev');
    expect(rec.pipelineEntryPoints).not.toContain('restirPtProduce');
    expect(rec.computePassLabels).not.toContain('vitrum.pt-webgpu.restirPt.produce');
    expect(warnSpy.mock.calls.flat().map(String).some((m) => m.includes('Skipping ReSTIR-PT reuse'))).toBe(true);

    warnSpy.mockRestore();
    engine.dispose();
  });

  it('ON: the three reuse pipelines were compiled from DISTINCT per-pass modules (no combined unit)', async () => {
    const rec = emptyRecorder();
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      restirPtReuse: true,
    });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(16));

    // Each reuse module must contain exactly ONE @compute entry point (the
    // per-pass split — the combined unit's duplicate @group/@binding slots would
    // not compile). The producer module has restirPtProduce and NOT the others.
    const producerModules = rec.shaderCodes.filter((c) => c.includes('fn restirPtProduce'));
    expect(producerModules.length).toBeGreaterThanOrEqual(1);
    for (const code of producerModules) {
      expect(code).toContain('fn restirPtProduce');
      expect(code).not.toContain('fn restirPtTemporal(');
      expect(code).not.toContain('fn restirPtSpatial(');
      expect(code).not.toContain('fn restirPtResolve(');
    }
    // The spatial module has restirPtSpatial and NOT the others.
    const spatialModules = rec.shaderCodes.filter((c) => c.includes('fn restirPtSpatial('));
    expect(spatialModules.length).toBeGreaterThanOrEqual(1);
    for (const code of spatialModules) {
      expect(code).not.toContain('fn restirPtProduce(');
      expect(code).not.toContain('fn restirPtResolve(');
    }
    engine.dispose();
  });

  it('ON: lite-tier request throws instead of silently disabling reuse', async () => {
    const rec = emptyRecorder();
    await expect(createPTEngine_WebGPU({
      // Force the lite tier explicitly; reuse must stay inert.
      device: makeFullTierDevice(rec),
      traceTier: 'lite',
      restirPtReuse: true,
    })).rejects.toThrow(/restirPtReuse requires traceTier "full"/);
  });

  it('ON: full-tier request throws if the device was not acquired with the reuse buffer floor', async () => {
    const rec = emptyRecorder();
    // 31 = PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE (N-directional +1: was 30, now 31).
    // 35 = PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE (31 + 4 RPT).
    // Use 31 so the device is full-tier (31 ≥ 31) but below the restirPtReuse floor (31 < 35),
    // ensuring the thrown error is the buffer-floor error, not the lite-tier error.
    const device = {
      ...makeFullTierDevice(rec),
      limits: { maxStorageBuffersPerShaderStage: 31, maxStorageTexturesPerShaderStage: 8 },
    } as unknown as GPUDevice;
    await expect(createPTEngine_WebGPU({
      device,
      restirPtReuse: true,
    })).rejects.toThrow(/restirPtReuse requires maxStorageBuffersPerShaderStage >= 35/);
  });
});

describe('ReSTIR-PT reuse compose — per-pass relocation to @group(0)', () => {
  it('relocates @group(4) reuse bindings onto @group(0) high bindings (portable, ≤4 groups)', () => {
    const base = RPT_GROUP0_BINDING_BASE; // 20
    for (const compose of [
      composeRestirPtProducerWgsl,
      composeRestirPtTemporalWgsl,
      composeRestirPtResolveWgsl,
    ]) {
      const wgsl = compose();
      // No @group(4) binding DECLARATION survives (the relocation rewrote them).
      expect(wgsl).not.toMatch(/@group\(4\)\s+@binding\(\d+\)/);
      // The relocated bindings live in @group(0) at >= base. At least the params
      // binding (b4 → base+4) is present in every pass.
      expect(wgsl).toContain(`@group(0) @binding(${base + 4})`);
    }
  });

  it('each per-pass module contains exactly ONE @compute entry point', () => {
    const producer = composeRestirPtProducerWgsl();
    const temporal = composeRestirPtTemporalWgsl();
    const resolve = composeRestirPtResolveWgsl();
    const countCompute = (s: string): number =>
      (s.match(/@compute @workgroup_size\(8, 8, 1\)/g) ?? []).length;
    expect(countCompute(producer)).toBe(1);
    expect(countCompute(temporal)).toBe(1);
    expect(countCompute(resolve)).toBe(1);
    expect(producer).toContain('fn restirPtProduce(');
    expect(temporal).toContain('fn restirPtTemporal(');
    expect(resolve).toContain('fn restirPtResolve(');
  });

  it('monomorphises the reservoir helpers — NO storage-pointer function parameter survives (the naga gap)', () => {
    // naga (dzn/lavapipe) rejects a `ptr<storage,…>` function parameter outright
    // (the non-core unrestricted_pointer_parameters feature is unavailable). The
    // compose helper monomorphises loadReservoirPTHero_{ro,rw} / store… so the
    // module-scope reservoir global is indexed DIRECTLY. Assert no storage-ptr
    // parameter remains, the original ptr-param helper defs are gone, and a
    // specialized `__rpt_*` variant exists. (The naga compile gate
    // wsl-gpu/scripts/restir-pt-compile-gate.ts is the executed proof; this pins
    // the invariant in CI so a compose regression is caught without a GPU.)
    for (const compose of [
      composeRestirPtProducerWgsl,
      composeRestirPtTemporalWgsl,
      composeRestirPtResolveWgsl,
    ]) {
      const wgsl = compose();
      expect(wgsl).not.toMatch(/ptr<storage,/); // no storage-ptr anywhere (params or otherwise)
      expect(wgsl).not.toMatch(/fn (?:load|store)ReservoirPTHero_(?:ro|rw)\(buf:/); // originals removed
      expect(wgsl).not.toMatch(/ReservoirPTHero_(?:ro|rw)\(&rpt_/); // no old-style call sites
      expect(wgsl).toMatch(/fn (?:load|store)ReservoirPTHero_\w+__rpt_\w+\(/); // a monomorphised variant exists
    }
  });

  it('the relocated bindings do NOT collide with the megakernel group-0 bindings (0..13)', () => {
    // The reuse bindings start at 20; assert none landed in 0..13.
    for (const compose of [
      composeRestirPtProducerWgsl,
      composeRestirPtTemporalWgsl,
      composeRestirPtResolveWgsl,
    ]) {
      const wgsl = compose();
      // Find the reuse-var declarations and confirm their binding >= 20.
      const reuseDecls = wgsl.match(/@group\(0\) @binding\((\d+)\) var<[^>]*> rpt_\w+/g) ?? [];
      expect(reuseDecls.length).toBeGreaterThanOrEqual(1);
      for (const decl of reuseDecls) {
        const b = Number(decl.match(/@binding\((\d+)\)/)![1]);
        expect(b).toBeGreaterThanOrEqual(RPT_GROUP0_BINDING_BASE);
      }
    }
  });
});
