/**
 * restirPtReuseWiring.test.ts — wiring contract for ReSTIR-PT
 * reservoir/reuse pre-passes (gpuResources.ts + index.ts).
 *
 * The load-bearing invariant of this increment is OFF-BYTE-IDENTITY: with the
 * `restirPtReuse` flag OFF (the default), NONE of the reuse resources/pipelines
 * are created and the default megakernel render is unchanged. ON allocates the
 * two compact reservoirs + result + params and creates four reuse pipelines.
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
import {
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
} from '../webgpuLimits.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import {
  composeRestirPtProducerWgsl,
  composeRestirPtTemporalWgsl,
  composeRestirPtSpatialWgsl,
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
  bindGroupBufferEntries: Array<{
    label: string;
    entries: Array<{ binding: number; bufferLabel: string | null }>;
  }>;
  computePassLabels: string[];
  clearBufferLabels: string[];
  bufferWrites: Array<{ label: string; bytes: Uint8Array }>;
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
    clearBuffer: vi.fn((buffer?: { label?: string }) => {
      rec.clearBufferLabels.push(buffer?.label ?? '');
    }),
    copyBufferToBuffer: vi.fn(),
    finish: vi.fn(() => ({})),
  };
  return {
    queue: {
      writeBuffer: vi.fn((buffer?: { label?: string }, _offset?: number, data?: BufferSource) => {
        let bytes = new Uint8Array();
        if (data instanceof ArrayBuffer) {
          bytes = new Uint8Array(data.slice(0));
        } else if (ArrayBuffer.isView(data)) {
          bytes = new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
        }
        rec.bufferWrites.push({ label: buffer?.label ?? '', bytes });
      }),
      writeTexture: vi.fn(),
      submit: vi.fn(),
    },
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
    createBindGroup: vi.fn((desc?: {
      label?: string;
      entries?: Array<{
        binding: number;
        resource: { buffer?: { label?: string } } | object;
      }>;
    }) => {
      rec.bindGroupLabels.push(desc?.label ?? '');
      rec.bindGroupBufferEntries.push({
        label: desc?.label ?? '',
        entries: (desc?.entries ?? []).map((entry) => ({
          binding: entry.binding,
          bufferLabel:
            'buffer' in entry.resource
              ? entry.resource.buffer?.label ?? null
              : null,
        })),
      });
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
  return {
    shaderCodes: [],
    pipelineEntryPoints: [],
    bufferLabels: [],
    bindGroupLabels: [],
    bindGroupBufferEntries: [],
    computePassLabels: [],
    clearBufferLabels: [],
    bufferWrites: [],
  };
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
    viewport: { width: size, height: size, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 1,
    quality: { samplesTarget: 4, bounces: 2, resolutionFactor: 1 },
  };
}

function latestRestirPtParamsWrite(rec: Recorder): ArrayBuffer {
  const write = rec.bufferWrites.filter((w) => w.label === 'vitrum.pt-webgpu.restirPt.params').at(-1);
  expect(write, 'ReSTIR-PT params UBO write').toBeDefined();
  return write!.bytes.buffer.slice(
    write!.bytes.byteOffset,
    write!.bytes.byteOffset + write!.bytes.byteLength,
  ) as ArrayBuffer;
}

describe('ReSTIR-PT reuse wiring — OFF by default (byte-identity)', () => {
  it('default-trace WGSL is untouched (the composed string is NOT modified by the reuse wiring)', () => {
    // The reuse path composes SEPARATE per-pass modules; it must never mutate the
    // default megakernel string. This is a cheap guard alongside the SHA pin in
    // wgslContract.test.ts (which is the authoritative byte-identity check).
    expect(PT_WEBGPU_TRACE_WGSL.length).toBeGreaterThan(100_000);
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
    expect(engine.capabilities.activeFeatures?.has('pt-webgpu-one-edge-gris-reconnection')).toBe(false);
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
  it('ON: a full-tier render creates the 4 reuse pipelines + two reservoir/result/params buffers', async () => {
    const rec = emptyRecorder();
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      oneEdgeReconnectionReuse: true,
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

    // Exactly two full-frame reservoirs + result + params were allocated.
    expect(rec.bufferLabels).toContain('vitrum.pt-webgpu.restirPt.reservoir.cur');
    expect(rec.bufferLabels).toContain('vitrum.pt-webgpu.restirPt.reservoir.prev');
    expect(rec.bufferLabels).not.toContain('vitrum.pt-webgpu.restirPt.reservoir.spatial');
    expect(rec.bufferLabels).toContain('vitrum.pt-webgpu.restirPt.result');
    expect(rec.bufferLabels).toContain('vitrum.pt-webgpu.restirPt.params');

    // Per-pass bind groups prevent active read/write aliases.
    for (const pass of ['producer', 'temporal', 'spatial', 'resolve']) {
      expect(rec.bindGroupLabels).toContain(
        `vitrum.pt-webgpu.restirPt.bindgroup0.${pass}`,
      );
    }
    // WebGPU usage validation covers every explicit-layout entry in a dispatch,
    // including shader-unused slots. The read-only history slot must therefore
    // never alias any read_write reuse slot in the same bind group.
    const base = RPT_GROUP0_BINDING_BASE;
    for (const pass of ['producer', 'temporal', 'spatial', 'resolve']) {
      const group = rec.bindGroupBufferEntries.find(
        (candidate) =>
          candidate.label ===
          `vitrum.pt-webgpu.restirPt.bindgroup0.${pass}`,
      );
      expect(group, `${pass} reuse bind group`).toBeDefined();
      const labelAt = (binding: number): string | null | undefined =>
        group!.entries.find((entry) => entry.binding === binding)?.bufferLabel;
      const readOnlyLabel = labelAt(base + 2);
      expect(readOnlyLabel).toBeTruthy();
      for (const readWriteBinding of [base, base + 1, base + 3, base + 5]) {
        expect(
          labelAt(readWriteBinding),
          `${pass} b${readWriteBinding} must not alias read-only b${base + 2}`,
        ).not.toBe(readOnlyLabel);
      }
    }
    // Runtime selection is reported through the typed active feature set.
    expect(engine.capabilities.activeFeatures?.has('pt-webgpu-one-edge-gris-reconnection')).toBe(true);

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

  it('keeps restirPtReuse as an exact compatibility alias', async () => {
    const rec = emptyRecorder();
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      restirPtReuse: true,
    });
    expect(engine.capabilities.activeFeatures?.has(
      'pt-webgpu-one-edge-gris-reconnection',
    )).toBe(true);
    engine.dispose();
  });

  it('rejects transmissive scenes before scene or reuse-resource publication', async () => {
    const rec = emptyRecorder();
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      oneEdgeReconnectionReuse: true,
    });
    const scene = makeScene();
    const transmissive: Scene = {
      ...scene,
      primitives: scene.primitives.map((primitive) => ({
        ...primitive,
        material: {
          ...primitive.material,
          transmission: 0.75,
        },
      })),
    };
    const buffersBefore = rec.bufferLabels.length;
    expect(() => engine.setScene(transmissive)).toThrow(
      /finite opaque one-edge reconnection only/,
    );
    expect(rec.bufferLabels).toHaveLength(buffersBefore);
    expect(rec.pipelineEntryPoints).not.toContain('restirPtProduce');
    engine.dispose();
  });

  it('rejects a transmissive material mutation before publication', async () => {
    const rec = emptyRecorder();
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      oneEdgeReconnectionReuse: true,
    });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(16));
    const clearsBefore = rec.clearBufferLabels.length;
    expect(() => engine.updatePrimitive!('mesh-a', {
      material: {
        baseColor: [0.8, 0.2, 0.1],
        roughness: 0.3,
        metallic: 0.1,
        transmission: 1,
      },
    })).toThrow(/finite opaque one-edge reconnection only/);
    expect(rec.clearBufferLabels).toHaveLength(clearsBefore);
    expect(engine.getScene!()?.primitives[0]?.material.transmission ?? 0).toBe(0);
    engine.dispose();
  });

  it('reset and resize clear or replace both temporal reservoirs', async () => {
    const rec = emptyRecorder();
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      oneEdgeReconnectionReuse: true,
    });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(16));
    rec.clearBufferLabels.length = 0;
    engine.reset();
    expect(rec.clearBufferLabels).toContain('vitrum.pt-webgpu.restirPt.reservoir.cur');
    expect(rec.clearBufferLabels).toContain('vitrum.pt-webgpu.restirPt.reservoir.prev');

    const curAllocationsBefore = rec.bufferLabels.filter(
      (label) => label === 'vitrum.pt-webgpu.restirPt.reservoir.cur',
    ).length;
    engine.renderFrame(frameInput(8));
    expect(rec.bufferLabels.filter(
      (label) => label === 'vitrum.pt-webgpu.restirPt.reservoir.cur',
    )).toHaveLength(curAllocationsBefore + 1);
    engine.dispose();
  });

  it('ON: params contain only the temporal confidence clamp and alignment padding', async () => {
    const rec = emptyRecorder();
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      restirPtReuse: true,
    });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(16));

    const u = new Uint32Array(latestRestirPtParamsWrite(rec));
    expect(u[0]).toBe(20);
    expect(u[1]).toBe(0);
    expect(u[2]).toBe(0);
    expect(u[3]).toBe(0);
    expect(u).toHaveLength(4);

    engine.dispose();
  });

  it.each([
    ['mClamp', NaN],
    ['mClamp', Infinity],
    ['mClamp', 0],
    ['mClamp', 1.5],
    ['mClamp', 4096],
  ] as const)('ON: rejects invalid explicit ReSTIR-PT %s=%s', async (key, value) => {
    await expect(createPTEngine_WebGPU({
      device: makeFullTierDevice(emptyRecorder()),
      restirPtReuse: true,
      restirPtReuseOptions: { [key]: value },
    })).rejects.toThrow(`restirPtReuseOptions.${key}`);
  });

  it('rejects unknown ReSTIR-PT tuning keys instead of silently ignoring typos', async () => {
    await expect(createPTEngine_WebGPU({
      device: makeFullTierDevice(emptyRecorder()),
      restirPtReuse: true,
      restirPtReuseOptions: { mClmap: 20 } as never,
    })).rejects.toThrow('restirPtReuseOptions contains unknown key(s): mClmap');
  });

  it('rejects non-empty legacy tuning when reuse is disabled', async () => {
    await expect(createPTEngine_WebGPU({
      device: makeFullTierDevice(emptyRecorder()),
      restirPtReuseOptions: { mClamp: 20 },
    })).rejects.toThrow('non-empty restirPtReuseOptions requires oneEdgeReconnectionReuse:true');
  });

  it('rejects the removed biased wCap option as an unknown key', async () => {
    await expect(createPTEngine_WebGPU({
      device: makeFullTierDevice(emptyRecorder()),
      restirPtReuse: true,
      restirPtReuseOptions: { wCap: 10 } as never,
    })).rejects.toThrow('restirPtReuseOptions contains unknown key(s): wCap');
  });

  it('ON: finite glossy reflection is part of the stable path', async () => {
    const rec = emptyRecorder();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      restirPtReuse: true,
    });

    warn.mockRestore();
    engine.dispose();
  });

  it('ON: over-budget reservoirs fail explicitly before reuse setup', async () => {
    const rec = emptyRecorder();
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      restirPtReuse: true,
    });
    engine.setScene(makeScene());
    expect(() => engine.renderFrame(frameInput(2048))).toThrow(
      /requires 256\.00 MiB per reservoir/,
    );

    expect(rec.bufferLabels).not.toContain('vitrum.pt-webgpu.restirPt.reservoir.cur');
    expect(rec.bufferLabels).not.toContain('vitrum.pt-webgpu.restirPt.reservoir.prev');
    expect(rec.pipelineEntryPoints).not.toContain('restirPtProduce');
    expect(rec.computePassLabels).not.toContain('vitrum.pt-webgpu.restirPt.produce');
    expect(warnSpy.mock.calls.flat().map(String).some((m) => m.includes('256.00 MiB per reservoir'))).toBe(true);

    warnSpy.mockRestore();
    engine.dispose();
  });

  it('ON: the four reuse pipelines were compiled from DISTINCT per-pass modules (no combined unit)', async () => {
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
    })).rejects.toThrow(/oneEdgeReconnectionReuse requires traceTier "full"/);
  });

  it('ON: full-tier request throws if the device was not acquired with the reuse buffer floor', async () => {
    const rec = emptyRecorder();
    // Use the full-tier floor so the device is full-tier but below the
    // restirPtReuse floor, ensuring the thrown error is the buffer-floor error,
    // not the lite-tier error.
    const device = {
      ...makeFullTierDevice(rec),
      limits: {
        maxStorageBuffersPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
        maxStorageTexturesPerShaderStage: 8,
      },
    } as unknown as GPUDevice;
    await expect(createPTEngine_WebGPU({
      device,
      restirPtReuse: true,
    })).rejects.toThrow(
      new RegExp(
        `oneEdgeReconnectionReuse requires maxStorageBuffersPerShaderStage >= ${PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE}`,
      ),
    );
  });
});

describe('ReSTIR-PT reuse compose — per-pass relocation to @group(0)', () => {
  it('relocates @group(4) reuse bindings onto @group(0) high bindings (portable, ≤4 groups)', () => {
    const base = RPT_GROUP0_BINDING_BASE; // 20
    for (const compose of [
      composeRestirPtProducerWgsl,
      composeRestirPtTemporalWgsl,
      composeRestirPtSpatialWgsl,
      composeRestirPtResolveWgsl,
    ]) {
      const wgsl = compose();
      // No @group(4) binding DECLARATION survives (the relocation rewrote them).
      expect(wgsl).not.toMatch(/@group\(4\)\s+@binding\(\d+\)/);
      const relocatedBindings = Array.from({ length: 6 }, (_, i) => base + i).join('|');
      expect(wgsl).toMatch(
        new RegExp(`@group\\(0\\) @binding\\((?:${relocatedBindings})\\)`),
      );
    }
    // Only temporal consumes the confidence clamp. Producer dimensions come
    // from FrameParams; spatial and resolve need no pass-local uniforms.
    expect(composeRestirPtTemporalWgsl()).toContain(`@group(0) @binding(${base + 4})`);
    expect(composeRestirPtProducerWgsl()).not.toContain(`@group(0) @binding(${base + 4})`);
    expect(composeRestirPtSpatialWgsl()).not.toContain(`@group(0) @binding(${base + 4})`);
    expect(composeRestirPtResolveWgsl()).not.toContain(`@group(0) @binding(${base + 4})`);
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
