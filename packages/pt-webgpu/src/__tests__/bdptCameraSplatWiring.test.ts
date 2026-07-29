import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BDPT_EXPLICIT_STRATEGY_MASK_WGSL } from '@vitrum/shared-samplers';
import { PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL } from '../wgsl/bdpt/bdptCameraSplat.wgsl.js';
import {
  PT_WEBGPU_BDPT_CONNECTION_WGSL,
  composePtWebgpuBdptConnectionWgsl,
} from '../wgsl/bdpt/bdptConnection.wgsl.js';
import {
  PT_WEBGPU_TRACE_WGSL,
  composePtWebgpuCompositeTraceWgsl,
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';
import {
  PT_WEBGPU_BDPT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
} from '../webgpuLimits.js';
import { PT_WEBGPU_FULL_SUPPORT_MANIFEST } from '../supportManifest.js';

const BDPT_OFF_SOURCE_SHA256 =
  // Re-pinned 2026-07-29: U11 removes every overwritten continuous-event
  // proposal-local throughput/PDF calculation from the shared sampler.
  '83c30898ae14f65773b25139456a2a2235203452e97bb18ec8e0313d7d4c1451';

function source(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), 'utf8');
}

function countDistinctStorageBufferBindings(wgsl: string): number {
  const bindings = new Set<string>();
  for (const match of wgsl.matchAll(
    /@group\((\d+)\)\s*@binding\((\d+)\)\s*var<storage/g,
  )) {
    bindings.add(`${match[1]}:${match[2]}`);
  }
  return bindings.size;
}

describe('BDPT t=1 camera-splat production wiring', () => {
  it('keeps the BDPT-off shader/resource surface unchanged', () => {
    const off = composePtWebgpuTraceWgsl(false);
    expect(off).toBe(PT_WEBGPU_TRACE_WGSL);
    expect(createHash('sha256').update(off).digest('hex')).toBe(
      BDPT_OFF_SOURCE_SHA256,
    );
    expect(off).not.toContain('bdptCameraSplatBuffer');
    expect(off).not.toContain('bdptResolveCameraSplats');
    expect(off).not.toContain('bdptAccumulateCameraSplatStrategies');
    expect(off).not.toContain('fn bdptExplicitConnectionStrategyIsValid(');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'var validExplicitStrategy = k >= 1u && k <= n - 2u;',
    );
    expect(composePtWebgpuBdptConnectionWgsl(true)).toContain(
      'fn bdptExplicitConnectionStrategyIsValid(',
    );
    expect(countDistinctStorageBufferBindings(off)).toBe(
      PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    );
  });

  it('composes the atomic t=1 strategy and resolver only for bdpt:true', () => {
    const on = composePtWebgpuTraceWgsl(true);
    expect(on).toContain(
      '@group(0) @binding(14) var<storage, read_write>\n  bdptCameraSplatBuffer',
    );
    expect(on).toContain('atomicCompareExchangeWeak(');
    expect(on).toContain('fn bdptAccumulateCameraSplatStrategies(');
    expect(on).toContain(
      'bdptAccumulateCameraSplatStrategies(gid.xy, heroPdf);',
    );
    expect(on).toContain('fn bdptResolveCameraSplats(');
    expect(on).toContain(
      'bdptCameraDirectionalPdfForDirection(ray.direction)',
    );
    expect(on).toContain('bdptStageCameraSample(');
    expect(on.match(/bdptStageCameraSample\(/g)).toHaveLength(2);
    // The legacy helper remains declared for source stability, but BDPT main no
    // longer calls it (definition only): the resolver owns persistent writes.
    expect(on.match(/accumulateFrame\(/g)).toHaveLength(1);
    expect(countDistinctStorageBufferBindings(on)).toBe(
      PT_WEBGPU_BDPT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    );
  });

  it('indexes the global atomic buffer without a storage-pointer parameter', () => {
    // Core WGSL does not permit storage-address-space pointers as function
    // parameters without the optional unrestricted_pointer_parameters feature.
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).not.toContain(
      'ptr<storage, atomic<u32>, read_write>',
    );
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).toContain(
      'atomicLoad(&bdptCameraSplatBuffer[wordIndex])',
    );
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).toContain(
      '&bdptCameraSplatBuffer[wordIndex], expected, bitcast<u32>(nextValue)',
    );
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).toContain(
      'bdptAtomicAddFiniteF32(base, value.x);',
    );
  });

  it('gives the composite megakernel the same atomic target at binding 14', () => {
    const composite = composePtWebgpuCompositeTraceWgsl(true);
    expect(composite).toContain('bdptCameraSplatBuffer');
    expect(composite).toContain('bdptAccumulateCameraSplatStrategies(gid.xy, heroPdf)');
    expect(composite).toContain('@group(0) @binding(23)');
  });

  it('pins the PBRT camera measure, finite-value CAS, and selected t=1 topology', () => {
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).toContain(
      '1.0 / (camera.imagePlaneArea * cosTheta * cosTheta * cosTheta)',
    );
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).toContain(
      'fn bdptCameraDirectionalPdfForDirection(direction: vec3f) -> f32',
    );
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).toContain(
      'cross(near10 - near00, near01 - near00)',
    );
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).toContain(
      'result.sampleWiOverPdf = result.cameraDirectionalPdf / distanceSquared',
    );
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).toContain('let n = c + 2u;');
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).toContain(
      'let selectedS = c + 1u;',
    );
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).toContain(
      'projection.cameraDirectionalPdf,\n    revLcMinus,',
    );
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).toContain(
      'lv2.xyz * lightScatter * surfaceCosine *',
    );
    // lv0.w records the sampled continuation from L_c to L_{c+1}. That edge
    // is outside the t=1 path ending at L_c, so a delta continuation cannot
    // suppress the independently evaluated finite L_c-to-camera connection.
    expect(PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL).not.toContain(
      'lv0.w == BDPT_KIND_DELTA',
    );
  });

  it('allocates, binds, clears, dispatches, and disposes the camera-splat cohort', () => {
    const gpu = source('../gpuResources.ts');
    const engine = source('../index.ts');
    expect(gpu).toContain('bdptCameraSplatBuffer: GPUBuffer | null = null');
    expect(gpu).toContain(
      "label: 'vitrum.pt-webgpu.bdpt.cameraSplats.buffer'",
    );
    expect(gpu).toContain('binding: 14');
    expect(gpu).toContain("entryPoint: 'bdptResolveCameraSplats'");
    expect(gpu).toContain('this.bdptCameraSplatBuffer?.destroy()');
    const clearAt = engine.indexOf(
      'encoder.clearBuffer(gpu.bdptCameraSplatBuffer);',
    );
    const traceAt = engine.indexOf(
      "label: 'vitrum.pt-webgpu.pathTrace.pass'",
    );
    const resolveAt = engine.indexOf(
      "label: 'vitrum.pt-webgpu.bdpt.cameraSplatResolve.pass'",
    );
    expect(clearAt).toBeGreaterThan(0);
    expect(traceAt).toBeGreaterThan(clearAt);
    expect(resolveAt).toBeGreaterThan(traceAt);
    expect(engine).toContain(
      'bdptCameraSplatBufBytes + scene.bufferBytes',
    );
  });

  it('admits s=n-1 in the canonical CPU+WGSL bounded-strategy mask', () => {
    expect(BDPT_EXPLICIT_STRATEGY_MASK_WGSL).toContain(
      'strategyS < 1u || strategyS >= pathVertexCount',
    );
    expect(BDPT_EXPLICIT_STRATEGY_MASK_WGSL).not.toContain(
      'strategyS > pathVertexCount - 2u',
    );
  });

  it('reports the now-executable native camera-splat strategy', () => {
    expect(
      PT_WEBGPU_FULL_SUPPORT_MANIFEST.bidirectionalPathTracing
        ?.cameraSplatStrategy,
    ).toBe('native');
  });
});
