/**
 * inferenceGraphReinit.test.ts — double-initialize leak guard.
 *
 * `InferenceGraph.initialize()` overwrites `_tensors` / `_allocatedBuffers`
 * with a fresh allocation. If it is ever called a SECOND time without an
 * intervening `dispose()`, every GPU buffer from the first allocation would be
 * orphaned (no remaining reference → never `.destroy()`'d → GPU memory leak).
 *
 * The defensive guard at the top of `initialize()` disposes the prior
 * allocation first. This test verifies that on a re-initialize, ALL of the
 * first allocation's buffers are destroyed (no leak), using a stub device that
 * records create/destroy calls — no real GPU required.
 */

import { describe, it, expect } from 'vitest';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';

installWebGPUPolyfills();

import { InferenceGraph } from '../InferenceGraph.js';
import { buildUNetSpec } from '../unetArchitecture.js';
import type { ModelWeights, LayerWeights } from '../weights.js';

interface StubBuffer {
  label: string;
  id: number;
  destroyed: boolean;
}

function makeStubDevice() {
  let nextBufId = 0;
  const allBuffers: StubBuffer[] = [];

  const device = {
    createBuffer(desc: { label?: string; size: number; usage: number; mappedAtCreation?: boolean }): StubBuffer {
      const buf: StubBuffer = { label: desc.label ?? '', id: nextBufId++, destroyed: false };
      (buf as unknown as Record<string, unknown>).destroy = () => { buf.destroyed = true; };
      if (desc.mappedAtCreation) {
        const ab = new ArrayBuffer(desc.size);
        (buf as unknown as Record<string, unknown>).getMappedRange = () => ab;
        (buf as unknown as Record<string, unknown>).unmap = () => {};
      }
      allBuffers.push(buf);
      return buf;
    },
    createShaderModule(_desc: unknown) {
      return {};
    },
    async createComputePipelineAsync(_desc: unknown) {
      return { getBindGroupLayout(_g: number) { return {}; } };
    },
    createBindGroup(_desc: unknown) {
      return {};
    },
    queue: { writeBuffer(_b: unknown, _o: number, _d: unknown) {} },
  } as unknown as GPUDevice;

  return { device, allBuffers };
}

function makeStubWeights(spec: ReturnType<typeof buildUNetSpec>): ModelWeights {
  const layers: LayerWeights[] = spec.layers
    .filter(l => l.kind === 'conv2d' || l.kind === 'transposedConv2d')
    .map(l => ({
      name: l.name,
      weights: new Float32Array(4),
      biases: new Float32Array(4),
    }));
  return { layers };
}

describe('InferenceGraph — double-initialize leak guard', () => {
  it('re-initialize destroys ALL of the first allocation\'s buffers', async () => {
    const spec = buildUNetSpec();
    const weights = makeStubWeights(spec);
    const { device, allBuffers } = makeStubDevice();

    const graph = new InferenceGraph(spec);
    await graph.initialize(device, weights, 32, 32);

    const firstGenBuffers = allBuffers.slice();
    expect(firstGenBuffers.length).toBeGreaterThan(0);
    // None destroyed yet.
    expect(firstGenBuffers.every(b => !b.destroyed)).toBe(true);

    // Second initialize WITHOUT a dispose() — guard must release the first gen.
    await graph.initialize(device, weights, 64, 64);

    // Every first-generation buffer must now be destroyed (no leak).
    const leaked = firstGenBuffers.filter(b => !b.destroyed);
    expect(leaked).toHaveLength(0);

    // The graph remains usable after re-initialize.
    expect(graph.ready).toBe(true);

    graph.dispose();
  });

  it('first initialize on a fresh instance destroys nothing', async () => {
    const spec = buildUNetSpec();
    const weights = makeStubWeights(spec);
    const { device, allBuffers } = makeStubDevice();

    const graph = new InferenceGraph(spec);
    await graph.initialize(device, weights, 32, 32);

    // The guard is a no-op on a never-initialized instance: nothing destroyed.
    expect(allBuffers.some(b => b.destroyed)).toBe(false);
    graph.dispose();
  });
});
