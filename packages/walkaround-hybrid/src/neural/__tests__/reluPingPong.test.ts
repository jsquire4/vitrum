/**
 * reluPingPong.test.ts — H28 enforcement test.
 *
 * Verifies that after `allocateGraph`, NO bind group assigns the same GPU buffer
 * to BOTH a read-only binding (0 = input) AND a read-write binding (3 = output)
 * for any relu layer.
 *
 * Before H28, in-place relu layers had `inputs: ['enc1_feat'], output: 'enc1_feat'`,
 * causing the same buffer to appear at binding 0 (read) and binding 3 (read_write)
 * — undefined behavior in WebGPU that can produce stale reads in any one-pass
 * compute execution.
 *
 * H28 fix in layerResourceAllocator.ts: for in-place relu layers, a distinct
 * `${layer.name}_out` buffer is allocated for binding 3 and the tensors map is
 * updated so downstream layers see the relu-written output.
 *
 * This test uses a minimal stub GPUDevice that records all createBuffer calls and
 * createBindGroup entries — no real GPU required.
 */

import { describe, it, expect } from 'vitest';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';
import { allocateGraph } from '../layerResourceAllocator.js';

// Install WebGPU global constants (GPUBufferUsage etc.) needed by allocateGraph.
installWebGPUPolyfills();
import { buildUNetSpec } from '../unetArchitecture.js';
import type { ModelWeights, LayerWeights } from '../weights.js';
import { computeTensorDims } from '../tensorDimSolver.js';

// ── Stub GPUDevice ────────────────────────────────────────────────────────────

interface StubBuffer {
  label: string;
  id: number;
}

interface BindGroupEntry {
  binding: number;
  buffer: StubBuffer;
}

interface RecordedBindGroup {
  label: string;
  entries: BindGroupEntry[];
}

function makeStubDevice() {
  let nextBufId = 0;
  const allBuffers: StubBuffer[] = [];
  const bindGroups: RecordedBindGroup[] = [];

  const device = {
    createBuffer(desc: { label?: string; size: number; usage: number; mappedAtCreation?: boolean }): StubBuffer {
      const buf = { label: desc.label ?? '', id: nextBufId++ };
      allBuffers.push(buf);
      if (desc.mappedAtCreation) {
        // Provide getMappedRange / unmap stubs so weight uploads don't throw.
        const ab = new ArrayBuffer(desc.size);
        (buf as unknown as Record<string, unknown>).getMappedRange = () => ab;
        (buf as unknown as Record<string, unknown>).unmap = () => {};
      }
      return buf;
    },
    createShaderModule(_desc: unknown) {
      return {};
    },
    async createComputePipelineAsync(_desc: unknown) {
      // Return a stub pipeline with a stub getBindGroupLayout.
      return {
        getBindGroupLayout(_group: number) {
          return {};
        },
      };
    },
    createBindGroup(desc: { label?: string; layout: unknown; entries: Array<{ binding: number; resource: { buffer: unknown } }> }): RecordedBindGroup {
      const entries: BindGroupEntry[] = desc.entries.map(e => ({
        binding: e.binding,
        buffer: e.resource.buffer as StubBuffer,
      }));
      const bg: RecordedBindGroup = { label: desc.label ?? '', entries };
      bindGroups.push(bg);
      return bg;
    },
    queue: {
      writeBuffer(_buf: unknown, _off: number, _data: unknown) {},
    },
  } as unknown as GPUDevice;

  return { device, allBuffers, bindGroups };
}

// ── Stub ModelWeights ─────────────────────────────────────────────────────────

function makeStubWeights(spec: ReturnType<typeof buildUNetSpec>): ModelWeights {
  const layers: LayerWeights[] = spec.layers
    .filter(l => l.kind === 'conv2d' || l.kind === 'transposedConv2d')
    .map(l => ({
      name: l.name,
      weights: new Float32Array(4), // minimal non-empty weight
      biases:  new Float32Array(4),
    }));
  return { layers };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('H28 — relu ping-pong (no aliased read+read_write bindings)', () => {
  it('after allocateGraph, no bind group has the same buffer at binding 0 (read) AND binding 3 (read_write)', async () => {
    const spec = buildUNetSpec();
    const W = 64, H = 64;
    const dimMap = computeTensorDims(spec, W, H);
    const weights = makeStubWeights(spec);
    const { device, bindGroups } = makeStubDevice();

    await allocateGraph(device, spec, weights, W, H, dimMap);

    // Check every recorded bind group.
    const violations: string[] = [];
    for (const bg of bindGroups) {
      const b0 = bg.entries.find(e => e.binding === 0);
      const b3 = bg.entries.find(e => e.binding === 3);
      if (b0 && b3 && b0.buffer === b3.buffer) {
        violations.push(`Bind group "${bg.label}": binding 0 === binding 3 (buf id=${(b0.buffer).id}, label="${b0.buffer.label}")`);
      }
    }

    expect(violations).toHaveLength(0);
  });

  it('relu layers produce distinct input and output buffers in their bind groups', async () => {
    const spec = buildUNetSpec();
    const W = 32, H = 32;
    const dimMap = computeTensorDims(spec, W, H);
    const weights = makeStubWeights(spec);
    const { device, bindGroups } = makeStubDevice();

    await allocateGraph(device, spec, weights, W, H, dimMap);

    // Find bind groups for relu layers (label contains 'relu').
    const reluBgs = bindGroups.filter(bg => bg.label.includes('relu'));
    // There must be at least one relu layer in the default UNet spec.
    expect(reluBgs.length).toBeGreaterThan(0);

    for (const bg of reluBgs) {
      const b0 = bg.entries.find(e => e.binding === 0);
      const b3 = bg.entries.find(e => e.binding === 3);
      expect(b0).toBeDefined();
      expect(b3).toBeDefined();
      if (b0 && b3) {
        // After H28 fix: input and output buffers must be DISTINCT objects.
        expect(b0.buffer).not.toBe(b3.buffer);
      }
    }
  });
});
