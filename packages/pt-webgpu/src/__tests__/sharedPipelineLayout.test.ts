/**
 * Regression for the WebGPU-conformance bug that blocked BDPT-ON from
 * dispatching on real hardware (lavapipe + dzn + Dawn): both compute pipelines
 * (path-trace `main` + BDPT `bdptExtendLightSubpath`) were created with
 * `layout:'auto'`. Auto-generated bind-group layouts are pipeline-exclusive per
 * the WebGPU spec, so a bind group built against the path-trace pipeline cannot
 * be set on the BDPT pipeline. The BDPT light-subpath pass reuses the path-trace
 * bind groups, so it was rejected at `setBindGroup` with "Exclusive pipelines
 * don't match".
 *
 * The fix gives both pipelines ONE explicit shared `GPUPipelineLayout` built
 * from explicit per-group `GPUBindGroupLayout`s. These tests pin that wiring
 * with a stub device (no real GPU): both pipelines must reference the SAME
 * pipeline-layout object, and the explicit group layouts must match the binding
 * indices / resource types the WGSL `@group/@binding` decls (and the prior auto
 * layout) declared, so the existing bind-group construction stays valid.
 */
import { describe, expect, it, vi } from 'vitest';
import { GpuResources } from '../gpuResources.js';

interface RecordedLayout {
  label?: string | undefined;
  entries: GPUBindGroupLayoutEntry[];
}
interface RecordedPipeline {
  label?: string | undefined;
  layout: unknown;
  entryPoint: string;
}

function installWebGpuConstStubs(): void {
  const g = globalThis as unknown as {
    GPUBufferUsage?: Record<string, number>;
    GPUShaderStage?: Record<string, number>;
  };
  if (g.GPUBufferUsage == null) {
    g.GPUBufferUsage = { STORAGE: 1 << 0, COPY_DST: 1 << 1, UNIFORM: 1 << 2 };
  }
  if (g.GPUShaderStage == null) {
    g.GPUShaderStage = { VERTEX: 1 << 0, FRAGMENT: 1 << 1, COMPUTE: 1 << 2 };
  }
}

function makeStubDevice() {
  const createdLayouts: RecordedLayout[] = [];
  const createdPipelineLayouts: {
    label?: string | undefined;
    bindGroupLayouts: unknown[];
    token: object;
  }[] = [];
  const createdPipelines: RecordedPipeline[] = [];

  // A unique sentinel per explicit GPUBindGroupLayout so we can confirm the
  // pipeline layout is built from them and the right one lands on bindGroupLayout.
  const createBindGroupLayout = vi.fn((desc: RecordedLayout) => {
    const token = { __bgl: desc.label, entries: desc.entries };
    createdLayouts.push(desc);
    return token as unknown as GPUBindGroupLayout;
  });
  const createPipelineLayout = vi.fn((desc: { label?: string; bindGroupLayouts: unknown[] }) => {
    const token = {};
    createdPipelineLayouts.push({ label: desc.label, bindGroupLayouts: desc.bindGroupLayouts, token });
    return token as unknown as GPUPipelineLayout;
  });
  const createComputePipeline = vi.fn(
    (desc: { label?: string; layout: unknown; compute: { entryPoint: string } }) => {
      createdPipelines.push({ label: desc.label, layout: desc.layout, entryPoint: desc.compute.entryPoint });
      return {
        // If anyone falls back to the auto-layout accessor this throws so the
        // test surfaces a regression rather than silently passing.
        getBindGroupLayout: () => {
          throw new Error('getBindGroupLayout must not be used with an explicit pipeline layout');
        },
      } as unknown as GPUComputePipeline;
    },
  );

  const device = {
    createBuffer: vi.fn((desc?: { label?: string }) => ({ label: desc?.label ?? '', destroy: vi.fn() })),
    createShaderModule: vi.fn(() => ({})),
    createBindGroupLayout,
    createPipelineLayout,
    createComputePipeline,
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;

  return {
    device,
    createdLayouts,
    createdPipelineLayouts,
    createdPipelines,
    createBindGroupLayout,
    createPipelineLayout,
  };
}

describe('pt-webgpu shared explicit pipeline layout (BDPT cross-pipeline bind-group fix)', () => {
  installWebGpuConstStubs();

  it('full+bdpt: both pipelines share ONE explicit GPUPipelineLayout (no layout:auto)', () => {
    const stub = makeStubDevice();
    const gpu = new GpuResources(stub.device, 'full', true);
    gpu.ensurePipeline();

    // Exactly one pipeline layout, built from 4 explicit bind-group layouts
    // (WS2 added group 3: the many-light importance-sampling tree buffer).
    expect(stub.createdPipelineLayouts).toHaveLength(1);
    expect(stub.createdPipelineLayouts[0]!.bindGroupLayouts).toHaveLength(4);

    // Two compute pipelines: path-trace `main` + BDPT `bdptExtendLightSubpath`.
    const entryPoints = stub.createdPipelines.map((p) => p.entryPoint).sort();
    expect(entryPoints).toEqual(['bdptExtendLightSubpath', 'main']);

    // BOTH pipelines reference the SAME explicit pipeline-layout object — never
    // the string 'auto'. This is the crux of the fix: shared layout → the same
    // bind groups can be set on both pipelines without "exclusive" rejection.
    const sharedLayout = stub.createdPipelineLayouts[0]!.token;
    for (const p of stub.createdPipelines) {
      expect(p.layout).toBe(sharedLayout);
      expect(p.layout).not.toBe('auto');
    }
  });

  it('full: bindGroupLayout/1/2/3 are the explicit layouts the shared pipeline layout uses', () => {
    const stub = makeStubDevice();
    const gpu = new GpuResources(stub.device, 'full', true);
    gpu.ensurePipeline();

    expect(gpu.bindGroupLayout).not.toBeNull();
    expect(gpu.bindGroupLayout1).not.toBeNull();
    expect(gpu.bindGroupLayout2).not.toBeNull();
    expect(gpu.bindGroupLayout3).not.toBeNull(); // WS2 light tree

    const pl = stub.createdPipelineLayouts[0]!.bindGroupLayouts;
    expect(pl[0]).toBe(gpu.bindGroupLayout);
    expect(pl[1]).toBe(gpu.bindGroupLayout1);
    expect(pl[2]).toBe(gpu.bindGroupLayout2);
    expect(pl[3]).toBe(gpu.bindGroupLayout3);
  });

  it('full: explicit group layouts match the WGSL binding indices/types the auto layout produced', () => {
    const stub = makeStubDevice();
    const gpu = new GpuResources(stub.device, 'full', true);
    gpu.ensurePipeline();

    // Group order is [group0, group1, group2, group3] as pushed into the layout.
    const [g0, g1, g2, g3] = stub.createdLayouts;
    const COMPUTE = (globalThis as unknown as { GPUShaderStage: { COMPUTE: number } }).GPUShaderStage.COMPUTE;

    // Group 0 — bindings 0..13 (full). Types must match material.wgsl.ts.
    expect(g0!.entries.map((e) => e.binding)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
    const byBinding = (entries: GPUBindGroupLayoutEntry[]) =>
      new Map(entries.map((e) => [e.binding, e]));
    const g0m = byBinding(g0!.entries);
    expect(g0m.get(0)!.storageTexture?.format).toBe('rgba16float'); // outputTexture
    expect(g0m.get(0)!.storageTexture?.access).toBe('write-only');
    expect(g0m.get(1)!.buffer?.type).toBe('uniform'); // params
    expect(g0m.get(2)!.buffer?.type).toBe('storage'); // accumBuffer (read_write)
    expect(g0m.get(3)!.buffer?.type).toBe('read-only-storage'); // positions
    expect(g0m.get(9)!.storageTexture?.format).toBe('rgba16float'); // normalDepth
    expect(g0m.get(12)!.storageTexture).toBeDefined(); // motionVectors
    expect(g0m.get(13)!.buffer?.type).toBe('storage'); // varianceMoments (read_write)
    for (const e of g0!.entries) expect(e.visibility).toBe(COMPUTE);

    // Group 1 — 11 read-only storage buffers (binding 10 = directionalLights, N-directional expansion).
    expect(g1!.entries).toHaveLength(11);
    for (const e of g1!.entries) {
      expect(e.buffer?.type).toBe('read-only-storage');
      expect(e.visibility).toBe(COMPUTE);
    }

    // Group 2 — TLAS table (0..4 read-only) + BDPT light-path/eye-stack (5,6 read_write).
    const g2m = byBinding(g2!.entries);
    expect(g2!.entries.map((e) => e.binding)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    for (const b of [0, 1, 2, 3, 4]) expect(g2m.get(b)!.buffer?.type).toBe('read-only-storage');
    expect(g2m.get(5)!.buffer?.type).toBe('storage'); // bdptLightPath
    expect(g2m.get(6)!.buffer?.type).toBe('storage'); // bdptEyeStack

    // Group 3 — WS2 light-tree buffer (0) + P2 material textures: meshUvs (1),
    // descriptors (2), sRGB texture_2d_array (3), sampler (4), LINEAR
    // texture_2d_array for normal/ORM (5) + A4 SPPM buffers: sppmPhotonCells (6),
    // sppmCellCounters (7), sppmStats uniform (8).
    expect(g3!.entries.map((e) => e.binding)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    const g3m = new Map(g3!.entries.map((e) => [e.binding, e]));
    expect(g3m.get(0)!.buffer?.type).toBe('read-only-storage'); // lightTree
    expect(g3m.get(1)!.buffer?.type).toBe('read-only-storage'); // meshUvs (P2)
    expect(g3m.get(2)!.buffer?.type).toBe('read-only-storage'); // descriptors (P2)
    expect(g3m.get(3)!.texture?.viewDimension).toBe('2d-array'); // materialTextures sRGB (P2)
    expect(g3m.get(4)!.sampler?.type).toBe('filtering'); // materialTexSampler (P2)
    expect(g3m.get(5)!.texture?.viewDimension).toBe('2d-array'); // materialTexturesLinear (P2)
    expect(g3m.get(6)!.buffer?.type).toBe('storage'); // A4: sppmPhotonCells (read_write)
    expect(g3m.get(7)!.buffer?.type).toBe('storage'); // A4: sppmCellCounters (read_write, atomic)
    expect(g3m.get(8)!.buffer?.type).toBe('uniform'); // A4: sppmStats (uniform)
    expect(g3m.get(0)!.visibility).toBe(COMPUTE);
  });

  it('full WITHOUT bdpt: still builds the explicit layout + path-trace pipeline (no bdpt pipeline)', () => {
    const stub = makeStubDevice();
    const gpu = new GpuResources(stub.device, 'full', false);
    gpu.ensurePipeline();

    expect(stub.createdPipelineLayouts).toHaveLength(1);
    expect(stub.createdPipelines.map((p) => p.entryPoint)).toEqual(['main']);
    expect(gpu.bdptSubpathPipeline).toBeNull();
    expect(gpu.bindGroupLayout).not.toBeNull();
  });

  // B12 — bindings 12/13/14 added for liteEnvTex + liteEnvCdfTex + liteLightTex.
  it('lite tier: single-group explicit layout (bindings 0..14), no group 1/2/3, no bdpt pipeline', () => {
    const stub = makeStubDevice();
    const gpu = new GpuResources(stub.device, 'lite', false);
    gpu.ensurePipeline();

    expect(stub.createdPipelineLayouts).toHaveLength(1);
    expect(stub.createdPipelineLayouts[0]!.bindGroupLayouts).toHaveLength(1);
    expect(gpu.bindGroupLayout1).toBeNull();
    expect(gpu.bindGroupLayout2).toBeNull();
    // WS2 — the lite tier keeps the uniform light pick and MUST NOT carry the
    // group-3 light-tree layout/binding.
    expect(gpu.bindGroupLayout3).toBeNull();
    expect(gpu.bdptSubpathPipeline).toBeNull();

    const [g0] = stub.createdLayouts;
    // B12 — bindings 12/13/14 are the lite texture slots (liteEnvTex, liteEnvCdfTex, liteLightTex).
    expect(g0!.entries.map((e) => e.binding)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
  });
});
