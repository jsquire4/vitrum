/**
 * sizingGpuDevice.ts — Size/alignment-validating GPUDevice stub for unit tests.
 *
 * Context (§H H53b):
 *   The "16B vs 32B dummy-buffer" bug class has recurred three times in this
 *   project:
 *     1. DDGI probe-update TLAS placeholder (ea88803) — 16→32 fix.
 *     2. RC dummy storage buffer (fixed same wave).
 *     3. ReSTIR merged-mode scene BGL (0bedd92) — 16→32 fix; hidden because the
 *        >1-mesh→TLAS auto-rule kept merged out of every multi-mesh test.
 *
 *   Each time the root cause is identical: `device.createBuffer({ size: 16, … })`
 *   is used as a placeholder for a binding whose WGSL struct stride is 32 bytes
 *   (array<BVHNode>).  WebGPU validates that the buffer's effective binding size
 *   is ≥ minBindingSize for the corresponding BGL entry.  In a real GPU driver
 *   this surfaces as "Binding size X less than minimum Y", which invalidates the
 *   WHOLE bind group and silently no-ops every pass that uses it.
 *
 * This helper provides:
 *   `createSizingGpuDevice(minSizeTable?)` — a mock GPUDevice that:
 *     - `createBuffer({ size, usage })` → records {size, usage}; asserts size>0
 *       and 4-byte alignment.  Returns a stub with a `.label` and `.destroy()`.
 *     - `createBindGroup({ layout, entries })` → if a `minSizeTable` is
 *       supplied, iterates buffer entries and asserts each buffer's effective
 *       binding size ≥ the declared minimum.  Returns a stub object.
 *
 * All recorded allocations are accessible via `device.allocations` for
 * assertions in tests.
 */

interface RecordedBuffer {
  label: string;
  size: number;
  usage: number;
}

interface BindGroupValidationError {
  binding: number;
  actualSize: number;
  minBindingSize: number;
}

/**
 * Per-binding minimum size table.  Keys are binding numbers; values are the
 * minimum effective binding size required by the shader (in bytes).
 *
 * Example: `{ 0: 32 }` means binding 0 must be backed by a buffer ≥ 32 bytes.
 */
export type MinBindingSizeTable = Record<number, number>;

export interface SizingGpuDevice {
  /** All createBuffer calls in order. */
  allocations: RecordedBuffer[];
  /** Validation errors from createBindGroup calls (empty on success). */
  bindGroupErrors: BindGroupValidationError[];

  // The minimal GPUDevice interface used by production bind-group builders.
  createBuffer(desc: { label?: string; size: number; usage: number; mappedAtCreation?: boolean }): GPUBuffer;
  createBindGroup(desc: {
    label?: string;
    layout: GPUBindGroupLayout;
    entries: Iterable<{ binding: number; resource: GPUBindingResource | { buffer: GPUBuffer; offset?: number; size?: number } }>;
  }): GPUBindGroup;
  createBindGroupLayout(desc: {
    label?: string;
    entries: Iterable<GPUBindGroupLayoutEntry>;
  }): GPUBindGroupLayout;
  createSampler(desc?: GPUSamplerDescriptor): GPUSampler;
  createTexture(desc: GPUTextureDescriptor): GPUTexture;
  queue: { writeBuffer(...args: unknown[]): void; writeTexture(...args: unknown[]): void; submit(...args: unknown[]): void };
}

/**
 * Create a size-validating GPUDevice stub.
 *
 * @param minSizeTable  Optional per-binding minimum binding size table.  When
 *                      provided, `createBindGroup` asserts that each buffer entry's
 *                      effective size ≥ the declared minimum.  Use this to catch
 *                      the 16-vs-32 dummy-buffer class at unit-test speed.
 */
export function createSizingGpuDevice(
  minSizeTable: MinBindingSizeTable = {},
): SizingGpuDevice {
  const allocations: RecordedBuffer[] = [];
  const bindGroupErrors: BindGroupValidationError[] = [];

  /** Map from stub buffer identity → its declared size (for binding validation). */
  const bufferSizes = new WeakMap<object, number>();

  function createBuffer(desc: {
    label?: string;
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): GPUBuffer {
    const { label = '<unlabeled>', size, usage } = desc;

    // Size invariants that WebGPU enforces on all real devices.
    if (size <= 0) {
      throw new RangeError(
        `[sizingGpuDevice] createBuffer size must be > 0, got ${size} (label: ${label})`,
      );
    }
    if (size % 4 !== 0) {
      throw new RangeError(
        `[sizingGpuDevice] createBuffer size must be 4-byte aligned, got ${size} (label: ${label})`,
      );
    }

    const stub = {
      label,
      size,
      usage,
      destroy() { /* no-op */ },
      getMappedRange() { return new ArrayBuffer(size); },
      unmap() { /* no-op */ },
    } as unknown as GPUBuffer;

    bufferSizes.set(stub as object, size);
    allocations.push({ label, size, usage });
    return stub;
  }

  function createBindGroup(desc: {
    label?: string;
    layout: GPUBindGroupLayout;
    entries: Iterable<{ binding: number; resource: unknown }>;
  }): GPUBindGroup {
    for (const entry of desc.entries) {
      const minSize = minSizeTable[entry.binding];
      if (minSize == null) continue;

      // Entries whose resource is a buffer binding are objects with a `buffer` key.
      const resource = entry.resource as { buffer?: object; offset?: number; size?: number };
      if (resource == null || typeof resource !== 'object' || !('buffer' in resource)) {
        continue; // sampler / texture view — not a buffer
      }

      const buf = resource.buffer as object;
      const bufSize = bufferSizes.get(buf) ?? 0;
      // Effective binding size = explicit `size` field if present, else full buffer.
      const effectiveSize = resource.size ?? bufSize;

      if (effectiveSize < minSize) {
        const err: BindGroupValidationError = {
          binding: entry.binding,
          actualSize: effectiveSize,
          minBindingSize: minSize,
        };
        bindGroupErrors.push(err);
        throw new RangeError(
          `[sizingGpuDevice] createBindGroup: binding ${entry.binding} effective size ` +
          `${effectiveSize} < minBindingSize ${minSize} (label: ${desc.label ?? '<unlabeled>'})`,
        );
      }
    }

    return {} as GPUBindGroup;
  }

  function createBindGroupLayout(desc: {
    label?: string;
    entries: Iterable<GPUBindGroupLayoutEntry>;
  }): GPUBindGroupLayout {
    void desc; // structural stub — layout compatibility isn't checked here
    return {} as GPUBindGroupLayout;
  }

  function createSampler(_desc?: GPUSamplerDescriptor): GPUSampler {
    return {} as GPUSampler;
  }

  function createTexture(_desc: GPUTextureDescriptor): GPUTexture {
    return {
      createView: () => ({}) as GPUTextureView,
      destroy: () => { /* no-op */ },
    } as unknown as GPUTexture;
  }

  const queue = {
    writeBuffer() { /* no-op */ },
    writeTexture() { /* no-op */ },
    submit() { /* no-op */ },
  };

  const device: SizingGpuDevice = {
    allocations,
    bindGroupErrors,
    createBuffer,
    createBindGroup,
    createBindGroupLayout,
    createSampler,
    createTexture,
    queue,
  };

  return device;
}
