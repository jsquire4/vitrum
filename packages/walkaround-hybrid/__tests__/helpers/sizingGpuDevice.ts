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
 *     - `createBuffer({ size, usage })` → records {size, usage}; asserts size>0,
 *       usage>0, integer usage, and 4-byte alignment.  Returns a stub with a
 *       `.label`, `.size`, `.usage`, and `.destroy()`.
 *     - `createBindGroupLayout({ entries })` → stores the layout entries on the
 *       returned token.
 *     - `createBindGroup({ layout, entries })` → validates required/missing
 *       binding indices, buffer binding ranges, minBindingSize, and required
 *       UNIFORM/STORAGE usage bits for buffer entries.  Returns a stub object.
 *
 * All recorded allocations are accessible via `device.allocations` for
 * assertions in tests.
 */

interface RecordedBuffer {
  label: string;
  size: number;
  usage: number;
}

interface RecordedBindGroupLayout {
  label: string;
  entries: GPUBindGroupLayoutEntry[];
}

interface RecordedBindGroup {
  label: string;
  entries: GPUBindGroupEntry[];
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
  /** All createBindGroupLayout calls in order. */
  bindGroupLayouts: RecordedBindGroupLayout[];
  /** All successfully-created createBindGroup calls in order. */
  bindGroups: RecordedBindGroup[];
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
  const bindGroupLayouts: RecordedBindGroupLayout[] = [];
  const bindGroups: RecordedBindGroup[] = [];
  const bindGroupErrors: BindGroupValidationError[] = [];

  /** Map from stub buffer identity → its declared size (for binding validation). */
  const bufferSizes = new WeakMap<object, number>();
  /** Map from stub buffer identity → its usage flags (for binding validation). */
  const bufferUsages = new WeakMap<object, number>();
  /** Map from stub layout identity → its layout entries. */
  const layoutEntries = new WeakMap<object, GPUBindGroupLayoutEntry[]>();

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
    if (!Number.isFinite(usage) || usage <= 0 || Math.floor(usage) !== usage) {
      throw new RangeError(
        `[sizingGpuDevice] createBuffer usage must be a positive integer, got ${usage} (label: ${label})`,
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
    bufferUsages.set(stub as object, usage);
    allocations.push({ label, size, usage });
    return stub;
  }

  function createBindGroup(desc: {
    label?: string;
    layout: GPUBindGroupLayout;
    entries: Iterable<{ binding: number; resource: unknown }>;
  }): GPUBindGroup {
    const entries = Array.from(desc.entries);
    const layout = desc.layout as object;
    const expectedEntries = layoutEntries.get(layout) ?? [];
    validateBindGroupBindings(desc.label, expectedEntries, entries);

    for (const entry of entries) {
      const layoutEntry = expectedEntries.find((candidate) => candidate.binding === entry.binding);
      const minSize = Math.max(
        minSizeTable[entry.binding] ?? 0,
        layoutEntry?.buffer?.minBindingSize ?? 0,
      );

      if (layoutEntry?.buffer == null && minSize === 0) {
        assertNonBufferResource(desc.label, layoutEntry, entry);
        continue;
      }

      // Entries whose resource is a buffer binding are objects with a `buffer` key
      // or a raw GPUBuffer-like object that carries size/usage fields.
      const resource = entry.resource as { buffer?: object; offset?: number; size?: number };
      const rawBufferLike = resource as unknown as { size?: number; usage?: number };
      const buffer = resource != null && typeof resource === 'object' && 'buffer' in resource
        ? resource.buffer
        : rawBufferLike != null && typeof rawBufferLike === 'object' && 'size' in rawBufferLike
          ? rawBufferLike as object
          : null;
      if (buffer == null) {
        throw new TypeError(
          `[sizingGpuDevice] createBindGroup: binding ${entry.binding} expected a buffer resource ` +
          `(label: ${desc.label ?? '<unlabeled>'})`,
        );
      }

      const bufSize = bufferSizes.get(buffer) ?? Number((buffer as { size?: number }).size ?? 0);
      const bufUsage = bufferUsages.get(buffer) ?? Number((buffer as { usage?: number }).usage ?? 0);
      validateBufferBinding(desc.label, entry.binding, resource, bufSize, bufUsage, minSize, layoutEntry, bindGroupErrors);
    }

    const bindGroup = { label: desc.label ?? '<unlabeled>', entries } as unknown as GPUBindGroup;
    bindGroups.push({ label: desc.label ?? '<unlabeled>', entries: entries as GPUBindGroupEntry[] });
    return bindGroup;
  }

  function createBindGroupLayout(desc: {
    label?: string;
    entries: Iterable<GPUBindGroupLayoutEntry>;
  }): GPUBindGroupLayout {
    const entries = Array.from(desc.entries, (entry) => ({ ...entry }));
    assertUniqueLayoutBindings(desc.label, entries);
    const layout = { label: desc.label ?? '<unlabeled>', entries } as unknown as GPUBindGroupLayout;
    layoutEntries.set(layout as object, entries);
    bindGroupLayouts.push({ label: desc.label ?? '<unlabeled>', entries });
    return layout;
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
    bindGroupLayouts,
    bindGroups,
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

function assertUniqueLayoutBindings(label: string | undefined, entries: GPUBindGroupLayoutEntry[]): void {
  const seen = new Set<number>();
  for (const entry of entries) {
    if (seen.has(entry.binding)) {
      throw new RangeError(
        `[sizingGpuDevice] createBindGroupLayout: duplicate binding ${entry.binding} ` +
        `(label: ${label ?? '<unlabeled>'})`,
      );
    }
    seen.add(entry.binding);
  }
}

function validateBindGroupBindings(
  label: string | undefined,
  layoutEntries: GPUBindGroupLayoutEntry[],
  entries: Array<{ binding: number; resource: unknown }>,
): void {
  if (layoutEntries.length === 0) return;

  const expected = new Set(layoutEntries.map((entry) => entry.binding));
  const seen = new Set<number>();
  for (const entry of entries) {
    if (seen.has(entry.binding)) {
      throw new RangeError(
        `[sizingGpuDevice] createBindGroup: duplicate binding ${entry.binding} ` +
        `(label: ${label ?? '<unlabeled>'})`,
      );
    }
    if (!expected.has(entry.binding)) {
      throw new RangeError(
        `[sizingGpuDevice] createBindGroup: binding ${entry.binding} is not present in layout ` +
        `(label: ${label ?? '<unlabeled>'})`,
      );
    }
    seen.add(entry.binding);
  }

  for (const binding of expected) {
    if (!seen.has(binding)) {
      throw new RangeError(
        `[sizingGpuDevice] createBindGroup: missing required binding ${binding} ` +
        `(label: ${label ?? '<unlabeled>'})`,
      );
    }
  }
}

function assertNonBufferResource(
  label: string | undefined,
  layoutEntry: GPUBindGroupLayoutEntry | undefined,
  entry: { binding: number; resource: unknown },
): void {
  if (layoutEntry?.texture == null && layoutEntry?.storageTexture == null && layoutEntry?.sampler == null) {
    return;
  }
  const resource = entry.resource;
  if (resource != null && typeof resource === 'object' && ('buffer' in resource || 'usage' in resource)) {
    throw new TypeError(
      `[sizingGpuDevice] createBindGroup: binding ${entry.binding} expected a non-buffer resource ` +
      `(label: ${label ?? '<unlabeled>'})`,
    );
  }
}

function validateBufferBinding(
  label: string | undefined,
  binding: number,
  resource: { offset?: number; size?: number },
  bufferSize: number,
  bufferUsage: number,
  minBindingSize: number,
  layoutEntry: GPUBindGroupLayoutEntry | undefined,
  bindGroupErrors: BindGroupValidationError[],
): void {
  if (!Number.isFinite(bufferSize) || bufferSize <= 0) {
    throw new RangeError(
      `[sizingGpuDevice] createBindGroup: binding ${binding} references an unknown or invalid buffer ` +
      `(label: ${label ?? '<unlabeled>'})`,
    );
  }

  const offset = resource.offset ?? 0;
  if (!Number.isFinite(offset) || offset < 0 || Math.floor(offset) !== offset) {
    throw new RangeError(
      `[sizingGpuDevice] createBindGroup: binding ${binding} has invalid offset ${offset} ` +
      `(label: ${label ?? '<unlabeled>'})`,
    );
  }
  if (offset >= bufferSize) {
    throw new RangeError(
      `[sizingGpuDevice] createBindGroup: binding ${binding} offset ${offset} leaves no bindable range ` +
      `in buffer size ${bufferSize} (label: ${label ?? '<unlabeled>'})`,
    );
  }

  const effectiveSize = resource.size ?? (bufferSize - offset);
  if (!Number.isFinite(effectiveSize) || effectiveSize <= 0 || Math.floor(effectiveSize) !== effectiveSize) {
    throw new RangeError(
      `[sizingGpuDevice] createBindGroup: binding ${binding} has invalid effective size ${effectiveSize} ` +
      `(label: ${label ?? '<unlabeled>'})`,
    );
  }
  if (offset + effectiveSize > bufferSize) {
    throw new RangeError(
      `[sizingGpuDevice] createBindGroup: binding ${binding} range ${offset}+${effectiveSize} ` +
      `exceeds buffer size ${bufferSize} (label: ${label ?? '<unlabeled>'})`,
    );
  }

  if (effectiveSize < minBindingSize) {
    const err: BindGroupValidationError = {
      binding,
      actualSize: effectiveSize,
      minBindingSize,
    };
    bindGroupErrors.push(err);
    throw new RangeError(
      `[sizingGpuDevice] createBindGroup: binding ${binding} effective size ` +
      `${effectiveSize} < minBindingSize ${minBindingSize} (label: ${label ?? '<unlabeled>'})`,
    );
  }

  const bufferType = layoutEntry?.buffer?.type ?? 'read-only-storage';
  const requiredUsage = bufferType === 'uniform'
    ? bufferUsageFlag('UNIFORM')
    : bufferUsageFlag('STORAGE');
  if ((bufferUsage & requiredUsage) === 0) {
    const expected = bufferType === 'uniform' ? 'UNIFORM' : 'STORAGE';
    throw new TypeError(
      `[sizingGpuDevice] createBindGroup: binding ${binding} buffer usage ${bufferUsage} ` +
      `does not include GPUBufferUsage.${expected} (label: ${label ?? '<unlabeled>'})`,
    );
  }
}

function bufferUsageFlag(name: 'UNIFORM' | 'STORAGE'): number {
  const usage = (globalThis as { GPUBufferUsage?: { UNIFORM?: number; STORAGE?: number } }).GPUBufferUsage;
  if (name === 'UNIFORM') return usage?.UNIFORM ?? 0x40;
  return usage?.STORAGE ?? 0x80;
}
