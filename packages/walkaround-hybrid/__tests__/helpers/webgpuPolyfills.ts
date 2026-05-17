/**
 * Shared WebGPU constant polyfills for the Node test environment.
 *
 * Vitest runs under happy-dom/jsdom which doesn't expose the WebGPU global
 * namespace objects (`GPUBufferUsage`, `GPUTextureUsage`). Production code
 * reads these as `GPUBufferUsage.STORAGE` etc., so tests that exercise that
 * code need the constants installed once before module import.
 *
 * Call `installWebGPUPolyfills()` at the top of each test file (it is
 * idempotent — re-calls are a no-op).
 */

const GPUBufferUsageValues = {
  MAP_READ:      0x0001,
  MAP_WRITE:     0x0002,
  COPY_SRC:      0x0004,
  COPY_DST:      0x0008,
  INDEX:         0x0010,
  VERTEX:        0x0020,
  UNIFORM:       0x0040,
  STORAGE:       0x0080,
  INDIRECT:      0x0100,
  QUERY_RESOLVE: 0x0200,
} as const;

const GPUTextureUsageValues = {
  COPY_SRC:          0x01,
  COPY_DST:          0x02,
  TEXTURE_BINDING:   0x04,
  STORAGE_BINDING:   0x08,
  RENDER_ATTACHMENT: 0x10,
} as const;

const GPUShaderStageValues = {
  VERTEX:   0x1,
  FRAGMENT: 0x2,
  COMPUTE:  0x4,
} as const;

const GPUMapModeValues = {
  READ:  0x0001,
  WRITE: 0x0002,
} as const;

export function installWebGPUPolyfills(): void {
  const g = globalThis as Record<string, unknown>;
  if (g['GPUBufferUsage'] === undefined) g['GPUBufferUsage'] = GPUBufferUsageValues;
  if (g['GPUTextureUsage'] === undefined) g['GPUTextureUsage'] = GPUTextureUsageValues;
  if (g['GPUShaderStage'] === undefined) g['GPUShaderStage'] = GPUShaderStageValues;
  if (g['GPUMapMode'] === undefined) g['GPUMapMode'] = GPUMapModeValues;
}
