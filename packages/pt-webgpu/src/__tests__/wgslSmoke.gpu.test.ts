/**
 * GPU smoke test — submits PT_WEBGPU_TRACE_WGSL to a real device and confirms
 * createComputePipelineAsync resolves without compile errors. Skipped under
 * Node / jsdom where navigator.gpu is unavailable; runs in Vitest's browser
 * mode or in environments that expose WebGPU.
 *
 * This is the minimum bar that catches WGSL syntax regressions that the
 * string-only contract tests cannot detect (missing constants, type mismatch
 * in main(), etc.).
 */
import { describe, expect, it, beforeAll } from 'vitest';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';

const gpu: GPU | undefined =
  typeof navigator !== 'undefined' && (navigator as { gpu?: GPU }).gpu
    ? (navigator as unknown as { gpu: GPU }).gpu
    : undefined;

describe.skipIf(!gpu)('PT_WEBGPU_TRACE_WGSL GPU smoke', () => {
  let device: GPUDevice | null = null;

  beforeAll(async () => {
    const adapter = await gpu!.requestAdapter();
    if (!adapter) return;
    device = await adapter.requestDevice();
  });

  it('compiles on a real WebGPU device without errors', async () => {
    if (!device) {
      // Adapter was not available; skip silently.
      return;
    }
    const module = device.createShaderModule({
      label: 'pt-webgpu-smoke',
      code: PT_WEBGPU_TRACE_WGSL,
    });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) {
      throw new Error(
        `WGSL compile error(s):\n${errors
          .map((e) => `  line ${e.lineNum}: ${e.message}`)
          .join('\n')}`,
      );
    }
    expect(errors.length).toBe(0);
    // Pipeline creation exercises full validation beyond syntactic compile.
    await device.createComputePipelineAsync({
      label: 'pt-webgpu-smoke-pipeline',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  });
});
