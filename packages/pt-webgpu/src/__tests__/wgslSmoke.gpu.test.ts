/**
 * GPU smoke test — submits PT_WEBGPU_TRACE_WGSL to a real device and confirms
 * (a) the WGSL parses without compile errors and (b) the pipeline either
 * creates successfully OR fails with the expected storage-buffer-limit
 * error (depending on what the adapter supports).
 *
 * Runs in vitest browser mode under headless Chromium with SwiftShader as
 * the WebGPU adapter. SwiftShader's Vulkan ICD caps
 * maxStorageBuffersPerShaderStage at 10; the bruteforce shader binds 18.
 * Real consumer GPUs (RTX, M-series, recent AMD) ship with 16+ and run
 * the full pipeline. Both code paths are covered here.
 */
import { describe, expect, it, beforeAll } from 'vitest';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';

/** Storage-buffer count in PT_WEBGPU_TRACE_WGSL's compute stage. Update if
 *  the shader's binding list changes. Used to gate the pipeline-create
 *  assertion against the active adapter's max. */
const REQUIRED_STORAGE_BUFFERS = 18;

const gpu: GPU | undefined =
  typeof navigator !== 'undefined' && (navigator as { gpu?: GPU }).gpu
    ? (navigator as unknown as { gpu: GPU }).gpu
    : undefined;

describe('PT_WEBGPU_TRACE_WGSL GPU smoke', () => {
  let adapter: GPUAdapter | null = null;
  let device: GPUDevice | null = null;

  beforeAll(async () => {
    if (!gpu) {
      throw new Error(
        'WebGPU navigator.gpu is unavailable — this test must run in vitest ' +
        'browser mode with --enable-unsafe-webgpu and a software/hardware ' +
        'WebGPU adapter present. See packages/pt-webgpu/vitest.gpu.config.ts.',
      );
    }
    adapter = await gpu.requestAdapter();
    if (!adapter) {
      throw new Error(
        'gpu.requestAdapter() returned null — no WebGPU adapter is available. ' +
        'Check the SwiftShader/Vulkan flags in vitest.gpu.config.ts.',
      );
    }
    device = await adapter.requestDevice({
      requiredLimits: {
        // Request the adapter's max so we can run the pipeline-create test
        // when the adapter actually supports the shader's binding count.
        maxStorageBuffersPerShaderStage:
          adapter.limits.maxStorageBuffersPerShaderStage,
      },
    });
  });

  it('WGSL parses without compile errors on a real Dawn-backed device', async () => {
    const module = device!.createShaderModule({
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
  });

  it('pipeline creates when the adapter supports the shader\'s storage-buffer count', async () => {
    const supported = adapter!.limits.maxStorageBuffersPerShaderStage;
    const module = device!.createShaderModule({
      label: 'pt-webgpu-pipeline-smoke',
      code: PT_WEBGPU_TRACE_WGSL,
    });

    if (supported >= REQUIRED_STORAGE_BUFFERS) {
      // Hardware path — full pipeline creation should succeed end-to-end.
      await device!.createComputePipelineAsync({
        label: 'pt-webgpu-smoke-pipeline',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      // No throw = pass; expect at least the assertion below to keep vitest happy.
      expect(supported).toBeGreaterThanOrEqual(REQUIRED_STORAGE_BUFFERS);
    } else {
      // Software path (SwiftShader-Vulkan caps at 10). Assert pipeline
      // creation fails with the EXACT limit error so a future regression
      // (e.g. the shader silently fitting under the cap) is detectable.
      let caughtMessage: string | null = null;
      try {
        await device!.createComputePipelineAsync({
          label: 'pt-webgpu-smoke-pipeline',
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        });
      } catch (err) {
        caughtMessage = String((err as Error)?.message ?? err);
      }
      expect(caughtMessage).not.toBeNull();
      expect(caughtMessage).toMatch(/storage buffers .* exceeds the maximum/i);
      // Also document the constraint on the adapter for diagnostic logs.
      expect(supported).toBeLessThan(REQUIRED_STORAGE_BUFFERS);
    }
  });
});
