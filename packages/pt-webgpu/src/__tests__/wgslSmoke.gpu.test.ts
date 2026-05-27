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
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import {
  PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
} from '../webgpuLimits.js';

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

  it('full pipeline creates when the adapter supports split full tier (10 buffers/group)', async () => {
    const supported = adapter!.limits.maxStorageBuffersPerShaderStage;
    const module = device!.createShaderModule({
      label: 'pt-webgpu-pipeline-smoke-full',
      code: PT_WEBGPU_TRACE_WGSL,
    });

    if (supported >= PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE) {
      await device!.createComputePipelineAsync({
        label: 'pt-webgpu-smoke-pipeline-full',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      expect(supported).toBeGreaterThanOrEqual(PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE);
    } else {
      let caughtMessage: string | null = null;
      try {
        await device!.createComputePipelineAsync({
          label: 'pt-webgpu-smoke-pipeline-full',
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        });
      } catch (err) {
        caughtMessage = String((err as Error)?.message ?? err);
      }
      expect(caughtMessage).not.toBeNull();
      expect(caughtMessage).toMatch(/storage buffers .* exceeds the maximum/i);
      expect(supported).toBeLessThan(PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE);
    }
  });

  it('lite WGSL parses without compile errors', async () => {
    const module = device!.createShaderModule({
      label: 'pt-webgpu-smoke-lite',
      code: PT_WEBGPU_TRACE_LITE_WGSL,
    });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) {
      throw new Error(
        `lite WGSL compile error(s):\n${errors
          .map((e) => `  line ${e.lineNum}: ${e.message}`)
          .join('\n')}`,
      );
    }
    expect(errors.length).toBe(0);
  });

  it('lite pipeline creates when the adapter supports the lite binding count', async () => {
    const supported = adapter!.limits.maxStorageBuffersPerShaderStage;
    const module = device!.createShaderModule({
      label: 'pt-webgpu-pipeline-smoke-lite',
      code: PT_WEBGPU_TRACE_LITE_WGSL,
    });

    if (supported >= PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE) {
      await device!.createComputePipelineAsync({
        label: 'pt-webgpu-smoke-pipeline-lite',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      expect(supported).toBeGreaterThanOrEqual(PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE);
    } else {
      let caughtMessage: string | null = null;
      try {
        await device!.createComputePipelineAsync({
          label: 'pt-webgpu-pipeline-smoke-lite',
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        });
      } catch (err) {
        caughtMessage = String((err as Error)?.message ?? err);
      }
      expect(caughtMessage).not.toBeNull();
      expect(caughtMessage).toMatch(/storage buffers .* exceeds the maximum/i);
    }
  });
});
