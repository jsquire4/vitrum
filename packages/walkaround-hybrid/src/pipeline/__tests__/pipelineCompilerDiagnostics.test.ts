import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import {
  compilePipelines,
  createCompositePipeline,
  emitShaderCompilationWarnings,
} from '../pipelineCompiler.js';

const warningMessage = (message: string): GPUCompilationMessage =>
  ({
    type: 'warning',
    message,
    lineNum: 7,
    linePos: 3,
    offset: 128,
    length: 12,
  }) as GPUCompilationMessage;

describe('pipeline compiler diagnostics', () => {
  it('inspects compilation info for every shader module before using it', async () => {
    const previousGPUShaderStage =
      (globalThis as { GPUShaderStage?: unknown }).GPUShaderStage;
    Object.defineProperty(globalThis, 'GPUShaderStage', {
      value: Object.freeze({ VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 }),
      configurable: true,
    });

    const createdLabels: string[] = [];
    const inspectedLabels: string[] = [];
    const asyncRenderDescriptors: GPURenderPipelineDescriptor[] = [];
    const syncRenderDescriptors: GPURenderPipelineDescriptor[] = [];
    const createPipelineLayout = vi.fn(
      (desc: GPUPipelineLayoutDescriptor) =>
        ({ bindGroupLayouts: desc.bindGroupLayouts }) as unknown as GPUPipelineLayout,
    );
    const device = {
      createShaderModule(desc: GPUShaderModuleDescriptor): GPUShaderModule {
        const label = String(desc.label ?? '');
        createdLabels.push(label);
        return {
          getCompilationInfo: async () => {
            inspectedLabels.push(label);
            return { messages: [] };
          },
        } as unknown as GPUShaderModule;
      },
      createBindGroupLayout: (desc: GPUBindGroupLayoutDescriptor) =>
        ({ entries: desc.entries }) as unknown as GPUBindGroupLayout,
      createPipelineLayout,
      createComputePipelineAsync: async () => ({} as GPUComputePipeline),
      createRenderPipelineAsync: async (desc: GPURenderPipelineDescriptor) => {
        asyncRenderDescriptors.push(desc);
        return {} as GPURenderPipeline;
      },
      createRenderPipeline: (desc: GPURenderPipelineDescriptor) => {
        syncRenderDescriptors.push(desc);
        return {} as GPURenderPipeline;
      },
    } as unknown as GPUDevice;
    const bglCache = {};

    try {
      await compilePipelines(device, bglCache, 'bgra8unorm', {
        ppgEnabled: true,
        regirEnabled: true,
      });

      expect(inspectedLabels.sort()).toEqual(createdLabels.sort());
      expect(asyncRenderDescriptors).toHaveLength(1);
      expect(asyncRenderDescriptors[0]?.fragment?.constants).toEqual({
        VT_ATTACHMENT_SRGB: 0,
        VT_TARGET_MAX_R: 1,
        VT_TARGET_MAX_G: 1,
        VT_TARGET_MAX_B: 1,
      });
      const moduleCount = createdLabels.length;
      createCompositePipeline(device, bglCache, 'rgba8unorm-srgb');
      createCompositePipeline(device, bglCache, 'rgba16float');
      expect(createdLabels).toHaveLength(moduleCount);
      expect(syncRenderDescriptors).toHaveLength(2);
      expect(syncRenderDescriptors[0]?.fragment?.constants).toEqual({
        VT_ATTACHMENT_SRGB: 1,
        VT_TARGET_MAX_R: 1,
        VT_TARGET_MAX_G: 1,
        VT_TARGET_MAX_B: 1,
      });
      expect(syncRenderDescriptors[1]?.fragment?.constants).toEqual({
        VT_ATTACHMENT_SRGB: 0,
        VT_TARGET_MAX_R: 65_504,
        VT_TARGET_MAX_G: 65_504,
        VT_TARGET_MAX_B: 65_504,
      });
      expect(syncRenderDescriptors[0]?.fragment?.targets).toEqual([
        { format: 'rgba8unorm-srgb' },
      ]);

      const layoutCallsBeforeInvalid = createPipelineLayout.mock.calls.length;
      expect(() =>
        createCompositePipeline(
          device,
          bglCache,
          'depth24plus' as GPUTextureFormat,
        ),
      ).toThrow(/swapChainFormat is unsupported/);
      expect(createPipelineLayout).toHaveBeenCalledTimes(
        layoutCallsBeforeInvalid,
      );
    } finally {
      if (previousGPUShaderStage === undefined) {
        delete (globalThis as { GPUShaderStage?: unknown }).GPUShaderStage;
      } else {
        Object.defineProperty(globalThis, 'GPUShaderStage', {
          value: previousGPUShaderStage,
          configurable: true,
        });
      }
    }
  });

  it('rejects an invalid initial format before creating any GPU object', async () => {
    const createShaderModule = vi.fn();
    const createPipelineLayout = vi.fn();
    const device = {
      createShaderModule,
      createPipelineLayout,
    } as unknown as GPUDevice;

    await expect(
      compilePipelines(
        device,
        {},
        'depth24plus' as GPUTextureFormat,
      ),
    ).rejects.toThrow(/swapChainFormat is unsupported/);
    expect(createShaderModule).not.toHaveBeenCalled();
    expect(createPipelineLayout).not.toHaveBeenCalled();
  });

  it('routes shader compilation warnings through structured warning sinks', () => {
    const warnings: EngineWarning[] = [];
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      emitShaderCompilationWarnings('shade', [warningMessage('unused variable')], {
        onWarning: (warning) => warnings.push(warning),
      });

      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toMatchObject({
        code: 'walkaround-hybrid.shader-compilation-warning',
        backend: 'walkaround-hybrid',
        phase: 'construction',
        method: 'initialize',
        details: {
          shaderLabel: 'shade',
          warnings: [
            {
              type: 'warning',
              message: 'unused variable',
              lineNum: 7,
              linePos: 3,
              offset: 128,
              length: 12,
            },
          ],
        },
      });
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('keeps console fallback for standalone compiler use', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      emitShaderCompilationWarnings('ris', [warningMessage('diagnostic')]);

      expect(warnSpy).toHaveBeenCalledOnce();
      expect(warnSpy.mock.calls[0]?.[0]).toBe("[ReSTIR] Shader warnings in 'ris':");
      expect(warnSpy.mock.calls[0]?.[1]).toEqual(['diagnostic']);
    } finally {
      warnSpy.mockRestore();
    }
  });
});
