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
      createPipelineLayout: (desc: GPUPipelineLayoutDescriptor) =>
        ({ bindGroupLayouts: desc.bindGroupLayouts }) as unknown as GPUPipelineLayout,
      createComputePipelineAsync: async () => ({} as GPUComputePipeline),
      createRenderPipelineAsync: async () => ({} as GPURenderPipeline),
      createRenderPipeline: () => ({} as GPURenderPipeline),
    } as unknown as GPUDevice;
    const bglCache = {};

    try {
      await compilePipelines(device, bglCache, 'bgra8unorm', {
        ppgEnabled: true,
        regirEnabled: true,
      });

      expect(inspectedLabels.sort()).toEqual(createdLabels.sort());
      const moduleCount = createdLabels.length;
      createCompositePipeline(device, bglCache, 'rgba8unorm');
      expect(createdLabels).toHaveLength(moduleCount);
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
