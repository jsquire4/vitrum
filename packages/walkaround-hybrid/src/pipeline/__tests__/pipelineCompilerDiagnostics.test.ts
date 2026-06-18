import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import { emitShaderCompilationWarnings } from '../pipelineCompiler.js';

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
