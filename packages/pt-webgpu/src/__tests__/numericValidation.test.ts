import { describe, expect, it } from 'vitest';
import { asMat4, type FrameInput } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import {
  assertPtWebgpuBdptFrameCameraSupported,
  validatePtWebgpuFrameInput,
} from '../ptWebgpuValidation.js';

const validationOnlyDevice = {
  createCommandEncoder() { return {}; },
} as unknown as GPUDevice;

function frameInput(): FrameInput {
  const identity = asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]));
  return {
    viewMatrix: identity,
    projMatrix: identity,
    cameraPosition: [0, 0, 0],
    viewport: { width: 64, height: 32, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 1,
  };
}

describe('pt-webgpu strict numeric construction validation', () => {
  it.each([
    ['maxSamplesPerPixel', NaN],
    ['maxSamplesPerPixel', Infinity],
    ['maxSamplesPerPixel', 0],
    ['maxSamplesPerPixel', 1.5],
    ['maxSamplesPerPixel', 0x1_0000_0000],
    ['maxBounces', NaN],
    ['maxBounces', 0],
    ['maxBounces', 1.5],
    ['maxBounces', 9],
  ] as const)('rejects %s=%s before GPU allocation', async (field, value) => {
    await expect(createPTEngine_WebGPU({
      device: validationOnlyDevice,
      [field]: value,
    })).rejects.toThrow(field);
  });

  it.each([
    ['mneeMaxIterations', NaN],
    ['mneeMaxIterations', 0],
    ['mneeMaxIterations', 1.5],
    ['mneeMaxIterations', 33],
    ['mneeMaxChainLength', Infinity],
    ['mneeMaxChainLength', 0],
    ['mneeMaxChainLength', 1.5],
    ['mneeMaxChainLength', 9],
  ] as const)('rejects causticOptions.%s=%s before GPU allocation', async (field, value) => {
    await expect(createPTEngine_WebGPU({
      device: validationOnlyDevice,
      causticOptions: { [field]: value },
    })).rejects.toThrow(`causticOptions.${field}`);
  });
});

describe('pt-webgpu strict per-frame numeric validation', () => {
  it('rejects orthographic projection for the perspective-only BDPT endpoint', () => {
    const orthographic = frameInput();
    expect(() => assertPtWebgpuBdptFrameCameraSupported(orthographic))
      .toThrow(/bdpt:true supports perspective camera projections only/);

    const perspective = new Float32Array(orthographic.projMatrix);
    perspective[11] = -1;
    perspective[15] = 0;
    expect(() => assertPtWebgpuBdptFrameCameraSupported({
      ...orthographic,
      projMatrix: asMat4(perspective),
    })).not.toThrow();
  });

  it('accepts valid quality dials and values above the engine caps for documented upper clamping', () => {
    expect(() => validatePtWebgpuFrameInput({
      ...frameInput(),
      quality: {
        samplesTarget: 100_000,
        bounces: 99,
        resolutionFactor: 0.5,
        exposure: 0,
        filteredGlossyFactor: 0,
      },
    })).not.toThrow();
  });

  it('rejects nonzero filteredGlossyFactor instead of silently ignoring it', () => {
    expect(() => validatePtWebgpuFrameInput({
      ...frameInput(),
      quality: { filteredGlossyFactor: 0.5 },
    })).toThrow(/filteredGlossyFactor is unsupported by pt-webgpu/);
  });

  it('accepts an omitted legacy camera position and rejects disagreement', () => {
    const { cameraPosition: _legacy, ...withoutPosition } = frameInput();
    expect(() => validatePtWebgpuFrameInput(withoutPosition)).not.toThrow();
    expect(() => validatePtWebgpuFrameInput({
      ...frameInput(),
      cameraPosition: [0, 0, 0.25],
    })).toThrow(/disagrees with inverse\(viewMatrix\)/);
  });

  it.each([
    ['frameIndex', NaN],
    ['frameIndex', -1],
    ['frameIndex', 1.5],
    ['frameSeed', Infinity],
    ['frameSeed', 0x1_0000_0000],
  ] as const)('rejects invalid %s=%s', (field, value) => {
    expect(() => validatePtWebgpuFrameInput({ ...frameInput(), [field]: value })).toThrow(field);
  });

  it.each([
    ['width', NaN],
    ['width', 0],
    ['width', 1.5],
    ['width', Number.MAX_SAFE_INTEGER + 1],
    ['height', Infinity],
    ['height', -1],
    ['height', 2.5],
    ['devicePixelRatio', 0],
  ] as const)('rejects invalid viewport.%s=%s', (field, value) => {
    const base = frameInput();
    expect(() => validatePtWebgpuFrameInput({
      ...base,
      viewport: { ...base.viewport, [field]: value },
    })).toThrow(`viewport.${field}`);
  });

  it.each([
    ['samplesTarget', NaN],
    ['samplesTarget', 0],
    ['samplesTarget', -1],
    ['samplesTarget', 1.5],
    ['bounces', Infinity],
    ['bounces', 0],
    ['bounces', -1],
    ['bounces', 1.5],
    ['resolutionFactor', NaN],
    ['resolutionFactor', 0],
    ['resolutionFactor', -0.25],
    ['resolutionFactor', 1.01],
    ['exposure', Infinity],
    ['exposure', Number.MAX_VALUE],
    ['exposure', Number.MIN_VALUE],
    ['exposure', -1],
    ['filteredGlossyFactor', NaN],
    ['filteredGlossyFactor', -0.01],
    ['filteredGlossyFactor', 1.01],
  ] as const)('rejects invalid quality.%s=%s', (field, value) => {
    expect(() => validatePtWebgpuFrameInput({
      ...frameInput(),
      quality: { [field]: value },
    })).toThrow(`quality.${field}`);
  });

  it('rejects non-finite matrix/vector components with exact field context', () => {
    const input = frameInput();
    const view = new Float32Array(input.viewMatrix);
    view[7] = NaN;
    expect(() => validatePtWebgpuFrameInput({ ...input, viewMatrix: asMat4(view) })).toThrow('viewMatrix[7]');
    expect(() => validatePtWebgpuFrameInput({
      ...input,
      cameraPosition: [0, Infinity, 0],
    })).toThrow('cameraPosition[1]');
  });
});
