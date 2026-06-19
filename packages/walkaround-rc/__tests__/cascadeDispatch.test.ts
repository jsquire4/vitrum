import { describe, expect, it, vi } from 'vitest';
import { RCDispatcher } from '../src/cascadeDispatch.js';

describe('RCDispatcher bind group layout', () => {
  it('declares rgba32float material atlas textures as unfilterable', () => {
    const g = globalThis as unknown as { GPUShaderStage?: { COMPUTE: number } };
    g.GPUShaderStage ??= { COMPUTE: 4 };

    const layouts: GPUBindGroupLayoutDescriptor[] = [];
    const device = {
      createBindGroupLayout: vi.fn((desc: GPUBindGroupLayoutDescriptor) => {
        layouts.push(desc);
        return {} as GPUBindGroupLayout;
      }),
    } as unknown as GPUDevice;

    const dispatcher = new RCDispatcher() as unknown as {
      _castBindGroupLayout(device: GPUDevice): GPUBindGroupLayout;
    };
    dispatcher._castBindGroupLayout(device);

    const entries = new Map(
      Array.from(layouts[0]!.entries).map((entry) => [entry.binding, entry]),
    );
    const textureEntry = (binding: number): GPUTextureBindingLayout | undefined =>
      (entries.get(binding) as (GPUBindGroupLayoutEntry & { texture?: GPUTextureBindingLayout }) | undefined)?.texture;

    expect(textureEntry(16)).toMatchObject({
      sampleType: 'unfilterable-float',
      viewDimension: '2d-array',
    });
    expect(textureEntry(17)).toMatchObject({
      sampleType: 'unfilterable-float',
      viewDimension: '2d',
    });
    expect(textureEntry(19)).toMatchObject({
      sampleType: 'unfilterable-float',
      viewDimension: '2d',
    });
    expect(textureEntry(20)).toMatchObject({
      sampleType: 'unfilterable-float',
      viewDimension: '2d',
    });
  });
});
