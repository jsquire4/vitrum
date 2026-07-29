import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';

const ENGINE_SOURCE = readFileSync(new URL('../index.ts', import.meta.url), 'utf8');
const GPU_RESOURCES_SOURCE = readFileSync(
  new URL('../gpuResources.ts', import.meta.url),
  'utf8',
);

describe('BDPT path state has no frame-global advance or shared path-storage seam', () => {
  it('removes the public override hook and group-2 path-buffer rebuild', () => {
    expect(ENGINE_SOURCE).not.toContain('bdptAdvanceFrame');
    expect(ENGINE_SOURCE).not.toContain('BdptLightPathBufferWebGPU');
    expect(GPU_RESOURCES_SOURCE).not.toContain('rebuildGroup2Only');
    expect(GPU_RESOURCES_SOURCE).not.toContain('bdptSubpathPipeline');
    expect(GPU_RESOURCES_SOURCE).not.toContain('bdptEyeStackBuffer');
    expect(GPU_RESOURCES_SOURCE).not.toContain('bdptLightPathBuffer');
  });

  it('keeps both bounded path prefixes in invocation-private WGSL memory', () => {
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain(
      'var<private> bdptLightPath: array<vec4f, 56>;',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'var<private> bdptEyeStackPrivate: array<BdptEyeVtx, 8>;',
    );
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).not.toContain('@group(2) @binding(5)');
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).not.toContain('@group(2) @binding(6)');
  });

  it('builds the light prefix as an ordinary function, not a separate compute pass', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'fn bdptBuildInvocationLightSubpath(pixel: vec2u)',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).not.toContain('@compute');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).not.toContain('bdptExtendLightSubpath');
  });
});
