import { describe, expect, it } from 'vitest';
import type { FrameInput, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import {
  SOBOL_TEXTURE_CHANNELS,
  SOBOL_TEXTURE_POINTS,
  SOBOL_TEXTURE_SIZE,
} from '@vitrum/shared-samplers';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';

const GREY: MaterialSpec = { baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0 };

function triScene(): Scene {
  const prim: MeshPrimitive = {
    kind: 'mesh',
    id: 'tri',
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material: GREY,
  };
  return { primitives: [prim], emitters: [], environment: { kind: 'none' } };
}

function frame(): FrameInput {
  return {
    viewMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]) as never,
    projMatrix: new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]) as never,
    cameraPosition: [0, 0, 5] as never,
    viewport: { width: 32, height: 32, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
    quality: { samplesTarget: 1 },
  };
}

async function renderWithRecord(engineOpts: Record<string, unknown> = {}): Promise<{
  readonly gl: WebGL2RenderingContext;
  readonly record: Map<string, unknown>;
}> {
  const record = new Map<string, unknown>();
  const gl = createMockGl(record);
  const engine = await createPTEngine_WebGL2({ device: gl, ...engineOpts });
  engine.setScene(triScene());
  engine.renderFrame(frame());
  return { gl, record };
}

function shaderSources(record: Map<string, unknown>): string {
  return ((record.get('__shaderSources') as readonly string[] | undefined) ?? []).join('\n');
}

function texImageUploads(record: Map<string, unknown>): readonly unknown[][] {
  return (record.get('__texImage2D') as readonly unknown[][] | undefined) ?? [];
}

describe('pt-webgl2 sampling options', () => {
  it('keeps PCG as the default random sequence', async () => {
    const { record } = await renderWithRecord();
    expect(shaderSources(record)).toContain('#define RANDOM_TYPE 0');
    const sobolUpload = texImageUploads(record).find((args) =>
      args[3] === SOBOL_TEXTURE_SIZE &&
      args[4] === SOBOL_TEXTURE_SIZE &&
      args[8] instanceof Float32Array &&
      args[8].length === SOBOL_TEXTURE_POINTS * SOBOL_TEXTURE_CHANNELS,
    );
    expect(sobolUpload).toBeUndefined();
  });

  it('uploads a real RGBA32F Sobol table when sampling=sobol', async () => {
    const { gl, record } = await renderWithRecord({ sampling: 'sobol' });
    const sources = shaderSources(record);
    expect(sources).toContain('#define RANDOM_TYPE 1');
    expect(sources).toContain('( vec2( x, y ) + 0.5 ) / vec2( dim )');
    const sobolUpload = texImageUploads(record).find((args) =>
      args[2] === gl.RGBA32F &&
      args[3] === SOBOL_TEXTURE_SIZE &&
      args[4] === SOBOL_TEXTURE_SIZE &&
      args[6] === gl.RGBA &&
      args[7] === gl.FLOAT &&
      args[8] instanceof Float32Array &&
      args[8].length === SOBOL_TEXTURE_POINTS * SOBOL_TEXTURE_CHANNELS,
    );
    expect(sobolUpload).toBeDefined();
  });
});
