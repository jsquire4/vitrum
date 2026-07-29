import { describe, expect, it } from 'vitest';
import type { FrameInput, MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';

const MATERIAL = { baseColor: [0.7, 0.5, 0.3] as const, roughness: 0.5, metallic: 0 };

function spectralBdptScene(): Scene {
  const panel: MeshPrimitive = {
    kind: 'mesh',
    id: 'panel',
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material: MATERIAL,
  };
  return {
    primitives: [panel],
    emitters: [{ kind: 'mesh-area', id: 'light', meshId: 'panel', color: [1, 1, 1], intensity: 4 }],
    environment: { kind: 'none' },
  };
}

function input(): FrameInput {
  return {
    viewMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]) as never,
    projMatrix: new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]) as never,
    cameraPosition: [0, 0, 5] as never,
    viewport: { width: 16, height: 16, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
    quality: { samplesTarget: 3 },
  };
}

function f32Bits(value: number): number {
  const array = new Float32Array([value]);
  return new Uint32Array(array.buffer)[0]!;
}

async function captureSequence(sampling: 'pcg' | 'sobol'): Promise<readonly number[]> {
  const scalarWrites = new Map<string, number[]>();
  const vectorWrites = new Map<string, number[][]>();
  const base = createMockGl();
  const gl = new Proxy(base, {
    get(target, prop, receiver): unknown {
      if (prop === 'getUniformLocation') {
        return (_program: unknown, name: string) => ({ __u: name });
      }
      if (prop === 'uniform1f') {
        return (location: { __u?: string }, value: number) => {
          const name = location.__u;
          if (name != null) {
            const values = scalarWrites.get(name) ?? [];
            values.push(value);
            scalarWrites.set(name, values);
          }
        };
      }
      if (prop === 'uniform3f') {
        return (location: { __u?: string }, x: number, y: number, z: number) => {
          const name = location.__u;
          if (name != null) {
            const values = vectorWrites.get(name) ?? [];
            values.push([x, y, z]);
            vectorWrites.set(name, values);
          }
        };
      }
      return Reflect.get(target, prop, receiver) as unknown;
    },
  });

  const engine = await createPTEngine_WebGL2({
    device: gl,
    spectral: true,
    bdpt: true,
    sampling,
  });
  engine.setScene(spectralBdptScene());
  for (let i = 0; i < 3; i++) engine.renderFrame(input());

  const wavelengths = scalarWrites.get('uBdptSharedWavelength') ?? [];
  const pdfs = scalarWrites.get('uBdptSharedWavelengthPdf') ?? [];
  const uploadsPerFrame = 4;
  expect(wavelengths).toHaveLength(3 * uploadsPerFrame);
  expect(pdfs).toHaveLength(3 * uploadsPerFrame);
  for (let frame = 0; frame < 3; frame++) {
    const firstUpload = frame * uploadsPerFrame;
    for (let pass = 1; pass < uploadsPerFrame; pass++) {
      expect(f32Bits(wavelengths[firstUpload]!)).toBe(
        f32Bits(wavelengths[firstUpload + pass]!),
      );
      expect(f32Bits(pdfs[firstUpload]!)).toBe(f32Bits(pdfs[firstUpload + pass]!));
    }
    expect(Number.isFinite(pdfs[firstUpload]) && pdfs[firstUpload]! > 0).toBe(true);
  }
  const sequence = [wavelengths[0]!, wavelengths[4]!, wavelengths[8]!];
  expect(new Set(sequence.map(f32Bits)).size).toBe(3);

  for (const name of ['iorCauchyA', 'iorCauchyB', 'iorCauchyC']) {
    expect(scalarWrites.has(name)).toBe(false);
  }
  engine.reset();
  engine.renderFrame(input());
  expect(f32Bits((scalarWrites.get('uBdptSharedWavelength') ?? [])[12]!))
    .toBe(f32Bits(wavelengths[0]!));
  expect(f32Bits((scalarWrites.get('uBdptSharedWavelengthPdf') ?? [])[12]!))
    .toBe(f32Bits(pdfs[0]!));
  engine.dispose();
  return sequence.map(f32Bits);
}

describe('spectral + BDPT production wavelength sequence', () => {
  it('advances per accumulated sample, resets deterministically, and is mode-independent', async () => {
    const pcg = await captureSequence('pcg');
    const sobol = await captureSequence('sobol');
    expect(pcg).toEqual(sobol);
  });
});
