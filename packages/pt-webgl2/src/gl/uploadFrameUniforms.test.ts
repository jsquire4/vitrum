import { describe, expect, it } from 'vitest';
import type { GlProgram } from './glProgram.js';
import type { FrameUniforms } from './glResources.js';
import { uploadFrameUniforms } from './uploadFrameUniforms.js';

const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function frameWithBdpt(bdpt: boolean): FrameUniforms {
  return {
    resolution: [64, 32],
    bounces: 4,
    transmissiveBounces: 4,
    filterGlossyFactor: 0,
    materialLodDepth: 0,
    cameraWorldMatrix: IDENTITY,
    invProjectionMatrix: IDENTITY,
    environmentIntensity: 1,
    environmentRotation: IDENTITY,
    backgroundBlur: 0,
    spectralEnabled: false,
    backgroundAlpha: 1,
    bdpt,
    bdptMaxLightBounces: 6,
    bdptSceneCenter: [3, 5, 7],
    bdptSceneRadius: 11,
    bdptSharedWavelengthNm: 550,
    bdptSharedWavelengthPdf: 0.25,
    iorCauchy: [0, 0, 0],
    dof: null,
    tonemapMode: 0,
    exposure: 1,
    outputColorSpace: 0,
  };
}

function recordingProgram(calls: string[]): GlProgram {
  return {
    use: () => true,
    setInt: (name: string, value: number) => calls.push(`int:${name}:${value}`),
    setFloat: (name: string, value: number) => calls.push(`float:${name}:${value}`),
    setVec2: (name: string, x: number, y: number) => calls.push(`vec2:${name}:${x},${y}`),
    setVec3: (name: string, x: number, y: number, z: number) =>
      calls.push(`vec3:${name}:${x},${y},${z}`),
    setMat4: (name: string) => calls.push(`mat4:${name}`),
    setFloatArray: (name: string) => calls.push(`array:${name}`),
  } as unknown as GlProgram;
}

describe('uploadFrameUniforms BDPT pass parity', () => {
  it('uploads scene-launch bounds to every BDPT eye/candidate program', () => {
    const calls: string[] = [];
    uploadFrameUniforms(recordingProgram(calls), 1, 17, frameWithBdpt(true));

    expect(calls).toContain('vec3:uBdptSceneCenter:3,5,7');
    expect(calls).toContain('float:uBdptSceneRadius:11');
    expect(calls).toContain('float:uBdptSharedWavelength:550');
    expect(calls).toContain('float:uBdptSharedWavelengthPdf:0.25');
  });

  it('does not upload BDPT-only uniforms for the unidirectional program', () => {
    const calls: string[] = [];
    uploadFrameUniforms(recordingProgram(calls), 1, 17, frameWithBdpt(false));

    expect(calls.some((call) => call.includes('uBdptScene'))).toBe(false);
    expect(calls.some((call) => call.includes('uBdptSharedWavelength'))).toBe(false);
  });
});
