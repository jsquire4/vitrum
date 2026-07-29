import { describe, expect, it } from 'vitest';
import { convertAnimations } from './animations.js';
import { GltfComponentType, type GltfJson } from './gltfTypes.js';

function f32(values: readonly number[]): ArrayBuffer {
  const buffer = new ArrayBuffer(values.length * 4);
  const view = new DataView(buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return buffer;
}

function u32(values: readonly number[]): ArrayBuffer {
  const buffer = new ArrayBuffer(values.length * 4);
  const view = new DataView(buffer);
  values.forEach((value, index) => view.setUint32(index * 4, value, true));
  return buffer;
}

function uintPointerFixture(values: readonly number[]): {
  gltf: GltfJson;
  buffers: Map<number, ArrayBuffer>;
} {
  const times = f32(values.map((_, index) => index));
  const outputs = u32(values);
  return {
    gltf: {
      asset: { version: '2.0' },
      buffers: [{ byteLength: times.byteLength }, { byteLength: outputs.byteLength }],
      bufferViews: [
        { buffer: 0, byteLength: times.byteLength },
        { buffer: 1, byteLength: outputs.byteLength },
      ],
      accessors: [
        {
          bufferView: 0,
          componentType: GltfComponentType.FLOAT,
          count: values.length,
          type: 'SCALAR',
        },
        {
          bufferView: 1,
          componentType: GltfComponentType.UNSIGNED_INT,
          count: values.length,
          type: 'SCALAR',
        },
      ],
      animations: [{
        samplers: [{ input: 0, output: 1, interpolation: 'STEP' }],
        channels: [{
          sampler: 0,
          target: {
            path: 'pointer',
            extensions: {
              KHR_animation_pointer: {
                pointer: '/extensions/KHR_lights_punctual/lights/0/intensity',
              },
            },
          },
        }],
      }],
      extensions: {
        KHR_lights_punctual: {
          lights: [{ type: 'point', intensity: 1 }],
        },
      },
    },
    buffers: new Map([[0, times], [1, outputs]]),
  };
}

describe('animation sampler integrity', () => {
  it('preserves the inclusive 2^24 UNSIGNED_INT bound exactly', () => {
    const { gltf, buffers } = uintPointerFixture([0, 2 ** 24]);
    const clips = convertAnimations(gltf, buffers, []);

    expect(Array.from(clips[0]!.channels[0]!.sampler.values)).toEqual([0, 2 ** 24]);
  });

  it('fails closed before UNSIGNED_INT animation outputs above 2^24 are rounded', () => {
    const { gltf, buffers } = uintPointerFixture([0, 2 ** 24 + 1]);

    expect(() => convertAnimations(gltf, buffers, [])).toThrow(
      /UNSIGNED_INT value 16777217.*exact animation-output integer bound 16777216.*lose integer precision/,
    );
  });

  it('applies the exact-integer bound to sparse animation output values', () => {
    const time = f32([0]);
    const sparseIndex = new Uint8Array([0]).buffer;
    const sparseValue = u32([2 ** 24 + 1]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      buffers: [
        { byteLength: time.byteLength },
        { byteLength: sparseIndex.byteLength },
        { byteLength: sparseValue.byteLength },
      ],
      bufferViews: [
        { buffer: 0, byteLength: time.byteLength },
        { buffer: 1, byteLength: sparseIndex.byteLength },
        { buffer: 2, byteLength: sparseValue.byteLength },
      ],
      accessors: [
        {
          bufferView: 0,
          componentType: GltfComponentType.FLOAT,
          count: 1,
          type: 'SCALAR',
        },
        {
          componentType: GltfComponentType.UNSIGNED_INT,
          count: 1,
          type: 'SCALAR',
          sparse: {
            count: 1,
            indices: {
              bufferView: 1,
              componentType: GltfComponentType.UNSIGNED_BYTE,
            },
            values: { bufferView: 2 },
          },
        },
      ],
      animations: [{
        samplers: [{ input: 0, output: 1, interpolation: 'STEP' }],
        channels: [{
          sampler: 0,
          target: {
            path: 'pointer',
            extensions: {
              KHR_animation_pointer: {
                pointer: '/extensions/KHR_lights_punctual/lights/0/intensity',
              },
            },
          },
        }],
      }],
      extensions: {
        KHR_lights_punctual: {
          lights: [{ type: 'point', intensity: 1 }],
        },
      },
    };

    expect(() => convertAnimations(
      gltf,
      new Map([[0, time], [1, sparseIndex], [2, sparseValue]]),
      [],
    )).toThrow(/UNSIGNED_INT value 16777217.*lose integer precision/);
  });

  it('rejects a single-key CUBICSPLINE sampler', () => {
    const times = f32([0]);
    const cubicOutput = f32([
      0, 0, 0, // in tangent
      1, 2, 3, // value
      0, 0, 0, // out tangent
    ]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      nodes: [{}],
      buffers: [
        { byteLength: times.byteLength },
        { byteLength: cubicOutput.byteLength },
      ],
      bufferViews: [
        { buffer: 0, byteLength: times.byteLength },
        { buffer: 1, byteLength: cubicOutput.byteLength },
      ],
      accessors: [
        {
          bufferView: 0,
          componentType: GltfComponentType.FLOAT,
          count: 1,
          type: 'SCALAR',
        },
        {
          bufferView: 1,
          componentType: GltfComponentType.FLOAT,
          count: 3,
          type: 'VEC3',
        },
      ],
      animations: [{
        samplers: [{ input: 0, output: 1, interpolation: 'CUBICSPLINE' }],
        channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
      }],
    };

    expect(() => convertAnimations(
      gltf,
      new Map([[0, times], [1, cubicOutput]]),
      [],
    )).toThrow(/CUBICSPLINE animation samplers require at least two keyframes/);
  });
});
