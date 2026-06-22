import { describe, expect, it } from 'vitest';

import {
  loadWeightsFromArrayBuffer,
  serializeWeightsToArrayBuffer,
  type ModelWeights,
} from '../weights.js';

function simpleWeights(): ModelWeights {
  return {
    layers: [
      {
        name: 'conv_in',
        weights: new Float32Array([0.25, -0.5, 1.0]),
        biases: new Float32Array([0.125]),
      },
    ],
  };
}

function appendBytes(buffer: ArrayBuffer, bytes: readonly number[]): ArrayBuffer {
  const out = new Uint8Array(buffer.byteLength + bytes.length);
  out.set(new Uint8Array(buffer), 0);
  out.set(bytes, buffer.byteLength);
  return out.buffer;
}

describe('vitrum-model weight loader', () => {
  it('rejects trailing bytes after the declared layer records', () => {
    const serialized = serializeWeightsToArrayBuffer(simpleWeights());
    const withTrailingBytes = appendBytes(serialized, [0xde, 0xad, 0xbe, 0xef]);

    expect(() => loadWeightsFromArrayBuffer(withTrailingBytes)).toThrow(
      /trailing 4 byte\(s\) after 1 layer record\(s\)/,
    );
  });

  it('round-trips exact buffers without trailing bytes', () => {
    const serialized = serializeWeightsToArrayBuffer(simpleWeights());
    const loaded = loadWeightsFromArrayBuffer(serialized);

    expect(loaded.layers).toHaveLength(1);
    const layer = loaded.layers[0];
    if (layer == null) throw new Error('expected one loaded layer');
    expect(layer.name).toBe('conv_in');
    expect(Array.from(layer.weights)).toEqual([0.25, -0.5, 1.0]);
    expect(Array.from(layer.biases)).toEqual([0.125]);
  });
});
