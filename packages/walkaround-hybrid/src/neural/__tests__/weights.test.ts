import { describe, expect, it } from 'vitest';

import {
  assessNeuralCheckpointProductionReadiness,
  loadWeightsFromArrayBuffer,
  serializeWeightsToArrayBuffer,
  type ModelWeights,
} from '../weights.js';
import { NEURAL_PREPROCESSING_CONTRACT } from '../preprocessing.js';

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

function productionWeights(): ModelWeights {
  return {
    ...simpleWeights(),
    formatVersion: 2,
    checkpoint: {
      id: 'unit-production',
      trainingSamples: 500,
      noisySpp: 1,
      cleanSpp: 4096,
      auxiliaryInputs: ['albedo', 'normal'],
      captureSource: 'unit-test',
      captureBackend: 'walkaround-hybrid',
      tonemap: 'linear',
      hardware: 'test-adapter',
      preprocessing: NEURAL_PREPROCESSING_CONTRACT,
      qualityReport: {
        status: 'pass',
        reportPath: 'unit-report.json',
        validationScenes: 8,
        psnrDb: 31.25,
        ssim: 0.91,
      },
    },
  };
}

function copyBuffer(buffer: ArrayBuffer): ArrayBuffer {
  return buffer.slice(0);
}

function replaceAscii(buffer: ArrayBuffer, from: string, to: string): ArrayBuffer {
  if (from.length !== to.length) throw new Error('replacement length must be stable');
  const out = new Uint8Array(copyBuffer(buffer));
  const needle = new TextEncoder().encode(from);
  const replacement = new TextEncoder().encode(to);
  outer: for (let i = 0; i <= out.length - needle.length; i++) {
    for (let j = 0; j < needle.length; j++) {
      if (out[i + j] !== needle[j]) continue outer;
    }
    out.set(replacement, i);
    return out.buffer;
  }
  throw new Error(`did not find '${from}' in model binary`);
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
  it('round-trips v2 checkpoint metadata deterministically and marks it production-ready', () => {
    const source = productionWeights();
    const first = serializeWeightsToArrayBuffer(source);
    const second = serializeWeightsToArrayBuffer(source);
    expect(Array.from(new Uint8Array(first))).toEqual(Array.from(new Uint8Array(second)));

    const loaded = loadWeightsFromArrayBuffer(first);
    expect(loaded.formatVersion).toBe(2);
    expect(loaded.checkpoint).toEqual(source.checkpoint);
    expect(assessNeuralCheckpointProductionReadiness(loaded)).toEqual({
      productionReady: true,
      missing: [],
      metadata: loaded.checkpoint,
    });
  });

  it('loads v1 as a compatibility fixture but does not certify it', () => {
    const legacy = loadWeightsFromArrayBuffer(
      serializeWeightsToArrayBuffer(simpleWeights(), { version: 1 }),
    );
    expect(legacy.formatVersion).toBe(1);
    expect(legacy.checkpoint).toBeUndefined();
    expect(assessNeuralCheckpointProductionReadiness(legacy)).toMatchObject({
      productionReady: false,
      missing: ['formatVersion=2', 'checkpoint metadata'],
    });
  });

  it('rejects impossible header counts and metadata lengths before allocation', () => {
    const layerCount = copyBuffer(serializeWeightsToArrayBuffer(productionWeights()));
    new DataView(layerCount).setUint32(8, 0xffff_ffff, true);
    expect(() => loadWeightsFromArrayBuffer(layerCount)).toThrow(/impossible layerCount/);

    const metadataLength = copyBuffer(serializeWeightsToArrayBuffer(productionWeights()));
    new DataView(metadataLength).setUint32(12, 0xffff_ffff, true);
    expect(() => loadWeightsFromArrayBuffer(metadataLength)).toThrow(/truncated reading v2 metadata JSON/);
  });

  it('rejects malformed UTF-8, unknown metadata enums, and preprocessing schema drift', () => {
    const invalidUtf8 = copyBuffer(serializeWeightsToArrayBuffer(productionWeights()));
    new Uint8Array(invalidUtf8)[16] = 0xff;
    expect(() => loadWeightsFromArrayBuffer(invalidUtf8)).toThrow(/invalid UTF-8 in v2 metadata JSON/);

    const unknownQuality = replaceAscii(
      serializeWeightsToArrayBuffer(productionWeights()),
      '"pass"',
      '"nope"',
    );
    expect(() => loadWeightsFromArrayBuffer(unknownQuality)).toThrow(
      /qualityReport\.status has unknown enum value 'nope'/,
    );

    const unknownPreprocessing = replaceAscii(
      serializeWeightsToArrayBuffer(productionWeights()),
      'linear-hdr-scaled',
      'linear-hdr-broken',
    );
    expect(() => loadWeightsFromArrayBuffer(unknownPreprocessing)).toThrow(
      /metadata\.preprocessing has an unsupported schema or enum value/,
    );
  });

  it('rejects non-finite and excessive parameter magnitudes while loading and serializing', () => {
    const legacy = serializeWeightsToArrayBuffer(simpleWeights(), { version: 1 });
    const nameLength = new DataView(legacy).getUint32(12, true);
    const firstWeightOffset = 12 + 4 + nameLength + 4;

    const excessive = copyBuffer(legacy);
    new DataView(excessive).setFloat32(firstWeightOffset, 2048, true);
    expect(() => loadWeightsFromArrayBuffer(excessive)).toThrow(/exceeds magnitude bound 1024/);

    const nonFinite = copyBuffer(legacy);
    new DataView(nonFinite).setFloat32(firstWeightOffset, Number.POSITIVE_INFINITY, true);
    expect(() => loadWeightsFromArrayBuffer(nonFinite)).toThrow(/is not finite/);

    expect(() => serializeWeightsToArrayBuffer({
      layers: [{
        name: 'conv_in',
        weights: new Float32Array([2048]),
        biases: new Float32Array(0),
      }],
    })).toThrow(/exceeds magnitude bound 1024/);
  });

});
