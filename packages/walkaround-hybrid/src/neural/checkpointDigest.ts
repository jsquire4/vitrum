import type { LayerWeights } from './weights.js';
import { sha256Hex } from './sha256.js';

/**
 * SHA-256 over canonical ordered parameter-layer records. Parameterless graph
 * nodes are excluded because exported checkpoints contain learned tensors only.
 */
export function neuralCheckpointPayloadSha256(
  layers: readonly LayerWeights[],
): string {
  const encoder = new TextEncoder();
  const records = layers
    .filter(layer => layer.weights.length > 0 || layer.biases.length > 0)
    .map(layer => ({ layer, name: encoder.encode(layer.name) }));
  const byteLength = records.reduce(
    (sum, record) =>
      sum + 4 + record.name.byteLength +
      4 + record.layer.weights.length * 4 +
      4 + record.layer.biases.length * 4,
    0,
  );
  const bytes = new Uint8Array(byteLength);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  const writeU32 = (value: number): void => {
    view.setUint32(offset, value, true);
    offset += 4;
  };
  const writeF32 = (value: number): void => {
    view.setFloat32(offset, value, true);
    offset += 4;
  };
  for (const { layer, name } of records) {
    writeU32(name.byteLength);
    bytes.set(name, offset);
    offset += name.byteLength;
    writeU32(layer.weights.length);
    for (const value of layer.weights) writeF32(value);
    writeU32(layer.biases.length);
    for (const value of layer.biases) writeF32(value);
  }
  return sha256Hex(bytes);
}
