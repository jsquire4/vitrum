import type { LayerKind } from './unetArchitecture.js';
import {
  neuralTensorWgslPreamble,
  type NeuralTensorStorageContract,
} from './tensorPrecision.js';

function replaceRequired(source: string, from: string, to: string): string {
  if (!source.includes(from)) {
    throw new Error(`[neural WGSL] precision transform lost required source fragment: ${from}`);
  }
  return source.replace(from, to);
}

/**
 * Specialize a canonical f32-activation kernel to f16 tensor storage. Learned
 * weights/biases and every convolution accumulator deliberately remain f32.
 * Required-fragment checks make source drift fail during initialization rather
 * than silently compiling a mixed layout.
 */
export function neuralLayerWgslForStorage(
  kind: Exclude<LayerKind, 'inputPack'>,
  source: string,
  storage: NeuralTensorStorageContract,
): string {
  if (storage.precision === 'f32') return source;
  let result = neuralTensorWgslPreamble(storage) + source;
  switch (kind) {
    case 'conv2d':
    case 'transposedConv2d':
      result = replaceRequired(result, 'input   : array<f32>', 'input   : array<f16>');
      result = replaceRequired(result, 'output  : array<f32>', 'output  : array<f16>');
      result = replaceRequired(
        result,
        'neuralSanitizeSigned(input[inIdx])',
        'neuralSanitizeSigned(f32(input[inIdx]))',
      );
      result = replaceRequired(
        result,
        'output[outIdx] = neuralSanitizeSigned(acc);',
        'output[outIdx] = f16(neuralSanitizeSigned(acc));',
      );
      return result;
    case 'relu':
      result = replaceRequired(result, 'input  : array<f32>', 'input  : array<f16>');
      result = replaceRequired(result, 'output : array<f32>', 'output : array<f16>');
      result = replaceRequired(
        result,
        'neuralSanitizeSigned(input[idx])',
        'neuralSanitizeSigned(f32(input[idx]))',
      );
      result = replaceRequired(
        result,
        'output[idx] = neuralSanitizeSigned(max(0.0, neuralSanitizeSigned(f32(input[idx]))));',
        'output[idx] = f16(neuralSanitizeSigned(max(0.0, neuralSanitizeSigned(f32(input[idx])))));',
      );
      return result;
    case 'skipAdd':
      result = replaceRequired(result, 'inputA : array<f32>', 'inputA : array<f16>');
      result = replaceRequired(result, 'inputB : array<f32>', 'inputB : array<f16>');
      result = replaceRequired(result, 'output : array<f32>', 'output : array<f16>');
      result = replaceRequired(
        result,
        'neuralSanitizeSigned(inputA[idx]) + neuralSanitizeSigned(inputB[idx])',
        'neuralSanitizeSigned(f32(inputA[idx])) + neuralSanitizeSigned(f32(inputB[idx]))',
      );
      result = replaceRequired(
        result,
        'output[idx] = neuralSanitizeSigned(',
        'output[idx] = f16(neuralSanitizeSigned(',
      );
      result = replaceRequired(result, '\n  );\n}', '\n  ));\n}');
      return result;
    case 'bilinearUpsample':
      result = replaceRequired(result, 'inputBuf  : array<f32>', 'inputBuf  : array<f16>');
      result = replaceRequired(result, 'outputBuf : array<f32>', 'outputBuf : array<f16>');
      result = replaceRequired(
        result,
        'return neuralSanitizeSigned(inputBuf[clampedY * params.inputW * params.channels + clampedX * params.channels + ch]);',
        'return neuralSanitizeSigned(f32(inputBuf[clampedY * params.inputW * params.channels + clampedX * params.channels + ch]));',
      );
      result = replaceRequired(
        result,
        'outputBuf[outIdx] = neuralSanitizeSigned(val);',
        'outputBuf[outIdx] = f16(neuralSanitizeSigned(val));',
      );
      return result;
  }
}
