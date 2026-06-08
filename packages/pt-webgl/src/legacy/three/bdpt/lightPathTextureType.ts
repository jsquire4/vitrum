import { FloatType, HalfFloatType, type TextureDataType } from 'three';

/**
 * ANGLE/D3D11 often breaks unidirectional PT when a RGBA32F light-path is bound.
 * Use RGBA16F on ANGLE stacks; full float elsewhere (SwiftShader uses CPU fill + no bind).
 */
export function bdptLightPathTextureType(rendererLabel: string): TextureDataType {
  return /angle/i.test(rendererLabel) ? HalfFloatType : FloatType;
}
