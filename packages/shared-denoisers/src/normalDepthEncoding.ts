/** Canonical encoded world-up normal used when normalDepth.w == 0 (no hit). */
export const NORMAL_DEPTH_NO_HIT_NORMAL = [0.5, 1, 0.5] as const;

/**
 * Physical depth placement for denoiser shaders.
 *
 * Standalone dispatchers bind a dedicated single-channel depth texture, while
 * converged renderer backends bind the canonical packed normalDepth texture
 * (`xyz = normal * .5 + .5`, `w = signed linear depth`). Shader composition
 * must select one explicitly; guessing from texel values breaks at sky pixels.
 */
export type NormalDepthTextureLayout =
  | 'separate-depth-r'
  | 'packed-normal-depth-w';

export const STANDALONE_DEPTH_TEXTURE_LAYOUT: NormalDepthTextureLayout =
  'separate-depth-r';
export const PACKED_NORMAL_DEPTH_TEXTURE_LAYOUT: NormalDepthTextureLayout =
  'packed-normal-depth-w';

/** WGSL component name for the selected physical depth layout. */
export function normalDepthWgslDepthComponent(
  layout: NormalDepthTextureLayout,
): 'r' | 'w' {
  return layout === 'packed-normal-depth-w' ? 'w' : 'r';
}

/**
 * Decode the backend-neutral normalDepth.xyz affine encoding.
 *
 * Converged backends store a world-space unit normal as n * 0.5 + 0.5. This
 * helper deliberately performs only that inverse affine transform: it does not
 * renormalize, preserving the existing OIDN input behavior and half-float
 * quantization characteristics.
 */
export function decodeNormalDepthWorldNormal(
  r: number,
  g: number,
  b: number,
): [number, number, number] {
  return [r * 2 - 1, g * 2 - 1, b * 2 - 1];
}

/**
 * WGSL counterpart of {@link decodeNormalDepthWorldNormal}.
 *
 * Keep shader consumers on the same affine encoding contract as CPU readback
 * paths instead of re-declaring (and eventually drifting from) it locally.
 */
export const NORMAL_DEPTH_DECODE_WGSL = /* wgsl */`
fn decodeNormalDepthWorldNormal(encoded: vec3f) -> vec3f {
  return encoded * 2.0 - vec3f(1.0);
}
`;
