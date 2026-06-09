/**
 * environment.ts — THREE.Scene environment → @vitrum/core SceneEnvironment.
 *
 * Resolves the scene's equirect/HDRI environment texture. Solid-color
 * backgrounds are not IBL sources and are treated as no environment.
 */

import type * as THREE from 'three';
import type { SceneEnvironment } from '@vitrum/core';

/** How resolved texture handles (env `hdri` + material maps) are represented:
 *  `'texture'` = a `THREE.Texture` (for the fork-wrapping `@vitrum/pt-webgl`);
 *  `'raw'` = a backend-neutral pixel payload (for the THREE-free path tracers). */
export type TexturePayloadMode = 'texture' | 'raw';

/** IEEE-754 half (uint16) → float32, no THREE dependency (keeps this module THREE-type-only). */
function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

/**
 * Extract a backend-neutral `{ width, height, data }` equirect payload (row-major
 * **RGB** float, 3 floats/pixel) from a THREE equirect `DataTexture`, for the
 * THREE-free path tracers (`@vitrum/pt-webgl2`, `@vitrum/pt-webgpu`) whose env
 * packers read raw pixels rather than a `THREE.Texture`. Returns `null` when the
 * texture has no readable CPU pixels (e.g. a GPU-only / cube / PMREM texture).
 *
 * Orientation + type handling mirror the fork's `EquirectHdrInfoUniform.preprocessEnvMap`
 * so the IBL is oriented identically: rows are reversed iff `texture.flipY`, and the
 * source is read at its true channel stride (RGBA→4 / RGB→3) with HalfFloat decoded.
 */
export function equirectTextureToPayload(
  tex: THREE.Texture,
): { width: number; height: number; data: Float32Array } | null {
  const img = tex.image as { data?: ArrayLike<number>; width?: number; height?: number } | undefined;
  const src = img?.data;
  const width = Number(img?.width ?? 0);
  const height = Number(img?.height ?? 0);
  if (src == null || typeof src.length !== 'number' || width <= 0 || height <= 0) return null;

  const stride = Math.max(3, Math.round(src.length / (width * height))); // RGBA→4, RGB→3
  const isHalf = src instanceof Uint16Array;
  const isFloat = src instanceof Float32Array;
  // Integer textures (UnsignedByte etc.) normalise by their max; float/half are linear.
  const intMax = isHalf || isFloat ? 0 : 2 ** (8 * ((src as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1)) - 1;
  const decode = (v: number): number => (isHalf ? halfToFloat(v) : isFloat ? v : intMax > 0 ? v / intMax : v);

  const flip = tex.flipY === true;
  const out = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    const sy = flip ? height - 1 - y : y;
    for (let x = 0; x < width; x += 1) {
      const si = (sy * width + x) * stride;
      const di = (y * width + x) * 3;
      out[di] = decode(Number(src[si] ?? 0));
      out[di + 1] = decode(Number(src[si + 1] ?? 0));
      out[di + 2] = decode(Number(src[si + 2] ?? 0));
    }
  }
  return { width, height, data: out };
}

/**
 * Resolve the @vitrum/core SceneEnvironment from a THREE.Scene.
 *
 * - `scene.environment` set → `{ kind: 'hdri', hdri: texture, intensity, rotationY }`.
 *   `intensity` mirrors `scene.environmentIntensity` (default 1); `rotationY`
 *   mirrors `scene.environmentRotation.y` (default 0). Capturing both fields
 *   prevents env intensity / rotation from being silently dropped on the
 *   THREE → vitrum → THREE round trip used by the PT engine env-update path.
 * - `scene.background` as a solid Color (no environment map) → `{ kind: 'none' }`.
 * - Nothing set → `{ kind: 'none' }`.
 *
 * Asymmetry note: this direction is the THREE → vitrum mapping. The reverse
 * direction (`vitrumSceneToThree`) does NOT substitute a procedural sky shader
 * — it warns and falls through to a dark background. Mapping the reverse here
 * — turning a `THREE.Sky` mesh / shader into a `ProceduralSkyEnvironment` —
 * is unimplemented because no host currently constructs three.js scenes from
 * a `THREE.Sky` source. If a host needs this, read uniforms `{ turbidity,
 * mieCoefficient, mieDirectionalG, rayleigh }` off the sky material and emit
 * `{ kind: 'procedural-sky', ... }`. Until then a solid-color background is
 * not an IBL source — treat as no environment.
 */
export function resolveEnvironment(
  threeScene: THREE.Scene,
  mode: TexturePayloadMode = 'texture',
): SceneEnvironment {
  // In 'raw' mode, emit a THREE-free {width,height,data} payload for THREE-free
  // backends (pt-webgl2 / pt-webgpu); fall back to the texture handle when the
  // texture has no readable CPU pixels. 'texture' (default) keeps the THREE.Texture
  // for the fork-wrapping @vitrum/pt-webgl (vitrumSceneToThree reads it back).
  const asHandle = (tex: THREE.Texture): unknown =>
    mode === 'raw' ? (equirectTextureToPayload(tex) ?? tex) : tex;

  if (threeScene.environment != null) {
    // THREE.Scene.environmentIntensity defaults to 1; environmentRotation is
    // a THREE.Euler. We capture only the Y rotation because HdriEnvironment
    // models a yaw around world up (matches WebGLPathTracer's equirect map
    // rotation behavior — full Euler isn't needed for hemispheric IBL).
    const intensity = threeScene.environmentIntensity ?? 1;
    const rotationY = threeScene.environmentRotation?.y ?? 0;
    return { kind: 'hdri', hdri: asHandle(threeScene.environment), intensity, rotationY };
  }
  // Some hosts use background-only HDRI setups (environment unset).
  if ((threeScene.background as THREE.Texture | null | undefined)?.isTexture === true) {
    const bg = threeScene.background as THREE.Texture;
    const intensity = threeScene.backgroundIntensity ?? 1;
    const rotationY = threeScene.backgroundRotation?.y ?? 0;
    return { kind: 'hdri', hdri: asHandle(bg), intensity, rotationY };
  }
  return { kind: 'none' };
}
