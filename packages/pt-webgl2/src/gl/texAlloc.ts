/**
 * D10.14: Consolidated GL texture allocation helper.
 *
 * `allocGlTexture(gl, spec)` unifies the previously duplicated create+guard+bind+
 * sampling+upload sequences that existed independently in:
 *   - `bvhTextureAdapter.ts` (`makeTex`)
 *   - `uploadSceneTextures.ts` (`uploadRgba32f` / `uploadRgba32fRect` / `uploadRgba32fArray`)
 *
 * All variants share:
 *   1. Context-loss guard (throws early with an actionable message)
 *   2. `MAX_TEXTURE_SIZE` / `MAX_ARRAY_TEXTURE_LAYERS` guard
 *   3. `gl.createTexture()`
 *   4. NEAREST + CLAMP_TO_EDGE sampling
 *   5. `texImage2D` or `texImage3D`
 */

export type TexAllocSpec =
  | {
      /** Square 2D texture — dims are `dim × dim` (RGBA32F or RGBA32UI BVH textures). */
      readonly kind: '2d';
      readonly dim: number;
      readonly internalFormat: number;
      readonly format: number;
      readonly type: number;
      readonly data: ArrayBufferView;
      readonly resourceName: string;
    }
  | {
      /** Non-square 2D texture — for equirect maps / CDF slabs. */
      readonly kind: 'rect';
      readonly width: number;
      readonly height: number;
      readonly internalFormat: number;
      readonly format: number;
      readonly type: number;
      readonly data: ArrayBufferView;
      readonly resourceName: string;
    }
  | {
      /** 2D array texture — for multi-layer vertex-attribute or material-texture arrays. */
      readonly kind: 'array';
      readonly dim: number;
      readonly layers: number;
      readonly internalFormat: number;
      readonly format: number;
      readonly type: number;
      readonly data: ArrayBufferView;
      readonly resourceName: string;
    };

/**
 * Allocate a GL texture from a spec, guarding against context loss and device
 * dimension limits. Returns the new `WebGLTexture`.
 *
 * Throws an `Error` with an actionable message when:
 *   - The GL context has been lost (before attempting any GL call)
 *   - The requested dimension exceeds `MAX_TEXTURE_SIZE`
 *   - The requested layer count exceeds `MAX_ARRAY_TEXTURE_LAYERS` (array only)
 */
export function allocGlTexture(
  gl: WebGL2RenderingContext,
  spec: TexAllocSpec,
): WebGLTexture {
  const { resourceName } = spec;

  // Guard context loss — `getParameter` returns 0 on a lost context, producing
  // misleading "needs 0² > 0²" messages; fail early with the correct error.
  if (gl.isContextLost()) {
    throw new Error(
      `pt-webgl2: WebGL context lost — cannot create ${resourceName} texture`,
    );
  }

  // Dimension guard.
  const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  if (spec.kind === '2d') {
    if (spec.dim > maxSize) {
      throw new Error(
        `pt-webgl2: ${resourceName} needs a ${spec.dim}² texture but this device only supports ` +
          `${maxSize}² — reduce triangle count or split the scene.`,
      );
    }
  } else if (spec.kind === 'rect') {
    const maxDim = Math.max(spec.width, spec.height);
    if (maxDim > maxSize) {
      throw new Error(
        `pt-webgl2: ${resourceName} needs a ${spec.width}×${spec.height} texture but this device only ` +
          `supports ${maxSize}² — reduce the environment map resolution.`,
      );
    }
  } else {
    // array
    if (spec.dim > maxSize) {
      throw new Error(
        `pt-webgl2: ${resourceName} needs a ${spec.dim}² texture but this device only supports ` +
          `${maxSize}² — reduce scene complexity.`,
      );
    }
    const maxLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
    if (spec.layers > maxLayers) {
      throw new Error(
        `pt-webgl2: ${resourceName} needs ${spec.layers} array-texture layers but this device only supports ` +
          `${maxLayers} — reduce the number of unique material textures in the scene.`,
      );
    }
  }

  const tex = gl.createTexture();
  if (tex == null) {
    throw new Error(
      `pt-webgl2: WebGL context lost — cannot create ${resourceName} texture`,
    );
  }

  const target = spec.kind === 'array' ? gl.TEXTURE_2D_ARRAY : gl.TEXTURE_2D;
  gl.bindTexture(target, tex);
  gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  if (spec.kind === 'array') {
    gl.texImage3D(
      target,
      0,
      spec.internalFormat,
      spec.dim,
      spec.dim,
      spec.layers,
      0,
      spec.format,
      spec.type,
      spec.data,
    );
  } else {
    const w = spec.kind === 'rect' ? spec.width : spec.dim;
    const h = spec.kind === 'rect' ? spec.height : spec.dim;
    gl.texImage2D(target, 0, spec.internalFormat, w, h, 0, spec.format, spec.type, spec.data);
  }

  return tex;
}
