// traceTier — the WebGL2 capability gate (WS5 §3). Mirrors pt-webgpu's
// `selectPtWebgpuTraceTier` / `resolvePtWebgpuTraceTier`, but WebGL2 has no
// storage buffers, so it gates on the WebGL2 limits that matter for the trace
// kernel instead:
//
//   - EXT_color_buffer_float : RGBA32F render targets are renderable. This is
//     REQUIRED at all (HDR accumulation has nowhere to go without it) — absence
//     throws rather than degrading.
//   - MAX_DRAW_BUFFERS >= 3   : MRT G-buffer (radiance + normal/depth + albedo).
//   - MAX_TEXTURE_IMAGE_UNITS >= 12 : the sampler budget the full kernel binds
//     (BVH, materials, lights, env, CMF, …).
//   - MAX_TEXTURE_SIZE >= 8192 : the square data textures (BVH nodes, vertex
//     attributes) the full pack packs into.
//
// `lite` tier: disables the auxiliary G-buffer outputs (gNormalDepth and gAlbedo
// at MRT attachments 1 and 2) by setting supportsAuxBuffers=false. This means
// FrameRendered.normalDepth and .albedo are null — denoising and post-processing
// that depend on those buffers have no input. The path-tracing kernel itself
// keeps the same bounce count, material-map sampling, optional BSDF lobes,
// spectral path, and emitter families as full tier; only aux-buffer products are
// missing below `full`.
// `lite` is the graceful-degradation tier for contexts where MAX_DRAW_BUFFERS < 3
// or MAX_TEXTURE_IMAGE_UNITS < 12 or MAX_TEXTURE_SIZE < 8192.
//
// `WebGl2TraceTier` is owned here (the tier-selection module) and re-exported by
// src/options.ts, so there is a single source of truth for the union.

export type WebGl2TraceTier = 'full' | 'lite';

const FULL_MIN_DRAW_BUFFERS = 3;
const FULL_MIN_TEXTURE_IMAGE_UNITS = 12;
const FULL_MIN_TEXTURE_SIZE = 8192;

/**
 * Pick full vs lite from WebGL2 limits. Throws if `EXT_color_buffer_float` is
 * absent (RGBA32F render targets are mandatory for HDR accumulation). When the
 * extension is present, returns `'full'` iff all of MAX_DRAW_BUFFERS,
 * MAX_TEXTURE_IMAGE_UNITS, and MAX_TEXTURE_SIZE meet the full-tier minimums;
 * otherwise `'lite'`.
 */
export function selectWebGl2TraceTier(gl: WebGL2RenderingContext): WebGl2TraceTier {
  const floatColor = gl.getExtension('EXT_color_buffer_float') != null;
  if (!floatColor) {
    throw new Error('pt-webgl2: EXT_color_buffer_float required (RGBA32F render targets)');
  }
  const drawBuffers = (gl.getParameter(gl.MAX_DRAW_BUFFERS) as number) ?? 1;
  const texUnits = (gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) as number) ?? 0;
  const maxTexSize = (gl.getParameter(gl.MAX_TEXTURE_SIZE) as number) ?? 0;
  if (
    drawBuffers >= FULL_MIN_DRAW_BUFFERS &&
    texUnits >= FULL_MIN_TEXTURE_IMAGE_UNITS &&
    maxTexSize >= FULL_MIN_TEXTURE_SIZE
  ) {
    return 'full';
  }
  return 'lite';
}

/**
 * Auto-detect unless `force` is set. `force: 'lite'` always succeeds (after the
 * mandatory EXT_color_buffer_float check). `force: 'full'` throws when the
 * context cannot meet the full-tier limits. Mirrors `resolvePtWebgpuTraceTier`.
 */
export function resolveWebGl2TraceTier(
  gl: WebGL2RenderingContext,
  force?: WebGl2TraceTier,
): WebGl2TraceTier {
  const tier = selectWebGl2TraceTier(gl); // also enforces EXT_color_buffer_float
  if (force === 'lite') {
    return 'lite';
  }
  if (force === 'full') {
    if (tier !== 'full') {
      const drawBuffers = gl.getParameter(gl.MAX_DRAW_BUFFERS) as number;
      const texUnits = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) as number;
      const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
      throw new Error(
        `pt-webgl2: traceTier=full requested but context reports ` +
          `MAX_DRAW_BUFFERS=${drawBuffers}, MAX_TEXTURE_IMAGE_UNITS=${texUnits}, ` +
          `MAX_TEXTURE_SIZE=${maxTexSize} (need >=${FULL_MIN_DRAW_BUFFERS} draw buffers, ` +
          `>=${FULL_MIN_TEXTURE_IMAGE_UNITS} texture units, >=${FULL_MIN_TEXTURE_SIZE} texture size).`,
      );
    }
    return 'full';
  }
  return tier;
}
