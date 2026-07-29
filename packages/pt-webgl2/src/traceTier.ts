// traceTier — the WebGL2 capability gate (WS5 §3). Mirrors pt-webgpu's
// `selectPtWebgpuTraceTier` / `resolvePtWebgpuTraceTier`, but WebGL2 has no
// storage buffers, so it gates on the WebGL2 limits that matter for the trace
// kernel instead:
//
//   - EXT_color_buffer_float : RGBA32F render targets are renderable. This is
//     REQUIRED at all (HDR accumulation has nowhere to go without it) — absence
//     throws rather than degrading.
//   - MAX_DRAW_BUFFERS >= 4 and MAX_COLOR_ATTACHMENTS >= 4: mandatory for the
//     four-attachment NEE candidate handoff used by BOTH tiers. The optional
//     full-tier G-buffer needs only three, so it cannot provide a meaningful
//     fallback below this shared kernel floor.
//   - MAX_TEXTURE_IMAGE_UNITS >= 16: the maximum selectable composed graph
//     (normalized + radiance material atlases, environment, Sobol, and BDPT) uses sixteen active
//     fragment samplers. WebGL2 requires implementations to expose at least 16,
//     but checking the real value fails closed on broken/virtualized contexts.
//
// `lite` tier: disables the renderer's auxiliary G-buffer outputs
// (gNormalDepth and gAlbedo at MRT attachments 1 and 2). This means
// FrameRendered.normalDepth and .albedo are absent — denoising and
// post-processing that depend on those buffers have no input. The path-tracing kernel itself
// keeps the same bounce count, material-map sampling, optional BSDF lobes,
// spectral path, and emitter families as full tier; only aux-buffer products are
// missing below `full`. It is an explicit lower-memory/output profile, not a
// way to bypass limits shared by the unchanged trace kernel.
//
// Texture-unit capacity is a shared kernel floor, not a tier dimension.
// Texture dimensions are scene-specific and are validated by each texture
// allocation against MAX_TEXTURE_SIZE.
//
// `WebGl2TraceTier` is owned here (the tier-selection module) and re-exported by
// src/options.ts, so there is a single source of truth for the union.

export type WebGl2TraceTier = 'full' | 'lite';

const REQUIRED_DRAW_BUFFERS = 4;
const REQUIRED_COLOR_ATTACHMENTS = 4;
const REQUIRED_TEXTURE_IMAGE_UNITS = 16;

/**
 * Validate the limits shared by both tiers and select the ordinary full profile.
 * `lite` is selected only when a host explicitly requests its lower-memory,
 * no-G-buffer output shape.
 */
export function selectWebGl2TraceTier(gl: WebGL2RenderingContext): WebGl2TraceTier {
  const floatColor = gl.getExtension('EXT_color_buffer_float') != null;
  if (!floatColor) {
    throw new Error('pt-webgl2: EXT_color_buffer_float required (RGBA32F render targets)');
  }
  const drawBuffers = (gl.getParameter(gl.MAX_DRAW_BUFFERS) as number) ?? 1;
  const colorAttachments =
    (gl.getParameter(gl.MAX_COLOR_ATTACHMENTS) as number) ?? 1;
  const textureImageUnits =
    (gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) as number) ?? 0;
  if (
    drawBuffers < REQUIRED_DRAW_BUFFERS ||
    colorAttachments < REQUIRED_COLOR_ATTACHMENTS ||
    textureImageUnits < REQUIRED_TEXTURE_IMAGE_UNITS
  ) {
    throw new Error(
      `pt-webgl2: the trace kernel requires at least ${REQUIRED_DRAW_BUFFERS} draw buffers, ` +
        `${REQUIRED_COLOR_ATTACHMENTS} color attachments for its NEE candidate handoff, ` +
        `and ${REQUIRED_TEXTURE_IMAGE_UNITS} fragment texture units for its maximum composed graph ` +
        `(context reports MAX_DRAW_BUFFERS=${drawBuffers}, ` +
        `MAX_COLOR_ATTACHMENTS=${colorAttachments}, ` +
        `MAX_TEXTURE_IMAGE_UNITS=${textureImageUnits})`,
    );
  }
  return 'full';
}

/**
 * Validate shared kernel requirements, then honor an explicit lower-memory
 * `lite` request. Auto and explicit `full` both select the full output profile.
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
    return 'full';
  }
  return tier;
}
