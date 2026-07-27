import {
  decodeNormalDepthWorldNormal,
  type ReadbackResult,
} from '@vitrum/shared-denoisers';

export type RgbDecode = (r: number, g: number, b: number) => readonly [number, number, number];

export type WebGlOidnReadbackResult = ReadbackResult;

/**
 * Convert a WebGL `readPixels(... RGBA, FLOAT, ...)` buffer into OIDN's RGB
 * HWC layout while flipping from GL's bottom-left origin to the engine's
 * top-left CPU convention.
 */
export function rgba32fBottomLeftToRgbF32(
  src: Float32Array,
  width: number,
  height: number,
  decode?: RgbDecode,
): Float32Array {
  if (width <= 0 || height <= 0) return new Float32Array(0);
  const expected = width * height * 4;
  if (src.length < expected) {
    throw new RangeError(
      `rgba32fBottomLeftToRgbF32: expected at least ${expected} floats, got ${src.length}`,
    );
  }
  const dst = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    const srcY = height - 1 - y;
    for (let x = 0; x < width; x++) {
      const srcOff = (srcY * width + x) * 4;
      const dstOff = (y * width + x) * 3;
      const r = src[srcOff]!;
      const g = src[srcOff + 1]!;
      const b = src[srcOff + 2]!;
      const decoded = decode != null ? decode(r, g, b) : ([r, g, b] as const);
      dst[dstOff] = decoded[0];
      dst[dstOff + 1] = decoded[1];
      dst[dstOff + 2] = decoded[2];
    }
  }
  return dst;
}

export interface WebGlOidnFramebufferSources {
  readonly colorFbo: WebGLFramebuffer | null;
  readonly auxFbo?: WebGLFramebuffer | null;
  readonly width: number;
  readonly height: number;
  readonly albedoAttachment?: GLenum | null;
  readonly normalDepthAttachment?: GLenum | null;
}

function readAttachmentRgb32f(
  gl: WebGL2RenderingContext,
  fbo: WebGLFramebuffer,
  attachment: GLenum,
  width: number,
  height: number,
  decode?: RgbDecode,
): Float32Array {
  const rgba = new Float32Array(width * height * 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.readBuffer(attachment);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, rgba);
  return rgba32fBottomLeftToRgbF32(rgba, width, height, decode);
}

export function readOidnInputsFromWebGlFbos(
  gl: WebGL2RenderingContext,
  sources: WebGlOidnFramebufferSources,
): WebGlOidnReadbackResult | null {
  const { colorFbo, auxFbo = null, width, height } = sources;
  if (width <= 0 || height <= 0 || colorFbo == null) return null;

  try {
    const color = readAttachmentRgb32f(gl, colorFbo, gl.COLOR_ATTACHMENT0, width, height);

    let albedo: Float32Array | undefined;
    if (auxFbo != null && sources.albedoAttachment != null) {
      albedo = readAttachmentRgb32f(gl, auxFbo, sources.albedoAttachment, width, height);
    }

    let normal: Float32Array | undefined;
    if (auxFbo != null && sources.normalDepthAttachment != null) {
      normal = readAttachmentRgb32f(
        gl,
        auxFbo,
        sources.normalDepthAttachment,
        width,
        height,
        decodeNormalDepthWorldNormal,
      );
    }

    return {
      color,
      ...(albedo !== undefined ? { albedo } : {}),
      ...(normal !== undefined ? { normal } : {}),
      width,
      height,
    };
  } finally {
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}
