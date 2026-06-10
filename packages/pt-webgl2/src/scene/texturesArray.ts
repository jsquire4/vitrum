// texturesArray — the material-map texture atlas (`sampler2DArray textures`) the
// fork GLSL samples via `texture(textures, vec3(uv, material.<map>))`, where the
// per-map float is a LAYER index into this array (materialsTexture assigns them).
//
// THREE-free: a `TextureRef.handle` is opaque (`EnvironmentMapRef = unknown`). For
// the THREE-free path tracer the handle must expose CPU pixels — either a raw
// `{ width, height, data }` payload (the on-ramp form, like the env G2 bridge) or a
// THREE `DataTexture`-shaped `{ image: { data, width, height } }`. Image/ImageBitmap
// sources (canvas readback) are a documented host-side follow-up; this packer reads
// the DataTexture/raw forms that cover procedural + baked textures.
//
// All layers share one dimension (sampler2DArray requirement): the max source dim,
// nearest-neighbour resampled. Provenance: gkjohnson/three-gpu-pathtracer (MIT).

import type { MaterialSpec } from '@vitrum/core';

/** The material-map fields the fork GLSL samples (others are inert until wired).
 *  D3 (Wave C) added the clearcoat/sheen/iridescence/specular maps — the fork
 *  `get_surface_record` ALREADY samples each (see clearcoatMap…specularIntensityMap
 *  in get_surface_record_function.glsl.js), only the packer wired them as NO_TEXTURE.
 *  aoMap/lightMap/bumpMap are NEW GLSL (added to material_struct + get_surface_record)
 *  so they are gathered here too. */
const SAMPLED_MAP_KEYS = [
  'baseColorMap', 'metallicMap', 'roughnessMap', 'transmissionMap',
  'emissiveMap', 'normalMap', 'alphaMap',
  // D3 — clearcoat / sheen / iridescence / specular maps (GLSL already samples).
  'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap',
  'sheenColorMap', 'sheenRoughnessMap',
  'iridescenceMap', 'iridescenceThicknessMap',
  'specularColorMap', 'specularIntensityMap',
  // D3 — aoMap / lightMap / bumpMap (new GLSL consumption sites).
  'aoMap', 'lightMap', 'bumpMap',
] as const satisfies ReadonlyArray<keyof MaterialSpec>;

interface RawPixels {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array; // RGBA, row-major, linear
}

export interface TextureAtlas {
  /** RGBA32F, `layerCount` layers each `dim × dim`. */
  readonly data: Float32Array;
  readonly dim: number;
  readonly layerCount: number;
  /** Map a `TextureRef.handle` (object identity) → its layer index in the array. */
  readonly layerOf: Map<unknown, number>;
}

/** IEEE-754 half (uint16) → float32 (DataTextures may ship HalfFloat). */
function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

/** Read RGBA float pixels from a TextureRef handle (raw payload or DataTexture-shaped). */
function readHandlePixels(handle: unknown): RawPixels | null {
  const h = handle as {
    width?: number; height?: number; data?: ArrayLike<number>;
    image?: { width?: number; height?: number; data?: ArrayLike<number> };
  } | null;
  if (h == null) return null;
  // raw {width,height,data} (on-ramp form) OR DataTexture {image:{...}}
  const src = h.data ?? h.image?.data;
  const width = Number(h.width ?? h.image?.width ?? 0);
  const height = Number(h.height ?? h.image?.height ?? 0);
  if (src == null || typeof src.length !== 'number' || width <= 0 || height <= 0) return null;

  const stride = Math.max(1, Math.round(src.length / (width * height))); // 4 RGBA / 3 RGB / 1 R
  const isHalf = src instanceof Uint16Array;
  const isFloat = src instanceof Float32Array;
  const intMax = isHalf || isFloat ? 0 : 2 ** (8 * ((src as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1)) - 1;
  const dec = (v: number): number => (isHalf ? halfToFloat(v) : isFloat ? v : intMax > 0 ? v / intMax : v);

  const out = new Float32Array(width * height * 4);
  for (let p = 0; p < width * height; p += 1) {
    const s = p * stride;
    out[p * 4] = dec(Number(src[s] ?? 0));
    out[p * 4 + 1] = dec(Number(src[s + (stride > 1 ? 1 : 0)] ?? 0));
    out[p * 4 + 2] = dec(Number(src[s + (stride > 2 ? 2 : 0)] ?? 0));
    out[p * 4 + 3] = stride >= 4 ? dec(Number(src[s + 3] ?? 1)) : 1;
  }
  return { width, height, data: out };
}

/** Nearest-neighbour resample `px` into a `dim × dim` RGBA float layer at `data[base..]`. */
function blitLayer(px: RawPixels, dim: number, data: Float32Array, base: number): void {
  for (let y = 0; y < dim; y += 1) {
    const sy = Math.min(px.height - 1, (y * px.height / dim) | 0);
    for (let x = 0; x < dim; x += 1) {
      const sx = Math.min(px.width - 1, (x * px.width / dim) | 0);
      const s = (sy * px.width + sx) * 4;
      const d = base + (y * dim + x) * 4;
      data[d] = px.data[s]!; data[d + 1] = px.data[s + 1]!;
      data[d + 2] = px.data[s + 2]!; data[d + 3] = px.data[s + 3]!;
    }
  }
}

/**
 * Build the material-map atlas: gather every unique readable map handle across the
 * scene materials, resample each to a common dim, and assign a layer index. Returns
 * `null` (no atlas → all map ids stay -1) when no readable textures exist.
 */
export function packTextureAtlas(materials: readonly MaterialSpec[]): TextureAtlas | null {
  // unique handles in first-seen order
  const handles: unknown[] = [];
  const seen = new Set<unknown>();
  for (const m of materials) {
    for (const key of SAMPLED_MAP_KEYS) {
      const ref = m[key] as { handle?: unknown } | undefined;
      const handle = ref?.handle;
      if (handle != null && !seen.has(handle)) { seen.add(handle); handles.push(handle); }
    }
  }
  if (handles.length === 0) return null;

  // Read pixels; drop unreadable handles (their map id stays -1).
  // H7 (2026-06-09): warn on each silently-dropped handle so the host knows which
  // texture assets were skipped. An unreadable handle means the texture sampling
  // will use the default (flat/no texture) for the corresponding map.
  const pixels: { handle: unknown; px: RawPixels }[] = [];
  for (const handle of handles) {
    const px = readHandlePixels(handle);
    if (px != null) {
      pixels.push({ handle, px });
    } else {
      console.warn(
        '[pt-webgl2] texture handle is not readable (no raw {width,height,data} or DataTexture-shaped image); ' +
          'the texture map will be ignored and the material will render without it. ' +
          `Handle: ${String(handle)}`,
      );
    }
  }
  if (pixels.length === 0) return null;

  const dim = Math.max(...pixels.map((p) => Math.max(p.px.width, p.px.height)));
  const layerCount = pixels.length;
  const data = new Float32Array(dim * dim * 4 * layerCount);
  const layerOf = new Map<unknown, number>();
  pixels.forEach(({ handle, px }, layer) => {
    layerOf.set(handle, layer);
    blitLayer(px, dim, data, layer * dim * dim * 4);
  });
  return { data, dim, layerCount, layerOf };
}

/** Upload the atlas as an RGBA32F TEXTURE_2D_ARRAY (NEAREST, ClampToEdge). */
export function uploadTextureAtlas(gl: WebGL2RenderingContext, atlas: TextureAtlas): WebGLTexture {
  // Size guards: exceed MAX_TEXTURE_SIZE or MAX_ARRAY_TEXTURE_LAYERS and the
  // texImage3D silently fails on most drivers — throw an actionable error first.
  if (gl.isContextLost()) {
    throw new Error('pt-webgl2: WebGL context lost — cannot create material texture atlas');
  }
  const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  if (atlas.dim > maxSize) {
    throw new Error(
      `pt-webgl2: material texture atlas needs a ${atlas.dim}² layer but this device only supports ` +
        `${maxSize}² — reduce the resolution of material textures in the scene.`,
    );
  }
  const maxLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  if (atlas.layerCount > maxLayers) {
    throw new Error(
      `pt-webgl2: material texture atlas needs ${atlas.layerCount} layers but this device only supports ` +
        `${maxLayers} — reduce the number of unique material textures in the scene.`,
    );
  }
  const tex = gl.createTexture();
  if (tex == null) throw new Error('pt-webgl2: WebGL context lost — cannot create material texture atlas');
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage3D(
    gl.TEXTURE_2D_ARRAY, 0, gl.RGBA32F, atlas.dim, atlas.dim, atlas.layerCount,
    0, gl.RGBA, gl.FLOAT, atlas.data,
  );
  return tex;
}
