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

import type { EngineWarning, MaterialSpec } from '@vitrum/core';

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
  // KHR_materials_anisotropy: RG = tangent direction, B = strength.
  'anisotropyMap',
  // KHR_materials_volume: G = scalar multiplier for thicknessFactor.
  'thicknessMap',
] as const satisfies ReadonlyArray<keyof MaterialSpec>;

// ── D10.12: TextureHandleHint ─────────────────────────────────────────────────
// Optional hints that a TextureRef handle can expose to make readHandlePixels
// unambiguous without relying on the stride heuristic (which can mis-classify
// 3-channel RGB data, for example). A host that provides these hints avoids the
// ambiguous-stride console.warn and gets deterministic decoding.
//
// Usage: attach a `__vitrum_hint__` property to the texture handle object, OR
// pass a wrapper that implements this interface as the handle.
//
// channels: 1 | 2 | 3 | 4 — number of channels per pixel in `data`.
//   If omitted, readHandlePixels falls back to the existing stride heuristic.
// dataType: 'uint8' | 'uint16' | 'float32' — encoding of each channel value.
//   If omitted, inferred from the ArrayLike type (Uint8Array→uint8 etc.).

export interface TextureHandleHint {
  readonly channels?: 1 | 2 | 3 | 4;
  readonly dataType?: 'uint8' | 'uint16' | 'float32';
  /**
   * Source encoding hint. By default, color/tint map roles are treated as sRGB
   * sources and converted into the atlas' linear RGBA32F payload; scalar/data
   * map roles stay linear. Set `colorSpace:'linear'` for a color map handle
   * that is already linear-light.
   */
  readonly colorSpace?: TextureSampleColorSpace;
}

export type TextureSampleColorSpace = 'srgb' | 'linear';

export interface TextureAtlasLayerMap {
  readonly srgb: ReadonlyMap<unknown, number>;
  readonly linear: ReadonlyMap<unknown, number>;
}

interface RawPixels {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array; // RGBA, row-major, normalized source values
  readonly sourceColorSpace?: TextureSampleColorSpace;
}

export interface TextureAtlas {
  /** RGBA32F, `layerCount` layers each `dim × dim`. */
  readonly data: Float32Array;
  readonly dim: number;
  readonly layerCount: number;
  /** Back-compat default map: first layer for a handle, regardless of role. */
  readonly layerOf: Map<unknown, number>;
  /** Role-aware layer maps; color/tint maps and data maps can share a handle safely. */
  readonly layerOfByColorSpace: TextureAtlasLayerMap;
}

export function textureAtlasLayerCapacity(layerCount: number, maxLayers: number): number {
  const count = Math.max(0, Math.floor(layerCount));
  const limit = Math.max(0, Math.floor(maxLayers));
  if (count === 0 || limit === 0) return 0;
  let capacity = 1;
  while (capacity < count + 1) capacity *= 2;
  return Math.min(limit, Math.max(count, capacity));
}

export interface TextureAtlasBuildOptions {
  readonly onWarning?: (warning: EngineWarning) => void;
  readonly warningPhase?: string;
  readonly warningMethod?: string;
}

function handleType(handle: unknown): string {
  return handle == null ? 'null' : Object.prototype.toString.call(handle);
}

function emitTextureWarning(
  options: TextureAtlasBuildOptions | undefined,
  warning: Omit<EngineWarning, 'backend' | 'phase' | 'method'>,
): void {
  const routed: EngineWarning = {
    ...warning,
    backend: 'pt-webgl2',
    phase: options?.warningPhase ?? 'scene-upload',
    ...(options?.warningMethod != null ? { method: options.warningMethod } : {}),
  };
  if (options?.onWarning != null) {
    options.onWarning(routed);
  } else {
    console.warn(routed.message);
  }
}

const SRGB_MAP_KEYS = new Set<keyof MaterialSpec>([
  'baseColorMap',
  'emissiveMap',
  'sheenColorMap',
  'specularColorMap',
]);

export function textureColorSpaceForMapKey(key: keyof MaterialSpec): TextureSampleColorSpace {
  return SRGB_MAP_KEYS.has(key) ? 'srgb' : 'linear';
}

function collectHandle(
  handles: { handle: unknown; colorSpace: TextureSampleColorSpace }[],
  seen: Map<unknown, Set<TextureSampleColorSpace>>,
  ref: { readonly handle?: unknown } | undefined,
  colorSpace: TextureSampleColorSpace,
): void {
  const handle = ref?.handle;
  if (handle == null) return;
  let seenSpaces = seen.get(handle);
  if (seenSpaces == null) {
    seenSpaces = new Set<TextureSampleColorSpace>();
    seen.set(handle, seenSpaces);
  }
  if (!seenSpaces.has(colorSpace)) {
    seenSpaces.add(colorSpace);
    handles.push({ handle, colorSpace });
  }
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

function srgbToLinear(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/** Read RGBA float pixels from a TextureRef handle (raw payload or DataTexture-shaped).
 *  D10.12: optional TextureHandleHint (`__vitrum_hint__` property on the handle, or the
 *  handle itself implementing TextureHandleHint) provides explicit channels/dataType to
 *  avoid the stride heuristic. A console.warn is emitted when the stride is ambiguous
 *  (not 1 or 4) and no hint is present, so hosts know to supply a hint. */
function readHandlePixels(
  handle: unknown,
  options?: TextureAtlasBuildOptions,
): RawPixels | null {
  const h = handle as {
    width?: number; height?: number; data?: ArrayLike<number>;
    image?: { width?: number; height?: number; data?: ArrayLike<number> };
    // D10.12: optional hint as a direct property on the handle
    __vitrum_hint__?: TextureHandleHint;
    channels?: number;
    dataType?: string;
    colorSpace?: string;
  } | null;
  if (h == null) return null;
  // raw {width,height,data} (on-ramp form) OR DataTexture {image:{...}}
  const src = h.data ?? h.image?.data;
  const width = Number(h.width ?? h.image?.width ?? 0);
  const height = Number(h.height ?? h.image?.height ?? 0);
  if (src == null || typeof src.length !== 'number' || width <= 0 || height <= 0) return null;

  // D10.12: resolve hint from __vitrum_hint__ property, or direct channels/dataType on handle.
  // Use type assertions at the object-literal level to satisfy exactOptionalPropertyTypes:
  // only include a property in the literal when the source value is non-null.
  const hint: TextureHandleHint | undefined = h.__vitrum_hint__ ?? (
    (h.channels != null || h.dataType != null)
      ? Object.assign(
          {} as TextureHandleHint,
          h.channels != null ? { channels: h.channels as TextureHandleHint['channels'] } : {},
          h.dataType != null ? { dataType: h.dataType as TextureHandleHint['dataType'] } : {},
          h.colorSpace != null ? { colorSpace: h.colorSpace as TextureHandleHint['colorSpace'] } : {},
        )
      : undefined
  );

  // Determine stride: hint takes priority; fall back to heuristic.
  const heuristicStride = Math.max(1, Math.round(src.length / (width * height)));
  const stride: number = hint?.channels ?? heuristicStride;

  // Warn on ambiguous stride (2 or 3 channels) without a hint — the heuristic
  // cannot distinguish 2-channel (RG) from a 2x oversized RGBA, for example.
  if (hint == null && stride !== 1 && stride !== 4) {
    emitTextureWarning(options, {
      code: 'pt-webgl2.texture-ambiguous-pixel-stride',
      message:
        `[pt-webgl2] texture handle has ambiguous pixel stride ${stride} ` +
        `(${src.length} values / ${width}×${height} pixels). ` +
        'Attach a __vitrum_hint__ = { channels: N } to the handle to resolve it deterministically.',
      details: {
        stride,
        valueCount: src.length,
        width,
        height,
        handleType: handleType(handle),
      },
    });
  }

  const isHalf = src instanceof Uint16Array;
  const isFloat = src instanceof Float32Array;
  // D10.12: respect explicit dataType hint for decoding.
  const hintIsHalf = hint?.dataType === 'uint16';
  const hintIsFloat = hint?.dataType === 'float32';
  const useHalf = hint?.dataType != null ? hintIsHalf : isHalf;
  const useFloat = hint?.dataType != null ? hintIsFloat : isFloat;
  const bpe = (src as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
  const intMax = useHalf || useFloat ? 0 : 2 ** (8 * bpe) - 1;
  const dec = (v: number): number => (useHalf ? halfToFloat(v) : useFloat ? v : intMax > 0 ? v / intMax : v);

  const out = new Float32Array(width * height * 4);
  for (let p = 0; p < width * height; p += 1) {
    const s = p * stride;
    out[p * 4] = dec(Number(src[s] ?? 0));
    out[p * 4 + 1] = dec(Number(src[s + (stride > 1 ? 1 : 0)] ?? 0));
    out[p * 4 + 2] = dec(Number(src[s + (stride > 2 ? 2 : 0)] ?? 0));
    out[p * 4 + 3] = stride >= 4 ? dec(Number(src[s + 3] ?? 1)) : 1;
  }
  const sourceColorSpace =
    hint?.colorSpace === 'srgb' || hint?.colorSpace === 'linear'
      ? hint.colorSpace
      : undefined;
  return { width, height, data: out, ...(sourceColorSpace ? { sourceColorSpace } : {}) };
}

/** Nearest-neighbour resample `px` into a `dim × dim` RGBA float layer at `data[base..]`. */
function blitLayer(
  px: RawPixels,
  dim: number,
  data: Float32Array,
  base: number,
  colorSpace: TextureSampleColorSpace,
): void {
  const decodeSrgb = colorSpace === 'srgb' && px.sourceColorSpace !== 'linear';
  for (let y = 0; y < dim; y += 1) {
    const sy = Math.min(px.height - 1, (y * px.height / dim) | 0);
    for (let x = 0; x < dim; x += 1) {
      const sx = Math.min(px.width - 1, (x * px.width / dim) | 0);
      const s = (sy * px.width + sx) * 4;
      const d = base + (y * dim + x) * 4;
      const r = px.data[s]!;
      const g = px.data[s + 1]!;
      const b = px.data[s + 2]!;
      data[d] = decodeSrgb ? srgbToLinear(r) : r;
      data[d + 1] = decodeSrgb ? srgbToLinear(g) : g;
      data[d + 2] = decodeSrgb ? srgbToLinear(b) : b;
      data[d + 3] = px.data[s + 3]!;
    }
  }
}

/**
 * Build the material-map atlas: gather every unique readable map handle across the
 * scene materials, resample each to a common dim, and assign a layer index. Returns
 * `null` (no atlas → all map ids stay -1) when no readable textures exist.
 */
export function packTextureAtlas(
  materials: readonly MaterialSpec[],
  options?: TextureAtlasBuildOptions,
): TextureAtlas | null {
  // unique (handle, color-space role) pairs in first-seen order
  const handles: { handle: unknown; colorSpace: TextureSampleColorSpace }[] = [];
  const seen = new Map<unknown, Set<TextureSampleColorSpace>>();
  for (const m of materials) {
    for (const key of SAMPLED_MAP_KEYS) {
      const ref = m[key] as { handle?: unknown } | undefined;
      collectHandle(handles, seen, ref, textureColorSpaceForMapKey(key));
    }
    collectHandle(handles, seen, m.frontLayer?.normalMap, 'linear');
    collectHandle(handles, seen, m.backLayer?.normalMap, 'linear');
  }
  if (handles.length === 0) return null;

  // Read pixels; drop unreadable handles (their map id stays -1).
  // H7 (2026-06-09): warn on each silently-dropped handle so the host knows which
  // texture assets were skipped. An unreadable handle means the texture sampling
  // will use the default (flat/no texture) for the corresponding map.
  const pixels: { handle: unknown; colorSpace: TextureSampleColorSpace; px: RawPixels }[] = [];
  for (const { handle, colorSpace } of handles) {
    const px = readHandlePixels(handle, options);
    if (px != null) {
      pixels.push({ handle, colorSpace, px });
    } else {
      emitTextureWarning(options, {
        code: 'pt-webgl2.texture-unreadable',
        message:
          '[pt-webgl2] texture handle is not readable (no raw {width,height,data} or DataTexture-shaped image); ' +
          'the texture map will be ignored and the material will render without it. ' +
          `Handle: ${String(handle)}`,
        details: {
          colorSpace,
          handleType: handleType(handle),
          handle: String(handle),
        },
      });
    }
  }
  if (pixels.length === 0) return null;

  const dim = Math.max(...pixels.map((p) => Math.max(p.px.width, p.px.height)));
  const layerCount = pixels.length;
  const data = new Float32Array(dim * dim * 4 * layerCount);
  const layerOf = new Map<unknown, number>();
  const layerOfByColorSpace: TextureAtlasLayerMap = {
    srgb: new Map<unknown, number>(),
    linear: new Map<unknown, number>(),
  };
  pixels.forEach(({ handle, colorSpace, px }, layer) => {
    if (!layerOf.has(handle)) layerOf.set(handle, layer);
    (layerOfByColorSpace[colorSpace] as Map<unknown, number>).set(handle, layer);
    blitLayer(px, dim, data, layer * dim * dim * 4, colorSpace);
  });
  return { data, dim, layerCount, layerOf, layerOfByColorSpace };
}

/** Upload the atlas as an RGBA32F TEXTURE_2D_ARRAY (NEAREST, ClampToEdge). */
export function uploadTextureAtlas(
  gl: WebGL2RenderingContext,
  atlas: TextureAtlas,
  opts?: { readonly layerCapacity?: number },
): WebGLTexture {
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
  const layerCapacity = opts?.layerCapacity ?? textureAtlasLayerCapacity(atlas.layerCount, maxLayers);
  if (layerCapacity < atlas.layerCount || layerCapacity > maxLayers) {
    throw new Error(
      `pt-webgl2: material texture atlas allocation requested ${layerCapacity} layers for ` +
        `${atlas.layerCount} live layers on a device with ${maxLayers} maximum layers.`,
    );
  }
  const tex = gl.createTexture();
  if (tex == null) throw new Error('pt-webgl2: WebGL context lost — cannot create material texture atlas');
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const uploadData = layerCapacity === atlas.layerCount
    ? atlas.data
    : (() => {
        const expanded = new Float32Array(atlas.dim * atlas.dim * 4 * layerCapacity);
        expanded.set(atlas.data);
        return expanded;
      })();
  gl.texImage3D(
    gl.TEXTURE_2D_ARRAY, 0, gl.RGBA32F, atlas.dim, atlas.dim, layerCapacity,
    0, gl.RGBA, gl.FLOAT, uploadData,
  );
  return tex;
}

export function updateTextureAtlasLayers(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
  atlas: TextureAtlas,
): void {
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
  gl.texSubImage3D(
    gl.TEXTURE_2D_ARRAY,
    0,
    0,
    0,
    0,
    atlas.dim,
    atlas.dim,
    atlas.layerCount,
    gl.RGBA,
    gl.FLOAT,
    atlas.data,
  );
}
