import type { MaterialSpec, TextureRef, TextureWrapMode, UvTransform } from '@vitrum/core';

interface RawHeightPixels {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number>;
  readonly channels: 1 | 2 | 3 | 4;
  readonly decode: (value: number) => number;
}

type DisplacementWarningSink = (warning: string) => void;

const MAX_DISPLACEMENT_SUBDIVISIONS = 4;
const MAX_MICRODISPLACED_TRIANGLES = 262_144;

export interface MicrodisplacedMeshGeometry {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly indices: Uint32Array;
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;
  readonly tangents?: Float32Array;
  readonly colors?: Float32Array;
  readonly subdivisions: number;
}

function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

function finiteOr(value: number | undefined, fallback: number): number {
  return value != null && Number.isFinite(value) ? value : fallback;
}

function textureHandleType(handle: unknown): string {
  return handle == null ? 'null' : Object.prototype.toString.call(handle);
}

function textureRefSourcePath(ref: TextureRef | undefined): string | undefined {
  if (ref == null) return undefined;
  for (const symbol of Object.getOwnPropertySymbols(ref)) {
    if (symbol.description !== 'vitrum.gltf.textureRefSource') continue;
    const value = (ref as unknown as Record<symbol, unknown>)[symbol];
    if (value == null || typeof value !== 'object') return undefined;
    const path = (value as { readonly path?: unknown }).path;
    return typeof path === 'string' ? path : undefined;
  }
  return undefined;
}

function warningPrefix(primitiveId: string, ref?: TextureRef): string {
  const sourcePath = textureRefSourcePath(ref);
  return `Primitive "${primitiveId}" displacementMap${sourcePath !== undefined ? ` at ${sourcePath}` : ''}`;
}

function handlePayload(handle: unknown): {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number> | undefined;
  readonly hint?: { readonly channels?: number; readonly dataType?: string };
} | null {
  if (handle == null || typeof handle !== 'object') return null;
  const h = handle as {
    readonly width?: number;
    readonly height?: number;
    readonly data?: ArrayLike<number>;
    readonly image?: {
      readonly width?: number;
      readonly height?: number;
      readonly data?: ArrayLike<number>;
    };
    readonly __vitrum_hint__?: { readonly channels?: number; readonly dataType?: string };
    readonly channels?: number;
    readonly dataType?: string;
  };
  const image = h.image;
  const hint = h.__vitrum_hint__ ?? (
    h.channels != null || h.dataType != null
      ? {
          ...(h.channels != null ? { channels: h.channels } : {}),
          ...(h.dataType != null ? { dataType: h.dataType } : {}),
        }
      : undefined
  );
  return {
    width: Number(h.width ?? image?.width ?? 0),
    height: Number(h.height ?? image?.height ?? 0),
    data: h.data ?? image?.data,
    ...(hint != null ? { hint } : {}),
  };
}

function decoderFor(data: ArrayLike<number>, dataType: string | undefined): (value: number) => number {
  if (dataType === 'float32' || data instanceof Float32Array || data instanceof Float64Array) {
    return (value) => value;
  }
  if (dataType === 'float16' || dataType === 'half-float') {
    return (value) => halfToFloat(value);
  }
  if (dataType === 'uint16') {
    return (value) => Math.min(1, Math.max(0, value / 65535));
  }
  if (data instanceof Uint16Array) {
    return (value) => Math.min(1, Math.max(0, value / 65535));
  }
  if (
    dataType === 'uint8' ||
    data instanceof Uint8Array ||
    data instanceof Uint8ClampedArray ||
    data instanceof Int8Array
  ) {
    return (value) => Math.min(1, Math.max(0, value / 255));
  }
  const bytesPerElement = (data as { readonly BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT;
  if (typeof bytesPerElement === 'number' && bytesPerElement > 0) {
    const max = 2 ** (8 * bytesPerElement) - 1;
    return (value) => Math.min(1, Math.max(0, value / max));
  }
  return (value) => value;
}

function readHeightPixels(
  ref: TextureRef,
  primitiveId: string,
  warn: DisplacementWarningSink,
): RawHeightPixels | null {
  const payload = handlePayload(ref.handle);
  if (payload == null) {
    warn(`${warningPrefix(primitiveId, ref)} handle is not CPU-readable (${textureHandleType(ref.handle)}); displacement skipped.`);
    return null;
  }
  const { width, height, data, hint } = payload;
  if (width <= 0 || height <= 0 || data == null || typeof data.length !== 'number') {
    warn(`${warningPrefix(primitiveId, ref)} handle is missing positive width/height/data; displacement skipped.`);
    return null;
  }
  const pixelCount = width * height;
  const inferredChannels = data.length / pixelCount;
  const channels = hint?.channels ?? inferredChannels;
  if (![1, 2, 3, 4].includes(channels) || !Number.isInteger(channels)) {
    warn(
      `${warningPrefix(primitiveId, ref)} has ${data.length} values for ${width}x${height} pixels; ` +
      'expected 1, 2, 3, or 4 channels. Displacement skipped.',
    );
    return null;
  }
  return {
    width,
    height,
    data,
    channels: channels as 1 | 2 | 3 | 4,
    decode: decoderFor(data, hint?.dataType),
  };
}

function wrapCoord(value: number, mode: TextureWrapMode | undefined): number {
  if (!Number.isFinite(value)) return 0;
  switch (mode ?? 'repeat') {
    case 'clamp-to-edge':
      return Math.min(1, Math.max(0, value));
    case 'mirrored-repeat': {
      const period = Math.floor(value);
      const frac = value - period;
      return Math.abs(period % 2) === 1 ? 1 - frac : frac;
    }
    case 'repeat':
    default:
      return value - Math.floor(value);
  }
}

function applyUvTransform(u: number, v: number, transform: UvTransform | undefined): readonly [number, number] {
  const sx = finiteOr(transform?.scale?.[0], 1);
  const sy = finiteOr(transform?.scale?.[1], 1);
  const rotation = finiteOr(transform?.rotation, 0);
  const x = u * sx;
  const y = v * sy;
  const c = Math.cos(rotation);
  const s = Math.sin(rotation);
  return [
    x * c - y * s + finiteOr(transform?.offset?.[0], 0),
    x * s + y * c + finiteOr(transform?.offset?.[1], 0),
  ];
}

function redAt(pixels: RawHeightPixels, x: number, y: number): number {
  const clampedX = Math.min(pixels.width - 1, Math.max(0, x));
  const clampedY = Math.min(pixels.height - 1, Math.max(0, y));
  const offset = (clampedY * pixels.width + clampedX) * pixels.channels;
  return pixels.decode(Number(pixels.data[offset] ?? 0));
}

function sampleHeight(pixels: RawHeightPixels, u: number, v: number, ref: TextureRef): number {
  const [tu, tv] = applyUvTransform(u, v, ref.transform);
  const wu = wrapCoord(tu, ref.wrapS);
  const wv = wrapCoord(tv, ref.wrapT);
  if (pixels.width === 1 && pixels.height === 1) {
    return redAt(pixels, 0, 0);
  }
  const x = wu * Math.max(0, pixels.width - 1);
  const y = wv * Math.max(0, pixels.height - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(pixels.width - 1, x0 + 1);
  const y1 = Math.min(pixels.height - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const h00 = redAt(pixels, x0, y0);
  const h10 = redAt(pixels, x1, y0);
  const h01 = redAt(pixels, x0, y1);
  const h11 = redAt(pixels, x1, y1);
  const hx0 = h00 * (1 - fx) + h10 * fx;
  const hx1 = h01 * (1 - fx) + h11 * fx;
  return hx0 * (1 - fy) + hx1 * fy;
}

function normalize3(x: number, y: number, z: number): readonly [number, number, number] {
  const len = Math.hypot(x, y, z);
  if (len > 1e-8 && Number.isFinite(len)) return [x / len, y / len, z / len];
  return [0, 1, 0];
}

/**
 * Apply CPU-readable displacement to authored mesh vertices only. For opt-in
 * diced geometry, use {@link maybeMicrodisplaceMeshGeometry}.
 */
export function maybeDisplaceMeshPositions(
  input: {
    readonly primitiveId: string;
    readonly material: MaterialSpec;
    readonly positions: Float32Array;
    readonly normals: Float32Array;
    readonly uvs?: Float32Array;
    readonly uv1?: Float32Array;
    readonly onWarning?: DisplacementWarningSink;
  },
): Float32Array | null {
  const ref = input.material.displacementMap;
  if (ref == null) return null;
  const warn = input.onWarning ?? (() => {});
  const texCoord = ref.texCoord ?? 0;
  if (texCoord !== 0 && texCoord !== 1) {
    warn(`${warningPrefix(input.primitiveId, ref)} requests TEXCOORD_${texCoord}; vertex displacement supports TEXCOORD_0/1 only. Skipped.`);
    return null;
  }
  const uvSource = texCoord === 1 ? input.uv1 : input.uvs;
  if (uvSource == null || uvSource.length === 0) {
    warn(`${warningPrefix(input.primitiveId, ref)} requests TEXCOORD_${texCoord}, but that UV channel is absent; displacement skipped.`);
    return null;
  }
  const pixels = readHeightPixels(ref, input.primitiveId, warn);
  if (pixels == null) return null;

  const vertexCount = Math.floor(input.positions.length / 3);
  const out = new Float32Array(input.positions.length);
  const scale = finiteOr(input.material.displacementScale, 1);
  const bias = finiteOr(input.material.displacementBias, 0);
  for (let i = 0; i < vertexCount; i += 1) {
    const px = input.positions[i * 3] ?? 0;
    const py = input.positions[i * 3 + 1] ?? 0;
    const pz = input.positions[i * 3 + 2] ?? 0;
    const [nx, ny, nz] = normalize3(
      input.normals[i * 3] ?? 0,
      input.normals[i * 3 + 1] ?? 1,
      input.normals[i * 3 + 2] ?? 0,
    );
    const u = uvSource[i * 2] ?? 0;
    const v = uvSource[i * 2 + 1] ?? 0;
    const amount = sampleHeight(pixels, u, v, ref) * scale + bias;
    out[i * 3] = px + nx * amount;
    out[i * 3 + 1] = py + ny * amount;
    out[i * 3 + 2] = pz + nz * amount;
  }
  return out;
}


function resolveDisplacementSubdivisions(
  material: MaterialSpec,
  primitiveId: string,
  warn: DisplacementWarningSink,
): number {
  const ref = material.displacementMap;
  const raw = material.displacementSubdivisions;
  if (raw == null || raw <= 0) return 0;
  if (!Number.isFinite(raw)) {
    warn(`${warningPrefix(primitiveId, ref)} displacementSubdivisions is non-finite; microdisplacement disabled.`);
    return 0;
  }
  const rounded = Math.floor(raw);
  if (rounded < 1) return 0;
  if (rounded !== raw) {
    warn(`${warningPrefix(primitiveId, ref)} displacementSubdivisions=${raw} is not an integer; using ${rounded}.`);
  }
  if (rounded > MAX_DISPLACEMENT_SUBDIVISIONS) {
    warn(
      `${warningPrefix(primitiveId, ref)} displacementSubdivisions=${rounded} exceeds the shared-BVH cap ` +
      `${MAX_DISPLACEMENT_SUBDIVISIONS}; using ${MAX_DISPLACEMENT_SUBDIVISIONS}.`,
    );
    return MAX_DISPLACEMENT_SUBDIVISIONS;
  }
  return rounded;
}

function generatedTriangleList(vertexCount: number): Uint32Array {
  const indices = new Uint32Array(vertexCount);
  for (let i = 0; i < vertexCount; i += 1) indices[i] = i;
  return indices;
}

function lerpScalar(
  values: Float32Array,
  stride: number,
  ia: number,
  ib: number,
  ic: number,
  component: number,
  wa: number,
  wb: number,
  wc: number,
  fallback: number,
): number {
  return (values[ia * stride + component] ?? fallback) * wa +
    (values[ib * stride + component] ?? fallback) * wb +
    (values[ic * stride + component] ?? fallback) * wc;
}

function accumulateFaceNormal(
  positions: readonly number[],
  normals: Float64Array,
  ia: number,
  ib: number,
  ic: number,
): void {
  const ax = positions[ia * 3] ?? 0;
  const ay = positions[ia * 3 + 1] ?? 0;
  const az = positions[ia * 3 + 2] ?? 0;
  const bx = positions[ib * 3] ?? 0;
  const by = positions[ib * 3 + 1] ?? 0;
  const bz = positions[ib * 3 + 2] ?? 0;
  const cx = positions[ic * 3] ?? 0;
  const cy = positions[ic * 3 + 1] ?? 0;
  const cz = positions[ic * 3 + 2] ?? 0;
  const abx = bx - ax;
  const aby = by - ay;
  const abz = bz - az;
  const acx = cx - ax;
  const acy = cy - ay;
  const acz = cz - az;
  const nx = aby * acz - abz * acy;
  const ny = abz * acx - abx * acz;
  const nz = abx * acy - aby * acx;
  if (!Number.isFinite(nx) || !Number.isFinite(ny) || !Number.isFinite(nz)) return;
  for (const i of [ia, ib, ic]) {
    normals[i * 3] = (normals[i * 3] ?? 0) + nx;
    normals[i * 3 + 1] = (normals[i * 3 + 1] ?? 0) + ny;
    normals[i * 3 + 2] = (normals[i * 3 + 2] ?? 0) + nz;
  }
}

/**
 * Dice triangle-list geometry and apply CPU-readable displacement at generated
 * vertices before BVH construction. This is intentionally uniform and bounded:
 * it is the first real microgeometry contract, not an adaptive micropolygon
 * renderer. Returns null when `displacementSubdivisions` is absent/zero or when
 * the requested map/UVs/cap cannot be honored, allowing callers to fall back to
 * the legacy vertex-only path.
 */
export function maybeMicrodisplaceMeshGeometry(input: {
  readonly primitiveId: string;
  readonly material: MaterialSpec;
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly indices?: Uint32Array | Uint16Array;
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;
  readonly tangents?: Float32Array;
  readonly colors?: Float32Array;
  readonly onWarning?: DisplacementWarningSink;
}): MicrodisplacedMeshGeometry | null {
  const ref = input.material.displacementMap;
  if (ref == null) return null;
  const warn = input.onWarning ?? (() => {});
  const subdivisions = resolveDisplacementSubdivisions(input.material, input.primitiveId, warn);
  if (subdivisions <= 0) return null;
  const texCoord = ref.texCoord ?? 0;
  if (texCoord !== 0 && texCoord !== 1) {
    warn(`${warningPrefix(input.primitiveId, ref)} requests TEXCOORD_${texCoord}; microdisplacement supports TEXCOORD_0/1 only. Falling back to vertex displacement.`);
    return null;
  }
  const uvSource = texCoord === 1 ? input.uv1 : input.uvs;
  if (uvSource == null || uvSource.length === 0) {
    warn(`${warningPrefix(input.primitiveId, ref)} requests TEXCOORD_${texCoord}, but that UV channel is absent; microdisplacement disabled.`);
    return null;
  }
  const pixels = readHeightPixels(ref, input.primitiveId, warn);
  if (pixels == null) return null;

  const vertexCount = Math.floor(input.positions.length / 3);
  const sourceIndices = input.indices ?? generatedTriangleList(vertexCount);
  const sourceTriCount = Math.floor(sourceIndices.length / 3);
  const steps = 1 << subdivisions;
  const generatedTriCount = sourceTriCount * steps * steps;
  if (generatedTriCount > MAX_MICRODISPLACED_TRIANGLES) {
    warn(
      `${warningPrefix(input.primitiveId, ref)} displacementSubdivisions=${subdivisions} would generate ` +
      `${generatedTriCount} triangles, above the shared-BVH safety cap ${MAX_MICRODISPLACED_TRIANGLES}; ` +
      'falling back to vertex displacement.',
    );
    return null;
  }

  const outPositions: number[] = [];
  const outNormalFallback: number[] = [];
  const outUvs: number[] | undefined = input.uvs != null ? [] : undefined;
  const outUv1: number[] | undefined = input.uv1 != null ? [] : undefined;
  const hasTangents = input.tangents != null && input.tangents.length >= vertexCount * 4;
  const outTangents: number[] | undefined = hasTangents ? [] : undefined;
  const colorStride = input.colors != null && input.colors.length >= vertexCount * 4
    ? 4
    : input.colors != null && input.colors.length >= vertexCount * 3
      ? 3
      : 0;
  const outColors: number[] | undefined = colorStride > 0 ? [] : undefined;
  const outIndices: number[] = [];
  const scale = finiteOr(input.material.displacementScale, 1);
  const bias = finiteOr(input.material.displacementBias, 0);

  const pushVertex = (
    ia: number,
    ib: number,
    ic: number,
    wa: number,
    wb: number,
    wc: number,
  ): number => {
    const px = lerpScalar(input.positions, 3, ia, ib, ic, 0, wa, wb, wc, 0);
    const py = lerpScalar(input.positions, 3, ia, ib, ic, 1, wa, wb, wc, 0);
    const pz = lerpScalar(input.positions, 3, ia, ib, ic, 2, wa, wb, wc, 0);
    const [nx, ny, nz] = normalize3(
      lerpScalar(input.normals, 3, ia, ib, ic, 0, wa, wb, wc, 0),
      lerpScalar(input.normals, 3, ia, ib, ic, 1, wa, wb, wc, 1),
      lerpScalar(input.normals, 3, ia, ib, ic, 2, wa, wb, wc, 0),
    );
    const du = lerpScalar(uvSource, 2, ia, ib, ic, 0, wa, wb, wc, 0);
    const dv = lerpScalar(uvSource, 2, ia, ib, ic, 1, wa, wb, wc, 0);
    const amount = sampleHeight(pixels, du, dv, ref) * scale + bias;
    const outIndex = Math.floor(outPositions.length / 3);
    outPositions.push(px + nx * amount, py + ny * amount, pz + nz * amount);
    outNormalFallback.push(nx, ny, nz);
    if (outUvs != null) {
      outUvs.push(
        lerpScalar(input.uvs!, 2, ia, ib, ic, 0, wa, wb, wc, 0),
        lerpScalar(input.uvs!, 2, ia, ib, ic, 1, wa, wb, wc, 0),
      );
    }
    if (outUv1 != null) {
      outUv1.push(
        lerpScalar(input.uv1!, 2, ia, ib, ic, 0, wa, wb, wc, 0),
        lerpScalar(input.uv1!, 2, ia, ib, ic, 1, wa, wb, wc, 0),
      );
    }
    if (outTangents != null) {
      const [tx, ty, tz] = normalize3(
        lerpScalar(input.tangents!, 4, ia, ib, ic, 0, wa, wb, wc, 1),
        lerpScalar(input.tangents!, 4, ia, ib, ic, 1, wa, wb, wc, 0),
        lerpScalar(input.tangents!, 4, ia, ib, ic, 2, wa, wb, wc, 0),
      );
      const handedness = lerpScalar(input.tangents!, 4, ia, ib, ic, 3, wa, wb, wc, 1) < 0 ? -1 : 1;
      outTangents.push(tx, ty, tz, handedness);
    }
    if (outColors != null) {
      outColors.push(
        lerpScalar(input.colors!, colorStride, ia, ib, ic, 0, wa, wb, wc, 1),
        lerpScalar(input.colors!, colorStride, ia, ib, ic, 1, wa, wb, wc, 1),
        lerpScalar(input.colors!, colorStride, ia, ib, ic, 2, wa, wb, wc, 1),
      );
      if (colorStride === 4) {
        outColors.push(lerpScalar(input.colors!, 4, ia, ib, ic, 3, wa, wb, wc, 1));
      }
    }
    return outIndex;
  };

  for (let tri = 0; tri < sourceTriCount; tri += 1) {
    const ia = sourceIndices[tri * 3] ?? 0;
    const ib = sourceIndices[tri * 3 + 1] ?? 0;
    const ic = sourceIndices[tri * 3 + 2] ?? 0;
    if (ia >= vertexCount || ib >= vertexCount || ic >= vertexCount) {
      warn(
        `${warningPrefix(input.primitiveId, ref)} triangle ${tri} references an out-of-range vertex index ` +
        `(i0=${ia}, i1=${ib}, i2=${ic}; vertexCount=${vertexCount}); microdisplacement disabled.`,
      );
      return null;
    }
    const rows: number[][] = [];
    for (let row = 0; row <= steps; row += 1) {
      const cells = steps - row;
      rows[row] = [];
      for (let col = 0; col <= cells; col += 1) {
        const wb = row / steps;
        const wc = col / steps;
        const wa = 1 - wb - wc;
        rows[row]![col] = pushVertex(ia, ib, ic, wa, wb, wc);
      }
    }
    for (let row = 0; row < steps; row += 1) {
      for (let col = 0; col < steps - row; col += 1) {
        const v0 = rows[row]![col]!;
        const v1 = rows[row + 1]![col]!;
        const v2 = rows[row]![col + 1]!;
        outIndices.push(v0, v1, v2);
        if (col < steps - row - 1) {
          const v3 = rows[row + 1]![col + 1]!;
          outIndices.push(v1, v3, v2);
        }
      }
    }
  }

  const normalAccum = new Float64Array(outPositions.length);
  for (let t = 0; t + 2 < outIndices.length; t += 3) {
    accumulateFaceNormal(outPositions, normalAccum, outIndices[t]!, outIndices[t + 1]!, outIndices[t + 2]!);
  }
  const outNormals = new Float32Array(outNormalFallback.length);
  for (let i = 0; i < outNormals.length; i += 3) {
    const [nx, ny, nz] = normalize3(normalAccum[i] ?? 0, normalAccum[i + 1] ?? 0, normalAccum[i + 2] ?? 0);
    const fallbackZero = nx === 0 && ny === 1 && nz === 0 &&
      Math.abs(normalAccum[i] ?? 0) <= 1e-12 &&
      Math.abs(normalAccum[i + 1] ?? 0) <= 1e-12 &&
      Math.abs(normalAccum[i + 2] ?? 0) <= 1e-12;
    if (fallbackZero) {
      outNormals[i] = outNormalFallback[i] ?? 0;
      outNormals[i + 1] = outNormalFallback[i + 1] ?? 1;
      outNormals[i + 2] = outNormalFallback[i + 2] ?? 0;
    } else {
      outNormals[i] = nx;
      outNormals[i + 1] = ny;
      outNormals[i + 2] = nz;
    }
  }

  return {
    positions: new Float32Array(outPositions),
    normals: outNormals,
    indices: new Uint32Array(outIndices),
    ...(outUvs != null ? { uvs: new Float32Array(outUvs) } : {}),
    ...(outUv1 != null ? { uv1: new Float32Array(outUv1) } : {}),
    ...(outTangents != null ? { tangents: new Float32Array(outTangents) } : {}),
    ...(outColors != null ? { colors: new Float32Array(outColors) } : {}),
    subdivisions,
  };
}
