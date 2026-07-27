const PT_WEBGPU_TEXTURE_SOURCE_BRAND = Symbol(
  'vitrum.pt-webgpu.webgpuTextureSource.brand',
);

let nextPtWebgpuTextureSourceId = 1;

export const PT_WEBGPU_TEXTURE_SOURCE_KIND =
  'vitrum.pt-webgpu.webgpu-texture-source' as const;

export type PtWebgpuTextureColorSpace = 'srgb' | 'linear';

export type PtWebgpuTextureCpuMirrorDataType =
  | 'uint8'
  | 'uint16'
  | 'float16'
  | 'half-float'
  | 'float32';

/** Maximum immutable CPU-mirror snapshot retained by one wrapped GPU source. */
export const PT_WEBGPU_CPU_MIRROR_SNAPSHOT_BUDGET_BYTES = 256 * 1024 * 1024;

/**
 * Immutable CPU texel snapshot paired with the selected GPU subresource.
 * Values are tightly packed in RGB(A) order. The snapshot is used only by
 * CPU-side distributions such as emissive-mesh NEE; forward shading continues
 * to sample the host-owned GPU texture.
 */
export interface PtWebgpuTextureCpuMirror {
  readonly width: number;
  readonly height: number;
  readonly channels: 1 | 2 | 3 | 4;
  readonly dataType: PtWebgpuTextureCpuMirrorDataType;
  readonly colorSpace: PtWebgpuTextureColorSpace;
  readonly data: ArrayLike<number>;
}

export interface PtWebgpuTextureCpuMirrorInput {
  readonly width: number;
  readonly height: number;
  readonly channels: 1 | 2 | 3 | 4;
  readonly dataType: PtWebgpuTextureCpuMirrorDataType;
  readonly colorSpace: PtWebgpuTextureColorSpace;
  readonly data: ArrayLike<number>;
}

/**
 * Immutable descriptor for a host-owned GPUTexture used by a pt-webgpu
 * material TextureRef. Device, format, transfer function, and selected
 * subresource are explicit so scene upload never guesses or reads back pixels.
 *
 * Recreate the descriptor after mutating the selected source subresource. Its
 * new `sourceId` makes the changed content a distinct scene-upload identity.
 */
export interface PtWebgpuTextureSource {
  readonly kind: typeof PT_WEBGPU_TEXTURE_SOURCE_KIND;
  readonly [PT_WEBGPU_TEXTURE_SOURCE_BRAND]: true;
  readonly sourceId: number;
  readonly device: GPUDevice;
  readonly texture: GPUTexture;
  readonly format: GPUTextureFormat;
  readonly colorSpace: PtWebgpuTextureColorSpace;
  readonly dimension: '2d';
  readonly ownership: 'host';
  readonly baseMipLevel: number;
  readonly arrayLayer: number;
  readonly width: number;
  readonly height: number;
  /**
   * Exact CPU snapshot of the selected mip/layer, when supplied by the host.
   * Emissive-map light distributions require this mirror because WebGPU has no
   * synchronous texture readback path during `setScene`.
   */
  readonly cpuMirror?: PtWebgpuTextureCpuMirror;
}

export interface PtWebgpuTextureSourceOptions {
  /** Must exactly match `texture.format`; no format inference is performed. */
  readonly format: GPUTextureFormat;
  /** Transfer function of RGB values stored in the selected source subresource. */
  readonly colorSpace: PtWebgpuTextureColorSpace;
  /** Source mip copied into material-array mip zero. Defaults to zero. */
  readonly baseMipLevel?: number;
  /** Source 2D-array layer copied into the material array. Defaults to zero. */
  readonly arrayLayer?: number;
  /**
   * Exact CPU-readable snapshot of the selected mip/layer. Required when this
   * source is used as an emissiveMap so NEE and forward-hit radiance agree.
   */
  readonly cpuMirror?: PtWebgpuTextureCpuMirrorInput;
}

function assertNonNegativeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer; received ${String(value)}.`);
  }
}

function cpuMirrorElementBytes(dataType: PtWebgpuTextureCpuMirrorDataType): number {
  return dataType === 'uint8' ? 1 : dataType === 'float32' ? 4 : 2;
}

function immutableNumericSnapshot(
  data: ArrayLike<number>,
  dataType: PtWebgpuTextureCpuMirrorDataType,
): ArrayLike<number> {
  const snapshot = dataType === 'uint8'
    ? Uint8Array.from(data)
    : dataType === 'float32'
      ? Float32Array.from(data)
      : Uint16Array.from(data);
  const target = Object.create(null) as { readonly length: number };
  Object.defineProperty(target, 'length', {
    value: snapshot.length,
    writable: false,
    enumerable: false,
    configurable: false,
  });
  Object.preventExtensions(target);
  return new Proxy(target, {
    get(current, property, receiver) {
      if (typeof property === 'string' && /^(0|[1-9][0-9]*)$/.test(property)) {
        return snapshot[Number(property)];
      }
      const reflected: unknown = Reflect.get(current, property, receiver);
      return reflected;
    },
    set() {
      throw new TypeError('PtWebgpuTextureSource cpuMirror data is immutable.');
    },
    defineProperty() {
      throw new TypeError('PtWebgpuTextureSource cpuMirror data is immutable.');
    },
    deleteProperty() {
      throw new TypeError('PtWebgpuTextureSource cpuMirror data is immutable.');
    },
  });
}

function createCpuMirrorSnapshot(
  input: PtWebgpuTextureCpuMirrorInput,
  expectedWidth: number,
  expectedHeight: number,
  sourceColorSpace: PtWebgpuTextureColorSpace,
): PtWebgpuTextureCpuMirror {
  if (input == null || typeof input !== 'object') {
    throw new TypeError('createPtWebgpuTextureSource: cpuMirror must be an object.');
  }
  if (input.width !== expectedWidth || input.height !== expectedHeight) {
    throw new RangeError(
      'createPtWebgpuTextureSource: cpuMirror dimensions must exactly match the selected ' +
      `GPU subresource (${expectedWidth}x${expectedHeight}); received ` +
      `${String(input.width)}x${String(input.height)}.`,
    );
  }
  if (![1, 2, 3, 4].includes(input.channels)) {
    throw new RangeError(
      'createPtWebgpuTextureSource: cpuMirror.channels must be 1, 2, 3, or 4; ' +
      `received ${String(input.channels)}.`,
    );
  }
  if (!['uint8', 'uint16', 'float16', 'half-float', 'float32'].includes(input.dataType)) {
    throw new RangeError(
      `createPtWebgpuTextureSource: unsupported cpuMirror.dataType ${String(input.dataType)}.`,
    );
  }
  if (input.colorSpace !== sourceColorSpace) {
    throw new RangeError(
      'createPtWebgpuTextureSource: cpuMirror.colorSpace must match the GPU source ' +
      `colorSpace (${sourceColorSpace}); received ${String(input.colorSpace)}.`,
    );
  }
  if (input.data == null || typeof input.data.length !== 'number') {
    throw new TypeError('createPtWebgpuTextureSource: cpuMirror.data must be array-like.');
  }
  const pixelCount = expectedWidth * expectedHeight;
  if (!Number.isSafeInteger(pixelCount)) {
    throw new RangeError(
      'createPtWebgpuTextureSource: cpuMirror pixel count exceeds safe integer range.',
    );
  }
  const expectedLength = pixelCount * input.channels;
  if (!Number.isSafeInteger(expectedLength)) {
    throw new RangeError(
      'createPtWebgpuTextureSource: cpuMirror element count exceeds safe integer range.',
    );
  }
  if (!Number.isSafeInteger(input.data.length) || input.data.length !== expectedLength) {
    throw new RangeError(
      `createPtWebgpuTextureSource: cpuMirror.data length must be exactly ${expectedLength}; ` +
      `received ${String(input.data.length)}.`,
    );
  }
  const snapshotBytes = expectedLength * cpuMirrorElementBytes(input.dataType);
  if (
    !Number.isSafeInteger(snapshotBytes) ||
    snapshotBytes > PT_WEBGPU_CPU_MIRROR_SNAPSHOT_BUDGET_BYTES
  ) {
    throw new RangeError(
      `createPtWebgpuTextureSource: cpuMirror immutable snapshot requires ${snapshotBytes} bytes; ` +
      `the per-source budget is ${PT_WEBGPU_CPU_MIRROR_SNAPSHOT_BUDGET_BYTES} bytes.`,
    );
  }
  const integerMax = input.dataType === 'uint8'
    ? 255
    : input.dataType === 'float32'
      ? null
      : 65535;
  for (let i = 0; i < input.data.length; i += 1) {
    const value = Number(input.data[i]);
    if (!Number.isFinite(value)) {
      throw new RangeError(
        `createPtWebgpuTextureSource: cpuMirror.data[${i}] must be finite.`,
      );
    }
    if (integerMax != null && (!Number.isInteger(value) || value < 0 || value > integerMax)) {
      throw new RangeError(
        `createPtWebgpuTextureSource: cpuMirror.data[${i}] must be an integer in ` +
        `[0, ${integerMax}] for ${input.dataType}.`,
      );
    }
  }
  return Object.freeze({
    width: expectedWidth,
    height: expectedHeight,
    channels: input.channels,
    dataType: input.dataType,
    colorSpace: input.colorSpace,
    data: immutableNumericSnapshot(input.data, input.dataType),
  });
}

/** Wrap a host-owned GPUTexture for zero-readback pt-webgpu material upload. */
export function createPtWebgpuTextureSource(
  device: GPUDevice,
  texture: GPUTexture,
  options: PtWebgpuTextureSourceOptions,
): PtWebgpuTextureSource {
  if (device == null || typeof device !== 'object') {
    throw new TypeError('createPtWebgpuTextureSource: device must be a GPUDevice.');
  }
  if (texture == null || typeof texture !== 'object' || typeof texture.createView !== 'function') {
    throw new TypeError('createPtWebgpuTextureSource: texture must be a GPUTexture.');
  }
  if (options?.format !== texture.format) {
    throw new RangeError(
      `createPtWebgpuTextureSource: declared format ${String(options?.format)} ` +
      `does not match texture.format ${String(texture.format)}.`,
    );
  }
  if (options.colorSpace !== 'srgb' && options.colorSpace !== 'linear') {
    throw new RangeError(
      'createPtWebgpuTextureSource: colorSpace must be "srgb" or "linear"; ' +
      `received ${String(options.colorSpace)}.`,
    );
  }
  if (options.format.endsWith('-srgb') && options.colorSpace !== 'srgb') {
    throw new RangeError(
      `createPtWebgpuTextureSource: native-sRGB format ${options.format} ` +
      'cannot be declared as linear.',
    );
  }
  if (texture.dimension !== '2d') {
    throw new RangeError(
      `createPtWebgpuTextureSource: only 2d textures are accepted; received ${String(texture.dimension)}.`,
    );
  }
  if (texture.sampleCount !== 1) {
    throw new RangeError(
      `createPtWebgpuTextureSource: multisampled textures are not valid material sources; ` +
      `sampleCount=${String(texture.sampleCount)}.`,
    );
  }
  if (!Number.isSafeInteger(texture.width) || texture.width < 1) {
    throw new RangeError(
      `createPtWebgpuTextureSource: texture.width must be a positive safe integer; received ${String(texture.width)}.`,
    );
  }
  if (!Number.isSafeInteger(texture.height) || texture.height < 1) {
    throw new RangeError(
      `createPtWebgpuTextureSource: texture.height must be a positive safe integer; received ${String(texture.height)}.`,
    );
  }
  if (!Number.isSafeInteger(texture.mipLevelCount) || texture.mipLevelCount < 1) {
    throw new RangeError(
      'createPtWebgpuTextureSource: texture.mipLevelCount must be a positive safe integer.',
    );
  }
  if (!Number.isSafeInteger(texture.depthOrArrayLayers) || texture.depthOrArrayLayers < 1) {
    throw new RangeError(
      'createPtWebgpuTextureSource: texture.depthOrArrayLayers must be a positive safe integer.',
    );
  }

  const baseMipLevel = options.baseMipLevel ?? 0;
  const arrayLayer = options.arrayLayer ?? 0;
  assertNonNegativeInteger(baseMipLevel, 'createPtWebgpuTextureSource: baseMipLevel');
  assertNonNegativeInteger(arrayLayer, 'createPtWebgpuTextureSource: arrayLayer');
  if (baseMipLevel >= texture.mipLevelCount) {
    throw new RangeError(
      `createPtWebgpuTextureSource: baseMipLevel ${baseMipLevel} is outside ` +
      `${texture.mipLevelCount} texture mip levels.`,
    );
  }
  if (baseMipLevel > 52) {
    throw new RangeError(
      'createPtWebgpuTextureSource: baseMipLevel is too large for exact subresource dimensions.',
    );
  }
  if (arrayLayer >= texture.depthOrArrayLayers) {
    throw new RangeError(
      `createPtWebgpuTextureSource: arrayLayer ${arrayLayer} is outside ` +
      `${texture.depthOrArrayLayers} texture layers.`,
    );
  }
  if (!Number.isSafeInteger(nextPtWebgpuTextureSourceId)) {
    throw new RangeError('createPtWebgpuTextureSource: descriptor identity space is exhausted.');
  }

  const width = Math.max(1, Math.floor(texture.width / (2 ** baseMipLevel)));
  const height = Math.max(1, Math.floor(texture.height / (2 ** baseMipLevel)));
  const cpuMirror = options.cpuMirror == null
    ? undefined
    : createCpuMirrorSnapshot(options.cpuMirror, width, height, options.colorSpace);

  const sourceId = nextPtWebgpuTextureSourceId;
  nextPtWebgpuTextureSourceId += 1;
  return Object.freeze({
    kind: PT_WEBGPU_TEXTURE_SOURCE_KIND,
    [PT_WEBGPU_TEXTURE_SOURCE_BRAND]: true as const,
    sourceId,
    device,
    texture,
    format: options.format,
    colorSpace: options.colorSpace,
    dimension: '2d' as const,
    ownership: 'host' as const,
    baseMipLevel,
    arrayLayer,
    width,
    height,
    ...(cpuMirror == null ? {} : { cpuMirror }),
  });
}

export function isPtWebgpuTextureSource(value: unknown): value is PtWebgpuTextureSource {
  if (value == null || typeof value !== 'object') return false;
  const candidate = value as Partial<PtWebgpuTextureSource> & {
    readonly [PT_WEBGPU_TEXTURE_SOURCE_BRAND]?: unknown;
  };
  return (
    candidate.kind === PT_WEBGPU_TEXTURE_SOURCE_KIND &&
    candidate[PT_WEBGPU_TEXTURE_SOURCE_BRAND] === true
  );
}
