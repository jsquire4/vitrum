const PT_WEBGPU_TEXTURE_SOURCE_BRAND = Symbol(
  'vitrum.pt-webgpu.webgpuTextureSource.brand',
);

// Symbol-property branding is forgeable through a Proxy get trap. Identity in
// this module-private set is the runtime nominal brand; only the validated,
// frozen object returned by createPtWebgpuTextureSource is admitted.
const PT_WEBGPU_TEXTURE_SOURCE_IDENTITIES = new WeakSet<object>();

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

function gpuColorFormatChannelCount(format: GPUTextureFormat): 1 | 2 | 3 | 4 | null {
  if (
    format.startsWith('rgba') ||
    format.startsWith('bgra') ||
    format === 'rgb10a2unorm'
  ) {
    return 4;
  }
  if (format.startsWith('rgb') || format === 'rg11b10ufloat') return 3;
  if (
    format.startsWith('rg8') ||
    format.startsWith('rg16') ||
    format.startsWith('rg32')
  ) {
    return 2;
  }
  if (
    format.startsWith('r8') ||
    format.startsWith('r16') ||
    format.startsWith('r32')
  ) {
    return 1;
  }
  return null;
}

type PtWebgpuTextureCpuMirrorSnapshot =
  | Uint8Array<ArrayBuffer>
  | Uint16Array<ArrayBuffer>
  | Float32Array<ArrayBuffer>;

function assertCpuMirrorDataIsNotShared(input: unknown): void {
  if (!ArrayBuffer.isView(input)) return;
  const inputBuffer = input.buffer;
  if (
    typeof SharedArrayBuffer !== 'undefined' &&
    inputBuffer instanceof SharedArrayBuffer
  ) {
    throw new TypeError(
      'createPtWebgpuTextureSource: SharedArrayBuffer-backed cpuMirror.data is not accepted; ' +
      'a concurrently mutable buffer cannot provide an exact immutable snapshot.',
    );
  }
}

function immutableNumericSnapshot(
  snapshot: PtWebgpuTextureCpuMirrorSnapshot,
): ArrayLike<number> {
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
  sourceFormat: GPUTextureFormat,
  sourceFormatChannels: 1 | 2 | 3 | 4 | null,
): PtWebgpuTextureCpuMirror {
  if (input == null || typeof input !== 'object') {
    throw new TypeError('createPtWebgpuTextureSource: cpuMirror must be an object.');
  }
  // Treat descriptor objects as untrusted input. Snapshot each property once so
  // accessor-backed inputs cannot validate one value and publish another.
  const inputWidth = input.width;
  const inputHeight = input.height;
  const inputChannels = input.channels;
  const inputDataType = input.dataType;
  const inputColorSpace = input.colorSpace;
  const inputData = input.data;
  if (inputWidth !== expectedWidth || inputHeight !== expectedHeight) {
    throw new RangeError(
      'createPtWebgpuTextureSource: cpuMirror dimensions must exactly match the selected ' +
      `GPU subresource (${expectedWidth}x${expectedHeight}); received ` +
      `${String(inputWidth)}x${String(inputHeight)}.`,
    );
  }
  if (![1, 2, 3, 4].includes(inputChannels)) {
    throw new RangeError(
      'createPtWebgpuTextureSource: cpuMirror.channels must be 1, 2, 3, or 4; ' +
      `received ${String(inputChannels)}.`,
    );
  }
  if (sourceFormatChannels != null && inputChannels !== sourceFormatChannels) {
    throw new RangeError(
      `createPtWebgpuTextureSource: cpuMirror.channels must match ${sourceFormat} ` +
      `(${sourceFormatChannels}); received ${String(inputChannels)}.`,
    );
  }
  if (!['uint8', 'uint16', 'float16', 'half-float', 'float32'].includes(inputDataType)) {
    throw new RangeError(
      `createPtWebgpuTextureSource: unsupported cpuMirror.dataType ${String(inputDataType)}.`,
    );
  }
  if (inputColorSpace !== sourceColorSpace) {
    throw new RangeError(
      'createPtWebgpuTextureSource: cpuMirror.colorSpace must match the GPU source ' +
      `colorSpace (${sourceColorSpace}); received ${String(inputColorSpace)}.`,
    );
  }
  if (inputData == null) {
    throw new TypeError('createPtWebgpuTextureSource: cpuMirror.data must be array-like.');
  }
  // Reject shared memory before observing length or allocating the destination.
  // Element-wise copying cannot make one coherent instant out of concurrent writes.
  assertCpuMirrorDataIsNotShared(inputData);
  // ArrayLike may be implemented by accessors. Read its length exactly once;
  // validation and snapshot publication must describe the same observation.
  const inputLength = inputData.length;
  if (typeof inputLength !== 'number') {
    throw new TypeError('createPtWebgpuTextureSource: cpuMirror.data must be array-like.');
  }
  const pixelCount = expectedWidth * expectedHeight;
  if (!Number.isSafeInteger(pixelCount)) {
    throw new RangeError(
      'createPtWebgpuTextureSource: cpuMirror pixel count exceeds safe integer range.',
    );
  }
  const expectedLength = pixelCount * inputChannels;
  if (!Number.isSafeInteger(expectedLength)) {
    throw new RangeError(
      'createPtWebgpuTextureSource: cpuMirror element count exceeds safe integer range.',
    );
  }
  if (!Number.isSafeInteger(inputLength) || inputLength !== expectedLength) {
    throw new RangeError(
      `createPtWebgpuTextureSource: cpuMirror.data length must be exactly ${expectedLength}; ` +
      `received ${String(inputLength)}.`,
    );
  }
  const snapshotBytes = expectedLength * cpuMirrorElementBytes(inputDataType);
  if (
    !Number.isSafeInteger(snapshotBytes) ||
    snapshotBytes > PT_WEBGPU_CPU_MIRROR_SNAPSHOT_BUDGET_BYTES
  ) {
    throw new RangeError(
      `createPtWebgpuTextureSource: cpuMirror immutable snapshot requires ${snapshotBytes} bytes; ` +
      `the per-source budget is ${PT_WEBGPU_CPU_MIRROR_SNAPSHOT_BUDGET_BYTES} bytes.`,
    );
  }
  const integerMax = inputDataType === 'uint8'
    ? 255
    : inputDataType === 'float32'
      ? null
      : 65535;
  // Allocate only after the exact byte budget is admitted. Each hostile or
  // accessor-backed element is then read once, validated, converted, and stored
  // directly in the final immutable staging buffer. A throwing getter leaves no
  // published snapshot or retained reservation: the local allocation is dropped.
  const snapshot: PtWebgpuTextureCpuMirrorSnapshot = inputDataType === 'uint8'
    ? new Uint8Array(expectedLength)
    : inputDataType === 'float32'
      ? new Float32Array(expectedLength)
      : new Uint16Array(expectedLength);
  for (let i = 0; i < expectedLength; i += 1) {
    const value = Number(inputData[i]);
    if (!Number.isFinite(value)) {
      throw new RangeError(
        `createPtWebgpuTextureSource: cpuMirror.data[${i}] must be finite.`,
      );
    }
    if (inputDataType === 'float32') {
      const packed = Math.fround(value);
      if (!Number.isFinite(packed)) {
        throw new RangeError(
          `createPtWebgpuTextureSource: cpuMirror.data[${i}] must be representable as finite float32.`,
        );
      }
      if (value !== 0 && packed === 0) {
        throw new RangeError(
          `createPtWebgpuTextureSource: cpuMirror.data[${i}] must not underflow to zero as float32.`,
        );
      }
      snapshot[i] = packed;
    }
    if (integerMax != null && (!Number.isInteger(value) || value < 0 || value > integerMax)) {
      throw new RangeError(
        `createPtWebgpuTextureSource: cpuMirror.data[${i}] must be an integer in ` +
        `[0, ${integerMax}] for ${inputDataType}.`,
      );
    }
    if (integerMax != null) snapshot[i] = value;
  }
  const source = Object.freeze({
    width: expectedWidth,
    height: expectedHeight,
    channels: inputChannels,
    dataType: inputDataType,
    colorSpace: inputColorSpace,
    data: immutableNumericSnapshot(snapshot),
  });
  return source;
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
  if (options == null || typeof options !== 'object') {
    throw new TypeError('createPtWebgpuTextureSource: options must be an object.');
  }
  // Snapshot both the caller descriptor and the observable WebGPU descriptor
  // once. Validation, subresource selection, and publication then use exactly
  // the values that were admitted.
  const format = options.format;
  const colorSpace = options.colorSpace;
  const requestedBaseMipLevel = options.baseMipLevel;
  const requestedArrayLayer = options.arrayLayer;
  const cpuMirrorInput = options.cpuMirror;
  const textureFormat = texture.format;
  const textureDimension = texture.dimension;
  const textureSampleCount = texture.sampleCount;
  const textureWidth = texture.width;
  const textureHeight = texture.height;
  const textureMipLevelCount = texture.mipLevelCount;
  const textureDepthOrArrayLayers = texture.depthOrArrayLayers;
  if (format !== textureFormat) {
    throw new RangeError(
      `createPtWebgpuTextureSource: declared format ${String(format)} ` +
      `does not match texture.format ${String(textureFormat)}.`,
    );
  }
  if (colorSpace !== 'srgb' && colorSpace !== 'linear') {
    throw new RangeError(
      'createPtWebgpuTextureSource: colorSpace must be "srgb" or "linear"; ' +
      `received ${String(colorSpace)}.`,
    );
  }
  if (format.endsWith('-srgb') && colorSpace !== 'srgb') {
    throw new RangeError(
      `createPtWebgpuTextureSource: native-sRGB format ${format} ` +
      'cannot be declared as linear.',
    );
  }
  if (textureDimension !== '2d') {
    throw new RangeError(
      `createPtWebgpuTextureSource: only 2d textures are accepted; received ${String(textureDimension)}.`,
    );
  }
  if (textureSampleCount !== 1) {
    throw new RangeError(
      `createPtWebgpuTextureSource: multisampled textures are not valid material sources; ` +
      `sampleCount=${String(textureSampleCount)}.`,
    );
  }
  if (!Number.isSafeInteger(textureWidth) || textureWidth < 1) {
    throw new RangeError(
      `createPtWebgpuTextureSource: texture.width must be a positive safe integer; received ${String(textureWidth)}.`,
    );
  }
  if (!Number.isSafeInteger(textureHeight) || textureHeight < 1) {
    throw new RangeError(
      `createPtWebgpuTextureSource: texture.height must be a positive safe integer; received ${String(textureHeight)}.`,
    );
  }
  if (!Number.isSafeInteger(textureMipLevelCount) || textureMipLevelCount < 1) {
    throw new RangeError(
      'createPtWebgpuTextureSource: texture.mipLevelCount must be a positive safe integer.',
    );
  }
  if (!Number.isSafeInteger(textureDepthOrArrayLayers) || textureDepthOrArrayLayers < 1) {
    throw new RangeError(
      'createPtWebgpuTextureSource: texture.depthOrArrayLayers must be a positive safe integer.',
    );
  }

  const baseMipLevel = requestedBaseMipLevel ?? 0;
  const arrayLayer = requestedArrayLayer ?? 0;
  assertNonNegativeInteger(baseMipLevel, 'createPtWebgpuTextureSource: baseMipLevel');
  assertNonNegativeInteger(arrayLayer, 'createPtWebgpuTextureSource: arrayLayer');
  if (baseMipLevel >= textureMipLevelCount) {
    throw new RangeError(
      `createPtWebgpuTextureSource: baseMipLevel ${baseMipLevel} is outside ` +
      `${textureMipLevelCount} texture mip levels.`,
    );
  }
  if (baseMipLevel > 52) {
    throw new RangeError(
      'createPtWebgpuTextureSource: baseMipLevel is too large for exact subresource dimensions.',
    );
  }
  if (arrayLayer >= textureDepthOrArrayLayers) {
    throw new RangeError(
      `createPtWebgpuTextureSource: arrayLayer ${arrayLayer} is outside ` +
      `${textureDepthOrArrayLayers} texture layers.`,
    );
  }
  if (!Number.isSafeInteger(nextPtWebgpuTextureSourceId)) {
    throw new RangeError('createPtWebgpuTextureSource: descriptor identity space is exhausted.');
  }

  const width = Math.max(1, Math.floor(textureWidth / (2 ** baseMipLevel)));
  const height = Math.max(1, Math.floor(textureHeight / (2 ** baseMipLevel)));
  const formatChannels = gpuColorFormatChannelCount(format);
  const cpuMirror = cpuMirrorInput == null
    ? undefined
    : createCpuMirrorSnapshot(
      cpuMirrorInput,
      width,
      height,
      colorSpace,
      format,
      formatChannels,
    );

  const sourceId = nextPtWebgpuTextureSourceId;
  nextPtWebgpuTextureSourceId += 1;
  const source = Object.freeze({
    kind: PT_WEBGPU_TEXTURE_SOURCE_KIND,
    [PT_WEBGPU_TEXTURE_SOURCE_BRAND]: true as const,
    sourceId,
    device,
    texture,
    format,
    colorSpace,
    dimension: '2d' as const,
    ownership: 'host' as const,
    baseMipLevel,
    arrayLayer,
    width,
    height,
    ...(cpuMirror == null ? {} : { cpuMirror }),
  });
  PT_WEBGPU_TEXTURE_SOURCE_IDENTITIES.add(source);
  return source;
}

export function isPtWebgpuTextureSource(value: unknown): value is PtWebgpuTextureSource {
  return value != null &&
    typeof value === 'object' &&
    PT_WEBGPU_TEXTURE_SOURCE_IDENTITIES.has(value);
}
