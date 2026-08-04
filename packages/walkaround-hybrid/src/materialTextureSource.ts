const WALKAROUND_WEBGPU_TEXTURE_SOURCE_BRAND = Symbol(
  'vitrum.walkaround.webgpuTextureSource.brand',
);

let nextWalkaroundWebGpuTextureSourceId = 1;

function createTextureSourceSessionSalt(): readonly [number, number] {
  const words = new Uint32Array(2);
  try {
    globalThis.crypto?.getRandomValues(words);
  } catch {
    // A session-only compatibility salt is not a security primitive. The
    // fallback merely has to differ across ordinary process/page lifetimes.
  }
  if (words[0] === 0 && words[1] === 0) {
    const now = Date.now();
    words[0] = (now ^ Math.floor(Math.random() * 0x1_0000_0000)) >>> 0;
    words[1] = (
      Math.floor(now / 0x1_0000_0000) ^
      Math.floor(Math.random() * 0x1_0000_0000) ^
      0x9e3779b9
    ) >>> 0;
  }
  return [words[0]!, words[1]!];
}

const WALKAROUND_TEXTURE_SOURCE_SESSION_SALT =
  createTextureSourceSessionSalt();

function hashTextureContentRevision(value: string): readonly [number, number] {
  // Two independently seeded FNV-1a lanes over UTF-16 code units. This is a
  // compatibility fingerprint, not a security boundary.
  let lo = 0x811c9dc5;
  let hi = 0x9e3779b9;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    lo = Math.imul(lo ^ (code & 0xff), 0x01000193) >>> 0;
    lo = Math.imul(lo ^ (code >>> 8), 0x01000193) >>> 0;
    hi = Math.imul(hi ^ code, 0x85ebca6b) >>> 0;
    hi = (hi ^ (hi >>> 13)) >>> 0;
  }
  return [lo, hi];
}

export const WALKAROUND_WEBGPU_TEXTURE_SOURCE_KIND =
  'vitrum.walkaround.webgpu-texture-source' as const;

export type WalkaroundTextureColorSpace = 'srgb' | 'linear';

export type WalkaroundTextureCpuMirrorDataType =
  | 'uint8'
  | 'uint16'
  | 'float16'
  | 'half-float'
  | 'float32';

/** Logical sampled channel count of a supported WebGPU material-map format. */
export function walkaroundTextureFormatChannelCount(
  format: GPUTextureFormat,
): 1 | 2 | 3 | 4 {
  switch (format) {
    case 'r8unorm':
    case 'r8snorm':
    case 'r16float':
    case 'r32float':
      return 1;
    case 'rg8unorm':
    case 'rg8snorm':
    case 'rg16float':
    case 'rg32float':
      return 2;
    case 'rg11b10ufloat':
    case 'rgb9e5ufloat':
      return 3;
    default:
      return 4;
  }
}

/** Maximum immutable CPU-mirror snapshot retained by one wrapped GPU source. */
export const WALKAROUND_CPU_MIRROR_SNAPSHOT_BUDGET_BYTES = 256 * 1024 * 1024;

/**
 * Immutable texel snapshot for the selected GPU mip/layer. When the source is
 * used radiometrically, the atlas publishes this snapshot for forward shading
 * as well as CPU emitter-distribution construction so both paths consume the
 * same encoded texels. Non-radiometric maps continue to use the host-owned GPU
 * texture directly.
 */
export interface WalkaroundTextureCpuMirror {
  readonly width: number;
  readonly height: number;
  readonly channels: 1 | 2 | 3 | 4;
  readonly dataType: WalkaroundTextureCpuMirrorDataType;
  readonly colorSpace: WalkaroundTextureColorSpace;
  readonly data: ArrayLike<number>;
}

export interface WalkaroundTextureCpuMirrorInput {
  readonly width: number;
  readonly height: number;
  readonly channels: 1 | 2 | 3 | 4;
  readonly dataType: WalkaroundTextureCpuMirrorDataType;
  readonly colorSpace: WalkaroundTextureColorSpace;
  readonly data: ArrayLike<number>;
}

/**
 * Immutable, host-owned WebGPU texture descriptor accepted by walkaround
 * material TextureRefs. The factory pins the source device, format, selected
 * mip/layer, dimensions, and transfer function so atlas upload never guesses
 * ownership or performs a CPU readback.
 *
 * Recreate the descriptor after mutating the selected source subresource. A
 * new descriptor receives a new session identity, which invalidates GI
 * material snapshots even when dimensions and metadata are unchanged. To make
 * snapshots portable across page/process lifetimes, supply a stable
 * `contentRevision` that changes whenever the selected texels change.
 */
export interface WalkaroundWebGpuTextureSource {
  readonly kind: typeof WALKAROUND_WEBGPU_TEXTURE_SOURCE_KIND;
  readonly [WALKAROUND_WEBGPU_TEXTURE_SOURCE_BRAND]: true;
  readonly sourceId: number;
  /** Low/high words used by GI-state compatibility fingerprints. */
  readonly compatibilityKeyLo: number;
  readonly compatibilityKeyHi: number;
  /** Host-authored persistent content/revision identity, when supplied. */
  readonly contentRevision?: string;
  readonly device: GPUDevice;
  readonly texture: GPUTexture;
  readonly format: GPUTextureFormat;
  readonly colorSpace: WalkaroundTextureColorSpace;
  readonly dimension: '2d';
  readonly ownership: 'host';
  readonly baseMipLevel: number;
  readonly arrayLayer: number;
  readonly width: number;
  readonly height: number;
  /**
   * Exact CPU snapshot of the selected mip/layer. Required for radiometric
   * emissive and light maps. A radiometric use publishes this immutable
   * snapshot as its canonical atlas layer so preflight, generated mips,
   * forward shading, and CPU-built emitter distributions consume identical
   * packed bytes. Ordinary material maps do not require it and continue to use
   * the host-owned GPU texture directly.
   */
  readonly cpuMirror?: WalkaroundTextureCpuMirror;
}

export interface WalkaroundWebGpuTextureSourceOptions {
  /** Must exactly match `texture.format`; no format inference is permitted. */
  readonly format: GPUTextureFormat;
  /** Transfer function of the stored RGB values, before atlas conversion. */
  readonly colorSpace: WalkaroundTextureColorSpace;
  /** Source mip copied into atlas mip zero. Defaults to zero. */
  readonly baseMipLevel?: number;
  /** Source 2D-array layer copied into the atlas. Defaults to zero. */
  readonly arrayLayer?: number;
  /**
   * Stable identity for the selected texel content. The same value may be
   * reused across sessions only when the selected mip/layer bytes and their
   * interpretation are identical; change it on every content mutation.
   *
   * When omitted, GI compatibility is deliberately session-scoped so a saved
   * state cannot be accepted after a reload merely because sourceId restarted.
   */
  readonly contentRevision?: string;
  /**
   * Exact CPU-readable snapshot of the selected mip/layer. The factory copies
   * and freezes it; later host mutations cannot desynchronise GPU shading from
   * radiometric preflight or mapped-emitter importance sampling. Required when
   * this source is used by an emissive or light map.
   */
  readonly cpuMirror?: WalkaroundTextureCpuMirrorInput;
}

function assertNonNegativeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer; received ${String(value)}.`);
  }
}

function cpuMirrorElementBytes(dataType: WalkaroundTextureCpuMirrorDataType): number {
  return dataType === 'uint8' ? 1 : dataType === 'float32' ? 4 : 2;
}

type MutableCpuMirrorSnapshot = Uint8Array | Uint16Array | Float32Array;

function allocateCpuMirrorSnapshot(
  length: number,
  dataType: WalkaroundTextureCpuMirrorDataType,
): MutableCpuMirrorSnapshot {
  return dataType === 'uint8'
    ? new Uint8Array(length)
    : dataType === 'float32'
      ? new Float32Array(length)
      : new Uint16Array(length);
}

function immutableNumericSnapshot(
  snapshot: MutableCpuMirrorSnapshot,
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
      throw new TypeError('WalkaroundWebGpuTextureSource cpuMirror data is immutable.');
    },
    defineProperty() {
      throw new TypeError('WalkaroundWebGpuTextureSource cpuMirror data is immutable.');
    },
    deleteProperty() {
      throw new TypeError('WalkaroundWebGpuTextureSource cpuMirror data is immutable.');
    },
  });
}

function createCpuMirrorSnapshot(
  input: WalkaroundTextureCpuMirrorInput,
  expectedWidth: number,
  expectedHeight: number,
  expectedChannels: 1 | 2 | 3 | 4,
  sourceFormat: GPUTextureFormat,
  sourceColorSpace: WalkaroundTextureColorSpace,
): WalkaroundTextureCpuMirror {
  if (input == null || typeof input !== 'object' || Array.isArray(input)) {
    throw new TypeError('createWalkaroundWebGpuTextureSource: cpuMirror must be an object.');
  }
  // Snapshot every user-controlled descriptor property exactly once. All
  // validation, allocation, and publication below use only these locals.
  const inputWidth = input.width;
  const inputHeight = input.height;
  const inputChannels = input.channels;
  const inputDataType = input.dataType;
  const inputColorSpace = input.colorSpace;
  const inputData = input.data;
  if (inputWidth !== expectedWidth || inputHeight !== expectedHeight) {
    throw new RangeError(
      'createWalkaroundWebGpuTextureSource: cpuMirror dimensions must exactly match the selected ' +
      `GPU subresource (${expectedWidth}x${expectedHeight}); received ` +
      `${String(inputWidth)}x${String(inputHeight)}.`,
    );
  }
  if (![1, 2, 3, 4].includes(inputChannels)) {
    throw new RangeError(
      'createWalkaroundWebGpuTextureSource: cpuMirror.channels must be 1, 2, 3, or 4; ' +
      `received ${String(inputChannels)}.`,
    );
  }
  if (inputChannels !== expectedChannels) {
    throw new RangeError(
      'createWalkaroundWebGpuTextureSource: cpuMirror.channels must match the ' +
      `selected GPU format ${sourceFormat} (${expectedChannels}); received ${String(inputChannels)}.`,
    );
  }
  if (!['uint8', 'uint16', 'float16', 'half-float', 'float32'].includes(inputDataType)) {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: unsupported cpuMirror.dataType ${String(inputDataType)}.`,
    );
  }
  if (inputColorSpace !== sourceColorSpace) {
    throw new RangeError(
      'createWalkaroundWebGpuTextureSource: cpuMirror.colorSpace must match the GPU source ' +
      `colorSpace (${sourceColorSpace}); received ${String(inputColorSpace)}.`,
    );
  }
  if (inputData == null) {
    throw new TypeError('createWalkaroundWebGpuTextureSource: cpuMirror.data must be array-like.');
  }
  // Read the user-controlled length once so a getter cannot pass validation
  // with one value and redirect the subsequent allocation/traversal.
  const inputDataLength = inputData.length;
  if (typeof inputDataLength !== 'number') {
    throw new TypeError('createWalkaroundWebGpuTextureSource: cpuMirror.data must be array-like.');
  }
  const pixelCount = expectedWidth * expectedHeight;
  const expectedLength = pixelCount * inputChannels;
  if (!Number.isSafeInteger(pixelCount) || !Number.isSafeInteger(expectedLength)) {
    throw new RangeError(
      'createWalkaroundWebGpuTextureSource: cpuMirror element count exceeds the safe integer range.',
    );
  }
  if (!Number.isSafeInteger(inputDataLength) || inputDataLength !== expectedLength) {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: cpuMirror.data length must be exactly ${expectedLength}; ` +
      `received ${String(inputDataLength)}.`,
    );
  }
  const snapshotBytes = expectedLength * cpuMirrorElementBytes(inputDataType);
  if (
    !Number.isSafeInteger(snapshotBytes) ||
    snapshotBytes > WALKAROUND_CPU_MIRROR_SNAPSHOT_BUDGET_BYTES
  ) {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: cpuMirror immutable snapshot requires ${snapshotBytes} bytes; ` +
      `the per-source budget is ${WALKAROUND_CPU_MIRROR_SNAPSHOT_BUDGET_BYTES} bytes.`,
    );
  }
  const integerMax = inputDataType === 'uint8'
    ? 255
    : inputDataType === 'float32'
      ? null
      : 65535;
  // Allocate only after the complete byte budget is known, then validate and
  // publish the exact same single read of every user-controlled element.
  const snapshot = allocateCpuMirrorSnapshot(expectedLength, inputDataType);
  for (let index = 0; index < expectedLength; index += 1) {
    const value = Number(inputData[index]);
    if (!Number.isFinite(value)) {
      throw new RangeError(
        `createWalkaroundWebGpuTextureSource: cpuMirror.data[${index}] must be finite.`,
      );
    }
    if (inputDataType === 'float32') {
      const canonical = Math.fround(value);
      if (!Number.isFinite(canonical)) {
        throw new RangeError(
          `createWalkaroundWebGpuTextureSource: cpuMirror.data[${index}] must be representable as finite float32.`,
        );
      }
      if (value !== 0 && canonical === 0) {
        throw new RangeError(
          `createWalkaroundWebGpuTextureSource: cpuMirror.data[${index}] is nonzero but underflows to zero in float32.`,
        );
      }
      snapshot[index] = canonical;
    }
    if (integerMax != null && (!Number.isInteger(value) || value < 0 || value > integerMax)) {
      throw new RangeError(
        `createWalkaroundWebGpuTextureSource: cpuMirror.data[${index}] must be an integer in ` +
        `[0, ${integerMax}] for ${inputDataType}.`,
      );
    }
    if (integerMax != null) {
      snapshot[index] = value;
    }
  }
  return Object.freeze({
    width: expectedWidth,
    height: expectedHeight,
    channels: inputChannels,
    dataType: inputDataType,
    colorSpace: inputColorSpace,
    data: immutableNumericSnapshot(snapshot),
  });
}

/**
 * Wrap a host-owned GPUTexture for zero-readback walkaround material upload.
 * The returned descriptor is nominal and frozen; hand-authored lookalikes are
 * intentionally rejected by {@link isWalkaroundWebGpuTextureSource}.
 */
export function createWalkaroundWebGpuTextureSource(
  device: GPUDevice,
  texture: GPUTexture,
  options: WalkaroundWebGpuTextureSourceOptions,
): WalkaroundWebGpuTextureSource {
  if (device == null || typeof device !== 'object') {
    throw new TypeError('createWalkaroundWebGpuTextureSource: device must be a GPUDevice.');
  }
  if (texture == null || typeof texture !== 'object' || typeof texture.createView !== 'function') {
    throw new TypeError('createWalkaroundWebGpuTextureSource: texture must be a GPUTexture.');
  }
  if (options == null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('createWalkaroundWebGpuTextureSource: options must be an object.');
  }
  // Options may be a getter-backed host object. Capture the complete tuple
  // once so validation, compatibility identity, mirror construction, and the
  // returned descriptor cannot observe different values.
  const sourceFormat = options.format;
  const sourceColorSpace = options.colorSpace;
  const contentRevision = options.contentRevision;
  const baseMipLevelInput = options.baseMipLevel;
  const arrayLayerInput = options.arrayLayer;
  const cpuMirrorInput = options.cpuMirror;
  if (sourceFormat !== texture.format) {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: declared format ${String(sourceFormat)} ` +
      `does not match texture.format ${String(texture.format)}.`,
    );
  }
  if (sourceColorSpace !== 'srgb' && sourceColorSpace !== 'linear') {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: colorSpace must be "srgb" or "linear"; ` +
      `received ${String(sourceColorSpace)}.`,
    );
  }
  if (
    contentRevision != null &&
    (
      typeof contentRevision !== 'string' ||
      contentRevision.length === 0 ||
      contentRevision.length > 4096
    )
  ) {
    throw new RangeError(
      'createWalkaroundWebGpuTextureSource: contentRevision must be a non-empty string of at most 4096 code units.',
    );
  }
  if (sourceFormat.endsWith('-srgb') && sourceColorSpace !== 'srgb') {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: native-sRGB format ${sourceFormat} ` +
      'cannot be declared as linear.',
    );
  }
  if (texture.dimension !== '2d') {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: only 2d textures are supported; ` +
      `received ${String(texture.dimension)}.`,
    );
  }
  if (texture.sampleCount !== 1) {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: multisampled textures are unsupported; ` +
      `sampleCount=${String(texture.sampleCount)}.`,
    );
  }
  if (!Number.isSafeInteger(texture.width) || texture.width < 1) {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: texture.width must be a positive safe integer; ` +
      `received ${String(texture.width)}.`,
    );
  }
  if (!Number.isSafeInteger(texture.height) || texture.height < 1) {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: texture.height must be a positive safe integer; ` +
      `received ${String(texture.height)}.`,
    );
  }
  if (!Number.isSafeInteger(texture.mipLevelCount) || texture.mipLevelCount < 1) {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: texture.mipLevelCount must be a positive safe integer; ` +
      `received ${String(texture.mipLevelCount)}.`,
    );
  }
  if (!Number.isSafeInteger(texture.depthOrArrayLayers) || texture.depthOrArrayLayers < 1) {
    throw new RangeError(
      'createWalkaroundWebGpuTextureSource: texture.depthOrArrayLayers must be a positive safe integer; ' +
      `received ${String(texture.depthOrArrayLayers)}.`,
    );
  }

  const baseMipLevel = baseMipLevelInput ?? 0;
  const arrayLayer = arrayLayerInput ?? 0;
  assertNonNegativeInteger(baseMipLevel, 'createWalkaroundWebGpuTextureSource: baseMipLevel');
  assertNonNegativeInteger(arrayLayer, 'createWalkaroundWebGpuTextureSource: arrayLayer');
  if (baseMipLevel >= texture.mipLevelCount) {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: baseMipLevel ${baseMipLevel} is outside ` +
      `${texture.mipLevelCount} texture mip levels.`,
    );
  }
  if (arrayLayer >= texture.depthOrArrayLayers) {
    throw new RangeError(
      `createWalkaroundWebGpuTextureSource: arrayLayer ${arrayLayer} is outside ` +
      `${texture.depthOrArrayLayers} texture layers.`,
    );
  }

  const width = Math.max(1, Math.floor(texture.width / (2 ** baseMipLevel)));
  const height = Math.max(1, Math.floor(texture.height / (2 ** baseMipLevel)));
  const expectedMirrorChannels =
    walkaroundTextureFormatChannelCount(sourceFormat);
  const cpuMirror = cpuMirrorInput == null
    ? undefined
    : createCpuMirrorSnapshot(
        cpuMirrorInput,
        width,
        height,
        expectedMirrorChannels,
        sourceFormat,
        sourceColorSpace,
      );
  if (!Number.isSafeInteger(nextWalkaroundWebGpuTextureSourceId)) {
    throw new RangeError(
      'createWalkaroundWebGpuTextureSource: descriptor identity space is exhausted.',
    );
  }
  const sourceId = nextWalkaroundWebGpuTextureSourceId;
  nextWalkaroundWebGpuTextureSourceId += 1;
  const compatibilityKey = contentRevision == null
    ? [
        (
          WALKAROUND_TEXTURE_SOURCE_SESSION_SALT[0] ^
          (sourceId >>> 0)
        ) >>> 0,
        (
          WALKAROUND_TEXTURE_SOURCE_SESSION_SALT[1] ^
          (Math.floor(sourceId / 0x1_0000_0000) >>> 0) ^
          Math.imul(sourceId >>> 0, 0x85ebca6b)
        ) >>> 0,
      ] as const
    : hashTextureContentRevision(
        `vitrum.walkaround.texture-content.v1:${contentRevision}`,
      );

  return Object.freeze({
    kind: WALKAROUND_WEBGPU_TEXTURE_SOURCE_KIND,
    [WALKAROUND_WEBGPU_TEXTURE_SOURCE_BRAND]: true as const,
    sourceId,
    compatibilityKeyLo: compatibilityKey[0],
    compatibilityKeyHi: compatibilityKey[1],
    device,
    texture,
    format: sourceFormat,
    colorSpace: sourceColorSpace,
    dimension: '2d' as const,
    ownership: 'host' as const,
    baseMipLevel,
    arrayLayer,
    width,
    height,
    ...(contentRevision != null
      ? { contentRevision }
      : {}),
    ...(cpuMirror ? { cpuMirror } : {}),
  });
}

export function isWalkaroundWebGpuTextureSource(
  value: unknown,
): value is WalkaroundWebGpuTextureSource {
  if (value == null || typeof value !== 'object') return false;
  const candidate = value as Partial<WalkaroundWebGpuTextureSource> & {
    readonly [WALKAROUND_WEBGPU_TEXTURE_SOURCE_BRAND]?: unknown;
  };
  return (
    candidate.kind === WALKAROUND_WEBGPU_TEXTURE_SOURCE_KIND &&
    candidate[WALKAROUND_WEBGPU_TEXTURE_SOURCE_BRAND] === true
  );
}
