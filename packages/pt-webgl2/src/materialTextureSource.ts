const PT_WEBGL2_TEXTURE_SOURCE_BRAND = Symbol(
  'vitrum.pt-webgl2.textureSource.brand',
);

let nextPtWebgl2TextureSourceId = 1;

export const PT_WEBGL2_TEXTURE_SOURCE_KIND =
  'vitrum.pt-webgl2.cpu-texture-source' as const;

export const PT_WEBGL2_TEXTURE_SNAPSHOT_BUDGET_BYTES = 256 * 1024 * 1024;

export type PtWebgl2TextureColorSpace = 'srgb' | 'linear';

export type PtWebgl2TextureDataType =
  | 'uint8'
  | 'uint16'
  | 'float16'
  | 'half-float'
  | 'float32';

export interface PtWebgl2RawTextureSourceInput {
  readonly width: number;
  readonly height: number;
  readonly channels: 1 | 2 | 3 | 4;
  readonly dataType: PtWebgl2TextureDataType;
  readonly data: ArrayLike<number>;
}

export interface PtWebgl2TextureSourceOptions {
  /** Transfer function of RGB values in the resulting immutable snapshot. */
  readonly colorSpace: PtWebgl2TextureColorSpace;
}

export interface PtWebgl2TextureCpuMirror {
  readonly width: number;
  readonly height: number;
  readonly channels: 1 | 2 | 3 | 4;
  readonly dataType: PtWebgl2TextureDataType;
  readonly colorSpace: PtWebgl2TextureColorSpace;
  readonly data: ArrayLike<number>;
}

/**
 * Immutable material-texture descriptor consumed by pt-webgl2's existing
 * `cpuMirror` atlas path. Browser image/canvas inputs are copied immediately;
 * the backend never retains or owns the source DOM object.
 */
export interface PtWebgl2TextureSource {
  readonly kind: typeof PT_WEBGL2_TEXTURE_SOURCE_KIND;
  readonly [PT_WEBGL2_TEXTURE_SOURCE_BRAND]: true;
  readonly sourceId: number;
  readonly ownership: 'immutable-snapshot';
  readonly width: number;
  readonly height: number;
  readonly colorSpace: PtWebgl2TextureColorSpace;
  readonly cpuMirror: PtWebgl2TextureCpuMirror;
}

type BrowserTextureSource = TexImageSource;

interface CanvasReadbackContext {
  drawImage(source: unknown, dx: number, dy: number, dw: number, dh: number): void;
  getImageData(sx: number, sy: number, sw: number, sh: number): {
    readonly width: number;
    readonly height: number;
    readonly data: ArrayLike<number>;
  };
}

interface CanvasReadbackSurface {
  width: number;
  height: number;
  getContext(
    type: '2d',
    options?: { readonly willReadFrequently?: boolean },
  ): CanvasReadbackContext | null;
}

function assertPositiveDimension(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive safe integer; received ${String(value)}.`);
  }
  return value;
}

function dataTypeBytes(dataType: PtWebgl2TextureDataType): number {
  return dataType === 'uint8' ? 1 : dataType === 'float32' ? 4 : 2;
}

function immutableNumericSnapshot(
  data: ArrayLike<number>,
  dataType: PtWebgl2TextureDataType,
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
  Object.defineProperty(target, Symbol.toStringTag, {
    value: dataType === 'uint8'
      ? 'Uint8Array'
      : dataType === 'float32'
        ? 'Float32Array'
        : 'Uint16Array',
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
      throw new TypeError('PtWebgl2TextureSource data is immutable.');
    },
    defineProperty() {
      throw new TypeError('PtWebgl2TextureSource data is immutable.');
    },
    deleteProperty() {
      throw new TypeError('PtWebgl2TextureSource data is immutable.');
    },
  });
}

function createMirror(
  input: PtWebgl2RawTextureSourceInput,
  colorSpace: PtWebgl2TextureColorSpace,
): PtWebgl2TextureCpuMirror {
  if (input == null || typeof input !== 'object' || Array.isArray(input)) {
    throw new TypeError('createPtWebgl2TextureSource: raw input must be an object.');
  }
  const width = assertPositiveDimension(input.width, 'createPtWebgl2TextureSource: width');
  const height = assertPositiveDimension(input.height, 'createPtWebgl2TextureSource: height');
  if (![1, 2, 3, 4].includes(input.channels)) {
    throw new RangeError(
      `createPtWebgl2TextureSource: channels must be 1, 2, 3, or 4; received ${String(input.channels)}.`,
    );
  }
  if (!['uint8', 'uint16', 'float16', 'half-float', 'float32'].includes(input.dataType)) {
    throw new RangeError(
      `createPtWebgl2TextureSource: unsupported dataType ${String(input.dataType)}.`,
    );
  }
  if (input.data == null || typeof input.data.length !== 'number') {
    throw new TypeError('createPtWebgl2TextureSource: data must be array-like.');
  }
  const pixelCount = width * height;
  const expectedLength = pixelCount * input.channels;
  const snapshotBytes = expectedLength * dataTypeBytes(input.dataType);
  if (
    !Number.isSafeInteger(pixelCount) ||
    !Number.isSafeInteger(expectedLength) ||
    !Number.isSafeInteger(snapshotBytes) ||
    snapshotBytes > PT_WEBGL2_TEXTURE_SNAPSHOT_BUDGET_BYTES
  ) {
    throw new RangeError(
      `createPtWebgl2TextureSource: immutable snapshot exceeds the ` +
      `${PT_WEBGL2_TEXTURE_SNAPSHOT_BUDGET_BYTES}-byte budget.`,
    );
  }
  if (!Number.isSafeInteger(input.data.length) || input.data.length !== expectedLength) {
    throw new RangeError(
      `createPtWebgl2TextureSource: data length must be exactly ${expectedLength}; ` +
      `received ${String(input.data.length)}.`,
    );
  }
  const integerMax = input.dataType === 'uint8'
    ? 255
    : input.dataType === 'float32'
      ? null
      : 65535;
  for (let index = 0; index < input.data.length; index += 1) {
    const value = Number(input.data[index]);
    if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
      throw new RangeError(
        `createPtWebgl2TextureSource: data[${index}] must be finite and representable as f32.`,
      );
    }
    if (integerMax != null && (!Number.isInteger(value) || value < 0 || value > integerMax)) {
      throw new RangeError(
        `createPtWebgl2TextureSource: data[${index}] must be an integer in [0, ${integerMax}] ` +
        `for ${input.dataType}.`,
      );
    }
  }
  return Object.freeze({
    width,
    height,
    channels: input.channels,
    dataType: input.dataType,
    colorSpace,
    data: immutableNumericSnapshot(input.data, input.dataType),
  });
}

function browserSourceDimension(source: unknown, axis: 'width' | 'height'): number {
  if (source == null || typeof source !== 'object') return 0;
  const record = source as Record<string, unknown>;
  const candidates = axis === 'width'
    ? ['naturalWidth', 'videoWidth', 'displayWidth', 'codedWidth', 'width']
    : ['naturalHeight', 'videoHeight', 'displayHeight', 'codedHeight', 'height'];
  for (const key of candidates) {
    const value = record[key];
    if (typeof value === 'number' && Number.isSafeInteger(value) && value > 0) return value;
  }
  return 0;
}

function createReadbackSurface(width: number, height: number): CanvasReadbackSurface {
  const runtime = globalThis as unknown as {
    readonly OffscreenCanvas?: new (width: number, height: number) => CanvasReadbackSurface;
    readonly document?: {
      createElement(tagName: 'canvas'): CanvasReadbackSurface;
    };
  };
  if (typeof runtime.OffscreenCanvas === 'function') {
    return new runtime.OffscreenCanvas(width, height);
  }
  if (runtime.document?.createElement != null) {
    const canvas = runtime.document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
  throw new Error(
    'createPtWebgl2TextureSource: browser image snapshotting requires OffscreenCanvas or document.createElement.',
  );
}

function snapshotBrowserSource(
  source: BrowserTextureSource,
  colorSpace: PtWebgl2TextureColorSpace,
): PtWebgl2TextureCpuMirror {
  const width = assertPositiveDimension(
    browserSourceDimension(source, 'width'),
    'createPtWebgl2TextureSource: browser source width',
  );
  const height = assertPositiveDimension(
    browserSourceDimension(source, 'height'),
    'createPtWebgl2TextureSource: browser source height',
  );
  const surface = createReadbackSurface(width, height);
  const context = surface.getContext('2d', { willReadFrequently: true });
  if (context == null) {
    throw new Error('createPtWebgl2TextureSource: could not acquire a 2D readback context.');
  }
  let imageData: ReturnType<CanvasReadbackContext['getImageData']>;
  try {
    context.drawImage(source, 0, 0, width, height);
    imageData = context.getImageData(0, 0, width, height);
  } catch (error) {
    throw new Error(
      'createPtWebgl2TextureSource: browser source could not be read synchronously; ' +
      `ensure it is loaded and origin-clean. ${String(error)}`,
    );
  }
  if (imageData.width !== width || imageData.height !== height) {
    throw new RangeError(
      `createPtWebgl2TextureSource: readback returned ${String(imageData.width)}x` +
      `${String(imageData.height)} for a ${width}x${height} source.`,
    );
  }
  return createMirror({
    width,
    height,
    channels: 4,
    dataType: 'uint8',
    data: imageData.data,
  }, colorSpace);
}

function isRawInput(value: unknown): value is PtWebgl2RawTextureSourceInput {
  if (value == null || typeof value !== 'object') return false;
  const candidate = value as Partial<PtWebgl2RawTextureSourceInput>;
  return candidate.data != null &&
    candidate.channels != null &&
    candidate.dataType != null;
}

function isImageDataInput(value: unknown): value is {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number>;
} {
  return value != null &&
    typeof value === 'object' &&
    Object.prototype.toString.call(value) === '[object ImageData]';
}

function snapshotImageData(
  source: { readonly width: number; readonly height: number; readonly data: ArrayLike<number> },
  colorSpace: PtWebgl2TextureColorSpace,
): PtWebgl2TextureCpuMirror {
  const backingType = Object.prototype.toString.call(source.data);
  return createMirror({
    width: source.width,
    height: source.height,
    channels: 4,
    dataType: backingType === '[object Uint8ClampedArray]' || backingType === '[object Uint8Array]'
      ? 'uint8'
      : 'float32',
    data: source.data,
  }, colorSpace);
}

/**
 * Snapshot a raw pixel payload or loaded browser TexImageSource into an
 * immutable descriptor accepted by every pt-webgl2 material-map slot.
 */
export function createPtWebgl2TextureSource(
  source: PtWebgl2RawTextureSourceInput | BrowserTextureSource,
  options: PtWebgl2TextureSourceOptions,
): PtWebgl2TextureSource {
  if (options == null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('createPtWebgl2TextureSource: options must be an object.');
  }
  if (options.colorSpace !== 'srgb' && options.colorSpace !== 'linear') {
    throw new RangeError(
      `createPtWebgl2TextureSource: colorSpace must be "srgb" or "linear"; ` +
      `received ${String(options.colorSpace)}.`,
    );
  }
  if (source == null || typeof source !== 'object') {
    throw new TypeError('createPtWebgl2TextureSource: source must be a pixel payload or TexImageSource.');
  }
  const cpuMirror = isRawInput(source)
    ? createMirror(source, options.colorSpace)
    : isImageDataInput(source)
      ? snapshotImageData(source, options.colorSpace)
      : snapshotBrowserSource(source, options.colorSpace);
  if (!Number.isSafeInteger(nextPtWebgl2TextureSourceId)) {
    throw new RangeError('createPtWebgl2TextureSource: descriptor identity space is exhausted.');
  }
  const sourceId = nextPtWebgl2TextureSourceId;
  nextPtWebgl2TextureSourceId += 1;
  return Object.freeze({
    kind: PT_WEBGL2_TEXTURE_SOURCE_KIND,
    [PT_WEBGL2_TEXTURE_SOURCE_BRAND]: true as const,
    sourceId,
    ownership: 'immutable-snapshot' as const,
    width: cpuMirror.width,
    height: cpuMirror.height,
    colorSpace: cpuMirror.colorSpace,
    cpuMirror,
  });
}

export function isPtWebgl2TextureSource(value: unknown): value is PtWebgl2TextureSource {
  if (value == null || typeof value !== 'object') return false;
  const candidate = value as Partial<PtWebgl2TextureSource> & {
    readonly [PT_WEBGL2_TEXTURE_SOURCE_BRAND]?: unknown;
  };
  return candidate.kind === PT_WEBGL2_TEXTURE_SOURCE_KIND &&
    candidate[PT_WEBGL2_TEXTURE_SOURCE_BRAND] === true;
}
