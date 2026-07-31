import { Buffer } from "node:buffer";
import { deflateSync } from "node:zlib";

export const RENDERER_DATASET_CAPTURE_SOURCE =
  "vitrum-walkaround-hybrid-neural-input-readback";
export const RENDERER_DATASET_SCHEMA =
  "vitrum.neural-denoiser.dataset.v1";
export const RENDERER_CAPTURE_CONFIG_SCHEMA =
  "vitrum.neural-denoiser.renderer-capture-config.v1";

const VHDR_MAGIC = 0x52444856;
const VHDR_VERSION = 1;
const U32_MAX = 0xffff_ffff;

/** @param {string} text @param {string} label */
function parseInteger(text, label) {
  if (!/^(?:0|[1-9]\d*)$/.test(text)) {
    throw new RangeError(`${label} must be a base-10 non-negative integer (got ${text})`);
  }
  const value = Number(text);
  if (!Number.isSafeInteger(value)) {
    throw new RangeError(`${label} exceeds the JavaScript safe-integer range`);
  }
  return value;
}

/**
 * @param {readonly string[]} argv
 */
export function parseRendererDatasetArgs(argv) {
  const args = {
    out: "data_renderer",
    pairs: 4,
    size: 128,
    cleanFrames: 4096,
    warmupFrames: 8,
    seed: 1984,
    scene: "cornell_box",
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const option = argv[index];
    const next = () => {
      const value = argv[index + 1];
      if (value == null || value.startsWith("--")) {
        throw new TypeError(`${option} requires a value`);
      }
      index += 1;
      return value;
    };
    switch (option) {
      case "--out":
        args.out = next();
        break;
      case "--pairs":
        args.pairs = parseInteger(next(), "--pairs");
        break;
      case "--size":
        args.size = parseInteger(next(), "--size");
        break;
      case "--clean-frames":
        args.cleanFrames = parseInteger(next(), "--clean-frames");
        break;
      case "--warmup-frames":
        args.warmupFrames = parseInteger(next(), "--warmup-frames");
        break;
      case "--seed":
        args.seed = parseInteger(next(), "--seed");
        break;
      case "--scene":
        args.scene = next();
        break;
      case "--help":
      case "-h":
        args.help = true;
        break;
      default:
        throw new TypeError(`Unknown argument: ${option}`);
    }
  }
  return validateRendererDatasetConfig(args);
}

/**
 * @param {{
 *   out: string,
 *   pairs: number,
 *   size: number,
 *   cleanFrames: number,
 *   warmupFrames: number,
 *   seed: number,
 *   scene: string,
 *   help?: boolean,
 * }} config
 */
export function validateRendererDatasetConfig(config) {
  if (typeof config.out !== "string" || config.out.trim().length === 0) {
    throw new TypeError("--out must be a non-empty path");
  }
  for (const [label, value, allowZero] of [
    ["--pairs", config.pairs, false],
    ["--size", config.size, false],
    ["--clean-frames", config.cleanFrames, false],
    ["--warmup-frames", config.warmupFrames, true],
    ["--seed", config.seed, true],
  ]) {
    if (!Number.isSafeInteger(value) || value < (allowZero ? 0 : 1)) {
      throw new RangeError(`${label} must be ${allowZero ? "a non-negative" : "a positive"} safe integer`);
    }
  }
  if (config.size < 8 || config.size > 8192 || config.size % 8 !== 0) {
    throw new RangeError("--size must be divisible by 8 and within 8..8192");
  }
  if (config.seed > U32_MAX) {
    throw new RangeError(`--seed must be within 0..${U32_MAX}`);
  }
  const totalFrames = config.warmupFrames + config.cleanFrames;
  if (!Number.isSafeInteger(totalFrames) || totalFrames > U32_MAX + 1) {
    throw new RangeError(
      `--warmup-frames + --clean-frames must not exceed ${U32_MAX + 1}`,
    );
  }
  if (config.scene !== "cornell_box") {
    throw new RangeError(`Unsupported --scene "${config.scene}"; expected cornell_box`);
  }
  return Object.freeze({ ...config, out: config.out.trim(), help: config.help === true });
}

/**
 * Stable u32 seed schedule. BigInt arithmetic avoids precision loss before the
 * explicit unsigned-32-bit reduction.
 */
export function rendererDatasetFrameSeed(baseSeed, pairIndex, frameIndex) {
  for (const [label, value] of [
    ["baseSeed", baseSeed],
    ["pairIndex", pairIndex],
    ["frameIndex", frameIndex],
  ]) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`${label} must be a non-negative safe integer`);
    }
  }
  return Number(BigInt.asUintN(
    32,
    BigInt(baseSeed) * 1_000_003n +
      BigInt(pairIndex) * 7_919n +
      BigInt(frameIndex) * 6_364_136_223_846_793_005n +
      1_442_695_040_888_963_407n,
  ));
}

/**
 * Validate and acquire the concrete walkaround-only capture method. This
 * deliberately does not accept an opaque callback/adapter: the generator must
 * invoke the shipped engine's `captureDenoiserTrainingInputs` method itself.
 *
 * @param {unknown} engine
 * @param {number} width
 * @param {number} height
 */
export async function captureRendererTrainingInput(engine, width, height) {
  if (
    engine == null ||
    typeof engine !== "object" ||
    typeof engine.captureDenoiserTrainingInputs !== "function"
  ) {
    throw new TypeError(
      "renderer dataset capture requires HybridEngine.captureDenoiserTrainingInputs()",
    );
  }
  const capture = await engine.captureDenoiserTrainingInputs();
  if (capture == null) {
    throw new Error("HybridEngine.captureDenoiserTrainingInputs() returned null after a rendered frame");
  }
  validateRendererTrainingCapture(capture, width, height);
  return capture;
}

/**
 * @param {unknown} value
 * @param {number} width
 * @param {number} height
 */
export function validateRendererTrainingCapture(value, width, height) {
  if (value == null || typeof value !== "object") {
    throw new TypeError("renderer training capture must be an object");
  }
  if (value.width !== width || value.height !== height) {
    throw new RangeError(
      `renderer capture dimensions ${String(value.width)}x${String(value.height)} ` +
        `do not match expected ${width}x${height}`,
    );
  }
  const expectedLength = width * height * 3;
  for (const field of ["radiance", "albedo", "worldNormal"]) {
    const array = value[field];
    if (!(array instanceof Float32Array) || array.length !== expectedLength) {
      throw new RangeError(
        `renderer capture ${field} must be Float32Array(${expectedLength})`,
      );
    }
    for (let index = 0; index < array.length; index += 1) {
      if (!Number.isFinite(array[index])) {
        throw new RangeError(`renderer capture ${field}[${index}] is non-finite`);
      }
    }
  }
}

export function createRgbAccumulator(width, height) {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError("accumulator dimensions must be positive safe integers");
  }
  return {
    sum: new Float64Array(width * height * 3),
    samples: 0,
  };
}

export function accumulateRendererRadiance(accumulator, radiance) {
  if (!(radiance instanceof Float32Array) || radiance.length !== accumulator.sum.length) {
    throw new RangeError("radiance does not match accumulator shape");
  }
  for (let index = 0; index < radiance.length; index += 1) {
    const value = radiance[index];
    if (!Number.isFinite(value)) {
      throw new RangeError(`radiance[${index}] is non-finite`);
    }
    accumulator.sum[index] += value;
  }
  accumulator.samples += 1;
}

export function finishRendererRadianceAverage(accumulator) {
  if (!Number.isSafeInteger(accumulator.samples) || accumulator.samples <= 0) {
    throw new RangeError("cannot finish an empty radiance accumulator");
  }
  const output = new Float32Array(accumulator.sum.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = accumulator.sum[index] / accumulator.samples;
  }
  return output;
}

export function encodeVhdr(linearRgb, width, height) {
  if (!(linearRgb instanceof Float32Array) || linearRgb.length !== width * height * 3) {
    throw new RangeError("VHDR source must be tightly packed Float32 RGB");
  }
  const header = Buffer.alloc(16);
  header.writeUInt32LE(VHDR_MAGIC, 0);
  header.writeUInt32LE(VHDR_VERSION, 4);
  header.writeUInt32LE(width, 8);
  header.writeUInt32LE(height, 12);
  const body = Buffer.alloc(linearRgb.length * Float32Array.BYTES_PER_ELEMENT);
  for (let index = 0; index < linearRgb.length; index += 1) {
    body.writeFloatLE(linearRgb[index], index * Float32Array.BYTES_PER_ELEMENT);
  }
  return Buffer.concat([header, body]);
}

function crc32(buffer) {
  let crc = ~0;
  for (let index = 0; index < buffer.length; index += 1) {
    crc ^= buffer[index];
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
  }
  return (~crc) >>> 0;
}

function pngChunk(type, data) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const checksum = Buffer.alloc(4);
  checksum.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([length, body, checksum]);
}

function toUnormByte(value) {
  return Math.max(0, Math.min(255, Math.round(value * 255)));
}

/**
 * Encode a tightly packed float RGB auxiliary into an 8-bit RGB PNG.
 * `signed` converts signed world normals through `n * 0.5 + 0.5`; albedo uses
 * the default unsigned path.
 */
export function encodeAuxiliaryPng(rgb, width, height, signed = false) {
  if (!(rgb instanceof Float32Array) || rgb.length !== width * height * 3) {
    throw new RangeError("PNG source must be tightly packed Float32 RGB");
  }
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const stride = width * 3;
  const rows = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y += 1) {
    const rowOffset = y * (stride + 1);
    rows[rowOffset] = 0;
    for (let channel = 0; channel < stride; channel += 1) {
      const source = rgb[y * stride + channel];
      rows[rowOffset + channel + 1] = toUnormByte(
        signed ? source * 0.5 + 0.5 : source,
      );
    }
  }
  return Buffer.concat([
    signature,
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", deflateSync(rows)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

export function rendererDatasetManifest(config) {
  const scenePath = `${config.out}/${config.scene}`;
  return {
    schema: RENDERER_DATASET_SCHEMA,
    id: `walkaround-renderer-seed-${config.seed}`,
    sceneCount: 1,
    sampleCount: config.pairs,
    noisySpp: 1,
    cleanReferenceSpp: config.cleanFrames,
    includesAlbedo: true,
    includesNormals: true,
    captureSource: RENDERER_DATASET_CAPTURE_SOURCE,
    tonemap: "linear-hdr",
    estimatorSampleUnit: "walkaround-renderer-frame",
    warmupFrames: config.warmupFrames,
    scenes: [{
      id: config.scene,
      sampleCount: config.pairs,
      noisyPath: `${scenePath}/noisy/`,
      cleanPath: `${scenePath}/clean/`,
      albedoPath: `${scenePath}/noisy/*_albedo.png`,
      normalPath: `${scenePath}/noisy/*_normal.png`,
    }],
  };
}

export function rendererCaptureConfigRecord(config) {
  return {
    schema: RENDERER_CAPTURE_CONFIG_SCHEMA,
    renderer: "@vitrum/walkaround-hybrid",
    captureMethod: "HybridEngine.captureDenoiserTrainingInputs",
    scene: config.scene,
    width: config.size,
    height: config.size,
    pairs: config.pairs,
    noisyFrames: 1,
    cleanFrames: config.cleanFrames,
    warmupFrames: config.warmupFrames,
    seed: config.seed,
    frameSeedSchedule: "rendererDatasetFrameSeed-v1-u32",
    denoiser: "none",
    checkerboard: false,
    targetFrameIntervalMs: null,
  };
}
