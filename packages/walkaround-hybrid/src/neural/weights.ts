/**
 * weights.ts — ModelWeights interface + binary weight loader for the vitrum neural denoiser.
 *
 * Binary format (.vitrum-model):
 * ─────────────────────────────
 * Header:
 *   [u32 magic=0xDEAF1984, u32 version=1, u32 layerCount]
 * Per layer:
 *   [u32 nameLen, char[nameLen] name (UTF-8),
 *    u32 weightCount, f32[weightCount] weights,
 *    u32 biasCount,   f32[biasCount]   biases]
 *
 * Weight layout per layer kind:
 *   Conv2D:          OIKW  (outputC × inputC × kH × kW)  — matches PyTorch Conv2d
 *   ConvTranspose2D: IOKW  (inputC × outputC × kH × kW) — matches PyTorch ConvTranspose2d
 *   All others:      No weights (empty arrays).
 *
 * This format is round-tripped by tools/neural-denoiser-training/export_weights.py.
 */

export const VITRUM_MODEL_MAGIC   = 0xDEAF1984 >>> 0;
export const VITRUM_MODEL_VERSION = 1;

// ── ModelWeights ──────────────────────────────────────────────────────────────

/** Per-layer weight payload. */
export interface LayerWeights {
  /** Layer name matching the LayerSpec.name in unetArchitecture.ts. */
  readonly name: string;
  /** Weight tensor as Float32Array. Length = 0 for non-parameterized layers. */
  readonly weights: Float32Array;
  /** Bias tensor as Float32Array. Length = 0 for layers with no bias. */
  readonly biases: Float32Array;
}

/** Full model weights for one U-Net checkpoint. */
export interface ModelWeights {
  /** Layer weights in execution order (matches UNetSpec.layers order). */
  readonly layers: readonly LayerWeights[];
}

// ── Spec validation ─────────────────────────────────────────────────────────

type WeightLayerSpec = {
  readonly name: string;
  readonly kind: string;
  readonly weightLayout: 'OIKW' | 'IOKW' | 'none';
  readonly params: {
    readonly inC: number;
    readonly outC: number;
    readonly kH?: number;
    readonly kW?: number;
  };
};

export interface WeightSpec {
  readonly layers: readonly WeightLayerSpec[];
}

/**
 * Validate that a checkpoint exactly matches the supplied inference spec before
 * any GPU buffers are allocated.
 *
 * Exported training checkpoints are allowed to omit parameterless layers
 * (`relu`, `skipAdd`, `inputPack`, etc.), but every parameterized conv layer
 * must be present exactly once with the expected weight/bias lengths and finite
 * f32 payloads. Unknown layers, duplicate layers, parameterless layers with
 * payloads, and malformed values throw synchronously so `denoiser:'neural'`
 * cannot silently run with placeholder weights.
 */
export function validateWeightsForSpec(spec: WeightSpec, weights: ModelWeights): void {
  const specByName = new Map<string, WeightLayerSpec>();
  for (const layer of spec.layers) {
    if (specByName.has(layer.name)) {
      throw new Error(`[validateWeightsForSpec] duplicate spec layer '${layer.name}'`);
    }
    specByName.set(layer.name, layer);
  }

  const supplied = new Set<string>();
  for (const layerWeights of weights.layers) {
    if (supplied.has(layerWeights.name)) {
      throw new Error(`[validateWeightsForSpec] duplicate weights for layer '${layerWeights.name}'`);
    }
    supplied.add(layerWeights.name);

    const layer = specByName.get(layerWeights.name);
    if (layer == null) {
      throw new Error(`[validateWeightsForSpec] unknown layer '${layerWeights.name}' in checkpoint`);
    }

    const expected = expectedParamCounts(layer);
    if (layerWeights.weights.length !== expected.weights) {
      throw new Error(
        `[validateWeightsForSpec] layer '${layerWeights.name}' weight length ` +
        `${layerWeights.weights.length} != expected ${expected.weights}`,
      );
    }
    if (layerWeights.biases.length !== expected.biases) {
      throw new Error(
        `[validateWeightsForSpec] layer '${layerWeights.name}' bias length ` +
        `${layerWeights.biases.length} != expected ${expected.biases}`,
      );
    }
    assertFiniteArray(layerWeights.weights, `${layerWeights.name}.weights`);
    assertFiniteArray(layerWeights.biases, `${layerWeights.name}.biases`);
  }

  for (const layer of spec.layers) {
    const expected = expectedParamCounts(layer);
    if ((expected.weights > 0 || expected.biases > 0) && !supplied.has(layer.name)) {
      throw new Error(`[validateWeightsForSpec] missing weights for layer '${layer.name}'`);
    }
  }
}

function expectedParamCounts(layer: WeightLayerSpec): { weights: number; biases: number } {
  if (layer.kind !== 'conv2d' && layer.kind !== 'transposedConv2d') {
    return { weights: 0, biases: 0 };
  }
  const { inC, outC, kH = 1, kW = 1 } = layer.params;
  return {
    weights: inC * outC * kH * kW,
    biases: outC,
  };
}

function assertFiniteArray(values: Float32Array, label: string): void {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(`[validateWeightsForSpec] ${label}[${i}] is not finite`);
    }
  }
}

// ── Random-weights helper (for examples + acceptance tests) ──────────────────

/**
 * Build a ModelWeights matching a U-Net spec with deterministic He-initialized
 * random weights. The denoising output will NOT be meaningful — this is for
 * pipeline-wiring smoke tests and the W10 example.
 * The repo tracks two limited research checkpoints under
 * `tools/neural-denoiser-training/checkpoints/`, but the walkaround package
 * does not ship production neural weights. Hosts opting into `denoiser:
 * 'neural'` should provide their own validated `.vitrum-model` payload.
 *
 * Deterministic via a Park-Miller LCG seeded from `seed` so the same call
 * with the same spec + seed produces bit-identical output across runs.
 *
 * @param spec  Architecture spec (typically `WALKAROUND_DENOISER_UNET_SPEC`).
 * @param seed  LCG seed (default 0xDEAF1984 — matches the model magic).
 *
 * ---
 * Repo-only research checkpoints
 * ------------------------------
 *
 * `tools/neural-denoiser-training/checkpoints/starter-v1.vitrum-model`
 *   Produced: 2026-06-10, CPU training (torch 2.12.0+cpu, Python 3.14).
 *   Dataset:  32 pairs × 128×128, Cornell box (CPU brute-force path tracer,
 *             1-spp noisy / 256-spp clean), generated by capture-dataset.mjs.
 *   Training: 20 epochs, batch=2, lr=1e-4, CosineAnnealingLR.
 *   Loss:     L1 + 0.1×SSIM, epoch-1 = 0.2965 → epoch-20 = 0.1198 (monotonic decrease).
 *   Params:   535,107 (f32), file size: ~2.04 MB.
 *   Quality:  STARTER artifact only — trained on a tiny Cornell-only CPU-rendered set.
 *             Reduces obvious 1-spp noise on Cornell-style scenes; generalizes
 *             poorly to varied geometry, materials, or lighting configurations.
 *             Kept for compatibility (the v2 checkpoint supersedes it for general use).
 *   Validation: round-trip loader test (neuralWeightsRoundTrip.test.ts) passes.
 *
 * `tools/neural-denoiser-training/checkpoints/v2-random.vitrum-model`
 *   Produced: 2026-06-10, CPU training (torch 2.12.0+cpu, Python 3.14).
 *   Dataset:  256 pairs × 128×128 — DIVERSE:
 *               128 Cornell box (CPU path tracer, seed=1984, 1-spp noisy / 256-spp clean)
 *               128 random scenes (per-pair randomized geometry, materials, lights,
 *               cameras — see capture-dataset.mjs --scene random):
 *                 • 2–5 objects (axis-aligned boxes + sphere approximations)
 *                 • Random diffuse albedo per object (full hue+saturation range)
 *                 • Random area-light position/size/intensity (5–20 W/sr·m²)
 *                 • 50% chance of coloured sky (gradient environment)
 *                 • Random camera position/aim/FOV (50°–90° vertical)
 *               Generated with capture-dataset.mjs --scene random --seed 42.
 *   Training: 100 epochs, batch=4, lr=1e-4, CosineAnnealingLR, patch=128
 *             (full-image, no random crop — dataset images are exactly 128×128).
 *   Loss:     L1 + 0.1×SSIM, epoch-1 = 0.1776 → epoch-100 = 0.0931 (monotonic).
 *             Loss plateau visible from epoch ~50 onward (0.095 → 0.093).
 *   Params:   535,107 (f32), file size: ~2.04 MB.
 *   Quality:  Better generalisation than v1 due to scene diversity — handles varied
 *             normals, depth distributions, albedo statistics, and multiple light
 *             placement configurations. Still CPU-trained at 128×128 resolution.
 *             Produces visibly smoother denoising on non-Cornell scenes compared to
 *             v1. Known ceiling: all training data is 128px Lambertian-only with no
 *             specular/metal/glass materials; will struggle on highly glossy content.
 *             Production-quality denoising requires a GPU-rendered HD dataset
 *             (see tools/neural-denoiser-training/README.md §GPU capture).
 *   Validation: round-trip loader test (neuralWeightsRoundTrip.test.ts) passes.
 */
export function buildRandomWeightsForSpec(
  spec: { layers: readonly { name: string; weightLayout: 'OIKW' | 'IOKW' | 'none'; params: { inC: number; outC: number; kH?: number; kW?: number } }[] },
  seed: number = VITRUM_MODEL_MAGIC,
): ModelWeights {
  let s = (seed >>> 0) || 1;
  const lcg01 = (): number => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
  const lcgRange = (lo: number, hi: number): number => lo + lcg01() * (hi - lo);

  const layers: LayerWeights[] = spec.layers.map((layer) => {
    if (layer.weightLayout === 'none') {
      return { name: layer.name, weights: new Float32Array(0), biases: new Float32Array(0) };
    }
    const { inC, outC, kH = 1, kW = 1 } = layer.params;
    const count = inC * outC * kH * kW;
    // He init: scale = sqrt(2 / fan_in). For conv2d fan_in = inC × kH × kW.
    // Same scale used for both OIKW (conv2d) and IOKW (transposedConv2d).
    const scale = Math.sqrt(2.0 / Math.max(1, inC * kH * kW));
    const weights = new Float32Array(count);
    for (let i = 0; i < count; i++) weights[i] = lcgRange(-scale, scale);
    const biases = new Float32Array(outC);
    for (let i = 0; i < outC; i++) biases[i] = lcgRange(-0.01, 0.01);
    return { name: layer.name, weights, biases };
  });

  return { layers };
}

// ── Binary loader ─────────────────────────────────────────────────────────────

/**
 * Load model weights from an ArrayBuffer in the vitrum-model binary format.
 *
 * @throws {Error} on magic mismatch, unsupported version, or truncated data.
 *
 * Byte-level layout:
 *   Offset 0: u32 magic (little-endian)
 *   Offset 4: u32 version
 *   Offset 8: u32 layerCount
 *   Offset 12: layer records (variable-length)
 *
 * Each layer record:
 *   u32 nameLen
 *   u8[nameLen] name (UTF-8, not null-terminated)
 *   u32 weightCount
 *   f32[weightCount] weights
 *   u32 biasCount
 *   f32[biasCount] biases
 */
export function loadWeightsFromArrayBuffer(bytes: ArrayBuffer): ModelWeights {
  const view = new DataView(bytes);
  let offset = 0;

  function readU32(): number {
    if (offset + 4 > bytes.byteLength) {
      throw new Error(`[loadWeightsFromArrayBuffer] truncated at offset ${offset}`);
    }
    const v = view.getUint32(offset, /* littleEndian */ true);
    offset += 4;
    return v;
  }

  function readF32Array(count: number): Float32Array {
    const byteLen = count * 4;
    if (offset + byteLen > bytes.byteLength) {
      throw new Error(
        `[loadWeightsFromArrayBuffer] truncated reading ${count} f32s at offset ${offset}`,
      );
    }
    // Slice to own the memory (avoids holding a reference to the full ArrayBuffer).
    const arr = new Float32Array(bytes.slice(offset, offset + byteLen));
    offset += byteLen;
    return arr;
  }

  function readString(len: number): string {
    if (offset + len > bytes.byteLength) {
      throw new Error(
        `[loadWeightsFromArrayBuffer] truncated reading string of length ${len} at offset ${offset}`,
      );
    }
    const arr = new Uint8Array(bytes, offset, len);
    offset += len;
    return new TextDecoder().decode(arr);
  }

  // ── Header ────────────────────────────────────────────────────────────────
  const magic = readU32();
  if (magic !== VITRUM_MODEL_MAGIC) {
    throw new Error(
      `[loadWeightsFromArrayBuffer] invalid magic: expected 0x${VITRUM_MODEL_MAGIC.toString(16).toUpperCase()}, ` +
      `got 0x${magic.toString(16).toUpperCase()}`,
    );
  }

  const version = readU32();
  if (version !== VITRUM_MODEL_VERSION) {
    throw new Error(
      `[loadWeightsFromArrayBuffer] unsupported version ${version} (expected ${VITRUM_MODEL_VERSION})`,
    );
  }

  const layerCount = readU32();
  const layers: LayerWeights[] = [];

  // ── Layer records ─────────────────────────────────────────────────────────
  for (let i = 0; i < layerCount; i++) {
    const nameLen     = readU32();
    const name        = readString(nameLen);
    const weightCount = readU32();
    const weights     = readF32Array(weightCount);
    const biasCount   = readU32();
    const biases      = readF32Array(biasCount);
    layers.push({ name, weights, biases });
  }

  if (offset !== bytes.byteLength) {
    throw new Error(
      `[loadWeightsFromArrayBuffer] trailing ${bytes.byteLength - offset} byte(s) after ` +
      `${layerCount} layer record(s) at offset ${offset}`,
    );
  }

  return { layers };
}

// ── Binary serialiser (CPU-side mirror of export_weights.py) ──────────────────

/**
 * Serialize ModelWeights to the vitrum-model binary format.
 *
 * This is a TypeScript mirror of `tools/neural-denoiser-training/export_weights.py`'s
 * output stage, used in tests for round-trip validation.
 */
export function serializeWeightsToArrayBuffer(weights: ModelWeights): ArrayBuffer {
  // Calculate total size.
  let totalBytes = 12; // magic + version + layerCount
  for (const layer of weights.layers) {
    const nameBytes = new TextEncoder().encode(layer.name);
    totalBytes += 4;                          // nameLen
    totalBytes += nameBytes.byteLength;       // name
    totalBytes += 4;                          // weightCount
    totalBytes += layer.weights.length * 4;   // weights (f32)
    totalBytes += 4;                          // biasCount
    totalBytes += layer.biases.length * 4;    // biases (f32)
  }

  const buf  = new ArrayBuffer(totalBytes);
  const view = new DataView(buf);
  const u8   = new Uint8Array(buf);
  let offset = 0;

  function writeU32(v: number): void {
    view.setUint32(offset, v >>> 0, /* littleEndian */ true);
    offset += 4;
  }

  function writeF32Array(arr: Float32Array): void {
    u8.set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength), offset);
    offset += arr.byteLength;
  }

  function writeString(s: string): void {
    const encoded = new TextEncoder().encode(s);
    writeU32(encoded.byteLength);
    u8.set(encoded, offset);
    offset += encoded.byteLength;
  }

  // ── Header ────────────────────────────────────────────────────────────────
  writeU32(VITRUM_MODEL_MAGIC);
  writeU32(VITRUM_MODEL_VERSION);
  writeU32(weights.layers.length);

  // ── Layer records ─────────────────────────────────────────────────────────
  for (const layer of weights.layers) {
    writeString(layer.name);
    writeU32(layer.weights.length);
    writeF32Array(layer.weights);
    writeU32(layer.biases.length);
    writeF32Array(layer.biases);
  }

  return buf;
}
