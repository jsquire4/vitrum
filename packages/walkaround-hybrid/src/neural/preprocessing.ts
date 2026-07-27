/** Versioned, finite preprocessing contract shared by training, export, and runtime. */
export interface NeuralPreprocessingContract {
  readonly version: 1;
  readonly color: 'linear-hdr-scaled';
  /** Divide clamped linear radiance by this value before inference. */
  readonly radianceScale: number;
  /** Clamp raw linear radiance to this trained ceiling before scaling. */
  readonly radianceClamp: number;
  readonly albedoRange: readonly [0, 1];
  readonly normalEncoding: 'signed-world-unit';
  readonly nonFinite: 'zero';
}
/** Bound used only to stop f32 accumulations from reaching infinity. */
export const NEURAL_INTERMEDIATE_MAGNITUDE_LIMIT = 65_504;

/**
 * Production preprocessing contract. A production checkpoint must contain an
 * exact copy of these trained scale/clamp semantics.
 */
export const NEURAL_PREPROCESSING_CONTRACT: NeuralPreprocessingContract = Object.freeze({
  version: 1,
  color: 'linear-hdr-scaled',
  radianceScale: 16,
  radianceClamp: 64,
  albedoRange: Object.freeze([0, 1] as const),
  normalEncoding: 'signed-world-unit',
  nonFinite: 'zero',
});

/** Compatibility behavior for v1/no-metadata checkpoints; never production-ready. */
export const NEURAL_LEGACY_PREPROCESSING_CONTRACT: NeuralPreprocessingContract = Object.freeze({
  version: 1,
  color: 'linear-hdr-scaled',
  radianceScale: 1,
  radianceClamp: NEURAL_INTERMEDIATE_MAGNITUDE_LIMIT,
  albedoRange: Object.freeze([0, 1] as const),
  normalEncoding: 'signed-world-unit',
  nonFinite: 'zero',
});

export function isNeuralPreprocessingContract(value: unknown): value is NeuralPreprocessingContract {
  if (value == null || typeof value !== 'object') return false;
  const c = value as Partial<NeuralPreprocessingContract>;
  return c.version === 1 &&
    c.color === 'linear-hdr-scaled' &&
    Number.isFinite(c.radianceScale) && (c.radianceScale ?? 0) > 0 &&
    Number.isFinite(c.radianceClamp) && (c.radianceClamp ?? 0) > 0 &&
    Array.isArray(c.albedoRange) && c.albedoRange[0] === 0 && c.albedoRange[1] === 1 &&
    c.normalEncoding === 'signed-world-unit' &&
    c.nonFinite === 'zero';
}

export function neuralPreprocessingContractsEqual(
  a: NeuralPreprocessingContract,
  b: NeuralPreprocessingContract,
): boolean {
  return a.version === b.version &&
    a.color === b.color &&
    a.radianceScale === b.radianceScale &&
    a.radianceClamp === b.radianceClamp &&
    a.albedoRange[0] === b.albedoRange[0] &&
    a.albedoRange[1] === b.albedoRange[1] &&
    a.normalEncoding === b.normalEncoding &&
    a.nonFinite === b.nonFinite;
}

export function preprocessingContractForCheckpoint(
  checkpoint: { readonly preprocessing?: NeuralPreprocessingContract } | undefined,
): NeuralPreprocessingContract {
  return checkpoint != null && isNeuralPreprocessingContract(checkpoint.preprocessing)
    ? checkpoint.preprocessing
    : NEURAL_LEGACY_PREPROCESSING_CONTRACT;
}

export function sanitizeNeuralSigned(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(
    -NEURAL_INTERMEDIATE_MAGNITUDE_LIMIT,
    Math.min(NEURAL_INTERMEDIATE_MAGNITUDE_LIMIT, value),
  );
}

export function preprocessNeuralRadiance(
  value: number,
  contract: NeuralPreprocessingContract = NEURAL_PREPROCESSING_CONTRACT,
): number {
  const finite = Number.isFinite(value) ? value : 0;
  return Math.max(0, Math.min(contract.radianceClamp, finite)) / contract.radianceScale;
}

export function postprocessNeuralRadiance(
  value: number,
  contract: NeuralPreprocessingContract = NEURAL_PREPROCESSING_CONTRACT,
): number {
  const finite = sanitizeNeuralSigned(value) * contract.radianceScale;
  return Math.max(0, Math.min(contract.radianceClamp, finite));
}

export function sanitizeNeuralAlbedo(value: number): number {
  return Math.max(0, Math.min(1, sanitizeNeuralSigned(value)));
}

export function sanitizeNeuralNormal(
  x: number,
  y: number,
  z: number,
): readonly [number, number, number] {
  const sx = sanitizeNeuralSigned(x);
  const sy = sanitizeNeuralSigned(y);
  const sz = sanitizeNeuralSigned(z);
  const lengthSquared = sx * sx + sy * sy + sz * sz;
  if (!Number.isFinite(lengthSquared) || lengthSquared < 1e-6) return [0, 1, 0];
  const inverseLength = 1 / Math.sqrt(lengthSquared);
  return [sx * inverseLength, sy * inverseLength, sz * inverseLength];
}

function wgslNumber(value: number): string {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError('invalid neural WGSL constant');
  return Number.isInteger(value) ? value.toFixed(1) : String(value);
}

/** WGSL helpers embedded by every neural kernel that can create or ingest data. */
export function neuralPreprocessingWgsl(
  contract: NeuralPreprocessingContract,
): string {
  if (!isNeuralPreprocessingContract(contract)) {
    throw new Error('[neural preprocessing] invalid preprocessing contract');
  }
  return /* wgsl */`
const NEURAL_INTERMEDIATE_LIMIT: f32 = ${wgslNumber(NEURAL_INTERMEDIATE_MAGNITUDE_LIMIT)};
const NEURAL_RADIANCE_SCALE: f32 = ${wgslNumber(contract.radianceScale)};
const NEURAL_RADIANCE_CLAMP: f32 = ${wgslNumber(contract.radianceClamp)};

fn neuralFinite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

fn neuralSanitizeSigned(value: f32) -> f32 {
  return clamp(select(0.0, value, neuralFinite(value)),
               -NEURAL_INTERMEDIATE_LIMIT,
               NEURAL_INTERMEDIATE_LIMIT);
}

fn neuralPreprocessRadiance(value: f32) -> f32 {
  return clamp(select(0.0, value, neuralFinite(value)), 0.0, NEURAL_RADIANCE_CLAMP)
       / NEURAL_RADIANCE_SCALE;
}

fn neuralPostprocessRadiance(value: f32) -> f32 {
  return clamp(neuralSanitizeSigned(value) * NEURAL_RADIANCE_SCALE,
               0.0, NEURAL_RADIANCE_CLAMP);
}

fn neuralSanitizeAlbedo(value: f32) -> f32 {
  return clamp(neuralSanitizeSigned(value), 0.0, 1.0);
}

fn neuralSanitizeNormal(value: vec3f) -> vec3f {
  let finiteValue = vec3f(
    neuralSanitizeSigned(value.x),
    neuralSanitizeSigned(value.y),
    neuralSanitizeSigned(value.z),
  );
  let lengthSquared = dot(finiteValue, finiteValue);
  let safe = select(vec3f(0.0, 1.0, 0.0), finiteValue,
                    neuralFinite(lengthSquared) && lengthSquared >= 1e-6);
  return normalize(safe);
}
`;
}

export const NEURAL_FINITE_WGSL = neuralPreprocessingWgsl(NEURAL_PREPROCESSING_CONTRACT);

/** Sanitize HWC RGB inputs into the exact runtime preprocessing contract. */
export function preprocessNeuralInputs(
  noisy: Float32Array,
  albedo: Float32Array,
  normals: Float32Array,
  contract: NeuralPreprocessingContract = NEURAL_PREPROCESSING_CONTRACT,
): Float32Array {
  if (noisy.length !== albedo.length || noisy.length !== normals.length || noisy.length % 3 !== 0) {
    throw new RangeError('[neural preprocessing] inputs must be equal-length interleaved RGB arrays');
  }
  const out = new Float32Array((noisy.length / 3) * 9);
  for (let p = 0; p < noisy.length / 3; p++) {
    const src = p * 3;
    const dst = p * 9;
    out[dst] = preprocessNeuralRadiance(noisy[src]!, contract);
    out[dst + 1] = preprocessNeuralRadiance(noisy[src + 1]!, contract);
    out[dst + 2] = preprocessNeuralRadiance(noisy[src + 2]!, contract);
    out[dst + 3] = sanitizeNeuralAlbedo(albedo[src]!);
    out[dst + 4] = sanitizeNeuralAlbedo(albedo[src + 1]!);
    out[dst + 5] = sanitizeNeuralAlbedo(albedo[src + 2]!);
    const normal = sanitizeNeuralNormal(normals[src]!, normals[src + 1]!, normals[src + 2]!);
    out[dst + 6] = normal[0];
    out[dst + 7] = normal[1];
    out[dst + 8] = normal[2];
  }
  return out;
}
