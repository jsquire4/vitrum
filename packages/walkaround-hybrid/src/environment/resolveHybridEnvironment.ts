import type {
  EnvironmentMapRef,
  HdriEnvironment,
  SceneEnvironment,
} from '@vitrum/core';

export type HybridSkyVec3 = [number, number, number];

export type HybridEnvironmentResolveMode =
  | 'none'
  | 'hdri-intensity-only'
  | 'hdri-raw-average'
  | 'hdri-extension-resolver'
  | 'procedural-sky-approx';

export interface HybridResolvedEnvironment {
  readonly mode: HybridEnvironmentResolveMode;
  readonly skyTint?: HybridSkyVec3;
  readonly skyIrradiance?: number;
  readonly proceduralSunDirection?: HybridSkyVec3;
  readonly proceduralSunIntensity?: number;
  readonly warnings: readonly string[];
}

export interface HybridEnvironmentMapResolverResult {
  /**
   * Unit-intensity tint for the opaque environment map handle. The resolver
   * applies SceneEnvironment.intensity to skyIrradiance after this callback.
   */
  readonly skyTint?: readonly [number, number, number];
  /**
   * Unit-intensity scalar for the opaque environment map handle. Defaults to 1
   * when omitted, so SceneEnvironment.intensity still has an effect.
   */
  readonly skyIrradiance?: number;
  readonly warnings?: readonly string[];
}

export type HybridEnvironmentMapResolver = (
  hdri: EnvironmentMapRef,
  environment: HdriEnvironment,
) => HybridEnvironmentMapResolverResult | null | undefined;

export interface HybridEnvironmentResolverExtensionNamespace {
  readonly resolveEnvironmentMap?: HybridEnvironmentMapResolver;
}

export interface HybridEnvironmentResolverExtensions {
  readonly 'walkaround-hybrid'?: HybridEnvironmentResolverExtensionNamespace;
}

export interface ResolveHybridEnvironmentOptions {
  readonly extensions?: HybridEnvironmentResolverExtensions | null;
}

interface RawNumericHdriPayload {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number>;
  readonly stride: 3 | 4;
}

type RawPayloadRead =
  | { readonly kind: 'raw'; readonly payload: RawNumericHdriPayload }
  | { readonly kind: 'malformed'; readonly warning: string }
  | { readonly kind: 'opaque' };

const DEFAULT_SUN_DIRECTION: HybridSkyVec3 = [0, 1, 0];

export function resolveHybridEnvironment(
  environment: SceneEnvironment | null,
  options: ResolveHybridEnvironmentOptions = {},
): HybridResolvedEnvironment {
  const env = environment ?? { kind: 'none' };
  switch (env.kind) {
    case 'none':
      return { mode: 'none', skyIrradiance: 0, warnings: [] };
    case 'procedural-sky':
      return resolveProceduralSkyEnvironment(env);
    case 'hdri':
      return resolveHdriEnvironment(env, options);
  }
}

function resolveProceduralSkyEnvironment(
  env: Extract<SceneEnvironment, { kind: 'procedural-sky' }>,
): HybridResolvedEnvironment {
  const warnings: string[] = [
    'procedural-sky is approximated as diffuse sky scalars; turbidity, rayleigh, mieDirectionalG, and directional sky distribution are not sampled by walkaround-hybrid.',
  ];
  const skyIrradiance = finiteNonNegativeScalar(
    env.intensity,
    1,
    warnings,
    'procedural-sky intensity',
  );
  const sunDirection = normalizeVec3(
    env.sunDirection,
    DEFAULT_SUN_DIRECTION,
    warnings,
    'procedural-sky sunDirection',
  );
  const tintBoost = Math.max(0.2, 1 - Math.max(0, env.mieCoefficient) * 10);
  return {
    mode: 'procedural-sky-approx',
    skyTint: [0.9 * tintBoost, 0.95, 1],
    skyIrradiance,
    proceduralSunDirection: sunDirection,
    proceduralSunIntensity: skyIrradiance,
    warnings,
  };
}

function resolveHdriEnvironment(
  env: HdriEnvironment,
  options: ResolveHybridEnvironmentOptions,
): HybridResolvedEnvironment {
  const warnings: string[] = [];
  const intensity = finiteNonNegativeScalar(env.intensity, 1, warnings, 'HDRI intensity');
  const raw = readRawNumericHdriPayload(env.hdri);
  if (raw.kind === 'raw') {
    return resolveRawHdriAverage(raw.payload, intensity, warnings);
  }

  if (raw.kind === 'malformed') {
    warnings.push(raw.warning);
  }

  const resolver = options.extensions?.['walkaround-hybrid']?.resolveEnvironmentMap;
  if (resolver !== undefined) {
    return resolveHdriWithExtensionResolver(env, resolver, intensity, warnings);
  }

  warnings.push(
    'HDRI environment handle is opaque to walkaround-hybrid; applying intensity only and leaving skyTint unchanged.',
  );
  return {
    mode: 'hdri-intensity-only',
    skyIrradiance: intensity,
    warnings,
  };
}

function resolveHdriWithExtensionResolver(
  env: HdriEnvironment,
  resolver: HybridEnvironmentMapResolver,
  intensity: number,
  warnings: string[],
): HybridResolvedEnvironment {
  let resolved: HybridEnvironmentMapResolverResult | null | undefined;
  try {
    resolved = resolver(env.hdri, env);
  } catch (err) {
    warnings.push(
      `HDRI environment resolver threw (${errorMessage(err)}); applying intensity only.`,
    );
    return {
      mode: 'hdri-intensity-only',
      skyIrradiance: intensity,
      warnings,
    };
  }

  if (resolved == null) {
    warnings.push(
      'HDRI environment resolver returned no result; applying intensity only and leaving skyTint unchanged.',
    );
    return {
      mode: 'hdri-intensity-only',
      skyIrradiance: intensity,
      warnings,
    };
  }

  if (resolved.warnings !== undefined) {
    warnings.push(...resolved.warnings.map((warning) => String(warning)));
  }
  const result: {
    mode: 'hdri-extension-resolver';
    skyTint?: HybridSkyVec3;
    skyIrradiance: number;
    warnings: readonly string[];
  } = {
    mode: 'hdri-extension-resolver',
    skyIrradiance:
      finiteNonNegativeScalar(
        resolved.skyIrradiance,
        1,
        warnings,
        'HDRI resolver skyIrradiance',
      ) * intensity,
    warnings,
  };
  const tint = finiteVec3(resolved.skyTint, warnings, 'HDRI resolver skyTint');
  if (tint !== undefined) {
    result.skyTint = tint;
  }
  return result;
}

function resolveRawHdriAverage(
  payload: RawNumericHdriPayload,
  intensity: number,
  warnings: string[],
): HybridResolvedEnvironment {
  const sum: HybridSkyVec3 = [0, 0, 0];
  let weightSum = 0;
  let clampedSamples = 0;

  for (let y = 0; y < payload.height; y += 1) {
    const theta = ((y + 0.5) / payload.height) * Math.PI;
    const rowWeight = Math.sin(theta);
    for (let x = 0; x < payload.width; x += 1) {
      const sampleOffset = (y * payload.width + x) * payload.stride;
      const r = finiteRadianceSample(payload.data[sampleOffset]);
      const g = finiteRadianceSample(payload.data[sampleOffset + 1]);
      const b = finiteRadianceSample(payload.data[sampleOffset + 2]);
      if (r.clamped) clampedSamples += 1;
      if (g.clamped) clampedSamples += 1;
      if (b.clamped) clampedSamples += 1;
      sum[0] += r.value * rowWeight;
      sum[1] += g.value * rowWeight;
      sum[2] += b.value * rowWeight;
      weightSum += rowWeight;
    }
  }

  if (clampedSamples > 0) {
    warnings.push(
      `HDRI raw payload had ${clampedSamples} non-finite or negative radiance sample(s); clamped them to 0.`,
    );
  }
  warnings.push(
    'HDRI raw payload is reduced to a solid-angle-weighted average color; directional lighting and rotationY are not represented by walkaround-hybrid sky scalars.',
  );

  if (weightSum <= 0) {
    return {
      mode: 'hdri-raw-average',
      skyTint: whiteSkyTint(),
      skyIrradiance: 0,
      warnings,
    };
  }

  const average: HybridSkyVec3 = [
    sum[0] / weightSum,
    sum[1] / weightSum,
    sum[2] / weightSum,
  ];
  const maxChannel = Math.max(average[0], average[1], average[2], 0);
  if (maxChannel <= 1e-12) {
    return {
      mode: 'hdri-raw-average',
      skyTint: whiteSkyTint(),
      skyIrradiance: 0,
      warnings,
    };
  }

  return {
    mode: 'hdri-raw-average',
    skyTint: [
      average[0] / maxChannel,
      average[1] / maxChannel,
      average[2] / maxChannel,
    ],
    skyIrradiance: maxChannel * intensity,
    warnings,
  };
}

function readRawNumericHdriPayload(value: EnvironmentMapRef): RawPayloadRead {
  if (typeof value !== 'object' || value === null) {
    return { kind: 'opaque' };
  }
  const candidate = value as {
    readonly width?: unknown;
    readonly height?: unknown;
    readonly data?: unknown;
  };
  const looksRaw =
    candidate.width !== undefined ||
    candidate.height !== undefined ||
    candidate.data !== undefined;
  if (!looksRaw) {
    return { kind: 'opaque' };
  }

  const width = Number(candidate.width);
  const height = Number(candidate.height);
  const data = candidate.data as { readonly length?: unknown } | null | undefined;
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width <= 0 ||
    height <= 0 ||
    data == null ||
    typeof data.length !== 'number'
  ) {
    return {
      kind: 'malformed',
      warning:
        'HDRI raw payload must provide positive integer width/height and array-like numeric data; applying intensity only.',
    };
  }

  const pixelCount = width * height;
  if (!Number.isSafeInteger(pixelCount) || pixelCount <= 0) {
    return {
      kind: 'malformed',
      warning:
        'HDRI raw payload dimensions overflow a safe pixel count; applying intensity only.',
    };
  }

  const length = data.length;
  if (!Number.isInteger(length) || length < pixelCount * 3) {
    return {
      kind: 'malformed',
      warning:
        'HDRI raw payload data is shorter than width * height * 3; applying intensity only.',
    };
  }

  const exactStride = length / pixelCount;
  const stride: 3 | 4 = exactStride === 4 ? 4 : 3;
  return {
    kind: 'raw',
    payload: {
      width,
      height,
      data: candidate.data as ArrayLike<number>,
      stride,
    },
  };
}

function finiteNonNegativeScalar(
  value: number | undefined,
  fallback: number,
  warnings: string[],
  label: string,
): number {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value)) {
    warnings.push(`${label} is not finite; defaulting to ${fallback}.`);
    return fallback;
  }
  if (value < 0) {
    warnings.push(`${label} is negative; clamping to 0.`);
    return 0;
  }
  return value;
}

function finiteVec3(
  value: readonly [number, number, number] | undefined,
  warnings: string[],
  label: string,
): HybridSkyVec3 | undefined {
  if (value === undefined) return undefined;
  const x = Number(value[0]);
  const y = Number(value[1]);
  const z = Number(value[2]);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    warnings.push(`${label} must contain finite numbers; ignoring it.`);
    return undefined;
  }
  return [x, y, z];
}

function normalizeVec3(
  value: readonly [number, number, number],
  fallback: HybridSkyVec3,
  warnings: string[],
  label: string,
): HybridSkyVec3 {
  const x = Number(value[0]);
  const y = Number(value[1]);
  const z = Number(value[2]);
  const len = Math.hypot(x, y, z);
  if (!Number.isFinite(len) || len < 1e-8) {
    warnings.push(`${label} is zero-length or non-finite; defaulting to [0, 1, 0].`);
    return [...fallback];
  }
  return [x / len, y / len, z / len];
}

function finiteRadianceSample(value: number | undefined): { value: number; clamped: boolean } {
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return { value: 0, clamped: true };
  }
  return { value: n, clamped: false };
}

function whiteSkyTint(): HybridSkyVec3 {
  return [1, 1, 1];
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
