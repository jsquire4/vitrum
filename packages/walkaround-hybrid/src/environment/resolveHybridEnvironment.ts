import type {
  EnvironmentMapRef,
  HdriEnvironment,
  SceneEnvironment,
} from '@vitrum/core';
import { bakePreethamSkyEquirect } from '@vitrum/shared-samplers';
import { buildDirectionalEnv, type DirectionalEnvData } from './equirectDirectional.js';

type HybridSkyVec3 = [number, number, number];

type HybridEnvironmentResolveMode =
  | 'none'
  | 'hdri-intensity-only'
  | 'hdri-raw-average'
  | 'hdri-extension-resolver'
  | 'procedural-sky-baked';

export interface HybridResolvedEnvironment {
  readonly mode: HybridEnvironmentResolveMode;
  readonly skyTint?: HybridSkyVec3;
  readonly skyIrradiance?: number;
  readonly proceduralSunDirection?: HybridSkyVec3;
  readonly proceduralSunIntensity?: number;
  readonly warnings: readonly string[];
  /**
   * B3 (road-to-100) — directional IBL payload. Present ONLY when a raw
   * pixel-backed HDRI was supplied (the `hdri-raw-average` mode). The host
   * uploads {@link DirectionalEnvData} as scene-group textures so the WGSL
   * sky-miss / GI-escape paths sample the ACTUAL map by direction; the
   * `skyTint`/`skyIrradiance` scalars above remain the fallback for backends/
   * scenes without directional data (the existing contract — unchanged). Absent
   * for opaque handles and all-black maps. Procedural sky fills this with the
   * shared finite Preetham equirect bake.
   */
  readonly directional?: DirectionalEnvData;
  /** HDRI Y-axis rotation in radians (H6 convention). 0 when not an HDRI. */
  readonly rotationY?: number;
  /**
   * Unit-intensity radiance multiplier for the directional map (the
   * SceneEnvironment.intensity). The WGSL applies it at sample time so the
   * uploaded `map` texels stay unit-intensity. Present with `directional`.
   */
  readonly directionalIntensity?: number;
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
  /**
   * Optional CPU-readable equirect payload for the opaque HDRI handle. When this
   * is a raw `{ width, height, data }` RGB/RGBA payload, walkaround builds the
   * same directional IBL map + importance CDFs used for native raw HDRI handles.
   * `skyTint` / `skyIrradiance` remain optional scalar fallback overrides; when
   * omitted, they are derived from this payload's solid-angle-weighted average.
   */
  readonly rawHdri?: EnvironmentMapRef;
  readonly warnings?: readonly string[];
}

export type HybridEnvironmentMapResolver = (
  hdri: EnvironmentMapRef,
  environment: HdriEnvironment,
) => HybridEnvironmentMapResolverResult | null | undefined;

interface HybridEnvironmentResolverExtensionNamespace {
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
  const warnings: string[] = [];
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
  const baked = bakePreethamSkyEquirect({
    sunDirection,
    turbidity: finiteScalarInRange(env.turbidity, 2, 1.5, 30, warnings, 'procedural-sky turbidity'),
    rayleigh: finiteScalarInRange(env.rayleigh, 1, 0, Number.POSITIVE_INFINITY, warnings, 'procedural-sky rayleigh'),
    mieCoefficient: finiteScalarInRange(env.mieCoefficient, 0.005, 0, Number.POSITIVE_INFINITY, warnings, 'procedural-sky mieCoefficient'),
    mieDirectionalG: finiteScalarInRange(env.mieDirectionalG, 0.8, -0.9999, 0.9999, warnings, 'procedural-sky mieDirectionalG'),
    intensity: skyIrradiance,
  });
  const payload: RawNumericHdriPayload = {
    width: baked.width,
    height: baked.height,
    data: baked.texels,
    stride: 4,
  };
  const average = averageRawRadiancePayload(payload, 1, warnings, 'procedural-sky bake');
  const directional = buildDirectionalEnv(payload) ?? undefined;
  return {
    mode: 'procedural-sky-baked',
    skyTint: average.skyTint,
    skyIrradiance: average.skyIrradiance,
    proceduralSunDirection: baked.sunDirection as HybridSkyVec3,
    // The baked equirect already contains the sun disk/corona and is sampled
    // through the env CDF, so do not request a second scalar procedural sun.
    proceduralSunIntensity: 0,
    warnings,
    ...(directional !== undefined
      ? { directional, rotationY: 0, directionalIntensity: 1 }
      : {}),
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
    const rotationY = finiteRotationY(env.rotationY, warnings);
    return resolveRawHdriAverage(raw.payload, intensity, rotationY, warnings);
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
  const rawResolved = resolveResolverRawHdri(resolved.rawHdri, intensity, env.rotationY, warnings);
  const tint = finiteVec3(resolved.skyTint, warnings, 'HDRI resolver skyTint') ?? rawResolved?.skyTint;
  return {
    mode: 'hdri-extension-resolver',
    ...(tint !== undefined ? { skyTint: tint } : {}),
    skyIrradiance:
      resolved.skyIrradiance !== undefined
        ? finiteNonNegativeScalar(
          resolved.skyIrradiance,
          1,
          warnings,
          'HDRI resolver skyIrradiance',
        ) * intensity
        : rawResolved?.skyIrradiance ?? intensity,
    warnings,
    ...(rawResolved?.directional !== undefined
      ? {
        directional: rawResolved.directional,
        rotationY: rawResolved.rotationY,
        directionalIntensity: rawResolved.directionalIntensity,
      }
      : {}),
  };
}

function resolveResolverRawHdri(
  rawHdri: EnvironmentMapRef | undefined,
  intensity: number,
  rotationYInput: number | undefined,
  warnings: string[],
): Pick<
  HybridResolvedEnvironment,
  'skyTint' | 'skyIrradiance' | 'directional' | 'rotationY' | 'directionalIntensity'
> | null {
  if (rawHdri === undefined) return null;
  const raw = readRawNumericHdriPayload(rawHdri);
  if (raw.kind === 'raw') {
    const rotationY = finiteRotationY(rotationYInput, warnings);
    const resolved = resolveRawHdriAverage(raw.payload, intensity, rotationY, warnings);
    return {
      ...(resolved.skyTint !== undefined ? { skyTint: resolved.skyTint } : {}),
      ...(resolved.skyIrradiance !== undefined ? { skyIrradiance: resolved.skyIrradiance } : {}),
      ...(resolved.directional !== undefined
        ? {
          directional: resolved.directional,
          rotationY: resolved.rotationY ?? 0,
          directionalIntensity: resolved.directionalIntensity ?? intensity,
        }
        : {}),
    };
  }
  if (raw.kind === 'malformed') {
    warnings.push(`HDRI resolver rawHdri was malformed: ${raw.warning}`);
  } else {
    warnings.push(
      'HDRI resolver rawHdri was opaque; using scalar resolver output only.',
    );
  }
  return null;
}

function resolveRawHdriAverage(
  payload: RawNumericHdriPayload,
  intensity: number,
  rotationY: number,
  warnings: string[],
): HybridResolvedEnvironment {
  // B3 — directional IBL payload (PBRT 2D distribution). Built from the same raw
  // pixels; additive to the scalar skyTint/skyIrradiance below (which stay the
  // fallback). Null for an all-black map (the scalar path then yields 0 irradiance).
  const directional = buildDirectionalEnv(payload) ?? undefined;
  const average = averageRawRadiancePayload(payload, intensity, warnings, 'HDRI raw payload');
  if (directional !== undefined) {
    warnings.push(
      'HDRI raw payload resolved to a directional IBL map (equirect + importance CDFs); the skyTint/skyIrradiance scalars below are the no-directional fallback only.',
    );
  } else {
    warnings.push(
      'HDRI raw payload is reduced to a solid-angle-weighted average color (no directional data — all-black or degenerate map); directional lighting and rotationY are not represented by walkaround-hybrid sky scalars.',
    );
  }

  return {
    mode: 'hdri-raw-average',
    skyTint: average.skyTint,
    skyIrradiance: average.skyIrradiance,
    warnings,
    ...(directional !== undefined
      ? { directional, rotationY, directionalIntensity: intensity }
      : {}),
  };
}

function finiteRotationY(value: number | undefined, warnings: string[]): number {
  if (value === undefined) return 0;
  if (!Number.isFinite(value)) {
    warnings.push('HDRI rotationY is not finite; defaulting to 0.');
    return 0;
  }
  return value;
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

function finiteScalarInRange(
  value: number | undefined,
  fallback: number,
  min: number,
  max: number,
  warnings: string[],
  label: string,
): number {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value)) {
    warnings.push(`${label} is not finite; defaulting to ${fallback}.`);
    return fallback;
  }
  if (value < min) {
    warnings.push(`${label} is below ${min}; clamping to ${min}.`);
    return min;
  }
  if (value > max) {
    warnings.push(`${label} is above ${max}; clamping to ${max}.`);
    return max;
  }
  return value;
}

function averageRawRadiancePayload(
  payload: RawNumericHdriPayload,
  intensity: number,
  warnings: string[],
  label: string,
): { skyTint: HybridSkyVec3; skyIrradiance: number } {
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
      `${label} had ${clampedSamples} non-finite or negative radiance sample(s); clamped them to 0.`,
    );
  }
  if (weightSum <= 0) {
    return { skyTint: whiteSkyTint(), skyIrradiance: 0 };
  }

  const average: HybridSkyVec3 = [
    sum[0] / weightSum,
    sum[1] / weightSum,
    sum[2] / weightSum,
  ];
  const maxChannel = Math.max(average[0], average[1], average[2], 0);
  if (maxChannel <= 1e-12) {
    return { skyTint: whiteSkyTint(), skyIrradiance: 0 };
  }
  return {
    skyTint: [
      average[0] / maxChannel,
      average[1] / maxChannel,
      average[2] / maxChannel,
    ],
    skyIrradiance: maxChannel * intensity,
  };
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
