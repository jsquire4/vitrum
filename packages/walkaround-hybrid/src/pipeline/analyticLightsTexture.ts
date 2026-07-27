import {
  assertBvhTextureFitsDevice,
  assertBvhTextureRefreshCapacity,
  uploadElementTexture,
  writeElementTexture,
} from './bvhTextureLimits.js';

const ANALYTIC_LIGHT_TEX_WIDTH = 4096;
const ANALYTIC_LIGHT_HEADER_VEC4S = 1;
const ANALYTIC_LIGHT_STRIDE_VEC4S = 4;
const ANALYTIC_LIGHT_ALIAS_STRIDE_VEC4S = 1;

export interface AnalyticLightsTexture {
  texture: GPUTexture;
  width: number;
  height: number;
}

function assertAnalyticLightsPayload(
  packed: Float32Array,
  lightCount: number,
): number {
  if (!Number.isSafeInteger(lightCount) || lightCount < 0) {
    throw new RangeError(
      `analyticLights lightCount must be a non-negative safe integer; received ${String(lightCount)}`,
    );
  }
  const payloadFloats = lightCount *
    (ANALYTIC_LIGHT_STRIDE_VEC4S + ANALYTIC_LIGHT_ALIAS_STRIDE_VEC4S) * 4;
  if (!Number.isSafeInteger(payloadFloats)) {
    throw new RangeError(`analyticLights payload size is not a safe integer for ${lightCount} lights`);
  }
  if (packed.length !== payloadFloats) {
    throw new RangeError(
      `analyticLights payload length must be exactly ${payloadFloats} floats for ${lightCount} lights; received ${packed.length}`,
    );
  }
  for (let i = 0; i < packed.length; i++) {
    if (!Number.isFinite(packed[i])) {
      throw new TypeError(`analyticLights payload[${i}] must be finite`);
    }
  }
  return payloadFloats;
}

function analyticLightsTextureSize(vec4Count: number): { width: number; height: number } {
  const count = Math.max(4, Math.floor(vec4Count));
  const width = Math.min(ANALYTIC_LIGHT_TEX_WIDTH, count);
  const height = Math.ceil(count / width);
  return { width, height };
}

export function uploadAnalyticLightsTexture(
  device: GPUDevice,
  packed: Float32Array,
  lightCount: number,
): AnalyticLightsTexture {
  assertAnalyticLightsPayload(packed, lightCount);
  const count = lightCount;
  const vec4Count = Math.max(
    4,
    ANALYTIC_LIGHT_HEADER_VEC4S +
      count * (ANALYTIC_LIGHT_STRIDE_VEC4S + ANALYTIC_LIGHT_ALIAS_STRIDE_VEC4S),
  );
  const { width, height } = analyticLightsTextureSize(vec4Count);
  assertBvhTextureFitsDevice('analyticLights', device, width, height, vec4Count);
  const texture = uploadElementTexture(device, {
    label: 'vitrum.analyticLights.rgba32float',
    format: 'rgba32float',
    width,
    height,
    bytesPerTexel: 16,
    elementsPerTexel: 4,
    makePadded: (n) => new Float32Array(n),
    fill: (padded) => fillAnalyticLights(padded, packed, count),
  });
  return { texture, width, height };
}

export function refreshAnalyticLightsTexture(
  device: GPUDevice,
  tex: AnalyticLightsTexture,
  packed: Float32Array,
  lightCount: number,
): void {
  assertAnalyticLightsPayload(packed, lightCount);
  const count = lightCount;
  assertBvhTextureRefreshCapacity(
    'analyticLights',
    tex.width,
    tex.height,
    Math.max(
      4,
      ANALYTIC_LIGHT_HEADER_VEC4S +
        count * (ANALYTIC_LIGHT_STRIDE_VEC4S + ANALYTIC_LIGHT_ALIAS_STRIDE_VEC4S),
    ),
  );
  writeElementTexture(device, tex.texture, {
    width: tex.width,
    height: tex.height,
    bytesPerTexel: 16,
    elementsPerTexel: 4,
    makePadded: (n) => new Float32Array(n),
    fill: (padded) => fillAnalyticLights(padded, packed, count),
  });
}

/** Populate header (count + alias offset), payload records, then aliases. */
function fillAnalyticLights(padded: Float32Array, packed: Float32Array, lightCount: number): void {
  padded[0] = lightCount;
  padded[1] = ANALYTIC_LIGHT_HEADER_VEC4S + lightCount * ANALYTIC_LIGHT_STRIDE_VEC4S;
  const payloadOffset = ANALYTIC_LIGHT_HEADER_VEC4S * 4;
  padded.set(packed, payloadOffset);
}
