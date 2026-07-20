import {
  assertBvhTextureFitsDevice,
  assertBvhTextureRefreshCapacity,
  uploadElementTexture,
  writeElementTexture,
} from './bvhTextureLimits.js';

const ANALYTIC_LIGHT_TEX_WIDTH = 4096;
const ANALYTIC_LIGHT_HEADER_VEC4S = 1;
const ANALYTIC_LIGHT_STRIDE_VEC4S = 4;

export interface AnalyticLightsTexture {
  texture: GPUTexture;
  width: number;
  height: number;
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
  const count = Math.max(0, Math.floor(lightCount));
  const vec4Count = Math.max(4, ANALYTIC_LIGHT_HEADER_VEC4S + count * ANALYTIC_LIGHT_STRIDE_VEC4S);
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
  const count = Math.max(0, Math.floor(lightCount));
  assertBvhTextureRefreshCapacity(
    'analyticLights',
    tex.width,
    tex.height,
    Math.max(4, ANALYTIC_LIGHT_HEADER_VEC4S + count * ANALYTIC_LIGHT_STRIDE_VEC4S),
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

/** Populate the padded rgba32float grid: light count in texel 0, packed
 *  point/spot payload starting at the header offset. */
function fillAnalyticLights(padded: Float32Array, packed: Float32Array, lightCount: number): void {
  padded[0] = lightCount;
  const payloadOffset = ANALYTIC_LIGHT_HEADER_VEC4S * 4;
  const payloadFloats = lightCount * ANALYTIC_LIGHT_STRIDE_VEC4S * 4;
  padded.set(
    packed.subarray(0, Math.min(packed.length, payloadFloats, padded.length - payloadOffset)),
    payloadOffset,
  );
}
