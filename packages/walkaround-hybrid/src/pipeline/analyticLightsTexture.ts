import {
  assertBvhTextureFitsDevice,
  assertBvhTextureRefreshCapacity,
} from './bvhTextureLimits.js';

const ANALYTIC_LIGHT_TEX_WIDTH = 4096;
const ANALYTIC_LIGHT_HEADER_VEC4S = 1;
const ANALYTIC_LIGHT_STRIDE_VEC4S = 4;
const TEX_BINDING = 0x04;
const COPY_DST = 0x02;

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
  const texture = device.createTexture({
    label: 'vitrum.analyticLights.rgba32float',
    size: { width, height, depthOrArrayLayers: 1 },
    format: 'rgba32float',
    usage: TEX_BINDING | COPY_DST,
  });
  writeAnalyticLightsTexture(device, texture, packed, count, width, height);
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
  writeAnalyticLightsTexture(device, tex.texture, packed, count, tex.width, tex.height);
}

function writeAnalyticLightsTexture(
  device: GPUDevice,
  texture: GPUTexture,
  packed: Float32Array,
  lightCount: number,
  width: number,
  height: number,
): void {
  const padded = new Float32Array(width * height * 4);
  padded[0] = lightCount;
  const payloadOffset = ANALYTIC_LIGHT_HEADER_VEC4S * 4;
  const payloadFloats = lightCount * ANALYTIC_LIGHT_STRIDE_VEC4S * 4;
  padded.set(
    packed.subarray(0, Math.min(packed.length, payloadFloats, padded.length - payloadOffset)),
    payloadOffset,
  );
  device.queue.writeTexture(
    { texture },
    padded.buffer,
    { bytesPerRow: width * 4 * 4, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );
}
