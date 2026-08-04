import type { SceneEnvironment } from '@vitrum/core';
import { buildRepresentedPmfF32 } from '@vitrum/shared-samplers';
import { LIGHT_PIXELS } from './lightsTexture.js';
import { TRI_LIGHT_PIXELS, type MeshAreaLightsData } from './meshAreaLights.js';
import type { EnvTextureData, LightsTextureData } from './sceneTextures.js';

function environmentProposalPower(
  env: EnvTextureData,
  authored: SceneEnvironment,
): number {
  if (env.map == null || env.conditional == null || !(env.totalSum > 0)) return 0;
  const scale = authored.kind === 'hdri' ? Math.fround(authored.intensity ?? 1) : 1;
  if (!Number.isFinite(scale) || scale < 0) {
    throw new RangeError(
      '@vitrum/pt-webgl2: environment proposal intensity must be finite and non-negative in Float32.',
    );
  }
  const power = env.totalSum * scale;
  if (!Number.isFinite(power) || power < 0) {
    throw new RangeError('@vitrum/pt-webgl2: environment proposal power overflowed.');
  }
  return power;
}

/**
 * Populate the spare RGBA32F lanes consumed by the BDPT global emitter draw.
 * The PMFs are exact multiples of 2^-24, matching both PCG and Sobol shader
 * variates, so every positive analytic/mesh/environment source is reachable
 * and the stored PDF is the proposal the shader actually samples.
 */
export function applyRepresentedEmitterProposalPmf(
  lights: LightsTextureData,
  meshLights: MeshAreaLightsData,
  env: EnvTextureData,
  authoredEnvironment: SceneEnvironment,
): void {
  const weights: number[] = [];
  for (let i = 0; i < lights.lightCount; i += 1) {
    weights.push(
      lights.proposalWeights?.[i] ??
        lights.data[i * LIGHT_PIXELS * 4 + 11] ??
        0,
    );
  }
  for (let i = 0; i < meshLights.triLightCount; i += 1) {
    weights.push(
      meshLights.proposalWeights?.[i] ??
        meshLights.data?.[i * TRI_LIGHT_PIXELS * 4 + 16] ??
        0,
    );
  }
  const envPower = environmentProposalPower(env, authoredEnvironment);
  const hasEnvironmentCandidate = env.map != null && env.conditional != null;
  if (hasEnvironmentCandidate) weights.push(envPower);

  const pmf = buildRepresentedPmfF32(weights);
  let cursor = 0;
  for (let i = 0; i < lights.lightCount; i += 1) {
    // s5.a is free for every analytic light kind.
    lights.data[i * LIGHT_PIXELS * 4 + 23] = pmf[cursor++] ?? 0;
  }
  for (let i = 0; i < meshLights.triLightCount; i += 1) {
    if (meshLights.data == null) {
      throw new Error('@vitrum/pt-webgl2: mesh proposal data is missing.');
    }
    // s4.b is free in the mesh-triangle record.
    meshLights.data[i * TRI_LIGHT_PIXELS * 4 + 18] = pmf[cursor++] ?? 0;
  }
  if (hasEnvironmentCandidate) {
    // The environment distribution texture uses .r/.g for conditional/marginal
    // CDFs; .b of texel zero carries the one global BDPT emitter PMF.
    env.conditional.data[2] = pmf[cursor++] ?? 0;
  }
  if (cursor !== pmf.length) {
    throw new Error('@vitrum/pt-webgl2: represented emitter proposal cursor mismatch.');
  }
}
