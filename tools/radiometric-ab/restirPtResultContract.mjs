// @ts-check

export const RESTIR_PT_RESULT_SCHEMA = 'vitrum.radiometric-ab.result.v1';
export const RESTIR_PT_RESULT_AB = 'restir-pt-reuse-on-vs-off';
export const RESTIR_PT_WEIGHT_MODE = 'shared-max-log-with-f32-output-storage';

/**
 * Build the exact capture metadata written by the ReSTIR-PT GPU producer.
 *
 * @param {{
 *   scene: string,
 *   traceTier: string,
 *   colorSpace: string,
 *   requireFullTier: boolean,
 *   requireRadiometricSignal: boolean,
 *   maxBounces: number,
 *   resolution: { W: number, H: number },
 *   roi: { x0: number, y0: number, x1: number, y1: number },
 *   meanFrames: number,
 *   varianceRuns: number,
 *   varianceFramesPerRun: number,
 *   effectiveMClamp: number,
 *   seeds: unknown,
 * }} input
 */
export function buildRestirPtCaptureConfig(input) {
  return {
    scene: input.scene,
    traceTier: input.traceTier,
    colorSpace: input.colorSpace,
    requireFullTier: input.requireFullTier,
    requireRadiometricSignal: input.requireRadiometricSignal,
    maxBounces: input.maxBounces,
    resolution: input.resolution,
    roi: input.roi,
    meanFrames: input.meanFrames,
    varianceRuns: input.varianceRuns,
    varianceFramesPerRun: input.varianceFramesPerRun,
    arms: {
      base: { oneEdgeReconnectionReuse: false },
      candidate: {
        oneEdgeReconnectionReuse: true,
        effectiveMClamp: input.effectiveMClamp,
      },
    },
    seeds: input.seeds,
  };
}

/**
 * Build the exact result shape written by `ab-restir-pt.mjs`.
 *
 * @param {{
 *   provenance: unknown,
 *   date: string,
 *   resolution: { W: number, H: number },
 *   roi: { x0: number, y0: number, x1: number, y1: number },
 *   meanFrames: number,
 *   varianceRuns: number,
 *   varianceFramesPerRun: number,
 *   captureConfig: unknown,
 *   deviceIdentity: unknown,
 *   base: unknown,
 *   rpt: unknown,
 *   globalRelErr: number,
 *   roiRelErr: number,
 *   varRatio: number,
 *   reservoirWeightStats: unknown,
 *   pairedSeedAnalysis: unknown,
 *   highFrameMeanAgreement: boolean,
 *   meanAgreement: boolean,
 *   varianceNotWorse: boolean,
 *   verdict: string,
 * }} input
 */
export function buildRestirPtResult(input) {
  return {
    schema: RESTIR_PT_RESULT_SCHEMA,
    provenance: input.provenance,
    ab: RESTIR_PT_RESULT_AB,
    date: input.date,
    resolution: input.resolution,
    roi: input.roi,
    meanFrames: input.meanFrames,
    varianceRuns: input.varianceRuns,
    varianceFramesPerRun: input.varianceFramesPerRun,
    captureConfig: input.captureConfig,
    deviceIdentity: input.deviceIdentity,
    base: input.base,
    rpt: input.rpt,
    globalRelErr: input.globalRelErr,
    roiRelErr: input.roiRelErr,
    varRatio: input.varRatio,
    weightMode: RESTIR_PT_WEIGHT_MODE,
    reservoirWeightStats: input.reservoirWeightStats,
    pairedSeedAnalysis: input.pairedSeedAnalysis,
    highFrameMeanAgreement: input.highFrameMeanAgreement,
    meanAgreement: input.meanAgreement,
    varianceNotWorse: input.varianceNotWorse,
    verdict: input.verdict,
  };
}
