import {
  assessNeuralCheckpointProductionReadiness,
  isNeuralCheckpointF16Compatible,
  resolveHybridNrcConfig,
  type HybridEngineOptions,
} from '@vitrum/walkaround-hybrid';

/**
 * Conditional features for an engine-owned adapter -> device request.
 *
 * `shader-f16` is deliberately not a global hybrid requirement: it is needed
 * only when the selected neural denoiser can consume a digest-bound,
 * mixed-precision-certified checkpoint or when the resolved NRC trainer
 * explicitly selects `useF16`. Externally supplied devices do not pass through
 * this function; their enabled feature set is checked by HybridEngine before
 * either subsystem allocates anything.
 */
export function requiredWalkaroundNeuralDeviceFeatures(
  adapter: Pick<GPUAdapter, 'features'>,
  options: Partial<HybridEngineOptions> | undefined,
): readonly GPUFeatureName[] {
  if (options == null) return [];
  const required = new Set<GPUFeatureName>();

  if (options.denoiser === 'neural' || options.denoiser === 'auto') {
    const adapterSupportsF16 = adapter.features.has('shader-f16');
    const weights = options.neuralWeights;
    if (assessNeuralCheckpointProductionReadiness(weights).productionReady) {
      const preference =
        options.extensions?.['walkaround-hybrid']?.neuralTensorStorage ?? 'auto';
      if (preference !== 'f32') {
        const certified = isNeuralCheckpointF16Compatible(weights);
        if (options.denoiser === 'neural' && preference === 'f16') {
          if (!certified) {
            throw new TypeError(
              "createEngine: denoiser:'neural' with neuralTensorStorage:'f16' requires " +
              'a digest-bound, passing mixed-precision checkpoint certificate',
            );
          }
          if (!adapterSupportsF16) {
            throw new TypeError(
              "createEngine: denoiser:'neural' with neuralTensorStorage:'f16' requires " +
              "an adapter supporting 'shader-f16'",
            );
          }
        }
        if (certified && adapterSupportsF16) required.add('shader-f16');
      }
    }
    // Invalid/missing weights deliberately preserve HybridEngine's canonical
    // checkpoint diagnostic instead of failing feature negotiation first.
  }

  if (options.nrcEnabled === true && resolveHybridNrcConfig(options).useF16) {
    const adapterSupportsF16 = adapter.features.has('shader-f16');
    if (!adapterSupportsF16) {
      throw new TypeError(
        "createEngine: nrcConfig.useF16=true requires " +
        "an adapter supporting 'shader-f16'",
      );
    }
    required.add('shader-f16');
  }

  return [...required];
}
