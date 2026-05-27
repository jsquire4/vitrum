/**
 * pt-webgpu gap-closure scenario → capture URL / engine options.
 */

export const PT_WEBGPU_GAP_SCENARIOS = [
  'ptwgpu-parity-material-fields',
  'rfe03-layered-front-back',
  'rfe05-caustic-strategy',
  'rfe07-11-sss-mixed-panels',
  'rfe08-13-spectral-payload',
  'rfe14-thinfilm-angle-shift',
  'rfe09-bridge-global-cmf',
];

/** Scenario IDs with committed baselines under tools/reference-renders/baseline/. */
export const PT_WEBGPU_BASELINE_SCENARIOS = [
  'ptwgpu-parity-material-fields',
  'rfe03-layered-front-back',
  'rfe05-caustic-strategy',
  'rfe07-11-sss-mixed-panels',
  'rfe08-13-spectral-payload',
  'rfe14-thinfilm-angle-shift',
  'rfe09-bridge-global-cmf',
];

export function captureQueryForScenario(scenario) {
  const params = new URLSearchParams();
  params.set('mode', 'ptwebgpu');
  params.set('vitrumGapScenario', scenario.scenarioId);
  params.set('vitrumSeed', String(scenario.seed));
  if (scenario.extensions?.['vitrum.ptWebgpu.spectralHeroWavelength']) {
    params.set('vitrumPtWebgpuSpectral', '1');
  }
  return params;
}
