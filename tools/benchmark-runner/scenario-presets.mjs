/**
 * Deterministic metadata for gap-closure scenarios (host capture pages may read query params).
 * Mirrors plan/gap-closure-acceptance-matrix.md.
 */
export const GAP_CLOSURE_SCENARIOS = [
  {
    scenarioId: 'rfe03-layered-front-back',
    seed: 1337,
    resolution: '1280x720',
    bounces: 8,
    spp: 512,
  },
  {
    scenarioId: 'rfe07-11-sss-mixed-panels',
    seed: 2027,
    resolution: '1280x720',
    bounces: 8,
    spp: 512,
  },
  {
    scenarioId: 'rfe08-13-spectral-payload',
    seed: 4242,
    resolution: '1280x720',
    bounces: 10,
    spp: 1024,
  },
  {
    scenarioId: 'rfe14-thinfilm-angle-shift',
    seed: 9001,
    resolution: '1280x720',
    bounces: 10,
    spp: 1024,
  },
  {
    scenarioId: 'rfe09-bridge-global-cmf',
    seed: 31415,
    resolution: '1024x1024',
    bounces: 8,
    spp: 256,
  },
  {
    scenarioId: 'rfe05-caustic-strategy',
    seed: 27182,
    resolution: '1280x720',
    bounces: 10,
    spp: 1024,
    causticVariants: ['none', 'manifold-nee', 'photon-map'],
  },
  {
    scenarioId: 'ptwgpu-parity-material-fields',
    seed: 777,
    resolution: '1280x720',
    bounces: 8,
    spp: 512,
  },
];
