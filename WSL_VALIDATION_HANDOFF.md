# WSL validation handoff

The local Codex pass fixed a set of contract/runtime issues in `three-bindings`,
`walkaround-hybrid`, and `stained-glass-extensions`. Please validate from WSL
with a WSL-native Node/npm install. If `which npm` resolves under `/mnt/c/`,
fix PATH or install Node inside WSL first; Windows npm cannot run this repo
reliably from the WSL filesystem.

## Code fixes to validate

- `HybridEngine.updateLighting({ primaryLightDir })` now republishes DDGI sun
  lights with the updated direction, and constructor-provided DDGI sun lights
  are initially oriented from `primaryLightDir`.
- `vitrumSceneToThree` now restores `alphaMode`, `alphaCutoff`, and `opacity`
  onto `MeshPhysicalMaterial`.
- `sceneFromThreeJS` / `vitrumSceneToThree` now preserve light `castShadow`;
  directional light `angularDiameter` round-trips through
  `userData.vitrumLightAngularDiameter`.
- `convertMaterial` now accepts only the documented `SpectralCurve` shape for
  `userData.vitrumSpectralAttenuation`.
- Neural `conv2d` implicit padding now uses one shared rule for tensor dims and
  packed layer uniforms: 3x3 defaults to padding 1, other kernels default to 0.

## Mechanical validation

Run from `/home/jsquire4/projects/vitrum`:

```bash
npm run typecheck
npm test
```

Targeted tests for this patch:

```bash
npm run test --workspace @vitrum/walkaround-hybrid -- \
  __tests__/hybridEngineLighting.test.ts \
  __tests__/neural.test.ts

npm run test --workspace @vitrum/three-bindings -- \
  src/__tests__/materialTextureRef.test.ts \
  src/__tests__/material-vitrum-roundtrip.test.ts \
  src/__tests__/sceneFromThreeJS.test.ts \
  src/__tests__/vitrumSceneToThree.test.ts
```

## GPU / visual validation

Run the cheap GPU gates first and report whether they pass, fail, or skip:

```bash
tools/gpu-env/run-gpu-validation.sh
npm run validate:gpu:smoke
```

Then run the broader GPU validator if the smoke pass is green:

```bash
npm run validate:gpu
```

For RC behavior/acceptance, generate fresh metrics with the benchmark runner
and feed the produced JSON paths into the gated tests:

```bash
npm run benchmark:rc-acceptance-full

VITRUM_RC_ACCEPTANCE=1 \
VITRUM_RC_ACCEPTANCE_METRICS=<path-to-rc-acceptance-metrics.json> \
npm run test --workspace @vitrum/walkaround-hybrid -- __tests__/rcAcceptance.gpu.test.ts

VITRUM_RC_BEHAVIOR_ACCEPTANCE=1 \
VITRUM_RC_BEHAVIOR_METRICS=<path-to-rc-behavior-metrics.json> \
npm run test --workspace @vitrum/walkaround-rc -- __tests__/rcBehavior.gpu.test.ts
```

## Still needs GPU-backed judgment

Do not mark these closed from mechanical tests alone:

- DDGI probe white-bounce/albedo model: needs radiometric A/B before changing.
- `pt-webgpu` photon-map radius/strategy weighting: needs caustic gap-closure
  renders, not just unit tests.
- SVGF-real conservative object-id fallback: larger G-buffer/object-id work.
- NRC stale-record clearing: validate record buffer behavior before patching.
- Stained-glass normal-map shadow perturbation: needs a product decision on
  whether to gate, remove, or replace with a physically justified option.
- OIDN readback race/dimension guard: narrow lifecycle edge; verify under
  resize/dispose stress before patching.

Also note the pre-existing mode-only changes in several `scripts/` and `tools/`
files. They were present before this code pass and should not be mixed into a
semantic commit unless intentionally normalized.
