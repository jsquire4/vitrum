// @ts-check
// Lightweight metadata shared by the synthetic glTF material sweep and behavioral PNG gate.

export const GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF = {
  fixture: "synthetic-material-sweep",
  label: "pt/gltf-material-sweep",
  goldenPath: "tools/reference-renders/gltf-material-sweep-behavioral/pt-gltf-material-sweep.png",
  thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
};

export function proofForMaterialSweepFixture(fixture) {
  return fixture === GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF.fixture
    ? GLTF_MATERIAL_SWEEP_BEHAVIORAL_PROOF
    : null;
}
