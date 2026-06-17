// @ts-check
// Lightweight metadata shared by the glTF topology behavioral PNG gate.

export const GLTF_TOPOLOGY_BEHAVIORAL_PROOFS = [
  {
    id: "point-line-fallback",
    kind: "vitrum-gltf-point-line-behavioral-goldens",
    label: "pt/gltf-point-line-fallback",
    fixture: "synthetic-points-lines-loop-strip",
    sourceModes: ["POINTS", "LINES", "LINE_LOOP", "LINE_STRIP"],
    proof: "fallback-generated-mesh",
    goldenPath: "tools/reference-renders/gltf-point-line-behavioral/pt-gltf-point-line-fallback.png",
    manifestPath: "tools/reference-renders/gltf-point-line-behavioral/manifest.json",
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
  {
    id: "triangle-strip-fan",
    kind: "vitrum-gltf-triangle-topology-behavioral-goldens",
    label: "pt/gltf-triangle-strip-fan",
    fixture: "synthetic-triangle-strip-fan",
    sourceModes: ["TRIANGLE_STRIP", "TRIANGLE_FAN"],
    proof: "adapter-generated-triangle-list",
    goldenPath: "tools/reference-renders/gltf-triangle-topology-behavioral/pt-gltf-triangle-strip-fan.png",
    manifestPath: "tools/reference-renders/gltf-triangle-topology-behavioral/manifest.json",
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
];

export function proofForGltfTopology(id) {
  return GLTF_TOPOLOGY_BEHAVIORAL_PROOFS.find((proof) => proof.id === id) ?? null;
}
