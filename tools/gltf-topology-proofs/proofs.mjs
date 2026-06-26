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
    sha256: "c3d467e4f3e004cc0d81158f942a4698e114b698177ef552b36e91528cdfaff0",
    width: 64,
    height: 64,
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
    sha256: "ead5aa951a2c846c51ec231cc0f21fcc91ca2adbb3efee7b7648b84ce80a770d",
    width: 64,
    height: 64,
    manifestPath: "tools/reference-renders/gltf-triangle-topology-behavioral/manifest.json",
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
];

export function proofForGltfTopology(id) {
  return GLTF_TOPOLOGY_BEHAVIORAL_PROOFS.find((proof) => proof.id === id) ?? null;
}
