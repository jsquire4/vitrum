// @ts-check
// Lightweight metadata shared by the real glTF import sweep and behavioral PNG gate.

export const REAL_GLTF_BEHAVIORAL_PROOFS = [
  {
    assetId: "box-textured-glb",
    label: "pt/gltf-real-box-textured",
    goldenPath: "tools/reference-renders/gltf-real-behavioral/pt-gltf-real-box-textured.png",
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
  {
    assetId: "cesium-milk-truck-draco",
    label: "pt/gltf-real-draco",
    goldenPath: "tools/reference-renders/gltf-real-behavioral/pt-gltf-real-draco.png",
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
  {
    assetId: "meshopt-cube-real",
    label: "pt/gltf-real-meshopt",
    goldenPath: "tools/reference-renders/gltf-real-behavioral/pt-gltf-real-meshopt.png",
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
  },
];

/** @param {string} assetId */
export function proofForRealGltfAsset(assetId) {
  return REAL_GLTF_BEHAVIORAL_PROOFS.find((proof) => proof.assetId === assetId) ?? null;
}
