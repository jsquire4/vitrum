// @ts-check
// Lightweight metadata shared by the real glTF import sweep and behavioral PNG gate.

export const REAL_GLTF_BEHAVIORAL_PROOFS = [
  {
    assetId: "box-textured-glb",
    label: "pt/gltf-real-box-textured",
    goldenPath: "tools/reference-renders/gltf-real-behavioral/pt-gltf-real-box-textured.png",
    sha256: "003df0782f4597eca33ffc31e8a3c90bc244f8f95c441c26043d703e1c8f1640",
    width: 64,
    height: 64,
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
    variants: {
      "dzn-full": {
        goldenPath: "tools/reference-renders/gltf-real-behavioral-dzn-full/pt-gltf-real-box-textured.png",
        sha256: "12d7847173872ef5fbee082c74649001d918b0a426652fffb0e8e96c2a4e7cae",
        width: 64,
        height: 64,
        thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
      },
    },
  },
  {
    assetId: "cesium-milk-truck-draco",
    label: "pt/gltf-real-draco",
    goldenPath: "tools/reference-renders/gltf-real-behavioral/pt-gltf-real-draco.png",
    sha256: "47737004d72092fd8e275078cd5c5e2d9b909b834821a27c4c9cd58f320ab13b",
    width: 64,
    height: 64,
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
    variants: {
      "dzn-full": {
        goldenPath: "tools/reference-renders/gltf-real-behavioral-dzn-full/pt-gltf-real-draco.png",
        sha256: "f2c60013a3c3cf179b48069ae9060d9cbfbe49daf75ee11bb7d0c0db89e0a615",
        width: 64,
        height: 64,
        thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
      },
    },
  },
  {
    assetId: "meshopt-cube-real",
    label: "pt/gltf-real-meshopt",
    goldenPath: "tools/reference-renders/gltf-real-behavioral/pt-gltf-real-meshopt.png",
    sha256: "8b1df25a346b9d902661efe077be3e044406d9a1e5c259d501f5604aa537e4e5",
    width: 64,
    height: 64,
    thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
    variants: {
      "dzn-full": {
        goldenPath: "tools/reference-renders/gltf-real-behavioral-dzn-full/pt-gltf-real-meshopt.png",
        sha256: "fc3583dfcb27c644b5cd9bb9d0b89242a95a451273ce76f0af02dff576b97b70",
        width: 64,
        height: 64,
        thresholds: { maxRmse: 8.0, maxMeanAbs: 4.0, maxAbs: 48 },
      },
    },
  },
];

/** @param {string} assetId */
export function proofForRealGltfAsset(assetId) {
  return REAL_GLTF_BEHAVIORAL_PROOFS.find((proof) => proof.assetId === assetId) ?? null;
}
