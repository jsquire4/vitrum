// @ts-check
// Pure metadata for public real glTF assets used by sweep/proof lanes.

/**
 * Shared golden-comparison thresholds for real-glTF capture harnesses.
 * Single source of truth so the browser pt-webgl2 capture harness and any other
 * capture lane stop re-hardcoding `{ maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 }`.
 * NOTE: checker files (check-status.mjs / check-proofs.mjs / check-dzn-status.mjs
 * / check-validation-queue.mjs) keep their OWN independent pinned thresholds by
 * design — this const is consumed by HARNESSES only.
 */
export const REAL_GLTF_GOLDEN_THRESHOLDS = Object.freeze({ maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 });

export const REAL_GLTF_ASSETS = [
  {
    id: "box-textured-glb",
    kind: "textured-glb",
    url: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/BoxTextured/glTF-Binary/BoxTextured.glb",
    expect: { minPrimitives: 1, minTextures: 1 },
  },
  {
    id: "cesium-milk-truck-draco",
    kind: "draco",
    url: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMilkTruck/glTF-Draco/CesiumMilkTruck.gltf",
    expect: { minPrimitives: 1, requiredExtensions: ["KHR_draco_mesh_compression"] },
  },
  {
    id: "meshopt-cube-real",
    kind: "meshopt",
    url: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/MeshoptCubeTest/glTF-Meshopt/MeshoptCubeTest.gltf",
    expect: {
      minPrimitives: 1,
      requiredExtensions: ["KHR_meshopt_compression"],
      allowedWarningSubstrings: ["sets doubleSided=true"],
    },
  },
];

/** @param {string} id */
export function getRealGltfAsset(id) {
  const asset = REAL_GLTF_ASSETS.find((candidate) => candidate.id === id);
  if (!asset) {
    throw new Error(`unknown real glTF asset "${id}"`);
  }
  return asset;
}
