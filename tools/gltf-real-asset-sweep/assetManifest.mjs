// @ts-check
// Pure metadata for public real glTF assets used by sweep/proof lanes.

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
