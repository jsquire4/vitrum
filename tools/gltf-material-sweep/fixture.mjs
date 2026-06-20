// @ts-check

export const SWEEP_MAPS = [
  "baseColorMap",
  "roughnessMap",
  "metallicMap",
  "normalMap",
  "aoMap",
  "emissiveMap",
  "transmissionMap",
  "specularIntensityMap",
  "specularColorMap",
  "sheenColorMap",
  "sheenRoughnessMap",
  "clearcoatMap",
  "clearcoatRoughnessMap",
  "clearcoatNormalMap",
  "iridescenceMap",
  "iridescenceThicknessMap",
  "anisotropyMap",
  "thicknessMap",
];

export const WALKAROUND_ATLAS_FIELDS = new Set([
  "baseColorMap",
  "normalMap",
  "roughnessMap",
  "metallicMap",
  "aoMap",
  "alphaMap",
  "emissiveMap",
  "transmissionMap",
  "thicknessMap",
  "lightMap",
  "specularColorMap",
  "specularIntensityMap",
  "clearcoatMap",
  "clearcoatRoughnessMap",
  "clearcoatNormalMap",
  "bumpMap",
  "sheenColorMap",
  "sheenRoughnessMap",
  "anisotropyMap",
  "iridescenceMap",
  "iridescenceThicknessMap",
]);

export const FIELD_TEXTURE_INDEX = new Map([
  ["baseColorMap", 0],
  ["roughnessMap", 1],
  ["metallicMap", 1],
  ["normalMap", 2],
  ["aoMap", 3],
  ["emissiveMap", 4],
  ["transmissionMap", 5],
  ["specularIntensityMap", 6],
  ["specularColorMap", 7],
  ["sheenColorMap", 8],
  ["sheenRoughnessMap", 9],
  ["clearcoatMap", 10],
  ["clearcoatRoughnessMap", 11],
  ["clearcoatNormalMap", 12],
  ["iridescenceMap", 13],
  ["iridescenceThicknessMap", 14],
  ["anisotropyMap", 15],
  ["thicknessMap", 16],
]);

function f32Buffer(values) {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

function concat(buffers) {
  const total = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const buffer of buffers) {
    out.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  }
  return out.buffer;
}

function texInfo(index) {
  return {
    index,
    texCoord: 0,
    extensions: {
      KHR_texture_transform: {
        texCoord: 1,
        offset: [0.01 * (index + 1), 0.02 * (index + 1)],
        scale: [1 + 0.1 * (index + 1), 2 + 0.1 * (index + 1)],
        rotation: 0.001 * (index + 1),
      },
    },
  };
}

export function expectedSamplerPolicy(textureIndex) {
  const minFilterModes = [
    { minFilter: "nearest", mipFilter: "none", usesMipmaps: false },
    { minFilter: "linear", mipFilter: "none", usesMipmaps: false },
    { minFilter: "nearest", mipFilter: "nearest", usesMipmaps: true },
    { minFilter: "linear", mipFilter: "nearest", usesMipmaps: true },
    { minFilter: "nearest", mipFilter: "linear", usesMipmaps: true },
    { minFilter: "linear", mipFilter: "linear", usesMipmaps: true },
  ];
  return {
    wrapS: textureIndex % 3 === 1 ? "clamp-to-edge" : textureIndex % 3 === 2 ? "mirrored-repeat" : "repeat",
    wrapT: textureIndex % 3 === 0 ? "repeat" : textureIndex % 3 === 1 ? "mirrored-repeat" : "clamp-to-edge",
    magFilter: textureIndex % 2 === 0 ? "nearest" : "linear",
    ...minFilterModes[textureIndex % minFilterModes.length],
  };
}

export function samplerPolicyIsNativeForBackend(backend, policy, field = undefined) {
  if (backend === "walkaround-hybrid") return false;
  if (backend === "pt-webgl2") {
    return policy.magFilter === "nearest" && policy.minFilter === "nearest" && policy.mipFilter === "none";
  }
  if (backend === "pt-webgpu" && field === "bumpMap") {
    return policy.magFilter === "linear" && policy.minFilter === "linear" && policy.mipFilter === "none";
  }
  return true;
}

export function makeSweepGltf() {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
  const buffer = concat([positions, imageBytes.buffer]);
  const imageOffset = positions.byteLength;
  const textureCount = 17;

  return {
    buffers: new Map([[0, buffer]]),
    gltf: {
      asset: { version: "2.0", generator: "vitrum-gltf-material-sweep" },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{
        pbrMetallicRoughness: {
          baseColorTexture: texInfo(0),
          metallicRoughnessTexture: texInfo(1),
        },
        normalTexture: { ...texInfo(2), scale: 0.5 },
        occlusionTexture: { ...texInfo(3), strength: 0.75 },
        emissiveFactor: [1, 1, 1],
        emissiveTexture: texInfo(4),
        extensions: {
          KHR_materials_transmission: {
            transmissionFactor: 0.8,
            transmissionTexture: texInfo(5),
          },
          KHR_materials_specular: {
            specularFactor: 0.5,
            specularTexture: texInfo(6),
            specularColorFactor: [0.9, 0.8, 0.7],
            specularColorTexture: texInfo(7),
          },
          KHR_materials_sheen: {
            sheenColorFactor: [0.5, 0.3, 0.1],
            sheenColorTexture: texInfo(8),
            sheenRoughnessFactor: 0.4,
            sheenRoughnessTexture: texInfo(9),
          },
          KHR_materials_clearcoat: {
            clearcoatFactor: 0.8,
            clearcoatTexture: texInfo(10),
            clearcoatRoughnessFactor: 0.1,
            clearcoatRoughnessTexture: texInfo(11),
            clearcoatNormalTexture: { ...texInfo(12), scale: 0.25 },
          },
          KHR_materials_iridescence: {
            iridescenceFactor: 0.7,
            iridescenceTexture: texInfo(13),
            iridescenceIor: 2.0,
            iridescenceThicknessMinimum: 200,
            iridescenceThicknessMaximum: 800,
            iridescenceThicknessTexture: texInfo(14),
          },
          KHR_materials_anisotropy: {
            anisotropyStrength: 0.6,
            anisotropyRotation: 1.0,
            anisotropyTexture: texInfo(15),
          },
          KHR_materials_volume: {
            thicknessFactor: 0.5,
            thicknessTexture: texInfo(16),
            attenuationDistance: 2.0,
          },
        },
      }],
      textures: Array.from({ length: textureCount }, (_, i) => ({ source: 0, sampler: i })),
      samplers: Array.from({ length: textureCount }, (_, i) => ({
        wrapS: i % 3 === 1 ? 33071 : i % 3 === 2 ? 33648 : undefined,
        wrapT: i % 3 === 0 ? undefined : i % 3 === 1 ? 33648 : 33071,
        magFilter: i % 2 === 0 ? 9728 : 9729,
        minFilter: [9728, 9729, 9984, 9985, 9986, 9987][i % 6],
      })),
      images: [{ bufferView: 1, mimeType: "image/png" }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3" }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        { buffer: 0, byteOffset: imageOffset, byteLength: imageBytes.byteLength },
      ],
      buffers: [{ byteLength: buffer.byteLength }],
    },
  };
}

export function makeSweepTextureDecodeHooks() {
  return {
    decodeImage: (bytes, mimeType) => ({ kind: "raw-image", mimeType, data: bytes }),
    decodePixels: (_handle, context) => ({
      width: 4,
      height: 4,
      channels: 4,
      dataType: "uint8",
      colorSpace: context.colorSpace,
      data: new Uint8Array(4 * 4 * 4).fill(255),
    }),
  };
}
