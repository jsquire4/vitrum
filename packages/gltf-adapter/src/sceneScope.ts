import type { GltfJson, GltfMaterial, GltfPrimitive } from './gltfTypes.js';
import {
  GLTF_TEXTURE_SOURCE_EXTENSIONS,
  type GltfTextureSourceExtension,
} from './textures.js';
import { resolveGltfMaterialAnimationPointer } from './materialPointerAnimation.js';

export interface GltfSceneReachability {
  readonly sceneIndex: number;
  readonly nodeIndices: ReadonlySet<number>;
  readonly meshIndices: ReadonlySet<number>;
  readonly primitiveKeys: ReadonlySet<string>;
  readonly materialIndices: ReadonlySet<number>;
  readonly textureIndices: ReadonlySet<number>;
  readonly imageIndices: ReadonlySet<number>;
  readonly bufferViewIndices: ReadonlySet<number>;
  readonly bufferIndices: ReadonlySet<number>;
  readonly skinIndices: ReadonlySet<number>;
  readonly cameraIndices: ReadonlySet<number>;
  readonly punctualLightIndices: ReadonlySet<number>;
}

export function gltfPrimitiveKey(meshIndex: number, primitiveIndex: number): string {
  return `${meshIndex}:${primitiveIndex}`;
}

export function collectPrimitiveMaterialIndices(primitive: GltfPrimitive): readonly number[] {
  const indices = new Set<number>();
  if (primitive.material !== undefined) indices.add(primitive.material);
  for (const mapping of primitive.extensions?.KHR_materials_variants?.mappings ?? []) {
    if (Number.isInteger(mapping.material) && mapping.material >= 0) {
      indices.add(mapping.material);
    }
  }
  return [...indices].sort((a, b) => a - b);
}

export function collectGltfSceneReachability(
  gltf: GltfJson,
  sceneIndex: number,
  textureSourceExtensions: readonly GltfTextureSourceExtension[] = [],
): GltfSceneReachability {
  const nodeIndices = new Set<number>();
  const meshIndices = new Set<number>();
  const primitiveKeys = new Set<string>();
  const materialIndices = new Set<number>();
  const textureIndices = new Set<number>();
  const imageIndices = new Set<number>();
  const bufferViewIndices = new Set<number>();
  const bufferIndices = new Set<number>();
  const skinIndices = new Set<number>();
  const cameraIndices = new Set<number>();
  const punctualLightIndices = new Set<number>();
  const scene = gltf.scenes?.[sceneIndex];

  const visitNode = (nodeIndex: number): void => {
    if (!Number.isInteger(nodeIndex) || nodeIndex < 0) return;
    if (nodeIndices.has(nodeIndex)) return;
    const node = gltf.nodes?.[nodeIndex];
    if (node == null) return;
    nodeIndices.add(nodeIndex);

    if (node.mesh !== undefined) {
      meshIndices.add(node.mesh);
      const mesh = gltf.meshes?.[node.mesh];
      for (const [primitiveIndex, primitive] of (mesh?.primitives ?? []).entries()) {
        primitiveKeys.add(gltfPrimitiveKey(node.mesh, primitiveIndex));
        for (const materialIndex of collectPrimitiveMaterialIndices(primitive)) {
          materialIndices.add(materialIndex);
        }
        collectPrimitiveBufferViews(gltf, primitive, bufferViewIndices);
      }
    }
    collectInstancingBufferViews(gltf, node, bufferViewIndices);
    if (node.skin !== undefined) {
      skinIndices.add(node.skin);
      const inverseBindMatrices = gltf.skins?.[node.skin]?.inverseBindMatrices;
      collectAccessorBufferViews(gltf, inverseBindMatrices, bufferViewIndices);
    }
    if (node.camera !== undefined) cameraIndices.add(node.camera);
    const lightRef = node.extensions?.KHR_lights_punctual;
    if (
      lightRef != null &&
      typeof lightRef === 'object' &&
      !Array.isArray(lightRef) &&
      typeof (lightRef as { readonly light?: unknown }).light === 'number'
    ) {
      punctualLightIndices.add((lightRef as { readonly light: number }).light);
    }
    for (const child of node.children ?? []) visitNode(child);
  };

  for (const root of scene?.nodes ?? []) visitNode(root);
  for (const animation of gltf.animations ?? []) {
    for (const channel of animation.channels ?? []) {
      if (channel.target.path === 'pointer') {
        const pointer = channel.target.extensions?.KHR_animation_pointer?.pointer;
        const pointerTarget = resolveGltfMaterialAnimationPointer(pointer);
        if (pointerTarget === undefined || !materialIndices.has(pointerTarget.materialIndex)) continue;
        const sampler = animation.samplers?.[channel.sampler];
        collectAccessorBufferViews(gltf, sampler?.input, bufferViewIndices);
        collectAccessorBufferViews(gltf, sampler?.output, bufferViewIndices);
        continue;
      }
      const targetNode = channel.target.node;
      if (targetNode === undefined || !nodeIndices.has(targetNode)) continue;
      const sampler = animation.samplers?.[channel.sampler];
      collectAccessorBufferViews(gltf, sampler?.input, bufferViewIndices);
      collectAccessorBufferViews(gltf, sampler?.output, bufferViewIndices);
    }
  }

  for (const materialIndex of materialIndices) {
    collectMaterialTextureIndices(gltf.materials?.[materialIndex], textureIndices);
  }

  const enabledTextureSourceExtensions = new Set(textureSourceExtensions);
  for (const textureIndex of textureIndices) {
    const imageIndex = selectedTextureImageIndex(
      gltf,
      textureIndex,
      enabledTextureSourceExtensions,
    );
    if (imageIndex !== undefined) imageIndices.add(imageIndex);
  }

  for (const imageIndex of imageIndices) {
    const bufferView = gltf.images?.[imageIndex]?.bufferView;
    if (bufferView !== undefined) bufferViewIndices.add(bufferView);
  }

  for (const bufferViewIndex of bufferViewIndices) {
    const bufferView = gltf.bufferViews?.[bufferViewIndex];
    const bufferIndex = bufferView?.buffer;
    if (bufferIndex !== undefined) bufferIndices.add(bufferIndex);
    const meshoptBufferIndex = meshoptCompressedBufferIndex(bufferView?.extensions);
    if (meshoptBufferIndex !== undefined) bufferIndices.add(meshoptBufferIndex);
  }

  return {
    sceneIndex,
    nodeIndices,
    meshIndices,
    primitiveKeys,
    materialIndices,
    textureIndices,
    imageIndices,
    bufferViewIndices,
    bufferIndices,
    skinIndices,
    cameraIndices,
    punctualLightIndices,
  };
}

function collectPrimitiveBufferViews(
  gltf: GltfJson,
  primitive: GltfPrimitive,
  out: Set<number>,
): void {
  for (const accessorIndex of Object.values(primitive.attributes ?? {})) {
    collectAccessorBufferViews(gltf, accessorIndex, out);
  }
  collectAccessorBufferViews(gltf, primitive.indices, out);
  for (const target of primitive.targets ?? []) {
    for (const accessorIndex of Object.values(target)) {
      collectAccessorBufferViews(gltf, accessorIndex, out);
    }
  }
  const draco = primitive.extensions?.KHR_draco_mesh_compression;
  if (isRecord(draco) && typeof draco.bufferView === 'number') out.add(draco.bufferView);
}

function collectAccessorBufferViews(
  gltf: GltfJson,
  accessorIndex: number | undefined,
  out: Set<number>,
): void {
  if (accessorIndex === undefined) return;
  const accessor = gltf.accessors?.[accessorIndex];
  const bufferView = accessor?.bufferView;
  if (bufferView !== undefined) out.add(bufferView);
  if (accessor?.sparse?.indices.bufferView !== undefined) out.add(accessor.sparse.indices.bufferView);
  if (accessor?.sparse?.values.bufferView !== undefined) out.add(accessor.sparse.values.bufferView);
}

function meshoptCompressedBufferIndex(extensions: Record<string, unknown> | undefined): number | undefined {
  const ext = extensions?.EXT_meshopt_compression ?? extensions?.KHR_meshopt_compression;
  if (!isRecord(ext) || typeof ext.buffer !== 'number') return undefined;
  return ext.buffer;
}

function collectInstancingBufferViews(
  gltf: GltfJson,
  node: NonNullable<GltfJson['nodes']>[number],
  out: Set<number>,
): void {
  const attributes = node.extensions?.EXT_mesh_gpu_instancing?.attributes;
  if (attributes == null) return;
  for (const accessorIndex of Object.values(attributes)) {
    collectAccessorBufferViews(gltf, accessorIndex, out);
  }
}

function collectMaterialTextureIndices(
  material: GltfMaterial | undefined,
  out: Set<number>,
): void {
  visitMaterialValue(material, out);
}

function visitMaterialValue(value: unknown, out: Set<number>): void {
  if (Array.isArray(value)) {
    for (const item of value) visitMaterialValue(item, out);
    return;
  }
  if (!isRecord(value)) return;
  if (typeof value.index === 'number' && Number.isInteger(value.index) && value.index >= 0) {
    out.add(value.index);
  }
  for (const child of Object.values(value)) visitMaterialValue(child, out);
}

function selectedTextureImageIndex(
  gltf: GltfJson,
  textureIndex: number,
  enabledExtensions: ReadonlySet<GltfTextureSourceExtension>,
): number | undefined {
  const texture = gltf.textures?.[textureIndex];
  if (texture == null) return undefined;
  for (const extName of GLTF_TEXTURE_SOURCE_EXTENSIONS) {
    if (!enabledExtensions.has(extName)) continue;
    const source = texture.extensions?.[extName]?.source;
    if (source !== undefined) return source;
  }
  return texture.source;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
