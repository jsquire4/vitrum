import type { GltfJson, GltfPrimitive } from './gltfTypes.js';

export interface GltfSceneReachability {
  readonly sceneIndex: number;
  readonly nodeIndices: ReadonlySet<number>;
  readonly meshIndices: ReadonlySet<number>;
  readonly primitiveKeys: ReadonlySet<string>;
  readonly materialIndices: ReadonlySet<number>;
  readonly bufferViewIndices: ReadonlySet<number>;
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
): GltfSceneReachability {
  const nodeIndices = new Set<number>();
  const meshIndices = new Set<number>();
  const primitiveKeys = new Set<string>();
  const materialIndices = new Set<number>();
  const bufferViewIndices = new Set<number>();
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
    if (node.skin !== undefined) {
      skinIndices.add(node.skin);
      const inverseBindMatrices = gltf.skins?.[node.skin]?.inverseBindMatrices;
      const bufferView = inverseBindMatrices === undefined
        ? undefined
        : gltf.accessors?.[inverseBindMatrices]?.bufferView;
      if (bufferView !== undefined) bufferViewIndices.add(bufferView);
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
      const targetNode = channel.target.node;
      if (targetNode === undefined || !nodeIndices.has(targetNode)) continue;
      const sampler = animation.samplers?.[channel.sampler];
      const inputBufferView = sampler?.input === undefined
        ? undefined
        : gltf.accessors?.[sampler.input]?.bufferView;
      const outputBufferView = sampler?.output === undefined
        ? undefined
        : gltf.accessors?.[sampler.output]?.bufferView;
      if (inputBufferView !== undefined) bufferViewIndices.add(inputBufferView);
      if (outputBufferView !== undefined) bufferViewIndices.add(outputBufferView);
    }
  }

  return {
    sceneIndex,
    nodeIndices,
    meshIndices,
    primitiveKeys,
    materialIndices,
    bufferViewIndices,
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
  const addAccessor = (accessorIndex: number | undefined): void => {
    if (accessorIndex === undefined) return;
    const bufferView = gltf.accessors?.[accessorIndex]?.bufferView;
    if (bufferView !== undefined) out.add(bufferView);
  };
  for (const accessorIndex of Object.values(primitive.attributes ?? {})) addAccessor(accessorIndex);
  addAccessor(primitive.indices);
  for (const target of primitive.targets ?? []) {
    for (const accessorIndex of Object.values(target)) addAccessor(accessorIndex);
  }
}
