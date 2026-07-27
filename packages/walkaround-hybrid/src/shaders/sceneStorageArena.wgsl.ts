import {
  SCENE_STORAGE_ARENA_HEADER_WORDS,
  SCENE_STORAGE_ARENA_MAGIC,
  SCENE_STORAGE_ARENA_SCHEMA,
  SCENE_STORAGE_ARENA_SCHEMA_WORD,
  SCENE_STORAGE_ARENA_VERSION,
  SCENE_STORAGE_SEGMENTS,
} from '../pipeline/sceneStorageArena.js';

const segmentIndex = (name: typeof SCENE_STORAGE_SEGMENTS[number]): number =>
  SCENE_STORAGE_SEGMENTS.indexOf(name);
const offsetWord = (name: typeof SCENE_STORAGE_SEGMENTS[number]): number =>
  8 + segmentIndex(name) * 2;
const countWord = (name: typeof SCENE_STORAGE_SEGMENTS[number]): number =>
  offsetWord(name) + 1;

/** Walkaround implementation of the binding-agnostic shared-BVH loader seam.
 * Three read-only raw arenas replace eleven independently-counted storage
 * bindings while retaining exact byte layouts and logical element counts. */
export const SCENE_STORAGE_ARENA_WGSL = /* wgsl */ `
const SCENE_ARENA_MAGIC: u32 = ${SCENE_STORAGE_ARENA_MAGIC}u;
const SCENE_ARENA_VERSION: u32 = ${SCENE_STORAGE_ARENA_VERSION}u;
const SCENE_ARENA_HEADER_WORDS: u32 = ${SCENE_STORAGE_ARENA_HEADER_WORDS}u;
const SCENE_ARENA_SCHEMA: u32 = ${SCENE_STORAGE_ARENA_SCHEMA}u;

// Arena lighting ABI. This type must precede sceneLoadEmitter: Naga resolves
// function names across declaration order, but requires struct types to exist
// before a function signature references them.
struct EmitterTri {
  vA: vec3f,
  sourceTriIndex: f32,
  vB: vec3f,
  sourceSubdivLevel: f32,
  vC: vec3f,
  sourceSubdivOrdinal: f32,
  normal: vec3f,
  area: f32,
  Le: vec3f,
  castShadowDisabled: f32,
};

@group(1) @binding(0) var<storage, read> sceneGeometryArena: array<u32>;
@group(1) @binding(1) var<storage, read> sceneTlasArena: array<u32>;
@group(1) @binding(2) var<storage, read> sceneLightingArena: array<u32>;

fn sceneGeometryArenaValid() -> bool {
  return sceneGeometryArena[0] == SCENE_ARENA_MAGIC &&
    sceneGeometryArena[1] == SCENE_ARENA_VERSION &&
    sceneGeometryArena[2] == 0u &&
    sceneGeometryArena[3] == 3u &&
    sceneGeometryArena[6] == SCENE_ARENA_HEADER_WORDS &&
    sceneGeometryArena[${SCENE_STORAGE_ARENA_SCHEMA_WORD}u] == SCENE_ARENA_SCHEMA;
}

fn sceneTlasArenaValid() -> bool {
  return sceneTlasArena[0] == SCENE_ARENA_MAGIC &&
    sceneTlasArena[1] == SCENE_ARENA_VERSION &&
    sceneTlasArena[2] == 1u &&
    sceneTlasArena[3] == 3u &&
    sceneTlasArena[6] == SCENE_ARENA_HEADER_WORDS &&
    sceneTlasArena[${SCENE_STORAGE_ARENA_SCHEMA_WORD}u] == SCENE_ARENA_SCHEMA &&
    sceneTlasArena[7] == sceneGeometryArena[7];
}

fn sceneLightingArenaValid() -> bool {
  return sceneLightingArena[0] == SCENE_ARENA_MAGIC &&
    sceneLightingArena[1] == SCENE_ARENA_VERSION &&
    sceneLightingArena[2] == 2u &&
    sceneLightingArena[3] == 3u &&
    sceneLightingArena[6] == SCENE_ARENA_HEADER_WORDS &&
    sceneLightingArena[${SCENE_STORAGE_ARENA_SCHEMA_WORD}u] == SCENE_ARENA_SCHEMA &&
    sceneLightingArena[7] == sceneGeometryArena[7];
}

fn sceneStorageArenasValid() -> bool {
  return sceneGeometryArenaValid() && sceneTlasArenaValid() && sceneLightingArenaValid();
}

fn sceneGeometryU32(word: u32) -> u32 {
  return sceneGeometryArena[word];
}

fn sceneTlasU32(word: u32) -> u32 {
  return sceneTlasArena[word];
}

fn sceneLightingU32(word: u32) -> u32 {
  return sceneLightingArena[word];
}

fn sceneGeometryVec4u(word: u32) -> vec4u {
  return vec4u(
    sceneGeometryU32(word),
    sceneGeometryU32(word + 1u),
    sceneGeometryU32(word + 2u),
    sceneGeometryU32(word + 3u),
  );
}

fn sceneGeometryVec4f(word: u32) -> vec4f {
  return bitcast<vec4f>(sceneGeometryVec4u(word));
}

fn sceneTlasVec4u(word: u32) -> vec4u {
  return vec4u(
    sceneTlasU32(word),
    sceneTlasU32(word + 1u),
    sceneTlasU32(word + 2u),
    sceneTlasU32(word + 3u),
  );
}

fn sceneTlasVec4f(word: u32) -> vec4f {
  return bitcast<vec4f>(sceneTlasVec4u(word));
}

fn sceneLightingVec4u(word: u32) -> vec4u {
  return vec4u(
    sceneLightingU32(word),
    sceneLightingU32(word + 1u),
    sceneLightingU32(word + 2u),
    sceneLightingU32(word + 3u),
  );
}

fn sceneLightingVec4f(word: u32) -> vec4f {
  return bitcast<vec4f>(sceneLightingVec4u(word));
}

fn bvhLoadNode(index: u32) -> BVHNode {
  let word = sceneGeometryArena[${offsetWord('bvhNodes')}u] + index * 8u;
  var node: BVHNode;
  node.boundsMin[0] = bitcast<f32>(sceneGeometryU32(word));
  node.boundsMin[1] = bitcast<f32>(sceneGeometryU32(word + 1u));
  node.boundsMin[2] = bitcast<f32>(sceneGeometryU32(word + 2u));
  node.boundsMax[0] = bitcast<f32>(sceneGeometryU32(word + 3u));
  node.boundsMax[1] = bitcast<f32>(sceneGeometryU32(word + 4u));
  node.boundsMax[2] = bitcast<f32>(sceneGeometryU32(word + 5u));
  node.rightChildOrTriOffset = sceneGeometryU32(word + 6u);
  node.splitAxisOrTriCount = sceneGeometryU32(word + 7u);
  return node;
}

fn bvhNodeCapacity() -> u32 {
  return sceneGeometryArena[${countWord('bvhNodes')}u];
}

fn bvhLoadIndex(index: u32) -> vec4u {
  let word = sceneGeometryArena[${offsetWord('bvhIndex')}u] + index * 4u;
  return sceneGeometryVec4u(word);
}

fn bvhIndexCount() -> u32 {
  return sceneGeometryArena[${countWord('bvhIndex')}u];
}

fn bvhLoadPosition(index: u32) -> vec4f {
  let word = sceneGeometryArena[${offsetWord('bvhPositions')}u] + index * 4u;
  return sceneGeometryVec4f(word);
}

fn bvhPositionCount() -> u32 {
  return sceneGeometryArena[${countWord('bvhPositions')}u];
}

fn sceneLoadBvhNormal(index: u32) -> vec4f {
  let word = sceneGeometryArena[${offsetWord('bvhNormals')}u] + index * 4u;
  return sceneGeometryVec4f(word);
}

fn sceneBvhNormalCount() -> u32 {
  return sceneGeometryArena[${countWord('bvhNormals')}u];
}

fn tlasLoadNode(index: u32) -> BVHNode {
  let word = sceneTlasArena[${offsetWord('tlasNodes')}u] + index * 8u;
  var node: BVHNode;
  node.boundsMin[0] = bitcast<f32>(sceneTlasU32(word));
  node.boundsMin[1] = bitcast<f32>(sceneTlasU32(word + 1u));
  node.boundsMin[2] = bitcast<f32>(sceneTlasU32(word + 2u));
  node.boundsMax[0] = bitcast<f32>(sceneTlasU32(word + 3u));
  node.boundsMax[1] = bitcast<f32>(sceneTlasU32(word + 4u));
  node.boundsMax[2] = bitcast<f32>(sceneTlasU32(word + 5u));
  node.rightChildOrTriOffset = sceneTlasU32(word + 6u);
  node.splitAxisOrTriCount = sceneTlasU32(word + 7u);
  return node;
}

fn tlasNodeCapacity() -> u32 {
  return sceneTlasArena[${countWord('tlasNodes')}u];
}

fn tlasLoadInstanceIndex(index: u32) -> u32 {
  return sceneTlasU32(
    sceneTlasArena[${offsetWord('tlasInstanceIndices')}u] + index,
  );
}

fn tlasInstanceIndexCount() -> u32 {
  return sceneTlasArena[${countWord('tlasInstanceIndices')}u];
}

fn tlasLoadBlasRoot(index: u32) -> u32 {
  return sceneTlasU32(
    sceneTlasArena[${offsetWord('tlasBlasRoots')}u] + index,
  );
}

fn tlasBlasRootCount() -> u32 {
  return sceneTlasArena[${countWord('tlasBlasRoots')}u];
}

fn tlasLoadWorldToLocalColumn(index: u32) -> vec4f {
  let word =
    sceneTlasArena[${offsetWord('tlasInstanceWorldToLocal')}u] + index * 4u;
  return sceneTlasVec4f(word);
}

fn tlasWorldToLocalColumnCount() -> u32 {
  return sceneTlasArena[${countWord('tlasInstanceWorldToLocal')}u] * 4u;
}

fn tlasLoadLocalToWorldColumn(index: u32) -> vec4f {
  let word =
    sceneTlasArena[${offsetWord('tlasInstanceLocalToWorld')}u] + index * 4u;
  return sceneTlasVec4f(word);
}

fn tlasLocalToWorldColumnCount() -> u32 {
  return sceneTlasArena[${countWord('tlasInstanceLocalToWorld')}u] * 4u;
}

fn sceneMneeFacetDomainCount() -> u32 {
  return sceneTlasArena[${countWord('mneeFacetDomains')}u];
}

fn sceneLoadMneeFacetDomainBase(index: u32) -> vec4u {
  let word =
    sceneTlasArena[${offsetWord('mneeFacetDomains')}u] + index * 8u;
  return sceneTlasVec4u(word);
}

fn sceneLoadMneeFacetDomainAlias(index: u32) -> vec4u {
  let word =
    sceneTlasArena[${offsetWord('mneeFacetDomains')}u] + index * 8u + 4u;
  return sceneTlasVec4u(word);
}

fn sceneEmitterCount() -> u32 {
  return sceneLightingArena[${countWord('emitters')}u];
}

fn sceneLoadEmitter(index: u32) -> EmitterTri {
  let word = sceneLightingArena[${offsetWord('emitters')}u] + index * 20u;
  var emitter: EmitterTri;
  emitter.vA = sceneLightingVec4f(word).xyz;
  emitter.sourceTriIndex = bitcast<f32>(sceneLightingU32(word + 3u));
  emitter.vB = sceneLightingVec4f(word + 4u).xyz;
  emitter.sourceSubdivLevel = bitcast<f32>(sceneLightingU32(word + 7u));
  emitter.vC = sceneLightingVec4f(word + 8u).xyz;
  emitter.sourceSubdivOrdinal = bitcast<f32>(sceneLightingU32(word + 11u));
  emitter.normal = sceneLightingVec4f(word + 12u).xyz;
  emitter.area = bitcast<f32>(sceneLightingU32(word + 15u));
  emitter.Le = sceneLightingVec4f(word + 16u).xyz;
  emitter.castShadowDisabled = bitcast<f32>(sceneLightingU32(word + 19u));
  return emitter;
}

fn sceneLoadEmitterCdf(index: u32) -> f32 {
  return bitcast<f32>(
    sceneLightingU32(sceneLightingArena[${offsetWord('emitterCdf')}u] + index),
  );
}


fn sceneEmitterCdfCount() -> u32 {
  return sceneLightingArena[${countWord('emitterCdf')}u];
}

fn sceneEmitterAliasCount() -> u32 {
  return sceneLightingArena[${countWord('emitterAlias')}u];
}

// Walker/Vose entry ABI: q (f32 bits), alias index, represented PMF (f32
// bits), pad. The PMF is the distribution encoded by the quantized table, not
// the ideal source weights, so consumers can divide by the exact wire PMF.
fn sceneLoadEmitterAlias(index: u32) -> vec4u {
  let word =
    sceneLightingArena[${offsetWord('emitterAlias')}u] + index * 4u;
  return sceneLightingVec4u(word);
}
`;
