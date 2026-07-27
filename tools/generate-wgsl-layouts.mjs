import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

/** Semantic FrameParams fields only — alignment padding is auto-inserted. */
const FRAME_FIELDS = [
  ['width', 'u32'],
  ['height', 'u32'],
  ['frameIndex', 'u32'],
  ['frameSeed', 'u32'],
  ['triangleCount', 'u32'],
  ['maxBounces', 'u32'],
  ['bvhNodeCount', 'u32'],
  ['analyticCount', 'u32'],
  ['pointLightCount', 'u32'],
  ['spotLightCount', 'u32'],
  ['rectAreaLightCount', 'u32'],
  ['meshAreaLightCount', 'u32'],
  ['mneeMaxIterations', 'u32'],
  ['mneeMaxChainLength', 'u32'],
  ['hasEnvironmentMap', 'u32'],
  ['causticStrategy', 'u32'],
  ['environmentMapWidth', 'u32'],
  ['environmentMapHeight', 'u32'],
  ['triIntersectEpsilon', 'f32'],
  ['tlasNodeCount', 'u32'],
  ['spectralEnabled', 'u32'],
  ['heroLambdaNm', 'f32'],
  ['heroPdf', 'f32'],
  ['bdptEnabled', 'u32'],
  ['bdptMaxLightBounces', 'u32'],
  ['bdptMaxEyeDepth', 'u32'],
  // WS2 — many-light importance sampling (full tier only). `lightTreeEnabled`
  // gates the power-weighted light-tree NEE pick; `lightTreeNodeCount` bounds the
  // GPU descent loop. Lite tier never sets these (keeps the uniform pick).
  ['lightTreeEnabled', 'u32'],
  ['lightTreeNodeCount', 'u32'],
  // Only xyz is semantic; the following scalar fills vec3f's aligned fourth lane.
  ['cameraPos', 'vec3f'],
  // H14-E: HDRI intensity remains in slot 31, separate from environmentSun.w.
  ['environmentHdriIntensity', 'f32'],
  ['environmentTint', 'vec4f'],
  ['environmentSun', 'vec4f'],
  ['invViewProj', 'mat4x4f'],
  ['viewProj', 'mat4x4f'],
  ['prevViewProj', 'mat4x4f'],
  // N-directional emitters (d67f0a3): kernel loops this many records from the
  // directionalLights storage buffer. APPENDED LAST so no earlier lane shifts.
  ['directionalLightCount', 'u32'],
  // BDPT pseudo-distant emitters are placed relative to the current scene bounds
  // instead of a Cornell-scale hardcoded radius.
  ['sceneCenterX', 'f32'],
  ['sceneCenterY', 'f32'],
  ['sceneCenterZ', 'f32'],
  ['sceneRadius', 'f32'],
  // Inverse rendering can ask the forward baseline to sum direct-light
  // candidates instead of sampling one emitter. Appended after the existing
  // scalar tail so no earlier field offsets move.
  ['directLightingMode', 'u32'],
];

const SIZE_BYTES = { u32: 4, f32: 4, vec3f: 12, vec3u: 12, vec4f: 16, mat4x4f: 64 };
const ALIGN_BYTES = { u32: 4, f32: 4, vec3f: 16, vec3u: 16, vec4f: 16, mat4x4f: 16 };

/**
 * Generic WGSL-struct layout solver. Given an ordered `[name, type]` field list
 * (pad fields included explicitly, matching the WGSL struct 1:1), returns the
 * byte offset of every field plus the 16-byte-rounded struct size. Used to emit
 * offset tables that the TS packers index by name instead of magic word indices.
 */
function layoutStruct(fields) {
  let offsetBytes = 0;
  const offsets = {};
  for (const [name, type] of fields) {
    const align = ALIGN_BYTES[type];
    const size = SIZE_BYTES[type];
    offsetBytes = Math.ceil(offsetBytes / align) * align;
    offsets[name] = offsetBytes;
    offsetBytes += size;
  }
  return { offsets, byteSize: Math.ceil(offsetBytes / 16) * 16 };
}

// ─── RCLight / RCLightBuffer entry layout ─────────────────────────────────────
// Mirrors `struct RCLight` in packages/walkaround-rc/src/wgsl/probeRayCast.wgsl.ts
// (64 bytes = 16 words). Emitted so wh/rc/packingHelpers.ts indexes by name.
const RC_LIGHT_FIELDS = [
  ['kind', 'u32'],
  ['distance', 'f32'],
  ['decay', 'f32'],
  ['_pad2', 'f32'],
  ['position', 'vec3f'],
  ['intensity', 'f32'],
  ['direction', 'vec3f'],
  ['innerCone', 'f32'],
  ['color', 'vec3f'],
  ['outerCone', 'f32'],
];

// Runtime raw-storage header. Offsets are words relative to the binding start.
const RC_LIGHT_BUFFER_HEADER_FIELDS = [
  ['count', 'u32'],
  ['entriesWordOffset', 'u32'],
  ['aliasWordOffset', 'u32'],
  ['abiMagic', 'u32'],
];

// ─── CascadeUniforms layout ───────────────────────────────────────────────────
// Mirrors `struct CascadeUniforms` in probeRayCast.wgsl.ts. The layout fills
// the established 40-word (160-byte) host uniform-buffer allocation exactly.
// Emitted so cascadeDispatch.ts buildCascadeUniformDataInto indexes named word
// offsets instead of raw magic slot indices (ui[29..31] etc.).
const CASCADE_UNIFORMS_FIELDS = [
  ['probeOriginWorld', 'vec3f'],
  ['_pad0', 'f32'],
  ['roomSize', 'vec3f'],
  ['_pad1', 'f32'],
  ['probeCount', 'vec3u'],
  ['raysPerProbe', 'u32'],
  ['rayGridSize', 'u32'],
  ['intervalNear', 'f32'],
  ['intervalFar', 'f32'],
  ['cascadeIndex', 'u32'],
  ['sunDirection', 'vec3f'],
  ['sunAngularRadius', 'f32'],
  ['sunColor', 'vec3f'],
  ['envIntensity', 'f32'],
  ['frameSeed', 'u32'],
  ['lastCascade', 'u32'],
  ['triIntersectEpsilon', 'f32'],
  ['bvhMode', 'u32'],
  ['tlasNodeCount', 'u32'],
  ['emitterCount', 'u32'],
  ['lightCount', 'u32'],
  ['sunCastShadowDisabled', 'u32'],
  ['emitterDataWordOffset', 'u32'],
  ['emitterAliasWordOffset', 'u32'],
  ['transmittedInterfaceBudget', 'u32'],
  // Appended inside the existing 160-byte host allocation so every prior
  // field offset remains ABI-stable.
  ['envRotationY', 'f32'],
  ['scalarSkyRadiance', 'vec3f'],
  ['hasDirectionalEnv', 'u32'],
];

function generateFrame() {
  let offsetBytes = 0;
  let padIndex = 0;
  const slotMap = [];
  const wgslFieldLines = [];

  for (const [name, type] of FRAME_FIELDS) {
    const align = ALIGN_BYTES[type];
    const size = SIZE_BYTES[type];
    const aligned = Math.ceil(offsetBytes / align) * align;
    const gap = aligned - offsetBytes;
    for (let g = 0; g < gap; g += 4) {
      const padName = `_padAuto${padIndex++}`;
      slotMap.push([padName, offsetBytes / 4]);
      wgslFieldLines.push(`  '${padName}: u32',`);
      offsetBytes += 4;
    }
    slotMap.push([name, offsetBytes / 4]);
    wgslFieldLines.push(`  '${name}: ${type}',`);
    offsetBytes += size;
  }

  const _f32Slots = Math.ceil(offsetBytes / 4); // computed but superseded by the 16-byte-aligned FRAME_PARAMS_F32_SLOTS below
  const slotLines = slotMap.map(([n, s]) => `  ${n}: ${s},`).join('\n');
  const fieldLines = wgslFieldLines.join('\n');
  return `/** AUTO-GENERATED by tools/generate-wgsl-layouts.mjs */\nexport const FRAME_PARAMS_BYTE_SIZE = ${Math.ceil(offsetBytes / 16) * 16};\nexport const FRAME_PARAMS_F32_SLOTS = ${Math.ceil(offsetBytes / 16) * 4};\n\nexport const FrameParamsSlot = {\n${slotLines}\n} as const;\n\nexport const FRAME_PARAMS_WGSL_FIELDS = [\n${fieldLines}\n] as const;\n`;
}

function emitOffsetObject(name, offsets) {
  const lines = Object.entries(offsets)
    .map(([k, v]) => `  ${k}: ${v},`)
    .join('\n');
  return `export const ${name} = {\n${lines}\n} as const;\n`;
}

function generateRC() {
  const light = layoutStruct(RC_LIGHT_FIELDS);
  const header = layoutStruct(RC_LIGHT_BUFFER_HEADER_FIELDS);
  return (
    `/** AUTO-GENERATED by tools/generate-wgsl-layouts.mjs */\n` +
    `export const RC_PARAMS_BYTE_SIZE = 64;\n\n` +
    `export const RCParamsOffset = {\n` +
    `  probeOriginWorld: 0,\n` +
    `  rcWeight: 12,\n` +
    `  roomSize: 16,\n` +
    `  enabled: 28,\n` +
    `  probeCount: 32,\n` +
    `  raysPerProbe: 44,\n` +
    `  rayGridSize: 48,\n` +
    `} as const;\n\n` +
    `/** Byte size of one \`RCLight\` entry (mirrors probeRayCast.wgsl.ts). */\n` +
    `export const RC_LIGHT_ENTRY_BYTES = ${light.byteSize};\n\n` +
    `/** Byte size of the runtime \`RCLightBuffer\` header section. */\n` +
    `export const RC_LIGHTS_HEADER_BYTES = ${header.byteSize};\n\n` +
    `/** Byte size of one Walker/Vose alias entry (q, alias, represented PMF, pad). */\n` +
    `export const RC_LIGHT_ALIAS_ENTRY_BYTES = 16;\n\n` +
    `/** Smallest valid runtime light buffer: a header with count=0. */\n` +
    `export const RC_LIGHTS_BUFFER_BYTES = RC_LIGHTS_HEADER_BYTES;\n` +
    `export const RC_LIGHTS_ABI_MAGIC = 0x31544352;\n\n` +
    `/** Byte offsets within the \`RCLightBuffer\` header. */\n` +
    emitOffsetObject('RCLightBufferHeaderOffset', {
      count: header.offsets.count,
      entriesWordOffset: header.offsets.entriesWordOffset,
      aliasWordOffset: header.offsets.aliasWordOffset,
      abiMagic: header.offsets.abiMagic,
    }) +
    `\n` +
    `/** Field byte offsets within one \`RCLight\` entry (relative to entry start). */\n` +
    emitOffsetObject('RCLightEntryOffset', {
      kind: light.offsets.kind,
      distance: light.offsets.distance,
      decay: light.offsets.decay,
      position: light.offsets.position,
      intensity: light.offsets.intensity,
      direction: light.offsets.direction,
      innerCone: light.offsets.innerCone,
      color: light.offsets.color,
      outerCone: light.offsets.outerCone,
    })
  );
}

function generateCascadeUniforms() {
  const cu = layoutStruct(CASCADE_UNIFORMS_FIELDS);
  return (
    `/** AUTO-GENERATED by tools/generate-wgsl-layouts.mjs */\n` +
    `export const CASCADE_UNIFORMS_BYTE_SIZE = ${cu.byteSize};\n\n` +
    `/** Field byte offsets within the WGSL \`struct CascadeUniforms\` (probeRayCast.wgsl.ts). */\n` +
    emitOffsetObject('CascadeUniformsOffset', cu.offsets)
  );
}

const frameOut = resolve(process.cwd(), 'packages/pt-webgpu/src/scene/frameParamsLayout.generated.ts');
const rcOut = resolve(process.cwd(), 'packages/walkaround-hybrid/src/rc/rcParamsLayout.generated.ts');
const cascadeOut = resolve(process.cwd(), 'packages/walkaround-rc/src/cascadeUniformsLayout.generated.ts');
writeFileSync(frameOut, generateFrame());
writeFileSync(rcOut, generateRC());
writeFileSync(cascadeOut, generateCascadeUniforms());
console.log('generated', frameOut, rcOut, cascadeOut);
