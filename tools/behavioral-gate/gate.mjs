#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env
// @ts-nocheck
/**
 * tools/behavioral-gate/gate.mjs
 *
 * Behavioral gate — exercises the full pt-webgpu and walkaround-hybrid engines
 * end-to-end on lavapipe (software Vulkan) for every production config.
 *
 * Per config:
 *   - boots the engine
 *   - renders N frames at 64×64
 *   - reads back the output texture
 *   - checks: zero GPU validation/OOM errors, all pixels finite, mean luminance
 *     meets the per-config expectation (see EXPECTATION_TABLE below)
 *
 * Usage (from repo root):
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
 *   npm run behavioral-gate
 *
 * Self-test mode (injects a black config and verifies detection):
 *   ... behavioral-gate -- --self-test
 *
 * Focused subset:
 *   ... behavioral-gate -- --filter gltf
 *
 * Exit codes:
 *   0 — all configs passed their expectations
 *   1 — one or more configs failed
 *
 * ── Expectation table ────────────────────────────────────────────────────────
 *
 * Each entry has:
 *   expected: 'ok' | 'known-residual'
 *
 *   'ok'            — gate FAILS if result is BLACK, GPU-ERROR, NaN, or ERROR
 *   'known-residual' — gate passes regardless of result; the entry records:
 *                       reason: human-readable explanation
 *                       planItem: road-to-100 / trust-remediation item number
 *                     As fixes land, the agent fixing the config updates the
 *                     table to 'ok' and removes reason/planItem.
 *
 * ── Naga gap patches ─────────────────────────────────────────────────────────
 *
 * - pt-webgpu: bdptLightPath storage texture → sampled texture when bdpt=off;
 *   isNan/isInf polyfills; strip 3-arg textureLoad mip level.
 * - walkaround-hybrid: uses the production nagaFix.mjs from tools/shader-gate/
 *   (ptr<storage> parameter rewrite, among others). Relative import — no
 *   absolute paths.
 *
 * IMPORTANT: all @vitrum/* imports are resolved via deno.json in this directory
 * (relative paths). Do NOT add absolute paths — see the stale-import lesson in
 * CLAUDE.md.
 */

import { createPTEngine_WebGPU } from "@vitrum/pt-webgpu";
import { createWalkaroundEngine_Hybrid } from "@vitrum/walkaround-hybrid";
import { asMat4 } from "@vitrum/core";
import { loadGltfForEngine } from "@vitrum/gltf-adapter";
import { Buffer } from "node:buffer";
import { PNG } from "npm:pngjs@7.0.0";
import { applyNagaFix } from "../shader-gate/nagaFix.mjs";
import {
  SWEEP_MAPS,
  makeSweepGltf,
  makeSweepTextureDecodeHooks,
} from "../gltf-material-sweep/fixture.mjs";

// ── CLI flags ─────────────────────────────────────────────────────────────────
const selfTest = Deno.args.includes("--self-test");
const updateGoldens = Deno.args.includes("--update-goldens");
function readFlagValue(name) {
  const eq = Deno.args.find((a) => a.startsWith(`${name}=`));
  if (eq) return eq.slice(name.length + 1);
  const i = Deno.args.indexOf(name);
  if (i >= 0) return Deno.args[i + 1] ?? "";
  return "";
}
const labelFilter = readFlagValue("--filter");

// ── Expectation table ─────────────────────────────────────────────────────────
// keyed by config label; missing entry defaults to { expected: 'ok' }.
const EXPECTATION_TABLE = {
  // pt-webgpu configs
  "pt/default":           { expected: "ok" },
  "pt/spectral":          { expected: "ok" },
  "pt/bdpt":              { expected: "ok" },
  "pt/caustic-manifold":  { expected: "ok" },
  "pt/caustic-photon":    { expected: "ok" }, // R7a-3 fixed: capacity 32 fits default maxStorageBufferBindingSize
  "pt/spectral+photon":  { expected: "ok" }, // item 21: spectral×photon-map gather fix (heroLambda passed to sppmGather)
  "pt/lite-tier":         { expected: "ok" },
  "pt/restirPtReuse":     { expected: "ok" }, // A1 done: composite megakernel folds rpt indirect into beauty (kernel.wgsl.ts:308-312, gpuResources.ts:930)
  "pt/skinned-mesh":      { expected: "ok" },
  "pt/analytic-sphere":   { expected: "ok" },
  "pt/point-light":       { expected: "ok" },
  // 2026-06-10: native analytic disc emitters — shape tag 1.0, concentric-disc
  // sampling (Shirley-Chiu), π·r² area in MIS pdf. Asserts non-black overall
  // luminance; the disc ceiling light must illuminate the Cornell box.
  "pt/disc-light":        { expected: "ok" },
  "pt/spot-light":        { expected: "ok" },
  "pt/directional-2":     { expected: "ok" },
  "pt/hdri-env":          { expected: "ok" },
  "pt/procedural-sky":    { expected: "ok" },
  "pt/spectral+bdpt":     { expected: "ok" },
  "pt/lite+hdri":         { expected: "ok" },
  "pt/lite+point-light":  { expected: "ok" },
  "pt/gltf-unlit":        { expected: "ok" },
  "pt/gltf-textured-pbr": { expected: "ok" },
  "pt/gltf-transmission": { expected: "ok" },
  "pt/gltf-skinned-animation": { expected: "ok" },
  "pt/gltf-draco-mock":   { expected: "ok" },
  "pt/gltf-material-sweep": { expected: "ok" },
  "pt/gltf-real-box-textured": { expected: "ok" },
  "pt/gltf-real-draco": { expected: "ok" },
  "pt/gltf-real-meshopt": { expected: "ok" },

  // walkaround configs
  "wh/default":           { expected: "ok" },
  "wh/rcEnabled":         { expected: "ok" }, // R7a-2 fixed: rc_tlas naga renames
  "wh/ppgEnabled":        { expected: "ok" },
  "wh/gtao-off":          { expected: "ok" },
  "wh/checkerboard":      { expected: "ok" },
  "wh/skinned-mesh":      { expected: "ok" },
  "wh/hdri-env":          { expected: "ok" },
  "wh/rect-area-emitter": { expected: "ok" },
  // item 4 (2026-06-10) — direct sun NEE default-on. Strong directional emitter,
  // no rect-area emitter; asserts non-black + higher luminance than the no-sun default.
  // Pin provenance: sun-NEE default-on, 2026-06-10 — RENDER-CHANGING for directional-lit scenes, A/B in R8-C.
  "wh/directional-sun":   { expected: "ok" },
  // B1 tail (2026-06-10) — glass refracted GI. Glass pane in front of a lit diffuse
  // wall; a rect-area emitter lights the back wall; the camera looks through the glass.
  // The glass pixel receives GI from the diffuse wall behind it via the refracted
  // reservoir. Assert non-black overall luminance — the emitter lights the scene
  // regardless of glass; the glass-GI term adds the indirect contribution seen through
  // the glass. RENDER-CHANGING for glass scenes, A/B in R8-C.
  "wh/glass-gi":          { expected: "ok" },
};

// ── Matrix ────────────────────────────────────────────────────────────────────

const PT_CONFIGS = [
  { label: "pt/default",          eng: {},                                    scene: {} },
  { label: "pt/spectral",         eng: { spectral: true },                    scene: {} },
  { label: "pt/bdpt",             eng: { bdpt: true },                        scene: {} },
  { label: "pt/caustic-manifold", eng: { causticStrategy: "manifold-nee" },   scene: {} },
  { label: "pt/caustic-photon",   eng: { causticStrategy: "photon-map" },     scene: {} },
  { label: "pt/spectral+photon",  eng: { spectral: true, causticStrategy: "photon-map" }, scene: {} },
  { label: "pt/lite-tier",        eng: { traceTier: "lite" },                 scene: {} },
  { label: "pt/restirPtReuse",    eng: { restirPtReuse: true },               scene: {} },
  { label: "pt/skinned-mesh",     eng: {},                                    scene: { skinned: true } },
  { label: "pt/analytic-sphere",  eng: {},                                    scene: { analytic: true } },
  { label: "pt/point-light",      eng: {},                                    scene: {
    emitters: [{ kind: "point", id: "pt-light", position: [0, 0.8, 0], color: [1,1,1], intensity: 4.0 }],
  }},
  // 2026-06-10: native analytic disc emitter — packed into rect stream with shape
  // tag 1.0 (Shirley-Chiu concentric-disc map, π·r² area, circle containment MIS).
  // The disc is on the ceiling facing down; the Cornell box surfaces should be lit.
  { label: "pt/disc-light",       eng: {},                                    scene: {
    emitters: [{ kind: "disc-area", id: "disc-light", position: [0, 0.95, 0],
      normal: [0, -1, 0], radius: 0.3, color: [1,1,1], intensity: 12.0 }],
  }},
  { label: "pt/spot-light",       eng: {},                                    scene: {
    emitters: [{ kind: "spot", id: "sp-light", position: [0, 0.8, 0], direction: [0,-1,0],
      color: [1,1,1], intensity: 6.0, angle: 0.8, penumbra: 0.2 }],
  }},
  { label: "pt/directional-2",    eng: {},                                    scene: {
    emitters: [
      { kind: "directional", id: "dir1", direction: [0.3,-0.8,0.5], color: [1,1,1], intensity: 2.0 },
      { kind: "directional", id: "dir2", direction: [-0.3,-0.8,-0.5], color: [0.8,0.9,1.0], intensity: 1.0 },
    ],
  }},
  { label: "pt/hdri-env",         eng: {},                                    scene: { hdri: true } },
  { label: "pt/procedural-sky",   eng: {},                                    scene: { sky: true } },
  { label: "pt/spectral+bdpt",    eng: { spectral: true, bdpt: true },        scene: {} },
  { label: "pt/lite+hdri",        eng: { traceTier: "lite" },                 scene: { hdri: true } },
  { label: "pt/lite+point-light", eng: { traceTier: "lite" },                 scene: {
    emitters: [{ kind: "point", id: "pt-light", position: [0, 0.8, 0], color: [1,1,1], intensity: 4.0 }],
  }},
  { label: "pt/gltf-unlit",        eng: {},                                    scene: { gltf: "unlit" } },
  { label: "pt/gltf-textured-pbr", eng: {},                                    scene: { gltf: "textured-pbr" } },
  { label: "pt/gltf-transmission", eng: {},                                    scene: { gltf: "transmission" } },
  { label: "pt/gltf-skinned-animation", eng: {},                               scene: { gltf: "skinned-animation" } },
  { label: "pt/gltf-draco-mock",   eng: {},                                    scene: { gltf: "draco-mock" } },
  { label: "pt/gltf-material-sweep", eng: {},                                  scene: { gltf: "material-sweep" } },
  { label: "pt/gltf-real-box-textured", eng: {},                                scene: { gltfReal: "box-textured-glb" } },
  { label: "pt/gltf-real-draco", eng: {},                                       scene: { gltfReal: "cesium-milk-truck-draco" } },
  { label: "pt/gltf-real-meshopt", eng: {},                                     scene: { gltfReal: "meshopt-cube-real" } },
];

const WH_CONFIGS = [
  { label: "wh/default",           eng: {},                                    scene: {} },
  { label: "wh/rcEnabled",         eng: { rcEnabled: true },                   scene: {} },
  { label: "wh/ppgEnabled",        eng: { ppgEnabled: true },                  scene: {} },
  { label: "wh/gtao-off",          eng: { gtaoEnabled: false },                scene: {} },
  { label: "wh/checkerboard",      eng: { checkerboardEnabled: true },         scene: {} },
  { label: "wh/skinned-mesh",      eng: {},                                    scene: { skinned: true } },
  { label: "wh/hdri-env",          eng: {},                                    scene: { hdri: true } },
  { label: "wh/rect-area-emitter", eng: {},                                    scene: {} },
  // item 4 (2026-06-10) — direct sun NEE: strong directional emitter, no rect-area
  // light. Opaque Cornell box surfaces should be lit via lo_sunNEE. LUM_THRESHOLD
  // gates BLACK detection (>0.005); a strong directional at intensity 3.0 reliably
  // exceeds that after 8 frames even on lavapipe.
  { label: "wh/directional-sun",   eng: {
      primaryLightDir:       [0.3, -0.8, 0.5],
      primaryLightIntensity: 3.0,
    },                                                                          scene: { directionalOnly: true } },
  // B1 tail (2026-06-10) — glass refracted GI: glass pane in front of a lit diffuse
  // wall. Glass primaries receive a refracted-GI reservoir built at the back wall.
  // Assert non-black overall luminance — the ceiling emitter lights the scene
  // regardless (direct light through glass); the glass-GI term adds the diffuse-wall
  // indirect contribution visible through the pane.
  { label: "wh/glass-gi",          eng: {},                                    scene: { glass: true } },
];

// ── Scene builder ─────────────────────────────────────────────────────────────

function makeQuad(id, verts, normal, color) {
  return {
    kind: "mesh", id,
    positions: new Float32Array(verts.flat()),
    normals:   new Float32Array([...normal, ...normal, ...normal, ...normal]),
    uvs:       new Float32Array(8),
    indices:   new Uint32Array([0, 2, 1, 2, 0, 3]),
    material:  { baseColor: color, roughness: 1.0, metallic: 0.0 },
  };
}

function buildCornellScene(opts = {}) {
  const primitives = [
    makeQuad("floor",      [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]], [0,1,0],  [0.8,0.8,0.8]),
    makeQuad("ceiling",    [[-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1]],     [0,-1,0], [0.8,0.8,0.8]),
    makeQuad("back-wall",  [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],     [0,0,-1], [0.8,0.8,0.8]),
    makeQuad("left-wall",  [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]], [1,0,0],  [0.75,0.1,0.1]),
    makeQuad("right-wall", [[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]],      [-1,0,0], [0.1,0.6,0.1]),
  ];

  if (opts.skinned) {
    // For pt-webgpu: skinIndices/skinWeights/bones/boneInverses contract
    // For walkaround-hybrid: joints/weights/bindMatrices/jointMatrices contract
    // We supply both sets so the same buildCornellScene works for both backends.
    const ident = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
    primitives.push({
      kind: "skinned-mesh", id: "skinned-box",
      positions:    new Float32Array([-0.3,-0.7,0.0, 0.3,-0.7,0.0, 0.3,-0.3,0.0, -0.3,-0.3,0.0]),
      normals:      new Float32Array([0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1]),
      uvs:          new Float32Array(8),
      indices:      new Uint32Array([0,2,1,2,0,3]),
      // pt-webgpu contract
      skinIndices:  new Uint32Array([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]),
      skinWeights:  new Float32Array([1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]),
      bones:        ident,
      boneInverses: ident,
      // walkaround-hybrid contract
      joints:        new Uint16Array([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]),
      weights:       new Float32Array([1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]),
      bindMatrices:  [ident],
      jointMatrices: [ident],
      material: { baseColor: [0.5,0.5,0.9], roughness: 0.6, metallic: 0.0 },
    });
  }

  if (opts.analytic) {
    primitives.push({
      kind: "analytic", id: "analytic-sphere",
      shape: { kind: "sphere", radius: 0.3 },
      transform: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0.2,-0.5,0.2,1]),
      material: { baseColor: [0.9,0.5,0.1], roughness: 0.2, metallic: 0.8 },
    });
  }

  // B1 tail (2026-06-10) — glass refracted GI scene: a glass pane placed between
  // the camera and the back wall. The rect-area emitter on the ceiling lights the
  // back wall; the camera looks through the glass pane. Glass primaries now receive
  // a refracted-GI reservoir built at the back wall behind the pane.
  // The pane is a quad at z=0.5 (camera is at z=2.5, back wall at z=1), normal
  // pointing toward camera. transmission=1.0 → decoded matColor.a ≈ 1.0 > 0.3
  // → isGlass=true in all walkaround shaders.
  if (opts.glass) {
    primitives.push({
      kind: "mesh", id: "glass-pane",
      positions: new Float32Array([-0.5,-0.5,0.5, 0.5,-0.5,0.5, 0.5,0.5,0.5, -0.5,0.5,0.5]),
      normals:   new Float32Array([0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1]),
      uvs:       new Float32Array(8),
      indices:   new Uint32Array([0,2,1, 2,0,3]),
      material:  { baseColor: [1.0,1.0,1.0], roughness: 0.05, metallic: 0.0, transmission: 1.0 },
    });
  }

  // directionalOnly: no rect-area emitter, so the only light is the primary
  // directional set on the engine (tests that lo_sunNEE lights opaque surfaces).
  const emitters = opts.directionalOnly ? [] : (opts.emitters ?? [{
    kind: "rect-area", id: "ceiling-light",
    position: [0, 0.95, 0],
    uAxis: [0, 0, 0.2], vAxis: [0.2, 0, 0],
    color: [1,1,1], intensity: 12.0,
  }]);

  let environment = { kind: "none" };
  if (opts.hdri) {
    const W = 4, H = 2;
    const texels = new Float32Array(W * H * 4);
    for (let i = 0; i < texels.length; i += 4) {
      texels[i] = 1.0; texels[i+1] = 1.0; texels[i+2] = 1.0; texels[i+3] = 1.0;
    }
    environment = { kind: "hdri", textureData: texels, width: W, height: H, intensity: 1.0 };
  }
  if (opts.sky) {
    environment = { kind: "procedural-sky", sunDirection: [0.5, 1.0, 0.3] };
  }

  return { primitives, emitters, environment };
}

// ── glTF fixture builder ─────────────────────────────────────────────────────

function exactArrayBuffer(view) {
  return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
}

function createGltfBufferBuilder() {
  const buffers = new Map();
  const gltf = {
    asset: { version: "2.0" },
    buffers: [],
    bufferViews: [],
    accessors: [],
    materials: [],
    meshes: [],
    nodes: [],
    scenes: [],
    scene: 0,
  };

  function addBufferView(view) {
    const data = view instanceof ArrayBuffer
      ? view
      : exactArrayBuffer(view);
    const bufferIndex = gltf.buffers.length;
    buffers.set(bufferIndex, data);
    gltf.buffers.push({ byteLength: data.byteLength });
    gltf.bufferViews.push({ buffer: bufferIndex, byteOffset: 0, byteLength: data.byteLength });
    return gltf.bufferViews.length - 1;
  }

  function addAccessor(view, type, componentType, componentCount, extra = {}) {
    const bufferView = addBufferView(view);
    gltf.accessors.push({
      bufferView,
      componentType,
      count: Math.floor(view.length / componentCount),
      type,
      ...extra,
    });
    return gltf.accessors.length - 1;
  }

  return { gltf, buffers, addBufferView, addAccessor };
}

const GLTF_QUAD = {
  positions: new Float32Array([
    -0.75, -0.65, 0,
     0.75, -0.65, 0,
     0.75,  0.65, 0,
    -0.75,  0.65, 0,
  ]),
  normals: new Float32Array([
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
  ]),
  uvs: new Float32Array([
    0, 0,
    1, 0,
    1, 1,
    0, 1,
  ]),
  indices: new Uint16Array([0, 1, 2, 0, 2, 3]),
};

function addQuadMesh(builder, materialIndex, extraAttributes = {}) {
  const position = builder.addAccessor(GLTF_QUAD.positions, "VEC3", 5126, 3, {
    min: [-0.75, -0.65, 0],
    max: [0.75, 0.65, 0],
  });
  const normal = builder.addAccessor(GLTF_QUAD.normals, "VEC3", 5126, 3);
  const uv = builder.addAccessor(GLTF_QUAD.uvs, "VEC2", 5126, 2);
  const indices = builder.addAccessor(GLTF_QUAD.indices, "SCALAR", 5123, 1);
  builder.gltf.meshes.push({
    primitives: [{
      attributes: { POSITION: position, NORMAL: normal, TEXCOORD_0: uv, ...extraAttributes },
      indices,
      material: materialIndex,
    }],
  });
  return builder.gltf.meshes.length - 1;
}

function finalizeSingleMeshGltf(builder, meshIndex, nodeExtra = {}) {
  builder.gltf.nodes.push({ mesh: meshIndex, ...nodeExtra });
  builder.gltf.scenes.push({ nodes: [0] });
  builder.gltf.scene = 0;
  return { gltf: builder.gltf, buffers: builder.buffers };
}

function makeQuadGltfFixture(kind) {
  const builder = createGltfBufferBuilder();
  const material = {
    pbrMetallicRoughness: {
      baseColorFactor: [0.9, 0.45, 0.18, 1],
      roughnessFactor: 0.65,
      metallicFactor: 0,
    },
  };
  let decodeImage;
  let decodePixels;

  if (kind === "unlit") {
    builder.gltf.extensionsUsed = ["KHR_materials_unlit"];
    material.extensions = { KHR_materials_unlit: {} };
    material.pbrMetallicRoughness.baseColorFactor = [0.85, 0.25, 0.05, 1];
  }

  if (kind === "textured-pbr") {
    const imageView = builder.addBufferView(new Uint8Array([0x76, 0x74, 0x72, 0x6d]));
    builder.gltf.images = [{ bufferView: imageView, mimeType: "image/x-vitrum-test" }];
    builder.gltf.samplers = [{ wrapS: 10497, wrapT: 10497, magFilter: 9728, minFilter: 9728 }];
    builder.gltf.textures = [{ source: 0, sampler: 0 }];
    material.pbrMetallicRoughness.baseColorTexture = { index: 0 };
    decodeImage = async (bytes, mimeType) => ({
      kind: "raw-image",
      mimeType,
      data: bytes,
    });
    decodePixels = async (_handle, context) => ({
      width: 2,
      height: 2,
      channels: 4,
      dataType: "uint8",
      colorSpace: context.colorSpace,
      data: new Uint8Array([
        255, 48, 48, 255,
        48, 255, 48, 255,
        48, 48, 255, 255,
        255, 255, 48, 255,
      ]),
    });
  }

  if (kind === "transmission") {
    builder.gltf.extensionsUsed = ["KHR_materials_transmission"];
    material.extensions = { KHR_materials_transmission: { transmissionFactor: 0.85 } };
    material.pbrMetallicRoughness.baseColorFactor = [0.9, 0.95, 1.0, 0.55];
    material.pbrMetallicRoughness.roughnessFactor = 0.05;
    material.alphaMode = "BLEND";
  }

  builder.gltf.materials.push(material);
  const mesh = addQuadMesh(builder, 0);
  return { ...finalizeSingleMeshGltf(builder, mesh), decodeImage, decodePixels };
}

function makeSkinnedAnimationGltfFixture() {
  const builder = createGltfBufferBuilder();
  builder.gltf.materials.push({
    pbrMetallicRoughness: {
      baseColorFactor: [0.35, 0.45, 0.95, 1],
      roughnessFactor: 0.55,
      metallicFactor: 0,
    },
  });
  const joints = builder.addAccessor(new Uint16Array([
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  ]), "VEC4", 5123, 4);
  const weights = builder.addAccessor(new Float32Array([
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
  ]), "VEC4", 5126, 4);
  const mesh = addQuadMesh(builder, 0, { JOINTS_0: joints, WEIGHTS_0: weights });
  const ibm = builder.addAccessor(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]), "MAT4", 5126, 16);
  const times = builder.addAccessor(new Float32Array([0, 1]), "SCALAR", 5126, 1);
  const rotations = builder.addAccessor(new Float32Array([
    0, 0, 0, 1,
    0, 0, 0.38268343, 0.92387953,
  ]), "VEC4", 5126, 4);
  builder.gltf.nodes.push({ mesh, skin: 0, children: [1] }, { name: "joint0" });
  builder.gltf.skins = [{ joints: [1], inverseBindMatrices: ibm }];
  builder.gltf.animations = [{
    samplers: [{ input: times, output: rotations, interpolation: "LINEAR" }],
    channels: [{ sampler: 0, target: { node: 1, path: "rotation" } }],
  }];
  builder.gltf.scenes.push({ nodes: [0] });
  builder.gltf.scene = 0;
  return { gltf: builder.gltf, buffers: builder.buffers };
}

function makeDracoMockGltfFixture() {
  const builder = createGltfBufferBuilder();
  const compressedView = builder.addBufferView(new Uint8Array([0xde, 0xc0, 0xde, 0x01]));
  builder.gltf.extensionsUsed = ["KHR_draco_mesh_compression"];
  builder.gltf.extensionsRequired = ["KHR_draco_mesh_compression"];
  builder.gltf.materials.push({
    pbrMetallicRoughness: {
      baseColorFactor: [0.95, 0.65, 0.15, 1],
      roughnessFactor: 0.5,
      metallicFactor: 0,
    },
  });
  builder.gltf.accessors.push(
    { componentType: 5126, count: 4, type: "VEC3" },
    { componentType: 5126, count: 4, type: "VEC3" },
    { componentType: 5126, count: 4, type: "VEC2" },
    { componentType: 5123, count: 6, type: "SCALAR" },
  );
  builder.gltf.meshes.push({
    primitives: [{
      attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 },
      indices: 3,
      material: 0,
      extensions: {
        KHR_draco_mesh_compression: {
          bufferView: compressedView,
          attributes: { POSITION: 10, NORMAL: 11, TEXCOORD_0: 12 },
        },
      },
    }],
  });
  const dracoDecode = (_bytes, attributeIds) => {
    if (attributeIds.POSITION !== 10 || attributeIds.NORMAL !== 11 || attributeIds.TEXCOORD_0 !== 12) {
      throw new Error("unexpected Draco attribute id map");
    }
    return {
      attributes: {
        POSITION: GLTF_QUAD.positions,
        NORMAL: GLTF_QUAD.normals,
        TEXCOORD_0: GLTF_QUAD.uvs,
      },
      indices: GLTF_QUAD.indices,
    };
  };
  return { ...finalizeSingleMeshGltf(builder, 0), dracoDecode };
}

async function buildGltfFixtureScene(kind) {
  const fixture =
    kind === "material-sweep" ? { ...makeSweepGltf(), ...makeSweepTextureDecodeHooks(), materialSweep: true }
      : kind === "skinned-animation" ? makeSkinnedAnimationGltfFixture()
      : kind === "draco-mock" ? makeDracoMockGltfFixture()
      : makeQuadGltfFixture(kind);
  const preparedScenes = [];
  const primitivePatches = [];
  const patchTarget = {
    setScene(scene) {
      preparedScenes.push(scene);
    },
    updatePrimitive(id, patch) {
      primitivePatches.push({ id, patch });
    },
  };
  const createEngine = async ({ scene, backend, asset }) => {
    if (backend !== "pt-webgpu") {
      throw new Error(`glTF behavioral gate selected unexpected backend "${backend}"`);
    }
    if (scene !== asset.scene) {
      throw new Error("glTF behavioral gate createEngine received a scene that does not match asset.scene");
    }
    return patchTarget;
  };
  const result = await loadGltfForEngine(fixture.gltf, {
    buffers: fixture.buffers,
    backend: "pt-webgpu",
    createEngine,
    ...(fixture.decodeImage ? {
      decodeTextures: true,
      textureTarget: "cpu-linear",
      decodeImage: fixture.decodeImage,
      decodePixels: fixture.decodePixels,
      ...(fixture.materialSweep ? { maxTextureSize: 8, warnOnNpotRepeatWrap: true } : {}),
    } : {}),
    ...(fixture.dracoDecode ? { dracoDecode: fixture.dracoDecode } : {}),
  });
  if (!result.attached || preparedScenes.length !== 1) {
    throw new Error("glTF behavioral gate did not exercise controller.attachEngine/setScene");
  }
  const importedScene = result.controller.scene;
  if (importedScene.primitives.length === 0) {
    throw new Error(`glTF fixture "${kind}" imported no primitives`);
  }
  const first = importedScene.primitives[0];
  if (kind === "unlit" && first.material?.shadingModel !== "unlit") {
    throw new Error("glTF unlit fixture lost KHR_materials_unlit");
  }
  if (kind === "textured-pbr" && first.material?.baseColorMap == null) {
    throw new Error("glTF textured-pbr fixture lost baseColorTexture");
  }
  if (kind === "textured-pbr") {
    const entry = result.textureDecodeReport.entries.find((candidate) =>
      candidate.materialField === "baseColorMap" && candidate.primitiveIndex === 0);
    const handle = first.material.baseColorMap.handle;
    if (result.decodedTextureCount !== 1 || result.unchangedTextureCount !== 0) {
      throw new Error(
        `glTF textured-pbr fixture did not exercise decode bridge ` +
        `(decoded=${result.decodedTextureCount}, unchanged=${result.unchangedTextureCount})`,
      );
    }
    if (result.textureDecodeWarnings.length > 0) {
      throw new Error(`glTF textured-pbr fixture emitted texture decode warnings: ${result.textureDecodeWarnings.join(" | ")}`);
    }
    if (entry == null || entry.handleKind !== "pixel-data" || entry.colorSpace !== "srgb" ||
        entry.backendReadiness.ptWebgpu !== "ready") {
      throw new Error(`glTF textured-pbr fixture did not surface a backend-ready CPU texture decode report`);
    }
    if (!(handle?.data instanceof Float32Array) || handle.__vitrum_hint__?.colorSpace !== "linear") {
      throw new Error("glTF textured-pbr fixture did not attach the CPU-linear decoded texture to the engine scene");
    }
    if (preparedScenes[0] !== result.asset.scene || result.asset.scene !== result.controller.scene) {
      throw new Error("glTF textured-pbr fixture attached a scene other than the decoded controller scene");
    }
  }
  if (kind === "material-sweep") {
    const missing = SWEEP_MAPS.filter((field) => !result.textureDecodeReport.entries.some((entry) =>
      entry.materialField === field && entry.primitiveIndex === 0));
    if (missing.length > 0) {
      throw new Error(`glTF material-sweep fixture missed textureDecodeReport fields: ${missing.join(", ")}`);
    }
    if (result.textureDecodeDiagnostics.length > 0) {
      throw new Error(
        `glTF material-sweep fixture emitted texture decode diagnostics: ` +
        `${result.textureDecodeDiagnostics.map((d) => d.message ?? d.code ?? String(d)).join(" | ")}`,
      );
    }
    if (result.textureDecodeWarnings.length > 0) {
      throw new Error(`glTF material-sweep fixture emitted texture decode warnings: ${result.textureDecodeWarnings.join(" | ")}`);
    }
    if (result.textureDecodeReport.mapCount !== 18 || result.textureDecodeReport.cpuReadableCount !== 18) {
      throw new Error(
        `glTF material-sweep fixture did not decode every material map ` +
        `(maps=${result.textureDecodeReport.mapCount}, cpu=${result.textureDecodeReport.cpuReadableCount})`,
      );
    }
    const missingHandles = SWEEP_MAPS.filter((field) => {
      const ref = first.material?.[field];
      return !(ref?.handle?.data instanceof Float32Array);
    });
    if (missingHandles.length > 0) {
      throw new Error(`glTF material-sweep fixture did not attach CPU-linear handles for: ${missingHandles.join(", ")}`);
    }
    if (preparedScenes[0] !== result.asset.scene || result.asset.scene !== result.controller.scene) {
      throw new Error("glTF material-sweep fixture attached a scene other than the decoded controller scene");
    }
  }
  if (kind === "transmission" && (first.material?.transmission ?? 0) <= 0) {
    throw new Error("glTF transmission fixture lost KHR_materials_transmission");
  }
  if (kind === "skinned-animation") {
    if (first.kind !== "skinned-mesh") throw new Error("glTF skinned fixture did not import a SkinnedMeshPrimitive");
    if ((result.asset.animations?.length ?? 0) === 0) throw new Error("glTF skinned fixture lost animation channels");
    const frame = result.controller.advance(0.5);
    if (frame.primitivePatches.length === 0 || primitivePatches.length === 0) {
      throw new Error("glTF skinned fixture controller advance did not patch the attached engine");
    }
  }
  if (kind === "draco-mock" && result.asset.warnings.some((w) => w.includes("KHR_draco_mesh_compression"))) {
    throw new Error(`glTF Draco mock emitted compression warning: ${result.asset.warnings.join(" | ")}`);
  }

  return {
    ...result.controller.scene,
    emitters: [{
      kind: "rect-area", id: "gltf-gate-light",
      position: [0, 0.85, 0.45],
      uAxis: [0.25, 0, 0], vAxis: [0, 0.25, 0],
      color: [1, 1, 1], intensity: 14.0,
    }],
    environment: { kind: "procedural-sky", sunDirection: [0.4, 1.0, 0.25] },
  };
}

function multiplyMat4(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

function transformPointMat4(m, x, y, z) {
  if (!m) return [x, y, z];
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ];
}

function includePoint(aabb, x, y, z) {
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
  aabb.min[0] = Math.min(aabb.min[0], x);
  aabb.min[1] = Math.min(aabb.min[1], y);
  aabb.min[2] = Math.min(aabb.min[2], z);
  aabb.max[0] = Math.max(aabb.max[0], x);
  aabb.max[1] = Math.max(aabb.max[1], y);
  aabb.max[2] = Math.max(aabb.max[2], z);
}

function computePrimitivePositionAabb(scene) {
  const aabb = {
    min: [Infinity, Infinity, Infinity],
    max: [-Infinity, -Infinity, -Infinity],
  };
  for (const primitive of scene.primitives ?? []) {
    const positions = primitive.positions;
    if (!(positions instanceof Float32Array) || positions.length < 3) continue;
    const transforms = primitive.kind === "instanced-mesh"
      ? (primitive.instances ?? [])
      : [primitive.transform];
    for (const transform of transforms.length > 0 ? transforms : [undefined]) {
      for (let i = 0; i + 2 < positions.length; i += 3) {
        const [x, y, z] = transformPointMat4(transform, positions[i], positions[i + 1], positions[i + 2]);
        includePoint(aabb, x, y, z);
      }
    }
  }
  return Number.isFinite(aabb.min[0]) ? aabb : null;
}

function normalizeSceneForBehavioralGate(scene) {
  const aabb = computePrimitivePositionAabb(scene);
  if (!aabb) return scene;
  const center = [
    (aabb.min[0] + aabb.max[0]) * 0.5,
    (aabb.min[1] + aabb.max[1]) * 0.5,
    (aabb.min[2] + aabb.max[2]) * 0.5,
  ];
  const extent = Math.max(
    aabb.max[0] - aabb.min[0],
    aabb.max[1] - aabb.min[1],
    aabb.max[2] - aabb.min[2],
  );
  if (!(extent > 0)) return scene;
  const s = 1.35 / extent;
  const normalization = asMat4(new Float32Array([
    s, 0, 0, 0,
    0, s, 0, 0,
    0, 0, s, 0,
    -center[0] * s, -center[1] * s, -center[2] * s, 1,
  ]));
  const primitives = (scene.primitives ?? []).map((primitive) => {
    if (primitive.kind === "instanced-mesh") {
      return {
        ...primitive,
        instances: (primitive.instances ?? []).map((instance) => asMat4(multiplyMat4(normalization, instance))),
      };
    }
    return {
      ...primitive,
      transform: asMat4(multiplyMat4(
        normalization,
        primitive.transform ?? new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      )),
    };
  });
  return { ...scene, primitives };
}

async function buildRealGltfAssetScene(assetId) {
  const {
    getRealGltfAsset,
    makeRealGltfDecodeHooks,
  } = await import("../gltf-real-asset-sweep/assets.mjs");
  const asset = getRealGltfAsset(assetId);
  const hooks = await makeRealGltfDecodeHooks();
  const preparedScenes = [];
  const patchTarget = {
    setScene(scene) {
      preparedScenes.push(scene);
    },
    updatePrimitive() {
      throw new Error("real glTF behavioral gate does not expect runtime primitive patches");
    },
  };
  const createEngine = async ({ scene, backend, asset: loadedAsset }) => {
    if (backend !== "pt-webgpu") {
      throw new Error(`real glTF behavioral gate selected unexpected backend "${backend}"`);
    }
    if (scene !== loadedAsset.scene) {
      throw new Error("real glTF behavioral gate createEngine received a scene that does not match asset.scene");
    }
    return patchTarget;
  };
  const result = await loadGltfForEngine(asset.url, {
    backend: "pt-webgpu",
    createEngine,
    decodeTextures: true,
    textureTarget: "cpu-linear",
    decodePixels: hooks.decodePixels,
    maxTextureSize: 4096,
    warnOnNpotRepeatWrap: true,
    dracoDecode: hooks.dracoDecode,
    meshoptDecode: hooks.meshoptDecode,
  });
  if (!result.attached || preparedScenes.length !== 1) {
    throw new Error(`real glTF asset "${assetId}" did not exercise controller.attachEngine/setScene`);
  }
  if (result.controller.scene.primitives.length < (asset.expect.minPrimitives ?? 0)) {
    throw new Error(`real glTF asset "${assetId}" imported too few primitives`);
  }
  if ((result.textureDecodeReport.mapCount ?? 0) < (asset.expect.minTextures ?? 0)) {
    throw new Error(`real glTF asset "${assetId}" decoded too few material texture maps`);
  }
  if (result.textureDecodeDiagnostics.length > 0) {
    throw new Error(
      `real glTF asset "${assetId}" emitted texture decode diagnostics: ` +
      result.textureDecodeDiagnostics.map((d) => d.message ?? d.code ?? String(d)).join(" | "),
    );
  }
  const used = new Set(result.asset.featureReport.extensions.used);
  for (const extension of asset.expect.requiredExtensions ?? []) {
    if (!used.has(extension)) {
      throw new Error(`real glTF asset "${assetId}" did not report expected extension ${extension}`);
    }
  }
  const allowedWarnings = asset.expect.allowedWarningSubstrings ?? [];
  const unexpectedWarnings = result.warnings.filter((warning) =>
    !allowedWarnings.some((needle) => warning.includes(needle))
  );
  if (unexpectedWarnings.length > 0) {
    throw new Error(`real glTF asset "${assetId}" emitted loader warnings: ${unexpectedWarnings.join(" | ")}`);
  }
  const normalized = normalizeSceneForBehavioralGate(result.controller.scene);
  return {
    ...normalized,
    emitters: [
      ...(normalized.emitters ?? []),
      {
        kind: "rect-area", id: `gltf-real-gate-light-${assetId}`,
        position: [0, 0.95, 0.65],
        uAxis: [0.45, 0, 0], vAxis: [0, 0.45, 0],
        color: [1, 1, 1], intensity: 18.0,
      },
    ],
    environment: { kind: "procedural-sky", sunDirection: [0.4, 1.0, 0.25] },
  };
}

async function buildGateScene(opts = {}) {
  if (opts.gltfReal) return buildRealGltfAssetScene(opts.gltfReal);
  if (opts.gltf) return buildGltfFixtureScene(opts.gltf);
  return buildCornellScene(opts);
}

// ── Camera ────────────────────────────────────────────────────────────────────

const W = 64, H = 64;

function makePerspectiveMatrix(fovDeg, aspect, near, far) {
  const f  = 1.0 / Math.tan((fovDeg * Math.PI) / 180 / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function makeLookAtMatrix(eye, center, up) {
  const fx = center[0]-eye[0], fy = center[1]-eye[1], fz = center[2]-eye[2];
  const fL  = Math.hypot(fx, fy, fz);
  const fnx = fx/fL, fny = fy/fL, fnz = fz/fL;
  const sx = fny*up[2]-fnz*up[1], sy = fnz*up[0]-fnx*up[2], sz = fnx*up[1]-fny*up[0];
  const sL  = Math.hypot(sx, sy, sz);
  const snx = sx/sL, sny = sy/sL, snz = sz/sL;
  const ux  = sny*fnz-snz*fny, uy = snz*fnx-snx*fnz, uz = snx*fny-sny*fnx;
  return new Float32Array([
    snx, ux, -fnx, 0,
    sny, uy, -fny, 0,
    snz, uz, -fnz, 0,
    -(snx*eye[0]+sny*eye[1]+snz*eye[2]),
    -(ux *eye[0]+uy *eye[1]+uz *eye[2]),
     fnx*eye[0]+fny*eye[1]+fnz*eye[2],
    1,
  ]);
}

// pt-webgpu camera
const PT_EYE    = [-0.05, 0, 2.75];
const PT_CENTER = [-0.05, -0.15, 0];
const ptProj    = asMat4(makePerspectiveMatrix(40, W / H, 0.1, 50));
const ptView    = asMat4(makeLookAtMatrix(PT_EYE, PT_CENTER, [0,1,0]));

// walkaround camera
const WH_EYE    = [0, 0, 2.5];
const WH_CENTER = [0, 0, 0];
const whProj    = asMat4(makePerspectiveMatrix(60, W / H, 0.1, 50));
const whView    = asMat4(makeLookAtMatrix(WH_EYE, WH_CENTER, [0,1,0]));

// ── Naga gap patches (pt-webgpu) ──────────────────────────────────────────────
// Mirrors render-pt-webgpu.ts and the /tmp prototypes.

function stripBdptMipArg(wgsl) {
  const NEEDLE = "textureLoad(bdptLightPath,";
  let result = "", i = 0;
  while (i < wgsl.length) {
    const start = wgsl.indexOf(NEEDLE, i);
    if (start < 0) { result += wgsl.slice(i); break; }
    const openParen = start + "textureLoad".length;
    result += wgsl.slice(i, openParen + 1);
    let depth = 1, j = openParen + 1;
    const commas = [];
    let closeParen = -1;
    while (j < wgsl.length) {
      const ch = wgsl[j];
      if (ch === "(") depth++;
      else if (ch === ")") { depth--; if (depth === 0) { closeParen = j; break; } }
      else if (ch === "," && depth === 1) commas.push(j);
      j++;
    }
    if (closeParen < 0) { result += wgsl.slice(openParen + 1); break; }
    if (commas.length >= 2) {
      result += wgsl.slice(openParen + 1, commas[commas.length - 1]) + ")";
    } else {
      result += wgsl.slice(openParen + 1, closeParen) + ")";
    }
    i = closeParen + 1;
  }
  return result;
}

function addBdptMipArg(wgsl) {
  const NEEDLE = "textureLoad(bdptLightPath,";
  let result = "", i = 0;
  while (i < wgsl.length) {
    const start = wgsl.indexOf(NEEDLE, i);
    if (start < 0) { result += wgsl.slice(i); break; }
    const openParen = start + "textureLoad".length;
    result += wgsl.slice(i, openParen + 1);
    let depth = 1, j = openParen + 1;
    const commas = [];
    let closeParen = -1;
    while (j < wgsl.length) {
      const ch = wgsl[j];
      if (ch === "(") depth++;
      else if (ch === ")") { depth--; if (depth === 0) { closeParen = j; break; } }
      else if (ch === "," && depth === 1) commas.push(j);
      j++;
    }
    if (closeParen < 0) { result += wgsl.slice(openParen + 1); break; }
    if (commas.length === 1) {
      result += wgsl.slice(openParen + 1, closeParen) + ", 0)";
    } else {
      result += wgsl.slice(openParen + 1, closeParen) + ")";
    }
    i = closeParen + 1;
  }
  return result;
}

function applyPtNagaGapFix(wgsl, bdptOn) {
  let fixed = stripBdptMipArg(wgsl);
  if (fixed.includes("isNan(") || fixed.includes("isInf(")) {
    const helpers = `\nfn isNan(v: vec3f) -> vec3<bool> { return v != v; }\nfn isInf(v: vec3f) -> vec3<bool> { return abs(v) >= vec3f(1e38); }\n`;
    const idx = fixed.indexOf('\nfn ');
    if (idx > 0) fixed = fixed.slice(0, idx) + helpers + fixed.slice(idx);
  }
  if (!bdptOn && fixed.includes('texture_storage_2d<rgba32float, read_write>')) {
    fixed = fixed.replace('texture_storage_2d<rgba32float, read_write>', 'texture_2d<f32>');
    fixed = addBdptMipArg(fixed);
    fixed = fixed.replace(/textureStore\s*\(\s*bdptLightPath\s*,[^;]+;/g,
      '// naga-gap-fix: textureStore(bdptLightPath) removed');
  }
  return fixed;
}

function patchDeviceForPt(device, bdptOn) {
  const orig = device.createShaderModule.bind(device);
  device.createShaderModule = (desc) => {
    if (typeof desc.code === "string" && desc.code.includes("bdptLightPath")) {
      return orig({ ...desc, code: applyPtNagaGapFix(desc.code, bdptOn) });
    }
    return orig(desc);
  };
  return () => { device.createShaderModule = orig; };
}

function patchDeviceForWh(device) {
  const orig = device.createShaderModule.bind(device);
  device.createShaderModule = (desc) => {
    if (typeof desc.code === "string") {
      try { return orig({ ...desc, code: applyNagaFix(desc.code) }); }
      catch { return orig(desc); }
    }
    return orig(desc);
  };
}

// ── Device acquisition ────────────────────────────────────────────────────────

async function acquirePtDevice(wantsFullTier) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
  const limits = {};
  if (wantsFullTier) {
    const sb = adapter.limits.maxStorageBuffersPerShaderStage ?? 8;
    const st = adapter.limits.maxStorageTexturesPerShaderStage ?? 4;
    if (sb >= 28) limits.maxStorageBuffersPerShaderStage = sb;
    if (st >= 5)  limits.maxStorageTexturesPerShaderStage = st;
  }
  const bg = adapter.limits.maxBindGroups ?? 4;
  if (bg > 4) limits.maxBindGroups = bg;
  return adapter.requestDevice(Object.keys(limits).length ? { requiredLimits: limits } : {});
}

async function acquireWhDevice() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
  const limits = {};
  const sb = adapter.limits.maxStorageBuffersPerShaderStage ?? 8;
  const st = adapter.limits.maxStorageTexturesPerShaderStage ?? 4;
  if (sb >= 16) limits.maxStorageBuffersPerShaderStage = sb;
  if (st >= 8)  limits.maxStorageTexturesPerShaderStage = st;
  const bg = adapter.limits.maxBindGroups ?? 4;
  if (bg > 4) limits.maxBindGroups = bg;
  return adapter.requestDevice(Object.keys(limits).length ? { requiredLimits: limits } : {});
}

// ── Readback helpers ──────────────────────────────────────────────────────────

/**
 * Readback an rgba16float (or rgba32float) texture via a blit-to-rgba8unorm
 * render pass, then CPU readback.  Returns Uint8Array of RGBA pixels.
 */
async function readbackAsRgba8(device, srcTex, texW, texH) {
  const dstTex = device.createTexture({
    size: { width: texW, height: texH },
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    label: "bg-blit-dst",
  });
  const blitMod = device.createShaderModule({
    label: "bg-blit",
    code: `
      @group(0) @binding(0) var srcTex: texture_2d<f32>;
      @group(0) @binding(1) var srcSmp: sampler;
      struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
      @vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
        var p = array<vec2f,3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
        var o: VO;
        o.pos = vec4f(p[vi], 0.0, 1.0);
        o.uv  = (p[vi] + vec2f(1,1)) * 0.5;
        o.uv.y = 1.0 - o.uv.y;
        return o;
      }
      @fragment fn fs(i: VO) -> @location(0) vec4f {
        return vec4f(clamp(textureSample(srcTex, srcSmp, i.uv).rgb, vec3f(0), vec3f(1)), 1.0);
      }
    `,
  });
  const bgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
  ]});
  const pipeline = device.createRenderPipeline({
    label: "bg-blit-pipeline",
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    vertex:   { module: blitMod, entryPoint: "vs" },
    fragment: { module: blitMod, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });
  const bg = device.createBindGroup({ layout: bgl, entries: [
    { binding: 0, resource: srcTex.createView() },
    { binding: 1, resource: device.createSampler({ magFilter: "nearest", minFilter: "nearest" }) },
  ]});
  const enc  = device.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{
    view: dstTex.createView(), loadOp: "clear", storeOp: "store",
    clearValue: { r: 0, g: 0, b: 0, a: 1 },
  }]});
  pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
  device.queue.submit([enc.finish()]);

  const bpr  = Math.ceil(texW * 4 / 256) * 256;
  const buf  = device.createBuffer({ size: bpr * texH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc2 = device.createCommandEncoder();
  enc2.copyTextureToBuffer({ texture: dstTex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: texH }, { width: texW, height: texH, depthOrArrayLayers: 1 });
  device.queue.submit([enc2.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buf.getMappedRange());
  const pixels = new Uint8Array(texW * texH * 4);
  for (let row = 0; row < texH; row++) {
    pixels.set(mapped.subarray(row * bpr, row * bpr + texW * 4), row * texW * 4);
  }
  buf.unmap(); buf.destroy(); dstTex.destroy();
  return pixels;
}

/** Readback bgra8unorm texture (walkaround swap chain). Returns RGBA pixels. */
async function readbackBgra8(device, tex, texW, texH) {
  const bpr  = Math.ceil(texW * 4 / 256) * 256;
  const buf  = device.createBuffer({ size: bpr * texH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc  = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: texH }, { width: texW, height: texH, depthOrArrayLayers: 1 });
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buf.getMappedRange());
  const pixels = new Uint8Array(texW * texH * 4);
  for (let row = 0; row < texH; row++) {
    pixels.set(mapped.subarray(row * bpr, row * bpr + texW * 4), row * texW * 4);
  }
  buf.unmap(); buf.destroy();
  // swap B↔R (bgra → rgba)
  for (let i = 0; i < pixels.length; i += 4) {
    const b = pixels[i]; pixels[i] = pixels[i+2]; pixels[i+2] = b;
  }
  return pixels;
}

function meanLuminance(pixels) {
  let sum = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    sum += 0.2126 * (pixels[i]/255) + 0.7152 * (pixels[i+1]/255) + 0.0722 * (pixels[i+2]/255);
  }
  return sum / (pixels.length / 4);
}

function hasNaN(pixels) {
  for (let i = 0; i < pixels.length; i++) {
    if (Number.isNaN(pixels[i])) return true;
  }
  return false;
}

const REAL_GLTF_GOLDENS = {
  "pt/gltf-real-box-textured": {
    path: "tools/reference-renders/gltf-real-behavioral/pt-gltf-real-box-textured.png",
    maxRmse: 8.0,
    maxMeanAbs: 4.0,
    maxAbs: 48,
  },
  "pt/gltf-real-draco": {
    path: "tools/reference-renders/gltf-real-behavioral/pt-gltf-real-draco.png",
    maxRmse: 8.0,
    maxMeanAbs: 4.0,
    maxAbs: 48,
  },
  "pt/gltf-real-meshopt": {
    path: "tools/reference-renders/gltf-real-behavioral/pt-gltf-real-meshopt.png",
    maxRmse: 8.0,
    maxMeanAbs: 4.0,
    maxAbs: 48,
  },
};

function pngFromPixels(pixels, width, height) {
  const png = new PNG({ width, height });
  png.data = Buffer.from(pixels);
  return PNG.sync.write(png);
}

function comparePixels(candidate, baseline) {
  if (candidate.length !== baseline.length) {
    throw new Error(`pixel buffer length mismatch (${candidate.length} vs ${baseline.length})`);
  }
  let sumSq = 0;
  let sumAbs = 0;
  let maxAbs = 0;
  for (let i = 0; i < candidate.length; i++) {
    const delta = candidate[i] - baseline[i];
    const abs = Math.abs(delta);
    sumSq += delta * delta;
    sumAbs += abs;
    maxAbs = Math.max(maxAbs, abs);
  }
  const n = candidate.length;
  return {
    rmse: Math.sqrt(sumSq / n),
    meanAbs: sumAbs / n,
    maxAbs,
  };
}

async function compareOrUpdateGolden(label, pixels) {
  const golden = REAL_GLTF_GOLDENS[label];
  if (!golden) return null;

  if (updateGoldens) {
    await Deno.mkdir("tools/reference-renders/gltf-real-behavioral", { recursive: true });
    await Deno.writeFile(golden.path, pngFromPixels(pixels, W, H));
    return {
      pass: true,
      updated: true,
      path: golden.path,
      rmse: 0,
      meanAbs: 0,
      maxAbs: 0,
    };
  }

  let decoded;
  try {
    decoded = PNG.sync.read(Buffer.from(await Deno.readFile(golden.path)));
  } catch (error) {
    return {
      pass: false,
      path: golden.path,
      error: `missing/unreadable golden PNG: ${error.message}`,
    };
  }
  if (decoded.width !== W || decoded.height !== H) {
    return {
      pass: false,
      path: golden.path,
      error: `golden PNG size ${decoded.width}x${decoded.height} does not match gate size ${W}x${H}`,
    };
  }
  const metrics = comparePixels(pixels, decoded.data);
  const pass =
    metrics.rmse <= golden.maxRmse &&
    metrics.meanAbs <= golden.maxMeanAbs &&
    metrics.maxAbs <= golden.maxAbs;
  return {
    pass,
    path: golden.path,
    ...metrics,
    thresholds: {
      maxRmse: golden.maxRmse,
      maxMeanAbs: golden.maxMeanAbs,
      maxAbs: golden.maxAbs,
    },
  };
}

async function waitForEngineReady(engine, label, timeoutMs = 120_000) {
  const deadline = Date.now() + timeoutMs;
  while (true) {
    if (engine.state === "ready") return;
    if (engine.state === "error") throw new Error(`${label}: engine.state === 'error'`);
    await new Promise(r => setTimeout(r, 50));
    if (engine.state === "ready") return;
    if (engine.state === "error") throw new Error(`${label}: engine.state === 'error'`);
    if (Date.now() > deadline) {
      throw new Error(`${label}: engine init timeout after ${Math.round(timeoutMs / 1000)} s`);
    }
  }
}

// ── Result classification ─────────────────────────────────────────────────────

const SPP = 8;
const LUM_THRESHOLD = 0.005;

/**
 * Run the expectation check for a single config result.
 * Returns { pass: boolean, note: string }.
 */
function checkExpectation(label, rawStatus, lum, errCount, nans) {
  const entry = EXPECTATION_TABLE[label] ?? { expected: "ok" };
  if (entry.expected === "known-residual") {
    return {
      pass: true,
      note: `KNOWN-RESIDUAL (${entry.planItem ?? "?"}) — ${rawStatus} — ${entry.reason}`,
    };
  }
  // expected === 'ok'
  if (rawStatus === "OK") return { pass: true, note: "" };
  return { pass: false, note: `FAIL — expected OK, got ${rawStatus} (lum=${lum.toFixed(4)}, gpuErrs=${errCount}, nan=${nans})` };
}

// ── pt-webgpu runner ──────────────────────────────────────────────────────────

async function runPtConfig(label, engineOpts, sceneOpts) {
  const bdptOn    = engineOpts.bdpt === true;
  const isLite    = engineOpts.traceTier === "lite";
  let device;
  try {
    device = await acquirePtDevice(!isLite);
  } catch (e) {
    return { label, rawStatus: "BLOCKED", lum: 0, errCount: 0, nans: false, errorMsg: e.message };
  }

  let errCount = 0;
  device.pushErrorScope("validation");
  device.pushErrorScope("out-of-memory");

  const unpatch = patchDeviceForPt(device, bdptOn);
  let engine  = null;
  let pixels  = null;
  let errorMsg = null;

  try {
    engine = await createPTEngine_WebGPU({
      device,
      maxBounces: 3,
      maxSamplesPerPixel: SPP,
      ...engineOpts,
    });

    const scene = await buildGateScene(sceneOpts);
    engine.setScene(scene);

    let frameOutput = null;
    for (let frame = 0; frame < SPP; frame++) {
      // eslint-disable-next-line no-loss-of-precision -- LCG constants intentionally exceed f64 mantissa; >>> 0 truncates to uint32 anyway
      const seed = ((frame * 6364136223846793005 + 1442695040888963407) >>> 0);
      frameOutput = engine.renderFrame({
        viewMatrix: ptView,
        projMatrix: ptProj,
        cameraPosition: PT_EYE,
        viewport: { width: W, height: H, devicePixelRatio: 1 },
        frameIndex: frame, frameSeed: seed,
        quality: { samplesTarget: SPP, bounces: 3, resolutionFactor: 1 },
      });
      await device.queue.onSubmittedWorkDone();
      if (frameOutput?.isConverged) break;
    }

    if (frameOutput?.kind === "rendered") {
      pixels = await readbackAsRgba8(device, frameOutput.primaryRadiance, W, H);
    }
  } catch (e) {
    errorMsg = e.message;
  } finally {
    unpatch();
    try { engine?.dispose(); } catch { /* best-effort cleanup — ignore */ }
  }

  const oomErr = await device.popErrorScope();
  const valErr = await device.popErrorScope();
  const gpuErrorMsg = [
    ...(oomErr ? [`oom: ${oomErr.message}`] : []),
    ...(valErr ? [`validation: ${valErr.message}`] : []),
  ].join(" | ");
  if (oomErr) errCount++;
  if (valErr) errCount++;
  device.destroy();

  if (errorMsg) return { label, rawStatus: "ERROR", lum: 0, errCount, nans: false, errorMsg, gpuErrorMsg };
  if (!pixels)  return { label, rawStatus: "ERROR", lum: 0, errCount, nans: false, errorMsg: "no pixels", gpuErrorMsg };

  const nans = hasNaN(pixels);
  const lum  = meanLuminance(pixels);
  let golden = null;
  if (!nans && errCount === 0 && lum >= LUM_THRESHOLD) {
    golden = await compareOrUpdateGolden(label, pixels);
  }
  const rawStatus = nans ? "NaN"
    : (errCount > 0 ? "GPU-ERROR"
      : (lum < LUM_THRESHOLD ? "BLACK"
        : (golden && !golden.pass ? "GOLDEN-DELTA" : "OK")));
  return { label, rawStatus, lum, errCount, nans, gpuErrorMsg, golden };
}

// ── walkaround-hybrid runner ──────────────────────────────────────────────────

async function runWhConfig(label, engineOpts, sceneOpts) {
  let device;
  try {
    device = await acquireWhDevice();
  } catch (e) {
    return { label, rawStatus: "BLOCKED", lum: 0, errCount: 0, nans: false, errorMsg: e.message };
  }

  let errCount = 0;
  device.pushErrorScope("validation");
  device.pushErrorScope("out-of-memory");

  patchDeviceForWh(device);

  let engine   = null;
  let pixels   = null;
  let swapTex  = null;
  let errorMsg = null;

  try {
    engine = await createWalkaroundEngine_Hybrid({
      device,
      width:  W, height: H,
      primaryLightDir:       [0.3, -0.8, 0.5],
      primaryLightIntensity: 0.6,
      skyTint:               [0.5, 0.7, 1.0],
      skyIrradiance:         0.15,
      verbose:               false,
      ppgEnabled:            false,
      rcEnabled:             false,
      denoiser:              "atrous-variance",
      ...engineOpts,
    });

    const scene = await buildGateScene(sceneOpts);
    engine.setScene(scene);

    // Poll until ready (state machine: uninitialized → initializing → ready).
    await waitForEngineReady(engine, label);

    swapTex = device.createTexture({
      label: "bg-wh-swap",
      size:  [W, H, 1],
      format: "bgra8unorm",
      usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
    });
    const swapView = swapTex.createView();

    for (let fi = 0; fi < SPP; fi++) {
      engine.renderFrame({
        viewMatrix: whView,
        projMatrix: whProj,
        cameraPosition: WH_EYE,
        viewport: { width: W, height: H, devicePixelRatio: 1 },
        frameIndex: fi,
        frameSeed:  fi * 1664525 + 1013904223,
        swapChainView:   swapView,
        swapChainFormat: "bgra8unorm",
      });
      await device.queue.onSubmittedWorkDone();
    }

    pixels = await readbackBgra8(device, swapTex, W, H);
  } catch (e) {
    errorMsg = e.message;
  } finally {
    try { swapTex?.destroy(); } catch { /* best-effort cleanup — ignore */ }
    try { engine?.dispose();  } catch { /* best-effort cleanup — ignore */ }
  }

  const oomErr = await device.popErrorScope();
  const valErr = await device.popErrorScope();
  const gpuErrorMsg = [
    ...(oomErr ? [`oom: ${oomErr.message}`] : []),
    ...(valErr ? [`validation: ${valErr.message}`] : []),
  ].join(" | ");
  if (oomErr) errCount++;
  if (valErr) errCount++;
  device.destroy();

  if (errorMsg) return { label, rawStatus: "ERROR", lum: 0, errCount, nans: false, errorMsg, gpuErrorMsg };
  if (!pixels)  return { label, rawStatus: "ERROR", lum: 0, errCount, nans: false, errorMsg: "no pixels", gpuErrorMsg };

  const nans = hasNaN(pixels);
  const lum  = meanLuminance(pixels);
  const rawStatus = nans ? "NaN" : (errCount > 0 ? "GPU-ERROR" : (lum < LUM_THRESHOLD ? "BLACK" : "OK"));
  return { label, rawStatus, lum, errCount, nans, gpuErrorMsg };
}

// ── Self-test config (injected when --self-test) ───────────────────────────────
// Forces a BLACK result so the gate can verify it catches a failing config.
async function runSelfTestConfig() {
  return {
    label: "__self-test/always-black",
    rawStatus: "BLACK",
    lum: 0.0,
    errCount: 0,
    nans: false,
  };
}

function formatGolden(golden) {
  if (!golden) return "";
  if (golden.updated) return `golden=updated:${golden.path}`;
  if (golden.error) return `golden=FAIL ${golden.error}`;
  const threshold = golden.thresholds
    ? ` <=(${golden.thresholds.maxRmse.toFixed(1)},${golden.thresholds.maxMeanAbs.toFixed(1)},${golden.thresholds.maxAbs})`
    : "";
  return `golden=${golden.pass ? "ok" : "FAIL"} rmse=${golden.rmse.toFixed(3)} meanAbs=${golden.meanAbs.toFixed(3)} maxAbs=${golden.maxAbs}${threshold}`;
}

// ── Main ──────────────────────────────────────────────────────────────────────

console.log("=== behavioral-gate ===");
console.log(`ICD: ${Deno.env.get("VK_ICD_FILENAMES") ?? "(not set)"}`);
console.log(`Resolution: ${W}×${H}, SPP: ${SPP}`);
if (selfTest) console.log("Mode: --self-test");
if (updateGoldens) console.log("Mode: --update-goldens");
if (labelFilter) console.log(`Filter: ${labelFilter}`);
console.log("");

const results = [];
const ptConfigs = labelFilter ? PT_CONFIGS.filter((cfg) => cfg.label.includes(labelFilter)) : PT_CONFIGS;
const whConfigs = labelFilter ? WH_CONFIGS.filter((cfg) => cfg.label.includes(labelFilter)) : WH_CONFIGS;
if (labelFilter && ptConfigs.length + whConfigs.length === 0) {
  console.error(`No behavioral-gate configs matched --filter=${labelFilter}`);
  Deno.exit(1);
}

console.log("── pt-webgpu ──");
for (const cfg of ptConfigs) {
  const r = await runPtConfig(cfg.label, cfg.eng, cfg.scene);
  results.push(r);
  const { pass, note } = checkExpectation(r.label, r.rawStatus, r.lum, r.errCount, r.nans);
  const marker = pass ? "PASS" : "FAIL";
  const goldenDetail = formatGolden(r.golden);
  const detail = r.errorMsg
    ? `${r.rawStatus} | ${r.errorMsg.replace(/\n/g, " ").slice(0, 160)}`
    : r.gpuErrorMsg
      ? `${r.rawStatus} | lum=${r.lum.toFixed(4)} gpuErrs=${r.errCount} nan=${r.nans} | ${r.gpuErrorMsg.replace(/\n/g, " ").slice(0, 220)}`
    : `${r.rawStatus} | lum=${r.lum.toFixed(4)} gpuErrs=${r.errCount} nan=${r.nans}`;
  const detailWithGolden = goldenDetail ? `${detail} | ${goldenDetail}` : detail;
  console.log(`  ${marker} | ${r.label.padEnd(28)} | ${detailWithGolden}${note ? " | " + note : ""}`);
}

console.log("");
console.log("── walkaround-hybrid ──");
for (const cfg of whConfigs) {
  const r = await runWhConfig(cfg.label, cfg.eng, cfg.scene);
  results.push(r);
  const { pass, note } = checkExpectation(r.label, r.rawStatus, r.lum, r.errCount, r.nans);
  const marker = pass ? "PASS" : "FAIL";
  const detail = r.errorMsg
    ? `${r.rawStatus} | ${r.errorMsg.replace(/\n/g, " ").slice(0, 160)}`
    : r.gpuErrorMsg
      ? `${r.rawStatus} | lum=${r.lum.toFixed(4)} gpuErrs=${r.errCount} nan=${r.nans} | ${r.gpuErrorMsg.replace(/\n/g, " ").slice(0, 220)}`
    : `${r.rawStatus} | lum=${r.lum.toFixed(4)} gpuErrs=${r.errCount} nan=${r.nans}`;
  console.log(`  ${marker} | ${r.label.padEnd(28)} | ${detail}${note ? " | " + note : ""}`);
}

// ── Self-test mode ────────────────────────────────────────────────────────────

if (selfTest) {
  console.log("");
  console.log("── self-test ──");
  // Inject a config that always returns BLACK — it has no EXPECTATION_TABLE entry
  // so it defaults to { expected: 'ok' } → should FAIL the gate.
  const stResult = await runSelfTestConfig();
  const { pass: stPass } = checkExpectation(stResult.label, stResult.rawStatus, stResult.lum, 0, false);
  if (!stPass) {
    console.log("  PASS | __self-test/always-black correctly detected as FAIL");
  } else {
    console.error("  FAIL | __self-test/always-black was NOT detected (gate broken)");
    Deno.exit(1);
  }

  // Verify that all production results match their expectations.
  const prodFails = results.filter(r => {
    const { pass } = checkExpectation(r.label, r.rawStatus, r.lum, r.errCount, r.nans);
    return !pass;
  });
  if (prodFails.length > 0) {
    console.error("\n--self-test aborted: production configs also failed:");
    for (const r of prodFails) console.error(`  ${r.label}: ${r.rawStatus}`);
    Deno.exit(1);
  }
  console.log("  --self-test PASSED");
  Deno.exit(0);
}

// ── Summary + exit code ───────────────────────────────────────────────────────

console.log("");
const failures = results.filter(r => {
  const { pass } = checkExpectation(r.label, r.rawStatus, r.lum, r.errCount, r.nans);
  return !pass;
});
const residuals = results.filter(r => {
  const entry = EXPECTATION_TABLE[r.label] ?? { expected: "ok" };
  return entry.expected === "known-residual";
});

console.log(`=== summary: ${results.length} configs total, ${failures.length} failures, ${residuals.length} known-residuals ===`);

if (failures.length > 0) {
  console.error("\nFAILED configs:");
  for (const r of failures) {
    console.error(`  ${r.label}: ${r.rawStatus} (lum=${r.lum.toFixed(4)}, gpuErrs=${r.errCount})`);
  }
  Deno.exit(1);
}

Deno.exit(0);
