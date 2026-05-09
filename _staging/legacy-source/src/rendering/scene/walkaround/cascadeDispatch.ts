/**
 * Per-frame cascade compute scheduler (§4.3).
 *
 * Builds the TSL compute nodes (once), updates uniforms each frame,
 * and dispatches:
 *   1. Probe ray-cast for each cascade (C0 → C4)
 *   2. Cascade merge bottom-up (C3 → C0)
 *
 * If WebGPU is not the active backend (or `debugFill` is set explicitly),
 * fills all cascades with constant debug colours instead of dispatching the
 * compute pipeline — keeps the GI receiver shading path live for fallback
 * environments / smoke testing without a real WebGPU adapter.
 *
 * The TSL compute graph has a complex type system; we use `any` at integration
 * boundaries where the three.js TSL type definitions lag the runtime.
 */

import { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import * as THREE from 'three';
import { storage, sampler, texture as textureNode, instanceIndex } from 'three/tsl';
import type { SceneBVH } from './bvhCompute';
import { CASCADE_DIMS, CASCADE_COUNT, fillCascadeDebug } from './cascadePyramid';
import type { CascadeBuffers } from './cascadePyramid';
import { probeRayCastKernel } from '../../shaders/walkaround/probeRayCast.wgsl';
import { cascadeMergeKernel } from '../../shaders/walkaround/cascadeMerge.wgsl';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyNode = any;

interface CascadePassHandle {
  computeNode:   AnyNode;
  uCascadeU:     AnyNode;
  /** Raw backing array so we can update uniforms per-frame without re-binding. */
  uniformRaw:    Float32Array;
}

interface MergePassHandle {
  computeNode:   AnyNode;
  uMerge:        AnyNode;
  /** Raw backing array for per-frame updates. */
  mergeRaw:      Float32Array;
}

interface DispatchHandles {
  castPasses:  CascadePassHandle[];
  mergePasses: MergePassHandle[];
}

export interface DispatchOpts {
  gl:             WebGPURenderer;
  sceneBVH:       SceneBVH;
  cascadeBuffers: CascadeBuffers;
  sunDirection:   THREE.Vector3;
  sunColor:       THREE.Color;
  envEquirect:    THREE.Texture | null;
  frameSeed:      number;
  /** Smoke-test / fallback mode: fill cascades with debug colours, skip ray-cast. */
  debugFill?:     boolean;
}

/**
 * Build all TSL compute nodes for the cascade pipeline (one-time setup).
 * Call once; cache on the renderer.
 */
function buildCascadeDispatch(
  sceneBVH:       SceneBVH,
  cascadeBuffers: CascadeBuffers,
  envEquirect:    THREE.Texture | null,
): DispatchHandles {
  // Env fallback.
  const fallbackEnv = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  fallbackEnv.needsUpdate = true;
  const envTex = envEquirect ?? fallbackEnv;

  // BVH read-only storage nodes.
  const bvhStorage  = storage(sceneBVH.bvhNodes,  'BVHNode',        sceneBVH.bvhNodes.count).toReadOnly();
  const idxStorage  = storage(sceneBVH.indices,   'vec3u',          sceneBVH.indices.count).toReadOnly();
  const posStorage  = storage(sceneBVH.positions, 'vec3f',          sceneBVH.positions.count).toReadOnly();
  const matStorage  = storage(sceneBVH.materials, 'MaterialEntry',  sceneBVH.materials.count).toReadOnly();
  const triMStorage = storage(sceneBVH.triMaterialId, 'u32',        sceneBVH.triMaterialId.count).toReadOnly();

  // Env texture + sampler nodes.
  // texture() wraps the THREE.Texture as a TSL TextureNode (needed for wgslFn params).
  // sampler() converts to a WGSL sampler node.
  const envTexNode     = textureNode(envTex);
  const envSamplerNode = sampler(envTex);

  // Build cast passes (one per cascade).
  const castPasses: CascadePassHandle[] = CASCADE_DIMS.map((dim, k) => {
    // Per-cascade output storage (writable).
    // Use the shared GPU attr from cascadeBuffers.gpuCascades so the lighting
    // node (which also uses cascadeBuffers.gpuCascades[0]) reads from the same
    // GPU buffer that the compute writes into.
    const cascadeAttr = cascadeBuffers.gpuCascades[k]!;
    const cascadeStorage = storage(cascadeAttr, 'vec4f', cascadeAttr.count);

    // CascadeUniforms as a read-only storage buffer (array<CascadeUniforms> of length 1).
    // We use storage instead of uniform because TSL's uniform() only handles primitive types;
    // custom struct uniforms throw "not declared". Storage buffers handle struct types fine.
    // Layout: see CascadeUniforms struct in probeRayCast.wgsl.ts (40 floats = 160 bytes).
    const uniformData = buildCascadeUniformData(k, cascadeBuffers, new THREE.Vector3(0, 1, 0), new THREE.Color(1, 1, 1), 1.0, 0);
    const uniformAttr = new StorageBufferAttribute(uniformData, 40);  // 40 floats per element, 1 element
    const uCascadeU = storage(uniformAttr, 'CascadeUniforms', 1).toReadOnly();

    const totalRays = dim.probes[0] * dim.probes[1] * dim.probes[2] * dim.rays;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const computeNode = (probeRayCastKernel as any)({
      bvh:           bvhStorage,
      geom_index:    idxStorage,
      geom_position: posStorage,
      materials:     matStorage,
      triMatId:      triMStorage,
      cascadeOut:    cascadeStorage,
      envMap:        envTexNode,
      envSampler:    envSamplerNode,
      u_arr:         uCascadeU,
      // instanceIndex is the TSL builtin for @builtin(global_invocation_id).x
      // wgslFn parameter matching: FunctionCallNode looks up each declared param by name.
      index:         instanceIndex,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    }).compute(totalRays, [64] as any);

    return { computeNode, uCascadeU, uniformRaw: uniformData };
  });

  // Build merge passes (bottom-up: C(N-2) → C0).
  const mergePasses: MergePassHandle[] = [];
  for (let lower = CASCADE_COUNT - 2; lower >= 0; lower--) {
    const lowerDim = CASCADE_DIMS[lower]!;
    const upperDim = CASCADE_DIMS[lower + 1]!;

    const lowerAttr = cascadeBuffers.gpuCascades[lower]!;
    const upperAttr = cascadeBuffers.gpuCascades[lower + 1]!;

    const lowerStorage = storage(lowerAttr, 'vec4f', lowerAttr.count);
    const upperStorage = storage(upperAttr, 'vec4f', upperAttr.count).toReadOnly();

    const mergeData  = buildMergeUniformData(lowerDim, upperDim, cascadeBuffers);
    // MergeUniforms as read-only storage buffer (same reason as CascadeUniforms — TSL
    // uniform() only handles primitive types; struct types need storage buffer approach).
    const mergeAttr  = new StorageBufferAttribute(mergeData, 20);  // 20 floats per element, 1 element
    const uMerge     = storage(mergeAttr, 'MergeUniforms', 1).toReadOnly();

    const totalLower = lowerDim.probes[0] * lowerDim.probes[1] * lowerDim.probes[2] * lowerDim.rays;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const computeNode = (cascadeMergeKernel as any)({
      upperCascade: upperStorage,
      lowerCascade: lowerStorage,
      m_arr:        uMerge,
      index:        instanceIndex,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    }).compute(totalLower, [64] as any);

    mergePasses.push({ computeNode, uMerge, mergeRaw: mergeData });
  }

  return { castPasses, mergePasses };
}

/** Pack CascadeUniforms struct into a new Float32Array (used at build time). */
function buildCascadeUniformData(
  k: number,
  cb: CascadeBuffers,
  sunDir: THREE.Vector3,
  sunColor: THREE.Color,
  envIntensity: number,
  frameSeed: number,
): Float32Array {
  const d = new Float32Array(40);
  buildCascadeUniformDataInto(d, k, cb, sunDir, sunColor, envIntensity, frameSeed);
  return d;
}

/** Write CascadeUniforms into an existing Float32Array (used per-frame to avoid realloc). */
function buildCascadeUniformDataInto(
  d: Float32Array,
  k: number,
  cb: CascadeBuffers,
  sunDir: THREE.Vector3,
  sunColor: THREE.Color,
  envIntensity: number,
  frameSeed: number,
): void {
  const dim = CASCADE_DIMS[k]!;
  const rayGridSize = Math.round(Math.sqrt(dim.rays));
  const { probeOriginWorld: o, roomSize: s } = cb;
  // CascadeUniforms layout (matches WGSL struct):
  // probeOriginWorld(3f), _pad0(f)
  // roomSize(3f), _pad1(f)
  // probeCount(3u), raysPerProbe(u)
  // rayGridSize(u), intervalNear(f), intervalFar(f), cascadeIndex(u)
  // sunDirection(3f), _pad2(f)
  // sunColor(3f), envIntensity(f)
  // frameSeed(u), lastCascade(u), _pad4(2u)
  // Total: 40 float/uint values = 160 bytes
  const ui = new Uint32Array(d.buffer);
  d[0]  = o.x; d[1]  = o.y; d[2]  = o.z; d[3]  = 0;
  d[4]  = s.x; d[5]  = s.y; d[6]  = s.z; d[7]  = 0;
  ui[8] = dim.probes[0]; ui[9] = dim.probes[1]; ui[10] = dim.probes[2];
  ui[11] = dim.rays;
  ui[12] = rayGridSize;
  d[13] = dim.intervalNear; d[14] = dim.intervalFar;
  ui[15] = k;
  d[16] = sunDir.x; d[17] = sunDir.y; d[18] = sunDir.z; d[19] = 0;
  d[20] = sunColor.r; d[21] = sunColor.g; d[22] = sunColor.b;
  d[23] = envIntensity;
  ui[24] = frameSeed;
  ui[25] = CASCADE_COUNT - 1;
  ui[26] = 0; ui[27] = 0;
}

/** Pack MergeUniforms struct into a Float32Array. */
function buildMergeUniformData(
  lowerDim: (typeof CASCADE_DIMS)[number],
  upperDim: (typeof CASCADE_DIMS)[number],
  cb: CascadeBuffers,
): Float32Array {
  // MergeUniforms layout (matches WGSL struct):
  // lowerProbeCount(3u), lowerRayCount(u)
  // upperProbeCount(3u), upperRayCount(u)
  // lowerRayGridSize(u), upperRayGridSize(u), _pad0(2u)
  // probeOriginWorld(3f), _pad1(f)
  // roomSize(3f), _pad2(f)
  // Total: 20 float/uint values = 80 bytes
  const d = new Float32Array(20);
  const ui = new Uint32Array(d.buffer);
  const { probeOriginWorld: o, roomSize: s } = cb;
  ui[0]  = lowerDim.probes[0]; ui[1]  = lowerDim.probes[1]; ui[2]  = lowerDim.probes[2];
  ui[3]  = lowerDim.rays;
  ui[4]  = upperDim.probes[0]; ui[5]  = upperDim.probes[1]; ui[6]  = upperDim.probes[2];
  ui[7]  = upperDim.rays;
  ui[8]  = Math.round(Math.sqrt(lowerDim.rays));
  ui[9]  = Math.round(Math.sqrt(upperDim.rays));
  ui[10] = 0; ui[11] = 0;
  d[12]  = o.x; d[13] = o.y; d[14] = o.z; d[15] = 0;
  d[16]  = s.x; d[17] = s.y; d[18] = s.z; d[19] = 0;
  return d;
}

/**
 * Dispatch the cascade compute pipeline for one frame.
 */
export async function dispatchCascadePasses(opts: DispatchOpts): Promise<void> {
  const { gl, sceneBVH, cascadeBuffers, sunDirection, sunColor, envEquirect, frameSeed } = opts;

  if (opts.debugFill) {
    // Explicit debug-fill request: skip ray-cast, fill with constants.
    fillCascadeDebug(cascadeBuffers);
    // The StorageBufferAttributes wrap the same Float32Arrays that
    // fillCascadeDebug just wrote, but their GPU backing buffers only
    // sync when needsUpdate=true. Without this the GPU side keeps the
    // last-uploaded content (zeros on first frame, last real compute
    // result thereafter), defeating the smoke-test path.
    for (const attr of cascadeBuffers.gpuCascades) attr.needsUpdate = true;
    return;
  }

  // Guard: compute dispatch requires a real WebGPU backend.
  // WebGPURenderer can auto-fall-back to WebGL; in that case GLSLNodeBuilder
  // is used and it cannot compile our WGSL-specific ptr<storage,...> syntax.
  // Fall back to debug-fill silently when WebGPU is not available.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const anyGl = gl as any;
  if (anyGl.backend?.isWebGPUBackend !== true) {
    fillCascadeDebug(cascadeBuffers);
    for (const attr of cascadeBuffers.gpuCascades) attr.needsUpdate = true;
    return;
  }

  // Lazy-build dispatch handles.
  const glAny = gl as AnyNode;
  if (!glAny._cascadeHandles) {
    try {
      glAny._cascadeHandles = buildCascadeDispatch(sceneBVH, cascadeBuffers, envEquirect);
    } catch (err: unknown) {
      console.error('[RC] buildCascadeDispatch failed:', err);
      return;
    }
  }
  const handles = glAny._cascadeHandles as DispatchHandles;

  // Update per-frame uniforms for each cascade.
  // Storage buffer approach: write directly into the backing Float32Array,
  // then mark the StorageBufferAttribute needsUpdate so the GPU re-uploads.
  for (let k = 0; k < CASCADE_COUNT; k++) {
    const handle = handles.castPasses[k]!;
    buildCascadeUniformDataInto(
      handle.uniformRaw, k, cascadeBuffers, sunDirection, sunColor, 1.0, frameSeed,
    );
    // uCascadeU is a StorageBufferNode; .value is the StorageBufferAttribute.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const sbAttr = (handle.uCascadeU as any).value as StorageBufferAttribute | undefined;
    if (sbAttr) sbAttr.needsUpdate = true;
  }

  // Dispatch cast passes (C0 → C4).
  for (let k = 0; k < CASCADE_COUNT; k++) {
    try {
      await gl.computeAsync(handles.castPasses[k]!.computeNode);
    } catch (err: unknown) {
      console.error(`[RC] cast pass C${k} dispatch error:`, err);
      return;  // don't attempt further passes if one fails
    }
  }

  // Dispatch merge passes bottom-up (C3 → C0).
  for (let k = 0; k < handles.mergePasses.length; k++) {
    try {
      await gl.computeAsync(handles.mergePasses[k]!.computeNode);
    } catch (err: unknown) {
      console.error(`[RC] merge pass ${k} dispatch error:`, err);
      return;
    }
  }
}
