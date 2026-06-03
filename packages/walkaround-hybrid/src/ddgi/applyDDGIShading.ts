/**
 * applyDDGIShading — walks the scene and injects DDGI diffuse indirect
 * into every visible Mesh material via `material.outputNode`.
 *
 * Pattern:
 *   material.outputNode =
 *     renderOutput(add(output, vec4(albedo * ddgiIrradiance, 0.0)),
 *                  ACESFilmicToneMapping, SRGBColorSpace)
 *
 * `output` is the built-in TSL property node holding the standard PBR-lit
 * output (direct + env-IBL). Adding to it preserves direct lighting while
 * injecting the DDGI diffuse indirect on top. The `renderOutput()` wrapper
 * applies ACES tone mapping + sRGB encoding inside the fragment shader because
 * the WebGPU renderer is pinned to NoToneMapping + LinearSRGBColorSpace at
 * the renderer level (see OUTPUT_TONE_MAPPING comment below for the
 * format-mismatch workaround that requires this).
 *
 * Extracted from `_staging/legacy-source/src/rendering/scene/walkaround/applyDDGIShading.ts`.
 * TSL imports preserved per extraction plan Option (i) — this is a Three.js
 * NodeMaterial customization hook, not a standalone compute kernel.
 * Requires `three/webgpu` + `three/tsl` as peer deps.
 */

import * as THREE from 'three';
import type { MeshPhysicalNodeMaterial, MeshStandardNodeMaterial } from 'three/webgpu';
import { StorageTexture } from 'three/webgpu';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyNode = any;
import { add, vec4, mul, output, materialColor, uniform, texture, positionWorld, normalWorld, wgslFn, renderOutput } from 'three/tsl';
import type { ProbeGrid, AtlasTextureSlot } from './probeGrid.js';
import { DDGI_SAMPLE_WGSL } from './ddgiSampleWgsl.js';
import { upgradeToNodeMaterial } from '../lib/nodeMaterialUpgrade.js';

// TSL binds textures via three.js Texture handles. ProbeGrid exposes
// backend-agnostic AtlasTextureSlot records (just width/height) so the
// compute path (probeUpdatePass) doesn't need to touch three/webgpu.
// This site is the TSL boundary — we wrap each slot in a StorageTexture
// here, cached by slot identity so TSL keeps its bindings stable across
// frames. The wrappers themselves carry no GPU data; three.js's WebGPU
// backend manages whatever GPUTexture it allocates for them.
//
// The parallel `_trackedStorageTextures` Set exists so `disposeApply-
// DDGIShadingCache()` can call `.dispose()` on each StorageTexture —
// dropping the WeakMap alone does NOT free the underlying GPU resource;
// the StorageTexture's `dispose()` method is what releases it.
let _slotStorageTextureCache = new WeakMap<AtlasTextureSlot, StorageTexture>();
const _trackedStorageTextures = new Set<StorageTexture>();
function slotToStorageTexture(slot: AtlasTextureSlot): StorageTexture {
  let tex = _slotStorageTextureCache.get(slot);
  if (!tex) {
    tex = new StorageTexture(slot.width, slot.height);
    tex.format = THREE.RGBAFormat;
    tex.type = THREE.HalfFloatType;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    _slotStorageTextureCache.set(slot, tex);
    _trackedStorageTextures.add(tex);
  }
  return tex;
}

/** Dispose all StorageTexture wrappers cached by `slotToStorageTexture`
 *  and reset the module-level caches. Call this from the host's engine
 *  teardown to release the underlying GPU resources — WeakMap-based GC
 *  is insufficient because StorageTexture owns a GPUTexture that only
 *  three.js's `.dispose()` call frees.
 *
 *  Idempotent. */
export function disposeApplyDDGIShadingCache(): void {
  for (const tex of _trackedStorageTextures) {
    try { tex.dispose(); } catch {}
  }
  _trackedStorageTextures.clear();
  _slotStorageTextureCache = new WeakMap();
  _ddgiSampleFnCached = null;
  _injectedMaterials = new WeakMap();
}

// WebGPU is configured with NoToneMapping + LinearSRGBColorSpace at the
// renderer level to keep Three's internal HDR `_frameBufferTarget` from
// engaging — that target triggers a copyFramebufferToTexture format mismatch
// (rgba16float source vs bgra8unorm dest) inside the WebGPU transmission
// compositor's `viewportMipTexture()` call, crashing every transmissive triangle.
//
// Fix: tone-map + sRGB-encode INSIDE each material's `outputNode` via TSL's
// `renderOutput()`. The renderer's HDR framebuffer stays disabled (transmission
// compositor safe), but the per-pixel output reaching the canvas is now properly
// gamma-encoded sRGB with ACES highlight roll-off.
const OUTPUT_TONE_MAPPING = THREE.ACESFilmicToneMapping;
const OUTPUT_COLOR_SPACE  = THREE.SRGBColorSpace;

// Singleton wgslFn for the DDGI sample function — created once, shared
// across all materials.
let _ddgiSampleFnCached: ReturnType<typeof wgslFn> | null = null;
function getDDGISampleFn(): ReturnType<typeof wgslFn> {
  if (!_ddgiSampleFnCached) {
    _ddgiSampleFnCached = wgslFn(DDGI_SAMPLE_WGSL);
  }
  return _ddgiSampleFnCached;
}

interface InjectedEntry {
  original: THREE.Material;
  upgraded: MeshPhysicalNodeMaterial | MeshStandardNodeMaterial;
}
let _injectedMaterials = new WeakMap<THREE.Mesh, InjectedEntry>();

export function applyDDGIShading(
  scene: THREE.Scene,
  probeGrid: ProbeGrid,
  enabled = true,
): void {
  if (!probeGrid.irradianceA || !probeGrid.irradianceB) return;

  const irrTex = slotToStorageTexture(probeGrid.irradianceReadTex);
  const visTex = slotToStorageTexture(probeGrid.visibilityReadTex);
  const p = probeGrid.params;

  scene.traverse((obj) => {
    if (!(obj instanceof THREE.Mesh)) return;
    const originalMat = obj.material as THREE.Material;
    if (!originalMat) return;

    if (_injectedMaterials.has(obj) && enabled) return;

    if (!enabled) {
      const entry = _injectedMaterials.get(obj);
      if (entry) {
        obj.material = entry.original;
        entry.upgraded.dispose();
        _injectedMaterials.delete(obj);
      }
      return;
    }

    const nodeMat = upgradeToNodeMaterial(originalMat);
    if (!nodeMat) return;

    obj.material = nodeMat;

    const sampleFn = getDDGISampleFn();

    const oX = uniform(p.origin.x);
    const oY = uniform(p.origin.y);
    const oZ = uniform(p.origin.z);
    const sp = uniform(p.spacing);
    const dx = uniform(p.dims.x);
    const dy = uniform(p.dims.y);
    const dz = uniform(p.dims.z);
    const iW = uniform(p.irradianceAtlasW);
    const iH = uniform(p.irradianceAtlasH);
    const vW = uniform(p.visibilityAtlasW);
    const vH = uniform(p.visibilityAtlasH);

    const irrTexNode = texture(irrTex);
    const visTexNode = texture(visTex);

    // DDGI irradiance sample at this fragment.
    const ddgiIrr = sampleFn(
      positionWorld, normalWorld,
      irrTexNode, visTexNode, irrTexNode.sampler,
      oX, oY, oZ, sp,
      dx, dy, dz,
      iW, iH, vW, vH,
    );

    // Lambertian receiver: outgoing diffuse from indirect = (albedo/π) · E_ddgi.
    // Atlas holds true irradiance E per Sweep M7 — albedo/π baking moved from
    // producer (probeUpdateRays.wgsl.ts) to consumer here.
    // Receiver equation: L_o_indirect = (albedo / π) · E_ddgi
    // Reference: Majercik 2019 §3; M7 DDGI Coherent Physical Model.
    const PI_INV = uniform(1.0 / Math.PI);
    // AnyNode casts below are needed because three/tsl typings are conservative
    // and don't model wgslFn return types precisely.
    const ddgiContrib = mul(ddgiIrr, mul(materialColor as AnyNode, PI_INV));

    const linearOutput = add(output, vec4(ddgiContrib as AnyNode, 0.0));
    (nodeMat as MeshPhysicalNodeMaterial & { outputNode: unknown }).outputNode =
      renderOutput(linearOutput, OUTPUT_TONE_MAPPING, OUTPUT_COLOR_SPACE);
    nodeMat.needsUpdate = true;
    _injectedMaterials.set(obj, { original: originalMat, upgraded: nodeMat });
  });
}
