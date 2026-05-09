/**
 * applyDDGIShading — walks the scene and injects DDGI diffuse indirect
 * into every visible Mesh material via `material.outputNode`.
 *
 * Pattern (§16.1 add-pattern):
 *   material.outputNode =
 *     renderOutput(add(output, vec4(albedo * ddgiIrradiance, 0.0)),
 *                  ACESFilmicToneMapping, SRGBColorSpace)
 *
 * `output` is the built-in TSL property node that holds the standard
 * PBR-lit output (direct + env-IBL). Adding to it preserves direct
 * lighting while injecting the DDGI diffuse indirect on top. The
 * `renderOutput()` wrapper applies ACES tone mapping + sRGB encoding
 * inside the fragment shader because the WebGPU renderer is pinned to
 * NoToneMapping + LinearSRGBColorSpace at the renderer level (see
 * OUTPUT_TONE_MAPPING comment below for the format-mismatch
 * workaround that requires this).
 *
 * Called from WalkaroundStage useEffect whenever the scene or DDGI
 * refs change.
 */

import * as THREE from 'three';
import type { MeshPhysicalNodeMaterial, MeshStandardNodeMaterial } from 'three/webgpu';
import { add, vec4, mul, output, materialColor, uniform, texture, positionWorld, normalWorld, wgslFn, renderOutput } from 'three/tsl';
import type { ProbeGrid } from './probeGrid';
import { DDGI_SAMPLE_WGSL } from './ddgiSampleWgsl';
import { upgradeToNodeMaterial } from './lib/nodeMaterialUpgrade';

// WebGPU is configured with NoToneMapping + LinearSRGBColorSpace at the
// renderer level (StudioScene's `flat` + `linear` Canvas props) to keep
// Three's internal HDR `_frameBufferTarget` from engaging — that target
// triggers a copyFramebufferToTexture format mismatch (rgba16float source
// vs bgra8unorm dest) inside the WebGPU transmission compositor's
// `viewportMipTexture()` call, crashing every transmissive triangle.
//
// The cost of leaving the renderer in linear/no-tone-map: scene output
// reaches the canvas as raw linear-sRGB radiance. On a gamma-encoded
// display this looks dramatically darker than expected — DDGI's bounce
// signal is real but invisible to the eye.
//
// Fix: tone-map + sRGB-encode INSIDE each material's `outputNode` (the
// fragment shader's final stage) via TSL's `renderOutput()`. The
// renderer's HDR framebuffer stays disabled (transmission compositor
// safe), but the per-pixel output reaching the canvas is now properly
// gamma-encoded sRGB with ACES highlight roll-off. Almost every visible
// surface in the walkaround/explore scene is MeshStandard or
// MeshPhysical (room walls/floor/ceiling, frame mouldings, glass
// faces) — those get DDGI injection AND tone mapping in this pass. The
// drei <Sky> / <Environment> backdrops use their own
// RawShaderMaterial / NodeMaterial paths and are not tone-mapped here;
// that's a known v1 trade-off (see WalkaroundStage's Sky comment).
const OUTPUT_TONE_MAPPING = THREE.ACESFilmicToneMapping;
const OUTPUT_COLOR_SPACE  = THREE.SRGBColorSpace;

// Singleton wgslFn for the DDGI sample function — created once, shared
// across all materials. DDGI_SAMPLE_WGSL is a single self-contained
// `ddgiSample` function (see that file's header for why no helpers).
let _ddgiSampleFnCached: ReturnType<typeof wgslFn> | null = null;
function getDDGISampleFn(): ReturnType<typeof wgslFn> {
  if (!_ddgiSampleFnCached) {
    _ddgiSampleFnCached = wgslFn(DDGI_SAMPLE_WGSL);
  }
  return _ddgiSampleFnCached;
}

/** Weak map of mesh → { original material we replaced, upgraded NodeMaterial
 *  we built }. Keyed on the mesh because we may swap `mesh.material` and
 *  need to remember what to restore on `enabled=false`. */
interface InjectedEntry {
  original: THREE.Material;
  upgraded: MeshPhysicalNodeMaterial | MeshStandardNodeMaterial;
}
const _injectedMaterials = new WeakMap<THREE.Mesh, InjectedEntry>();

export function applyDDGIShading(
  scene: THREE.Scene,
  probeGrid: ProbeGrid,
  enabled = true,
): void {
  if (!probeGrid.irradianceA || !probeGrid.irradianceB) return;

  const irrTex = probeGrid.irradianceReadTex as unknown as THREE.Texture;
  const visTex = probeGrid.visibilityReadTex as unknown as THREE.Texture;
  const p = probeGrid.params;

  scene.traverse((obj) => {
    if (!(obj instanceof THREE.Mesh)) return;
    const originalMat = obj.material as THREE.Material;
    if (!originalMat) return;

    // If already injected, nothing to do (enabled path) — or we restored
    // the material on a prior `enabled=false` pass and there is nothing
    // left to undo.
    if (_injectedMaterials.has(obj) && enabled) return;

    if (!enabled) {
      // Disable path — restore the original material if we swapped it.
      const entry = _injectedMaterials.get(obj);
      if (entry) {
        obj.material = entry.original;
        entry.upgraded.dispose();
        _injectedMaterials.delete(obj);
      }
      return;
    }

    // Upgrade to a NodeMaterial sibling. Three.js's WebGPU renderer does
    // this internally (see NodeLibrary.fromMaterial) but does NOT replace
    // mesh.material, so `outputNode` set on the user's material never
    // reaches the shader. We swap explicitly here so DDGI injection takes
    // effect. Materials that aren't phys/standard (e.g. MeshBasicMaterial
    // used by background helpers) skip without error.
    const nodeMat = upgradeToNodeMaterial(originalMat);
    if (!nodeMat) return;

    // Swap the mesh's material. Future renderer passes pick up the new
    // material identity through their own cache (RenderObjects keys on
    // material.version + identity).
    obj.material = nodeMat;

    const sampleFn = getDDGISampleFn();

    // Build uniform nodes for grid params.
    const oX = uniform(p.origin.x);
    const oY = uniform(p.origin.y);
    const oZ = uniform(p.origin.z);
    const sp = uniform(p.spacing);
    // Use f32 for dims to avoid u32 type mismatch in TSL → will be cast inside WGSL.
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

    // Diffuse indirect = irradiance × albedo / π (Lambertian BRDF).
    const PI_INV = uniform(1.0 / Math.PI);
    const ddgiContrib = mul(ddgiIrr, mul(materialColor, PI_INV));

    // Add DDGI on top of existing output (preserves direct lighting),
    // then tone-map + sRGB-encode the linear result so the canvas
    // receives gamma-correct pixels even though the renderer itself is
    // pinned to NoToneMapping + LinearSRGBColorSpace (see header
    // comment for why the renderer can't carry the post-transform).
    const linearOutput = add(output, vec4(ddgiContrib, 0.0));
    (nodeMat as MeshPhysicalNodeMaterial & { outputNode: unknown }).outputNode =
      renderOutput(linearOutput, OUTPUT_TONE_MAPPING, OUTPUT_COLOR_SPACE);
    nodeMat.needsUpdate = true;
    _injectedMaterials.set(obj, { original: originalMat, upgraded: nodeMat });
  });
}

