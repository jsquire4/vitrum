/**
 * GIReceiver — wraps receiver meshes with the walkaround GI material node.
 *
 * Converts receiver meshes (walls, floor, ceiling, fixture bodies) to
 * MeshPhysicalNodeMaterial and adds the walkaround GI contribution to their
 * lightingNode.
 *
 * Glass meshes (userData.glassPiece === true) are NOT receivers — they
 * contribute light through transmission but don't receive cascade GI.
 *
 * Extracted from `_staging/legacy-source/src/rendering/scene/walkaround/giReceiver.ts`.
 * De-React-ified: `useGIReceiverConverter` hook (useEffect, useRef, useThree) replaced
 * with a `GIReceiver` class exposing `wrap(scene, cascadeBuffers)` and `unwrap()`.
 * TSL imports (`MeshPhysicalNodeMaterial`, `output`, `renderOutput`) preserved per
 * extraction plan Option (i) — these are Three.js NodeMaterial customization hooks.
 * Requires `three/webgpu` + `three/tsl` as peer deps.
 */

import * as THREE from 'three';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import { output, renderOutput } from 'three/tsl';
import type { CascadeBuffers } from './cascadePyramid.js';
import { buildWalkaroundLightingNode } from './walkaroundDiffuseLighting.js';

// WebGPU is pinned to NoToneMapping + LinearSRGBColorSpace at the renderer level
// to avoid the WebGPU transmission compositor format mismatch (rgba16float source
// vs bgra8unorm dest inside `viewportMipTexture()`).
//
// Fix: tone-map + sRGB-encode INSIDE each receiver material's `outputNode` via
// TSL's `renderOutput()`. Cross-port from DDGI (applyDDGIShading.ts) and ReSTIR
// (composite.wgsl.ts acesFilm fit). Same Narkowicz 2015 ACES fit all three engines use.
const OUTPUT_TONE_MAPPING = THREE.ACESFilmicToneMapping;
const OUTPUT_COLOR_SPACE  = THREE.SRGBColorSpace;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyNode = any;

/** Tag applied to materials that have been wrapped with GI. */
const GI_TAG = Symbol('walkaroundGI');

interface GIMaterial {
  mat: MeshPhysicalNodeMaterial;
  originalMat: THREE.Material;
  [GI_TAG]?: boolean;
}

/**
 * Predicate for skipping a mesh when wrapping the scene with GI receivers.
 * Library consumers pass this to `GIReceiver` via `isExcluded` to opt
 * specific meshes out of GI without forking this file.
 */
export type GIReceiverExclusionPredicate = (mesh: THREE.Mesh) => boolean;

const DEFAULT_IS_EXCLUDED: GIReceiverExclusionPredicate = (mesh) =>
  Boolean(mesh.userData['glassPiece']);

function isGIReceiver(
  obj: THREE.Object3D,
  isExcluded: GIReceiverExclusionPredicate,
): obj is THREE.Mesh {
  if (!(obj instanceof THREE.Mesh)) return false;
  if (isExcluded(obj)) return false;
  const mat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
  if (!mat) return false;
  return (
    mat instanceof THREE.MeshStandardMaterial ||
    mat instanceof THREE.MeshPhysicalMaterial
  );
}

/**
 * Wrap a source material with GI receiver capabilities.
 * Returns a new MeshPhysicalNodeMaterial with giNode added to lightingNode.
 */
function makeGIReceiverMaterial(
  srcMat: THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial,
  giNode: AnyNode,
): MeshPhysicalNodeMaterial {
  const nm = new MeshPhysicalNodeMaterial();

  nm.color       = srcMat.color.clone();
  nm.roughness   = srcMat.roughness;
  nm.metalness   = srcMat.metalness;
  nm.emissive    = srcMat.emissive.clone();
  nm.emissiveIntensity = srcMat.emissiveIntensity;
  nm.opacity     = srcMat.opacity;
  nm.transparent = srcMat.transparent;
  nm.side        = srcMat.side;
  if (srcMat.map)           nm.map           = srcMat.map;
  if (srcMat.normalMap)     nm.normalMap     = srcMat.normalMap;
  if (srcMat.roughnessMap)  nm.roughnessMap  = srcMat.roughnessMap;
  if (srcMat.metalnessMap)  nm.metalnessMap  = srcMat.metalnessMap;

  if (srcMat instanceof THREE.MeshPhysicalMaterial) {
    nm.transmission    = srcMat.transmission;
    nm.ior             = srcMat.ior;
    nm.thickness       = srcMat.thickness;
    nm.attenuationColor    = srcMat.attenuationColor.clone();
    nm.attenuationDistance = srcMat.attenuationDistance;
    nm.clearcoat       = srcMat.clearcoat;
    nm.clearcoatRoughness = srcMat.clearcoatRoughness;
    nm.iridescence     = srcMat.iridescence;
    nm.iridescenceIOR  = srcMat.iridescenceIOR;
  }

  // Add walkaround GI as an additive indirect-light term via emissiveNode.
  // NodeMaterial adds emissiveNode to outgoingLightNode AFTER the standard
  // PBR direct-lighting computation.
  nm.emissiveNode = giNode as AnyNode;

  // Tone-map + sRGB-encode the linear PBR + GI output.
  (nm as MeshPhysicalNodeMaterial & { outputNode: unknown }).outputNode =
    renderOutput(output, OUTPUT_TONE_MAPPING, OUTPUT_COLOR_SPACE);

  (nm as unknown as GIMaterial)[GI_TAG] = true;
  return nm;
}

export interface GIReceiverOptions {
  /**
   * Optional predicate for opting specific meshes out of GI wrapping.
   * Defaults to skipping any mesh with `userData.glassPiece === true`.
   */
  isExcluded?: GIReceiverExclusionPredicate;
}

/**
 * GIReceiver — class-based GI material wrapper (de-React-ified).
 *
 * Lifecycle:
 *   1. Construct with `new GIReceiver(opts?)`.
 *   2. Call `wrap(scene, cascadeBuffers)` to inject GI into all receiver meshes.
 *      Re-calling with new `cascadeBuffers` unwraps previous materials first.
 *   3. Call `unwrap()` to restore all original materials (e.g. on teardown).
 */
export class GIReceiver {
  private _wrappedMeshes = new Map<THREE.Mesh, THREE.Material>();
  private readonly _isExcluded: GIReceiverExclusionPredicate;

  constructor(opts: GIReceiverOptions = {}) {
    this._isExcluded = opts.isExcluded ?? DEFAULT_IS_EXCLUDED;
  }

  /**
   * Walk `scene` and wrap qualifying mesh materials with walkaround GI.
   * If `cascadeBuffers` is null, does nothing.
   * Re-wrapping: unwraps any previously wrapped meshes before re-applying.
   */
  wrap(scene: THREE.Scene, cascadeBuffers: CascadeBuffers | null): void {
    if (!cascadeBuffers) return;

    // Build the GI lighting node from the cascade buffers.  The Fn closure inside
    // buildWalkaroundLightingNode captures the storage reference at build time, so
    // any change of cascadeBuffers (StorageBufferAttribute reallocation on bounds
    // resize) requires a full rebuild + re-wrap.
    const { lightingNode: giNode } = buildWalkaroundLightingNode(cascadeBuffers);

    // Restore previous wraps before re-wrapping.
    this.unwrap();

    scene.traverse((obj) => {
      if (!isGIReceiver(obj, this._isExcluded)) return;
      const mat = (Array.isArray(obj.material) ? obj.material[0] : obj.material) as
        THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial;

      // Don't double-wrap.
      if ((mat as unknown as GIMaterial)[GI_TAG]) return;

      const wrapped = makeGIReceiverMaterial(mat, giNode);
      this._wrappedMeshes.set(obj, obj.material as THREE.Material);
      obj.material = wrapped;
    });
  }

  /**
   * Restore all original materials. Call on teardown or before re-wrapping.
   */
  unwrap(): void {
    this._wrappedMeshes.forEach((originalMat, mesh) => {
      mesh.material = originalMat;
    });
    this._wrappedMeshes.clear();
  }
}
