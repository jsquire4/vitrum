/**
 * GI receiver material wrapper (§8.2).
 *
 * Converts receiver meshes (walls, floor, ceiling, fixture bodies) to
 * MeshPhysicalNodeMaterial and adds the walkaround GI contribution to their
 * lightingNode.
 *
 * Glass meshes (userData.glassPiece === true) are NOT receivers — they
 * contribute light through transmission but don't receive cascade GI.
 *
 * React hook useGIReceiverConverter walks the scene once on mount, wraps
 * qualifying materials, and re-wraps when the scene topology changes.
 */

import { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import { output, renderOutput } from 'three/tsl';
import type { CascadeBuffers } from './cascadePyramid';
import { buildWalkaroundLightingNode } from './walkaroundDiffuseLighting';

// WebGPU is pinned to NoToneMapping + LinearSRGBColorSpace at the renderer
// level (WalkaroundStage.tsx:146-147) to keep Three's internal HDR
// `_frameBufferTarget` from engaging — that target collides with the
// WebGPU transmission compositor (rgba16float source vs bgra8unorm dest
// inside `viewportMipTexture()`).  The cost: scene output reaches the
// canvas as raw linear-sRGB radiance, which a gamma-encoded display
// renders ~5x too dark on mid-tones.
//
// Fix: tone-map + sRGB-encode INSIDE each receiver material's `outputNode`
// (the fragment shader's final stage) via TSL's `renderOutput()`.  The
// renderer's HDR framebuffer stays disabled (transmission compositor
// safe), but the per-pixel output reaching the canvas is now properly
// gamma-encoded sRGB with ACES highlight roll-off.
//
// Cross-port from DDGI (applyDDGIShading.ts:52,231) and ReSTIR
// (composite.wgsl.ts:69-90 acesFilm fit).  Same Narkowicz 2015 ACES fit
// the other two engines use; ensures all three engines tonemap to the
// same target on the same scene.
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
 * Library consumers pass this via `useGIReceiverConverter({ isExcluded })`
 * to opt specific meshes out of GI without forking giReceiver. The
 * stained-glass-app demo's default skips meshes tagged with
 * `userData.glassPiece` (panels are emitters, not receivers).
 */
export type GIReceiverExclusionPredicate = (mesh: THREE.Mesh) => boolean;

const DEFAULT_IS_EXCLUDED: GIReceiverExclusionPredicate = (mesh) =>
  Boolean(mesh.userData['glassPiece']);

/** Check if mesh should receive GI. */
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

  // Copy standard PBR parameters.
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

  // Copy physical params if applicable.
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

  // Add walkaround GI as an additive indirect-light term.
  // NodeMaterial adds emissiveNode to outgoingLightNode AFTER the standard
  // PBR direct-lighting computation. This makes GI purely additive, preserving
  // all existing diffuse/specular/shadow shading.
  nm.emissiveNode = giNode as AnyNode;

  // Tone-map + sRGB-encode the linear PBR + GI output so the canvas (pinned
  // to NoToneMapping + LinearSRGBColorSpace at the renderer level) receives
  // gamma-correct pixels.  See OUTPUT_TONE_MAPPING comment block above for
  // the full rationale (transmission-compositor format-mismatch workaround).
  // `output` is the built-in TSL property node holding the standard PBR-lit
  // result (direct + emissive/GI).  Without this wrapper the canvas shows
  // raw linear radiance interpreted by the monitor as gamma-encoded — every
  // mid-tone reads ~5x too dark and saturated cells wash out to pastel.
  (nm as MeshPhysicalNodeMaterial & { outputNode: unknown }).outputNode =
    renderOutput(output, OUTPUT_TONE_MAPPING, OUTPUT_COLOR_SPACE);

  (nm as unknown as GIMaterial)[GI_TAG] = true;
  return nm;
}

export interface GIReceiverConverterOpts {
  /**
   * Optional predicate for opting specific meshes out of GI wrapping.
   * Defaults to skipping any mesh with `userData.glassPiece === true`
   * (the stained-glass-app convention; panels are emitters, not
   * receivers). Library consumers pass their own predicate to map
   * their scene's mesh-classification convention onto GI receivership.
   */
  isExcluded?: GIReceiverExclusionPredicate;
}

/**
 * React hook: walk scene tree, wrap receiver materials with GI on mount.
 * Re-wraps when cascadeBuffers changes.
 */
export function useGIReceiverConverter(
  cascadeBuffers: CascadeBuffers | null,
  opts: GIReceiverConverterOpts = {},
): void {
  const { scene } = useThree();
  const wrappedMeshes = useRef<Map<THREE.Mesh, THREE.Material>>(new Map());
  const isExcluded = opts.isExcluded ?? DEFAULT_IS_EXCLUDED;

  useEffect(() => {
    if (!cascadeBuffers) return;

    // Build the GI node from the cascade buffers. The Fn closure inside
    // buildWalkaroundLightingNode captures the storage reference at build
    // time, so any change of cascadeBuffers (StorageBufferAttribute
    // reallocation on bounds resize) requires a full rebuild + re-wrap —
    // the previous "updateBuffers" shortcut couldn't actually patch the
    // captured node and the wrapped meshes would silently keep the stale
    // node graph. Audit B2 fix from 2026-05-07 sweep.
    const { lightingNode: giNode } = buildWalkaroundLightingNode(cascadeBuffers);

    // Restore previous wraps before re-wrapping (scene may have changed).
    wrappedMeshes.current.forEach((originalMat, mesh) => {
      mesh.material = originalMat;
    });
    wrappedMeshes.current.clear();

    // Walk scene and wrap qualifying meshes.
    scene.traverse((obj) => {
      if (!isGIReceiver(obj, isExcluded)) return;
      const mat = (Array.isArray(obj.material) ? obj.material[0] : obj.material) as
        THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial;

      // Don't double-wrap.
      if ((mat as unknown as GIMaterial)[GI_TAG]) return;

      const wrapped = makeGIReceiverMaterial(mat, giNode);
      wrappedMeshes.current.set(obj, obj.material as THREE.Material);
      obj.material = wrapped;
    });

    return () => {
      // Restore on unmount.
      wrappedMeshes.current.forEach((originalMat, mesh) => {
        mesh.material = originalMat;
      });
      wrappedMeshes.current.clear();
    };
  }, [scene, cascadeBuffers]);
}
