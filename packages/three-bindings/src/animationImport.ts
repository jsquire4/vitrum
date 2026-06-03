// animationImport.ts — THREE.AnimationClip[] (as produced by GLTFLoader) →
// @vitrum/core AnimationClip[] (P3).
//
// Each THREE KeyframeTrack targets "<nodeName>.<property>". We resolve the node
// to its uuid (the same id convertMesh assigns to a primitive) and map the
// property to the glTF target path. Tracks whose node can't be resolved, or
// whose property isn't a transform / morph-weights channel, are faithfully
// SKIPPED (not crashed) — the host gets exactly the channels vitrum can address.

import * as THREE from 'three';
import type {
  AnimationClip,
  AnimationChannel,
  AnimationInterpolation,
  AnimationTargetPath,
} from '@vitrum/core';

const PROPERTY_TO_PATH: Readonly<Record<string, AnimationTargetPath>> = {
  position: 'translation',
  quaternion: 'rotation',
  scale: 'scale',
  morphTargetInfluences: 'weights',
};

function mapInterpolation(track: THREE.KeyframeTrack): AnimationInterpolation {
  // THREE.InterpolateDiscrete = 2300, InterpolateLinear = 2301, InterpolateSmooth = 2302.
  const mode = (track as { getInterpolation?: () => number }).getInterpolation?.();
  if (mode === THREE.InterpolateDiscrete) return 'STEP';
  if (mode === THREE.InterpolateSmooth) return 'CUBICSPLINE';
  return 'LINEAR';
}

/**
 * Convert THREE.AnimationClip[] into vitrum AnimationClip[]. The `root` is the
 * scene the clips animate (used to resolve each track's node name → uuid).
 */
export function convertAnimations(
  clips: ReadonlyArray<THREE.AnimationClip>,
  root: THREE.Object3D,
): AnimationClip[] {
  const out: AnimationClip[] = [];
  for (const clip of clips) {
    const channels: AnimationChannel[] = [];
    for (const track of clip.tracks) {
      const parsed = THREE.PropertyBinding.parseTrackName(track.name);
      const path = PROPERTY_TO_PATH[parsed.propertyName ?? ''];
      if (path === undefined) continue; // not a transform / weights channel
      const node = THREE.PropertyBinding.findNode(root, parsed.nodeName) as THREE.Object3D | undefined;
      if (node == null) continue; // unresolved target
      channels.push({
        target: { node: node.uuid, path },
        sampler: {
          times: track.times instanceof Float32Array ? track.times : new Float32Array(track.times),
          values: track.values instanceof Float32Array ? track.values : new Float32Array(track.values),
          interpolation: mapInterpolation(track),
        },
      });
    }
    out.push({
      ...(clip.name ? { name: clip.name } : {}),
      duration: clip.duration,
      channels,
    });
  }
  return out;
}
