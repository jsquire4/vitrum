// Scene description — backend-agnostic.
//
// Animation — keyframe clips (glTF 2.0 data model). This is the DATA contract
// only; sampling a clip to per-frame poses and mixing is a host/engine concern
// (the P3 animation workflow). No backend consumes clips directly today — a
// host evaluates a clip at time t and pushes the resulting transforms / morph
// weights through `updatePrimitive` (positions/transform) like any other
// per-frame update.

import type { SceneNodeId } from './math.js';

/** Interpolation mode for a keyframe sampler (glTF 2.0). */
export type AnimationInterpolation = 'LINEAR' | 'STEP' | 'CUBICSPLINE';

/** Which node property a channel animates (glTF 2.0 target path). */
export type AnimationTargetPath = 'translation' | 'rotation' | 'scale' | 'weights';

/**
 * A keyframe sampler: input times → output values. `values` is flat; the
 * component count per keyframe depends on the channel's `path`
 * (translation/scale = 3, rotation = 4 (quaternion xyzw), weights = morph-target
 * count), tripled for CUBICSPLINE (in-tangent, value, out-tangent).
 */
export interface AnimationSampler {
  readonly times: Float32Array;                    // seconds; length = keyframe count
  readonly values: Float32Array;                   // flat output values
  readonly interpolation?: AnimationInterpolation; // default 'LINEAR'
}

/** One channel: animate `target.path` of `target.node` from `sampler`. */
export interface AnimationChannel {
  readonly target: { readonly node: SceneNodeId; readonly path: AnimationTargetPath };
  readonly sampler: AnimationSampler;
}

/**
 * A named animation clip — a set of channels driving node transforms / morph
 * weights over time. Mirrors glTF 2.0 / THREE.AnimationClip. Evaluated by the
 * host or a P3 engine-side sampler to poses; no backend reads clips directly.
 */
export interface AnimationClip {
  readonly name?: string;
  readonly duration: number;                       // seconds (max sampler time)
  readonly channels: ReadonlyArray<AnimationChannel>;
}
