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

/** Which property a channel animates (glTF 2.0 target path plus KHR_animation_pointer). */
export type AnimationTargetPath = 'translation' | 'rotation' | 'scale' | 'weights' | 'pointer';

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

/** One channel: animate `target.path` of `target.node` from `sampler`.
 *
 * `path:"pointer"` carries a KHR_animation_pointer JSON pointer instead of a
 * node-local TRS/weights target. Engines do not consume clips directly; adapter
 * controllers resolve supported pointers to ordinary scene/material patches.
 */
export interface AnimationChannel {
  readonly target: {
    readonly node: SceneNodeId;
    readonly path: AnimationTargetPath;
    readonly pointer?: string;
  };
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

const ANIMATION_PATHS: ReadonlySet<string> = new Set([
  'translation', 'rotation', 'scale', 'weights', 'pointer',
]);
const ANIMATION_INTERPOLATIONS: ReadonlySet<string> = new Set([
  'LINEAR', 'STEP', 'CUBICSPLINE',
]);

function isRuntimeArray(value: unknown): boolean {
  return Array.isArray(value);
}

function animationError(path: string, message: string): never {
  throw new RangeError(`validateAnimationClip: ${path} ${message}`);
}

function assertAnimationFloatArray(value: unknown, path: string): asserts value is Float32Array {
  if (!(value instanceof Float32Array)) {
    throw new TypeError(`validateAnimationClip: ${path} must be a Float32Array`);
  }
  for (let index = 0; index < value.length; index += 1) {
    if (!Number.isFinite(value[index])) {
      animationError(`${path}[${index}]`, `must be finite (got ${String(value[index])})`);
    }
  }
}

/**
 * Validate a complete animation clip at the public CPU-sampling boundary.
 * Malformed sampler shapes are rejected instead of being floor-divided and
 * zero-filled, which otherwise turns corrupt asset data into plausible poses.
 */
export function validateAnimationClip(clip: AnimationClip): void {
  if (clip == null || typeof clip !== 'object' || Array.isArray(clip)) {
    throw new TypeError('validateAnimationClip: clip must be a non-array object');
  }
  if (!Number.isFinite(clip.duration) || clip.duration < 0) {
    animationError('duration', `must be finite and >= 0 (got ${String(clip.duration)})`);
  }
  if (!isRuntimeArray(clip.channels)) {
    throw new TypeError('validateAnimationClip: channels must be an array');
  }

  const targets = new Set<string>();
  for (let channelIndex = 0; channelIndex < clip.channels.length; channelIndex += 1) {
    const channel = clip.channels[channelIndex];
    const channelPath = `channels[${channelIndex}]`;
    if (channel == null || typeof channel !== 'object' || Array.isArray(channel)) {
      throw new TypeError(`validateAnimationClip: ${channelPath} must be an object`);
    }
    const target = channel.target;
    if (target == null || typeof target !== 'object' || Array.isArray(target)) {
      throw new TypeError(`validateAnimationClip: ${channelPath}.target must be an object`);
    }
    if (typeof target.node !== 'string' || target.node.length === 0) {
      throw new TypeError(`validateAnimationClip: ${channelPath}.target.node must be a non-empty string`);
    }
    if (typeof target.path !== 'string' || !ANIMATION_PATHS.has(target.path)) {
      animationError(
        `${channelPath}.target.path`,
        `is unsupported (got ${String(target.path)})`,
      );
    }
    if (target.path === 'pointer') {
      if (typeof target.pointer !== 'string' || target.pointer.length === 0) {
        throw new TypeError(
          `validateAnimationClip: ${channelPath}.target.pointer must be a non-empty string for path "pointer"`,
        );
      }
    } else if (target.pointer !== undefined) {
      throw new TypeError(
        `validateAnimationClip: ${channelPath}.target.pointer is only valid for path "pointer"`,
      );
    }
    const targetKey = `${target.node}\u0000${target.path}\u0000${target.pointer ?? ''}`;
    if (targets.has(targetKey)) {
      animationError(channelPath, 'duplicates an earlier channel target');
    }
    targets.add(targetKey);

    const sampler = channel.sampler;
    if (sampler == null || typeof sampler !== 'object' || Array.isArray(sampler)) {
      throw new TypeError(`validateAnimationClip: ${channelPath}.sampler must be an object`);
    }
    assertAnimationFloatArray(sampler.times, `${channelPath}.sampler.times`);
    assertAnimationFloatArray(sampler.values, `${channelPath}.sampler.values`);
    if (sampler.times.length === 0) {
      animationError(`${channelPath}.sampler.times`, 'must contain at least one keyframe');
    }
    for (let keyIndex = 0; keyIndex < sampler.times.length; keyIndex += 1) {
      const keyTime = sampler.times[keyIndex]!;
      if (keyTime < 0) {
        animationError(`${channelPath}.sampler.times[${keyIndex}]`, 'must be >= 0');
      }
      if (keyIndex > 0 && !(keyTime > sampler.times[keyIndex - 1]!)) {
        animationError(
          `${channelPath}.sampler.times[${keyIndex}]`,
          'must be strictly greater than the previous key time',
        );
      }
    }
    const lastTime = sampler.times[sampler.times.length - 1]!;
    if (lastTime > clip.duration) {
      animationError(
        `${channelPath}.sampler.times[${sampler.times.length - 1}]`,
        `exceeds clip duration ${clip.duration}`,
      );
    }
    const interpolation = sampler.interpolation ?? 'LINEAR';
    if (!ANIMATION_INTERPOLATIONS.has(interpolation)) {
      animationError(
        `${channelPath}.sampler.interpolation`,
        `is unsupported (got ${String(interpolation)})`,
      );
    }
    const cubicFactor = interpolation === 'CUBICSPLINE' ? 3 : 1;
    const fixedComponents = target.path === 'rotation'
      ? 4
      : target.path === 'translation' || target.path === 'scale'
        ? 3
        : undefined;
    const valueDivisor = sampler.times.length * cubicFactor;
    const components = fixedComponents ?? sampler.values.length / valueDivisor;
    if (!Number.isSafeInteger(components) || components <= 0) {
      animationError(
        `${channelPath}.sampler.values`,
        `length ${sampler.values.length} does not encode a positive integral component count`,
      );
    }
    const expectedValues = valueDivisor * components;
    if (sampler.values.length !== expectedValues) {
      animationError(
        `${channelPath}.sampler.values`,
        `has length ${sampler.values.length}; expected ${expectedValues}`,
      );
    }
    if (target.path === 'rotation') {
      const stride = components * cubicFactor;
      const valueOffset = interpolation === 'CUBICSPLINE' ? components : 0;
      for (let keyIndex = 0; keyIndex < sampler.times.length; keyIndex += 1) {
        const offset = keyIndex * stride + valueOffset;
        const magnitude = Math.hypot(
          sampler.values[offset]!, sampler.values[offset + 1]!,
          sampler.values[offset + 2]!, sampler.values[offset + 3]!,
        );
        if (!(magnitude > 1e-8) || !Number.isFinite(magnitude)) {
          animationError(
            `${channelPath}.sampler.values`,
            `contains a zero-length rotation quaternion at keyframe ${keyIndex}`,
          );
        }
      }
    }
  }
}

// ────────────────────────────────────────────────────────────────────────────
// CPU clip sampler — evaluate a clip at time t → per-channel values the host
// pushes through updatePrimitive(transform/positions). Backend-agnostic; no GPU.
// ────────────────────────────────────────────────────────────────────────────

/** One channel's value sampled at a given time. */
export interface SampledChannel {
  readonly node: SceneNodeId;
  readonly path: AnimationTargetPath;
  readonly pointer?: string;
  /** translation/scale = 3, rotation = 4 (quat xyzw), weights = morph count. */
  readonly value: Float32Array;
}

/** Component count for a path; `weights` is inferred from the buffer stride. */
function pathComponents(
  path: AnimationTargetPath,
  valuesLen: number,
  keyframeCount: number,
  cubic: boolean,
): number {
  if (path === 'rotation') return 4;
  if (path === 'translation' || path === 'scale') return 3;
  const div = (cubic ? 3 : 1) * Math.max(keyframeCount, 1);
  return valuesLen / div;
}

function findInterval(times: Float32Array, time: number): { i0: number; i1: number; t: number } {
  const n = times.length;
  if (n === 0) return { i0: 0, i1: 0, t: 0 };
  if (time <= (times[0] ?? 0)) return { i0: 0, i1: 0, t: 0 };
  if (time >= (times[n - 1] ?? 0)) return { i0: n - 1, i1: n - 1, t: 0 };
  let i0 = 0;
  while (i0 < n - 1 && (times[i0 + 1] ?? 0) <= time) i0 += 1;
  const i1 = Math.min(i0 + 1, n - 1);
  const dt = (times[i1] ?? 0) - (times[i0] ?? 0);
  const t = dt > 0 ? (time - (times[i0] ?? 0)) / dt : 0;
  return { i0, i1, t };
}

function slerpQuat(out: Float32Array, v: Float32Array, o0: number, o1: number, t: number): void {
  const aInv = 1 / Math.hypot(v[o0]!, v[o0 + 1]!, v[o0 + 2]!, v[o0 + 3]!);
  const bInv = 1 / Math.hypot(v[o1]!, v[o1 + 1]!, v[o1 + 2]!, v[o1 + 3]!);
  const ax = v[o0]! * aInv, ay = v[o0 + 1]! * aInv;
  const az = v[o0 + 2]! * aInv, aw = v[o0 + 3]! * aInv;
  let bx = v[o1]! * bInv, by = v[o1 + 1]! * bInv;
  let bz = v[o1 + 2]! * bInv, bw = v[o1 + 3]! * bInv;
  let cos = ax * bx + ay * by + az * bz + aw * bw;
  if (cos < 0) { cos = -cos; bx = -bx; by = -by; bz = -bz; bw = -bw; }
  let s0: number, s1: number;
  if (1 - cos > 1e-6) {
    const omega = Math.acos(Math.max(-1, Math.min(cos, 1)));
    const sin = Math.sin(omega);
    s0 = Math.sin((1 - t) * omega) / sin;
    s1 = Math.sin(t * omega) / sin;
  } else {
    s0 = 1 - t;
    s1 = t;
  }
  out[0] = s0 * ax + s1 * bx;
  out[1] = s0 * ay + s1 * by;
  out[2] = s0 * az + s1 * bz;
  out[3] = s0 * aw + s1 * bw;
}

function normalizeSampledQuat(value: Float32Array): void {
  const len = Math.hypot(value[0]!, value[1]!, value[2]!, value[3]!);
  if (len > 1e-8 && Number.isFinite(len)) {
    value[0] = value[0]! / len;
    value[1] = value[1]! / len;
    value[2] = value[2]! / len;
    value[3] = value[3]! / len;
  } else {
    throw new RangeError('sampleAnimationClip: interpolated rotation quaternion has zero length');
  }
}

/**
 * Evaluate a clip at `time` (seconds, clamped to the clip range). Returns one
 * `SampledChannel` per channel with the interpolated value. Rotation channels
 * use quaternion slerp; CUBICSPLINE uses the glTF Hermite basis.
 */
export function sampleAnimationClip(clip: AnimationClip, time: number): SampledChannel[] {
  validateAnimationClip(clip);
  if (!Number.isFinite(time)) {
    throw new RangeError(`sampleAnimationClip: time must be finite (got ${String(time)})`);
  }
  const out: SampledChannel[] = [];
  for (const ch of clip.channels) {
    const times = ch.sampler.times;
    const values = ch.sampler.values;
    const interpolation = ch.sampler.interpolation ?? 'LINEAR';
    const cubic = interpolation === 'CUBICSPLINE';
    const n = pathComponents(ch.target.path, values.length, times.length, cubic);
    const stride = cubic ? n * 3 : n;
    const { i0, i1, t } = findInterval(times, time);
    const value = new Float32Array(n);
    if (interpolation === 'STEP' || i0 === i1) {
      const off = cubic ? i0 * stride + n : i0 * stride; // cubic: middle (value) third
      for (let k = 0; k < n; k += 1) value[k] = values[off + k] ?? 0;
    } else if (cubic) {
      const dt = (times[i1] ?? 0) - (times[i0] ?? 0);
      const t2 = t * t, t3 = t2 * t;
      const h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
      const b0 = i0 * stride, b1 = i1 * stride;
      for (let k = 0; k < n; k += 1) {
        const v0 = values[b0 + n + k] ?? 0;       // value of kf0
        const out0 = values[b0 + 2 * n + k] ?? 0; // out-tangent of kf0
        const v1 = values[b1 + n + k] ?? 0;       // value of kf1
        const in1 = values[b1 + k] ?? 0;          // in-tangent of kf1
        value[k] = h00 * v0 + h10 * dt * out0 + h01 * v1 + h11 * dt * in1;
      }
    } else if (ch.target.path === 'rotation') {
      slerpQuat(value, values, i0 * stride, i1 * stride, t);
    } else {
      const o0 = i0 * stride, o1 = i1 * stride;
      for (let k = 0; k < n; k += 1) {
        const a = values[o0 + k] ?? 0;
        const b = values[o1 + k] ?? 0;
        value[k] = a + (b - a) * t;
      }
    }
    if (ch.target.path === 'rotation') normalizeSampledQuat(value);
    out.push({
      node: ch.target.node,
      path: ch.target.path,
      ...(ch.target.pointer !== undefined ? { pointer: ch.target.pointer } : {}),
      value,
    });
  }
  return out;
}
