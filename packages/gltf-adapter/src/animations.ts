// animations.ts — glTF animations → core AnimationClip[] (GLTF-03).
//
// Parses glTF §3.11 animations (samplers + channels) into the @vitrum/core
// keyframe data model (`AnimationClip` / `AnimationChannel` /
// `AnimationSampler`). The core model mirrors glTF directly, so conversion is
// mostly accessor unpacking plus validation:
//
//   - times  = sampler.input accessor (SCALAR float seconds)
//   - values = sampler.output accessor (float-converted; normalized int
//     rotation outputs are de-normalized by unpackAccessorFloat; sparse
//     accessors are supported)
//   - interpolation LINEAR / STEP / CUBICSPLINE pass through; unknown values
//     warn and degrade to LINEAR.
//
// Channel-target identity: a glTF channel targets a NODE; the adapter's scene
// flattening means one node may map to zero (joint/empty nodes) or many
// (multi-primitive mesh) ScenePrimitives. Channels therefore carry the stable
// node id `animationNodeId(nodeIndex)` = `gltf-node-<index>`, and
// `gltfToScene()` returns an `animationTargets` record mapping that node id to
// the primitive ids created from the node's mesh. See GltfToSceneResult JSDoc
// for the host-side evaluation contract.
//
// Sampling/evaluation itself is `sampleAnimationClip` in @vitrum/core (which
// normalizes rotation outputs for every interpolation mode as of the
// 2026-06-11 CORE-01 wave).
//
// Reference: glTF 2.0 specification (Khronos Group), §3.11 Animations
// https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html#animations

import type {
  AnimationChannel,
  AnimationClip,
  AnimationInterpolation,
  AnimationTargetPath,
} from '@vitrum/core';
import type { GltfJson } from './gltfTypes.js';
import { unpackAccessorFloat } from './accessors.js';

/** Stable channel-target node id for glTF node `nodeIndex` (`gltf-node-<i>`). */
export function animationNodeId(nodeIndex: number): string {
  return `gltf-node-${nodeIndex}`;
}

const VALID_PATHS: ReadonlySet<string> = new Set(['translation', 'rotation', 'scale', 'weights']);
const VALID_INTERPOLATIONS: ReadonlySet<string> = new Set(['LINEAR', 'STEP', 'CUBICSPLINE']);

/** Fixed per-keyframe component count for TRS paths (weights is inferred from
 *  the buffer by the core sampler, so it is validated loosely here). */
function trsComponents(path: AnimationTargetPath): number | undefined {
  if (path === 'rotation') return 4;
  if (path === 'translation' || path === 'scale') return 3;
  return undefined; // weights
}

/**
 * Convert all glTF animations to core AnimationClips.
 *
 * Unreadable/malformed samplers and channels are skipped with a warning;
 * animations with zero importable channels are dropped with a warning.
 */
export function convertAnimations(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  warnings: string[],
): AnimationClip[] {
  const clips: AnimationClip[] = [];
  const animations = gltf.animations ?? [];

  for (const [animIdx, anim] of animations.entries()) {
    const label = anim.name ?? `#${animIdx}`;
    const samplers = anim.samplers ?? [];
    const gltfChannels = anim.channels ?? [];
    const channels: AnimationChannel[] = [];
    let duration = 0;

    // Decode each referenced sampler once (multiple channels may share one).
    const decoded = new Map<number, { times: Float32Array; values: Float32Array; interpolation: AnimationInterpolation } | null>();
    const decodeSampler = (samplerIdx: number): { times: Float32Array; values: Float32Array; interpolation: AnimationInterpolation } | null => {
      if (decoded.has(samplerIdx)) return decoded.get(samplerIdx)!;
      let result: { times: Float32Array; values: Float32Array; interpolation: AnimationInterpolation } | null = null;
      const sampler = samplers[samplerIdx];
      if (!sampler) {
        warnings.push(
          `[vitrum/gltf-adapter] Animation "${label}" references sampler ${samplerIdx} which does ` +
            'not exist. Channel skipped.',
        );
      } else {
        let interpolation: AnimationInterpolation = 'LINEAR';
        if (sampler.interpolation !== undefined) {
          if (VALID_INTERPOLATIONS.has(sampler.interpolation)) {
            interpolation = sampler.interpolation as AnimationInterpolation;
          } else {
            warnings.push(
              `[vitrum/gltf-adapter] Animation "${label}" sampler ${samplerIdx} has unknown ` +
                `interpolation "${sampler.interpolation}". Falling back to LINEAR.`,
            );
          }
        }
        try {
          const times = unpackAccessorFloat(gltf, buffers, sampler.input, warnings);
          const values = unpackAccessorFloat(gltf, buffers, sampler.output, warnings);
          result = { times, values, interpolation };
        } catch (e) {
          warnings.push(
            `[vitrum/gltf-adapter] Failed to read animation "${label}" sampler ${samplerIdx}: ` +
              `${String(e)} Channel skipped.`,
          );
        }
      }
      decoded.set(samplerIdx, result);
      return result;
    };

    for (const [chIdx, ch] of gltfChannels.entries()) {
      const path = ch.target?.path;
      if (path === undefined || !VALID_PATHS.has(path)) {
        warnings.push(
          `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} targets unsupported ` +
            `path "${String(path)}" (supported: translation, rotation, scale, weights). Channel skipped.`,
        );
        continue;
      }
      const nodeIdx = ch.target?.node;
      if (nodeIdx === undefined) {
        warnings.push(
          `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} has no target node ` +
            '(extension-targeted channels are not supported). Channel skipped.',
        );
        continue;
      }
      if (!gltf.nodes?.[nodeIdx]) {
        warnings.push(
          `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} targets node ${nodeIdx} ` +
            'which does not exist. Channel skipped.',
        );
        continue;
      }

      const samplerData = decodeSampler(ch.sampler);
      if (!samplerData) continue;
      const { times, values, interpolation } = samplerData;
      const typedPath = path as AnimationTargetPath;

      // Validate value-buffer shape (strict for TRS; loose multiple-of check
      // for weights since the morph-target count lives in the mesh).
      const cubicFactor = interpolation === 'CUBICSPLINE' ? 3 : 1;
      const comps = trsComponents(typedPath);
      if (comps !== undefined) {
        if (values.length !== times.length * comps * cubicFactor) {
          warnings.push(
            `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} (${typedPath}) has ` +
              `${values.length} output floats but ${times.length} keyframes expect ` +
              `${times.length * comps * cubicFactor}. Channel skipped.`,
          );
          continue;
        }
      } else if (times.length > 0 && values.length % (times.length * cubicFactor) !== 0) {
        warnings.push(
          `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} (weights) output length ` +
            `${values.length} is not a multiple of keyframe count ${times.length}` +
            `${cubicFactor === 3 ? ' × 3 (CUBICSPLINE)' : ''}. Channel skipped.`,
        );
        continue;
      }

      if (times.length > 0) {
        duration = Math.max(duration, times[times.length - 1] ?? 0);
      }
      channels.push({
        target: { node: animationNodeId(nodeIdx), path: typedPath },
        sampler: { times, values, interpolation },
      });
    }

    if (channels.length === 0) {
      warnings.push(
        `[vitrum/gltf-adapter] Animation "${label}" has no importable channels and was dropped.`,
      );
      continue;
    }
    clips.push({
      ...(anim.name !== undefined ? { name: anim.name } : {}),
      duration,
      channels,
    });
  }

  return clips;
}
