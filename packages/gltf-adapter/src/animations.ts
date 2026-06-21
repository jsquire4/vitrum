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
import { unpackAccessorFloat, type GltfAccessorDiagnosticSink } from './accessors.js';
import { resolveGltfMaterialAnimationPointer, supportedGltfMaterialAnimationPointers } from './materialPointerAnimation.js';

/** Stable channel-target node id for glTF node `nodeIndex` (`gltf-node-<i>`). */
export function animationNodeId(nodeIndex: number): string {
  return `gltf-node-${nodeIndex}`;
}

export function animationPointerId(pointer: string): string {
  return `gltf-pointer:${pointer}`;
}

const VALID_PATHS: ReadonlySet<string> = new Set(['translation', 'rotation', 'scale', 'weights', 'pointer']);
const VALID_INTERPOLATIONS: ReadonlySet<string> = new Set(['LINEAR', 'STEP', 'CUBICSPLINE']);

export type GltfAnimationImportDiagnosticCode =
  | 'missing-animation-sampler'
  | 'unknown-animation-interpolation'
  | 'unreadable-animation-sampler'
  | 'unsupported-animation-target-path'
  | 'missing-animation-target-node'
  | 'animation-target-node-not-found'
  | 'invalid-animation-output-count'
  | 'dropped-animation';

export interface GltfAnimationImportDiagnostic {
  readonly severity: 'warning';
  readonly code: GltfAnimationImportDiagnosticCode;
  readonly path: string;
  readonly message: string;
  readonly animationIndex?: number;
  readonly channelIndex?: number;
  readonly samplerIndex?: number;
  readonly nodeIndex?: number;
  readonly targetPath?: string;
}

export type GltfAnimationImportDiagnosticSink = (
  diagnostic: GltfAnimationImportDiagnostic,
) => void;

export interface ConvertAnimationsOptions {
  /**
   * Optional selected-scene node filter. Channels whose target node is absent
   * from this set are outside the current import scene and are skipped without
   * warnings, matching feature-report/resource-fetch scoping.
   */
  readonly reachableNodeIndices?: ReadonlySet<number>;
  readonly reachableMaterialIndices?: ReadonlySet<number>;
}

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
  onDiagnostic?: GltfAnimationImportDiagnosticSink,
  onAccessorDiagnostic?: GltfAccessorDiagnosticSink,
  options: ConvertAnimationsOptions = {},
): AnimationClip[] {
  const clips: AnimationClip[] = [];
  const animations = gltf.animations ?? [];
  const reachableNodeIndices = options.reachableNodeIndices;
  const reachableMaterialIndices = options.reachableMaterialIndices;

  for (const [animIdx, anim] of animations.entries()) {
    const label = anim.name ?? `#${animIdx}`;
    const samplers = anim.samplers ?? [];
    const gltfChannels = anim.channels ?? [];
    const channels: AnimationChannel[] = [];
    let hasReachableChannel = reachableNodeIndices === undefined;
    let duration = 0;

    // Decode each referenced sampler once (multiple channels may share one).
    const decoded = new Map<number, { times: Float32Array; values: Float32Array; interpolation: AnimationInterpolation } | null>();
    const decodeSampler = (samplerIdx: number): { times: Float32Array; values: Float32Array; interpolation: AnimationInterpolation } | null => {
      if (decoded.has(samplerIdx)) return decoded.get(samplerIdx)!;
      let result: { times: Float32Array; values: Float32Array; interpolation: AnimationInterpolation } | null = null;
      const sampler = samplers[samplerIdx];
      if (!sampler) {
        emitAnimationDiagnostic(warnings, onDiagnostic, {
          severity: 'warning',
          code: 'missing-animation-sampler',
          path: `animations[${animIdx}].samplers[${samplerIdx}]`,
          animationIndex: animIdx,
          samplerIndex: samplerIdx,
          message:
            `[vitrum/gltf-adapter] Animation "${label}" references sampler ${samplerIdx} which does ` +
            'not exist. Channel skipped.',
        });
      } else {
        let interpolation: AnimationInterpolation = 'LINEAR';
        if (sampler.interpolation !== undefined) {
          if (VALID_INTERPOLATIONS.has(sampler.interpolation)) {
            interpolation = sampler.interpolation as AnimationInterpolation;
          } else {
            emitAnimationDiagnostic(warnings, onDiagnostic, {
              severity: 'warning',
              code: 'unknown-animation-interpolation',
              path: `animations[${animIdx}].samplers[${samplerIdx}].interpolation`,
              animationIndex: animIdx,
              samplerIndex: samplerIdx,
              message:
                `[vitrum/gltf-adapter] Animation "${label}" sampler ${samplerIdx} has unknown ` +
                `interpolation "${sampler.interpolation}". Falling back to LINEAR.`,
            });
          }
        }
        try {
          const times = unpackAccessorFloat(gltf, buffers, sampler.input, warnings, onAccessorDiagnostic);
          const values = unpackAccessorFloat(gltf, buffers, sampler.output, warnings, onAccessorDiagnostic);
          result = { times, values, interpolation };
        } catch (e) {
          emitAnimationDiagnostic(warnings, onDiagnostic, {
            severity: 'warning',
            code: 'unreadable-animation-sampler',
            path: `animations[${animIdx}].samplers[${samplerIdx}]`,
            animationIndex: animIdx,
            samplerIndex: samplerIdx,
            message:
              `[vitrum/gltf-adapter] Failed to read animation "${label}" sampler ${samplerIdx}: ` +
              `${String(e)} Channel skipped.`,
          });
        }
      }
      decoded.set(samplerIdx, result);
      return result;
    };

    for (const [chIdx, ch] of gltfChannels.entries()) {
      const path = ch.target?.path;
      if (path === undefined || !VALID_PATHS.has(path)) {
        emitAnimationDiagnostic(warnings, onDiagnostic, {
          severity: 'warning',
          code: 'unsupported-animation-target-path',
          path: `animations[${animIdx}].channels[${chIdx}].target.path`,
          animationIndex: animIdx,
          channelIndex: chIdx,
          targetPath: path,
          message:
            `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} targets unsupported ` +
            `path "${String(path)}" (supported: translation, rotation, scale, weights, pointer). Channel skipped.`,
        });
        continue;
      }
      if (path === 'pointer') {
        const pointer = ch.target?.extensions?.KHR_animation_pointer?.pointer;
        const pointerTarget = resolveGltfMaterialAnimationPointer(pointer);
        if (
          reachableMaterialIndices !== undefined &&
          pointerTarget !== undefined &&
          !reachableMaterialIndices.has(pointerTarget.materialIndex)
        ) {
          continue;
        }
        hasReachableChannel = true;
        if (pointerTarget === undefined) {
          emitAnimationDiagnostic(warnings, onDiagnostic, {
            severity: 'warning',
            code: 'unsupported-animation-target-path',
            path: `animations[${animIdx}].channels[${chIdx}].target.extensions.KHR_animation_pointer.pointer`,
            animationIndex: animIdx,
            channelIndex: chIdx,
            targetPath: 'pointer',
            message:
              `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} targets unsupported ` +
              `KHR_animation_pointer JSON pointer "${String(pointer)}". Supported material pointers: ` +
              `${supportedGltfMaterialAnimationPointers().join(', ')}. Channel skipped.`,
          });
          continue;
        }

        const samplerData = decodeSampler(ch.sampler);
        if (!samplerData) continue;
        const { times, values, interpolation } = samplerData;
        const cubicFactor = interpolation === 'CUBICSPLINE' ? 3 : 1;
        const expected = times.length * pointerTarget.components * cubicFactor;
        if (values.length !== expected) {
          emitAnimationDiagnostic(warnings, onDiagnostic, {
            severity: 'warning',
            code: 'invalid-animation-output-count',
            path: `animations[${animIdx}].channels[${chIdx}].sampler`,
            animationIndex: animIdx,
            channelIndex: chIdx,
            samplerIndex: ch.sampler,
            targetPath: 'pointer',
            message:
              `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} (${pointer}) has ` +
              `${values.length} output floats but ${times.length} keyframes expect ${expected}. Channel skipped.`,
          });
          continue;
        }
        if (times.length > 0) {
          duration = Math.max(duration, times[times.length - 1] ?? 0);
        }
        channels.push({
          target: { node: animationPointerId(pointerTarget.pointer), path: 'pointer', pointer: pointerTarget.pointer },
          sampler: { times, values, interpolation },
        });
        continue;
      }

      const nodeIdx = ch.target?.node;
      if (reachableNodeIndices !== undefined) {
        if (nodeIdx === undefined || !reachableNodeIndices.has(nodeIdx)) continue;
        hasReachableChannel = true;
      }
      if (nodeIdx === undefined) {
        emitAnimationDiagnostic(warnings, onDiagnostic, {
          severity: 'warning',
          code: 'missing-animation-target-node',
          path: `animations[${animIdx}].channels[${chIdx}].target.node`,
          animationIndex: animIdx,
          channelIndex: chIdx,
          targetPath: path,
          message:
            `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} has no target node ` +
            '(extension-targeted channels are not supported). Channel skipped.',
        });
        continue;
      }
      if (!gltf.nodes?.[nodeIdx]) {
        emitAnimationDiagnostic(warnings, onDiagnostic, {
          severity: 'warning',
          code: 'animation-target-node-not-found',
          path: `animations[${animIdx}].channels[${chIdx}].target.node`,
          animationIndex: animIdx,
          channelIndex: chIdx,
          nodeIndex: nodeIdx,
          targetPath: path,
          message:
            `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} targets node ${nodeIdx} ` +
            'which does not exist. Channel skipped.',
        });
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
          emitAnimationDiagnostic(warnings, onDiagnostic, {
            severity: 'warning',
            code: 'invalid-animation-output-count',
            path: `animations[${animIdx}].channels[${chIdx}].sampler`,
            animationIndex: animIdx,
            channelIndex: chIdx,
            samplerIndex: ch.sampler,
            nodeIndex: nodeIdx,
            targetPath: typedPath,
            message:
              `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} (${typedPath}) has ` +
              `${values.length} output floats but ${times.length} keyframes expect ` +
              `${times.length * comps * cubicFactor}. Channel skipped.`,
          });
          continue;
        }
      } else if (times.length > 0 && values.length % (times.length * cubicFactor) !== 0) {
        emitAnimationDiagnostic(warnings, onDiagnostic, {
          severity: 'warning',
          code: 'invalid-animation-output-count',
          path: `animations[${animIdx}].channels[${chIdx}].sampler`,
          animationIndex: animIdx,
          channelIndex: chIdx,
          samplerIndex: ch.sampler,
          nodeIndex: nodeIdx,
          targetPath: typedPath,
          message:
            `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} (weights) output length ` +
            `${values.length} is not a multiple of keyframe count ${times.length}` +
            `${cubicFactor === 3 ? ' × 3 (CUBICSPLINE)' : ''}. Channel skipped.`,
        });
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
      if (!hasReachableChannel) continue;
      emitAnimationDiagnostic(warnings, onDiagnostic, {
        severity: 'warning',
        code: 'dropped-animation',
        path: `animations[${animIdx}]`,
        animationIndex: animIdx,
        message: `[vitrum/gltf-adapter] Animation "${label}" has no importable channels and was dropped.`,
      });
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

function emitAnimationDiagnostic(
  warnings: string[],
  onDiagnostic: GltfAnimationImportDiagnosticSink | undefined,
  diagnostic: GltfAnimationImportDiagnostic,
): void {
  warnings.push(diagnostic.message);
  onDiagnostic?.(diagnostic);
}
