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
import { GltfComponentType, type GltfJson } from './gltfTypes.js';
import { unpackAccessorFloat, type GltfAccessorDiagnosticSink } from './accessors.js';
import {
  GltfResourceLimitError,
  type ImportResourceLedger,
} from './importResourceBudget.js';
import {
  resolveGltfAnimationPointer,
  supportedGltfAnimationPointers,
  gltfAnimationPointerTargetComponentCount,
  gltfAnimationPointerTargetDefinitionError,
  gltfAnimationPointerInterpolationError,
  gltfAnimationPointerOutputAccessorError,
  gltfAnimationPointerTargetIdentity,
  gltfAnimationTargetsConflict,
  gltfNativeAnimationTargetIdentity,
  gltfAnimationPointerValuesError,
  type GltfAnimationPointerTarget,
  type GltfAnimationTargetIdentity,
} from './animationPointer.js';

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
  | 'animation-pointer-target-undefined'
  | 'invalid-animation-pointer-output-accessor'
  | 'invalid-animation-pointer-interpolation'
  | 'invalid-animation-pointer-value'
  | 'duplicate-animation-target'
  | 'invalid-animation-output-count'
  | 'dropped-animation';

export interface GltfAnimationImportDiagnostic {
  readonly severity: 'error'  ;
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

export class GltfAnimationImportError extends Error {
  readonly diagnostic: GltfAnimationImportDiagnostic;

  constructor(diagnostic: GltfAnimationImportDiagnostic) {
    super(diagnostic.message);
    this.name = 'GltfAnimationImportError';
    this.diagnostic = diagnostic;
  }
}

export interface ConvertAnimationsOptions {
  /**
   * Optional selected-scene node filter. Channels whose target node is absent
   * from this set are outside the current import scene and are skipped without
   * warnings, matching feature-report/resource-fetch scoping.
   */
  readonly reachableNodeIndices?: ReadonlySet<number>;
  readonly reachableMaterialIndices?: ReadonlySet<number>;
  readonly reachableCameraIndices?: ReadonlySet<number>;
  readonly reachablePunctualLightIndices?: ReadonlySet<number>;
  /** Shared monotonic resource ledger for the complete import operation. */
  readonly resourceLedger?: ImportResourceLedger;
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
 * Unreadable or malformed reachable samplers/channels reject conversion
 * atomically. Channels outside the selected scene are intentionally omitted.
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

  for (const [animIdx, anim] of animations.entries()) {
    const label = anim.name ?? `#${animIdx}`;
    const samplers = anim.samplers ?? [];
    const gltfChannels = anim.channels ?? [];
    const channels: AnimationChannel[] = [];
    let hasReachableChannel = reachableNodeIndices === undefined;
    let duration = 0;
    const claimedTargets: GltfAnimationTargetIdentity[] = [];

    // Decode each referenced sampler once (multiple channels may share one).
    const decoded = new Map<number, { times: Float32Array; values: Float32Array; interpolation: AnimationInterpolation } | null>();
    const decodeSampler = (samplerIdx: number): { times: Float32Array; values: Float32Array; interpolation: AnimationInterpolation } | null => {
      if (decoded.has(samplerIdx)) return decoded.get(samplerIdx)!;
      let result: { times: Float32Array; values: Float32Array; interpolation: AnimationInterpolation } | null = null;
      const sampler = samplers[samplerIdx];
      if (!sampler) {
        emitAnimationDiagnostic(warnings, onDiagnostic, {
          severity: 'error',
          code: 'missing-animation-sampler',
          path: `animations[${animIdx}].samplers[${samplerIdx}]`,
          animationIndex: animIdx,
          samplerIndex: samplerIdx,
          message:
            `[vitrum/gltf-adapter] Animation "${label}" references sampler ${samplerIdx} which does ` +
            'not exist. Import rejected.',
        });
      } else {
        let interpolation: AnimationInterpolation = 'LINEAR';
        if (sampler.interpolation !== undefined) {
          if (VALID_INTERPOLATIONS.has(sampler.interpolation)) {
            interpolation = sampler.interpolation as AnimationInterpolation;
          } else {
            emitAnimationDiagnostic(warnings, onDiagnostic, {
              severity: 'error',
              code: 'unknown-animation-interpolation',
              path: `animations[${animIdx}].samplers[${samplerIdx}].interpolation`,
              animationIndex: animIdx,
              samplerIndex: samplerIdx,
              message:
                `[vitrum/gltf-adapter] Animation "${label}" sampler ${samplerIdx} has unknown ` +
                `interpolation "${sampler.interpolation}". Import rejected.`,
            });
          }
        }
        try {
          const times = unpackAccessorFloat(
            gltf,
            buffers,
            sampler.input,
            warnings,
            onAccessorDiagnostic,
            options.resourceLedger,
          );
          const values = unpackAccessorFloat(
            gltf,
            buffers,
            sampler.output,
            warnings,
            onAccessorDiagnostic,
            options.resourceLedger,
          );
          const inputAccessor = gltf.accessors?.[sampler.input];
          if (
            !inputAccessor || inputAccessor.type !== 'SCALAR' ||
            inputAccessor.componentType !== GltfComponentType.FLOAT ||
            inputAccessor.normalized === true || times.length === 0 || !times.every(Number.isFinite) ||
            !values.every(Number.isFinite)
          ) {
            throw new Error('animation input must be a non-empty, finite, non-normalized FLOAT SCALAR accessor and output values must be finite');
          }
          for (let keyframe = 0; keyframe < times.length; keyframe++) {
            const time = times[keyframe]!;
            if (time < 0 || (keyframe > 0 && time <= times[keyframe - 1]!)) {
              throw new Error('animation input times must be non-negative and strictly increasing');
            }
          }
          result = { times, values, interpolation };
        } catch (e) {
          if (e instanceof GltfResourceLimitError) throw e;
          emitAnimationDiagnostic(warnings, onDiagnostic, {
            severity: 'error',
            code: 'unreadable-animation-sampler',
            path: `animations[${animIdx}].samplers[${samplerIdx}]`,
            animationIndex: animIdx,
            samplerIndex: samplerIdx,
            message:
              `[vitrum/gltf-adapter] Failed to read animation "${label}" sampler ${samplerIdx}: ` +
              `${String(e)} Import rejected.`,
          });
        }
      }
      decoded.set(samplerIdx, result);
      return result;
    };

    for (const [chIdx, ch] of gltfChannels.entries()) {
      const targetNodeIndex = ch.target?.node;
      if (
        ch.target?.path !== 'pointer' &&
        reachableNodeIndices !== undefined &&
        targetNodeIndex !== undefined &&
        !reachableNodeIndices.has(targetNodeIndex)
      ) {
        continue;
      }
      const path = ch.target?.path;
      if (path === undefined || !VALID_PATHS.has(path)) {
        emitAnimationDiagnostic(warnings, onDiagnostic, {
          severity: 'error',
          code: 'unsupported-animation-target-path',
          path: `animations[${animIdx}].channels[${chIdx}].target.path`,
          animationIndex: animIdx,
          channelIndex: chIdx,
          targetPath: path,
          message:
            `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} targets unsupported ` +
            `path "${String(path)}" (supported: translation, rotation, scale, weights, pointer). Import rejected.`,
        });
        continue;
      }
      if (path === 'pointer') {
        const pointer = ch.target?.extensions?.KHR_animation_pointer?.pointer;
        const pointerTarget = resolveGltfAnimationPointer(pointer);
        if (pointerTarget !== undefined && !pointerTargetIsReachable(pointerTarget, options)) {
          continue;
        }
        hasReachableChannel = true;
        if (pointerTarget === undefined) {
          emitAnimationDiagnostic(warnings, onDiagnostic, {
            severity: 'error',
            code: 'unsupported-animation-target-path',
            path: `animations[${animIdx}].channels[${chIdx}].target.extensions.KHR_animation_pointer.pointer`,
            animationIndex: animIdx,
            channelIndex: chIdx,
            targetPath: 'pointer',
            message:
              `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} targets unsupported ` +
              `KHR_animation_pointer JSON pointer "${String(pointer)}". Supported pointers: ` +
              `${supportedGltfAnimationPointers().join(', ')}. Import rejected.`,
          });
          continue;
        }
        const definitionError = gltfAnimationPointerTargetDefinitionError(gltf, pointerTarget);
        if (definitionError !== undefined) {
          emitAnimationDiagnostic(warnings, onDiagnostic, {
            severity: 'error',
            code: 'animation-pointer-target-undefined',
            path: `animations[${animIdx}].channels[${chIdx}].target.extensions.KHR_animation_pointer.pointer`,
            animationIndex: animIdx,
            channelIndex: chIdx,
            targetPath: 'pointer',
            message:
              `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} pointer "${pointer}" ` +
              `does not resolve to a property defined by this asset: ${definitionError}. Import rejected.`,
          });
          continue;
        }
        const samplerData = decodeSampler(ch.sampler);
        if (!samplerData) continue;
        const { times, values, interpolation } = samplerData;
        const outputAccessorError = gltfAnimationPointerOutputAccessorError(
          gltf,
          pointerTarget,
          gltf.accessors?.[anim.samplers?.[ch.sampler]?.output ?? -1],
        );
        if (outputAccessorError !== undefined) {
          emitAnimationDiagnostic(warnings, onDiagnostic, {
            severity: 'error',
            code: 'invalid-animation-pointer-output-accessor',
            path: `animations[${animIdx}].samplers[${ch.sampler}].output`,
            animationIndex: animIdx,
            channelIndex: chIdx,
            samplerIndex: ch.sampler,
            targetPath: 'pointer',
            message:
              `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} (${pointer}) has an ` +
              `incompatible output accessor: ${outputAccessorError}. Import rejected.`,
          });
          continue;
        }
        const interpolationError = gltfAnimationPointerInterpolationError(pointerTarget, interpolation);
        if (interpolationError !== undefined) {
          emitAnimationDiagnostic(warnings, onDiagnostic, {
            severity: 'error',
            code: 'invalid-animation-pointer-interpolation',
            path: `animations[${animIdx}].samplers[${ch.sampler}].interpolation`,
            animationIndex: animIdx,
            channelIndex: chIdx,
            samplerIndex: ch.sampler,
            targetPath: 'pointer',
            message:
              `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} (${pointer}) targets a ` +
              `property with incompatible interpolation: ${interpolationError}. Import rejected.`,
          });
          continue;
        }
        const cubicFactor = interpolation === 'CUBICSPLINE' ? 3 : 1;
        const componentCount = gltfAnimationPointerTargetComponentCount(gltf, pointerTarget);
        if (componentCount === undefined) continue;
        const pointerStride = times.length * cubicFactor;
        const expected = pointerStride * componentCount;
        const validOutputCount = values.length === expected;
        if (!validOutputCount) {
          emitAnimationDiagnostic(warnings, onDiagnostic, {
            severity: 'error',
            code: 'invalid-animation-output-count',
            path: `animations[${animIdx}].channels[${chIdx}].sampler`,
            animationIndex: animIdx,
            channelIndex: chIdx,
            samplerIndex: ch.sampler,
            targetPath: 'pointer',
            message:
              `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} (${pointer}) has ` +
              `${values.length} output floats but ${times.length} keyframes expect ` +
              `${expected}. Import rejected.`,
          });
          continue;
        }
        const valuesError = gltfAnimationPointerValuesError(
          gltf,
          pointerTarget,
          values,
          times.length,
          interpolation,
        );
        if (valuesError !== undefined) {
          emitAnimationDiagnostic(warnings, onDiagnostic, {
            severity: 'error',
            code: 'invalid-animation-pointer-value',
            path: `animations[${animIdx}].samplers[${ch.sampler}].output`,
            animationIndex: animIdx,
            channelIndex: chIdx,
            samplerIndex: ch.sampler,
            targetPath: 'pointer',
            message:
              `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} (${pointer}) has an invalid ` +
              `animated value: ${valuesError}. Import rejected.`,
          });
          continue;
        }
        const identity = gltfAnimationPointerTargetIdentity(pointerTarget);
        if (!claimAnimationTarget(claimedTargets, identity)) {
          emitAnimationDiagnostic(warnings, onDiagnostic, duplicateTargetDiagnostic(
            label, animIdx, chIdx, 'pointer', pointerTarget.pointer,
          ));
          continue;
        }
        if (times.length > 0) {
          duration = Math.max(duration, times[times.length - 1] ?? 0);
        }
        channels.push(pointerTarget.kind === 'node'
          ? {
              target: { node: animationNodeId(pointerTarget.nodeIndex), path: pointerTarget.path },
              sampler: { times, values, interpolation },
            }
          : {
              target: { node: animationPointerId(pointerTarget.pointer), path: 'pointer', pointer: pointerTarget.pointer },
              sampler: { times, values, interpolation },
            });
        continue;
      }

      const nodeIdx = targetNodeIndex;
      if (reachableNodeIndices !== undefined) {
        if (nodeIdx === undefined || !reachableNodeIndices.has(nodeIdx)) continue;
        hasReachableChannel = true;
      }
      if (nodeIdx === undefined) {
        emitAnimationDiagnostic(warnings, onDiagnostic, {
          severity: 'error',
          code: 'missing-animation-target-node',
          path: `animations[${animIdx}].channels[${chIdx}].target.node`,
          animationIndex: animIdx,
          channelIndex: chIdx,
          targetPath: path,
          message:
            `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} has no target node ` +
            '(extension-targeted channels are not supported). Import rejected.',
        });
        continue;
      }
      if (!gltf.nodes?.[nodeIdx]) {
        emitAnimationDiagnostic(warnings, onDiagnostic, {
          severity: 'error',
          code: 'animation-target-node-not-found',
          path: `animations[${animIdx}].channels[${chIdx}].target.node`,
          animationIndex: animIdx,
          channelIndex: chIdx,
          nodeIndex: nodeIdx,
          targetPath: path,
          message:
            `[vitrum/gltf-adapter] Animation "${label}" channel ${chIdx} targets node ${nodeIdx} ` +
            'which does not exist. Import rejected.',
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
            severity: 'error',
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
              `${times.length * comps * cubicFactor}. Import rejected.`,
          });
          continue;
        }
      } else if (times.length > 0 && values.length % (times.length * cubicFactor) !== 0) {
        emitAnimationDiagnostic(warnings, onDiagnostic, {
          severity: 'error',
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
            `${cubicFactor === 3 ? ' × 3 (CUBICSPLINE)' : ''}. Import rejected.`,
        });
        continue;
      }

      if (!claimAnimationTarget(claimedTargets, gltfNativeAnimationTargetIdentity(nodeIdx, path))) {
        emitAnimationDiagnostic(warnings, onDiagnostic, duplicateTargetDiagnostic(
          label, animIdx, chIdx, path, `nodes[${nodeIdx}].${path}`,
        ));
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
        severity: 'error',
        code: 'dropped-animation',
        path: `animations[${animIdx}]`,
        animationIndex: animIdx,
        message: `[vitrum/gltf-adapter] Animation "${label}" has no importable channels. Import rejected.`,
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

function claimAnimationTarget(
  claimed: GltfAnimationTargetIdentity[],
  candidate: GltfAnimationTargetIdentity,
): boolean {
  const conflict = claimed.some((existing) => gltfAnimationTargetsConflict(existing, candidate));
  if (conflict) return false;
  claimed.push(candidate);
  return true;
}

function duplicateTargetDiagnostic(
  label: string,
  animationIndex: number,
  channelIndex: number,
  targetPath: string,
  targetLabel: string,
): GltfAnimationImportDiagnostic {
  return {
    severity: 'error',
    code: 'duplicate-animation-target',
    path: `animations[${animationIndex}].channels[${channelIndex}].target`,
    animationIndex,
    channelIndex,
    targetPath,
    message:
      `[vitrum/gltf-adapter] Animation "${label}" channel ${channelIndex} targets ` +
      `"${targetLabel}", which conflicts with an earlier channel target. Import rejected.`,
  };
}

function pointerTargetIsReachable(
  target: GltfAnimationPointerTarget,
  options: ConvertAnimationsOptions,
): boolean {
  switch (target.kind) {
    case 'material-property':
    case 'material-texture-transform':
      return options.reachableMaterialIndices?.has(target.materialIndex) ?? true;
    case 'node':
    case 'node-weight':
    case 'node-visibility':
      return options.reachableNodeIndices?.has(target.nodeIndex) ?? true;
    case 'camera':
      return options.reachableCameraIndices?.has(target.cameraIndex) ?? true;
    case 'punctual-light':
      return options.reachablePunctualLightIndices?.has(target.lightIndex) ?? true;
  }
}

function emitAnimationDiagnostic(
  warnings: string[],
  onDiagnostic: GltfAnimationImportDiagnosticSink | undefined,
  diagnostic: GltfAnimationImportDiagnostic,
): never {
  const errorDiagnostic: GltfAnimationImportDiagnostic = {
    ...diagnostic,
    severity: 'error',
  };
  try {
    onDiagnostic?.(errorDiagnostic);
  } catch {
    // A diagnostic observer cannot suppress a strict import failure.
  }
  if (!onDiagnostic) warnings.push(errorDiagnostic.message);
  throw new GltfAnimationImportError(errorDiagnostic);
}
