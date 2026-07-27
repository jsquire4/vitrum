/**
 * paramResolution.ts — pt-webgpu adapter over the shared inverse scaffolding
 * parameter resolution / field validation / scene read+patch (WS5).
 *
 * The field-metadata table and its four descriptor-driven consumers (kind
 * classification, clamp range, scene read, scene patch) now live in
 * `@vitrum/core` (`inverse-scaffolding.ts`, `MATERIAL_PARAM_DESCRIPTORS` /
 * `EMITTER_PARAM_DESCRIPTORS`) as the single source of truth shared with
 * pt-webgl2. This module re-exports those symbols and adapts the two validators
 * whose signatures carry a per-backend detail: `validateParam` gates
 * material/emitter fields against the active runtime profile's
 * `BackendSupportMode` — that runtime-profile capability gate is a pt-webgpu-only
 * availability flag the shared table carries but pt-webgl2 leaves unset. Scene
 * read/patch and clamp ranges are byte-identical to the pre-split
 * implementation.
 */

import type {
  InverseParam,
  MaterialSpec,
  Scene,
  SceneEmitter,
  BackendSupportMode,
} from '@vitrum/core';
import type { ResolvedParamTarget } from './optimizer.js';
import {
  validateParam as sharedValidateParam,
  validateInitialSceneValue as sharedValidateInitialSceneValue,
} from '@vitrum/core/inverse-scaffolding';

export type { ParamSlot } from '@vitrum/core/inverse-scaffolding';
export {
  MATERIAL_RGB_FIELDS,
  MATERIAL_VEC2_FIELDS,
  MATERIAL_SCALAR_FIELDS,
  EMITTER_RGB_FIELDS,
  EMITTER_SCALAR_FIELDS,
  defaultClampRange,
  findPrimitive,
  readSceneValue,
  materialPatch,
  emitterPatch,
} from '@vitrum/core/inverse-scaffolding';

import type { ParamSlot } from '@vitrum/core/inverse-scaffolding';

/** Validate a resolved parameter for the pt-webgpu runtime: material/emitter
 *  fields are gated against the active
 *  runtime profile's support details (a field/emitter kind reported
 *  `unsupported` is not optimizable). */
export function validateParam(
  scene: Scene,
  param: InverseParam,
  target: ResolvedParamTarget,
  materialSupportDetails?: Readonly<Partial<Record<keyof MaterialSpec, BackendSupportMode>>>,
  emitterSupportDetails?: Readonly<Partial<Record<SceneEmitter['kind'], BackendSupportMode>>>,
): void {
  sharedValidateParam(scene, param, target, {
    backend: 'pt-webgpu',
    ...(materialSupportDetails != null
      ? { materialSupportDetails: materialSupportDetails }
      : {}),
    ...(emitterSupportDetails != null
      ? { emitterSupportDetails: emitterSupportDetails }
      : {}),
  });
}

export function validateInitialSceneValue(
  slot: ParamSlot,
  value: readonly number[],
  fromExplicitInitial: boolean,
): void {
  sharedValidateInitialSceneValue(slot, value, fromExplicitInitial, 'pt-webgpu');
}
