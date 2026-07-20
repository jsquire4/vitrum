/**
 * paramResolution.ts — pt-webgpu InverseSession parameter path resolution,
 * field validation, and scene read/write (WS5).
 *
 * Extracted verbatim from `inverseSession.ts` (behavior-preserving): every
 * `createInverseSession:` error string, clamp range, and scene read/patch
 * mapping is byte-identical to the pre-split implementation. The session class
 * imports these back and the diagnostics module reuses `ParamSlot` +
 * `findPrimitive` from here.
 */

import type {
  InverseParam,
  MaterialSpec,
  Scene,
  SceneEmitter,
  ScenePrimitive,
  Vec2,
  Vec3,
  BackendSupportMode,
} from '@vitrum/core';
import type { ResolvedParamTarget } from './optimizer.js';

/** One resolved optimized parameter slot in the flat parameter vector. */
export interface ParamSlot {
  readonly param: InverseParam;
  readonly target: ResolvedParamTarget;
  readonly offset: number;
  readonly length: number;
}

export const MATERIAL_RGB_FIELDS = new Set([
  'baseColor',
  'emissive',
  'attenuationColor',
  'specularColor',
  'sheenColor',
  'scatteringCoefficientRGB',
]);
export const MATERIAL_VEC2_FIELDS = new Set([
  'iridescenceThicknessRange',
]);
export const MATERIAL_SCALAR_FIELDS = new Set([
  'roughness',
  'metallic',
  'emissiveIntensity',
  'opacity',
  'alphaCutoff',
  'ior',
  'transmission',
  'thickness',
  'attenuationDistance',
  'specularIntensity',
  'clearcoat',
  'clearcoatRoughness',
  'sheen',
  'sheenRoughness',
  'iridescence',
  'iridescenceIor',
  'anisotropy',
  'anisotropyRotation',
  'normalScale',
  'bumpScale',
  'clearcoatNormalScale',
  'aoMapIntensity',
  'lightMapIntensity',
  'envMapIntensity',
  'dispersionAbbeNumber',
  'scatteringCoefficient',
  'scatteringAnisotropy',
  'displacementScale',
  'displacementBias',
]);
export const EMITTER_RGB_FIELDS = new Set(['color']);
export const EMITTER_SCALAR_FIELDS = new Set(['intensity']);

export function validateParam(
  scene: Scene,
  param: InverseParam,
  target: ResolvedParamTarget,
  materialSupportDetails?: Readonly<Partial<Record<keyof MaterialSpec, BackendSupportMode>>>,
  emitterSupportDetails?: Readonly<Partial<Record<SceneEmitter['kind'], BackendSupportMode>>>,
): void {
  if (param.kind === 'texture') {
    throw new Error(
      `createInverseSession: parameter kind 'texture' (path "${param.path}") is reserved ` +
        'for Phase 2 (texture optimization) and is not yet differentiable in pt-webgpu.',
    );
  }
  if (target.domain === 'materials') {
    const prim = findPrimitive(scene, target.id);
    if (prim == null) {
      throw new Error(
        `createInverseSession: no primitive with id "${target.id}" for path "${param.path}".`,
      );
    }
    const isRgb = MATERIAL_RGB_FIELDS.has(target.field);
    const isVec2 = MATERIAL_VEC2_FIELDS.has(target.field);
    const isScalar = MATERIAL_SCALAR_FIELDS.has(target.field);
    if (!isRgb && !isVec2 && !isScalar) {
      throw new Error(
        `createInverseSession: material field "${target.field}" (path "${param.path}") is not ` +
          `optimizable. Supported: ${[
            ...MATERIAL_RGB_FIELDS,
            ...MATERIAL_VEC2_FIELDS,
            ...MATERIAL_SCALAR_FIELDS,
          ].join(', ')}.`,
      );
    }
    const materialField = target.field as keyof MaterialSpec;
    if (materialSupportDetails?.[materialField] === 'unsupported') {
      throw new Error(
        `createInverseSession: material field "${target.field}" (path "${param.path}") is not ` +
          'optimizable on the active pt-webgpu runtime profile because that profile reports ' +
          'the field as unsupported.',
      );
    }
    assertKind(param, isRgb ? 'rgb' : isVec2 ? 'vec2' : 'scalar');
  } else {
    const emitter = scene.emitters.find((e) => e.id === target.id);
    if (emitter == null) {
      throw new Error(
        `createInverseSession: no emitter with id "${target.id}" for path "${param.path}".`,
      );
    }
    if (emitterSupportDetails?.[emitter.kind] === 'unsupported') {
      throw new Error(
        `createInverseSession: emitter kind "${emitter.kind}" (path "${param.path}") is not ` +
          'optimizable on the active pt-webgpu runtime profile because that profile reports ' +
          'the emitter kind as unsupported.',
      );
    }
    const isRgb = EMITTER_RGB_FIELDS.has(target.field);
    const isScalar = EMITTER_SCALAR_FIELDS.has(target.field);
    if (!isRgb && !isScalar) {
      throw new Error(
        `createInverseSession: emitter field "${target.field}" (path "${param.path}") is not ` +
          `optimizable. Supported: ${[...EMITTER_RGB_FIELDS, ...EMITTER_SCALAR_FIELDS].join(', ')}.`,
      );
    }
    assertKind(param, isRgb ? 'rgb' : 'scalar');
  }
}

/** Field-aware default [min, max] clamp range, used when a parameter doesn't
 *  supply its own `min`/`max`. baseColor / roughness / metallic / emissive
 *  saturate at [0, 1] (physical reflectance / microfacet range);
 *  attenuationColor uses the finite material-packer clamp [1e-4, 1]; emissive
 *  intensity and emitter intensity / color are non-negative but unbounded above
 *  (an explicit `max` from the host narrows them); `ior` is bounded to the
 *  dielectric range [1, 2.5] the material decoder clamps to (material.wgsl.ts:615
 *  `clamp(m2.y, 1.0, 2.5)`) — optimizing outside it would hit a flat clamp and
 *  stall. */
export function defaultClampRange(field: string): [number, number] {
  switch (field) {
    case 'baseColor':
    case 'specularColor':
    case 'sheenColor':
    case 'roughness':
    case 'metallic':
    case 'opacity':
    case 'alphaCutoff':
    case 'transmission':
    case 'specularIntensity':
    case 'clearcoat':
    case 'clearcoatRoughness':
    case 'sheen':
    case 'sheenRoughness':
    case 'iridescence':
    case 'anisotropy':
    case 'aoMapIntensity':
      return [0, 1];
    case 'attenuationColor':
      return [1e-4, 1];
    case 'ior':
      return [1, 2.5];
    case 'attenuationDistance':
      return [1e-6, Infinity];
    case 'dispersionAbbeNumber':
      return [0, Infinity];
    case 'scatteringCoefficient':
      return [0, Infinity];
    case 'scatteringAnisotropy':
      return [-0.95, 0.95];
    case 'iridescenceIor':
      return [1, 3];
    case 'iridescenceThicknessRange':
      return [0, Infinity];
    case 'anisotropyRotation':
    case 'displacementScale':
    case 'displacementBias':
      return [-Infinity, Infinity];
    case 'emissive':
    case 'emissiveIntensity':
    case 'thickness':
    case 'normalScale':
    case 'bumpScale':
    case 'clearcoatNormalScale':
    case 'lightMapIntensity':
    case 'envMapIntensity':
    case 'color':
    case 'intensity':
      return [0, Infinity];
    default:
      return [0, Infinity];
  }
}

function assertKind(param: InverseParam, expected: 'rgb' | 'vec2' | 'scalar'): void {
  if (param.kind !== expected) {
    throw new Error(
      `createInverseSession: parameter "${param.path}" is declared kind '${param.kind}' ` +
        `but the resolved field is '${expected}'.`,
    );
  }
}

export function validateInitialSceneValue(
  slot: ParamSlot,
  value: readonly number[],
  fromExplicitInitial: boolean,
): void {
  if (slot.target.domain !== 'materials' || slot.target.field !== 'attenuationDistance') return;
  const distance = value[0];
  if (Number.isFinite(distance) && distance! > 0) return;
  const source = fromExplicitInitial ? 'initial' : 'scene';
  throw new Error(
    `createInverseSession: parameter "${slot.param.path}" requires a finite positive ${source} ` +
      'attenuationDistance. Undefined or Infinity means "no finite absorbing medium" in the ' +
      'renderer, so pt-webgpu cannot forward-difference this parameter without an explicit ' +
      'finite seed. Set parameter.initial to start fitting a finite medium.',
  );
}

export function findPrimitive(scene: Scene, id: string): ScenePrimitive | null {
  return scene.primitives.find((p) => p.id === id) ?? null;
}

export function readSceneValue(scene: Scene, target: ResolvedParamTarget, length: number): number[] {
  if (target.domain === 'materials') {
    const prim = findPrimitive(scene, target.id)!;
    const m = prim.material;
    switch (target.field) {
      case 'baseColor': return [...m.baseColor];
      case 'roughness': return [m.roughness];
      case 'metallic': return [m.metallic];
      case 'emissive': return [...(m.emissive ?? [0, 0, 0])];
      case 'emissiveIntensity': return [m.emissiveIntensity ?? 1];
      case 'opacity': return [m.opacity ?? 1];
      case 'alphaCutoff': return [m.alphaCutoff ?? 0.5];
      case 'ior': return [m.ior ?? 1.5];
      case 'transmission': return [m.transmission ?? 0];
      case 'thickness': return [m.thickness ?? 0];
      case 'attenuationColor': return [...(m.attenuationColor ?? [1, 1, 1])];
      case 'attenuationDistance': return [m.attenuationDistance ?? Number.POSITIVE_INFINITY];
      case 'dispersionAbbeNumber': return [m.dispersionAbbeNumber ?? 0];
      case 'scatteringCoefficient': return [m.scatteringCoefficient ?? 0];
      case 'scatteringAnisotropy': return [m.scatteringAnisotropy ?? 0];
      case 'scatteringCoefficientRGB': return [...(m.scatteringCoefficientRGB ?? [0, 0, 0])];
      case 'specularColor': return [...(m.specularColor ?? [1, 1, 1])];
      case 'specularIntensity': return [m.specularIntensity ?? 1];
      case 'clearcoat': return [m.clearcoat ?? 0];
      case 'clearcoatRoughness': return [m.clearcoatRoughness ?? 0];
      case 'sheen': return [m.sheen ?? 0];
      case 'sheenColor': return [...(m.sheenColor ?? [1, 1, 1])];
      case 'sheenRoughness': return [m.sheenRoughness ?? 0];
      case 'iridescence': return [m.iridescence ?? 0];
      case 'iridescenceIor': return [m.iridescenceIor ?? 1.3];
      case 'iridescenceThicknessRange': return [...(m.iridescenceThicknessRange ?? [100, 400])];
      case 'anisotropy': return [m.anisotropy ?? 0];
      case 'anisotropyRotation': return [m.anisotropyRotation ?? 0];
      case 'normalScale': return [m.normalScale ?? 1];
      case 'bumpScale': return [m.bumpScale ?? 1];
      case 'clearcoatNormalScale': return [m.clearcoatNormalScale ?? 1];
      case 'aoMapIntensity': return [m.aoMapIntensity ?? 1];
      case 'lightMapIntensity': return [m.lightMapIntensity ?? 1];
      case 'envMapIntensity': return [m.envMapIntensity ?? 1];
      case 'displacementScale': return [m.displacementScale ?? 1];
      case 'displacementBias': return [m.displacementBias ?? 0];
      default: break;
    }
  } else {
    const e = scene.emitters.find((em) => em.id === target.id)!;
    switch (target.field) {
      case 'color': return [...e.color];
      case 'intensity': return [e.intensity];
      default: break;
    }
  }
  // unreachable — validateParam already rejected unknown fields
  return new Array<number>(length).fill(0);
}

export function materialPatch(field: string, value: number[]): Partial<MaterialSpec> {
  switch (field) {
    case 'baseColor': return { baseColor: value as unknown as Vec3 };
    case 'roughness': return { roughness: value[0]! };
    case 'metallic': return { metallic: value[0]! };
    case 'emissive': return { emissive: value as unknown as Vec3 };
    case 'emissiveIntensity': return { emissiveIntensity: value[0]! };
    case 'opacity': return { opacity: value[0]! };
    case 'alphaCutoff': return { alphaCutoff: value[0]! };
    case 'ior': return { ior: value[0]! };
    case 'transmission': return { transmission: value[0]! };
    case 'thickness': return { thickness: value[0]! };
    case 'attenuationColor': return { attenuationColor: value as unknown as Vec3 };
    case 'attenuationDistance': return { attenuationDistance: value[0]! };
    case 'dispersionAbbeNumber': return { dispersionAbbeNumber: value[0]! };
    case 'scatteringCoefficient': return { scatteringCoefficient: value[0]! };
    case 'scatteringAnisotropy': return { scatteringAnisotropy: value[0]! };
    case 'scatteringCoefficientRGB': return { scatteringCoefficientRGB: value as unknown as Vec3 };
    case 'specularColor': return { specularColor: value as unknown as Vec3 };
    case 'specularIntensity': return { specularIntensity: value[0]! };
    case 'clearcoat': return { clearcoat: value[0]! };
    case 'clearcoatRoughness': return { clearcoatRoughness: value[0]! };
    case 'sheen': return { sheen: value[0]! };
    case 'sheenColor': return { sheenColor: value as unknown as Vec3 };
    case 'sheenRoughness': return { sheenRoughness: value[0]! };
    case 'iridescence': return { iridescence: value[0]! };
    case 'iridescenceIor': return { iridescenceIor: value[0]! };
    case 'iridescenceThicknessRange':
      return {
        iridescenceThicknessRange: [
          Math.max(value[0] ?? 100, 0),
          Math.max(value[1] ?? 400, 0),
        ] as unknown as Vec2,
      };
    case 'anisotropy': return { anisotropy: value[0]! };
    case 'anisotropyRotation': return { anisotropyRotation: value[0]! };
    case 'normalScale': return { normalScale: value[0]! };
    case 'bumpScale': return { bumpScale: value[0]! };
    case 'clearcoatNormalScale': return { clearcoatNormalScale: value[0]! };
    case 'aoMapIntensity': return { aoMapIntensity: value[0]! };
    case 'lightMapIntensity': return { lightMapIntensity: value[0]! };
    case 'envMapIntensity': return { envMapIntensity: value[0]! };
    case 'displacementScale': return { displacementScale: value[0]! };
    case 'displacementBias': return { displacementBias: value[0]! };
    default: throw new Error(`inverse: unsupported material field "${field}".`);
  }
}

export function emitterPatch(field: string, value: number[]): Partial<SceneEmitter> {
  switch (field) {
    case 'color': return { color: value as unknown as Vec3 };
    case 'intensity': return { intensity: value[0]! };
    default: throw new Error(`inverse: unsupported emitter field "${field}".`);
  }
}
