/**
 * pathReplayDiagnostics.ts — the pt-webgpu InverseSession path-replay
 * eligibility diagnostic subsystem (WS5).
 *
 * Extracted verbatim from `inverseSession.ts` (behavior-preserving): the set of
 * pure `pathReplay*` / `diagnosePathReplay*` / material-issue functions that
 * decide whether each optimized slot can route through the scoped single-bounce
 * direct-light path-replay adjoint, and — when it cannot — produce the exact
 * `InverseSessionDiagnostic` a host observes. No dependency on the session-class
 * instance; the class calls `collectPathReplayDiagnostics`,
 * `pathReplayRenderRegimeIssue`, `diagnosePathReplaySlot`, and
 * `isPathReplayZeroGradientSlot`.
 *
 * The 11 near-identical `materialIssueFor*` variants that existed pre-split are
 * collapsed into a single table-driven `materialIssueForField`, parameterized by
 * `{ allowIridescence, allowAnisotropy, primaryHitOnly, unlit }`. Divergent unlit
 * messages are preserved byte-for-byte as explicit table rows (see
 * `MATERIAL_ISSUE_TABLE`), so every emitted diagnostic message + ordering is
 * identical to the pre-collapse implementation.
 */

import type {
  AnalyticPrimitive,
  InverseSessionDiagnostic,
  MaterialSpec,
  Scene,
  SceneEmitter,
  ScenePrimitive,
  BackendSupportMode,
} from '@vitrum/core';
import { analyticPrimitiveToMesh } from '@vitrum/core';
import { luminance, readEnvironmentMapPixels } from '@vitrum/shared-samplers';
import {
  MESH_AREA_LIGHT_TRI_CAP,
  meshAreaEmitterAdjointRangeForScene,
} from '../scene/emitterPacking.js';
import { type ParamSlot, findPrimitive } from './paramResolution.js';

export interface InversePathReplayRenderContext {
  readonly bounces?: number;
  readonly spectral?: boolean;
  readonly bdpt?: boolean;
  readonly restirPtReuse?: boolean;
  readonly causticStrategy?: 'none' | 'manifold-nee' | 'photon-map';
  readonly directLighting?: 'sampled-selection' | 'summed-expectation';
  readonly cameraVisibleEmitters?: boolean;
  readonly implicitEmissiveMeshLights?: boolean;
}

export interface InversePathReplayGeometryCapabilities {
  readonly supportedAnalyticShapes?: ReadonlySet<string>;
}

/**
 * Material fields for which the local adjoint has an implementation route.
 * This table is not a release-support claim: InverseSession intersects it with
 * the end-to-end proof set, which currently contains only material `emissive`.
 *
 *  - `baseColor`, `roughness` — the original Phase-1 BSDF partials
 *    (`dBrdf_dBaseColor` / `dBrdf_dRoughness`). The native full-render gate
 *    currently differs from finite differences by 13–22%, so public sessions
 *    force both fields to finite differences.
 *    `baseColor` also covers baseColorMap / COLOR_0-aware
 *    `shadingModel:'unlit'` primary hits:
 *    forward contributes `throughput · baseColor` and terminates, so the
 *    derivative is the direct contribution-level identity rather than a BRDF
 *    partial or light-domain term.
 *  - `metallic` — the opaque base-BRDF partial through the diffuse fade-out
 *    and F0 blend in the same direct-light replay domain.
 *  - `emissive` / `emissiveIntensity` — the camera-direct emission partials
 *    (`∂rendered_c/∂emissive_c = throughput · emissiveIntensity` and
 *    `∂rendered_c/∂emissiveIntensity = throughput · emissive_c`, scattered at
 *    the PRIMARY hit where the camera sees the emissive surface directly — NOT
 *    a NEE term, so they need no light). `emissive` is GPU-validated end-to-end
 *    by `tools/gpu-env/inverse-fit-deno.ts`: all three channels sign-match the
 *    full-render finite difference with 1.9–2.7% relative error on lavapipe.
 *    The earlier divergent trial scattered emissive
 *    inside the NEE loop / without folding the live emissiveIntensity through the
 *    descriptor; the fix scatters it at the primary hit gated by the matId match
 *    and hands the fixed emissiveIntensity in the descriptor `.w` (bitcast f32).
 *  - `specularColor`, `specularIntensity` — KHR_materials_specular dielectric F0
 *    controls through the same frozen direct-light BRDF derivative. These affect
 *    the diffuse/specular partition and the specular Fresnel colour for
 *    non-metallic and partially-metallic surfaces; fully metallic surfaces still
 *    source F0 from baseColor and therefore have zero derivative here.
 *  - `clearcoat`, `clearcoatRoughness`, `clearcoatNormalScale` —
 *    KHR_materials_clearcoat direct lobe controls. Clearcoat/
 *    clearcoatRoughness maps replay as local scalar chain-rule factors;
 *    clearcoatNormalMap replays its independent tangent-frame normal for the
 *    additive clearcoat lobe.
 *  - `sheen`, `sheenColor`, `sheenRoughness` — KHR_materials_sheen direct lobe
 *    controls through the additive Charlie lobe. Sheen color/roughness maps
 *    replay as local chain-rule factors.
 *  - `iridescence` — KHR_materials_iridescence scalar through the
 *    thin-film-modified base F0 in the same scoped opaque direct-light domain.
 *    Iridescence maps replay as local scalar chain-rule factors.
 *  - `iridescenceIor` and `iridescenceThicknessRange` — scalar thin-film IOR
 *    and authored min/max thickness ranges. Iridescence thickness maps replay by
 *    collapsing the range to the sampled forward value for the BRDF derivative;
 *    thickness-range gradients chain the sampled texel (or map-free `V·H`) back
 *    to the min/max endpoints.
 *  - `aoMapIntensity` — local derivative of glTF AO's
 *    `mix(1, sampledR, intensity)` baseColor multiplier.
 *  - `lightMapIntensity` — additive primary-hit baked-radiance partial:
 *    `∂rendered/∂lightMapIntensity = lightMapRadianceTexel` in the same
 *    camera-direct emission slot as `emissive`, with no direct-light requirement.
 *  - baseColorMap/COLOR_0, roughnessMap/metallicMap, AO, specular
 *    color/intensity maps, clearcoat/sheen maps, iridescence/thickness maps,
 *    anisotropy maps, and light maps are replayed as local chain-rule factors
 *    clearcoat-normal maps for the clearcoat lobe, and light maps are replayed
 *    as local chain-rule factors for the lit BRDF / primary-hit emission domains
 *    above. Additive primary-hit
 *    emissiveMap/lightMap terms are allowed on BRDF/unlit targets because they
 *    do not change the derivative of those optimized fields; dLossDRendered
 *    already contains their forward contribution. Dormant alpha/transmission
 *    maps may stay on path replay when the readable coverage/transport factor
 *    is provably opaque or zero-effective; active alpha visibility,
 *    transmission/thickness transport, and displacement geometry remain
 *    finite-difference fallbacks until those terms are mirrored.
 *  - `anisotropy` / `anisotropyRotation` — map-free scalar anisotropic-GGX
 *    controls through a local symmetric derivative of the direct-light specular
 *    lobe. Anisotropy maps replay the B-channel strength multiplier and RG
 *    rotation offset used by the forward shader.
 *
 * `ior` is deliberately NOT here — it optimizes via finite difference (correct,
 * just slower) and has a GPU-validated analytic partial (`dFrDielectric_dIor` —
 * analytic == FD in isolation, `ADJOINT_EMISSIVE_IOR_FD_WGSL`) ready for a future
 * path-replay wire: `∂evaluateBrdf/∂ior ≡ 0` in the current forward
 * (opaque F0 is controlled by KHR_materials_specular/baseColor, not ior), and the single-bounce
 * adjoint doesn't trace the transmissive Fresnel partition where ior IS
 * differentiable.
 *
 * This is the implementation-capable set, not the release-proof manifest.
 * `inverseSession` intersects it with the engine's end-to-end GPU-fit proof
 * manifest before advertising or dispatching session-level path replay.
 */
const ADJOINT_ELIGIBLE_FIELDS = new Set([
  'baseColor',
  'roughness',
  'metallic',
  'aoMapIntensity',
  'lightMapIntensity',
  'emissive',
  'emissiveIntensity',
  'specularColor',
  'specularIntensity',
  'clearcoat',
  'clearcoatRoughness',
  'sheen',
  'sheenColor',
  'sheenRoughness',
  'iridescence',
  'iridescenceIor',
  'iridescenceThicknessRange',
  'anisotropy',
  'anisotropyRotation',
  'envMapIntensity',
  'normalScale',
  'bumpScale',
  'clearcoatNormalScale',
]);
const ADJOINT_ELIGIBLE_EMITTER_FIELDS = new Set(['color', 'intensity']);
const ADJOINT_ADDITIVE_LOBE_FIELDS = new Set([
  'clearcoat',
  'clearcoatRoughness',
  'sheen',
  'sheenColor',
  'sheenRoughness',
]);
const PATH_REPLAY_TRANSPORT_ONLY_FIELDS = new Set([
  'ior',
  'transmission',
  'thickness',
  'attenuationColor',
  'attenuationDistance',
  'dispersionAbbeNumber',
  'scatteringCoefficient',
  'scatteringAnisotropy',
  'scatteringCoefficientRGB',
]);
const PATH_REPLAY_VISIBILITY_ONLY_FIELDS = new Set(['opacity', 'alphaCutoff']);
const PATH_REPLAY_GEOMETRY_ONLY_FIELDS = new Set(['displacementScale', 'displacementBias']);
const PATH_REPLAY_ALPHA_STABLE_OPAQUE_MARGIN = 1e-3;

type PathReplayUnsupportedCode = InverseSessionDiagnostic['code'];
type PathReplayMaterialIssue = {
  readonly code?: PathReplayUnsupportedCode;
  readonly message: string;
  readonly details: Record<string, string | number | boolean | readonly string[]>;
};

export function collectPathReplayDiagnostics(
  scene: Scene,
  slots: readonly ParamSlot[],
  options: {
    readonly hasHook: boolean;
    readonly iridescenceOptimizedPrimitiveIds: ReadonlySet<string>;
    readonly renderContext: InversePathReplayRenderContext;
    readonly geometryCapabilities: InversePathReplayGeometryCapabilities;
    readonly emitterSupportDetails:
      | Readonly<Partial<Record<SceneEmitter['kind'], BackendSupportMode>>>
      | undefined;
  },
): InverseSessionDiagnostic[] {
  const diagnostics: InverseSessionDiagnostic[] = [];
  if (!options.hasHook) {
    diagnostics.push({
      severity: 'info',
      code: 'path-replay-hook-missing',
      message:
        '[vitrum/pt-webgpu] InverseSession requested path-replay, but this engine instance ' +
        'does not expose the adjoint gradient hook; using finite-difference.',
    });
  }

  const renderRegimeIssue = pathReplayRenderRegimeIssue(options.renderContext);
  if (renderRegimeIssue != null) {
    diagnostics.push({
      severity: 'info',
      code: 'path-replay-unsupported-render-regime',
      message:
        '[vitrum/pt-webgpu] InverseSession requested path-replay, but the most recent ' +
        `${renderRegimeIssue.message}; using finite-difference.`,
      details: renderRegimeIssue.details,
    });
  }

  for (const slot of slots) {
    diagnostics.push(...diagnosePathReplaySlot(
      scene,
      slot,
      options.iridescenceOptimizedPrimitiveIds,
      options.geometryCapabilities,
      options.renderContext,
      options.emitterSupportDetails,
    ));
  }
  return diagnostics;
}

export function pathReplayRenderRegimeIssue(
  context: InversePathReplayRenderContext,
): { readonly message: string; readonly details: Record<string, string | number | boolean> } | null {
  if (context.spectral === true) {
    return {
      message: 'forward baseline used spectral transport that the scoped RGB direct-light adjoint does not mirror',
      details: { spectral: true },
    };
  }
  if (typeof context.bounces === 'number' && Number.isFinite(context.bounces) && context.bounces > 1) {
    return {
      message: `forward baseline used ${context.bounces} bounces while the adjoint pass is single-bounce direct-light only`,
      details: { bounces: context.bounces, supportedBounces: 1 },
    };
  }
  if (context.bdpt === true) {
    return {
      message: 'forward baseline used BDPT contributions that the scoped path-replay adjoint does not mirror',
      details: { bdpt: true, unsupportedFeature: 'bdpt' },
    };
  }
  if (context.restirPtReuse === true) {
    return {
      message: 'forward baseline used ReSTIR-PT reuse contributions that the scoped path-replay adjoint does not mirror',
      details: { restirPtReuse: true, unsupportedFeature: 'restir-pt-reuse' },
    };
  }
  if (context.causticStrategy != null && context.causticStrategy !== 'none') {
    return {
      message: `forward baseline used caustic strategy "${context.causticStrategy}" that the scoped path-replay adjoint does not mirror`,
      details: { causticStrategy: context.causticStrategy, unsupportedFeature: 'caustic-strategy' },
    };
  }
  return null;
}

export function diagnosePathReplaySlot(
  scene: Scene,
  slot: ParamSlot,
  iridescenceOptimizedPrimitiveIds: ReadonlySet<string>,
  geometryCapabilities: InversePathReplayGeometryCapabilities,
  renderContext: InversePathReplayRenderContext,
  emitterSupportDetails?: Readonly<Partial<Record<SceneEmitter['kind'], BackendSupportMode>>>,
): InverseSessionDiagnostic[] {
  const path = slot.param.path;
  const target = slot.target;
  if (target.domain !== 'materials') {
    return diagnosePathReplayEmitterSlot(
      scene,
      slot,
      geometryCapabilities,
      renderContext,
      emitterSupportDetails,
    );
  }
  if (isPathReplayZeroGradientSlot(scene, slot)) {
    return [];
  }
  if (!ADJOINT_ELIGIBLE_FIELDS.has(target.field)) {
    const finiteDifferenceOnlyIssue = pathReplayFiniteDifferenceOnlyFieldIssue(target.field);
    if (finiteDifferenceOnlyIssue != null) {
      return [{
        severity: 'info',
        code: finiteDifferenceOnlyIssue.code,
        path,
        message:
          `[vitrum/pt-webgpu] InverseSession path "${path}" targets material field ` +
          `"${target.field}", ${finiteDifferenceOnlyIssue.message}; using finite-difference.`,
        details: finiteDifferenceOnlyIssue.details,
      }];
    }
    return [{
      severity: 'info',
      code: 'path-replay-unsupported-field',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" targets material field ` +
        `"${target.field}", which is not in the path-replay differentiable field set; ` +
        'using finite-difference.',
      details: { field: target.field },
    }];
  }

  const prim = findPrimitive(scene, target.id);
  if (prim == null) return [];
  const primitiveIssue = pathReplayPrimitiveIssue(prim, geometryCapabilities);
  if (primitiveIssue != null) {
    return [{
      severity: 'info',
      code: 'path-replay-unsupported-primitive',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" targets primitive "${target.id}", ` +
        `${primitiveIssue.message}; using finite-difference.`,
      details: primitiveIssue.details,
    }];
  }
  const sceneGeometryIssue = pathReplaySceneGeometryIssue(scene, geometryCapabilities);
  if (sceneGeometryIssue != null) {
    return [{
      severity: 'info',
      code: 'path-replay-unsupported-scene-geometry',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" needs whole-scene triangle replay, ` +
        `${sceneGeometryIssue.message}; using finite-difference.`,
      details: sceneGeometryIssue.details,
    }];
  }

  const materialIssue = pathReplayMaterialIssue(
    scene,
    prim,
    target.field,
    iridescenceOptimizedPrimitiveIds.has(target.id),
    renderContext,
  );
  if (materialIssue != null) {
    return [{
      severity: 'info',
      code: materialIssue.code ?? 'path-replay-unsupported-material',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" is outside the scoped path-replay ` +
        `material domain (${materialIssue.message}); using finite-difference.`,
      details: materialIssue.details,
    }];
  }

  if (pathReplayTargetRequiresLighting(target.field, prim)) {
    const lightingIssue = pathReplayLightingIssue(scene, renderContext, emitterSupportDetails);
    if (lightingIssue != null) {
      return [{
        severity: 'info',
        code: lightingIssue.code ?? 'path-replay-unsupported-lighting',
        path,
        message:
          `[vitrum/pt-webgpu] InverseSession path "${path}" needs the direct-light replay domain, ` +
          `${lightingIssue.message}; using finite-difference.`,
        details: lightingIssue.details,
      }];
    }
  }
  return [];
}

export function isPathReplayZeroGradientSlot(scene: Scene, slot: ParamSlot): boolean {
  const target = slot.target;
  if (target.domain !== 'materials') return false;
  if (target.field !== 'opacity' && target.field !== 'alphaCutoff') return false;
  const prim = findPrimitive(scene, target.id);
  if (prim == null) return false;
  if ((prim.material.alphaMode ?? 'opaque') === 'opaque') return true;
  return pathReplayMaskCoverageIsStablyOpaque(prim.material, prim);
}

function pathReplayFiniteDifferenceOnlyFieldIssue(
  field: string,
): {
  readonly code:
    | 'path-replay-unsupported-transport'
    | 'path-replay-unsupported-visibility'
    | 'path-replay-unsupported-geometry';
  readonly message: string;
  readonly details: Record<string, string | readonly string[]>;
} | null {
  if (PATH_REPLAY_TRANSPORT_ONLY_FIELDS.has(field)) {
    return {
      code: 'path-replay-unsupported-transport',
      message:
        'which changes transmissive/medium transport that the scoped path-replay adjoint does not mirror yet',
      details: {
        field,
        finiteDifferenceReason: 'transport',
        affectedTerms: ['fresnel-partition', 'refraction-direction', 'medium-attenuation'],
      },
    };
  }
  if (PATH_REPLAY_VISIBILITY_ONLY_FIELDS.has(field)) {
    return {
      code: 'path-replay-unsupported-visibility',
      message:
        'which changes alpha coverage and visibility discontinuities that the scoped path-replay adjoint does not mirror yet',
      details: {
        field,
        finiteDifferenceReason: 'visibility',
        affectedTerms: ['alpha-coverage', 'ray-visibility', 'shadow-visibility'],
      },
    };
  }
  if (PATH_REPLAY_GEOMETRY_ONLY_FIELDS.has(field)) {
    return {
      code: 'path-replay-unsupported-geometry',
      message:
        'which changes displacement geometry that the scoped path-replay adjoint does not mirror yet',
      details: {
        field,
        finiteDifferenceReason: 'geometry',
        affectedTerms: ['micro-displacement', 'bvh-geometry', 'visibility'],
      },
    };
  }
  return null;
}

function diagnosePathReplayEmitterSlot(
  scene: Scene,
  slot: ParamSlot,
  geometryCapabilities: InversePathReplayGeometryCapabilities,
  renderContext: InversePathReplayRenderContext,
  emitterSupportDetails?: Readonly<Partial<Record<SceneEmitter['kind'], BackendSupportMode>>>,
): InverseSessionDiagnostic[] {
  const path = slot.param.path;
  const target = slot.target;
  if (!ADJOINT_ELIGIBLE_EMITTER_FIELDS.has(target.field)) {
    return [{
      severity: 'info',
      code: 'path-replay-unsupported-field',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" targets emitter field ` +
        `"${target.field}", which is not in the path-replay differentiable emitter field set; ` +
        'using finite-difference.',
      details: { field: target.field },
    }];
  }

  const emitter = scene.emitters.find((e) => e.id === target.id);
  if (emitter == null) return [];
  const emitterIssue = pathReplayEmitterTargetIssue(scene, emitter, target.field);
  if (emitterIssue != null) {
    return [{
      severity: 'info',
      code: 'path-replay-unsupported-emitter',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" targets emitter "${target.id}", ` +
        `${emitterIssue.message}; using finite-difference.`,
      details: emitterIssue.details,
    }];
  }

  const lightingIssue = pathReplayLightingIssue(scene, renderContext, emitterSupportDetails);
  if (lightingIssue != null) {
    return [{
      severity: 'info',
      code: lightingIssue.code ?? 'path-replay-unsupported-lighting',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" needs the direct-light replay domain, ` +
        `${lightingIssue.message}; using finite-difference.`,
      details: lightingIssue.details,
    }];
  }

  const receiverIssue = pathReplayEmitterReceiverSceneIssue(scene, geometryCapabilities);
  if (receiverIssue != null) {
    return [{
      severity: 'info',
      code: receiverIssue.code ?? 'path-replay-unsupported-receiver',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" differentiates an emitter through ` +
        `scene receivers, but ${receiverIssue.message}; using finite-difference.`,
      details: receiverIssue.details,
    }];
  }
  return [];
}

function pathReplayPrimitiveIssue(
  primitive: ScenePrimitive,
  geometryCapabilities: InversePathReplayGeometryCapabilities,
): { message: string; details: Record<string, string | number | boolean> } | null {
  if (primitive.kind === 'analytic') {
    return pathReplayAnalyticPrimitiveIssue(primitive, geometryCapabilities);
  }
  if (primitive.kind !== 'mesh' && primitive.kind !== 'skinned-mesh' && primitive.kind !== 'instanced-mesh') {
    const primitiveKind = (primitive as { readonly kind: string }).kind;
    return {
      message: `primitive kind "${primitiveKind}" is not triangle-backed for path-replay`,
      details: { primitiveKind },
    };
  }
  if (primitive.kind === 'instanced-mesh' && primitive.instances.length === 0) {
    return {
      message: 'instanced-mesh target has zero instances in the replayed scene',
      details: { primitiveKind: primitive.kind, instanceCount: 0 },
    };
  }
  return null;
}

function pathReplaySceneGeometryIssue(
  scene: Scene,
  geometryCapabilities: InversePathReplayGeometryCapabilities,
): { message: string; details: Record<string, string | number | boolean> } | null {
  for (const primitive of scene.primitives) {
    if (primitive.kind === 'mesh' || primitive.kind === 'skinned-mesh') continue;
    if (primitive.kind === 'instanced-mesh') continue;
    if (primitive.kind === 'analytic') {
      const analyticIssue = pathReplayAnalyticPrimitiveIssue(primitive, geometryCapabilities);
      if (analyticIssue == null) continue;
      return {
        message:
          `scene primitive "${primitive.id}" has analytic shape "${primitive.shape}", ` +
          analyticIssue.message,
        details: { primitiveId: primitive.id, primitiveKind: primitive.kind, ...analyticIssue.details },
      };
    }
    const primitiveId = (primitive as { readonly id: string }).id;
    const primitiveKind = (primitive as { readonly kind: string }).kind;
    return {
      message: `scene primitive "${primitiveId}" has kind "${primitiveKind}", outside the flat triangle replay domain`,
      details: { primitiveId, primitiveKind },
    };
  }
  return null;
}

function pathReplayAnalyticPrimitiveIssue(
  primitive: AnalyticPrimitive,
  geometryCapabilities: InversePathReplayGeometryCapabilities,
): { message: string; details: Record<string, string | number | boolean> } | null {
  const supportedAnalyticShapes = geometryCapabilities.supportedAnalyticShapes;
  if (supportedAnalyticShapes == null || !supportedAnalyticShapes.has(primitive.shape)) {
    return {
      message:
        `analytic shape "${primitive.shape}" has no exact path-replay geometry implementation on this engine tier`,
      details: {
        primitiveKind: primitive.kind,
        analyticShape: primitive.shape,
        finiteDifferenceReason: 'analytic-shape-exact-replay-unavailable',
      },
    };
  }

  try {
    const mesh = analyticPrimitiveToMesh(primitive, { preferFallbackMesh: false });
    const vertexCount = Math.floor(mesh.positions.length / 3);
    const triangleCount = Math.floor((mesh.indices?.length ?? vertexCount) / 3);
    if (triangleCount <= 0) {
      return {
        message: `analytic shape "${primitive.shape}" produced empty replay geometry`,
        details: {
          primitiveKind: primitive.kind,
          analyticShape: primitive.shape,
          finiteDifferenceReason: 'analytic-replay-geometry-empty',
        },
      };
    }
  } catch {
    return {
      message: `analytic shape "${primitive.shape}" could not produce exact path-replay geometry`,
      details: {
        primitiveKind: primitive.kind,
        analyticShape: primitive.shape,
        finiteDifferenceReason: 'analytic-replay-geometry-failed',
      },
    };
  }
  return null;
}

function pathReplayEmitterTargetIssue(
  scene: Scene,
  emitter: SceneEmitter,
  field: string,
): { message: string; details: Record<string, string | number | readonly string[]> } | null {
  switch (emitter.kind) {
    case 'directional': {
      return null;
    }
    case 'point':
    case 'spot':
    case 'rect-area':
    case 'disc-area':
      return null;
    case 'mesh-area': {
      const range = meshAreaEmitterAdjointRangeForScene(scene, emitter.id);
      if (range == null) {
        return {
          message: 'mesh-area emitter target produces no contiguous packed triangle range',
          details: { emitterKind: emitter.kind },
        };
      }
      if (range.capped) {
        return {
          message:
            'mesh-area emitter target exceeds the exact adjoint replay triangle capacity',
          details: {
            emitterKind: emitter.kind,
            triangleCount: range.totalMeshAreaTriangles,
            triangleLimit: MESH_AREA_LIGHT_TRI_CAP,
            finiteDifferenceReason: 'mesh-area-replay-capacity',
          },
        };
      }
      const mappedEmissionIssue = meshAreaEmitterMappedEmissionIssue(scene, emitter, field);
      if (mappedEmissionIssue != null) return mappedEmissionIssue;
      return null;
    }
    default: {
      const emitterKind = (emitter as { readonly kind: string }).kind;
      return {
        message: `emitter kind "${emitterKind}" is outside the path-replay target domain`,
        details: { emitterKind },
      };
    }
  }
}

function meshAreaEmitterMappedEmissionIssue(
  scene: Scene,
  emitter: Extract<SceneEmitter, { readonly kind: 'mesh-area' }>,
  field: string,
): { message: string; details: Record<string, string | number | readonly string[]> } | null {
  const primitive = scene.primitives.find((p) => p.id === emitter.meshId);
  if (primitive == null || primitive.kind === 'analytic') return null;
  if (primitive.material.emissiveMap == null) return null;
  if (field === 'color' || field === 'intensity') {
    return null;
  }
  return {
    message:
      'mesh-area emitter target uses material emissiveMap radiance outside the scoped emitter color/intensity replay fields',
    details: {
      emitterKind: emitter.kind,
      meshId: emitter.meshId,
      unsupportedMaterialFields: ['emissiveMap'],
    },
  };
}

function pathReplayEmitterReceiverSceneIssue(
  scene: Scene,
  geometryCapabilities: InversePathReplayGeometryCapabilities,
): PathReplayMaterialIssue | null {
  for (const primitive of scene.primitives) {
    const primitiveIssue = pathReplayPrimitiveIssue(primitive, geometryCapabilities);
    if (primitiveIssue != null) {
      return {
        message: `receiver primitive "${primitive.id}" ${primitiveIssue.message}`,
        details: { primitiveId: primitive.id, ...primitiveIssue.details },
      };
    }
    const materialIssue = pathReplayEmitterReceiverMaterialIssue(primitive);
    if (materialIssue != null) {
      return {
        ...(materialIssue.code != null ? { code: materialIssue.code } : {}),
        message: `receiver primitive "${primitive.id}" has material outside the scoped direct-light replay domain (${materialIssue.message})`,
        details: { primitiveId: primitive.id, ...materialIssue.details },
      };
    }
  }
  return null;
}

function pathReplayEmitterReceiverMaterialIssue(
  primitive: ScenePrimitive,
): PathReplayMaterialIssue | null {
  const material = primitive.material;
  const common = materialIssueCommon(material, { allowIridescence: true, allowAnisotropy: true }, primitive);
  if (common != null) return common;
  const maps = listPathReplayTransportOrGeometryMaps(material);
  if (maps.length > 0) {
    return materialMapIssue(maps);
  }
  return null;
}

function pathReplayMaterialIssue(
  scene: Scene,
  primitive: ScenePrimitive,
  field: string,
  iridescenceCoupled: boolean,
  renderContext: InversePathReplayRenderContext,
): PathReplayMaterialIssue | null {
  const material = primitive.material;
  if (field === 'baseColor' && material.shadingModel === 'unlit') {
    return materialIssueForPrimaryHit(material, primitive);
  }
  if (field === 'emissive' || field === 'emissiveIntensity') {
    const foldIssue = pathReplayMaterialEmissiveFoldIssue(scene, primitive, renderContext);
    if (foldIssue != null) return foldIssue;
    return materialIssueForField(material, primitive, MATERIAL_ISSUE_TABLE.emissive);
  }
  if (field === 'aoMapIntensity') {
    return materialIssueForField(material, primitive, MATERIAL_ISSUE_TABLE.aoMapIntensity);
  }
  if (field === 'lightMapIntensity') {
    return materialIssueForField(material, primitive, MATERIAL_ISSUE_TABLE.lightMapIntensity);
  }
  if (field === 'envMapIntensity') {
    return materialIssueForField(material, primitive, MATERIAL_ISSUE_TABLE.envMapIntensity);
  }
  if (field === 'iridescence' || field === 'iridescenceIor' || field === 'iridescenceThicknessRange') {
    return materialIssueForField(material, primitive, MATERIAL_ISSUE_TABLE.iridescence);
  }
  if (field === 'anisotropy' || field === 'anisotropyRotation') {
    return materialIssueForField(material, primitive, MATERIAL_ISSUE_TABLE.anisotropy);
  }
  if (field === 'normalScale') {
    return materialIssueForField(material, primitive, MATERIAL_ISSUE_TABLE.normalScale);
  }
  if (field === 'bumpScale') {
    return materialIssueForField(material, primitive, MATERIAL_ISSUE_TABLE.bumpScale);
  }
  if (field === 'clearcoatNormalScale') {
    return materialIssueForField(material, primitive, MATERIAL_ISSUE_TABLE.clearcoatNormalScale);
  }
  if (iridescenceCoupled) {
    return {
      message: 'another optimized parameter on this material targets iridescence, which is coupled to this BRDF field',
      details: { reason: 'coupled-iridescence-parameter' },
    };
  }
  if (ADJOINT_ADDITIVE_LOBE_FIELDS.has(field)) {
    return materialIssueForField(material, primitive, MATERIAL_ISSUE_TABLE.additiveLobe);
  }
  return materialIssueForField(material, primitive, MATERIAL_ISSUE_TABLE.brdf);
}

function pathReplayMaterialEmissiveFoldIssue(
  scene: Scene,
  primitive: ScenePrimitive,
  renderContext: InversePathReplayRenderContext,
): PathReplayMaterialIssue | null {
  if (renderContext.cameraVisibleEmitters !== true) return null;
  const foldedEmitter = scene.emitters.find((emitter) =>
    emitter.kind === 'mesh-area' &&
    emitter.meshId === primitive.id &&
    directEmitterContributes(scene, emitter)
  );
  if (foldedEmitter == null) return null;
  return {
    code: 'path-replay-unsupported-material',
    message:
      'material emissive is overwritten by camera-visible mesh-area emitter folding; optimize the emitter color/intensity target instead',
    details: {
      primitiveId: primitive.id,
      emitterId: foldedEmitter.id,
      emitterKind: foldedEmitter.kind,
      finiteDifferenceReason: 'mesh-area-emissive-fold',
    },
  };
}

/**
 * Table-driven replacement for the 11 near-identical `materialIssueFor*`
 * variants. Each row is byte-equivalent to the pre-split dedicated function:
 *
 *  - `primaryHitOnly` rows (emissive, lightMapIntensity) route straight to
 *    `materialIssueForPrimaryHit` — no unlit special-case, no common/map check.
 *  - the remaining rows carry an `unlit` behavior (`'primary-hit'` → delegate to
 *    `materialIssueForPrimaryHit`, or a verbatim unlit message object) plus the
 *    `{ allowIridescence, allowAnisotropy }` flags fed to `materialIssueCommon`,
 *    followed by the shared map check.
 *
 * The divergent unlit messages are preserved exactly (they are observable), so
 * this collapse does not change any emitted diagnostic string or ordering.
 */
interface MaterialIssueSpec {
  readonly primaryHitOnly?: true;
  readonly allowIridescence?: boolean;
  readonly allowAnisotropy?: boolean;
  /** Either delegate to primary-hit on unlit, or emit a verbatim unlit issue. */
  readonly unlit?: 'primary-hit' | PathReplayMaterialIssue;
}

const MATERIAL_ISSUE_TABLE = {
  emissive: { primaryHitOnly: true },
  lightMapIntensity: { primaryHitOnly: true },
  brdf: {
    allowIridescence: true,
    allowAnisotropy: true,
    unlit: { message: 'unlit materials only support path-replay for baseColor primary-hit fitting', details: { reason: 'unlit' } },
  },
  additiveLobe: {
    allowIridescence: true,
    allowAnisotropy: true,
    unlit: { message: 'unlit materials do not evaluate clearcoat/sheen direct-light lobes', details: { reason: 'unlit' } },
  },
  aoMapIntensity: {
    allowIridescence: false,
    allowAnisotropy: true,
    unlit: 'primary-hit',
  },
  envMapIntensity: {
    allowIridescence: true,
    allowAnisotropy: true,
    unlit: { message: 'unlit materials do not evaluate environment direct-light BRDF intensity', details: { reason: 'unlit' } },
  },
  iridescence: {
    allowIridescence: true,
    allowAnisotropy: false,
    unlit: { message: 'unlit materials do not evaluate the iridescence direct-light lobe', details: { reason: 'unlit' } },
  },
  anisotropy: {
    allowIridescence: false,
    allowAnisotropy: true,
    unlit: { message: 'unlit materials do not evaluate the anisotropic direct-light lobe', details: { reason: 'unlit' } },
  },
  normalScale: {
    allowIridescence: true,
    allowAnisotropy: true,
    unlit: { message: 'unlit materials do not evaluate the normal-mapped direct-light lobe', details: { reason: 'unlit' } },
  },
  bumpScale: {
    allowIridescence: true,
    allowAnisotropy: true,
    unlit: { message: 'unlit materials do not evaluate the bump-mapped direct-light lobe', details: { reason: 'unlit' } },
  },
  clearcoatNormalScale: {
    allowIridescence: true,
    allowAnisotropy: true,
    unlit: { message: 'unlit materials do not evaluate the clearcoat direct-light lobe', details: { reason: 'unlit' } },
  },
} as const satisfies Record<string, MaterialIssueSpec>;

function materialIssueForField(
  material: MaterialSpec,
  primitive: ScenePrimitive,
  spec: MaterialIssueSpec,
): PathReplayMaterialIssue | null {
  if (spec.primaryHitOnly === true) {
    return materialIssueForPrimaryHit(material, primitive);
  }
  if (material.shadingModel === 'unlit') {
    if (spec.unlit === 'primary-hit') {
      return materialIssueForPrimaryHit(material, primitive);
    }
    if (spec.unlit != null) {
      return spec.unlit;
    }
  }
  const common = materialIssueCommon(
    material,
    { allowIridescence: spec.allowIridescence ?? false, allowAnisotropy: spec.allowAnisotropy ?? false },
    primitive,
  );
  if (common != null) return common;
  const maps = listPathReplayTransportOrGeometryMaps(material);
  if (maps.length > 0) {
    return materialMapIssue(maps);
  }
  return null;
}

function materialIssueForPrimaryHit(
  material: MaterialSpec,
  primitive: ScenePrimitive,
): PathReplayMaterialIssue | null {
  const alphaVisibilityIssue = pathReplayAlphaVisibilityIssue(material, primitive);
  if (alphaVisibilityIssue != null) return alphaVisibilityIssue;
  if (material.displacementMap != null) {
    return materialMapIssue(['displacementMap']);
  }
  if (material.extensions != null && Object.keys(material.extensions).length > 0) {
    return { message: 'opaque MaterialSpec.extensions are not replayed by the adjoint pass', details: { field: 'extensions' } };
  }
  return null;
}

function materialIssueCommon(
  material: MaterialSpec,
  options: { readonly allowIridescence: boolean; readonly allowAnisotropy: boolean },
  primitive: ScenePrimitive,
): PathReplayMaterialIssue | null {
  const alphaVisibilityIssue = pathReplayAlphaVisibilityIssue(material, primitive);
  if (alphaVisibilityIssue != null) return alphaVisibilityIssue;
  if (pathReplayEffectiveTransmissionMayTransport(material)) {
    return {
      code: 'path-replay-unsupported-transport',
      message: 'transmission transport is not replayed',
      details: {
        field: 'transmission',
        value: material.transmission ?? 0,
        finiteDifferenceReason: 'transport',
        affectedTerms: ['fresnel-partition', 'refraction-direction', 'medium-attenuation'],
      },
    };
  }
  if (!options.allowIridescence && (material.iridescence ?? 0) > 0) {
    return { message: 'iridescence is coupled to the optimized BRDF field', details: { field: 'iridescence', value: material.iridescence ?? 0 } };
  }
  if (!options.allowAnisotropy && (material.anisotropy ?? 0) > 0) {
    return { message: 'anisotropy is coupled to the optimized BRDF field', details: { field: 'anisotropy', value: material.anisotropy ?? 0 } };
  }
  if (pathReplayLayeredMaterialAffectsBrdf(material)) {
    return {
      code: 'path-replay-unsupported-transport',
      message: 'layered/thin-film material stacks are not replayed',
      details: {
        field: 'layeredMaterial',
        finiteDifferenceReason: 'transport',
        affectedTerms: ['layer-selection', 'thin-film-phase', 'transmission-mis'],
      },
    };
  }
  if (pathReplaySpectralOrDispersionAffectsTransport(material)) {
    return {
      code: 'path-replay-unsupported-transport',
      message: 'spectral/dispersion material transport is not replayed',
      details: {
        field: 'spectralOrDispersion',
        finiteDifferenceReason: 'transport',
        affectedTerms: ['hero-wavelength', 'spectral-attenuation', 'dispersion-ior'],
      },
    };
  }
  if (pathReplayScatteringAffectsTransport(material)) {
    return {
      code: 'path-replay-unsupported-transport',
      message: 'volume/scattering material transport is not replayed',
      details: {
        field: 'scattering',
        finiteDifferenceReason: 'transport',
        affectedTerms: ['volume-walk', 'medium-scattering', 'phase-function'],
      },
    };
  }
  if (material.extensions != null && Object.keys(material.extensions).length > 0) {
    return { message: 'opaque MaterialSpec.extensions are not replayed by the adjoint pass', details: { field: 'extensions' } };
  }
  return null;
}

function pathReplayAlphaVisibilityIssue(
  material: MaterialSpec,
  primitive: ScenePrimitive,
): PathReplayMaterialIssue | null {
  const alphaMode = material.alphaMode ?? 'opaque';
  if (alphaMode === 'opaque') return null;

  const opacity = material.opacity ?? 1;
  const alphaCutoff = material.alphaCutoff ?? 0.5;
  const coverage = pathReplayAlphaCoverage(material, primitive);
  if (alphaMode === 'mask' && coverage.known && coverage.min >= alphaCutoff + PATH_REPLAY_ALPHA_STABLE_OPAQUE_MARGIN) {
    return null;
  }
  if (alphaMode === 'blend' && coverage.known && coverage.min >= 1 - PATH_REPLAY_ALPHA_STABLE_OPAQUE_MARGIN) {
    return null;
  }

  const alphaInputs = [...coverage.affectedInputs];
  if (alphaMode === 'mask' && opacity < alphaCutoff) alphaInputs.push('opacity<alphaCutoff');
  if (alphaMode === 'blend' && opacity < 1) alphaInputs.push('opacity<1');

  if (alphaInputs.length === 0) return null;

  return {
    code: 'path-replay-unsupported-visibility',
    message: `alphaMode "${alphaMode}" changes visibility/coverage`,
    details: {
      field: 'alphaMode',
      value: alphaMode,
      opacity,
      alphaCutoff,
      finiteDifferenceReason: 'visibility',
      affectedInputs: alphaInputs,
      affectedTerms: ['alpha-coverage', 'ray-visibility', 'shadow-visibility'],
    },
  };
}

function pathReplayMaskCoverageIsStablyOpaque(material: MaterialSpec, primitive: ScenePrimitive): boolean {
  if ((material.alphaMode ?? 'opaque') !== 'mask') return false;
  const coverage = pathReplayAlphaCoverage(material, primitive);
  return coverage.known &&
    coverage.min >= (material.alphaCutoff ?? 0.5) + PATH_REPLAY_ALPHA_STABLE_OPAQUE_MARGIN;
}

function pathReplayAlphaCoverage(
  material: MaterialSpec,
  primitive: ScenePrimitive,
): { readonly known: boolean; readonly min: number; readonly affectedInputs: readonly string[] } {
  const inputs: string[] = [];
  let known = true;
  let min = material.opacity ?? 1;

  const baseColorAlpha = textureChannelMinimum(material.baseColorMap, 3, 'baseColorMap.a');
  if (!baseColorAlpha.known) known = false;
  min *= baseColorAlpha.min;
  inputs.push(...baseColorAlpha.affectedInputs);

  const alphaMap = textureChannelMinimum(material.alphaMap, 0, 'alphaMap');
  if (!alphaMap.known) known = false;
  min *= alphaMap.min;
  inputs.push(...alphaMap.affectedInputs);

  const vertexAlpha = primitiveVertexAlphaMinimum(primitive);
  min *= vertexAlpha.min;
  inputs.push(...vertexAlpha.affectedInputs);

  return { known, min, affectedInputs: inputs };
}

function asTextureHandle(value: unknown): unknown {
  if (value == null) return null;
  if (typeof value === 'object' && 'handle' in value) {
    return (value as { readonly handle?: unknown }).handle ?? null;
  }
  return value;
}

function textureChannelMinimum(
  value: unknown,
  channel: number,
  affectedInput: string,
): { readonly known: boolean; readonly min: number; readonly affectedInputs: readonly string[] } {
  const stats = textureChannelStats(value, channel, affectedInput);
  return { known: stats.known, min: stats.min, affectedInputs: stats.affectedInputs };
}

function textureChannelMaximum(
  value: unknown,
  channel: number,
  affectedInput: string,
): { readonly known: boolean; readonly max: number; readonly affectedInputs: readonly string[] } {
  const stats = textureChannelStats(value, channel, affectedInput);
  return { known: stats.known, max: stats.max, affectedInputs: stats.affectedInputs };
}

function textureChannelStats(
  value: unknown,
  channel: number,
  affectedInput: string,
): { readonly known: boolean; readonly min: number; readonly max: number; readonly affectedInputs: readonly string[] } {
  if (value == null) return { known: true, min: 1, max: 1, affectedInputs: [] };
  const handle = asTextureHandle(value);
  if (handle == null || typeof handle !== 'object') {
    return { known: false, min: 0, max: 1, affectedInputs: [affectedInput] };
  }
  const h = handle as {
    readonly width?: number;
    readonly height?: number;
    readonly data?: ArrayLike<number>;
    readonly image?: { readonly width?: number; readonly height?: number; readonly data?: ArrayLike<number> };
  };
  const src = h.data ?? h.image?.data;
  const width = Math.floor(Number(h.width ?? h.image?.width ?? 0));
  const height = Math.floor(Number(h.height ?? h.image?.height ?? 0));
  if (src == null || typeof src.length !== 'number' || width <= 0 || height <= 0) {
    return { known: false, min: 0, max: 1, affectedInputs: [affectedInput] };
  }

  const pixelCount = width * height;
  if (pixelCount <= 0 || src.length < pixelCount) {
    return { known: false, min: 0, max: 1, affectedInputs: [affectedInput] };
  }
  const hint = textureHint(handle);
  const heuristicStride = src.length / pixelCount;
  const stride = hint?.channels ?? heuristicStride;
  if (!Number.isInteger(stride) || stride <= 0 || stride > 4) {
    return { known: false, min: 0, max: 1, affectedInputs: [affectedInput] };
  }
  if (channel >= stride) return { known: true, min: 1, max: 1, affectedInputs: [] };

  let min = 1;
  let max = 0;
  for (let p = 0; p < pixelCount; p += 1) {
    const sample = decodedUnitAlpha(Number(src[p * stride + channel] ?? 1), src, hint?.dataType);
    if (!Number.isFinite(sample)) {
      return { known: false, min: 0, max: 1, affectedInputs: [affectedInput] };
    }
    const clamped = Math.max(0, Math.min(1, sample));
    min = Math.min(min, clamped);
    max = Math.max(max, clamped);
  }
  return {
    known: true,
    min,
    max,
    affectedInputs: min < 1 - PATH_REPLAY_ALPHA_STABLE_OPAQUE_MARGIN ? [affectedInput] : [],
  };
}

function textureHint(handle: unknown): {
  readonly channels?: number;
  readonly dataType?: string;
} | undefined {
  if (handle == null || typeof handle !== 'object') return undefined;
  const h = handle as {
    readonly __vitrum_hint__?: { readonly channels?: number; readonly dataType?: string };
    readonly channels?: number;
    readonly dataType?: string;
  };
  return h.__vitrum_hint__ ?? (
    h.channels != null || h.dataType != null
      ? {
          ...(h.channels != null ? { channels: h.channels } : {}),
          ...(h.dataType != null ? { dataType: h.dataType } : {}),
        }
      : undefined
  );
}

function decodedUnitAlpha(value: number, src: ArrayLike<number>, hintDataType: string | undefined): number {
  if (hintDataType === 'float32' || src instanceof Float32Array || src instanceof Float64Array) return value;
  if (hintDataType === 'uint16' || src instanceof Uint16Array) return value / 65535;
  if (hintDataType === 'uint8' || src instanceof Uint8Array || src instanceof Uint8ClampedArray) return value / 255;
  if (src instanceof Uint32Array) return value / 4294967295;
  if (src instanceof Int16Array) return value / 32767;
  if (src instanceof Int32Array) return value / 2147483647;
  const bpe = (src as { readonly BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
  const max = 2 ** (8 * bpe) - 1;
  return max > 0 ? value / max : value;
}

function primitiveVertexAlphaMinimum(
  primitive: ScenePrimitive,
): { readonly min: number; readonly affectedInputs: readonly string[] } {
  if (primitive.kind === 'analytic') return { min: 1, affectedInputs: [] };
  const colors = primitive.colors;
  if (colors == null || colors.length === 0) return { min: 1, affectedInputs: [] };
  const vertexCount = Math.floor(primitive.positions.length / 3);
  if (vertexCount <= 0 || colors.length < vertexCount * 4) return { min: 1, affectedInputs: [] };
  let min = 1;
  for (let i = 0; i < vertexCount; i += 1) {
    const alpha = colors[i * 4 + 3] ?? 1;
    if (!Number.isFinite(alpha)) return { min: 0, affectedInputs: ['COLOR_0.a'] };
    min = Math.min(min, Math.max(0, Math.min(1, alpha)));
  }
  return {
    min,
    affectedInputs: min < 1 - PATH_REPLAY_ALPHA_STABLE_OPAQUE_MARGIN ? ['COLOR_0.a'] : [],
  };
}

function materialMapIssue(maps: readonly string[]): PathReplayMaterialIssue {
  const categories = new Set(maps.map(pathReplayMaterialMapCategory));
  const details: Record<string, string | readonly string[]> = { unsupportedMaterialFields: maps };
  if (categories.size === 1) {
    const category = categories.values().next().value as PathReplayMaterialMapCategory;
    switch (category) {
      case 'transport':
        details.finiteDifferenceReason = 'transport';
        details.affectedTerms = ['fresnel-partition', 'refraction-direction', 'medium-attenuation'];
        return {
          code: 'path-replay-unsupported-transport',
          message: `transport maps are not replayed: ${maps.join(', ')}`,
          details,
        };
      case 'visibility':
        details.finiteDifferenceReason = 'visibility';
        details.affectedTerms = ['alpha-coverage', 'ray-visibility', 'shadow-visibility'];
        return {
          code: 'path-replay-unsupported-visibility',
          message: `visibility maps are not replayed: ${maps.join(', ')}`,
          details,
        };
      case 'normal':
        details.finiteDifferenceReason = 'normal';
        details.affectedTerms = ['normal-map-frame', 'bump-gradient', 'clearcoat-normal-frame'];
        return {
          code: 'path-replay-unsupported-normal',
          message: `normal maps are not replayed: ${maps.join(', ')}`,
          details,
        };
      case 'geometry':
        details.finiteDifferenceReason = 'geometry';
        details.affectedTerms = ['micro-displacement', 'bvh-geometry', 'visibility'];
        return {
          code: 'path-replay-unsupported-geometry',
          message: `geometry maps are not replayed: ${maps.join(', ')}`,
          details,
        };
    }
  }
  details.finiteDifferenceReason = 'mixed-material-domain';
  return {
    code: 'path-replay-unsupported-material',
    message: `mixed transport/visibility/normal/geometry maps are not replayed: ${maps.join(', ')}`,
    details,
  };
}

type PathReplayMaterialMapCategory = 'transport' | 'visibility' | 'normal' | 'geometry';

function pathReplayMaterialMapCategory(field: string): PathReplayMaterialMapCategory {
  switch (field) {
    case 'transmissionMap':
    case 'thicknessMap':
      return 'transport';
    case 'alphaMap':
      return 'visibility';
    case 'normalMap':
    case 'bumpMap':
    case 'clearcoatNormalMap':
      return 'normal';
    case 'displacementMap':
    default:
      return 'geometry';
  }
}

function pathReplayTargetRequiresLighting(field: string, primitive: ScenePrimitive): boolean {
  const material = primitive.material;
  if (field === 'emissive' || field === 'emissiveIntensity') return false;
  if (field === 'lightMapIntensity') return false;
  if (field === 'baseColor' && isPathReplayCompatibleUnlitBaseColorMaterial(primitive)) return false;
  if (field === 'aoMapIntensity' && material.shadingModel === 'unlit') return false;
  return true;
}

function pathReplayLightingIssue(
  scene: Scene,
  context: InversePathReplayRenderContext,
  emitterSupportDetails?: Readonly<Partial<Record<SceneEmitter['kind'], BackendSupportMode>>>,
): {
  code?: InverseSessionDiagnostic['code'];
  message: string;
  details: Record<string, string | number | readonly string[]>;
} | null {
  const unsupportedByProfile = scene.emitters.find((e) =>
    emitterSupportDetails?.[e.kind] === 'unsupported' &&
    directEmitterContributes(scene, e)
  );
  if (unsupportedByProfile != null) {
    return {
      message:
        `emitter "${unsupportedByProfile.id}" (${unsupportedByProfile.kind}) is unsupported by ` +
        'the active pt-webgpu runtime profile',
      details: {
        emitterId: unsupportedByProfile.id,
        emitterKind: unsupportedByProfile.kind,
        supportMode: 'unsupported',
      },
    };
  }

  const unsupported = (scene.emitters as unknown as ReadonlyArray<{
    readonly id: string;
    readonly kind: string;
    readonly angularDiameter?: number;
    readonly color?: readonly number[];
    readonly intensity?: number;
  }>).find((e) =>
    e.kind !== 'directional' &&
    e.kind !== 'point' &&
    e.kind !== 'spot' &&
    e.kind !== 'rect-area' &&
    e.kind !== 'disc-area' &&
    e.kind !== 'mesh-area' &&
    directEmitterContributes(scene, e)
  );
  if (unsupported != null) {
    return {
      message: `emitter "${unsupported.id}" (${unsupported.kind}) is outside the deterministic direct-light replay domain`,
      details: {
        emitterId: unsupported.id,
        emitterKind: unsupported.kind,
        ...(unsupported.kind === 'directional' && unsupported.angularDiameter != null
          ? { angularDiameter: unsupported.angularDiameter }
          : {}),
      },
    };
  }

  const environmentIssue = pathReplayEnvironmentIssue(scene.environment);
  if (environmentIssue != null) {
    return environmentIssue;
  }

  const candidates = directLightCandidateLabels(scene, context);
  if (candidates.length > 1 && context.directLighting !== 'summed-expectation') {
    return {
      code: 'path-replay-unsupported-light-selection',
      message:
        'scene has multiple contributing direct-light candidates, but the forward context uses sampled ' +
        'light selection; the scoped adjoint sums the replay domain and only matches baselines that also ' +
        'sum direct-light expectations',
      details: {
        candidateCount: candidates.length,
        candidates,
        directLighting: context.directLighting ?? 'sampled-selection',
      },
    };
  }
  return null;
}

function pathReplayEnvironmentIssue(environment: {
  readonly kind?: string;
  readonly intensity?: number;
} | undefined): {
  code: InverseSessionDiagnostic['code'];
  message: string;
  details: Record<string, string | number | readonly string[]>;
} | null {
  if (environment == null || environment.kind == null || environment.kind === 'none') return null;
  if (!((environment.intensity ?? 1) > 0)) return null;
  if (environment.kind === 'hdri' || environment.kind === 'procedural-sky') return null;
  return {
    code: 'path-replay-unsupported-environment',
    message:
      `environment kind "${environment.kind}" is outside the scoped path-replay environment replay domain`,
    details: {
      environmentKind: environment.kind,
      supportedEnvironmentKinds: ['none', 'hdri', 'procedural-sky'],
    },
  };
}

function directLightCandidateLabels(
  scene: Scene,
  context: InversePathReplayRenderContext,
): readonly string[] {
  const candidates: string[] = [];
  for (const emitter of scene.emitters as unknown as ReadonlyArray<{
    readonly id?: string | number;
    readonly kind: string;
    readonly color?: readonly number[];
    readonly intensity?: number;
    readonly meshId?: string;
    readonly normal?: readonly number[];
    readonly radius?: number;
  }>) {
    if (!directEmitterContributes(scene, emitter)) continue;
    candidates.push(`emitter:${String(emitter.id ?? '(unnamed)')}:${emitter.kind}`);
  }
  for (const primitive of scene.primitives) {
    if (!implicitEmissiveMeshContributes(scene, primitive, context)) continue;
    candidates.push(`implicit-emissive-mesh:${primitive.id}`);
  }
  if (environmentContributesDirectLight(scene.environment)) {
    candidates.push(`environment:${scene.environment.kind}`);
  }
  return candidates;
}

function implicitEmissiveMeshContributes(
  scene: Scene,
  primitive: ScenePrimitive,
  context: InversePathReplayRenderContext,
): boolean {
  if (context.implicitEmissiveMeshLights !== true) return false;
  if (!isTriangleBackedPrimitiveForReplay(primitive)) return false;
  if (!materialHasPositiveImplicitEmission(primitive.material)) return false;
  const explicitMeshEmitter = scene.emitters.some((emitter) =>
    emitter.kind === 'mesh-area' &&
    emitter.meshId === primitive.id &&
    directEmitterContributes(scene, emitter)
  );
  if (explicitMeshEmitter) return false;
  return true;
}

function isTriangleBackedPrimitiveForReplay(primitive: ScenePrimitive): boolean {
  return primitive.kind === 'mesh' ||
    primitive.kind === 'skinned-mesh' ||
    primitive.kind === 'instanced-mesh';
}

function materialHasPositiveImplicitEmission(material: MaterialSpec): boolean {
  const intensity = material.emissiveIntensity ?? 1;
  if (!(intensity > 0)) return false;
  if (material.emissiveMap != null) return true;
  const emissive = material.emissive ?? [0, 0, 0];
  return ((emissive[0] ?? 0) > 0 || (emissive[1] ?? 0) > 0 || (emissive[2] ?? 0) > 0);
}

function directEmitterContributes(scene: Scene, emitter: {
  readonly id?: string | number;
  readonly kind?: string;
  readonly color?: readonly number[];
  readonly intensity?: number;
  readonly normal?: readonly number[];
  readonly radius?: number;
  readonly uAxis?: readonly number[];
  readonly vAxis?: readonly number[];
}): boolean {
  const intensity = emitter.intensity ?? 1;
  if (!(intensity > 0)) return false;
  const color = emitter.color ?? [1, 1, 1];
  if (!((color[0] ?? 0) > 0 || (color[1] ?? 0) > 0 || (color[2] ?? 0) > 0)) return false;
  if (emitter.kind === 'mesh-area') {
    return meshAreaEmitterAdjointRangeForScene(scene, String(emitter.id ?? '')) != null;
  }
  if (emitter.kind === 'disc-area') {
    if (!Number.isFinite(emitter.radius) || (emitter.radius ?? 0) < 1e-8) return false;
    const n = emitter.normal ?? [0, 0, 0];
    const nLen = Math.hypot(n[0] ?? 0, n[1] ?? 0, n[2] ?? 0);
    if (!Number.isFinite(nLen) || nLen < 1e-8) return false;
  }
  if (emitter.kind === 'rect-area') {
    const u = emitter.uAxis ?? [0, 0, 0];
    const v = emitter.vAxis ?? [0, 0, 0];
    const cx = (u[1] ?? 0) * (v[2] ?? 0) - (u[2] ?? 0) * (v[1] ?? 0);
    const cy = (u[2] ?? 0) * (v[0] ?? 0) - (u[0] ?? 0) * (v[2] ?? 0);
    const cz = (u[0] ?? 0) * (v[1] ?? 0) - (u[1] ?? 0) * (v[0] ?? 0);
    const area = 4 * Math.hypot(cx, cy, cz);
    if (!Number.isFinite(area) || area < 1e-8) return false;
  }
  return true;
}

function environmentContributesDirectLight(environment: {
  readonly kind?: string;
  readonly hdri?: unknown;
  readonly intensity?: number;
} | undefined): boolean {
  if (environment == null || environment.kind == null || environment.kind === 'none') return false;
  if (!((environment.intensity ?? 1) > 0)) return false;
  if (environment.kind === 'hdri') {
    return hdriHasPositiveLuminance(environment.hdri);
  }
  return true;
}

function hdriHasPositiveLuminance(handle: unknown): boolean {
  const hdri = readEnvironmentMapPixels(handle);
  if (hdri == null) return false;
  const width = Math.floor(hdri.width);
  const height = Math.floor(hdri.height);
  const data = hdri.data;
  if (width <= 0 || height <= 0 || data.length < width * height * 4) return false;
  let totalWeight = 0;
  for (let i = 0; i < width * height; i += 1) {
    const r = Number(data[i * 4] ?? 0);
    const g = Number(data[i * 4 + 1] ?? 0);
    const b = Number(data[i * 4 + 2] ?? 0);
    const y = (i / width) | 0;
    const theta = ((y + 0.5) / height) * Math.PI;
    totalWeight += Math.max(0, luminance(r, g, b) * Math.sin(theta));
    if (totalWeight > 1e-12) return true;
  }
  return false;
}

function isPathReplayCompatibleUnlitBaseColorMaterial(primitive: ScenePrimitive): boolean {
  const m = primitive.material;
  if (m.shadingModel !== 'unlit') return false;
  if (pathReplayAlphaVisibilityIssue(m, primitive) != null) return false;
  if (pathReplayEffectiveTransmissionMayTransport(m)) return false;
  if (pathReplayLayeredMaterialAffectsBrdf(m)) return false;
  if (pathReplaySpectralOrDispersionAffectsTransport(m)) return false;
  if (pathReplayScatteringAffectsTransport(m)) return false;
  if (m.extensions != null && Object.keys(m.extensions).length > 0) return false;
  return !hasPathReplayTransportOrGeometryMap(m);
}

function hasPathReplayTransportOrGeometryMap(m: MaterialSpec): boolean {
  return listPathReplayTransportOrGeometryMaps(m).length > 0;
}

function listPathReplayTransportOrGeometryMaps(m: MaterialSpec): readonly string[] {
  const out: string[] = [];
  if (pathReplayTransmissionMapAffectsTransport(m)) out.push('transmissionMap');
  if (pathReplayThicknessMapAffectsTransport(m)) out.push('thicknessMap');
  if (pathReplayAlphaMapAffectsVisibility(m)) out.push('alphaMap');
  if (m.displacementMap != null) out.push('displacementMap');
  return out;
}

function pathReplayTransmissionMapAffectsTransport(m: MaterialSpec): boolean {
  if (m.transmissionMap == null || (m.transmission ?? 0) <= 0) return false;
  const transmission = textureChannelMaximum(m.transmissionMap, 0, 'transmissionMap');
  return !transmission.known || transmission.max > 0;
}

function pathReplayThicknessMapAffectsTransport(m: MaterialSpec): boolean {
  return m.thicknessMap != null && pathReplayEffectiveTransmissionMayTransport(m);
}

function pathReplaySpectralOrDispersionAffectsTransport(m: MaterialSpec): boolean {
  return pathReplayEffectiveTransmissionMayTransport(m) &&
    (m.spectralAttenuation != null || m.dispersionAbbeNumber != null);
}

function pathReplayScatteringAffectsTransport(m: MaterialSpec): boolean {
  if (!pathReplayEffectiveTransmissionMayTransport(m)) return false;
  if ((m.scatteringCoefficient ?? 0) > 0) return true;
  const rgb = m.scatteringCoefficientRGB;
  return rgb != null && ((rgb[0] ?? 0) > 0 || (rgb[1] ?? 0) > 0 || (rgb[2] ?? 0) > 0);
}

function pathReplayEffectiveTransmissionMayTransport(m: MaterialSpec): boolean {
  const scalar = m.transmission ?? 0;
  if (scalar <= 0) return false;
  if (m.transmissionMap == null) return true;
  const transmission = textureChannelMaximum(m.transmissionMap, 0, 'transmissionMap');
  return !transmission.known || scalar * transmission.max > 0;
}

function pathReplayLayeredMaterialAffectsBrdf(m: MaterialSpec): boolean {
  return pathReplayLayerAffectsBrdf(m.frontLayer) ||
    pathReplayLayerAffectsBrdf(m.backLayer) ||
    ((m.thinFilmStack?.layers.length ?? 0) > 0);
}

function pathReplayLayerAffectsBrdf(layer: MaterialSpec['frontLayer']): boolean {
  if (layer == null) return false;
  const tx = layer.transmission;
  return Math.abs((tx[0] ?? 1) - 1) > 1e-6 ||
    Math.abs((tx[1] ?? 1) - 1) > 1e-6 ||
    Math.abs((tx[2] ?? 1) - 1) > 1e-6 ||
    layer.roughness != null ||
    layer.normalMap != null;
}

function pathReplayAlphaMapAffectsVisibility(m: MaterialSpec): boolean {
  if (m.alphaMap == null || (m.alphaMode ?? 'opaque') === 'opaque') return false;
  const coverage = textureChannelMinimum(m.alphaMap, 0, 'alphaMap');
  return !coverage.known || coverage.min < 1 - PATH_REPLAY_ALPHA_STABLE_OPAQUE_MARGIN;
}
