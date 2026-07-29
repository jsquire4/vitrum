/**
 * Fail-closed eligibility checks for pt-webgpu's certified inverse replay path.
 *
 * Production path replay differentiates exactly one quantity: RGB
 * `MaterialSpec.emissive` at an opaque, triangle-backed primary hit. It replays
 * one-bounce RGB visibility and applies the analytic identity
 * `d rendered.rgb / d emissive.rgb = emissiveIntensity`. Every BSDF, light,
 * emitter, mapped-emission, analytic-shape, layered, spectral, and multi-bounce
 * case is outside this route. Users can select the separate explicit
 * `finite-difference` inverse method for those cases.
 */
import type {
  InverseSessionDiagnostic,
  Scene,
} from '@vitrum/core';
import { type ParamSlot, findPrimitive } from './paramResolution.js';
import {
  emissiveReplayPrimitiveIssue,
  emissiveReplaySceneIssue,
  emissiveReplayTargetIssue,
} from './emissivePathReplayDomain.js';

export interface InversePathReplayRenderContext {
  readonly bounces?: number;
  readonly spectral?: boolean;
  readonly bdpt?: boolean;
  readonly restirPtReuse?: boolean;
  readonly causticStrategy?: 'none' | 'manifold-nee' | 'photon-map';
  readonly cameraVisibleEmitters?: boolean;
}

export function collectPathReplayDiagnostics(
  scene: Scene,
  slots: readonly ParamSlot[],
  options: {
    readonly hasHook: boolean;
    readonly renderContext: InversePathReplayRenderContext;
  },
): InverseSessionDiagnostic[] {
  const diagnostics: InverseSessionDiagnostic[] = [];
  if (!options.hasHook) {
    diagnostics.push({
      severity: 'warning',
      code: 'path-replay-hook-missing',
      message:
        '[vitrum/pt-webgpu] InverseSession requested path-replay, but this engine instance does not expose the certified emissive adjoint hook.',
    });
  }

  const renderRegimeIssue = pathReplayRenderRegimeIssue(options.renderContext);
  if (renderRegimeIssue != null) {
    diagnostics.push({
      severity: 'warning',
      code: 'path-replay-unsupported-render-regime',
      message:
        `[vitrum/pt-webgpu] InverseSession requested path-replay, but ${renderRegimeIssue.message}.`,
      details: renderRegimeIssue.details,
    });
  }

  for (const slot of slots) {
    diagnostics.push(...diagnosePathReplaySlot(scene, slot));
  }
  return diagnostics;
}

export function pathReplayRenderRegimeIssue(
  context: InversePathReplayRenderContext,
): {
  readonly message: string;
  readonly details: Record<string, string | number | boolean>;
} | null {
  if (context.bounces !== 1) {
    return {
      message:
        context.bounces == null
          ? 'the forward bounce count was not supplied, so one-bounce equivalence cannot be proven'
          : `the forward baseline used ${context.bounces} bounces while emissive replay requires exactly one`,
      details: {
        ...(context.bounces == null ? {} : { bounces: context.bounces }),
        supportedBounces: 1,
        missingRenderContext: context.bounces == null,
      },
    };
  }
  if (context.spectral !== false) {
    return {
      message:
        context.spectral == null
          ? 'the spectral render mode was not supplied, so RGB equivalence cannot be proven'
          : 'the forward baseline used spectral transport that RGB emissive replay does not mirror',
      details: {
        spectral: context.spectral === true,
        missingRenderContext: context.spectral == null,
        unsupportedFeature: 'spectral',
      },
    };
  }
  if (context.bdpt !== false) {
    return {
      message:
        context.bdpt == null
          ? 'the BDPT render mode was not supplied, so estimator equivalence cannot be proven'
          : 'the forward baseline used BDPT contributions that emissive replay does not mirror',
      details: {
        bdpt: context.bdpt === true,
        missingRenderContext: context.bdpt == null,
        unsupportedFeature: 'bdpt',
      },
    };
  }
  if (context.restirPtReuse !== false) {
    return {
      message:
        context.restirPtReuse == null
          ? 'the ReSTIR-PT render mode was not supplied, so estimator equivalence cannot be proven'
          : 'the forward baseline used ReSTIR-PT reuse that emissive replay does not mirror',
      details: {
        restirPtReuse: context.restirPtReuse === true,
        missingRenderContext: context.restirPtReuse == null,
        unsupportedFeature: 'restir-pt-reuse',
      },
    };
  }
  if (context.causticStrategy !== 'none') {
    return {
      message:
        context.causticStrategy == null
          ? 'the caustic strategy was not supplied, so estimator equivalence cannot be proven'
          : `the forward baseline used caustic strategy "${context.causticStrategy}" that emissive replay does not mirror`,
      details: {
        ...(context.causticStrategy == null
          ? {}
          : { causticStrategy: context.causticStrategy }),
        missingRenderContext: context.causticStrategy == null,
        unsupportedFeature: 'caustic-strategy',
      },
    };
  }
  if (context.cameraVisibleEmitters !== true) {
    return {
      message:
        context.cameraVisibleEmitters == null
          ? 'camera-visible emitters were not supplied, so primary emissive-hit equivalence cannot be proven'
          : 'the forward baseline suppressed camera-visible primitive emission while emissive replay differentiates primary emissive hits',
      details: {
        cameraVisibleEmitters: false,
        missingRenderContext: context.cameraVisibleEmitters == null,
        requiredFeature: 'camera-visible-emitters',
      },
    };
  }
  return null;
}

function diagnosePathReplaySlot(
  scene: Scene,
  slot: ParamSlot,
): InverseSessionDiagnostic[] {
  const path = slot.param.path;
  const target = slot.target;
  if (target.domain !== 'materials') {
    return [{
      severity: 'warning',
      code: 'path-replay-unsupported-param-domain',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" targets an emitter; certified path replay accepts only material emissive RGB.`,
      details: {
        domain: target.domain,
        field: target.field,
      },
    }];
  }
  if (target.field !== 'emissive' || slot.length !== 3) {
    return [{
      severity: 'warning',
      code: 'path-replay-unsupported-field',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" is not material emissive RGB, the sole certified path-replay field.`,
      details: {
        domain: target.domain,
        field: target.field,
        expectedField: 'emissive',
        expectedLength: 3,
      },
    }];
  }

  const primitive = findPrimitive(scene, target.id);
  if (primitive == null) return [];
  const targetPrimitiveIssue = emissiveReplayPrimitiveIssue(primitive);
  if (targetPrimitiveIssue != null) {
    return [{
      severity: 'warning',
      code: targetPrimitiveIssue.code,
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" is outside exact emissive replay: ${targetPrimitiveIssue.message}.`,
      details: targetPrimitiveIssue.details,
    }];
  }

  const sceneIssue = emissiveReplaySceneIssue(scene);
  if (sceneIssue != null) {
    return [{
      severity: 'warning',
      code: sceneIssue.code,
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" is outside exact emissive replay: ${sceneIssue.message}.`,
      details: sceneIssue.details,
    }];
  }

  const targetIssue = emissiveReplayTargetIssue(scene, primitive);
  if (targetIssue != null) {
    return [{
      severity: 'warning',
      code: targetIssue.code,
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" is outside exact emissive replay: ${targetIssue.message}.`,
      details: targetIssue.details,
    }];
  }
  return [];
}
