// backendCompatibilityReconcile.ts — post-import + post-texture-decode reconciliation
// of per-backend compatibility (D15-7). Extracted verbatim from assetLoader.ts; the
// two public entry points (reconcileBackendCompatibilityAfterSceneImport /
// reconcileBackendCompatibilityAfterTextureDecode) plus the two vertex-color bake
// helpers the loaders call directly are exported. The texture issue-name predicates
// are imported from the shared compatibilityIssuePredicates module (I4-2 / D15-8).

import {
  getPrimitiveActiveColorSet,
  type Scene,
  type ScenePrimitive,
} from '@vitrum/core';
import type { GltfJson } from './gltfTypes.js';
import type {
  GltfBackendCompatibility,
  GltfBackendPolicy,
  GltfCompatibilityIssue,
  GltfFeatureReport,
} from './featureReport.js';
import type {
  DecodeSceneTextureDiagnostic,
  GltfBackendTextureStatus,
  GltfTextureDecodeReport,
} from './texturePipeline.js';
import { resolveGltfAnimationPointer } from './animationPointer.js';
import {
  collectGltfSceneReachability,
  collectPrimitiveMaterialIndices,
  gltfPrimitiveKey,
} from './sceneScope.js';
import {
  TEXTURE_DECODE_DIAGNOSTIC_ISSUE_PREFIX,
  TEXTURE_READINESS_ISSUE_PREFIX,
  isTextureDecodeDiagnosticIssue,
  isTextureReadinessIssue,
} from './compatibilityIssuePredicates.js';

const SPEC_GLOSS_ALPHA_ISSUE =
  'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha';
const EMISSIVE_MAP_TEXEL_PDF_ISSUE = 'emissiveMap.texelPdf';
const DEGRADED_TEXTURE_DECODE_DIAGNOSTICS: ReadonlySet<DecodeSceneTextureDiagnostic['code']> = new Set([
  'decoded-texture-exceeds-max-size',
  'decoded-texture-npot-repeat-wrap',
]);

export function reconcileBackendCompatibilityAfterSceneImport(
  compatibility: readonly GltfBackendCompatibility[],
  scene: Scene,
  textureDecodeReport: GltfTextureDecodeReport,
  allowLiteVertexColorBake: boolean,
): readonly GltfBackendCompatibility[] {
  const liteVertexColorsBakeable = allowLiteVertexColorBake && sceneHasOnlyPtWebgpuLiteBakeableVertexColors(scene);

  return compatibility.map((candidate) => {
    if (candidate.profileId !== 'pt-webgpu-lite' || !liteVertexColorsBakeable) {
      return compatibilityWithTextureReadiness(candidate, textureDecodeReport);
    }
    const issues = candidate.issues.filter((issue) =>
      !(issue.category === 'primitive' && issue.name === 'vertexColors')
    );
    return compatibilityWithTextureReadiness(
      issues.length === candidate.issues.length ? candidate : compatibilityWithIssues(candidate, issues),
      textureDecodeReport,
    );
  });
}

function sceneHasOnlyPtWebgpuLiteBakeableVertexColors(scene: Scene): boolean {
  for (const primitive of scene.primitives) {
    const colors = getPrimitiveActiveColorSet(primitive);
    if (colors != null && colors.length > 0 && !ptWebgpuLiteCanBakeVertexColors(primitive)) {
      return false;
    }
  }
  return true;
}

export function hasReachableMaterialPointerAnimationForColoredPrimitive(
  gltf: GltfJson,
  reachability: ReturnType<typeof collectGltfSceneReachability>,
): boolean {
  const coloredMaterialIndices = new Set<number>();
  for (const meshIndex of reachability.meshIndices) {
    const mesh = gltf.meshes?.[meshIndex];
    if (mesh == null) continue;
    for (const [primitiveIndex, primitive] of (mesh.primitives ?? []).entries()) {
      if (!reachability.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))) continue;
      if (primitive.attributes?.COLOR_0 === undefined) continue;
      for (const materialIndex of collectPrimitiveMaterialIndices(primitive)) {
        coloredMaterialIndices.add(materialIndex);
      }
    }
  }
  if (coloredMaterialIndices.size === 0) return false;

  for (const animation of gltf.animations ?? []) {
    for (const channel of animation.channels ?? []) {
      if (channel.target.path !== 'pointer') continue;
      const pointerTarget = resolveGltfAnimationPointer(
        channel.target.extensions?.KHR_animation_pointer?.pointer,
      );
      if (
        pointerTarget !== undefined &&
        (pointerTarget.kind === 'material-property' || pointerTarget.kind === 'material-texture-transform') &&
        coloredMaterialIndices.has(pointerTarget.materialIndex)
      ) {
        return true;
      }
    }
  }
  return false;
}

export function bakePtWebgpuLiteCompatibleVertexColors(scene: Scene): Scene {
  let changed = false;
  const primitives = scene.primitives.map((primitive): ScenePrimitive => {
    if (primitive.kind === 'analytic') return primitive;
    const color = ptWebgpuLiteBakeableVertexColor(primitive);
    if (color == null) return primitive;
    changed = true;
    const { colors: _legacyColor0, ...withoutLegacyColor0 } = primitive;
    const material = primitive.material;
    return {
      ...withoutLegacyColor0,
      vertexColorSet: null,
      material: {
        ...material,
        baseColor: [
          material.baseColor[0] * color[0],
          material.baseColor[1] * color[1],
          material.baseColor[2] * color[2],
        ],
      },
    };
  });
  return changed ? { ...scene, primitives } : scene;
}

function ptWebgpuLiteBakeableVertexColor(primitive: ScenePrimitive): readonly [number, number, number] | null {
  const colors = getPrimitiveActiveColorSet(primitive);
  if (colors == null || colors.length === 0) return null;
  if (!ptWebgpuLiteCanBakeVertexColors(primitive)) return null;
  return [colors[0] ?? 1, colors[1] ?? 1, colors[2] ?? 1];
}

function ptWebgpuLiteCanBakeVertexColors(primitive: ScenePrimitive): boolean {
  const colors = getPrimitiveActiveColorSet(primitive);
  const positions = (primitive as { readonly positions?: Float32Array }).positions;
  if (colors == null || colors.length === 0) return true;
  if (positions == null || positions.length === 0) return false;
  const vertexCount = Math.floor(positions.length / 3);
  const stride = colors.length >= vertexCount * 4
    ? 4
    : colors.length >= vertexCount * 3
      ? 3
      : 0;
  if (vertexCount === 0 || stride === 0) return false;
  const r = colors[0] ?? 1;
  const g = colors[1] ?? 1;
  const b = colors[2] ?? 1;
  const eps = 1e-6;
  for (let i = 0; i < vertexCount; i += 1) {
    const o = i * stride;
    if (
      Math.abs((colors[o] ?? 1) - r) > eps ||
      Math.abs((colors[o + 1] ?? 1) - g) > eps ||
      Math.abs((colors[o + 2] ?? 1) - b) > eps
    ) {
      return false;
    }
    if (stride === 4 && Math.abs((colors[o + 3] ?? 1) - 1) > eps) {
      return false;
    }
  }
  return true;
}

export function reconcileBackendCompatibilityAfterTextureDecode(
  compatibility: readonly GltfBackendCompatibility[],
  report: GltfTextureDecodeReport,
  diagnostics: readonly DecodeSceneTextureDiagnostic[],
  featureReport: GltfFeatureReport,
  backendPolicy: GltfBackendPolicy,
): readonly GltfBackendCompatibility[] {
  const reconciled = compatibility.map((candidate) => {
    const issues = candidate.issues.filter((issue) => {
      if (isTextureReadinessIssue(issue)) return false;
      if (isTextureDecodeDiagnosticIssue(issue)) return false;
      if (specGlossAlphaIssueSatisfiedByDecode(issue, report, diagnostics, featureReport)) return false;
      if (emissiveTexelPdfIssueSatisfiedByDecode(issue, candidate, report)) return false;
      return true;
    });
    const decodeDiagnosticIssues = textureDecodeDiagnosticIssuesForCandidate(
      issues.length === candidate.issues.length ? candidate : compatibilityWithIssues(candidate, issues),
      diagnostics,
    );
    return compatibilityWithTextureReadiness(
      compatibilityWithIssues(
        issues.length === candidate.issues.length ? candidate : compatibilityWithIssues(candidate, issues),
        [...issues, ...decodeDiagnosticIssues],
      ),
      report,
    );
  });
  return rerankBackendCompatibility(reconciled, backendPolicy);
}

function specGlossAlphaIssueSatisfiedByDecode(
  issue: GltfCompatibilityIssue,
  report: GltfTextureDecodeReport,
  diagnostics: readonly DecodeSceneTextureDiagnostic[],
  featureReport: GltfFeatureReport,
): boolean {
  if (issue.name !== SPEC_GLOSS_ALPHA_ISSUE) return false;
  const paths = featureReport.materials.issuePaths.specGlossGlossinessAlpha;
  const requiredPaths = paths !== undefined && paths.length > 0 ? paths : [issue.path];
  return requiredPaths.every((path) =>
    !diagnostics.some((diagnostic) =>
      diagnostic.code === 'spec-gloss-alpha-bake-unavailable' &&
      diagnostic.path === path
    ) &&
    report.entries.some((entry) =>
      entry.materialField === 'roughnessMap' &&
      entry.path === path &&
      entry.handleKind === 'pixel-data' &&
      entry.handleColorSpace === 'linear'
    )
  );
}

function emissiveTexelPdfIssueSatisfiedByDecode(
  issue: GltfCompatibilityIssue,
  candidate: GltfBackendCompatibility,
  report: GltfTextureDecodeReport,
): boolean {
  if (
    issue.name !== EMISSIVE_MAP_TEXEL_PDF_ISSUE ||
    issue.support !== 'approximate' ||
    (candidate.profileId !== 'pt-webgl2' && candidate.profileId !== 'pt-webgpu')
  ) {
    return false;
  }
  const entries = report.entries.filter((entry) => entry.materialField === 'emissiveMap');
  if (entries.length === 0) return false;
  const key = textureReadinessKey(candidate);
  if (key == null) return false;
  return entries.every((entry) =>
    (entry.handleKind === 'pixel-data' || entry.handleKind === 'data-texture') &&
    entry.handleColorSpace === 'linear' &&
    entry.backendReadiness[key] === 'ready'
  );
}

function compatibilityWithTextureReadiness(
  candidate: GltfBackendCompatibility,
  report: GltfTextureDecodeReport,
): GltfBackendCompatibility {
  const baseIssues = candidate.issues.filter((issue) => !isTextureReadinessIssue(issue));
  const textureIssues = textureReadinessIssuesForCandidate(
    issuesMatch(candidate.issues, baseIssues) ? candidate : compatibilityWithIssues(candidate, baseIssues),
    report,
  );
  if (textureIssues.length === 0 && issuesMatch(candidate.issues, baseIssues)) return candidate;
  return compatibilityWithIssues(candidate, [...baseIssues, ...textureIssues]);
}

function textureReadinessIssuesForCandidate(
  candidate: GltfBackendCompatibility,
  report: GltfTextureDecodeReport,
): GltfCompatibilityIssue[] {
  const key = textureReadinessKey(candidate);
  if (key == null) return [];

  const unsupportedMaterialFields = new Set(
    candidate.issues
      .filter((issue) => issue.category === 'material' && issue.support === 'unsupported')
      .map((issue) => issue.name),
  );

  const issues: GltfCompatibilityIssue[] = [];
  for (const entry of report.entries) {
    if (unsupportedMaterialFields.has(entry.materialField)) continue;
    const status = entry.backendReadiness[key];
    const support = textureReadinessSupport(status);
    if (support == null) continue;
    issues.push({
      category: 'texture',
      name: `${TEXTURE_READINESS_ISSUE_PREFIX}${entry.materialField}`,
      support,
      path: entry.path,
      message:
        support === 'requires-hook'
          ? `Backend profile ${candidate.profileId} needs a decoded or backend-native texture handle for ` +
            `"${entry.materialField}" at ${entry.path}; current handle is ${entry.handleKind}.`
          : `Backend profile ${candidate.profileId} does not consume "${entry.materialField}" texture data ` +
            `at ${entry.path}; current handle is ${entry.handleKind}.`,
    });
  }
  return issues;
}

function textureDecodeDiagnosticIssuesForCandidate(
  candidate: GltfBackendCompatibility,
  diagnostics: readonly DecodeSceneTextureDiagnostic[],
): GltfCompatibilityIssue[] {
  const unsupportedMaterialFields = new Set(
    candidate.issues
      .filter((issue) => issue.category === 'material' && issue.support === 'unsupported')
      .map((issue) => issue.name),
  );
  const issues: GltfCompatibilityIssue[] = [];
  const seen = new Set<string>();
  for (const diagnostic of diagnostics) {
    if (!DEGRADED_TEXTURE_DECODE_DIAGNOSTICS.has(diagnostic.code)) continue;
    if (unsupportedMaterialFields.has(diagnostic.materialField)) continue;
    const name = `${TEXTURE_DECODE_DIAGNOSTIC_ISSUE_PREFIX}${diagnostic.code}:${diagnostic.materialField}`;
    const key = `${name}\n${diagnostic.path}`;
    if (seen.has(key)) continue;
    seen.add(key);
    issues.push({
      category: 'texture',
      name,
      support: 'approximate',
      path: diagnostic.path,
      message:
        `Texture decode diagnostic ${diagnostic.code} for "${diagnostic.materialField}" at ` +
        `${diagnostic.path} affects backend profile ${candidate.profileId}: ${diagnostic.message}`,
    });
  }
  return issues;
}

function textureReadinessKey(
  candidate: GltfBackendCompatibility,
): keyof GltfTextureDecodeReport['entries'][number]['backendReadiness'] | undefined {
  if (candidate.backend === 'pt-webgl2') return 'ptWebgl2';
  if (candidate.backend === 'pt-webgpu') return 'ptWebgpu';
  if (candidate.backend === 'walkaround-hybrid') return 'walkaroundHybrid';
  return undefined;
}

function textureReadinessSupport(status: GltfBackendTextureStatus): GltfCompatibilityIssue['support'] | null {
  if (status === 'ready') return null;
  if (status === 'opaque') return 'requires-hook';
  return 'unsupported';
}

function issuesMatch(
  a: readonly GltfCompatibilityIssue[],
  b: readonly GltfCompatibilityIssue[],
): boolean {
  return a.length === b.length && a.every((issue, index) => issue === b[index]);
}

export function rerankBackendCompatibility(
  compatibility: readonly GltfBackendCompatibility[],
  policy: GltfBackendPolicy,
): readonly GltfBackendCompatibility[] {
  const preferred = policy === 'realtime'
    ? ['walkaround-hybrid', 'pt-webgpu', 'pt-webgpu-lite', 'pt-webgl2']
    : ['pt-webgl2', 'pt-webgpu', 'pt-webgpu-lite', 'walkaround-hybrid'];
  const order = new Map(preferred.map((profileId, index) => [profileId, index]));
  return [...compatibility].sort((a, b) => {
    if (policy === 'strict') {
      const aBad = a.unsupportedCount + a.approximateCount + a.requiresHookCount;
      const bBad = b.unsupportedCount + b.approximateCount + b.requiresHookCount;
      if (aBad !== bBad) return aBad - bBad;
    } else {
      if (a.unsupportedCount !== b.unsupportedCount) return a.unsupportedCount - b.unsupportedCount;
      if (a.requiresHookCount !== b.requiresHookCount) return a.requiresHookCount - b.requiresHookCount;
      if (a.approximateCount !== b.approximateCount) return a.approximateCount - b.approximateCount;
    }
    return (order.get(a.profileId) ?? 99) - (order.get(b.profileId) ?? 99);
  });
}

function compatibilityWithIssues(
  candidate: GltfBackendCompatibility,
  issues: readonly GltfCompatibilityIssue[],
): GltfBackendCompatibility {
  let unsupportedCount = 0;
  let approximateCount = 0;
  let requiresHookCount = 0;

  for (const issue of issues) {
    if (issue.support === 'unsupported') {
      unsupportedCount += 1;
    } else if (
      issue.support === 'approximate' ||
      issue.support === 'fallback-generated-mesh' ||
      issue.support === 'fallback-rebuild'
    ) {
      approximateCount += 1;
    } else if (issue.support === 'requires-hook') {
      requiresHookCount += 1;
    }
  }

  return {
    ...candidate,
    unsupportedCount,
    approximateCount,
    nativeCount: candidate.nativeCount,
    requiresHookCount,
    issues,
    isCompatible: unsupportedCount === 0,
  };
}
