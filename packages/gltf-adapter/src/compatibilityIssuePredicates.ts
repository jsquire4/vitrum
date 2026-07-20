// compatibilityIssuePredicates.ts — the single source of truth for the
// texture-compatibility issue vocabulary (I4-2 / D15-8). The `texture-readiness:`
// and `texture-decode:` issue-name prefixes and their predicate functions were
// previously duplicated across assetLoader.ts, engineBridge.ts, and
// @vitrum/engine's gltf.ts (inlined). Centralizing them here means a future prefix
// change is made in exactly one place; every consumer imports these predicates.

import type { GltfCompatibilityIssue } from './featureReport.js';

/** Issue-name prefix for per-texture backend-readiness rows appended after decode. */
export const TEXTURE_READINESS_ISSUE_PREFIX = 'texture-readiness:';

/** Issue-name prefix for per-texture decode-diagnostic rows appended after decode. */
export const TEXTURE_DECODE_DIAGNOSTIC_ISSUE_PREFIX = 'texture-decode:';

/** True when `issue` is a per-texture backend-readiness compatibility row. */
export function isTextureReadinessIssue(issue: GltfCompatibilityIssue): boolean {
  return issue.category === 'texture' && issue.name.startsWith(TEXTURE_READINESS_ISSUE_PREFIX);
}

/** True when `issue` is a per-texture decode-diagnostic compatibility row. */
export function isTextureDecodeDiagnosticIssue(issue: GltfCompatibilityIssue): boolean {
  return issue.category === 'texture' && issue.name.startsWith(TEXTURE_DECODE_DIAGNOSTIC_ISSUE_PREFIX);
}
