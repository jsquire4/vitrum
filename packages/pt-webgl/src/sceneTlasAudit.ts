/**
 * @deprecated Import `auditSceneNeedsTlas` from `@vitrum/core` — kept for pt-webgl callers.
 */

import { auditSceneNeedsTlas, type SceneTlasAudit } from '@vitrum/core';

export type PtWebglTlasAudit = SceneTlasAudit;

export const auditPtWebglSceneForTlas = auditSceneNeedsTlas;
