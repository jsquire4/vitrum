// @ts-check

export const GAP_CLOSURE_CAPTURE_PROVENANCE_SCHEMA = 'vitrum.gap-closure.capture-provenance.v1';

export class GapClosureProofValidationError extends Error {
  /** @param {string} message */
  constructor(message) {
    super(message);
    this.name = 'GapClosureProofValidationError';
  }
}

/**
 * @param {Array<{ scenarioId: string }>} scenarios
 * @param {string[]} requestedIds
 */
export function selectGapClosureScenarios(scenarios, requestedIds) {
  if (requestedIds.length === 0) return scenarios;
  const duplicates = requestedIds.filter((id, index) => requestedIds.indexOf(id) !== index);
  if (duplicates.length > 0) {
    throw new GapClosureProofValidationError(
      `VITRUM_GAP_SCENARIOS contains duplicate ids: ${[...new Set(duplicates)].join(', ')}`,
    );
  }
  const known = new Set(scenarios.map((scenario) => scenario.scenarioId));
  const unknown = requestedIds.filter((id) => !known.has(id));
  if (unknown.length > 0) {
    throw new GapClosureProofValidationError(
      `VITRUM_GAP_SCENARIOS matched no scenario for: ${unknown.join(', ')}`,
    );
  }
  const selected = scenarios.filter((scenario) => requestedIds.includes(scenario.scenarioId));
  if (selected.length === 0) {
    throw new GapClosureProofValidationError('VITRUM_GAP_SCENARIOS selected zero scenarios');
  }
  return selected;
}

/**
 * @param {{ mtimeMs: number, size: number, sha256: string } | null} before
 * @param {{ mtimeMs: number, size: number, sha256: string } | null} after
 * @param {number} startedAtMs
 */
export function evaluateCaptureFreshness(before, after, startedAtMs) {
  if (after == null) return { fresh: false, reason: 'capture artifact is missing' };
  if (
    !Number.isFinite(after.mtimeMs) ||
    !Number.isSafeInteger(after.size) ||
    !/^[0-9a-f]{64}$/.test(after.sha256)
  ) {
    return { fresh: false, reason: 'capture artifact metadata is invalid' };
  }
  const changed =
    before == null ||
    after.mtimeMs !== before.mtimeMs ||
    after.size !== before.size ||
    after.sha256 !== before.sha256;
  if (!changed) {
    return { fresh: false, reason: 'capture adapter left the pre-run artifact unchanged' };
  }
  if (!Number.isFinite(startedAtMs) || after.mtimeMs < startedAtMs - 2_000) {
    return { fresh: false, reason: 'capture artifact timestamp predates this run' };
  }
  return { fresh: true, reason: null };
}

/**
 * @param {Record<string, any>} scenario
 * @param {{ baselineExists: boolean, baselineHash: string | null, baselinePath: string }} input
 */
export function mechanicalGapClosureRow(scenario, input) {
  return {
    ...scenario,
    status: 'SKIPPED',
    evidenceClass: 'MECHANICAL',
    beforeImageHash: input.baselineHash,
    afterImageHash: null,
    deltaSummary: input.baselineExists
      ? 'Mechanical inspection only: baseline PNG exists; no GPU candidate was captured.'
      : `Mechanical inspection only: baseline PNG is missing at ${input.baselinePath}.`,
    perfBaselineMsPerSample: null,
    perfCandidateMsPerSample: null,
    passFail: 'SKIPPED',
  };
}

/** @param {any} artifact */
function isArtifactState(artifact) {
  return (
    artifact != null &&
    typeof artifact === 'object' &&
    typeof artifact.artifactPath === 'string' &&
    artifact.artifactPath.length > 0 &&
    Number.isFinite(artifact.mtimeMs) &&
    Number.isSafeInteger(artifact.size) &&
    artifact.size > 0 &&
    /^[0-9a-f]{64}$/.test(artifact.sha256)
  );
}

/**
 * @param {Array<Record<string, any>>} results
 */
export function strictGapClosureProblems(results) {
  const problems = [];
  for (const row of results) {
    const label = String(row.scenarioId ?? '(unknown)');
    if (row.passFail !== 'PASS') {
      problems.push(`${label}: passFail=${String(row.passFail)}`);
      continue;
    }
    if (row.evidenceClass !== 'LIVE_GPU') {
      problems.push(`${label}: evidenceClass=${String(row.evidenceClass)}`);
      continue;
    }
    const provenance = row.captureProvenance;
    if (
      provenance?.schema !== GAP_CLOSURE_CAPTURE_PROVENANCE_SCHEMA ||
      !isArtifactState(provenance.baseline) ||
      !Array.isArray(provenance.candidates) ||
      provenance.candidates.length === 0
    ) {
      problems.push(`${label}: capture provenance is incomplete`);
      continue;
    }
    const candidates = /** @type {Array<Record<string, any>>} */ (provenance.candidates);
    const badCandidate = candidates.find((candidate) => {
      const captureStartedAtMs =
        typeof candidate?.captureStartedAt === 'string'
          ? Date.parse(candidate.captureStartedAt)
          : Number.NaN;
      const capturedAtMs =
        typeof candidate?.capturedAt === 'string' ? Date.parse(candidate.capturedAt) : Number.NaN;
      const previousArtifact = candidate?.previousArtifact;
      const previousArtifactValid =
        previousArtifact === null ||
        (isArtifactState(previousArtifact) &&
          previousArtifact.artifactPath === candidate.artifactPath);
      const freshness = evaluateCaptureFreshness(
        previousArtifactValid ? previousArtifact : null,
        /** @type {{ mtimeMs: number, size: number, sha256: string }} */ (candidate),
        captureStartedAtMs,
      );
      return (
        candidate?.fresh !== true ||
        !isArtifactState(candidate) ||
        candidate.artifactPath === provenance.baseline.artifactPath ||
        !previousArtifactValid ||
        !Number.isFinite(captureStartedAtMs) ||
        !Number.isFinite(capturedAtMs) ||
        Math.abs(capturedAtMs - candidate.mtimeMs) > 1 ||
        !freshness.fresh
      );
    });
    const candidatePaths = candidates.map((candidate) => candidate?.artifactPath);
    if (badCandidate != null || new Set(candidatePaths).size !== candidatePaths.length) {
      problems.push(`${label}: candidate provenance is stale or not distinct`);
    }
  }
  return problems;
}
