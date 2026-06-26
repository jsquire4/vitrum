import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  formatDetailedSummary,
  formatSummary,
  parseQueue,
  summarizeQueue,
} from '../../tools/road-to-100/next-actions.mjs';

test('road-to-100 next-actions separates code, proof, provisioning, and future rows', () => {
  const queue = parseQueue({
    currentAsOf: '2026-06-22',
    implementationQueue: [
      { id: 'IMPL-1', status: 'source-needed', title: 'bounded code bug' },
    ],
    validationQueue: [
      { id: 'VQ-PARTIAL', status: 'partial-proof-green', title: 'partial proof', remaining: 'needs A/B' },
      {
        id: 'VQ-RESEARCH',
        status: 'partial-proof-green',
        workClass: 'research-promotion',
        title: 'research promotion',
        remaining: 'needs estimator redesign',
      },
      { id: 'VQ-HOST', status: 'host-blocked', title: 'host proof', remaining: 'needs browser' },
      { id: 'VQ-LEARNED', status: 'provisioning-needed', title: 'learned', remaining: 'needs checkpoint' },
      { id: 'VQ-GREEN', status: 'committed-proof-green', title: 'done', remaining: 'done' },
    ],
    futureContractRows: [
      { id: 'FC-ONE', status: 'future-contract', title: 'future API' },
    ],
  });

  const summary = summarizeQueue(queue);
  assert.equal(summary.activeCodeBlocked, true);
  assert.deepEqual(summary.implementation.map((row) => row.id), ['IMPL-1']);
  assert.deepEqual(summary.proofRows.map((row) => row.id), ['VQ-PARTIAL', 'VQ-HOST']);
  assert.deepEqual(summary.researchPromotionRows.map((row) => row.id), ['VQ-RESEARCH']);
  assert.deepEqual(summary.provisioningRows.map((row) => row.id), ['VQ-LEARNED']);
  assert.deepEqual(summary.futureRows.map((row) => row.id), ['FC-ONE']);

  const text = formatSummary(summary);
  assert.match(text, /implementationQueue: 1/);
  assert.match(text, /code-now: IMPL-1/);
  assert.match(text, /proof-or-adapter-work: 2 \(VQ-PARTIAL, VQ-HOST\)/);
  assert.match(text, /research-promotion-work: 1 \(VQ-RESEARCH\)/);
  assert.match(text, /provisioning-work: 1 \(VQ-LEARNED\)/);
  assert.match(text, /future-contract: 1 \(FC-ONE\)/);
});

test('road-to-100 next-actions reports code freeze when implementation queue is empty', () => {
  const summary = summarizeQueue(parseQueue({
    currentAsOf: '2026-06-22',
    implementationQueue: [],
    validationQueue: [
      {
        id: 'VQ-RESEARCH',
        status: 'partial-proof-green',
        workClass: 'research-promotion',
        title: 'research promotion',
        remaining: 'needs estimator redesign',
      },
      { id: 'VQ-PARTIAL', status: 'partial-proof-green', title: 'partial proof', remaining: 'needs A/B' },
    ],
    futureContractRows: [],
  }));

  assert.equal(summary.activeCodeBlocked, false);
  assert.match(
    formatSummary(summary),
    /code-now: none \(do not reopen source work unless implementationQueue gains a source-verified row\)/,
  );
});

test('road-to-100 next-actions details include commands, remaining work, and future blockers', () => {
  const summary = summarizeQueue(parseQueue({
    currentAsOf: '2026-06-22',
    implementationQueue: [],
    validationQueue: [
      {
        id: 'VQ-HOST',
        status: 'host-blocked',
        kind: 'validation',
        title: 'browser proof',
        command: 'npm run proof',
        promotionCommand: 'npm run proof:required',
        allCasesHighSppCommand: 'npm run proof:all-spp64',
        executionScope: 'external-browser-host',
        blockedBy: 'browser readback is unavailable on this host',
        nextLocalAction: 'none until the browser host changes',
        rerunPolicy: 'do not rerun required proof on WSL',
        remaining: 'needs browser host',
      },
      {
        id: 'VQ-RESEARCH',
        status: 'partial-proof-green',
        kind: 'validation',
        workClass: 'research-promotion',
        title: 'research promotion',
        command: 'npm run research-proof',
        remaining: 'needs estimator redesign',
      },
      {
        id: 'VQ-LEARNED',
        status: 'provisioning-needed',
        kind: 'provisioning',
        title: 'learned systems',
        command: 'npm run learned',
        proofArtifacts: [
          {
            path: 'tools/learned-systems/learned-systems-status.json',
            json: { verdict: 'PASS' },
          },
        ],
        remaining: 'needs production checkpoint',
      },
    ],
    futureContractRows: [
      {
        id: 'FC-ONE',
        status: 'future-contract',
        title: 'future API',
        currentContract: 'fallback is explicit',
        decisionBlockers: ['define contract', 'add backend grades'],
      },
    ],
  }));

  const text = formatDetailedSummary(summary);
  assert.match(text, /\[road-to-100-next-actions:details\]/);
  assert.match(text, /proofOrAdapterWork: 1/);
  assert.match(text, /VQ-HOST: browser proof \[host-blocked\]/);
  assert.match(text, /command: npm run proof/);
  assert.match(text, /allCasesHighSppCommand: npm run proof:all-spp64/);
  assert.match(text, /promotionCommand: npm run proof:required/);
  assert.match(text, /executionScope: external-browser-host/);
  assert.match(text, /blockedBy: browser readback is unavailable on this host/);
  assert.match(text, /nextLocalAction: none until the browser host changes/);
  assert.match(text, /rerunPolicy: do not rerun required proof on WSL/);
  assert.match(text, /remaining: needs browser host/);
  assert.match(text, /researchPromotionWork: 1/);
  assert.match(text, /VQ-RESEARCH: research promotion \[partial-proof-green\]/);
  assert.match(text, /workClass: research-promotion/);
  assert.match(text, /provisioningWork: 1/);
  assert.match(text, /VQ-LEARNED: learned systems \[provisioning-needed\]/);
  assert.match(text, /proofArtifactStatus:/);
  assert.match(text, /tools\/learned-systems\/learned-systems-status\.json: verdict=PASS/);
  assert.match(text, /futureContract: 1/);
  assert.match(text, /currentContract: fallback is explicit/);
  assert.match(text, /decisionBlockers:/);
  assert.match(text, /\* define contract/);
  assert.match(text, /\* add backend grades/);
});

test('committed road unresolved rows carry enforced execution metadata', async () => {
  const [queueRaw, checkerSource] = await Promise.all([
    readFile(new URL('../../tools/road-to-100/validation-queue.json', import.meta.url), 'utf8'),
    readFile(new URL('../../tools/road-to-100/check-validation-queue.mjs', import.meta.url), 'utf8'),
  ]);
  const queue = JSON.parse(queueRaw);
  const unresolvedStatuses = new Set([
    'partial-proof-green',
    'host-blocked',
    'evidence-needed',
    'provisioning-needed',
    'decision-needed',
  ]);
  const allowedScopes = new Set([
    'external-browser-host',
    'external-real-adapter-validation',
    'external-real-adapter-throughput',
    'research-design-and-real-adapter-validation',
    'asset-provisioning-and-quality-ab',
  ]);
  const unresolvedRows = queue.validationQueue.filter((row) => unresolvedStatuses.has(row.status));
  assert.deepEqual(unresolvedRows.map((row) => row.id), [
    'VQ-GLTF-BROWSER-PTWEBGL2',
    'VQ-RADIOMETRIC-PT',
    'VQ-WALKAROUND-RADIOMETRIC-AB',
    'VQ-RENDERER-FIDELITY-PROOF',
    'VQ-CWBVH-DEFAULT-PROMOTION',
    'VQ-LEARNED-SYSTEMS',
  ]);
  for (const row of unresolvedRows) {
    assert.ok(allowedScopes.has(row.executionScope), row.id + ' executionScope should be classified');
    assert.equal(typeof row.blockedBy, 'string', row.id + ' blockedBy should be present');
    assert.ok(row.blockedBy.length > 20, row.id + ' blockedBy should explain the blocker');
    assert.equal(typeof row.nextLocalAction, 'string', row.id + ' nextLocalAction should be present');
    assert.ok(row.nextLocalAction.length > 10, row.id + ' nextLocalAction should guide local work');
    assert.equal(typeof row.rerunPolicy, 'string', row.id + ' rerunPolicy should be present');
    assert.match(row.rerunPolicy, /rerun/i, row.id + ' rerunPolicy should say when to rerun');
  }
  assert.match(checkerSource, /UNRESOLVED_VALIDATION_STATUSES/);
  assert.match(checkerSource, /ALLOWED_EXECUTION_SCOPES/);
  assert.match(checkerSource, /function assertUnresolvedExecutionMetadata/);
  assert.match(checkerSource, /assertUnresolvedExecutionMetadata\(row\)/);
  assert.match(checkerSource, /contains unexpected id/);

  const pngArtifacts = queue.validationQueue.flatMap((row) => row.proofArtifacts ?? [])
    .filter((artifact) => artifact.type === 'png' || artifact.path?.endsWith?.('.png'));
  assert.ok(pngArtifacts.length >= 40, 'Road validation queue should pin the committed PNG proof set');
  for (const artifact of pngArtifacts) {
    assert.match(artifact.sha256, /^[0-9a-f]{64}$/, artifact.path + ' should pin sha256');
    assert.ok(Number.isInteger(artifact.width) && artifact.width > 0, artifact.path + ' should pin width');
    assert.ok(Number.isInteger(artifact.height) && artifact.height > 0, artifact.path + ' should pin height');
  }
  assert.match(checkerSource, /function assertPngIdentity/);
  assert.match(checkerSource, /sha256Hex/);
  assert.match(checkerSource, /PNG SHA-256/);
  assert.match(checkerSource, /REQUIRED_GLTF_REAL_MANIFEST_ROWS/);
  assert.match(checkerSource, /function assertRealGltfManifestCoverage/);
  assert.match(checkerSource, /baseGoldenPath/);
  assert.match(checkerSource, /dznFullGoldenPath/);
  assert.match(checkerSource, /requiredExtensions must pin/);
});
