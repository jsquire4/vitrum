import assert from 'node:assert/strict';
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
  assert.match(text, /promotionCommand: npm run proof:required/);
  assert.match(text, /remaining: needs browser host/);
  assert.match(text, /researchPromotionWork: 1/);
  assert.match(text, /VQ-RESEARCH: research promotion \[partial-proof-green\]/);
  assert.match(text, /workClass: research-promotion/);
  assert.match(text, /provisioningWork: 1/);
  assert.match(text, /VQ-LEARNED: learned systems \[provisioning-needed\]/);
  assert.match(text, /futureContract: 1/);
  assert.match(text, /currentContract: fallback is explicit/);
  assert.match(text, /decisionBlockers:/);
  assert.match(text, /\* define contract/);
  assert.match(text, /\* add backend grades/);
});
