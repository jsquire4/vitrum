import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const queuePath = path.join(rootDir, 'tools/road-to-100/validation-queue.json');
const queue = JSON.parse(readFileSync(queuePath, 'utf8'));

const expectedFutureContractIds = [
  'FC-ADJOINT-FULL-PATH-PARITY',
  'FC-ARBITRARY-UV-ARRAYS',
  'FC-DISPLACEMENT-MICROTESSELLATION',
  'FC-NATIVE-INSTANCED-SKINNED-MORPHED',
  'FC-NATIVE-POINT-LINE',
  'FC-TRANSPARENT-GI-TRANSPORT',
  'FC-WALKAROUND-SPECIALTY-MATERIAL-TRANSPORT',
];

function futureRow(id) {
  const row = queue.futureContractRows.find((candidate) => candidate.id === id);
  assert.ok(row, `expected future-contract row ${id}`);
  return row;
}

function sourcePaths(row) {
  return row.sourceEvidence.map((evidence) => evidence.path);
}

test('committed road-to-100 queue has no active implementation rows', () => {
  assert.deepEqual(
    queue.implementationQueue,
    [],
    'source-verified code gaps must be added explicitly to implementationQueue before code work resumes',
  );
});

test('future-contract rows stay bounded as decision/API work', () => {
  const rows = queue.futureContractRows;
  assert.deepEqual(
    rows.map((row) => row.id).sort(),
    expectedFutureContractIds,
    'future-contract row set changed; update the boundary test with the source-backed decision',
  );

  for (const row of rows) {
    assert.equal(row.status, 'future-contract', `${row.id} must not masquerade as an active code gap`);
    assert.equal(row.codeNowBounded, false, `${row.id} must stay out of code-now execution`);
    assert.ok(row.currentContract.length > 24, `${row.id} needs an explicit current contract`);
    assert.ok(Array.isArray(row.sourceEvidence), `${row.id} needs source evidence`);
    assert.ok(row.sourceEvidence.length > 0, `${row.id} needs at least one source evidence entry`);
    assert.ok(Array.isArray(row.decisionBlockers), `${row.id} needs decision blockers`);
    assert.ok(row.decisionBlockers.length >= 2, `${row.id} needs at least two decision blockers`);
  }
});

test('future-contract boundary text pins the current truthful API surface', () => {
  const displacement = futureRow('FC-DISPLACEMENT-MICROTESSELLATION');
  assert.match(displacement.currentContract, /consumed approximately/);
  assert.match(displacement.currentContract, /no tessellation or microgeometry synthesis/);
  assert.deepEqual(sourcePaths(displacement), [
    'packages/core/src/engine/promiseLedger.ts',
    'packages/core/src/__tests__/ledgerVsCapabilities.test.ts',
  ]);

  const transparentTransport = futureRow('FC-TRANSPARENT-GI-TRANSPORT');
  assert.match(transparentTransport.currentContract, /OIT and alpha-aware direct\/shadow handling/);
  assert.match(transparentTransport.currentContract, /not full ReSTIR\/GI transport vertices/);
  assert.deepEqual(sourcePaths(transparentTransport), [
    'packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts',
    'packages/walkaround-hybrid/src/__tests__/transparentAlphaTransportContract.test.ts',
  ]);

  const specialtyTransport = futureRow('FC-WALKAROUND-SPECIALTY-MATERIAL-TRANSPORT');
  assert.match(specialtyTransport.currentContract, /spectralAttenuation/);
  assert.match(specialtyTransport.currentContract, /thinFilmStack\/full-layer-stack transport remain unsupported/);
  assert.match(specialtyTransport.currentContract, /PT backends unless hosts accept those walkaround approximations/);
  assert.deepEqual(sourcePaths(specialtyTransport), [
    'packages/core/src/engine/promiseLedger.ts',
    'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
  ]);

  const pointLine = futureRow('FC-NATIVE-POINT-LINE');
  assert.match(pointLine.currentContract, /generated mesh fallbacks/);
  assert.deepEqual(sourcePaths(pointLine), [
    'packages/gltf-adapter/src/primitiveModeFallback.ts',
    'packages/gltf-adapter/README.md',
  ]);

  const uvArrays = futureRow('FC-ARBITRARY-UV-ARRAYS');
  assert.match(uvArrays.currentContract, /Single high TEXCOORD_N lossless remap/);
  assert.match(uvArrays.currentContract, /true arbitrary UV arrays require a new core\/backend contract/);
  assert.deepEqual(sourcePaths(uvArrays), [
    'packages/gltf-adapter/src/gltfToScene.ts',
    'packages/core/src/scene/material.ts',
  ]);

  const instancedSkinning = futureRow('FC-NATIVE-INSTANCED-SKINNED-MORPHED');
  assert.match(instancedSkinning.currentContract, /Renderable fallback expansion/);
  assert.match(instancedSkinning.currentContract, /native instanced skinned\/morphed primitives/);
  assert.deepEqual(sourcePaths(instancedSkinning), [
    'packages/core/src/scene/primitives.ts',
    'packages/gltf-adapter/src/featureReport.ts',
    'packages/gltf-adapter/src/gltfToScene.ts',
  ]);

  const adjoint = futureRow('FC-ADJOINT-FULL-PATH-PARITY');
  assert.match(adjoint.currentContract, /scoped direct-light path replay/);
  assert.match(adjoint.currentContract, /finite-difference fallback diagnostics/);
  assert.match(adjoint.currentContract, /indirect paths, and full-path parity cases/);
  assert.deepEqual(sourcePaths(adjoint), [
    'packages/pt-webgpu/src/inverse/inverseSession.ts',
    'packages/pt-webgpu/src/__tests__/inverseSession.test.ts',
  ]);
});
