import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { PNG } from 'pngjs';
import {
  RENDERER_DATASET_CAPTURE_SOURCE,
  accumulateRendererRadiance,
  captureRendererTrainingInput,
  createRgbAccumulator,
  encodeAuxiliaryPng,
  encodeVhdr,
  finishRendererRadianceAverage,
  parseRendererDatasetArgs,
  rendererDatasetFrameSeed,
  rendererDatasetManifest,
} from '../../tools/neural-denoiser-training/renderer-dataset-contract.mjs';
import { publishRendererDatasetGeneration } from '../../tools/neural-denoiser-training/renderer-dataset-publication.mjs';

test('renderer dataset CLI is deterministic and rejects malformed numeric input', () => {
  const first = parseRendererDatasetArgs([
    '--out',
    'capture',
    '--pairs',
    '7',
    '--size',
    '64',
    '--clean-frames',
    '4096',
    '--warmup-frames',
    '12',
    '--seed',
    '42',
  ]);
  const second = parseRendererDatasetArgs([
    '--out',
    'capture',
    '--pairs',
    '7',
    '--size',
    '64',
    '--clean-frames',
    '4096',
    '--warmup-frames',
    '12',
    '--seed',
    '42',
  ]);
  assert.deepEqual(first, second);
  assert.equal(rendererDatasetFrameSeed(42, 3, 9), rendererDatasetFrameSeed(42, 3, 9));
  assert.notEqual(rendererDatasetFrameSeed(42, 3, 9), rendererDatasetFrameSeed(42, 3, 10));
  assert.throws(() => parseRendererDatasetArgs(['--pairs', '1e3']), /base-10/);
  assert.throws(() => parseRendererDatasetArgs(['--size', '63']), /divisible by 8/);
  assert.throws(() => parseRendererDatasetArgs(['--clean-frames', '0']), /positive/);
  assert.throws(() => parseRendererDatasetArgs(['--seed', '4294967296']), /0\.\.4294967295/);
  assert.throws(
    () => parseRendererDatasetArgs(['--warmup-frames', '4294967296', '--clean-frames', '1']),
    /must not exceed/,
  );
  assert.throws(() => parseRendererDatasetArgs(['--unknown']), /Unknown argument/);
});

test('capture contract invokes the concrete engine method and validates exact arrays', async () => {
  let calls = 0;
  const capture = {
    width: 1,
    height: 1,
    radiance: new Float32Array([1, 2, 3]),
    albedo: new Float32Array([0.25, 0.5, 0.75]),
    worldNormal: new Float32Array([-1, 0, 1]),
  };
  const engine = {
    async captureDenoiserTrainingInputs() {
      calls += 1;
      return capture;
    },
  };
  assert.equal(await captureRendererTrainingInput(engine, 1, 1), capture);
  assert.equal(calls, 1);
  await assert.rejects(
    captureRendererTrainingInput({ captureDenoiserTrainingInputs: async () => null }, 1, 1),
    /returned null/,
  );
  await assert.rejects(captureRendererTrainingInput({}, 1, 1), /HybridEngine/);
});

test('production generator imports the shipped engine and does not delegate to the CPU smoke', async () => {
  const source = await readFile(
    new URL('../../tools/neural-denoiser-training/capture-renderer-dataset.mjs', import.meta.url),
    'utf8',
  );
  const denoConfig = JSON.parse(
    await readFile(
      new URL('../../tools/neural-denoiser-training/deno.json', import.meta.url),
      'utf8',
    ),
  );
  assert.match(source, /createWalkaroundEngine_Hybrid/);
  assert.match(source, /captureRendererTrainingInput\(engine/);
  assert.doesNotMatch(source, /from\s+["'][^"']*capture-dataset\.mjs/);
  assert.equal(
    denoConfig.imports['@vitrum/stained-glass-extensions'],
    '../../packages/stained-glass-extensions/src/index.ts',
  );
});

test('float64 accumulation and VHDR encoding preserve raw linear radiance', () => {
  const accumulator = createRgbAccumulator(1, 1);
  accumulateRendererRadiance(accumulator, new Float32Array([1, 2, 3]));
  accumulateRendererRadiance(accumulator, new Float32Array([3, 4, 5]));
  const average = finishRendererRadianceAverage(accumulator);
  assert.deepEqual(average, new Float32Array([2, 3, 4]));

  const vhdr = encodeVhdr(average, 1, 1);
  assert.equal(vhdr.readUInt32LE(0), 0x52444856);
  assert.equal(vhdr.readUInt32LE(4), 1);
  assert.equal(vhdr.readUInt32LE(8), 1);
  assert.equal(vhdr.readUInt32LE(12), 1);
  assert.equal(vhdr.readFloatLE(16), 2);
  assert.equal(vhdr.readFloatLE(20), 3);
  assert.equal(vhdr.readFloatLE(24), 4);
});

test('auxiliary PNG encoding is standards-readable and preserves signed-normal mapping', () => {
  const encoded = encodeAuxiliaryPng(new Float32Array([-1, 0, 1]), 1, 1, true);
  const decoded = PNG.sync.read(encoded);
  assert.equal(decoded.width, 1);
  assert.equal(decoded.height, 1);
  assert.deepEqual([...decoded.data], [0, 128, 255, 255]);
});

test('manifest records real renderer provenance and all required dataset lanes', () => {
  const config = parseRendererDatasetArgs([
    '--out',
    'renderer-data',
    '--pairs',
    '500',
    '--size',
    '128',
    '--clean-frames',
    '4096',
    '--warmup-frames',
    '8',
    '--seed',
    '1984',
  ]);
  const manifest = rendererDatasetManifest(config);
  assert.equal(manifest.schema, 'vitrum.neural-denoiser.dataset.v1');
  assert.equal(manifest.captureSource, RENDERER_DATASET_CAPTURE_SOURCE);
  assert.equal(manifest.noisySpp, 1);
  assert.equal(manifest.cleanReferenceSpp, 4096);
  assert.equal(manifest.includesAlbedo, true);
  assert.equal(manifest.includesNormals, true);
  assert.equal(manifest.tonemap, 'linear-hdr');
  assert.equal(manifest.scenes[0].sampleCount, 500);
  assert.match(manifest.scenes[0].noisyPath, /renderer-data\/cornell_box\/noisy/);
  assert.match(manifest.scenes[0].cleanPath, /renderer-data\/cornell_box\/clean/);
});

test('dataset generation publishes once and refuses a smaller in-place rerun', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'vitrum-neural-publication-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const output = join(root, 'dataset');

  await publishRendererDatasetGeneration(output, async (staging) => {
    const noisy = join(staging, 'cornell_box', 'noisy');
    await mkdir(noisy, { recursive: true });
    for (let index = 1; index <= 3; index += 1) {
      await writeFile(
        join(noisy, `frame_${String(index).padStart(4, '0')}.bin`),
        `generation-one-${index}`,
      );
    }
    await writeFile(join(staging, 'dataset-manifest.json'), JSON.stringify({ sampleCount: 3 }));
  });

  let rerunInvoked = false;
  await assert.rejects(
    publishRendererDatasetGeneration(output, async () => {
      rerunInvoked = true;
    }),
    /already exists/,
  );
  assert.equal(rerunInvoked, false);
  assert.deepEqual((await readdir(join(output, 'cornell_box', 'noisy'))).sort(), [
    'frame_0001.bin',
    'frame_0002.bin',
    'frame_0003.bin',
  ]);
  assert.deepEqual(JSON.parse(await readFile(join(output, 'dataset-manifest.json'), 'utf8')), {
    sampleCount: 3,
  });
});

test('failed generation never exposes a stale manifest or training artifacts', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'vitrum-neural-publication-failure-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const output = join(root, 'dataset');

  await assert.rejects(
    publishRendererDatasetGeneration(output, async (staging) => {
      const noisy = join(staging, 'cornell_box', 'noisy');
      await mkdir(noisy, { recursive: true });
      await writeFile(join(noisy, 'frame_0001.bin'), 'partial');
      await writeFile(join(staging, 'dataset-manifest.json'), JSON.stringify({ sampleCount: 1 }));
      throw new Error('injected renderer failure');
    }),
    /injected renderer failure/,
  );

  await assert.rejects(readFile(join(output, 'dataset-manifest.json')), /ENOENT/);
  await assert.rejects(readdir(output), /ENOENT/);
  assert.equal(
    (await readdir(root)).some((entry) => entry.startsWith('.dataset.vitrum-generation-')),
    false,
  );
});
