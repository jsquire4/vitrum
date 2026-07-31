#!/usr/bin/env -S deno run --config=deno.json --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write
// @ts-nocheck
/**
 * Generate neural-denoiser pairs from the shipped walkaround-hybrid renderer.
 *
 * This is the production-aligned peer of `capture-dataset.mjs` (which remains a
 * CPU-only format smoke). Each sample is read from the exact live textures the
 * neural runtime consumes: shade-pass linear radiance, demodulated albedo, and
 * the world-normal G-buffer. A one-frame noisy input and a deterministic
 * multi-frame renderer average share the same scene, camera, lighting, warm
 * estimator state, and seed schedule.
 *
 * Usage from the repository root:
 *
 *   deno run --config tools/neural-denoiser-training/deno.json \
 *     --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write \
 *     tools/neural-denoiser-training/capture-renderer-dataset.mjs \
 *     --out data_renderer --pairs 500 --size 128 --clean-frames 4096 \
 *     --warmup-frames 8 --seed 1984
 */

import { createWalkaroundEngine_Hybrid } from '@vitrum/walkaround-hybrid';
import { asMat4 } from '@vitrum/core';
import { acquireWhDevice, makeLookAtMatrix, makePerspectiveMatrix } from '../lib/whHarness.mjs';
import {
  accumulateRendererRadiance,
  captureRendererTrainingInput,
  createRgbAccumulator,
  encodeAuxiliaryPng,
  encodeVhdr,
  finishRendererRadianceAverage,
  parseRendererDatasetArgs,
  rendererCaptureConfigRecord,
  rendererDatasetFrameSeed,
  rendererDatasetManifest,
} from './renderer-dataset-contract.mjs';
import { publishRendererDatasetGeneration } from './renderer-dataset-publication.mjs';

function printHelp() {
  console.log('capture-renderer-dataset.mjs — shipped-renderer neural dataset capture');
  console.log('  --out <dir>             new output root; must not exist (default data_renderer)');
  console.log('  --pairs <n>             aligned pairs (default 4; production minimum 500)');
  console.log('  --size <px>             square size, divisible by 8 (default 128)');
  console.log('  --clean-frames <n>      renderer frames averaged for clean target (default 4096)');
  console.log('  --warmup-frames <n>     estimator warmup before capture (default 8)');
  console.log('  --seed <u32-ish>        deterministic base seed (default 1984)');
  console.log('  --scene cornell_box     deterministic scene family (only current value)');
}

function makeRng(seed) {
  return { value: seed >>> 0 || 1 };
}

function random01(rng) {
  rng.value = (Math.imul(rng.value, 1_664_525) + 1_013_904_223) >>> 0;
  return rng.value / 0x1_0000_0000;
}

function randomRange(rng, low, high) {
  return low + (high - low) * random01(rng);
}

function makeQuad(id, vertices, normal, color) {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array(vertices.flat()),
    normals: new Float32Array([...normal, ...normal, ...normal, ...normal]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material: {
      baseColor: color,
      roughness: 1,
      metallic: 0,
      specularIntensity: 0,
    },
  };
}

export function buildCornellScene(baseSeed, pairIndex) {
  const rng = makeRng(rendererDatasetFrameSeed(baseSeed, pairIndex, 0));
  const neutral = randomRange(rng, 0.62, 0.84);
  const red = [
    randomRange(rng, 0.55, 0.82),
    randomRange(rng, 0.04, 0.16),
    randomRange(rng, 0.04, 0.16),
  ];
  const green = [
    randomRange(rng, 0.04, 0.16),
    randomRange(rng, 0.42, 0.72),
    randomRange(rng, 0.04, 0.18),
  ];
  const primitives = [
    makeQuad(
      'floor',
      [
        [-1, -1, -1],
        [1, -1, -1],
        [1, -1, 1],
        [-1, -1, 1],
      ],
      [0, 1, 0],
      [neutral, neutral, neutral],
    ),
    makeQuad(
      'ceiling',
      [
        [-1, 1, 1],
        [1, 1, 1],
        [1, 1, -1],
        [-1, 1, -1],
      ],
      [0, -1, 0],
      [neutral, neutral, neutral],
    ),
    makeQuad(
      'back-wall',
      [
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
      ],
      [0, 0, -1],
      [neutral, neutral, neutral],
    ),
    makeQuad(
      'left-wall',
      [
        [-1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, -1],
      ],
      [1, 0, 0],
      red,
    ),
    makeQuad(
      'right-wall',
      [
        [1, -1, -1],
        [1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
      ],
      [-1, 0, 0],
      green,
    ),
  ];

  const lightHalfSize = randomRange(rng, 0.16, 0.34);
  const lightX = randomRange(rng, -0.35, 0.35);
  const lightZ = randomRange(rng, -0.35, 0.35);
  const warm = randomRange(rng, 0.78, 1);
  return {
    primitives,
    emitters: [
      {
        kind: 'rect-area',
        id: 'ceiling-light',
        position: [lightX, 0.96, lightZ],
        // u × v points down into the box.
        uAxis: [lightHalfSize, 0, 0],
        vAxis: [0, 0, lightHalfSize],
        color: [1, warm, randomRange(rng, 0.68, warm)],
        intensity: randomRange(rng, 7, 18),
      },
    ],
    environment: { kind: 'none' },
  };
}

function buildCamera(baseSeed, pairIndex, size) {
  const rng = makeRng(rendererDatasetFrameSeed(baseSeed, pairIndex, 1));
  const eye = [
    randomRange(rng, -0.38, 0.38),
    randomRange(rng, -0.25, 0.32),
    randomRange(rng, 2.25, 2.85),
  ];
  const center = [
    randomRange(rng, -0.2, 0.2),
    randomRange(rng, -0.25, 0.2),
    randomRange(rng, -0.2, 0.2),
  ];
  return {
    eye,
    view: asMat4(makeLookAtMatrix(eye, center, [0, 1, 0])),
    projection: asMat4(makePerspectiveMatrix(randomRange(rng, 48, 68), 1, 0.1, 50)),
    viewport: { width: size, height: size, devicePixelRatio: 1 },
  };
}

async function waitForReady(engine, timeoutMs = 120_000) {
  const deadline = Date.now() + timeoutMs;
  while (engine.state !== 'ready') {
    if (engine.state === 'error' || engine.state === 'disposed') {
      throw new Error(`walkaround engine entered ${engine.state} during initialization`);
    }
    if (Date.now() >= deadline) {
      throw new Error(`walkaround engine did not become ready within ${timeoutMs} ms`);
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
}

async function writeFileAtomic(path, bytes) {
  const temporary = `${path}.tmp-${Deno.pid}`;
  try {
    await Deno.writeFile(temporary, bytes);
    await Deno.rename(temporary, path);
  } catch (error) {
    try {
      await Deno.remove(temporary);
    } catch {
      // Preserve the write/rename error.
    }
    throw error;
  }
}

async function writeJsonAtomic(path, value) {
  await writeFileAtomic(path, new TextEncoder().encode(`${JSON.stringify(value, null, 2)}\n`));
}

async function renderPair(device, config, pairIndex) {
  const size = config.size;
  let engine = null;
  let swapTexture = null;
  try {
    engine = await createWalkaroundEngine_Hybrid({
      device,
      width: size,
      height: size,
      primaryLightDir: [0.3, -0.8, 0.5],
      primaryLightIntensity: 0,
      skyTint: [0, 0, 0],
      skyIrradiance: 0,
      maxBounces: 2,
      denoiser: 'none',
      targetFrameIntervalMs: null,
      checkerboardRendering: false,
      ppgEnabled: false,
      nrcEnabled: false,
      rcEnabled: false,
      gtaoMode: 'on',
      verbose: false,
    });
    engine.setScene(buildCornellScene(config.seed, pairIndex));
    await waitForReady(engine);

    swapTexture = device.createTexture({
      label: `vitrum.neural-dataset.swap.${pairIndex}`,
      size: [size, size, 1],
      format: 'bgra8unorm',
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.TEXTURE_BINDING,
    });
    const swapChainView = swapTexture.createView();
    const camera = buildCamera(config.seed, pairIndex, size);
    const noisy = createRgbAccumulator(size, size);
    const clean = createRgbAccumulator(size, size);
    let auxiliaries = null;
    const totalFrames = config.warmupFrames + config.cleanFrames;

    for (let frame = 0; frame < totalFrames; frame += 1) {
      const frameSeed = rendererDatasetFrameSeed(config.seed, pairIndex, frame + 2);
      const output = engine.renderFrame({
        viewMatrix: camera.view,
        projMatrix: camera.projection,
        cameraPosition: camera.eye,
        viewport: camera.viewport,
        frameIndex: frame,
        frameSeed,
        swapChainView,
        swapChainFormat: 'bgra8unorm',
        quality: {
          bounces: 2,
          resolutionFactor: 1,
          tonemap: 'none',
          exposure: 1,
          outputColorSpace: 'linear',
        },
      });
      if (output.kind !== 'rendered') {
        throw new Error(
          `walkaround renderer skipped pair ${pairIndex + 1}, frame ${frame}; ` +
            'the capture runner disables its frame throttle, so a skip is not a valid sample',
        );
      }
      await device.queue.onSubmittedWorkDone();
      if (frame < config.warmupFrames) continue;

      const capture = await captureRendererTrainingInput(engine, size, size);
      if (auxiliaries == null) auxiliaries = capture;
      if (noisy.samples === 0) {
        accumulateRendererRadiance(noisy, capture.radiance);
      }
      accumulateRendererRadiance(clean, capture.radiance);

      const sample = frame - config.warmupFrames + 1;
      const progressInterval = Math.max(1, Math.floor(config.cleanFrames / 16));
      if (sample % progressInterval === 0 || sample === config.cleanFrames) {
        console.log(
          `[renderer-capture] pair ${pairIndex + 1}/${config.pairs}: ` +
            `${sample}/${config.cleanFrames} clean frames`,
        );
      }
    }

    if (auxiliaries == null) {
      throw new Error(`pair ${pairIndex + 1} produced no renderer capture`);
    }
    return {
      noisy: finishRendererRadianceAverage(noisy),
      clean: finishRendererRadianceAverage(clean),
      albedo: auxiliaries.albedo,
      worldNormal: auxiliaries.worldNormal,
    };
  } finally {
    try {
      swapTexture?.destroy();
    } catch {
      // Preserve the render/capture outcome.
    }
    try {
      engine?.dispose();
    } catch {
      // Preserve the render/capture outcome.
    }
  }
}

export async function runRendererDatasetCapture(config) {
  await publishRendererDatasetGeneration(config.out, async (stagingRoot) => {
    const sceneDirectory = `${stagingRoot}/${config.scene}`;
    const noisyDirectory = `${sceneDirectory}/noisy`;
    const cleanDirectory = `${sceneDirectory}/clean`;
    await Deno.mkdir(noisyDirectory, { recursive: true });
    await Deno.mkdir(cleanDirectory, { recursive: true });

    const device = await acquireWhDevice();
    try {
      console.log(
        `[renderer-capture] scene=${config.scene} pairs=${config.pairs} ` +
          `size=${config.size} noisy=1 renderer-frame ` +
          `clean=${config.cleanFrames} renderer-frames warmup=${config.warmupFrames} ` +
          `seed=${config.seed}`,
      );
      for (let pairIndex = 0; pairIndex < config.pairs; pairIndex += 1) {
        const captured = await renderPair(device, config, pairIndex);
        const tag = `frame_${String(pairIndex + 1).padStart(4, '0')}`;
        await Promise.all([
          writeFileAtomic(
            `${noisyDirectory}/${tag}.bin`,
            encodeVhdr(captured.noisy, config.size, config.size),
          ),
          writeFileAtomic(
            `${cleanDirectory}/${tag}.bin`,
            encodeVhdr(captured.clean, config.size, config.size),
          ),
          writeFileAtomic(
            `${noisyDirectory}/${tag}_albedo.png`,
            encodeAuxiliaryPng(captured.albedo, config.size, config.size),
          ),
          writeFileAtomic(
            `${noisyDirectory}/${tag}_normal.png`,
            encodeAuxiliaryPng(captured.worldNormal, config.size, config.size, true),
          ),
        ]);
        console.log(`[renderer-capture] wrote ${tag}`);
      }
      // Manifests are written only inside the unpublished generation. The final
      // output directory appears in one sibling rename after every declared pair
      // and both manifests are present.
      await writeJsonAtomic(
        `${stagingRoot}/capture-config.json`,
        rendererCaptureConfigRecord(config),
      );
      await writeJsonAtomic(
        `${stagingRoot}/dataset-manifest.json`,
        rendererDatasetManifest(config),
      );
    } finally {
      try {
        device.destroy();
      } catch {
        // Preserve the capture outcome.
      }
    }
  });
  console.log(`[renderer-capture] complete → ${config.out}`);
}

if (import.meta.main) {
  try {
    const config = parseRendererDatasetArgs(Deno.args);
    if (config.help) {
      printHelp();
    } else {
      await runRendererDatasetCapture(config);
    }
  } catch (error) {
    console.error(
      `[renderer-capture] ${error instanceof Error ? (error.stack ?? error.message) : String(error)}`,
    );
    Deno.exitCode = 1;
  }
}
