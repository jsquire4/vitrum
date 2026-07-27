// End-to-end inverse-rendering gradient gate on a native WebGPU adapter.
// Run: deno run --no-check --unstable-webgpu -A tools/gpu-env/inverse-fit-deno.ts

import { asMat4, type Scene } from '../../packages/core/src/index.ts';
import {
  createPTEngine_WebGPU,
  ptWebgpuRequiredLimitsForAdapter,
} from '../../packages/pt-webgpu/src/index.ts';

type ProvenField = 'emissive';
type SamplingMode = 'pcg' | 'sobol';
type GeometryProfile = 'identity' | 'transformed' | 'instanced';

const PROOF_SAMPLES = Math.max(
  1,
  Math.floor(Number(Deno.env.get('VITRUM_INVERSE_PROOF_SAMPLES') ?? '32')),
);

interface GradientEvidence {
  readonly field: ProvenField;
  readonly sampling: SamplingMode;
  readonly geometry: GeometryProfile;
  readonly finiteDifference: readonly number[];
  readonly pathReplay: readonly number[];
  readonly diagnostics: readonly unknown[];
  readonly parameterMethods: readonly string[];
  readonly relativeError: readonly number[];
  readonly passed: boolean;
}

const gpu = (navigator as unknown as { gpu: GPU }).gpu;
const adapter = await gpu.requestAdapter();
if (adapter == null) throw new Error('No WebGPU adapter is available.');

const info = adapter.info;
const requiredLimits = ptWebgpuRequiredLimitsForAdapter(adapter);
const device = await adapter.requestDevice({ requiredLimits });

function identityMat(): Float32Array {
  const matrix = new Float32Array(16);
  matrix[0] = 1;
  matrix[5] = 1;
  matrix[10] = 1;
  matrix[15] = 1;
  return matrix;
}

const frame = {
  viewMatrix: asMat4(identityMat()),
  projMatrix: asMat4(identityMat()),
  cameraPosition: [0, 0, 0] as [number, number, number],
  viewport: { width: 1, height: 1, devicePixelRatio: 1 },
  frameIndex: 0,
  frameSeed: 0x1234_5678,
  quality: {
    samplesTarget: 1,
    bounces: 1,
    resolutionFactor: 1,
    tonemap: 'none' as const,
    outputColorSpace: 'linear' as const,
  },
};

function translated(x: number, y: number, z: number): ReturnType<typeof asMat4> {
  const matrix = identityMat();
  matrix[12] = x;
  matrix[13] = y;
  matrix[14] = z;
  return asMat4(matrix);
}

function makeScene(
  field: ProvenField,
  geometry: GeometryProfile,
): Scene {
  const positions = geometry === 'identity'
    ? new Float32Array([-4, -4, 2, 0, 4, 2, 4, -4, 2])
    : new Float32Array([-4, -4, 0, 0, 4, 0, 4, -4, 0]);
  const common = {
    positions,
    normals: new Float32Array([0, 0, -1, 0, 0, -1, 0, 0, -1]),
    material: {
      baseColor: [0.31, 0.46, 0.67] as [number, number, number],
      roughness: 0.43,
      metallic: 0.27,
      emissive: [0.2, 0.3, 0.4] as [number, number, number],
      emissiveIntensity: 1,
    },
  };
  const primitive = geometry === 'instanced'
    ? {
        kind: 'instanced-mesh' as const,
        id: 'panel',
        ...common,
        instances: [translated(0, 0, 2), translated(12, 0, 2)],
      }
    : {
        kind: 'mesh' as const,
        id: 'panel',
        ...common,
        ...(geometry === 'transformed' ? { transform: translated(0, 0, 2) } : {}),
      };
  return {
    primitives: [primitive],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function relativeError(reference: number, actual: number): number {
  return Math.abs(actual - reference) / Math.max(Math.abs(reference), 1e-4);
}

async function gradient(
  engine: Awaited<ReturnType<typeof createPTEngine_WebGPU>>,
  field: ProvenField,
  geometry: GeometryProfile,
  method: 'finite-difference' | 'path-replay',
): Promise<{
  gradient: number[];
  diagnostics: readonly unknown[];
  parameterMethods: readonly string[];
}> {
  engine.setScene(makeScene(field, geometry));
  engine.renderFrame(frame);
  const session = engine.createInverseSession!({
    target: {
      data: new Float32Array([0.7, 0.5, 0.3]),
      width: 1,
      height: 1,
      channels: 3,
    },
    parameters: [{
      path: `materials.panel.${field}`,
      kind: 'rgb',
    }],
    method,
    loss: 'l2',
    samplesPerStep: PROOF_SAMPLES,
    optimizer: {
      learningRate: 1e-8,
      fdEpsilon: 1e-3,
    },
  });
  try {
    const result = await session.step();
    if (
      method === 'path-replay' &&
      (
        session.method !== 'path-replay' ||
        session.parameterMethods[0] !== 'path-replay' ||
        (session.diagnostics?.length ?? 0) !== 0
      )
    ) {
      throw new Error(
        `${field}/${geometry}: requested path replay degraded to ${session.method}: ` +
          JSON.stringify({
            parameterMethods: session.parameterMethods,
            diagnostics: session.diagnostics,
          }),
      );
    }
    return {
      gradient: [...result.gradient[0]!],
      diagnostics: session.diagnostics ?? [],
      parameterMethods: [...session.parameterMethods],
    };
  } finally {
    session.dispose();
  }
}

const evidence: GradientEvidence[] = [];
for (const sampling of ['pcg', 'sobol'] as const) {
  const engine = await createPTEngine_WebGPU({
    device,
    traceTier: 'full',
    sampling,
    maxBounces: 1,
    maxSamplesPerPixel: 32,
  });
  const cases: Array<readonly [ProvenField, GeometryProfile]> = [
    ['emissive', 'identity'],
    ['emissive', 'transformed'],
    ['emissive', 'instanced'],
  ];
  for (const [field, geometry] of cases) {
    console.error(`inverse-proof: ${sampling}/${field}/${geometry}/${PROOF_SAMPLES}spp`);
    const fd = await gradient(engine, field, geometry, 'finite-difference');
    const replay = await gradient(engine, field, geometry, 'path-replay');
    const errors = fd.gradient.map((reference, index) =>
      relativeError(reference, replay.gradient[index]!),
    );
    const hasSignal = fd.gradient.every((component) => Math.abs(component) > 1e-5);
    const finite = [...fd.gradient, ...replay.gradient, ...errors].every(Number.isFinite);
    const sameSigns = fd.gradient.every(
      (component, index) => Math.sign(component) === Math.sign(replay.gradient[index]!),
    );
    evidence.push({
      field,
      sampling,
      geometry,
      finiteDifference: fd.gradient,
      pathReplay: replay.gradient,
      diagnostics: replay.diagnostics,
      parameterMethods: replay.parameterMethods,
      relativeError: errors,
      passed:
        hasSignal &&
        finite &&
        sameSigns &&
        errors.every((error) => error <= 0.08),
    });
  }
  await device.queue.onSubmittedWorkDone();
  engine.dispose();
}

await device.queue.onSubmittedWorkDone();
device.destroy();

const passed = evidence.every((entry) => entry.passed);
console.log(JSON.stringify({
  adapter: {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
  },
  requiredLimits,
  evidence,
  passed,
}, null, 2));

if (!passed) Deno.exit(1);
