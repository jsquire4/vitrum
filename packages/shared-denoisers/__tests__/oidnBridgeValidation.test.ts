import { afterEach, describe, expect, it, vi } from 'vitest';

import type {
  OIDNDenoiseInputs,
  OIDNDenoiseOptions,
} from '../src/oidnBridge.js';

afterEach(() => {
  vi.doUnmock('onnxruntime-web');
  vi.resetModules();
});

describe('denoiseFinal public-boundary validation', () => {
  it('rejects malformed requests before loading ORT or acquiring a session', async () => {
    vi.resetModules();
    let ortLoads = 0;
    vi.doMock('onnxruntime-web', () => {
      ortLoads += 1;
      return {
        Tensor: class {},
        InferenceSession: { create: vi.fn() },
      };
    });
    const bridge = await import('../src/oidnBridge.js');
    const rgb = (values = 3): Float32Array => new Float32Array(values);
    const requests: Array<{
      readonly inputs: OIDNDenoiseInputs;
      readonly opts: OIDNDenoiseOptions;
      readonly message: string;
    }> = [
      {
        inputs: { color: rgb(), width: 1, height: 1 },
        opts: { modelUrl: '' },
        message: 'modelUrl must be a non-empty string',
      },
      {
        inputs: { color: rgb(), width: 1, height: 1 },
        opts: { modelUrl: '   ' },
        message: 'modelUrl must be a non-empty string',
      },
      ...[0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY].map((width) => ({
        inputs: { color: rgb(), width, height: 1 },
        opts: { modelUrl: '/model.onnx' },
        message: 'width must be a positive safe integer',
      })),
      ...[0, -1, 1.5, Number.NaN, Number.NEGATIVE_INFINITY].map((height) => ({
        inputs: { color: rgb(), width: 1, height },
        opts: { modelUrl: '/model.onnx' },
        message: 'height must be a positive safe integer',
      })),
      {
        inputs: {
          color: rgb(),
          width: Number.MAX_SAFE_INTEGER,
          height: 2,
        },
        opts: { modelUrl: '/model.onnx' },
        message: 'exceeds the safe integer range',
      },
      {
        inputs: { color: rgb(2), width: 1, height: 1 },
        opts: { modelUrl: '/model.onnx' },
        message: 'color length must be exactly 3; received 2',
      },
      {
        inputs: { color: rgb(4), width: 1, height: 1 },
        opts: { modelUrl: '/model.onnx' },
        message: 'color length must be exactly 3; received 4',
      },
      {
        inputs: { color: rgb(), normal: rgb(2), width: 1, height: 1 },
        opts: { modelUrl: '/model.onnx' },
        message: 'normal length must be exactly 3; received 2',
      },
      {
        inputs: { color: rgb(), albedo: rgb(4), width: 1, height: 1 },
        opts: { modelUrl: '/model.onnx' },
        message: 'albedo length must be exactly 3; received 4',
      },
      {
        inputs: {
          color: new Float32Array([0, Number.NaN, 0]),
          width: 1,
          height: 1,
        },
        opts: { modelUrl: '/model.onnx' },
        message: 'color[1] must be finite',
      },
      {
        inputs: {
          color: rgb(),
          normal: new Float32Array([0, Number.POSITIVE_INFINITY, 0]),
          width: 1,
          height: 1,
        },
        opts: { modelUrl: '/model.onnx' },
        message: 'normal[1] must be finite',
      },
      {
        inputs: {
          color: rgb(),
          albedo: new Float32Array([0, 0, Number.NEGATIVE_INFINITY]),
          width: 1,
          height: 1,
        },
        opts: { modelUrl: '/model.onnx' },
        message: 'albedo[2] must be finite',
      },
    ];

    for (const request of requests) {
      await expect(
        bridge.denoiseFinal(request.inputs, request.opts),
      ).rejects.toThrow(request.message);
    }
    const malformedOptions: ReadonlyArray<{
      readonly opts: unknown;
      readonly message: string;
    }> = [
      { opts: null, message: 'options must be an object' },
      { opts: [], message: 'options must be an object' },
      {
        opts: { modelUrl: '/model.onnx', unexpected: true },
        message: 'options contains unknown key(s): unexpected',
      },
      {
        opts: { modelUrl: '/model.onnx', executionProviders: [] },
        message: 'executionProviders must be a non-empty array',
      },
      {
        opts: { modelUrl: '/model.onnx', executionProviders: 'wasm' },
        message: 'executionProviders must be a non-empty array',
      },
      {
        opts: { modelUrl: '/model.onnx', executionProviders: ['cuda'] },
        message: 'unsupported execution provider cuda',
      },
      {
        opts: {
          modelUrl: '/model.onnx',
          executionProviders: ['wasm', 'wasm'],
        },
        message: "executionProviders contains duplicate 'wasm'",
      },
      {
        opts: { modelUrl: '/model.onnx', tensorNames: null },
        message: 'tensorNames must be an object',
      },
      {
        opts: { modelUrl: '/model.onnx', tensorNames: [] },
        message: 'tensorNames must be an object',
      },
      {
        opts: {
          modelUrl: '/model.onnx',
          tensorNames: { unexpected: 'value' },
        },
        message: 'tensorNames contains unknown key(s): unexpected',
      },
      {
        opts: {
          modelUrl: '/model.onnx',
          tensorNames: { output: '   ' },
        },
        message: 'tensorNames.output must be a non-empty string',
      },
      {
        opts: {
          modelUrl: '/model.onnx',
          tensorNames: { normal: 42 },
        },
        message: 'tensorNames.normal must be a non-empty string',
      },
      {
        opts: {
          modelUrl: '/model.onnx',
          tensorNames: { color: 'shared', normal: 'shared' },
        },
        message: 'feed tensor names must be unique',
      },
      {
        opts: {
          modelUrl: '/model.onnx',
          tensorNames: { color: 'normal' },
        },
        message: 'feed tensor names must be unique',
      },
    ];
    for (const malformed of malformedOptions) {
      await expect(bridge.denoiseFinal(
        { color: rgb(), width: 1, height: 1 },
        malformed.opts as OIDNDenoiseOptions,
      )).rejects.toThrow(malformed.message);
    }
    await expect(
      bridge.preloadOIDNModel({ modelUrl: '' }),
    ).rejects.toThrow('modelUrl must be a non-empty string');
    await expect(
      bridge.acquireOIDNSession({ modelUrl: '  ' }),
    ).rejects.toThrow('modelUrl must be a non-empty string');
    expect(() => bridge.releaseOIDNCacheEntry({ modelUrl: '' }))
      .toThrow('modelUrl must be a non-empty string');
    expect(() => bridge.releaseOIDNCacheEntry({
      modelUrl: '/model.onnx',
      executionProviders: [],
    })).toThrow('executionProviders must be a non-empty array');
    expect(ortLoads).toBe(0);
  });

  it('preserves valid HWC-to-NCHW feed and NCHW-to-HWC result layout', async () => {
    vi.resetModules();
    const tensors: Array<{
      readonly data: Float32Array;
      readonly dims: number[];
    }> = [];
    const run = vi.fn(async () => ({
        output: {
          data: new Float32Array([
          101, 104, 107, 110,
          102, 105, 108, 111,
            103, 106, 109, 112,
          ]),
          dims: [1, 3, 2, 2],
      },
    }));
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {
        constructor(_type: string, data: Float32Array, dims: number[]) {
          tensors.push({ data, dims });
        }
      },
      InferenceSession: {
        create: vi.fn(async () => ({ run, release: vi.fn() })),
      },
    }));
    const bridge = await import('../src/oidnBridge.js');
    const result = await bridge.denoiseFinal(
      {
        color: new Float32Array([
          1, 2, 3,
          4, 5, 6,
          7, 8, 9,
          10, 11, 12,
        ]),
        width: 2,
        height: 2,
      },
      { modelUrl: '/layout.onnx', executionProviders: ['wasm'] },
    );

    expect(tensors).toHaveLength(1);
    expect(tensors[0]?.dims).toEqual([1, 3, 2, 2]);
    expect(Array.from(tensors[0]!.data)).toEqual([
      1, 4, 7, 10,
      2, 5, 8, 11,
      3, 6, 9, 12,
    ]);
    expect(Array.from(result)).toEqual([
      101, 102, 103,
      104, 105, 106,
      107, 108, 109,
      110, 111, 112,
    ]);
    bridge.clearOIDNCache();
    await Promise.resolve();
  });

  it('rejects malformed model output instead of zero-filling it', async () => {
    vi.resetModules();
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {},
      InferenceSession: {
        create: vi.fn(async () => ({
          run: vi.fn(async () => ({
              output: {
                data: new Float32Array(2),
                dims: [1, 3, 1, 1],
              },
          })),
          release: vi.fn(),
        })),
      },
    }));
    const bridge = await import('../src/oidnBridge.js');
    await expect(bridge.denoiseFinal(
      { color: new Float32Array(3), width: 1, height: 1 },
      { modelUrl: '/bad-output.onnx', executionProviders: ['wasm'] },
    )).rejects.toThrow('model output length must be exactly 3; received 2');
    bridge.clearOIDNCache();
    await Promise.resolve();
  });

  it('requires the exact explicit output name and never applies its legacy alias', async () => {
    vi.resetModules();
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {},
      InferenceSession: {
        create: vi.fn(async () => ({
          run: vi.fn(async () => ({
            color: {
              data: new Float32Array([1, 2, 3]),
              dims: [1, 3, 1, 1],
            },
          })),
          release: vi.fn(),
        })),
      },
    }));
    const bridge = await import('../src/oidnBridge.js');

    await expect(bridge.denoiseFinal(
      { color: new Float32Array(3), width: 1, height: 1 },
      {
        modelUrl: '/explicit-output.onnx',
        executionProviders: ['wasm'],
        tensorNames: { output: 'denoised' },
      },
    )).rejects.toThrow("Expected output named 'denoised'. Got keys: color");
    bridge.clearOIDNCache();
    await Promise.resolve();
  });

  it('retains the color output alias only when no output name is configured', async () => {
    vi.resetModules();
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {},
      InferenceSession: {
        create: vi.fn(async () => ({
          run: vi.fn(async () => ({
            color: {
              data: new Float32Array([7, 8, 9]),
              dims: [1, 3, 1, 1],
            },
          })),
          release: vi.fn(),
        })),
      },
    }));
    const bridge = await import('../src/oidnBridge.js');

    await expect(bridge.denoiseFinal(
      { color: new Float32Array(3), width: 1, height: 1 },
      { modelUrl: '/legacy-output.onnx', executionProviders: ['wasm'] },
    )).resolves.toEqual(new Float32Array([7, 8, 9]));
    bridge.clearOIDNCache();
    await Promise.resolve();
  });

  it('rejects a same-length output whose declared layout is not exact NCHW', async () => {
    vi.resetModules();
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {},
      InferenceSession: {
        create: vi.fn(async () => ({
          run: vi.fn(async () => ({
            output: {
              data: new Float32Array(12),
              dims: [1, 2, 2, 3],
            },
          })),
          release: vi.fn(),
        })),
      },
    }));
    const bridge = await import('../src/oidnBridge.js');

    await expect(bridge.denoiseFinal(
      { color: new Float32Array(12), width: 2, height: 2 },
      { modelUrl: '/hwc-output.onnx', executionProviders: ['wasm'] },
    )).rejects.toThrow(
      'model output dims must be exactly [1, 3, 2, 2]; received [1, 2, 2, 3]',
    );
    bridge.clearOIDNCache();
    await Promise.resolve();
  });
});
