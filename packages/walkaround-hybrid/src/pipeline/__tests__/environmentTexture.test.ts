/**
 * B3 (road-to-100) — directional IBL GPU-resource host (placeholder + upload +
 * float16 conversion). Device-mocked; verifies the EnvParams uniform packing,
 * the placeholder hasEnv=0 vs uploaded hasEnv=1, and the Float32→Float16 path.
 */
import { describe, it, expect, vi } from 'vitest';
import {
  createPlaceholderEnvironment,
  uploadEnvironment,
  clearEnvironment,
  float32ArrayToFloat16,
} from '../environmentTexture.js';
import { buildDirectionalEnv } from '../../environment/equirectDirectional.js';

interface WriteBufferCall { offset: number; data: ArrayBuffer; }
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyMockFn = ReturnType<typeof vi.fn<any, any>>;
interface MockTexture { format: string; width: number; height: number; destroy: AnyMockFn; createView: AnyMockFn; }
type TextureWriteCall = [
  { texture: MockTexture },
  Float32Array,
  { bytesPerRow: number },
];
function mockDevice(texDims?: { width: number; height: number }) {
  const writeBufferCalls: WriteBufferCall[] = [];
  const writeTextureCalls: unknown[] = [];
  const device = {
    createTexture: vi.fn((d: { format: string; size: { width: number; height: number } }) => {
      const tex: MockTexture = {
        format: d.format,
        // Use explicitly provided dims when the mock needs to simulate a
        // same-size existing texture (for the reuse test).
        width:  texDims?.width  ?? d.size.width,
        height: texDims?.height ?? d.size.height,
        destroy: vi.fn(),
        createView: vi.fn(() => ({})),
      };
      return tex;
    }),
    createSampler: vi.fn(() => ({})),
    createBuffer: vi.fn((d: { size: number }) => ({ size: d.size, destroy: vi.fn() })),
    queue: {
      writeBuffer: vi.fn((_buf: unknown, offset: number, data: ArrayBuffer) => {
        writeBufferCalls.push({ offset, data: data.slice(0) });
      }),
      writeTexture: vi.fn((...args: unknown[]) => { writeTextureCalls.push(args); }),
    },
  } as unknown as GPUDevice;
  return { device, writeBufferCalls, writeTextureCalls };
}

function readParams(data: ArrayBuffer) {
  const u = new Uint32Array(data);
  const f = new Float32Array(data);
  return { hasEnv: u[0], width: u[1], height: u[2], rotationY: f[3], intensity: f[4] };
}

describe('environmentTexture — B3 directional IBL host', () => {
  it('placeholder writes hasEnv=0', () => {
    const { device, writeBufferCalls } = mockDevice();
    const env = createPlaceholderEnvironment(device);
    expect(writeBufferCalls).toHaveLength(1);
    expect(readParams(writeBufferCalls[0]!.data).hasEnv).toBe(0);
    expect(env.rotationY).toBe(0);
    expect(env.intensity).toBe(0);
    expect(env.hasDirectionalEnvironment).toBe(false);
  });

  it('upload writes hasEnv=1 + dims + rotationY + intensity', () => {
    const { device, writeBufferCalls, writeTextureCalls } = mockDevice();
    const env = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 4, height: 2, stride: 3,
      data: new Float32Array(4 * 2 * 3).fill(1),
    })!;
    const uploaded = uploadEnvironment(device, env, data, 0.5, 2.0);
    const last = writeBufferCalls.at(-1)!;
    const p = readParams(last.data);
    expect(p.hasEnv).toBe(1);
    expect(p.width).toBe(4);
    expect(p.height).toBe(2);
    expect(p.rotationY).toBeCloseTo(0.5);
    expect(p.intensity).toBeCloseTo(2.0);
    expect(uploaded.rotationY).toBeCloseTo(0.5);
    expect(uploaded.intensity).toBeCloseTo(2.0);
    expect(uploaded.hasDirectionalEnvironment).toBe(true);
    // radiance + pdf + marginal + conditional → 4 writeTexture calls.
    expect(writeTextureCalls.length).toBe(4);
  });

  it('clear resets to hasEnv=0', () => {
    const { device, writeBufferCalls } = mockDevice();
    const env0 = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 2, height: 1, stride: 3, data: new Float32Array([1, 1, 1, 2, 2, 2]),
    })!;
    const env1 = uploadEnvironment(device, env0, data, 0, 1);
    const cleared = clearEnvironment(device, env1);
    expect(readParams(writeBufferCalls.at(-1)!.data).hasEnv).toBe(0);
    expect(cleared.rotationY).toBe(0);
    expect(cleared.intensity).toBe(0);
    expect(cleared.hasDirectionalEnvironment).toBe(false);
  });

  it('radiance is rgba32float and pdf/CDF textures are r32float', () => {
    const { device } = mockDevice();
    let env = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 2, height: 1, stride: 3, data: new Float32Array([1, 1, 1, 2, 2, 2]),
    })!;
    env = uploadEnvironment(device, env, data, 0, 1);
    expect((env.map as unknown as { format: string }).format).toBe('rgba32float');
    expect((env.pdf as unknown as { format: string }).format).toBe('r32float');
    expect((env.marginal as unknown as { format: string }).format).toBe('r32float');
    expect((env.conditional as unknown as { format: string }).format).toBe('r32float');
  });

  // ── Transactional replacement, including same-size uploads ─────────────────

  it('same-size re-upload stages a complete replacement before retiring the old set', () => {
    // Build a mock device whose createTexture reports width=4 / height=2 so the
    // existing textures appear to already be the right size.
    const { device } = mockDevice({ width: 4, height: 2 });
    let env = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 4, height: 2, stride: 3,
      data: new Float32Array(4 * 2 * 3).fill(0.5),
    })!;
    const mapBefore = env.map;
    const pdfBefore = env.pdf;
    const marginalBefore = env.marginal;
    const conditionalBefore = env.conditional;

    env = uploadEnvironment(device, env, data, 0.1, 1.5);

    // Even same-sized updates stage a complete replacement so a later failed
    // write cannot partially mutate the live environment.
    expect(env.map).not.toBe(mapBefore);
    expect(env.pdf).not.toBe(pdfBefore);
    expect(env.marginal).not.toBe(marginalBefore);
    expect(env.conditional).not.toBe(conditionalBefore);
    expect((mapBefore as unknown as { destroy: ReturnType<typeof vi.fn> }).destroy).toHaveBeenCalledOnce();
    expect((pdfBefore as unknown as { destroy: ReturnType<typeof vi.fn> }).destroy).toHaveBeenCalledOnce();
    expect((marginalBefore as unknown as { destroy: ReturnType<typeof vi.fn> }).destroy).toHaveBeenCalledOnce();
    expect((conditionalBefore as unknown as { destroy: ReturnType<typeof vi.fn> }).destroy).toHaveBeenCalledOnce();
  });

  it('different-size re-upload destroys old textures and creates new ones', () => {
    // Default mock: createTexture reports whatever size it was asked for.
    // The placeholder will report 1×1; the upload asks for 4×2 → mismatch → recreate.
    const { device } = mockDevice();
    let env = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 4, height: 2, stride: 3,
      data: new Float32Array(4 * 2 * 3).fill(0.5),
    })!;
    const mapBefore = env.map;

    env = uploadEnvironment(device, env, data, 0, 1);

    // The old map must have been destroyed.
    expect((mapBefore as unknown as { destroy: ReturnType<typeof vi.fn> }).destroy).toHaveBeenCalled();
    // And a new one was created at the larger size.
    expect(env.map).not.toBe(mapBefore);
  });

  it('marginal texture is H wide and 1 tall (H×1 convention, matching WGSL textureLoad(…, vec2i(row, 0)))', () => {
    // Pin the marginal allocation convention — H×1, not 1×H — so a future
    // refactor cannot silently swap the dimensions and break the WGSL consumer.
    const { device } = mockDevice();
    let env = createPlaceholderEnvironment(device);
    const w = 4; const h = 2;
    const data = buildDirectionalEnv({
      width: w, height: h, stride: 3,
      data: new Float32Array(w * h * 3).fill(0.5),
    })!;
    env = uploadEnvironment(device, env, data, 0, 1);
    const marginal = env.marginal as unknown as { width: number; height: number };
    expect(marginal.width).toBe(h);   // H wide
    expect(marginal.height).toBe(1);  // 1 tall
  });

  it('preserves the live environment and destroys candidates when an upload fails', () => {
    const { device } = mockDevice();
    const env = createPlaceholderEnvironment(device);
    const oldTextures = [env.map, env.pdf, env.marginal, env.conditional] as unknown as MockTexture[];
    const data = buildDirectionalEnv({
      width: 4, height: 2, stride: 3,
      data: new Float32Array(4 * 2 * 3).fill(0.5),
    })!;
    const writeTexture = device.queue.writeTexture as unknown as AnyMockFn;
    writeTexture
      .mockImplementationOnce(() => undefined)
      .mockImplementationOnce(() => { throw new Error('injected upload failure'); });

    expect(() => uploadEnvironment(device, env, data, 0, 1))
      .toThrow('injected upload failure');

    for (const texture of oldTextures) expect(texture.destroy).not.toHaveBeenCalled();
    const createTexture = device.createTexture as unknown as AnyMockFn;
    const candidates = createTexture.mock.results.slice(-4)
      .map((result) => result.value as MockTexture);
    for (const texture of candidates) expect(texture.destroy).toHaveBeenCalledOnce();
  });

  it('rounds environment radiance to IEEE f16 ties-to-even', () => {
    const bits = float32ArrayToFloat16(new Float32Array([
      2 ** -24, 2 ** -25, 1 + 2 ** -11, 1 + 3 * 2 ** -11,
      1.99951171875, 65504, Number.POSITIVE_INFINITY, Number.NaN,
    ]));
    expect(Array.from(bits.slice(0, 7))).toEqual([
      0x0001, 0x0000, 0x3c00, 0x3c02, 0x4000, 0x7bff, 0x7c00,
    ]);
    expect(bits[7]! & 0x7c00).toBe(0x7c00);
    expect(bits[7]! & 0x03ff).not.toBe(0);
  });

  it('uploads HDR radiance and sharp PDFs as full-range Float32 payloads', () => {
    const { device, writeTextureCalls } = mockDevice();
    const env = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 2,
      height: 1,
      stride: 3,
      data: new Float32Array([1_000_000, 2, 3, 1, 1, 1]),
    })!;
    data.pdf[0] = 1_000_000;
    uploadEnvironment(device, env, data, 0, 1);
    const mapWrite = writeTextureCalls[0] as TextureWriteCall;
    const pdfWrite = writeTextureCalls[1] as TextureWriteCall;
    expect(mapWrite[0].texture.format).toBe('rgba32float');
    expect(mapWrite[2].bytesPerRow).toBe(2 * 4 * 4);
    expect(mapWrite[1][0]).toBe(1_000_000);
    expect(pdfWrite[0].texture.format).toBe('r32float');
    expect(pdfWrite[1][0]).toBe(1_000_000);
  });

  it('uploads only the addressed bytes of radiance and PDF subarray views', () => {
    const { device, writeTextureCalls } = mockDevice();
    const env = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 2,
      height: 1,
      stride: 3,
      data: new Float32Array([3, 4, 5, 6, 7, 8]),
    })!;
    const mapBacking = new Float32Array(data.map.length + 4).fill(-1);
    const mapView = mapBacking.subarray(2, 2 + data.map.length);
    mapView.set(data.map);
    const pdfBacking = new Float32Array(data.pdf.length + 4).fill(-1);
    const pdfView = pdfBacking.subarray(2, 2 + data.pdf.length);
    pdfView.set(data.pdf);

    uploadEnvironment(
      device,
      env,
      { ...data, map: mapView, pdf: pdfView },
      0,
      1,
    );

    const mapWrite = writeTextureCalls[0] as TextureWriteCall;
    const pdfWrite = writeTextureCalls[1] as TextureWriteCall;
    expect(mapWrite[1]).toBe(mapView);
    expect(pdfWrite[1]).toBe(pdfView);
    expect(Array.from(mapWrite[1])).toEqual(Array.from(data.map));
    expect(Array.from(pdfWrite[1])).toEqual(Array.from(data.pdf));
  });
});
