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
} from '../environmentTexture.js';
import { buildDirectionalEnv } from '../../environment/equirectDirectional.js';

interface WriteBufferCall { offset: number; data: ArrayBuffer; }
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyMockFn = ReturnType<typeof vi.fn<any, any>>;
interface MockTexture { format: string; width: number; height: number; destroy: AnyMockFn; createView: AnyMockFn; }
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
    createPlaceholderEnvironment(device);
    expect(writeBufferCalls).toHaveLength(1);
    expect(readParams(writeBufferCalls[0]!.data).hasEnv).toBe(0);
  });

  it('upload writes hasEnv=1 + dims + rotationY + intensity', () => {
    const { device, writeBufferCalls, writeTextureCalls } = mockDevice();
    const env = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 4, height: 2, stride: 3,
      data: new Float32Array(4 * 2 * 3).fill(1),
    })!;
    void uploadEnvironment(device, env, data, 0.5, 2.0);
    const last = writeBufferCalls.at(-1)!;
    const p = readParams(last.data);
    expect(p.hasEnv).toBe(1);
    expect(p.width).toBe(4);
    expect(p.height).toBe(2);
    expect(p.rotationY).toBeCloseTo(0.5);
    expect(p.intensity).toBeCloseTo(2.0);
    // map + marginal + conditional → 3 writeTexture calls.
    expect(writeTextureCalls.length).toBe(3);
  });

  it('clear resets to hasEnv=0', () => {
    const { device, writeBufferCalls } = mockDevice();
    const env0 = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 2, height: 1, stride: 3, data: new Float32Array([1, 1, 1, 2, 2, 2]),
    })!;
    const env1 = uploadEnvironment(device, env0, data, 0, 1);
    void clearEnvironment(device, env1);
    expect(readParams(writeBufferCalls.at(-1)!.data).hasEnv).toBe(0);
  });

  it('env_map is created as rgba16float; CDF textures as r32float', () => {
    const { device } = mockDevice();
    let env = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 2, height: 1, stride: 3, data: new Float32Array([1, 1, 1, 2, 2, 2]),
    })!;
    env = uploadEnvironment(device, env, data, 0, 1);
    expect((env.map as unknown as { format: string }).format).toBe('rgba16float');
    expect((env.marginal as unknown as { format: string }).format).toBe('r32float');
    expect((env.conditional as unknown as { format: string }).format).toBe('r32float');
  });

  // ── Item 17b: same-size reuse (write-in-place, no destroy+recreate) ─────────

  it('same-size re-upload reuses the existing GPUTexture objects (no destroy)', () => {
    // Build a mock device whose createTexture reports width=4 / height=2 so the
    // existing textures appear to already be the right size.
    const { device } = mockDevice({ width: 4, height: 2 });
    let env = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 4, height: 2, stride: 3,
      data: new Float32Array(4 * 2 * 3).fill(0.5),
    })!;
    const mapBefore = env.map;
    const marginalBefore = env.marginal;
    const conditionalBefore = env.conditional;

    env = uploadEnvironment(device, env, data, 0.1, 1.5);

    // The GPUTexture objects must be the same references — not destroyed and
    // recreated — because the dimensions already match.
    expect(env.map).toBe(mapBefore);
    expect(env.marginal).toBe(marginalBefore);
    expect(env.conditional).toBe(conditionalBefore);

    // destroy() must NOT have been called on the old textures.
    expect((mapBefore as unknown as { destroy: ReturnType<typeof vi.fn> }).destroy).not.toHaveBeenCalled();
    expect((marginalBefore as unknown as { destroy: ReturnType<typeof vi.fn> }).destroy).not.toHaveBeenCalled();
    expect((conditionalBefore as unknown as { destroy: ReturnType<typeof vi.fn> }).destroy).not.toHaveBeenCalled();
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
});
