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
function mockDevice() {
  const writeBufferCalls: WriteBufferCall[] = [];
  const writeTextureCalls: unknown[] = [];
  const device = {
    createTexture: vi.fn((d: { format: string }) => ({
      format: d.format,
      destroy: vi.fn(),
      createView: vi.fn(() => ({})),
    })),
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
    let env = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 4, height: 2, stride: 3,
      data: new Float32Array(4 * 2 * 3).fill(1),
    })!;
    env = uploadEnvironment(device, env, data, 0.5, 2.0);
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
    let env = createPlaceholderEnvironment(device);
    const data = buildDirectionalEnv({
      width: 2, height: 1, stride: 3, data: new Float32Array([1, 1, 1, 2, 2, 2]),
    })!;
    env = uploadEnvironment(device, env, data, 0, 1);
    env = clearEnvironment(device, env);
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
});
