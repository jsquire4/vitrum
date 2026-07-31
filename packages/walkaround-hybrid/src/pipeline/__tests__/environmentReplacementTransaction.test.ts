import { describe, expect, it, vi } from 'vitest';
import { buildDirectionalEnv } from '../../environment/equirectDirectional.js';
import { BvhBufferHost } from '../BvhBufferHost.js';
import {
  createPlaceholderEnvironment,
  disposeEnvironment,
} from '../environmentTexture.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyMockFn = ReturnType<typeof vi.fn<any, any>>;
type MockTexture = {
  readonly label: string;
  readonly destroy: AnyMockFn;
  readonly createView: AnyMockFn;
};

function directional(value: number) {
  return buildDirectionalEnv({
    width: 2,
    height: 1,
    stride: 3,
    data: new Float32Array([
      value, value, value,
      value, value, value,
    ]),
  })!;
}

function mockDevice() {
  const textures: MockTexture[] = [];
  const buffers: Array<{
    readonly label: string;
    readonly destroy: AnyMockFn;
  }> = [];
  let failViews = false;
  const device = {
    createTexture: vi.fn((descriptor: { label?: string }) => {
      const texture: MockTexture = {
        label: descriptor.label ?? '',
        destroy: vi.fn(),
        createView: vi.fn(() => {
          if (failViews) throw new Error('injected createView failure');
          return { texture };
        }),
      };
      textures.push(texture);
      return texture;
    }),
    createSampler: vi.fn(() => ({})),
    createBuffer: vi.fn((descriptor: { label?: string }) => {
      const buffer = {
        label: descriptor.label ?? '',
        destroy: vi.fn(),
      };
      buffers.push(buffer);
      return buffer;
    }),
    queue: {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
    },
  } as unknown as GPUDevice;
  return {
    device,
    textures,
    buffers,
    failViews: () => {
      failViews = true;
    },
  };
}

describe('directional environment replacement transaction', () => {
  it('leaves the old generation live when candidate view creation fails', () => {
    const mock = mockDevice();
    const host = new BvhBufferHost();
    host.updateEnvironment(mock.device, directional(1), 0, 1);
    const oldBindings = host.envBindings()!;
    const oldLiveTextures = mock.textures.filter(
      (texture) => texture.destroy.mock.calls.length === 0,
    );
    const candidateStart = mock.textures.length;

    mock.failViews();
    expect(() => host.prepareEnvironmentReplacement(
      mock.device,
      directional(2),
      0.5,
      3,
    )).toThrow('injected createView failure');

    expect(host.envBindings()).toEqual(oldBindings);
    for (const texture of oldLiveTextures) {
      expect(texture.destroy).not.toHaveBeenCalled();
    }
    for (const texture of mock.textures.slice(candidateStart)) {
      expect(texture.destroy).toHaveBeenCalledOnce();
    }
  });

  it('publishes only by pointer swap and retires old resources at finalize', () => {
    const mock = mockDevice();
    const host = new BvhBufferHost();
    host.updateEnvironment(mock.device, directional(1), 0, 1);
    const oldBindings = host.envBindings()!;
    const oldLiveTextures = mock.textures.filter(
      (texture) => texture.destroy.mock.calls.length === 0,
    );

    const prepared = host.prepareEnvironmentReplacement(
      mock.device,
      directional(2),
      0.25,
      2,
    );
    expect(host.envBindings()).toEqual(oldBindings);
    prepared.commit();
    expect(host.envBindings()!.textureView)
      .toBe(prepared.bindings.textureView);
    for (const texture of oldLiveTextures) {
      expect(texture.destroy).not.toHaveBeenCalled();
    }

    prepared.finalize();
    for (const texture of oldLiveTextures) {
      expect(texture.destroy).toHaveBeenCalledOnce();
    }
  });

  it('restores the old pointer and destroys only the candidate on rollback', () => {
    const mock = mockDevice();
    const host = new BvhBufferHost();
    host.updateEnvironment(mock.device, directional(1), 0, 1);
    const oldBindings = host.envBindings()!;
    const oldLiveTextures = mock.textures.filter(
      (texture) => texture.destroy.mock.calls.length === 0,
    );
    const candidateStart = mock.textures.length;

    const prepared = host.prepareEnvironmentReplacement(
      mock.device,
      directional(2),
      0,
      1,
    );
    prepared.commit();
    prepared.rollback();

    expect(host.envBindings()).toEqual(oldBindings);
    for (const texture of oldLiveTextures) {
      expect(texture.destroy).not.toHaveBeenCalled();
    }
    for (const texture of mock.textures.slice(candidateStart)) {
      expect(texture.destroy).toHaveBeenCalledOnce();
    }
  });

  it('retires every resource best-effort when one destroy throws', () => {
    const mock = mockDevice();
    const env = createPlaceholderEnvironment(mock.device);
    const resources = [
      env.map,
      env.pdf,
      env.marginal,
      env.conditional,
      env.paramsBuffer,
    ] as unknown as Array<{ destroy: AnyMockFn }>;
    resources[0]!.destroy.mockImplementationOnce(() => {
      throw new Error('hostile destroy');
    });

    expect(() => disposeEnvironment(env)).not.toThrow();
    for (const resource of resources) {
      expect(resource.destroy).toHaveBeenCalledOnce();
    }
  });
});
