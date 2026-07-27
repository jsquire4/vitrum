import { describe, expect, it, vi } from 'vitest';
import { RCDispatcher } from '../src/cascadeDispatch.js';

function hostileResource(index: number) {
  return {
    destroy: vi.fn(() => {
      if (index === 0 || index === 5) throw new Error(`injected destroy failure ${index}`);
    }),
  };
}

describe('RCDispatcher resource teardown', () => {
  it('releases every uniform, placeholder, and dummy buffer despite hostile destroys', () => {
    const owned = Array.from({ length: 13 }, (_, index) => hostileResource(index));
    const dispatcher = new RCDispatcher();
    const internal = dispatcher as unknown as {
      _handles: unknown;
      _bindingSignature: unknown;
    };
    internal._handles = {
      ownedResources: owned,
    };
    internal._bindingSignature = {};

    expect(() => dispatcher.dispose()).not.toThrow();
    for (const resource of owned) expect(resource.destroy).toHaveBeenCalledOnce();
    expect(internal._handles).toBeNull();
    expect(internal._bindingSignature).toBeNull();

    dispatcher.dispose();
    for (const resource of owned) expect(resource.destroy).toHaveBeenCalledOnce();
  });
});
