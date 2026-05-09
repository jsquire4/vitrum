import { describe, it, expect } from 'vitest';
import { createPTEngine_WebGL2 } from '../index.js';

describe('createPTEngine_WebGL2', () => {
  it('rejects null/invalid device', async () => {
    await expect(createPTEngine_WebGL2({ device: null as never })).rejects.toThrow(TypeError);
    await expect(
      createPTEngine_WebGL2({ device: {} as never }),
    ).rejects.toThrow(TypeError);
  });
});
