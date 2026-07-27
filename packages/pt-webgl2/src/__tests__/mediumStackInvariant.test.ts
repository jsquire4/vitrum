import { describe, expect, it } from 'vitest';
import { RENDER_MAIN } from '../glsl/renderMain.glsl.js';
import * as AttenuateHitFns from '../glsl/render/attenuate_hit_function.glsl.js';
import { ATTENUATE_HIT_SCALAR_RICH_GLSL } from '../glsl/render/attenuate_hit_scalar_rich.glsl.js';
import * as BdptLightSubpath from '../glsl/render/bdpt_light_subpath.glsl.js';
import { FOG_MATERIAL_GLSL } from '../glsl/shader/structs/fog_material.glsl.js';

const attenuate_hit_function = (
  AttenuateHitFns as unknown as Record<string, string>
)['attenuate_hit_function']!;
const bdpt_light_subpath = (
  BdptLightSubpath as unknown as Record<string, string>
)['bdpt_light_subpath']!;

function enter(stack: number[], materialId: number): boolean {
  if (stack.length >= 8) return false;
  stack.push(materialId);
  return true;
}

function leave(stack: number[], materialId: number): boolean {
  if (stack.length === 0 || stack[stack.length - 1] !== materialId) return false;
  stack.pop();
  return true;
}

describe('WebGL2 nested-medium LIFO invariant', () => {
  it('accepts A -> B -> B-exit -> A-exit and refreshes the enclosing top', () => {
    const stack: number[] = [];
    expect(enter(stack, 11)).toBe(true);
    expect(enter(stack, 22)).toBe(true);
    expect(stack).toEqual([11, 22]);
    expect(leave(stack, 22)).toBe(true);
    expect(stack).toEqual([11]);
    expect(leave(stack, 11)).toBe(true);
    expect(stack).toEqual([]);
  });

  it('rejects an out-of-order A exit while B is current and leaves the stack unchanged', () => {
    const stack = [11, 22];
    expect(leave(stack, 11)).toBe(false);
    expect(stack).toEqual([11, 22]);

    const leaveSource = FOG_MATERIAL_GLSL.slice(
      FOG_MATERIAL_GLSL.indexOf('bool leaveMedium'),
    );
    expect(leaveSource).toContain('int top = stack.count - 1;');
    expect(leaveSource).toContain('stack.materialIds[ top ] != materialId');
    expect(leaveSource).toContain('stack.count = top;');
    expect(leaveSource).not.toContain('removeIndex');
  });

  it('fails closed at every eye, visibility, scalar-rich, and BDPT stack transition', () => {
    expect(RENDER_MAIN).toMatch(
      /bool mediumStackValid[\s\S]*?leaveMedium\([\s\S]*?if \( ! mediumStackValid \) break;/,
    );
    for (const visibilitySource of [
      attenuate_hit_function,
      ATTENUATE_HIT_SCALAR_RICH_GLSL,
    ]) {
      expect(visibilitySource).toMatch(
        /bool stackValid[\s\S]*?leaveMedium\([\s\S]*?if \( ! stackValid \) \{[\s\S]*?result = true;[\s\S]*?break;/,
      );
    }
    expect(bdpt_light_subpath).toMatch(
      /bool stackValid[\s\S]*?leaveMedium\([\s\S]*?if \( ! stackValid \) break;/,
    );
  });

  it('round-trips BDPT medium order without sorting or deduplicating repeated ids', () => {
    expect(bdpt_light_subpath).toContain(
      'float materialId = float( stack.materialIds[ i ] );',
    );
    expect(bdpt_light_subpath).toContain(
      'stack.materialIds[ i ] = uint( round( packedIds[ i ] ) );',
    );
    expect(bdpt_light_subpath).toContain(
      'The final packed id remains the top/innermost medium.',
    );

    const stack: number[] = [];
    expect(enter(stack, 7)).toBe(true);
    expect(enter(stack, 7)).toBe(true);
    const packed = [...stack];
    const unpacked = [...packed];
    expect(leave(unpacked, 7)).toBe(true);
    expect(unpacked).toEqual([7]);
    expect(leave(unpacked, 7)).toBe(true);
    expect(unpacked).toEqual([]);
  });
});
