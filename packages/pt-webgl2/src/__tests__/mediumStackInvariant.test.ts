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

interface MediumIdentity {
  readonly boundaryId: number;
  readonly materialId: number;
}

function enter(stack: MediumIdentity[], boundaryId: number, materialId: number): boolean {
  if (stack.length >= 8) return false;
  if (boundaryId === 0) return false;
  stack.push({ boundaryId, materialId });
  return true;
}

function leave(stack: MediumIdentity[], boundaryId: number, materialId: number): boolean {
  const top = stack.at(-1);
  if (
    top == null || top.boundaryId !== boundaryId ||
    top.materialId !== materialId
  ) return false;
  stack.pop();
  return true;
}

describe('WebGL2 nested-medium LIFO invariant', () => {
  it('accepts A -> B -> B-exit -> A-exit and refreshes the enclosing top', () => {
    const stack: MediumIdentity[] = [];
    expect(enter(stack, 101, 11)).toBe(true);
    expect(enter(stack, 202, 22)).toBe(true);
    expect(stack).toEqual([
      { boundaryId: 101, materialId: 11 },
      { boundaryId: 202, materialId: 22 },
    ]);
    expect(leave(stack, 202, 22)).toBe(true);
    expect(stack).toEqual([{ boundaryId: 101, materialId: 11 }]);
    expect(leave(stack, 101, 11)).toBe(true);
    expect(stack).toEqual([]);
  });

  it('rejects an out-of-order A exit while B is current and leaves the stack unchanged', () => {
    const stack: MediumIdentity[] = [
      { boundaryId: 101, materialId: 11 },
      { boundaryId: 202, materialId: 22 },
    ];
    expect(leave(stack, 101, 11)).toBe(false);
    expect(stack).toHaveLength(2);

    const leaveSource = FOG_MATERIAL_GLSL.slice(
      FOG_MATERIAL_GLSL.indexOf('bool leaveMedium'),
    );
    expect(leaveSource).toContain('int top = stack.count - 1;');
    expect(leaveSource).toContain('stack.boundaryIds[ top ] != boundaryId');
    expect(leaveSource).toContain('stack.materialIds[ top ] != materialId');
    expect(leaveSource).toContain('stack.count = top;');
    expect(leaveSource).not.toContain('removeIndex');
  });

  it('fails closed at radiance and BDPT transitions while visibility never collapses an interface', () => {
    expect(RENDER_MAIN).toMatch(
      /bool mediumStackValid[\s\S]*?leaveMedium\([\s\S]*?if \( ! mediumStackValid \) break;/,
    );
    for (const visibilitySource of [
      attenuate_hit_function,
      ATTENUATE_HIT_SCALAR_RICH_GLSL,
    ]) {
      expect(visibilitySource).toContain('filterShadowMediumStack(');
      expect(visibilitySource).toContain('opticalVisibilitySegmentTransmittance(');
      expect(visibilitySource).not.toContain('enterMedium(');
      expect(visibilitySource).not.toContain('leaveMedium(');
      expect(visibilitySource).toMatch(/accepted physical surface[\s\S]*?result = true;[\s\S]*?break;/);
    }
    expect(bdpt_light_subpath).toMatch(
      /bool stackValid[\s\S]*?leaveMedium\([\s\S]*?if \( ! stackValid \)[\s\S]*?writeBdptInvalidVertex[\s\S]*?return;/,
    );
  });

  it('round-trips BDPT medium order without conflating same-material boundaries', () => {
    expect(bdpt_light_subpath).toContain(
      'float materialId = float( stack.materialIds[ i ] );',
    );
    expect(bdpt_light_subpath).toContain(
      'stack.materialIds[ i ] = uint( round( packedIds[ i ] ) );',
    );
    expect(bdpt_light_subpath).toContain(
      'stack.boundaryIds[ i ] = uint( packedBoundaryId );',
    );
    expect(bdpt_light_subpath).toContain('bdptStoredBoundaryRowsValid(');
    expect(bdpt_light_subpath).toContain(
      'The final packed id remains the top/innermost medium.',
    );

    const stack: MediumIdentity[] = [];
    expect(enter(stack, 41, 7)).toBe(true);
    expect(enter(stack, 42, 7)).toBe(true);
    const packed = [...stack];
    const unpacked = [...packed];
    expect(leave(unpacked, 41, 7)).toBe(false);
    expect(unpacked).toHaveLength(2);
    expect(leave(unpacked, 42, 7)).toBe(true);
    expect(unpacked).toEqual([{ boundaryId: 41, materialId: 7 }]);
    expect(leave(unpacked, 41, 7)).toBe(true);
    expect(unpacked).toEqual([]);
  });
});
