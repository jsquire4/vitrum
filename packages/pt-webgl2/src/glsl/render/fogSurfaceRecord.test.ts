import { describe, expect, it } from 'vitest';
import { RENDER_MAIN } from '../renderMain.glsl.js';
import * as SurfaceRecordSource from '../shader/structs/surface_record_struct.glsl.js';
import * as RenderStructSource from './render_structs.glsl.js';
import * as TraceSceneSource from './trace_scene_function.glsl.js';

const surfaceRecordStruct = (
  SurfaceRecordSource as unknown as Record<string, string>
).surface_record_struct!;
const renderStructs = (
  RenderStructSource as unknown as Record<string, string>
).render_structs!;
const traceScene = (
  TraceSceneSource as unknown as Record<string, string>
).trace_scene_function!;

function blockBetween(source: string, start: string, end: string): string {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  expect(from).toBeGreaterThanOrEqual(0);
  expect(to).toBeGreaterThan(from);
  return source.slice(from, to);
}

describe('setFogSurfaceRecord', () => {
  it('initializes miss distance before BVH traversal and fog free-flight comparison', () => {
    const missDistance = traceScene.indexOf('surfaceHit.dist = INFINITY;');
    const traversal = traceScene.indexOf(
      'bool hit = ray.minimumDistanceExclusive >= 0.0',
    );
    const fogComparison = traceScene.indexOf(
      'particleDist < segmentLimit',
    );
    expect(missDistance).toBeGreaterThanOrEqual(0);
    expect(traversal).toBeGreaterThan(missDistance);
    expect(fogComparison).toBeGreaterThan(traversal);
    expect(traceScene).not.toContain('particleDist + RAY_OFFSET');
  });

  it('initializes every SurfaceRecord field before publishing the medium vertex', () => {
    const structBody = blockBetween(
      surfaceRecordStruct,
      'struct SurfaceRecord {',
      '};',
    );
    const fieldNames = [...structBody.matchAll(
      /^\s*(?:bool|float|uint|vec3|mat3)\s+([A-Za-z_]\w*)\s*;/gm,
    )].map((match) => match[1]!);
    const initializer = blockBetween(
      renderStructs,
      'void setFogSurfaceRecord(',
      'surf = fogSurface;',
    );
    const initializedFields = new Set(
      [...initializer.matchAll(/\bfogSurface\.([A-Za-z_]\w*)\s*=/g)]
        .map((match) => match[1]!),
    );

    expect(fieldNames.length).toBeGreaterThan(40);
    expect(fieldNames.filter((field) => !initializedFields.has(field))).toEqual([]);
  });

  it('keeps accepted medium collisions on HG transport with deterministic environment ownership', () => {
    expect(renderStructs).toContain('fogSurface.frontFace = true;');
    expect(renderStructs).toContain('fogSurface.sssSigmaT = 0.0;');
    expect(renderStructs).toContain('fogSurface.envMapIntensity = 1.0;');
    expect(RENDER_MAIN).toContain('surf.volumeParticle');
    expect(RENDER_MAIN).toContain('state.envMapIntensity = surf.envMapIntensity;');
    expect(RENDER_MAIN).toContain('if ( surf.volumeParticle ) neeFlags |= 2u;');
    expect(RENDER_MAIN).not.toContain('surf.sssSigmaT > 0.0');
  });
});
