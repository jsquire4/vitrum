// Per-scene PT sample-target hook — bumps the convergence target when the
// scene contains a shaded fixture (table lamp, pendant, ceiling flush).
//
// PathTracingLayer consumes the hook output as the <Pathtracer samples={N} />
// prop AND publishes it to the __PT__ debug bridge so the e2e timing specs
// auto-track via readPtSnapshot(page).targetSamples.

import { useSelector } from 'react-redux';
import type { RootState } from '@/store';
import { isFixtureLight } from './lightSourceTypes';
import { isShadedFixture } from './fixtureCatalog';
import {
  PT_TARGET_SAMPLES_BASE,
  PT_TARGET_SAMPLES_FIXTURES,
} from '../pathtracerConstants';

export function usePTSampleTarget(): number {
  return useSelector((s: RootState) => {
    for (const id of s.lighting.allIds) {
      const l = s.lighting.byId[id];
      if (l && isFixtureLight(l) && isShadedFixture(l.fixtureType)) {
        return PT_TARGET_SAMPLES_FIXTURES;
      }
    }
    return PT_TARGET_SAMPLES_BASE;
  });
}
