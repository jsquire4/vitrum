/**
 * PR-5.1 — DDGI probe rays share ReSTIR `SceneBVHBuffers` (no separate SceneBvh build).
 */

export {
  makeRestirBvhSnapshot as makeDdgiRestirBvhSnapshot,
  type RestirBvhSnapshot as DdgiRestirBvhSnapshot,
} from '../restir/restirBvhSnapshot.js';
