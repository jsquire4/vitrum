/**
 * D6.9 — PPG descent drift gate.
 *
 * The sTree and dTree descent functions in ppgUpdate.wgsl.ts and ppgPdf.wgsl.ts
 * are semantically identical — only the packed-arena load helper names differ
 * between the update and pdf kernels. This test extracts those function bodies,
 * canonicalises those helper names and any comment-only differences, and asserts
 * the normalised bodies are identical.
 *
 * A failure here means one descent was edited without mirroring the change in
 * the other — see the MUST-MATCH comments in both files.
 */

import { describe, expect, it } from 'vitest';
import { buildPpgUpdateWgsl } from '../ppgUpdate.wgsl.js';
import { PPG_PDF_WGSL } from '../ppgPdf.wgsl.js';

/** Extract the body of a named WGSL `fn` (from `fn name(` to its matching `}`). */
function extractFnBody(wgsl: string, fnName: string): string {
  const start = wgsl.indexOf(`fn ${fnName}(`);
  if (start === -1) throw new Error(`Function '${fnName}' not found in WGSL`);
  // Find the opening brace of the function body.
  const braceStart = wgsl.indexOf('{', start);
  if (braceStart === -1) throw new Error(`No opening brace for '${fnName}'`);
  // Walk forward matching braces to find the closing brace.
  let depth = 0;
  let i = braceStart;
  for (; i < wgsl.length; i++) {
    if (wgsl[i] === '{') depth++;
    else if (wgsl[i] === '}') {
      depth--;
      if (depth === 0) break;
    }
  }
  return wgsl.slice(braceStart, i + 1);
}

/**
 * Normalise a WGSL function body for comparison:
 * 1. Strip single-line (//) comments so comment-only differences are ignored.
 * 2. Collapse all whitespace runs to a single space.
 * 3. Replace packed-arena load helper tokens so the read-write update-arena and
 *    read-only query-arena spellings are treated as equivalent.
 */
function normalise(body: string): string {
  return body
    .replace(/\/\/[^\n]*/g, '')          // strip line comments
    .replace(/\s+/g, ' ')                 // collapse whitespace
    .trim()
    // Canonicalize update/query arena helper names.
    .replace(/ppgArenaLoadSTreeF32Update/g, 'ppgArenaLoadSTreeF32')
    .replace(/ppgArenaLoadDTreeF32Update/g, 'ppgArenaLoadDTreeF32')
    // Canonicalize the parameter name used for the octahedral UV input.
    // The update kernel uses `uv`; the pdf kernel uses `octUV`; both
    // are the same semantic variable — normalise to `uv` for comparison.
    .replace(/\boctUV\b/g, 'uv');
}

describe('PPG descent drift gate (D6.9)', () => {
  const updateWgsl = buildPpgUpdateWgsl(341);
  const pdfWgsl = PPG_PDF_WGSL;

  it('sTree descent bodies are identical (modulo arena-helper tokens)', () => {
    const updateBody = extractFnBody(updateWgsl, 'sTreeFindLeafBase');
    const pdfBody    = extractFnBody(pdfWgsl,    'ppgSTreeFindLeafBase');
    expect(normalise(updateBody)).toBe(normalise(pdfBody));
  });

  it('dTree descent bodies are identical (modulo arena-helper tokens)', () => {
    const updateBody = extractFnBody(updateWgsl, 'dTreeFindLeafBase');
    const pdfBody    = extractFnBody(pdfWgsl,    'ppgDTreeFindLeafBase');
    expect(normalise(updateBody)).toBe(normalise(pdfBody));
  });
});
