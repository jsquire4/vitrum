/**
 * W4-A4 byte-equivalence snapshot for the composed brute-force PT shader.
 *
 * Background: the WGSL kernel was previously a single 1908-LOC template
 * literal. W4-A4 split it into ~14 per-concern modules composed by
 * `wgsl/pathTraceBruteforce.wgsl.ts`. To guarantee no semantic drift across
 * the split (and any future module edits), this test snapshots the composed
 * string against a frozen golden saved alongside.
 *
 * Updating the golden: any intentional change to the composed WGSL output
 * must be accompanied by an explicit golden update. Reviewers should look at
 * the golden diff to verify the change is intended (and behaviour-preserving,
 * unless the commit explicitly notes a behaviour change).
 *
 * Golden location: `src/__tests__/__snapshots__/pathTraceBruteforce.golden.wgsl`
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const GOLDEN_PATH = join(__dirname, '__snapshots__', 'pathTraceBruteforce.golden.wgsl');

describe('pathTraceBruteforce WGSL composition', () => {
  it('composed WGSL matches the byte-equivalent golden snapshot', () => {
    const golden = readFileSync(GOLDEN_PATH, 'utf-8');
    if (PT_WEBGPU_TRACE_WGSL !== golden) {
      // Report first divergent line index for fast debugging when the diff is small.
      const composedLines = PT_WEBGPU_TRACE_WGSL.split('\n');
      const goldenLines = golden.split('\n');
      const minLen = Math.min(composedLines.length, goldenLines.length);
      let firstDiff = -1;
      for (let i = 0; i < minLen; i += 1) {
        if (composedLines[i] !== goldenLines[i]) {
          firstDiff = i;
          break;
        }
      }
      if (firstDiff === -1 && composedLines.length !== goldenLines.length) {
        firstDiff = minLen;
      }
      const context =
        firstDiff >= 0
          ? `\nFirst divergence at line ${firstDiff + 1}:\n` +
            `  composed: ${JSON.stringify(composedLines[firstDiff])}\n` +
            `  golden:   ${JSON.stringify(goldenLines[firstDiff])}`
          : '';
      throw new Error(
        `PT_WEBGPU_TRACE_WGSL drift detected (composed ${PT_WEBGPU_TRACE_WGSL.length}B vs golden ${golden.length}B).${context}\n` +
          `If this change is intentional, regenerate the golden via:\n` +
          `  npx tsx -e "import('./src/wgsl/pathTraceBruteforce.wgsl.ts').then(m => require('fs').writeFileSync('src/__tests__/__snapshots__/pathTraceBruteforce.golden.wgsl', m.PT_WEBGPU_TRACE_WGSL))"`,
      );
    }
    expect(PT_WEBGPU_TRACE_WGSL).toBe(golden);
  });

  it('composed WGSL preserves load-bearing markers from the pre-split monolith', () => {
    // Sanity checks: these markers MUST be present in any valid composition,
    // and would catch a missing module import in the orchestrator.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('struct FrameParams');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('@compute @workgroup_size(8, 8, 1)');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn main(@builtin(global_invocation_id) gid: vec3u)');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn traceClosest');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn traceAny');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn sampleNextBounceDirection');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn manifoldNeeContribution');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn photonMapContribution');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn evaluateBrdf');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn brdfDirectionalPdf');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn glossyReflectionSample');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn decodeMaterial');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn sampleEnvironmentImportance');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn bsdfAreaLightConnectionContribution');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn bsdfEnvironmentConnectionContribution');
  });

  it('dead-code `sampleMeshAreaLight` is not present in the composed output', () => {
    // Sweep finding F9: `fn sampleMeshAreaLight` was uncalled. W4-A4 removed
    // it during the split — assert it stays gone.
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/\bfn sampleMeshAreaLight\b/);
  });
});
