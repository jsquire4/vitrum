export function gltfBrowserProofStatusExitCode(verdict) {
  if (verdict === 'PASS') return 0;
  if (verdict === 'HOST-BLOCKED') return 2;
  return 1;
}
