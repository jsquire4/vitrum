import fs from 'node:fs';
import path from 'node:path';

const target = path.resolve(
	process.cwd(),
	'./src/materials/pathtracing/PhysicalPathTracingMaterial.js'
);

const source = fs.readFileSync( target, 'utf8' );

function requirePattern( pattern, label ) {
	if ( ! pattern.test( source ) ) {
		console.error( `[caustic-regression] Missing expected shader path: ${ label }` );
		process.exit( 1 );
	}
}

requirePattern( /uniform\s+int\s+uCausticStrategy\s*;/, 'uCausticStrategy uniform' );
requirePattern( /uniform\s+float\s+uMneeMaxIterations\s*;/, 'uMneeMaxIterations uniform' );
requirePattern( /uniform\s+float\s+uMneeMaxChainLength\s*;/, 'uMneeMaxChainLength uniform' );
requirePattern( /if\s*\(\s*uCausticStrategy\s*==\s*1\s*\)/, 'manifold strategy branch' );
requirePattern( /if\s*\(\s*uCausticStrategy\s*==\s*2\s*\)/, 'photon strategy branch' );
requirePattern( /const\s+int\s+PHOTON_SAMPLES\s*=\s*8\s*;/, 'deterministic photon sample count' );
requirePattern( /for\s*\(\s*int\s+walkIter\s*=\s*0;\s*walkIter\s*<\s*16;/, 'bounded manifold loop' );

if ( /causticGain\s*=/.test( source ) || /gain\s*=\s*0\.04\s*\+\s*0\.12/.test( source ) ) {
	console.error( '[caustic-regression] Found legacy ad-hoc gain path; expected deterministic strategy implementation.' );
	process.exit( 1 );
}

console.log( '[caustic-regression] pass' );
