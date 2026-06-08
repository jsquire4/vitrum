import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

function read(relPath) {
	return readFileSync(resolve(process.cwd(), relPath), 'utf8');
}

function expectMatch(text, pattern, message) {
	if (!pattern.test(text)) {
		throw new Error(message);
	}
}

function expectNoMatch(text, pattern, message) {
	if (pattern.test(text)) {
		throw new Error(message);
	}
}

const renderStructs = read('./src/materials/pathtracing/glsl/render_structs.glsl.js');
const directLight = read('./src/materials/pathtracing/glsl/direct_light_contribution_function.glsl.js');
const bsdf = read('./src/shader/bsdf/bsdf_functions.glsl.js');
const spectral = read('./src/shader/bsdf/spectral_accumulator.glsl.js');
const util = read('./src/shader/common/util_functions.glsl.js');
const materialMain = read('./src/materials/pathtracing/PhysicalPathTracingMaterial.js');
const fogFunctions = read('./src/shader/bsdf/fog_functions.glsl.js');
const volumeMarch = read('./src/shader/bsdf/volume_march.glsl.js');
const lightSampling = read('./src/shader/sampling/light_sampling_functions.glsl.js');
const attenuateHit = read('./src/materials/pathtracing/glsl/attenuate_hit_function.glsl.js');
const bdptConnection = read('./src/materials/pathtracing/glsl/bdpt_connection.glsl.js');
const bdptLightSubpath = read('./src/materials/pathtracing/glsl/bdpt_light_subpath.glsl.js');
const webglPathTracer = read('./src/core/WebGLPathTracer.js');
const surfaceRecordStruct = read('./src/shader/structs/surface_record_struct.glsl.js');
const lightsStruct = read('./src/shader/structs/lights_struct.glsl.js');
const pathTracingSceneGenerator = read('./src/core/PathTracingSceneGenerator.js');
const uvUnwrapper = read('./src/utils/UVUnwrapper.js');
const quiltPathTracingRenderer = read('./src/core/QuiltPathTracingRenderer.js');

expectMatch(renderStructs, /float wavelength;/, 'RenderState missing wavelength field');
expectMatch(renderStructs, /float wavelengthPdf;/, 'RenderState missing wavelengthPdf field');
expectMatch(renderStructs, /vec3 throughput;/, 'RenderState missing RGB throughput field');
expectNoMatch(renderStructs, /throughputColor/, 'RenderState still contains legacy throughputColor');

expectMatch(
	directLight,
	/wavelengthToRGB\s*\(\s*state\.wavelength\s*,\s*state\.throughput\s*,\s*state\.wavelengthPdf\s*\)/,
	'direct_light_contribution is not using hero-wavelength throughput conversion',
);
expectMatch(
	directLight,
	/bsdfResult\s*\(\s*worldWo,\s*lightRec\.direction,\s*surf,\s*state\.wavelength,/,
	'direct light path must pass state.wavelength into bsdfResult',
);
expectMatch(
	directLight,
	/bsdfResult\s*\(\s*worldWo,\s*envDirection,\s*surf,\s*state\.wavelength,/,
	'environment light path must pass state.wavelength into bsdfResult',
);
expectNoMatch(directLight, /throughputColor/, 'direct_light_contribution still references throughputColor');
expectMatch(materialMain, /uniform float uRadianceClamp;/, 'PhysicalPathTracingMaterial missing radiance clamp uniform');
// Sprint 5: output color names have moved across fork eras:
// gl_FragColor (legacy), gColor (MRT location 0), pc_fragColor (current three.js GLSL3 output).
expectMatch(
	materialMain,
	/sampleLuminance\s*=\s*dot\s*\(\s*(?:gl_FragColor|gColor|pc_fragColor)\.rgb,\s*vec3\s*\(\s*0\.2126,\s*0\.7152,\s*0\.0722\s*\)\s*\)/,
	'PhysicalPathTracingMaterial must clamp final sample luminance for firefly control',
);
expectMatch(spectral, /uniform int uSpectralRendering;/, 'spectral accumulator missing preview-mode gate uniform');
expectMatch(
	spectral,
	/if\s*\(\s*uSpectralRendering\s*==\s*0\s*\)\s*return throughput/,
	'spectral accumulator must keep low-SPP preview path RGB-stable by default',
);
expectMatch(
	bsdf,
	/if\s*\(\s*uSpectralRendering\s*==\s*0\s*\)\s*return max\s*\(\s*rgb,\s*vec3\s*\(\s*0\.0\s*\)\s*\)/,
	'pathThroughputFromRgb must keep RGB throughput in the default preview path',
);

expectMatch(
	bsdf,
	/ScatterRecord bsdfSample\(\s*vec3 worldWo,\s*SurfaceRecord surf,\s*float heroWavelength\s*\)/,
	'bsdfSample signature missing hero wavelength parameter',
);
expectMatch(
	bsdf,
	/ScatterRecord sssSample\(\s*vec3 worldWo,\s*SurfaceRecord surf,\s*float heroWavelength\s*\)/,
	'sssSample signature missing hero wavelength parameter',
);
expectNoMatch(bsdf, /TODO\(sprint-7-flags\)/, 'Stale sprint-7 flags TODO still present in bsdf_functions');
const evalSpectrumIndex = bsdf.indexOf('float evalSpectrum( vec3 coeffs, float lambda )');
const evalSpectrumAtHeroIndex = bsdf.indexOf('float evalSpectrumAtHero( float lambdaNm )');
if (evalSpectrumIndex === -1 || evalSpectrumAtHeroIndex === -1 || evalSpectrumIndex > evalSpectrumAtHeroIndex) {
	throw new Error('evalSpectrum must be declared before evalSpectrumAtHero for GLSL compile order');
}
expectMatch(
	bsdf,
	/float bsdfResult\s*\(\s*vec3 worldWo,\s*vec3 worldWi,\s*SurfaceRecord surf,\s*float heroWavelength,\s*inout vec3 color\s*\)/,
	'bsdfResult must thread hero wavelength for NEE consistency',
);
expectMatch(
	bsdf,
	/return ggxPDF\s*\(\s*wo,\s*wh,\s*filteredRoughness\s*\)\s*\/\s*denom/,
	'transmissionEval must use GGX BTDF Jacobian PDF (Walter et al.)',
);
expectMatch(
	bsdf,
	/vec3 transmissionDirection[\s\S]*?ggxDirection\s*\(\s*wo,\s*vec2\s*\(\s*filteredRoughness\s*\),\s*rand2\s*\(\s*13\s*\)\s*\)/,
	'transmissionDirection must sample GGX half-vectors to match transmissionEval PDF',
);
expectMatch(
	bsdf,
	/vec3 dispersionTransmissionDirection[\s\S]*?ggxDirection\s*\(\s*wo,\s*vec2\s*\(\s*filteredRoughness\s*\),\s*rand2\s*\(\s*13\s*\)\s*\)/,
	'dispersionTransmissionDirection must sample GGX half-vectors to match transmissionEval PDF',
);
expectNoMatch(bsdf, /incorrect PDF/, 'transmissionEval must not retain incorrect-PDF TODO');
expectMatch(util, /float heroWeightFromRgb\(/, 'heroWeightFromRgb helper missing');

// ── Jakob & Hanika 2019 RGB→spectrum upsampling is actually consumed ─────────
// The host upsamples a representative medium albedo into u_jakobCoeffs; without
// these assertions the coefficients would die at the uniform (their previous
// state). The integrator must (a) define a gated mediumAlbedoHero helper that
// routes through evalSpectrumAtHero only when upsampling is active, and (b) call
// it at the volume-scatter and SSS single-scatter albedo sites.
expectMatch(
	bsdf,
	/float mediumAlbedoHero\(\s*vec3 rgb,\s*float heroWavelength\s*\)\s*\{[\s\S]*?if\s*\(\s*spectralUpsamplingActive\(\)\s*\)\s*\{[\s\S]*?return evalSpectrumAtHero\(\s*heroWavelength\s*\)/,
	'mediumAlbedoHero must route the hero albedo through evalSpectrumAtHero under the spectral gate',
);
expectMatch(
	bsdf,
	/float mediumAlbedoHero\(\s*vec3 rgb,\s*float heroWavelength\s*\)\s*\{[\s\S]*?return heroScalarFromRgb\(\s*rgb,\s*heroWavelength\s*\)/,
	'mediumAlbedoHero must fall back to the legacy heroScalarFromRgb projection when the gate is off',
);
// The gate must require BOTH spectral rendering on AND a non-flat coefficient set.
// The flat (0,0,0) default evaluates to S ≡ ½ and would wash colour, so it must
// NOT route through evalSpectrum — the default RGB path stays bit-identical.
expectMatch(
	bsdf,
	/bool spectralUpsamplingActive\(\)\s*\{\s*return uSpectralRendering == 1 &&\s*\(\s*u_jakobCoeffs\.x != 0\.0 \|\| u_jakobCoeffs\.y != 0\.0 \|\| u_jakobCoeffs\.z != 0\.0\s*\)/,
	'spectralUpsamplingActive must gate on spectral rendering AND a non-default u_jakobCoeffs so the flat default never routes through evalSpectrum',
);
// evalSpectrumAtHero must be declared before mediumAlbedoHero (GLSL compile order),
// and mediumAlbedoHero before its call sites further down bsdf_functions.
const evalSpectrumAtHeroIdx = bsdf.indexOf('float evalSpectrumAtHero( float lambdaNm )');
const mediumAlbedoHeroIdx = bsdf.indexOf('float mediumAlbedoHero( vec3 rgb, float heroWavelength )');
if (evalSpectrumAtHeroIdx === -1 || mediumAlbedoHeroIdx === -1 || evalSpectrumAtHeroIdx > mediumAlbedoHeroIdx) {
	throw new Error('mediumAlbedoHero must be declared after evalSpectrumAtHero for GLSL compile order');
}
expectMatch(
	bsdf,
	/sssRec\.throughput = mediumAlbedoThroughput\(\s*surf\.sssAlbedo,\s*heroWavelength\s*\)\s*\*\s*beerLambert/,
	'SSS single-scatter albedo must flow through mediumAlbedoHero (gated Jakob–Hanika reflectance)',
);
expectMatch(
	materialMain,
	/state\.throughput \*= mediumAlbedoThroughput\(\s*u_scatterAlbedo,\s*state\.wavelength\s*\)\s*\*\s*transmittance/,
	'volume single-scatter albedo must flow through mediumAlbedoHero (gated Jakob–Hanika reflectance)',
);
// The legacy preview-stable RGB projection must still exist as the default branch.
expectMatch(util, /dot\(\s*rgb,\s*vec3\(\s*tR,\s*tG,\s*tB\s*\)\s*\)/, 'heroScalarFromRgb smoothstep projection must remain as the default-path fallback');
expectMatch(
	util,
	/readSpectralAttenuationMu\s*\(\s*sampler2D materialsTex/,
	'util_functions must sample packed spectral μ from materials texture',
);
expectMatch(
	util,
	/exp\s*\(\s*-\s*muLambda\s*\*\s*dist\s*\)/,
	'transmissionAttenuationHero must apply Beer-Lambert with interpolated spectral μ',
);
expectMatch(
	util,
	/transmissionAttenuationHero\s*\(\s*sampler2D materialsTex/,
	'transmissionAttenuationHero must accept materials sampler for spectral lookup',
);
expectNoMatch(
	lightSampling,
	/denominator is potentially zero/,
	'light sampling must not retain zero-denominator TODOs in area-light PDF code',
);
expectMatch(
	lightSampling,
	/lightRec\.pdf\s*=\s*max\s*\(/,
	'area-light sampling must clamp PDFs to finite positive values',
);

expectMatch(
	attenuateHit,
	/#if\s+FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION[\s\S]*if\s*\(\s*isShadowRay\s*&&\s*material\.normalMap\s*!=\s*-\s*1\s*\)/,
	'stained-glass shadow normal-map perturbation must be behind an explicit opt-in feature gate',
);
expectNoMatch(
	attenuateHit,
	/\n\s*if\s*\(\s*material\.normalMap\s*!=\s*-\s*1\s*\)\s*\{/,
	'normal-map shadow-ray perturbation must not run from the default unguarded path',
);
expectNoMatch(
	attenuateHit,
	/#define\s+FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION\s+1\b/,
	'FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION must not be enabled by default in the shader chunk',
);
expectMatch(
	materialMain,
	/FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION:\s*0/,
	'PhysicalPathTracingMaterial must define shadow normal perturbation as explicit opt-in default 0',
);

expectMatch(
	materialMain,
	/bsdfSample\s*\(\s*-\s*ray\.direction,\s*surf,\s*state\.wavelength\s*\)/,
	'PhysicalPathTracingMaterial must call bsdfSample with hero wavelength',
);
expectMatch(
	materialMain,
	/sssSample\s*\(\s*-\s*ray\.direction,\s*surf,\s*state\.wavelength\s*\)/,
	'PhysicalPathTracingMaterial must call sssSample with hero wavelength',
);

// ── BDPT full Veach §10.3 connection MIS (GPU port of bdptConnectionMIS_full) ──
// The 2-strategy approximation (bdptMISWeight2) must be GONE, replaced by the
// full multi-strategy recurrence: a ConvertDensity (PBRT destination-cosine
// half-G) area-measure ratio sweep over the merged path, β=2 power heuristic.
expectNoMatch(
	bdptConnection,
	/bdptMISWeight2/,
	'bdpt_connection must drop the 2-strategy bdptMISWeight2 for the full §10.3 sweep',
);
expectMatch(
	bdptConnection,
	/float bdptMISWeightFull\(/,
	'bdpt_connection must define the full Veach §10.3 multi-strategy MIS weight',
);
expectMatch(
	bdptConnection,
	/float bdptConvertDensitySAtoArea\(\s*float pdfSA,\s*vec3 fromPos,\s*vec3 destPos,\s*vec3 destNorm\s*\)/,
	'bdpt_connection must use PBRT Vertex::ConvertDensity (destination-cosine half-G)',
);
// The reverse density at the eye vertex must be the PBRT-correct (non-symmetric)
// bsdfResult with wo/wi as the connection geometry dictates — NOT a symmetric
// cos/π. revLc swaps (eeToPrev, connDir); fwdEeMinus swaps (connDir, eeToPrev).
expectMatch(
	bdptConnection,
	/float revLc = bsdfResult\(\s*eeToPrev,\s*connDir,/,
	'eye reverse density (revLc) must re-invoke bsdfResult with wo→E_{e-1}, wi=connDir (D1)',
);
expectMatch(
	bdptConnection,
	/fwdEeMinus = bsdfResult\(\s*connDir,\s*eeToPrev,/,
	'eye fwdEeMinus override must re-invoke bsdfResult with wo=connDir, wi→E_{e-1} (D1)',
);
// The MIS denominator must be the β=2 power heuristic (sum of squared pdfs).
expectMatch(
	bdptConnection,
	/return \(\s*ps \* ps\s*\)\s*\/\s*denom/,
	'bdpt_connection MIS weight must be the β=2 power heuristic ps²/Σpᵢ²',
);
// The light subpath must store SOLID-ANGLE pdfs — the baked-in geometry term is
// gone (ConvertDensity applies the Jacobian at connection time instead).
expectNoMatch(
	bdptLightSubpath,
	/pdfFwd = pdfScatter \* max\( gTerm/,
	'light subpath must NOT bake the geometry term into pdfFwd (store SA pdf)',
);
expectNoMatch(
	bdptLightSubpath,
	/pdfRev = pdfRevScatter \* max\( gTerm/,
	'light subpath must NOT bake the geometry term into pdfRev (store SA pdf)',
);
expectMatch(
	bdptLightSubpath,
	/float pdfFwd = pdfScatter;/,
	'light subpath pdfFwd must be the bare solid-angle scatter pdf (no G)',
);
// The eye loop must thread the eye-subpath scratch stack into the connection.
expectMatch(
	materialMain,
	/float bdptPrevScatterPdf = 1\.0;/,
	'eye loop must track the real forward scatter pdf (replaces hardcoded eyePdfFwd=1.0)',
);
expectMatch(
	materialMain,
	/bdptEyePos, bdptEyeNrm, bdptEyePdfFwd, bdptEyePdfRev, bdptEyeSpec/,
	'evaluateBdptConnection must receive the threaded eye-subpath scratch stack',
);
expectNoMatch(
	materialMain,
	/scatterRec\.pdf,\s*\/\/ eyePdfFwd/,
	'eye loop must not pass the old scalar eyePdfFwd=scatterRec.pdf hack to the connection',
);
expectMatch(
	bdptConnection,
	/Intentional biased firefly clamp[\s\S]*rather than an unbiased BDPT reference[\s\S]*#define BDPT_CONTRIBUTION_CLAMP 100\.0/,
	'BDPT_CONTRIBUTION_CLAMP must be documented as intentional biased firefly control, not unbiased BDPT',
);
expectMatch(
	bdptConnection,
	/return clamp\(\s*contribution,\s*vec3\(\s*0\.0\s*\),\s*vec3\(\s*BDPT_CONTRIBUTION_CLAMP\s*\)\s*\)/,
	'BDPT connection contribution must remain clamped through BDPT_CONTRIBUTION_CLAMP',
);

// Lifecycle hygiene: generated cube-to-equirect helpers and secondary render
// targets are easy to leak because they are off the main PathTracingRenderer.
expectMatch(
	webglPathTracer,
	/_lowResPathTracer\.dispose\(\)/,
	'WebGLPathTracer.dispose must dispose the dynamic low-res PathTracingRenderer',
);
expectMatch(
	webglPathTracer,
	/_colorBackground\?\.\s*dispose\(\)/,
	'WebGLPathTracer.dispose must dispose the cached color background texture',
);
expectMatch(
	webglPathTracer,
	/generator\.dispose\(\);[\s\S]*?this\._internalBackground = background/,
	'cube-background update must dispose the CubeToEquirectGenerator after conversion',
);
expectMatch(
	webglPathTracer,
	/material\.envMapInfo\.updateFrom\(\s*environment\s*\);[\s\S]*?environment\.dispose\(\);[\s\S]*?generator\.dispose\(\);/,
	'cube-environment update must dispose the generated equirect texture and generator after envMapInfo upload',
);

// Low-traffic fork utilities should stay constructible and compatible with
// modern THREE even when app code rarely exercises them.
expectNoMatch(
	pathTracingSceneGenerator,
	/materialUuids\.length !== length/,
	'PathTracingSceneGenerator must compare material UUID cache against materials.length, not undefined length',
);
expectNoMatch(
	uvUnwrapper,
	/AddMeshStatus/,
	'UVUnwrapper must not reference the undefined AddMeshStatus enum',
);
expectMatch(
	uvUnwrapper,
	/statusCode !== 0/,
	'UVUnwrapper must treat xatlas status 0 as success',
);
expectNoMatch(
	uvUnwrapper,
	/\.addAttribute\(/,
	'UVUnwrapper must use BufferGeometry.setAttribute with modern THREE',
);
expectNoMatch(
	uvUnwrapper,
	/mesh\.geometry = newGeometry/,
	'UVUnwrapper.generate must return the unwrapped geometry instead of assigning to an undefined mesh',
);
expectMatch(
	uvUnwrapper,
	/return newGeometry;/,
	'UVUnwrapper.generate must return the unwrapped BufferGeometry',
);
expectMatch(
	quiltPathTracingRenderer,
	/set samples\(\s*v\s*\)/,
	'QuiltPathTracingRenderer must provide a samples setter for PathTracingRenderer construction/reset',
);
expectNoMatch(fogFunctions, /sampleFogVolume/, 'dead sampleFogVolume helper must stay deleted');
expectNoMatch(volumeMarch, /equiAngularPdf/, 'dead equiAngularPdf helper must stay deleted');
expectNoMatch(spectral, /float sampleHeroWavelength\(/, 'legacy Y-only GLSL sampleHeroWavelength helper must stay deleted');
expectNoMatch(materialMain, /uniform float u_ior0|uniform float u_dispersionStrength|u_ior0:\s*\{|u_dispersionStrength:\s*\{/, 'dead Sprint-8 dispersion uniforms must stay deleted');
expectNoMatch(surfaceRecordStruct, /activeLayerRoughness/, 'write-only SurfaceRecord.activeLayerRoughness must stay deleted');
expectNoMatch(lightsStruct, /float near;/, 'write-only Light.near field must stay deleted');

// ── GLSL call-closure gate (2026-06-06) ─────────────────────────────────────
// The regex pins above are string checks — they CANNOT catch a call to a
// function that does not exist (the `activeLayerWeight` vs
// `activeLayerThroughput` dangling call shipped green through this script and
// broke the whole fork fragment shader at GL compile time). This section
// extracts every function CALL from the *.glsl.js chunk library and asserts
// each callee is either a GLSL builtin, a struct constructor, or a function
// DEFINED somewhere in the fork's embedded GLSL (chunks + the material JS
// template literals). It is the GLSL analogue of the WGSL naga-compile gate.
{
	const { readdirSync, statSync } = await import('node:fs');
	const { join } = await import('node:path');

	const walk = (dir, out = []) => {
		for (const name of readdirSync(dir)) {
			const p = join(dir, name);
			if (statSync(p).isDirectory()) walk(p, out);
			else if (name.endsWith('.js')) out.push(p);
		}
		return out;
	};

	const srcRoot = resolve(process.cwd(), './src');
	const allJs = walk(srcRoot);
	const chunkFiles = allJs.filter((p) => p.endsWith('.glsl.js'));

	// Pull template-literal bodies (the embedded GLSL) out of a JS file.
	// `${...}` interpolations are replaced with the placeholder type `float`
	// so JS-parameterized definitions (e.g. sobol.glsl.js `${ type } sobolReverseBits(`)
	// remain visible as definitions while JS call noise inside interpolations
	// disappears. Comments are stripped; `#define` macro names are collected as
	// definitions BEFORE preprocessor lines are stripped.
	const collectDefines = (glsl, into) => {
		for (const m of glsl.matchAll(/^\s*#define\s+([a-zA-Z_]\w*)/gm)) into.add(m[1]);
	};
	const glslOf = (file, definesInto) => {
		const text = readFileSync(file, 'utf8');
		const literals = text.match(/`[^`]*`/gs) ?? [];
		let glsl = literals.join('\n');
		for (let i = 0; i < 4 && /\$\{[^{}]*\}/.test(glsl); i += 1) {
			glsl = glsl.replace(/\$\{[^{}]*\}/g, 'float');
		}
		glsl = glsl
			.replace(/\/\*[\s\S]*?\*\//g, ' ')
			.replace(/\/\/[^\n]*/g, ' ');
		if (definesInto) collectDefines(glsl, definesInto);
		return glsl.replace(/^\s*#[^\n]*/gm, ' ');
	};

	// DEFINITIONS universe: every embedded GLSL template literal in src/**.js
	// (chunks may call functions defined by the including material).
	const defined = new Set();
	let defsGlsl = '';
	for (const f of allJs) defsGlsl += glslOf(f, defined) + '\n';
	for (const m of defsGlsl.matchAll(
		/(?:^|[;{}\s])(?:float|u?int|bool|void|[iub]?vec[234]|mat[234](?:x[234])?|[A-Z]\w*)\s+([a-zA-Z_]\w*)\s*\(/g,
	)) {
		defined.add(m[1]);
	}
	for (const m of defsGlsl.matchAll(/struct\s+([A-Z]\w*)/g)) defined.add(m[1]);

	// Symbols provided by EXTERNAL GLSL composed into the final shader at
	// runtime — not defined in this repo, verified by provenance:
	//   - THREE ShaderChunk `#include <common>` (PhysicalPathTracingMaterial.js:231):
	//     saturate, pow2, pow3, pow4, luminance, texture2D (compat alias).
	//   - three-mesh-bvh `BVHShaderGLSL.*` (PhysicalPathTracingMaterial.js:244-245):
	//     uTexelFetch1D, bvhIntersectFirstHit.
	for (const ext of [
		'saturate', 'pow2', 'pow3', 'pow4', 'luminance', 'texture2D',
		'uTexelFetch1D', 'bvhIntersectFirstHit',
	]) {
		defined.add(ext);
	}

	const GLSL_BUILTINS = new Set((
		'abs acos acosh all any asin asinh atan atanh bitCount bitfieldExtract bitfieldInsert bitfieldReverse ' +
		'ceil clamp cos cosh cross degrees determinant dFdx dFdy distance dot equal exp exp2 faceforward ' +
		'findLSB findMSB floatBitsToInt floatBitsToUint floor fma fract frexp fwidth greaterThan ' +
		'greaterThanEqual intBitsToFloat inverse inversesqrt isinf isnan ldexp length lessThan lessThanEqual ' +
		'log log2 matrixCompMult max min mix mod modf normalize not notEqual outerProduct packHalf2x16 ' +
		'packSnorm2x16 packUnorm2x16 pow radians reflect refract round roundEven sign sin sinh smoothstep ' +
		'sqrt step tan tanh texelFetch texture textureGrad textureLod textureSize transpose trunc ' +
		'uintBitsToFloat unpackHalf2x16 unpackSnorm2x16 unpackUnorm2x16 ' +
		'float int uint bool vec2 vec3 vec4 ivec2 ivec3 ivec4 uvec2 uvec3 uvec4 bvec2 bvec3 bvec4 ' +
		'mat2 mat3 mat4 mat2x2 mat2x3 mat2x4 mat3x2 mat3x3 mat3x4 mat4x2 mat4x3 mat4x4 ' +
		'if for while switch return defined layout'
	).split(/\s+/));

	const undefinedCalls = new Map();
	for (const f of chunkFiles) {
		const body = glslOf(f);
		for (const m of body.matchAll(/([a-zA-Z_]\w*)\s*\(/g)) {
			const callee = m[1];
			if (GLSL_BUILTINS.has(callee) || defined.has(callee)) continue;
			const rel = f.slice(srcRoot.length + 1);
			if (!undefinedCalls.has(callee)) undefinedCalls.set(callee, rel);
		}
	}
	if (undefinedCalls.size > 0) {
		const lines = [...undefinedCalls.entries()].map(([fn, file]) => `  ${fn}() — first seen in src/${file}`);
		throw new Error(
			`GLSL call-closure FAILED — ${undefinedCalls.size} call(s) to functions defined nowhere in the fork GLSL:\n` +
			lines.join('\n'),
		);
	}
	console.log(`GLSL call-closure: OK (${defined.size} definitions cover all chunk calls).`);
}

console.log('Shader smoke checks passed.');
