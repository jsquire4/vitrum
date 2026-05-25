import yargs from 'yargs';
import puppeteer from 'puppeteer';
import path from 'path';
import fs from 'fs';
import fetch from 'node-fetch';
import { exec } from 'child_process';

const excludeList = [
	'khronos-MetalRoughSpheres-LDR',
	'khronos-BoxInterleaved',
];

let totalTime = 0;
const argv = yargs( process.argv.slice( 2 ) )
	.usage( 'Usage: $0 <command> [options]' )
	.option( 'output-path', {
		describe: 'Output directory for the files.',
		alias: 'o',
		type: 'string',
		default: './screenshots/golden/',
	} )
	.option( 'scenario', {
		describe: 'The name of one scenario to run.',
		alias: 's',
		type: 'string'
	} )
	.option( 'headless', {
		describe: 'Whether to run in a headless mode.',
		alias: 'h',
		type: 'boolean',
		default: false
	} )
	.option( 'base-url', {
		describe: 'Base URL for the viewer test page.',
		type: 'string',
		default: 'http://localhost:5173',
	} )
	.option( 'start-server', {
		describe: 'Start the local dev server before capturing.',
		type: 'boolean',
		default: false,
	} )
	.option( 'samples', {
		describe: 'Samples per pixel for each screenshot.',
		type: 'number',
		default: 64,
	} )
	.option( 'timeout-ms', {
		describe: 'Render-complete timeout per scenario in milliseconds.',
		type: 'number',
		default: 240000,
	} )
	.argv;

( async () => {

	const req = await fetch( 'https://raw.githubusercontent.com/google/model-viewer/master/packages/render-fidelity-tools/test/config.json' );
	const { scenarios } = await req.json();
	const folderPath = path.resolve( process.cwd(), argv[ 'output-path' ] );
	console.log( `Saving to "${ folderPath }"\n` );

	if ( argv[ 'start-server' ] ) {

		console.log( 'Running test page service' );
		exec( 'npm run start' );

	}

	await waitForViewer( argv[ 'base-url' ] );

	fs.mkdirSync( folderPath, { recursive: true } );

	try {

		if ( argv.scenario ) {

			const scenario = scenarios.find( s => s.name === argv.scenario );
			if ( ! scenario ) {

				console.error( `Scenario "${ argv.scenario }" does not exist.` );
				process.exit( 1 );

			} else {

				await saveScreenshot( scenario, folderPath );

			}

		} else {

			for ( const key in scenarios ) {

				const scenario = scenarios[ key ];
				if ( excludeList.includes( scenario.name ) ) {

					console.log( `Skipping ${ scenario.name }` );

				} else {

					console.log( `Rendering ${ scenario.name }` );
					await saveScreenshot( scenario, folderPath );

				}

			}

		}

		console.log( `\nTotal Time: ${ ( 1e-3 * totalTime ).toFixed( 2 ) }s` );
		process.exit( 0 );

	} catch ( e ) {

		console.error( e );
		process.exit( 1 );

	}

} )();

async function saveScreenshot( scenario, targetFolder ) {

	const name = scenario.name;
	const args = argv.headless
		? [
			'--use-angle=swiftshader',
			'--enable-unsafe-swiftshader',
			'--ignore-gpu-blocklist',
			'--disable-dev-shm-usage',
		]
		: [];
	const browser = await puppeteer.launch( {

		defaultViewport: null,
		args,
		headless: argv.headless,

	} );

	const page = await browser.newPage();
	let fatalViewerError = null;
	page.on( 'console', msg => {

		if ( msg.type() === 'error' ) {

			const text = msg.text();
			console.error( `[viewer console] ${ text }` );
			if (
				/Shader Error|A WebGL context could not be created|Failed to create a WebGL2 context|Program Info Log/i.test( text )
			) {

				fatalViewerError = new Error( `Viewer runtime error: ${ text }` );

			}

		}

	} );
	page.on( 'pageerror', err => {

		console.error( `[viewer pageerror] ${ err.message }` );
		fatalViewerError = err;

	} );

	await page.evaluateOnNewDocument( () => {

		window.__rftRenderComplete = false;
		window.addEventListener( 'render-complete', () => {

			window.__rftRenderComplete = true;

		} );

	} );

	await page.goto(
		`${ argv[ 'base-url' ] }/viewerTest.html?hideUI=true&scale=1&tiles=4&samples=${ argv.samples }#${ name }`,
		{ waitUntil: 'networkidle0' },
	);

	try {
		if ( fatalViewerError ) {

			throw fatalViewerError;

		}

		const startTime = performance.now();
		await page.waitForFunction(
			() => window.__rftRenderComplete === true,
			{ timeout: argv[ 'timeout-ms' ] },
		);

		const deltaTime = performance.now() - startTime;
		console.log( `\tin ${ ( 1e-3 * deltaTime ).toFixed( 2 ) }s` );
		totalTime += deltaTime;
		if ( fatalViewerError ) {

			throw fatalViewerError;

		}

	} catch ( e ) {

		console.error( e.message );
		await browser.close();
		throw e;

	}

	// https://stackoverflow.com/questions/11335460/how-do-i-parse-a-data-url-in-node
	// https://stackoverflow.com/questions/65914988/how-to-save-a-canvas-as-an-image-using-puppeteer
	const dataUrl = await page.evaluate( () => {

		const canvas = document.querySelector( 'canvas' );
		return canvas.toDataURL();

	} );

	const [ info, data ] = dataUrl.split( ',' );
	const [ , ext ] = info.match( /^data:.+\/(.+);base64/ );
	const buffer = Buffer.from( data, 'base64' );
	fs.writeFileSync( `${ targetFolder }/${ name }.${ ext }`, buffer );

	await browser.close();

}

async function waitForViewer( baseUrl ) {

	const targetUrl = `${ baseUrl }/viewerTest.html`;
	const timeoutMs = 30000;
	const pollMs = 500;
	const started = Date.now();
	while ( Date.now() - started < timeoutMs ) {

		try {

			const res = await fetch( targetUrl, { method: 'GET' } );
			if ( res.ok ) {

				return;

			}

		} catch ( _err ) {

			// Retry until timeout.

		}

		await new Promise( resolve => setTimeout( resolve, pollMs ) );

	}

	throw new Error( `Viewer endpoint did not become ready: ${ targetUrl }` );

}
