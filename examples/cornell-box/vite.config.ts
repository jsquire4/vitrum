import { defineConfig } from 'vite';
import path from 'node:path';
import fs from 'node:fs';

const repoRoot = path.resolve(__dirname, '../..');
// Capture sink directory — accepts POST /__capture?name=<scenario> uploads
// from the Playwright capture adapter. Earlier revisions hard-coded a
// per-sweep date ("post-sweep-20260512") that aged out without anyone
// rotating it; settled on a stable name so reference renders accumulate
// in one place.
const captureDir = path.resolve(repoRoot, 'tools/reference-renders/cornell-box-captures');
const MAX_CAPTURE_BYTES = 64 * 1024 * 1024;

export default defineConfig({
  resolve: {
    dedupe: ['three', 'three-mesh-bvh', 'three-gpu-pathtracer'],
    alias: {
      '@vitrum/core': path.resolve(repoRoot, 'packages/core/src/index.ts'),
      '@vitrum/pt-webgl': path.resolve(repoRoot, 'packages/pt-webgl/src/index.ts'),
      '@vitrum/shared-denoisers': path.resolve(repoRoot, 'packages/shared-denoisers/src/index.ts'),
      '@vitrum/three-bindings': path.resolve(repoRoot, 'packages/three-bindings/src/index.ts'),
    },
  },
  server: {
    port: 5174,
  },
  plugins: [
    {
      name: 'vitrum-capture-sink',
      configureServer(server) {
        fs.mkdirSync(captureDir, { recursive: true });
        server.middlewares.use('/__capture', (req, res) => {
          if (req.method !== 'POST') {
            res.statusCode = 405;
            res.end('POST only');
            return;
          }
          const url = new URL(req.url ?? '/', 'http://x');
          const name = (url.searchParams.get('name') ?? 'capture').replace(/[^a-zA-Z0-9._-]/g, '_');
          const filePath = path.join(captureDir, `${name}.png`);
          const chunks: Buffer[] = [];
          let total = 0;
          req.on('data', (c) => {
            total += c.length;
            if (total > MAX_CAPTURE_BYTES) {
              chunks.length = 0;
              res.statusCode = 413;
              res.end(`capture payload too large (>${MAX_CAPTURE_BYTES} bytes)`);
              req.destroy();
              return;
            }
            chunks.push(c);
          });
          req.on('end', () => {
            const buf = Buffer.concat(chunks);
            fs.writeFileSync(filePath, buf);
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ ok: true, bytes: buf.length, path: filePath }));
          });
          req.on('error', (err) => {
            res.statusCode = 500;
            res.end(String(err));
          });
        });
      },
    },
  ],
});
