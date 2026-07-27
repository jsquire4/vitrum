// ESLint flat config for the vitrum workspace.
//
// Scope:
//   - TypeScript everywhere (`packages/*/src`, `packages/*/__tests__`,
//     `examples/*`, `tools/*`).
//   - React + react-hooks rules layered onto packages that ship JSX/TSX
//     (currently `@vitrum/dev` and `@vitrum/engine`).
//   - Prettier compatibility via `eslint-config-prettier` — formatting rules
//     stay in Prettier; ESLint focuses on correctness + type-safety.
//
// Two parser scopes:
//   1. `packages/*/src/**` — type-checked (slow but catches `no-unsafe-*`,
//      `no-floating-promises`, etc.). Uses the workspace tsconfig.json.
//   2. Everything else — non-type-checked (fast, ignores type-aware rules).
//
// This config is a starting bar, not a clean-room. The goal is to expose
// the current violation count so the team can decide what to clean up
// rather than auto-fixing the entire codebase in one commit. Run
// `npx eslint .` to see the bar; run `npx eslint . --fix` to auto-fix the
// safe subset.

import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactPlugin from 'eslint-plugin-react';
import reactHooksPlugin from 'eslint-plugin-react-hooks';
import prettierConfig from 'eslint-config-prettier';
import globals from 'globals';

export default tseslint.config(
  // ── Global ignores ───────────────────────────────────────────────────────
  {
    ignores: [
      '**/node_modules/**',
      '**/dist/**',
      '**/build/**',
      '**/coverage/**',
      '**/*.tsbuildinfo',
      '_staging/**',
      // Disabled generated GitNexus helper; it is not part of the runtime or
      // the active validation workflow (see AGENTS.md).
      '.gitnexus/**',
      'plan/**',
      'external_requests/**',
      'tools/reference-renders/**',
      // Generated / vendored / non-source assets
      '**/*.wgsl',
      '**/*.glsl',
      'packages/gltf-adapter/src/vendor/draco_decoder_browser.js',
    ],
  },

  // ── Base JS recommended (applies to all matched files) ───────────────────
  js.configs.recommended,

  // ── TypeScript recommended-type-checked for package source ───────────────
  ...tseslint.configs.recommendedTypeChecked.map((cfg) => ({
    ...cfg,
    files: ['packages/*/src/**/*.{ts,tsx}'],
  })),

  // ── Type-aware parser settings + globals for package source ──────────────
  {
    files: ['packages/*/src/**/*.{ts,tsx}'],
    languageOptions: {
      parser: tseslint.parser,
      ecmaVersion: 2022,
      sourceType: 'module',
      parserOptions: {
        project: ['./tsconfig.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      globals: {
        ...globals.browser,
        ...globals.worker,
        ...globals.node,
        // WebGPU types come from @webgpu/types — but the runtime globals
        // (GPUBufferUsage, GPUTextureUsage, etc.) are not in eslint-globals.
        GPUBufferUsage: 'readonly',
        GPUTextureUsage: 'readonly',
        GPUShaderStage: 'readonly',
        GPUMapMode: 'readonly',
        GPUColorWrite: 'readonly',
      },
    },
  },

  // ── TypeScript (non-type-checked) for tests, examples, tools, scripts ────
  ...tseslint.configs.recommended.map((cfg) => ({
    ...cfg,
    files: [
      'packages/*/__tests__/**/*.{ts,tsx}',
      'packages/*/vitest.config.{ts,mjs,js}',
      'packages/*/vitest.gpu.config.{ts,mjs,js}',
      'examples/**/*.{ts,tsx}',
      'tools/**/*.{ts,tsx,mjs,js}',
      'scripts/**/*.{ts,mjs,js}',
    ],
  })),

  // ── Non-type-aware parser + globals for tests/examples/tools ─────────────
  {
    files: [
      'packages/*/__tests__/**/*.{ts,tsx}',
      'packages/*/vitest.config.{ts,mjs,js}',
      'packages/*/vitest.gpu.config.{ts,mjs,js}',
      'examples/**/*.{ts,tsx}',
      'tools/**/*.{ts,tsx,mjs,js}',
      'scripts/**/*.{ts,mjs,js}',
    ],
    languageOptions: {
      parser: tseslint.parser,
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        ...globals.browser,
        ...globals.worker,
        ...globals.node,
        ...globals.vitest,
        GPUBufferUsage: 'readonly',
        GPUTextureUsage: 'readonly',
        GPUShaderStage: 'readonly',
        GPUMapMode: 'readonly',
        GPUColorWrite: 'readonly',
      },
    },
  },

  // ── Project-wide custom rules (TS files only) ────────────────────────────
  {
    files: ['**/*.{ts,tsx}'],
    plugins: {
      '@typescript-eslint': tseslint.plugin,
    },
    rules: {
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
          destructuredArrayIgnorePattern: '^_',
          ignoreRestSiblings: true,
        },
      ],
    },
    linterOptions: {
      // Surface stale `// eslint-disable-next-line` comments so they don't
      // accumulate as the code evolves.
      reportUnusedDisableDirectives: 'warn',
    },
  },

  // ── Deno gate scripts: @ts-nocheck is intentional (sloppy-imports mode) ─────
  // These .mjs files are run by Deno with --sloppy-imports; @ts-nocheck is the
  // documented workaround to prevent VS Code / typescript-eslint from trying to
  // type-check the cross-runtime dynamic imports.
  // Also apply the underscore-prefix unused-var convention consistently with .ts files.
  {
    files: ['tools/**/*.mjs', 'scripts/**/*.mjs'],
    rules: {
      '@typescript-eslint/ban-ts-comment': 'off',
      '@typescript-eslint/no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
          destructuredArrayIgnorePattern: '^_',
          ignoreRestSiblings: true,
        },
      ],
      'no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
          destructuredArrayIgnorePattern: '^_',
          ignoreRestSiblings: true,
        },
      ],
    },
  },

  // ── Test files: relax rules that false-positive in Vitest/Jest stubs ────────
  // `unbound-method` fires on `expect(spy.method).toHaveBeenCalled()` patterns
  // where Vitest matchers handle method extraction safely — it is not a real
  // scoping hazard in that context.
  // `require-await` fires on stub/mock implementations that satisfy an async
  // interface but happen not to need await — suppressing inline would produce
  // ~35 disable comments across test files.
  // `no-unsafe-*` / `no-redundant-type-constituents` / `restrict-template-expressions`
  // fire systematically on Vitest mock introspection (`.mock.calls[0][0]`),
  // `Array(n)` patterns, and test-helper types that don't need production-grade
  // type narrowing. Suppressing per-site would create ~100+ noise comments.
  {
    files: [
      'packages/*/__tests__/**/*.{ts,tsx}',
      'packages/*/src/__tests__/**/*.{ts,tsx}',
      'packages/**/__tests__/**/*.{ts,tsx}',
      'packages/**/*.test.{ts,tsx}',
    ],
    rules: {
      '@typescript-eslint/unbound-method': 'off',
      '@typescript-eslint/require-await': 'off',
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/no-unsafe-argument': 'off',
      '@typescript-eslint/no-unsafe-return': 'off',
      '@typescript-eslint/no-unsafe-call': 'off',
      '@typescript-eslint/no-redundant-type-constituents': 'off',
      '@typescript-eslint/restrict-template-expressions': 'off',
    },
  },

  // ── React + react-hooks for packages that ship JSX/TSX ───────────────────
  {
    files: ['packages/dev/**/*.{ts,tsx}', 'packages/engine/**/*.{ts,tsx}'],
    plugins: {
      react: reactPlugin,
      'react-hooks': reactHooksPlugin,
    },
    languageOptions: {
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
    settings: {
      react: { version: 'detect' },
    },
    rules: {
      ...reactPlugin.configs.recommended.rules,
      // React 17+ JSX transform — `import React from 'react'` no longer
      // required at every callsite.
      'react/react-in-jsx-scope': 'off',
      'react/prop-types': 'off', // We use TypeScript types instead.
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
    },
  },

  // ── Prettier — disable rules that conflict with the formatter ────────────
  // MUST come last so it overrides earlier formatting rules.
  prettierConfig,
);
