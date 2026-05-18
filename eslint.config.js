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
      'plan/**',
      'external_requests/**',
      'tools/reference-renders/**',
      // Generated / vendored / non-source assets
      '**/*.wgsl',
      '**/*.glsl',
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
