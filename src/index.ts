/**
 * eslint-plugin-jax-js
 *
 * ESLint plugin for catching jax-js array memory leaks at edit time.
 */

import type { ESLint } from "eslint";

import noUnnecessaryRef from "./rules/no-unnecessary-ref";
import noUseAfterConsume from "./rules/no-use-after-consume";
import requireConsume from "./rules/require-consume";

const plugin: ESLint.Plugin = {
  meta: {
    name: "eslint-plugin-jax-js",
    // Keep in sync with version in package.json
    version: "0.1.0",
  },
  rules: {
    "no-unnecessary-ref": noUnnecessaryRef,
    "no-use-after-consume": noUseAfterConsume,
    "require-consume": requireConsume,
  },
  configs: {},
};

// Self-referential recommended config (ESLint flat config style)
plugin.configs!.recommended = {
  plugins: { "@jax-js": plugin },
  rules: {
    "@jax-js/no-unnecessary-ref": "warn",
    "@jax-js/no-use-after-consume": "warn",
    "@jax-js/require-consume": "warn",
  },
};

export default plugin;
