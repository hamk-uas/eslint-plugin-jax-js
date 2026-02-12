/**
 * @hamk-uas/eslint-plugin-jax-js
 *
 * ESLint plugin for catching jax-js array memory leaks at edit time.
 * Community plugin maintained by HAMK UAS — not an official jax-js package.
 */

import type { ESLint } from "eslint";

import pkg from "../package.json" with { type: "json" };
import noUnnecessaryRef from "./rules/no-unnecessary-ref";
import noUseAfterConsume from "./rules/no-use-after-consume";
import requireConsume from "./rules/require-consume";

const plugin: ESLint.Plugin = {
  meta: {
    name: "@hamk-uas/eslint-plugin-jax-js",
    version: pkg.version,
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
