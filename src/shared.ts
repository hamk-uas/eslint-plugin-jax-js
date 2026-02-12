/**
 * Shared utilities and constant sets for @hamk-uas/eslint-plugin-jax-js rules.
 *
 * Most sets are derived from the API surface file (api-surface.generated.ts).
 * Update that file when jax-js adds new methods or getters.
 *
 * Only two sets require manual curation:
 *   - CONSUMING_TERMINAL_METHODS (methods that return non-Array values)
 *   - UNAMBIGUOUS_ARRAY_METHODS  (methods unique to jax-js, not on standard JS types)
 */

import type * as ESTree from "estree";

import { EXTRACTED_GETTERS, EXTRACTED_METHODS } from "./api-surface.generated";

// ---------------------------------------------------------------------------
// Auto-derived sets (from api-surface.generated.ts)
// ---------------------------------------------------------------------------

/**
 * Methods that consume the array and return a non-Array value.
 * MANUAL — update when adding a new terminal method to the Array class.
 */
export const CONSUMING_TERMINAL_METHODS = new Set<string>([
  "data",
  "dataSync",
  "js",
  "jsAsync",
  "item",
  "dispose",
  // Future-proof: JAX methods likely coming to jax-js
  "tolist",
  "tobytes",
]);

/**
 * Methods that are syntactically methods but do NOT consume the array.
 *
 * `toString`  — returns a string representation without consuming the array.
 *              Unlike most Array methods, it does not decrement the reference
 *              count, matching standard JS `Object.prototype.toString` semantics.
 * `blockUntilReady` — sync barrier that returns `this` unchanged; used for
 *                     timing/benchmarking without affecting ownership.
 */
const NON_CONSUMING_METHODS = new Set<string>(["toString", "blockUntilReady"]);

/**
 * Getters/properties/methods that are NOT consuming.
 * Includes extracted getters plus manually-listed non-consuming methods.
 */
export const NON_CONSUMING_PROPS = new Set<string>([
  ...EXTRACTED_GETTERS,
  ...NON_CONSUMING_METHODS,
]);

/**
 * Methods on Array/Tracer that consume `this` and return a new Array.
 * Auto-computed: all extracted methods minus terminal and non-consuming ones.
 */
export const ARRAY_RETURNING_METHODS = new Set<string>(
  EXTRACTED_METHODS.filter(
    (m) => !CONSUMING_TERMINAL_METHODS.has(m) && !NON_CONSUMING_METHODS.has(m),
  ),
);

/**
 * Factory / constructor names known to return a jax-js Array.
 * MANUAL — these are top-level/namespace functions that create arrays from
 * scratch (not class methods). They span multiple source files and are
 * not worth auto-extracting since the set is small and very stable.
 */
export const ARRAY_FACTORY_NAMES = new Set<string>([
  "array",
  "zeros",
  "ones",
  "full",
  "eye",
  "identity",
  "arange",
  "linspace",
  "logspace",
  "zerosLike",
  "onesLike",
  "fullLike",
  "tri",
  "tril",
  "triu",
  "diag",
  // Future-proof: JAX factory functions likely coming to jax-js
  "asarray",
  "concatenate",
  "stack",
  "vstack",
  "hstack",
  "dstack",
  "tile",
  "repeat",
  "meshgrid",
  "where",
  "empty",
  "emptyLike",
]);

/**
 * Subset of ARRAY_RETURNING_METHODS that don't exist on standard JS types,
 * so seeing them strongly implies jax-js Array usage.
 * MANUAL — requires knowing which names collide with standard JS APIs.
 */
export const UNAMBIGUOUS_ARRAY_METHODS = new Set<string>([
  "neg",
  "astype",
  "transpose",
  "reshape",
  "ravel",
  "argsort",
  "diagonal",
  "greaterEqual",
  "lessEqual",
  "notEqual",
  "flatten",
  // Future-proof: JAX array methods likely coming to jax-js
  "squeeze",
  "expandDims",
  "swapaxes",
  "broadcastTo",
  "conj",
  "clip",
]);

// ---------------------------------------------------------------------------
// Array-init heuristics (shared by require-consume & no-use-after-consume)
// ---------------------------------------------------------------------------

/** Does a callee look like `array(...)`, `np.zeros(...)`, etc.? */
function isArrayFactory(node: ESTree.CallExpression): boolean {
  const c = node.callee;
  if (c.type === "Identifier") return ARRAY_FACTORY_NAMES.has(c.name);
  return (
    c.type === "MemberExpression" &&
    !c.computed &&
    c.property.type === "Identifier" &&
    ARRAY_FACTORY_NAMES.has(c.property.name)
  );
}

/** Does an initializer expression look like it produces a jax-js Array? */
export function isArrayInit(
  init: ESTree.Expression | null | undefined,
): boolean {
  if (!init) return false;
  const unwrapped = unwrapTransparentExpression(init as any) as ESTree.Expression;

  if (unwrapped.type === "CallExpression" && isArrayFactory(unwrapped)) return true;
  if (
    unwrapped.type === "CallExpression" &&
    unwrapped.callee.type === "MemberExpression" &&
    !unwrapped.callee.computed &&
    unwrapped.callee.property.type === "Identifier" &&
    UNAMBIGUOUS_ARRAY_METHODS.has(unwrapped.callee.property.name)
  ) {
    return true;
  }
  if (
    unwrapped.type === "MemberExpression" &&
    !unwrapped.computed &&
    unwrapped.property.type === "Identifier" &&
    unwrapped.property.name === "ref"
  ) {
    return true;
  }
  if (unwrapped.type === "AwaitExpression") return isArrayInit(unwrapped.argument as any);
  return false;
}

/**
 * Unwrap expression wrappers that are transparent to the value.
 *
 * This keeps the plugin effective in TypeScript projects where array inits are
 * often wrapped (casts, non-null assertions, parentheses).
 */
function unwrapTransparentExpression(node: any): any {
  let current = node;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    if (!current || typeof current.type !== "string") return current;
    if (
      current.type === "ParenthesizedExpression" ||
      current.type === "TSAsExpression" ||
      current.type === "TSNonNullExpression" ||
      current.type === "TSSatisfiesExpression" ||
      current.type === "TSTypeAssertion"
    ) {
      current = current.expression;
      continue;
    }
    return current;
  }
}

// ---------------------------------------------------------------------------
// Scope & AST utilities
// ---------------------------------------------------------------------------

/**
 * Walk up scope chains to find a variable by name.
 */
export function findVariable(scope: any, name: string): any | null {
  let current = scope;
  while (current) {
    const found = current.variables?.find(
      (v: { name: string }) => v.name === name,
    );
    if (found) return found;
    current = current.upper;
  }
  return null;
}

/**
 * Get the parent node (handles missing `.parent` gracefully).
 */
export function parentOf(node: ESTree.Node): ESTree.Node | undefined {
  return (node as any).parent as ESTree.Node | undefined;
}

/**
 * Check if a reference only accesses a non-consuming property
 * (`.shape`, `.dtype`, etc.) — meaning it does NOT consume the array.
 */
export function isNonConsumingReference(ref: {
  identifier: ESTree.Identifier;
}): boolean {
  const parent = parentOf(ref.identifier);
  if (!parent) return false;

  if (
    parent.type === "MemberExpression" &&
    parent.object === ref.identifier &&
    !parent.computed &&
    parent.property.type === "Identifier" &&
    NON_CONSUMING_PROPS.has(parent.property.name)
  ) {
    return true;
  }

  return false;
}

/**
 * Check if a variable is a "borrowed" binding whose `.ref` usage is
 * intentional (callback parameters and for-of/for-in loop variables).
 *
 * - Callback params: `.map(t => t.ref)` — cloning borrowed element
 * - For-of vars: `for (const t of arr) t.ref;` — rc-bumping borrowed ref
 */
export function isBorrowedBinding(variable: any): boolean {
  for (const def of variable.defs ?? []) {
    // Callback parameter: arrow/function expression passed as an argument
    if (def.type === "Parameter") {
      const funcNode = def.node;
      if (
        funcNode?.type === "ArrowFunctionExpression" ||
        funcNode?.type === "FunctionExpression"
      ) {
        const parent = parentOf(funcNode);
        if (
          parent?.type === "CallExpression" &&
          (parent as ESTree.CallExpression).arguments.includes(funcNode as any)
        ) {
          return true;
        }
      }
    }

    // For-of / for-in loop variable
    if (def.type === "Variable") {
      const grandParent = def.parent ? parentOf(def.parent) : undefined;
      if (
        grandParent?.type === "ForOfStatement" ||
        grandParent?.type === "ForInStatement"
      ) {
        return true;
      }
    }
  }
  return false;
}
