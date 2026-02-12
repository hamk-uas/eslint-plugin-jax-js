// Extracted from jax-js v0.1.9 — edit manually when jax-js adds new methods.
// See https://github.com/ekzhang/jax-js for the upstream source.
//
// The derived constant sets (NON_CONSUMING_PROPS, ARRAY_RETURNING_METHODS,
// CONSUMING_TERMINAL_METHODS) are computed in shared.ts from these lists.

/** Public getters on Tracer and Array classes (non-consuming accesses). */
export const EXTRACTED_GETTERS = [
  "aval",
  "device",
  "dtype",
  "ndim",
  "ref",
  "refCount",
  "shape",
  "size",
  "weakType",
] as const;

/** Public methods on Tracer and Array classes (consuming operations). */
export const EXTRACTED_METHODS = [
  "add",
  "all",
  "any",
  "argsort",
  "astype",
  "blockUntilReady",
  "data",
  "dataSync",
  "diagonal",
  "dispose",
  "div",
  "equal",
  "flatten",
  "greater",
  "greaterEqual",
  "item",
  "js",
  "jsAsync",
  "less",
  "lessEqual",
  "max",
  "mean",
  "min",
  "mod",
  "mul",
  "neg",
  "notEqual",
  "prod",
  "ravel",
  "reshape",
  "slice",
  "sort",
  "sub",
  "sum",
  "toString",
  "transpose",
] as const;
