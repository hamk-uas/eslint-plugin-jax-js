/**
 * @jax-js/no-use-after-consume
 *
 * Warns when a variable holding a jax-js Array is used after being consumed.
 *
 * In jax-js's consuming ownership model, most operations dispose their
 * input arrays. Using an array after it has been consumed is a
 * use-after-free bug.
 *
 * Consumption sources (in order of specificity):
 *   1. Method call on the array: `x.add(1)`, `x.dispose()`
 *   2. Passing the array to ANY function call: `np.multiply(x, y)`,
 *      `myHelper(x)`, `obj.process(x)` — under move semantics, passing
 *      an array transfers ownership.
 *
 * Exceptions:
 *   - Known safe callees that never consume (console.log, expect, etc.)
 *     are automatically excluded.
 *   - A `// @jax-borrow` comment on the call line suppresses consumption
 *     tracking for that call, for user-defined non-consuming helpers.
 *
 * To keep an array alive past a consuming call, use `.ref` at the
 * consuming site: `x.ref.add(1)` instead of `x.add(1)`.
 */

import type { Rule } from "eslint";
import type * as ESTree from "estree";

import {
  ARRAY_RETURNING_METHODS,
  CONSUMING_TERMINAL_METHODS,
  findVariable,
  isArrayInit,
  parentOf,
} from "../shared";

// ---------------------------------------------------------------------------
// Safe-callee lists (calls known NOT to consume jax-js arrays)
// ---------------------------------------------------------------------------

/**
 * Object names whose method calls (`obj.method(x)`) never consume jax arrays.
 *
 * console — logging/debugging (console.log, console.warn, etc.)
 * assert  — Node.js assert module (assert.deepEqual, assert.ok, etc.)
 */
const SAFE_CALLEE_OBJECTS = new Set<string>(["console", "assert"]);

/**
 * Bare function names (`func(x)`) known not to consume jax arrays.
 *
 * expect — vitest/jest assertion entry point (expect(x).toBeDefined())
 * assert — Node.js assert() or chai assert()
 * String — type coercion, calls toString()
 * Number — type coercion for scalar arrays
 */
const SAFE_CALLEE_NAMES = new Set<string>([
  "expect",
  "assert",
  "String",
  "Number",
]);

/** Is the callee known not to consume jax arrays? */
function isSafeCallee(callee: ESTree.Expression): boolean {
  if (callee.type === "Identifier") {
    return SAFE_CALLEE_NAMES.has(callee.name);
  }
  if (
    callee.type === "MemberExpression" &&
    callee.object.type === "Identifier"
  ) {
    return SAFE_CALLEE_OBJECTS.has(callee.object.name);
  }
  return false;
}

/** Human-readable description of a callee expression. */
function describeCallee(callee: ESTree.Expression): string {
  if (callee.type === "Identifier") {
    return `${callee.name}()`;
  }
  if (
    callee.type === "MemberExpression" &&
    !callee.computed &&
    callee.property.type === "Identifier"
  ) {
    if (callee.object.type === "Identifier") {
      return `${callee.object.name}.${callee.property.name}()`;
    }
    // nested: a.b.c()
    if (
      callee.object.type === "MemberExpression" &&
      !callee.object.computed &&
      callee.object.property.type === "Identifier" &&
      callee.object.object.type === "Identifier"
    ) {
      return `${callee.object.object.name}.${callee.object.property.name}.${callee.property.name}()`;
    }
  }
  return "a function call";
}

// ---------------------------------------------------------------------------
// Consumption detection
// ---------------------------------------------------------------------------

/**
 * If this reference consumes the variable, return a description of how.
 *
 * Detected patterns:
 *   1. Method call on the array:  x.add(1), x.dispose()
 *   2. Argument to a jax-js namespace function: np.multiply(x, y)
 */
interface ConsumingSite {
  description: string;
  /** The identifier node at the consuming site (for .ref insertion). */
  identifier: ESTree.Identifier;
  /** "method" = x.method(), "argument" = np.func(x) */
  kind: "method" | "argument";
}

function getConsumingSite(identifier: ESTree.Identifier): ConsumingSite | null {
  const parent = parentOf(identifier);
  if (!parent) return null;

  // Pattern 1: x.method(...)
  if (
    parent.type === "MemberExpression" &&
    parent.object === identifier &&
    !parent.computed &&
    parent.property.type === "Identifier"
  ) {
    const prop = parent.property.name;
    if (
      !ARRAY_RETURNING_METHODS.has(prop) &&
      !CONSUMING_TERMINAL_METHODS.has(prop)
    ) {
      return null;
    }
    const gp = parentOf(parent);
    if (
      gp?.type === "CallExpression" &&
      (gp as ESTree.CallExpression).callee === parent
    ) {
      return {
        description: `.${prop}()`,
        identifier,
        kind: "method",
      };
    }
    return null;
  }

  // Pattern 2: x passed as argument to any function call.
  // Under move semantics, passing an array transfers ownership.
  // Exceptions: known safe callees (console.log, etc.) — the @jax-borrow
  // comment directive is checked separately in the main loop.
  if (
    parent.type === "CallExpression" &&
    (parent as ESTree.CallExpression).arguments.includes(identifier as any)
  ) {
    const callee = (parent as ESTree.CallExpression).callee;
    if (callee.type === "Super") return null;
    if (isSafeCallee(callee)) return null;
    return { description: describeCallee(callee), identifier, kind: "argument" };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Scope helpers
// ---------------------------------------------------------------------------

/**
 * Check if a reference originates from a nested function relative to the
 * variable's defining scope. Closure references are unreliable for
 * consumption tracking — the function might not execute, might throw
 * first (e.g., `expect(() => jaxFunc(x)).toThrow()`), or execute
 * after other code has already consumed the variable.
 */
function isInNestedFunction(ref: any, variable: any): boolean {
  let scope = ref.from;
  while (scope && scope !== variable.scope) {
    if (scope.type === "function") return true;
    scope = scope.upper;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Early-terminating if-branch detection
// ---------------------------------------------------------------------------

/**
 * If `node` is inside the consequent of an `if` whose body definitely
 * terminates (return / throw / continue / break), return the IfStatement.
 * Otherwise return null.
 *
 * This lets us reset consumption tracking after the `if`, since the code
 * following the `if` is a mutually exclusive branch.
 */
function getTerminatingIfAncestor(
  node: ESTree.Node,
): ESTree.IfStatement | null {
  let current: ESTree.Node = node;
  let parent = parentOf(current);
  while (parent) {
    // We only care about the consequent (then-branch), not the alternate.
    if (
      parent.type === "IfStatement" &&
      (parent as ESTree.IfStatement).consequent === current
    ) {
      if (blockTerminates(current)) return parent as ESTree.IfStatement;
    }
    // Also handle: direct child of a BlockStatement that is the consequent
    if (parent.type === "BlockStatement") {
      const gp = parentOf(parent);
      if (
        gp?.type === "IfStatement" &&
        (gp as ESTree.IfStatement).consequent === parent
      ) {
        if (blockTerminates(parent)) return gp as ESTree.IfStatement;
      }
    }
    // Stop at function boundaries.
    if (
      parent.type === "ArrowFunctionExpression" ||
      parent.type === "FunctionExpression" ||
      parent.type === "FunctionDeclaration"
    ) {
      break;
    }
    current = parent;
    parent = parentOf(current);
  }
  return null;
}

/** Does a statement or block definitely terminate (return/throw/break/continue)? */
function blockTerminates(node: ESTree.Node): boolean {
  if (
    node.type === "ReturnStatement" ||
    node.type === "ThrowStatement" ||
    node.type === "BreakStatement" ||
    node.type === "ContinueStatement"
  ) {
    return true;
  }
  if (node.type === "BlockStatement") {
    const body = (node as ESTree.BlockStatement).body;
    return body.length > 0 && blockTerminates(body[body.length - 1]);
  }
  if (node.type === "IfStatement") {
    const ifStmt = node as ESTree.IfStatement;
    return (
      blockTerminates(ifStmt.consequent) &&
      !!ifStmt.alternate &&
      blockTerminates(ifStmt.alternate)
    );
  }
  return false;
}

// ---------------------------------------------------------------------------
// Consume-and-reassign detection
// ---------------------------------------------------------------------------

/**
 * Check if `identifier` (being consumed) is inside the RHS of an assignment
 * that writes BACK to the same variable.
 *
 * Pattern: `x = x.method(...)` or `x = np.func(x, ...)`
 * These consume and immediately reassign, so x is valid afterward.
 */
function isConsumeAndReassign(
  identifier: ESTree.Identifier,
  varName: string,
): boolean {
  let node: ESTree.Node = identifier;
  let parent = parentOf(node);
  while (parent) {
    if (
      parent.type === "AssignmentExpression" &&
      (parent as ESTree.AssignmentExpression).right === node &&
      (parent as ESTree.AssignmentExpression).left.type === "Identifier" &&
      ((parent as ESTree.AssignmentExpression).left as ESTree.Identifier)
        .name === varName
    ) {
      return true;
    }
    // Stop at statement or function boundaries.
    if (
      parent.type === "ExpressionStatement" ||
      parent.type === "VariableDeclaration" ||
      parent.type === "ReturnStatement" ||
      parent.type === "ArrowFunctionExpression" ||
      parent.type === "FunctionExpression" ||
      parent.type === "FunctionDeclaration"
    ) {
      break;
    }
    node = parent;
    parent = parentOf(node);
  }
  return false;
}

// ---------------------------------------------------------------------------
// Rule
// ---------------------------------------------------------------------------

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow using a jax-js Array after it has been consumed by a method call or jax-js function",
      recommended: true,
    },
    hasSuggestions: true,
    messages: {
      useAfterConsume:
        "`{{name}}` is used after being consumed by `{{consumedBy}}` (line {{consumedLine}}). " +
        "Use `.ref` at the consuming site to keep the array alive. " +
        "(Can be ignored inside jit.)",
      suggestRef: "Insert `.ref` at the consuming site (line {{consumedLine}})",
    },
    schema: [],
  },

  create(context) {
    const sourceCode = context.sourceCode ?? context.getSourceCode();
    const sourceLines = (sourceCode.getText?.() ?? "").split("\n");

    /**
     * Check if the call containing `identifier` has a `// @jax-borrow`
     * comment directive on the same line or the line above.
     */
    function lineHasJaxSafe(identifier: ESTree.Identifier): boolean {
      const call = parentOf(identifier);
      const line = call?.loc?.start.line ?? identifier.loc?.start.line;
      if (!line) return false;
      for (const ln of [line - 1, line]) {
        if (ln < 1) continue;
        const lineText = sourceLines[ln - 1];
        if (lineText && /\/[/*].*@jax-borrow/.test(lineText)) return true;
      }
      return false;
    }

    return {
      VariableDeclarator(node: ESTree.VariableDeclarator) {
        if (node.id.type !== "Identifier") return;
        if (!isArrayInit(node.init)) return;

        const varName = node.id.name;
        const variable = findVariable(sourceCode.getScope(node), varName);
        if (!variable) return;

        // All references excluding the declaration itself, in source order.
        const refs = variable.references.filter(
          (ref: any) => ref.identifier !== node.id,
        );

        let consumedBy: ConsumingSite | null = null;
        /** If the consuming site is inside a terminating if-branch, track the IfStatement. */
        let consumedInIf: ESTree.IfStatement | null = null;

        for (const ref of refs) {
          // Write reference (reassignment) resets consumption tracking.
          if ((ref as any).isWrite()) {
            consumedBy = null;
            consumedInIf = null;
            continue;
          }

          if (!(ref as any).isRead()) continue;

          if (consumedBy !== null) {
            // If the consuming site was inside a terminating if-branch,
            // and this reference is AFTER that if (not inside it),
            // the two usages are mutually exclusive — reset.
            if (consumedInIf) {
              const ifEnd = consumedInIf.range?.[1] ?? 0;
              const refStart = ref.identifier.range?.[0] ?? 0;
              if (refStart >= ifEnd) {
                consumedBy = null;
                consumedInIf = null;
                // Fall through to re-evaluate this ref as a potential consuming site.
              }
            }
          }

          if (consumedBy !== null) {
            const consumedLine = String(
              consumedBy.identifier.loc?.start.line ?? "?",
            );
            context.report({
              node: ref.identifier,
              messageId: "useAfterConsume",
              data: {
                name: varName,
                consumedBy: consumedBy.description,
                consumedLine,
              },
              suggest: [
                {
                  messageId: "suggestRef",
                  data: { consumedLine },
                  fix: (fixer) =>
                    fixer.insertTextAfter(consumedBy!.identifier, ".ref"),
                },
              ],
            });
            continue;
          }

          // Skip closure references for consumption detection — they may
          // not execute (e.g., inside expect(() => ...).toThrow()).
          if (isInNestedFunction(ref, variable)) continue;

          const site = getConsumingSite(ref.identifier);
          if (site && !isConsumeAndReassign(ref.identifier, varName)) {
            // User can suppress arbitrary-call consumption via // @jax-borrow
            if (site.kind === "argument" && lineHasJaxSafe(site.identifier)) {
              continue;
            }
            consumedBy = site;
            consumedInIf = getTerminatingIfAncestor(ref.identifier);
          }
        }
      },
    };
  },
};

export default rule;
