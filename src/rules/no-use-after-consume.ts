/**
 * @jax-js/no-use-after-consume
 *
 * Warns when a variable holding a jax-js Array is used after being consumed.
 *
 * In jax-js's consuming ownership model, most operations dispose their
 * input arrays. Using an array after it has been consumed (by a method
 * call like `.add()` or by passing it to a jax-js function like
 * `np.multiply()`) is a use-after-free bug.
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
  isJaxNamespaceCall,
  JAX_NAMESPACES,
  parentOf,
} from "../shared";

// ---------------------------------------------------------------------------
// Jax-js namespace description (uses shared detection)
// ---------------------------------------------------------------------------

/** Import aliases commonly used for jax-js modules. */
// (JAX_NAMESPACES is now imported from shared.ts)

/**
 * If the callee is a jax-js namespace function, return a human-readable
 * description like `np.multiply()`. Otherwise return null.
 */
function describeJaxCall(callee: ESTree.Expression): string | null {
  if (
    callee.type !== "MemberExpression" ||
    callee.computed ||
    callee.property.type !== "Identifier"
  ) {
    return null;
  }
  if (!isJaxNamespaceCall(callee)) return null;
  const prop = callee.property.name;
  // np.func(), lax.func()
  if (
    callee.object.type === "Identifier" &&
    JAX_NAMESPACES.has(callee.object.name)
  ) {
    return `${callee.object.name}.${prop}()`;
  }
  // lax.linalg.func() — nested namespace
  if (
    callee.object.type === "MemberExpression" &&
    !callee.object.computed &&
    callee.object.property.type === "Identifier" &&
    callee.object.object.type === "Identifier" &&
    JAX_NAMESPACES.has(callee.object.object.name)
  ) {
    return `${callee.object.object.name}.${callee.object.property.name}.${prop}()`;
  }
  return null;
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

  // Pattern 2: np.func(x), lax.linalg.func(x)
  if (
    parent.type === "CallExpression" &&
    (parent as ESTree.CallExpression).arguments.includes(identifier as any)
  ) {
    const callee = (parent as ESTree.CallExpression).callee;
    if (callee.type !== "Super") {
      const desc = describeJaxCall(callee);
      if (desc) return { description: desc, identifier, kind: "argument" };
    }
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
            consumedBy = site;
            consumedInIf = getTerminatingIfAncestor(ref.identifier);
          }
        }
      },
    };
  },
};

export default rule;
