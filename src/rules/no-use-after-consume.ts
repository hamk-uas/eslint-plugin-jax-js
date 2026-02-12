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
 * JSON    — JSON.stringify / JSON.parse (serialization)
 * Object  — Object.keys, Object.values, Object.entries, etc.
 * Array   — Array.isArray, Array.from (introspection/conversion)
 * Math    — Math.min, Math.max, etc. (numeric, not array-consuming)
 */
const SAFE_CALLEE_OBJECTS = new Set<string>([
  "console",
  "assert",
  "JSON",
  "Object",
  "Array",
  "Math",
]);

/**
 * Bare function names (`func(x)`) known not to consume jax arrays.
 *
 * expect    — vitest/jest assertion entry point (expect(x).toBeDefined())
 * assert    — Node.js assert() or chai assert()
 * String    — type coercion, calls toString()
 * Number    — type coercion for scalar arrays
 * Boolean   — type coercion
 * parseInt  — numeric conversion
 * parseFloat — numeric conversion
 * isNaN     — numeric check
 * isFinite  — numeric check
 * describe  — test framework
 * it        — test framework
 * test      — test framework
 */
const SAFE_CALLEE_NAMES = new Set<string>([
  "expect",
  "assert",
  "String",
  "Number",
  "Boolean",
  "parseInt",
  "parseFloat",
  "isNaN",
  "isFinite",
  "describe",
  "it",
  "test",
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

/**
 * Check if two nodes are in mutually exclusive branches of the same
 * conditional or logical expression. For example, in `cond ? a : b`,
 * `a` and `b` are mutually exclusive.
 */
function areInMutuallyExclusiveBranches(
  node1: ESTree.Node,
  node2: ESTree.Node,
): boolean {
  // IfStatement branches (if/else and if/else-if/else chains):
  // nodes are mutually exclusive if one is in the consequent subtree
  // and the other is in the alternate subtree of the same IfStatement.
  const ifAncestors1 = getIfAncestors(node1);
  for (const ifStmt of ifAncestors1) {
    const side1 = getIfBranchSide(node1, ifStmt);
    const side2 = getIfBranchSide(node2, ifStmt);
    if (side1 && side2 && side1 !== side2) return true;
  }

  // Find the common conditional ancestor
  const cond1 = getConditionalAncestor(node1);
  const cond2 = getConditionalAncestor(node2);
  if (!cond1 || !cond2 || cond1 !== cond2) return false;

  if (cond1.type === "ConditionalExpression") {
    const inConsequent1 = isDescendantOf(node1, cond1.consequent);
    const inAlternate1 = isDescendantOf(node1, cond1.alternate);
    const inConsequent2 = isDescendantOf(node2, cond1.consequent);
    const inAlternate2 = isDescendantOf(node2, cond1.alternate);
    // One in consequent, other in alternate
    return (inConsequent1 && inAlternate2) || (inAlternate1 && inConsequent2);
  }

  // For logical expressions (&&, ||, ??), both operands (left/right) where
  // right is the short-circuited branch. If both are in the right operand,
  // they'll both execute or neither will — they're NOT mutually exclusive.
  // But consumption in left and reference in right are sequential, not exclusive.
  return false;
}

function getIfAncestors(node: ESTree.Node): ESTree.IfStatement[] {
  const result: ESTree.IfStatement[] = [];
  let current: ESTree.Node | undefined = node;
  while (current) {
    const parent = parentOf(current);
    if (!parent) break;
    if (parent.type === "IfStatement") {
      result.push(parent as ESTree.IfStatement);
    }
    current = parent;
  }
  return result;
}

function getIfBranchSide(
  node: ESTree.Node,
  ifStmt: ESTree.IfStatement,
): "consequent" | "alternate" | null {
  let current: ESTree.Node | undefined = node;
  while (current) {
    const parent = parentOf(current);
    if (!parent) break;
    if (parent === ifStmt) {
      if (ifStmt.consequent === current) return "consequent";
      if (ifStmt.alternate === current) return "alternate";
      return null;
    }
    current = parent;
  }
  return null;
}

/**
 * Check if `node` is a descendant of (or equal to) `ancestor`.
 */
function isDescendantOf(node: ESTree.Node, ancestor: ESTree.Node): boolean {
  let current: ESTree.Node | undefined = node;
  while (current) {
    if (current === ancestor) return true;
    current = parentOf(current);
  }
  return false;
}

// ---------------------------------------------------------------------------
// Expression evaluation order detection
// ---------------------------------------------------------------------------

/**
 * Check if `refNode` appears inside the arguments of a call expression
 * where the callee is a method on the SAME variable being consumed.
 *
 * Example: `ret.reshape([1, ...ret.shape])`
 * Here `ret.shape` is inside the arguments of `ret.reshape(...)`.
 * JS evaluates arguments before the method call, so `ret.shape` is
 * read while `ret` is still alive. This is NOT a use-after-consume.
 *
 * @param refNode — the identifier node of the reference being checked
 * @param consumingSite — the consuming site that already consumed the variable
 */
function isInArgumentsOfSameConsumingCall(
  refNode: ESTree.Identifier,
  consumingSite: ConsumingSite,
): boolean {
  const consumingIdent = consumingSite.identifier;
  let callExpr: ESTree.Node | undefined;

  if (consumingSite.kind === "method") {
    // x.method(...) — ident → MemberExpression → CallExpression
    const memberExpr = parentOf(consumingIdent);
    if (!memberExpr || memberExpr.type !== "MemberExpression") return false;
    callExpr = parentOf(memberExpr);
  } else if (consumingSite.kind === "argument") {
    // np.func(x, ...) — ident is a direct argument of a CallExpression
    callExpr = parentOf(consumingIdent);
  } else {
    return false;
  }

  if (!callExpr || callExpr.type !== "CallExpression") return false;

  // Check if refNode is a descendant of one of the arguments of that call.
  return isDescendantOfArguments(refNode, callExpr as ESTree.CallExpression);
}

/**
 * Check if a node is a descendant of any argument in a CallExpression.
 */
function isDescendantOfArguments(
  node: ESTree.Node,
  call: ESTree.CallExpression,
): boolean {
  let current: ESTree.Node | undefined = node;
  while (current) {
    if ((call.arguments as ESTree.Node[]).includes(current)) return true;
    current = parentOf(current);
    if (current === call) return false; // Reached the call without matching an arg
  }
  return false;
}

// ---------------------------------------------------------------------------
// Conditional / ternary / logical expression detection
// ---------------------------------------------------------------------------

/**
 * Check if the consuming identifier is inside a conditional branch
 * (ternary, logical &&/||/??, or if-else) that might not execute.
 *
 * If consumption is inside:
 *   - A ternary consequent/alternate → only one branch runs
 *   - An `&&` / `||` / `??` right operand → short-circuit may skip it
 *
 * Returns the outermost conditional expression node if detected, null otherwise.
 */
function getConditionalAncestor(
  node: ESTree.Node,
): ESTree.ConditionalExpression | ESTree.LogicalExpression | null {
  let current: ESTree.Node = node;
  let parent = parentOf(current);
  while (parent) {
    if (parent.type === "ConditionalExpression") {
      const cond = parent as ESTree.ConditionalExpression;
      // Only if the consuming node is in the consequent or alternate, not the test
      if (current === cond.consequent || current === cond.alternate) {
        return cond;
      }
    }
    if (parent.type === "LogicalExpression") {
      const logical = parent as ESTree.LogicalExpression;
      // Only if the consuming node is in the right operand (short-circuited)
      if (current === logical.right) {
        return logical;
      }
    }
    // Stop at statement boundaries
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
    current = parent;
    parent = parentOf(current);
  }
  return null;
}

// ---------------------------------------------------------------------------
// Loop detection
// ---------------------------------------------------------------------------

/**
 * Check if a node is inside the body of a loop statement (for, while,
 * do-while, for-of, for-in).
 *
 * Consumption inside a loop body should NOT be treated as definitely
 * consuming for the entire scope — the variable may be alive at loop
 * start (first iteration) and code after the loop.
 *
 * Returns the loop node if found, null otherwise.
 */
function getEnclosingLoop(
  node: ESTree.Node,
): ESTree.Node | null {
  let current: ESTree.Node = node;
  let parent = parentOf(current);
  while (parent) {
    if (
      (parent.type === "ForStatement" && current === (parent as ESTree.ForStatement).body) ||
      (parent.type === "WhileStatement" && current === (parent as ESTree.WhileStatement).body) ||
      (parent.type === "DoWhileStatement" && current === (parent as ESTree.DoWhileStatement).body) ||
      (parent.type === "ForOfStatement" && current === (parent as ESTree.ForOfStatement).body) ||
      (parent.type === "ForInStatement" && current === (parent as ESTree.ForInStatement).body)
    ) {
      return parent;
    }
    // Blocks that are the body of loops
    if (parent.type === "BlockStatement") {
      const gp = parentOf(parent);
      if (
        gp &&
        (
          (gp.type === "ForStatement" && (gp as ESTree.ForStatement).body === parent) ||
          (gp.type === "WhileStatement" && (gp as ESTree.WhileStatement).body === parent) ||
          (gp.type === "DoWhileStatement" && (gp as ESTree.DoWhileStatement).body === parent) ||
          (gp.type === "ForOfStatement" && (gp as ESTree.ForOfStatement).body === parent) ||
          (gp.type === "ForInStatement" && (gp as ESTree.ForInStatement).body === parent)
        )
      ) {
        return gp;
      }
    }
    // Stop at function boundaries
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
 * If `node` is inside the consequent OR alternate of an `if` whose
 * branch definitely terminates (return / throw / continue / break),
 * return the IfStatement. Otherwise return null.
 *
 * This lets us reset consumption tracking after the `if`, since the code
 * following the `if` is a mutually exclusive branch.
 *
 * Handles both:
 *   - Consumption in then-branch that terminates → code after if is safe
 *   - Consumption in else-branch that terminates → code after if is safe
 */
function getTerminatingIfAncestor(
  node: ESTree.Node,
): ESTree.IfStatement | null {
  let current: ESTree.Node = node;
  let parent = parentOf(current);
  while (parent) {
    if (parent.type === "IfStatement") {
      const ifStmt = parent as ESTree.IfStatement;
      // Check if consumption is in the consequent (then-branch)
      if (ifStmt.consequent === current && blockTerminates(current)) {
        return ifStmt;
      }
      // Check if consumption is in the alternate (else-branch)
      if (ifStmt.alternate === current && blockTerminates(current)) {
        return ifStmt;
      }
    }
    // Also handle: direct child of a BlockStatement that is the consequent/alternate
    if (parent.type === "BlockStatement") {
      const gp = parentOf(parent);
      if (gp?.type === "IfStatement") {
        const ifStmt = gp as ESTree.IfStatement;
        if (ifStmt.consequent === parent && blockTerminates(parent)) {
          return ifStmt;
        }
        if (ifStmt.alternate === parent && blockTerminates(parent)) {
          return ifStmt;
        }
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
    function lineHasJaxBorrow(identifier: ESTree.Identifier): boolean {
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
        /** If the consuming site is inside a conditional expression, track it. */
        let consumedInConditional: ESTree.ConditionalExpression | ESTree.LogicalExpression | null = null;
        /** If the consuming site is inside a loop body, track the loop. */
        let consumedInLoop: ESTree.Node | null = null;

        for (const ref of refs) {
          // Write reference (reassignment) resets consumption tracking.
          if ((ref as any).isWrite()) {
            consumedBy = null;
            consumedInIf = null;
            consumedInConditional = null;
            consumedInLoop = null;
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
                consumedInConditional = null;
                consumedInLoop = null;
                // Fall through to re-evaluate this ref as a potential consuming site.
              }
            }

            // If the consuming site was inside a conditional expression
            // (ternary or logical), and this reference is AFTER it,
            // the consumption might not have occurred — reset.
            if (consumedBy !== null && consumedInConditional) {
              const condEnd = consumedInConditional.range?.[1] ?? 0;
              const refStart = ref.identifier.range?.[0] ?? 0;
              if (refStart >= condEnd) {
                consumedBy = null;
                consumedInConditional = null;
                consumedInIf = null;
                consumedInLoop = null;
                // Fall through to re-evaluate.
              }
            }

            // If the consuming site was inside a loop body, and this
            // reference is AFTER the loop (not inside it), reset —
            // the variable could be alive at loop entry and code after.
            if (consumedBy !== null && consumedInLoop) {
              const loopEnd = consumedInLoop.range?.[1] ?? 0;
              const refStart = ref.identifier.range?.[0] ?? 0;
              if (refStart >= loopEnd) {
                consumedBy = null;
                consumedInLoop = null;
                consumedInIf = null;
                consumedInConditional = null;
                // Fall through to re-evaluate.
              }
            }
          }

          if (consumedBy !== null) {
            // Check for expression evaluation order: if this reference is
            // inside the arguments of the SAME call that consumed the var
            // (e.g., `x.reshape([1, ...x.shape])`), JS evaluates arguments
            // before the method call — so this is NOT a use-after-consume.
            if (isInArgumentsOfSameConsumingCall(ref.identifier, consumedBy)) {
              continue;
            }

            // If both the consuming site and this reference are in mutually
            // exclusive branches of the same ternary (e.g., `cond ? x.add(1) : x.sub(1)`),
            // they never both execute — skip.
            if (areInMutuallyExclusiveBranches(ref.identifier, consumedBy.identifier)) {
              // Check if this ref is ALSO a consuming site. If so, BOTH branches
              // consume → the consumption is unconditional. Update consumedBy
              // but clear consumedInConditional since both paths lead to consumption.
              const siteInOtherBranch = getConsumingSite(ref.identifier);
              if (siteInOtherBranch && !isConsumeAndReassign(ref.identifier, varName)) {
                consumedBy = siteInOtherBranch;
                consumedInConditional = null; // both branches consume → unconditional
                consumedInIf = getTerminatingIfAncestor(ref.identifier);
                consumedInLoop = getEnclosingLoop(ref.identifier);
                continue;
              }
              // Only one branch consumes — reset for code after the conditional.
              consumedBy = null;
              consumedInIf = null;
              consumedInConditional = null;
              consumedInLoop = null;
              // Fall through to the consuming-site detection below.
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
            if (site.kind === "argument" && lineHasJaxBorrow(site.identifier)) {
              continue;
            }
            consumedBy = site;
            consumedInIf = getTerminatingIfAncestor(ref.identifier);
            consumedInConditional = getConditionalAncestor(ref.identifier);
            consumedInLoop = getEnclosingLoop(ref.identifier);
          }
        }
      },
    };
  },
};

export default rule;
