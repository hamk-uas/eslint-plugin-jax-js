/**
 * @jax-js/no-unnecessary-ref
 *
 * Warns when `.ref` is used on a variable that is never referenced again
 * after the `.ref` chain. This is a guaranteed array memory leak.
 *
 * `.ref` increments the reference count. If the variable holding the
 * original array is never used again, its rc stays at 1 forever (leak).
 *
 * Provides an autofix that removes `.ref` from the chain.
 */

import type { Rule } from "eslint";
import type * as ESTree from "estree";

import {
  findVariable,
  isBorrowedBinding,
  isNonConsumingReference,
  parentOf,
} from "../shared";

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow `.ref` when the variable is not used again afterward (guaranteed leak)",
      recommended: true,
    },
    fixable: "code",
    hasSuggestions: true,
    messages: {
      unnecessaryRef:
        "Unnecessary `.ref` — `{{name}}` is not used after this, so the `.ref` creates a leaked reference. Remove `.ref` to let the operation consume the array directly.",
      unnecessaryRefOnlyProps:
        "Unnecessary `.ref` — `{{name}}` is only used for non-consuming property access ({{props}}) afterward. Add `.dispose()` after the last use to avoid a leak.",
      addDispose:
        "Add `.dispose()` after the last use of `{{name}}` to fix the leak.",
    },
    schema: [],
  },

  create(context) {
    const sourceCode = context.sourceCode ?? context.getSourceCode();

    return {
      'MemberExpression[property.name="ref"][computed=false]'(
        node: ESTree.MemberExpression,
      ) {
        if (node.object.type !== "Identifier") return;

        const varName = node.object.name;
        const variable = findVariable(sourceCode.getScope(node), varName);
        if (!variable) return;

        // Borrowed bindings (callback params, for-of vars) use `.ref`
        // for intentional cloning — skip.
        if (isBorrowedBinding(variable)) return;

        // If the `.ref` is inside a nested function (closure capturing the
        // variable from an outer scope), skip. The closure may be invoked
        // multiple times (e.g., by grad(), jit(), vmap()), and the `.ref`
        // keeps the captured array alive across invocations.
        const refScope = sourceCode.getScope(node);
        if (isClosureCapture(variable.scope, refScope)) return;

        const allRefs = variable.references;
        const idx = allRefs.findIndex((r: any) => r.identifier === node.object);
        if (idx === -1) return;

        // If the variable is consumed BEFORE this `.ref` in source order,
        // the `.ref` is justified (keeps array alive for a later use).
        // e.g., `equal(a, min(a.ref, ...))` or `foo(x); x.ref.dataSync()`.
        // Any non-property read counts as a potential consuming use because
        // jax-js uses move semantics — passing an array to any function
        // transfers ownership.
        for (const ref of allRefs.slice(0, idx)) {
          if (ref.isRead() && !isNonConsumingReference(ref)) return;
        }

        const after = allRefs.slice(idx + 1);

        if (after.length === 0) {
          context.report({
            node,
            messageId: "unnecessaryRef",
            data: { name: varName },
            fix: (fixer) => removeRef(fixer, node, sourceCode),
          });
          return;
        }

        // If any later reference is itself a `.ref` access, the current
        // `.ref` is needed — the variable participates in multiple
        // consuming chains and must survive past each one.
        for (const ref of after) {
          if (!ref.isRead()) continue;
          const parent = parentOf(ref.identifier);
          if (
            parent?.type === "MemberExpression" &&
            (parent as ESTree.MemberExpression).object === ref.identifier &&
            !(parent as ESTree.MemberExpression).computed &&
            (parent as ESTree.MemberExpression).property.type === "Identifier" &&
            ((parent as ESTree.MemberExpression).property as ESTree.Identifier)
              .name === "ref"
          ) {
            return;
          }
        }

        // Collect non-consuming prop names; bail on any consuming ref.
        const props: string[] = [];
        const allNonConsuming = after.every((ref: any) => {
          if (!ref.isRead()) return true;
          if (!isNonConsumingReference(ref)) return false;
          const p = (
            (ref.identifier as any).parent.property as ESTree.Identifier
          ).name;
          if (!props.includes(p)) props.push(p);
          return true;
        });

        if (allNonConsuming && props.length > 0) {
          // The `.ref` keeps the array alive for later non-consuming
          // accesses (.shape, .dtype, etc.). Removing `.ref` would
          // let the chained operation consume the array, crashing
          // the later accesses. Report as a warning without autofix;
          // the real fix is to add `.dispose()` after the last use.
          const lastReadRef = [...after].reverse().find((r: any) => r.isRead());
          context.report({
            node,
            messageId: "unnecessaryRefOnlyProps",
            data: {
              name: varName,
              props: props.map((p) => `.${p}`).join(", "),
            },
            suggest: lastReadRef
              ? [
                  {
                    messageId: "addDispose",
                    data: { name: varName },
                    fix: (fixer) => {
                      const lastId = lastReadRef.identifier;
                      const lastParent = parentOf(lastId);
                      // Find the end of the statement containing the last use.
                      let stmtNode: ESTree.Node = lastParent ?? lastId;
                      while (
                        parentOf(stmtNode) &&
                        (parentOf(stmtNode) as any).type !== "Program" &&
                        (parentOf(stmtNode) as any).type !== "BlockStatement"
                      ) {
                        stmtNode = parentOf(stmtNode)!;
                      }
                      const stmtEnd = stmtNode.range
                        ? stmtNode.range[1]
                        : null;
                      if (stmtEnd === null) return null;
                      return fixer.insertTextAfterRange(
                        [stmtEnd, stmtEnd],
                        `\n${varName}.dispose();`,
                      );
                    },
                  },
                ]
              : [],
          });
        }
      },
    };
  },
};

/** Remove `.ref` from `x.ref` or `x.ref.method()`. */
function removeRef(
  fixer: Rule.RuleFixer,
  node: ESTree.MemberExpression,
  sourceCode: any,
): Rule.Fix | null {
  const dot = sourceCode.getTokenAfter(node.object, {
    filter: (t: { value: string }) => t.value === ".",
  });
  const ref = dot && sourceCode.getTokenAfter(dot);
  return dot && ref ? fixer.removeRange([dot.range[0], ref.range[1]]) : null;
}

/**
 * Check if there is a function scope boundary between the variable's
 * defining scope and the reference scope. If so, the reference is a
 * closure capture — the `.ref` may be needed for multi-invocation safety.
 */
function isClosureCapture(variableScope: any, refScope: any): boolean {
  let scope = refScope;
  while (scope && scope !== variableScope) {
    if (scope.type === "function") return true;
    scope = scope.upper;
  }
  return false;
}

export default rule;
