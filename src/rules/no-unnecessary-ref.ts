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
    messages: {
      unnecessaryRef:
        "Unnecessary `.ref` — `{{name}}` is not used after this, so the `.ref` creates a leaked reference. Remove `.ref` to let the operation consume the array directly.",
      unnecessaryRefOnlyProps:
        "Unnecessary `.ref` — `{{name}}` is only used for non-consuming property access ({{props}}) afterward. Add `.dispose()` or remove `.ref`.",
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

        const allRefs = variable.references;
        const idx = allRefs.findIndex((r: any) => r.identifier === node.object);
        if (idx === -1) return;

        // If the variable is consumed BEFORE this `.ref` in source order,
        // the `.ref` is justified (keeps array alive for a later use).
        // e.g., `equal(a, min(a.ref, ...))`.
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
          context.report({
            node,
            messageId: "unnecessaryRefOnlyProps",
            data: {
              name: varName,
              props: props.map((p) => `.${p}`).join(", "),
            },
            fix: (fixer) => removeRef(fixer, node, sourceCode),
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

export default rule;
