/**
 * @jax-js/require-consume
 *
 * Warns when a variable holding a jax-js Array is never consumed — i.e.,
 * never passed to a consuming operation, returned, yielded, or explicitly
 * disposed.
 *
 * Non-consuming accesses like `.shape`, `.dtype`, `.ndim`, `.size`,
 * `.device`, `.refCount` do NOT count as consumption.
 *
 * This rule uses heuristics to identify variables that likely hold jax-js
 * Arrays by checking the right-hand side of their declarations. It flags
 * variables whose only usages are non-consuming property accesses.
 */

import type { Rule } from "eslint";
import type * as ESTree from "estree";

import {
  ARRAY_RETURNING_METHODS,
  CONSUMING_TERMINAL_METHODS,
  findVariable,
  isArrayInit,
  NON_CONSUMING_PROPS,
  parentOf,
} from "../shared";

// ---------------------------------------------------------------------------
// Consuming-usage detection
// ---------------------------------------------------------------------------

/** TS transparent wrapper node types. */
const TS_WRAPPERS = new Set([
  "TSAsExpression",
  "TSNonNullExpression",
  "TSSatisfiesExpression",
  "TSTypeAssertion",
]);

/** Is a particular reference to a variable "consuming"? */
function isConsumingUsage(identifier: ESTree.Identifier): boolean {
  const parent = parentOf(identifier);
  if (!parent) return true; // can't prove non-consuming → conservative

  // TS cast wrappers — assume consuming (conservative)
  if (TS_WRAPPERS.has((parent as any).type)) return true;

  // Member access: `x.prop`
  if (parent.type === "MemberExpression" && parent.object === identifier) {
    if (parent.computed) return true;
    const prop = (parent.property as ESTree.Identifier).name;
    if (CONSUMING_TERMINAL_METHODS.has(prop)) return true;
    if (ARRAY_RETURNING_METHODS.has(prop)) return true;
    if (prop === "ref") {
      // `.ref` as a standalone expression (`x.ref;`) doesn't consume.
      // In any other context the ref result is captured → consuming.
      const gp = parentOf(parent);
      return !!gp && gp.type !== "ExpressionStatement";
    }
    if (NON_CONSUMING_PROPS.has(prop)) return false;
    return true; // unknown prop → conservative
  }

  // Passed as argument, spread, returned, yielded, assigned, etc.
  if (
    parent.type === "CallExpression" &&
    parent.arguments.includes(identifier as any)
  )
    return true;
  if (
    parent.type === "SpreadElement" ||
    parent.type === "ReturnStatement" ||
    parent.type === "YieldExpression" ||
    parent.type === "ArrayExpression" ||
    parent.type === "TemplateLiteral" ||
    parent.type === "ConditionalExpression" ||
    parent.type === "LogicalExpression" ||
    parent.type === "BinaryExpression"
  ) {
    return true;
  }
  if (parent.type === "AssignmentExpression" && parent.right === identifier)
    return true;
  if (parent.type === "VariableDeclarator" && parent.init === identifier)
    return true;
  if (parent.type === "Property" && parent.value === identifier) return true;

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
        "Require jax-js Arrays to be consumed or disposed before going out of scope",
      recommended: true,
    },
    hasSuggestions: true,
    messages: {
      neverConsumed:
        "Array `{{name}}` is never consumed or disposed. Call `.dispose()` or pass it to an operation. " +
        "(Can be ignored inside jit.)",
      onlyPropertyAccess:
        "Array `{{name}}` is only used for non-consuming property access ({{props}}). Call `.dispose()` when done. " +
        "(Can be ignored inside jit.)",
      suggestDispose: "Add `{{name}}.dispose()` after last use",
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

        const readRefs = variable.references.filter(
          (ref: any) => ref.isRead() && ref.identifier !== node.id,
        );

        if (readRefs.length === 0) {
          // Walk up from the declarator to find the VariableDeclaration
          // to insert `.dispose()` after.
          let insertAfter: ESTree.Node = node;
          const parent = parentOf(node);
          if (parent?.type === "VariableDeclaration") insertAfter = parent;

          context.report({
            node: node.id,
            messageId: "neverConsumed",
            data: { name: varName },
            suggest: [
              {
                messageId: "suggestDispose",
                data: { name: varName },
                fix: (fixer) =>
                  fixer.insertTextAfter(insertAfter, `\n${varName}.dispose();`),
              },
            ],
          });
          return;
        }

        const nonConsumingProps: string[] = [];
        let consumed = false;

        for (const ref of readRefs) {
          if (isConsumingUsage(ref.identifier)) {
            consumed = true;
            break;
          }
          const parent = parentOf(ref.identifier);
          if (
            parent?.type === "MemberExpression" &&
            !parent.computed &&
            parent.property.type === "Identifier"
          ) {
            const p = parent.property.name;
            if (!nonConsumingProps.includes(p)) nonConsumingProps.push(p);
          }
        }

        if (!consumed) {
          // Find the last reference (or the declarator itself) for
          // suggesting where to insert `.dispose()`.
          const lastRef =
            readRefs.length > 0 ? readRefs[readRefs.length - 1] : null;
          const lastNode = lastRef ? lastRef.identifier : node.id;

          // Walk up to the enclosing ExpressionStatement or
          // VariableDeclaration to insert `.dispose()` after it.
          let insertAfter: ESTree.Node = lastNode;
          let p = parentOf(insertAfter);
          while (
            p &&
            p.type !== "ExpressionStatement" &&
            p.type !== "VariableDeclaration"
          ) {
            insertAfter = p;
            p = parentOf(insertAfter);
          }
          if (p) insertAfter = p;

          const suggest: Rule.SuggestionReportDescriptor[] = [
            {
              messageId: "suggestDispose",
              data: { name: varName },
              fix: (fixer) =>
                fixer.insertTextAfter(insertAfter, `\n${varName}.dispose();`),
            },
          ];

          context.report({
            node: node.id,
            messageId:
              nonConsumingProps.length > 0
                ? "onlyPropertyAccess"
                : "neverConsumed",
            data: {
              name: varName,
              props: nonConsumingProps.map((p) => `.${p}`).join(", "),
            },
            suggest,
          });
        }
      },
    };
  },
};

export default rule;
