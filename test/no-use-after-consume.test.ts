import { RuleTester } from "eslint";
import { describe, test } from "vitest";

import rule from "../src/rules/no-use-after-consume";

const tester = new RuleTester();

describe("no-use-after-consume", () => {
  test("passes rule tester", () => {
    tester.run("no-use-after-consume", rule, {
      valid: [
        // Consumed by method call, no later use
        "const x = np.zeros([3]); x.add(1);",

        // .ref keeps variable alive for later use
        "const x = np.zeros([3]); const y = x.ref.add(1); x.dispose();",

        // Non-consuming property access before consuming method
        "const x = np.zeros([3]); console.log(x.shape); x.dispose();",

        // Not an array init — rule doesn't apply
        "const x = getValue(); x.add(1); x.shape;",

        // Reassignment resets consumption tracking
        "let x = np.zeros([3]); x.dispose(); x = np.ones([3]); x.add(1);",

        // Passed to non-jax function — not tracked as consuming
        "const x = np.zeros([3]); foo(x); bar(x);",

        // Multiple .ref usages — .ref doesn't invalidate
        "const x = np.zeros([3]); x.ref.add(1); x.ref.mul(2); x.dispose();",

        // Only property accesses then dispose
        "const x = np.zeros([3]); x.shape; x.dtype; x.dispose();",

        // Property accesses before consuming method at end
        "const x = np.zeros([3]); console.log(x.shape); x.add(1);",

        // Array created via .ref, consumed once
        "const x = y.ref; x.add(1);",

        // Passed to jax function, no later use
        "const x = np.zeros([3]); np.multiply(x, 2);",

        // Method access without calling it (no parentheses — not consuming)
        "const x = np.zeros([3]); const fn = x.add; x.dispose();",

        // Consume-and-reassign: x = x.method() — x is valid afterward
        "let x = np.zeros([3]); x = x.add(1); x.dispose();",

        // Consume-and-reassign via jax function: x = np.func(x)
        "let x = np.zeros([3]); x = np.reshape(x, [1, 3]); x.dispose();",

        // blockUntilReady — doesn't consume the array
        "const x = np.zeros([3]); x.blockUntilReady(); x.dispose();",

        // Closure references — consumption inside closures is not tracked
        // (function might throw before consuming, e.g. expect().toThrow())
        "const x = np.zeros([3]); expect(() => np.reshape(x, [99])).toThrow(); x.dispose();",
        "const x = np.array([1]); expect(() => x.add(1)).toThrow(); x.dispose();",

        // Mutually exclusive if-return branches — consumed in if that returns,
        // then used again after the if (lax.ts dot_general pattern)
        {
          code: `
            function f() {
              const x = np.zeros([3]);
              if (cond) {
                return x.reshape([1, 3]);
              }
              return x.reshape([3, 1]);
            }
          `,
        },

        // Same pattern with throw instead of return
        {
          code: `
            function f() {
              const x = np.zeros([3]);
              if (cond) {
                throw x.dispose();
              }
              x.add(1);
            }
          `,
        },
      ],

      invalid: [
        // Use after method call (.add)
        {
          code: "const x = np.zeros([3]); x.add(1); x.shape;",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output: "const x = np.zeros([3]); x.ref.add(1); x.shape;",
                },
              ],
            },
          ],
        },

        // Use after .dispose()
        {
          code: "const x = np.zeros([3]); x.dispose(); x.shape;",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output: "const x = np.zeros([3]); x.ref.dispose(); x.shape;",
                },
              ],
            },
          ],
        },

        // Double dispose
        {
          code: "const x = np.array([1]); x.dispose(); x.dispose();",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output:
                    "const x = np.array([1]); x.ref.dispose(); x.dispose();",
                },
              ],
            },
          ],
        },

        // Use after .js()
        {
          code: "const x = np.zeros([3]); x.js(); x.shape;",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output: "const x = np.zeros([3]); x.ref.js(); x.shape;",
                },
              ],
            },
          ],
        },

        // Multiple uses after consume
        {
          code: "const x = np.zeros([3]); x.add(1); x.shape; x.dtype;",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output:
                    "const x = np.zeros([3]); x.ref.add(1); x.shape; x.dtype;",
                },
              ],
            },
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output:
                    "const x = np.zeros([3]); x.ref.add(1); x.shape; x.dtype;",
                },
              ],
            },
          ],
        },

        // Consuming method after initial consume
        {
          code: "const x = np.zeros([3]); x.add(1); x.mul(2);",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output: "const x = np.zeros([3]); x.ref.add(1); x.mul(2);",
                },
              ],
            },
          ],
        },

        // Use after passing to jax namespace function
        {
          code: "const x = np.zeros([3]); np.multiply(x, 2); x.shape;",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output:
                    "const x = np.zeros([3]); np.multiply(x.ref, 2); x.shape;",
                },
              ],
            },
          ],
        },

        // Use after passing to nested jax namespace (lax.linalg)
        {
          code: "const x = np.array([[1,0],[0,1]]); lax.linalg.cholesky(x); x.shape;",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output:
                    "const x = np.array([[1,0],[0,1]]); lax.linalg.cholesky(x.ref); x.shape;",
                },
              ],
            },
          ],
        },

        // Consume then pass to non-jax function (still flagged — already consumed)
        {
          code: "const x = np.zeros([3]); x.dispose(); foo(x);",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output: "const x = np.zeros([3]); x.ref.dispose(); foo(x);",
                },
              ],
            },
          ],
        },
      ],
    });
  });
});
