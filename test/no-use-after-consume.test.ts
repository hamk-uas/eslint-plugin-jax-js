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

        // Passed to non-jax function — consumes under move semantics, but no later use
        "const x = np.zeros([3]); foo(x);",

        // console.log is safe-listed — does NOT consume
        "const x = np.zeros([3]); console.log(x); x.dispose();",

        // Non-consuming member access passed to another function — x itself is not an argument
        "const x = np.zeros([3]); JSON.stringify(x.shape); x.dispose();",

        // expect() is safe-listed — does NOT consume
        "const x = np.zeros([3]); expect(x).toBeDefined(); x.dispose();",

        // assert() is safe-listed — does NOT consume
        "const x = np.zeros([3]); assert(x); x.dispose();",

        // assert.deepEqual is safe-listed (object method) — does NOT consume
        "const x = np.zeros([3]); assert.deepEqual(x, y); x.dispose();",

        // @jax-borrow comment directive suppresses consumption tracking
        {
          code: `
            const x = np.zeros([3]);
            myLogger(x); // @jax-borrow
            x.dispose();
          `,
        },

        // @jax-borrow on the line above the call
        {
          code: `
            const x = np.zeros([3]);
            // @jax-borrow
            myLogger(x);
            x.dispose();
          `,
        },

        // consume-and-reassign via user function: x = foo(x)
        "let x = np.zeros([3]); x = myHelper(x); x.dispose();",

        // super() call — not tracked as consuming
        {
          code: `
            class Foo extends Bar {
              constructor() {
                const x = np.zeros([3]);
                super();
                x.dispose();
              }
            }
          `,
        },

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

        // Else-branch termination: consumed in else that returns
        {
          code: `
            function f() {
              const x = np.zeros([3]);
              if (cond) {
                // do nothing
              } else {
                return x.reshape([1, 3]);
              }
              x.add(1);
            }
          `,
        },

        // Else-branch termination with throw
        {
          code: `
            function f() {
              const x = np.zeros([3]);
              if (err) {
                // skip
              } else {
                throw x.dispose();
              }
              x.reshape([3, 1]);
            }
          `,
        },

        // Ternary expression — consumption in one branch doesn't affect after
        {
          code: `
            const x = np.zeros([3]);
            const y = cond ? x.add(1) : other;
            x.dispose();
          `,
        },

        // Logical OR short-circuit — right side may not execute
        {
          code: `
            const x = np.zeros([3]);
            const y = fallback || x.add(1);
            x.dispose();
          `,
        },

        // Logical AND short-circuit — right side may not execute
        {
          code: `
            const x = np.zeros([3]);
            const y = cond && x.reshape([1, 3]);
            x.dispose();
          `,
        },

        // Nullish coalescing — right side may not execute
        {
          code: `
            const x = np.zeros([3]);
            const y = prev ?? x.add(1);
            x.dispose();
          `,
        },

        // Expression evaluation order: x.reshape([...x.shape])
        // Arguments are evaluated before the method call,
        // so x.shape is read while x is still alive.
        {
          code: `
            const x = np.zeros([3]);
            x.reshape([1, ...x.shape]);
          `,
        },

        // Expression evaluation order: accessing property in args of consuming call
        {
          code: `
            const x = np.zeros([3]);
            const y = x.reshape([x.shape[0], 1]);
          `,
        },

        // Expression evaluation order: kind:'argument' — np.multiply(x, x.shape)
        // All arguments are evaluated before the function call, so x.shape
        // is read while x is still alive.
        "const x = np.zeros([3]); foo(x, x.shape);",

        // Loop with consume-and-reassign — safe pattern
        "let x = np.zeros([3]); while (cond) { x = x.add(1); } x.dispose();",

        // JSON.stringify is safe-listed
        "const x = np.zeros([3]); JSON.stringify(x); x.dispose();",

        // Array.isArray is safe-listed
        "const x = np.zeros([3]); Array.isArray(x); x.dispose();",

        // Math.min is safe-listed
        "const x = np.zeros([3]); Math.min(x); x.dispose();",

        // Boolean() is safe-listed
        "const x = np.zeros([3]); Boolean(x); x.dispose();",
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

        // Use after passing to user function (move semantics)
        {
          code: "const x = np.zeros([3]); foo(x); x.shape;",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output: "const x = np.zeros([3]); foo(x.ref); x.shape;",
                },
              ],
            },
          ],
        },

        // Use after passing to user object method (move semantics)
        {
          code: "const x = np.zeros([3]); obj.process(x); x.shape;",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output:
                    "const x = np.zeros([3]); obj.process(x.ref); x.shape;",
                },
              ],
            },
          ],
        },

        // Double pass to user function — second is use-after-consume
        {
          code: "const x = np.zeros([3]); foo(x); bar(x);",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output: "const x = np.zeros([3]); foo(x.ref); bar(x);",
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

        // Ternary: both branches consume, then use after — still flagged
        // because the variable IS definitely consumed by one branch or the other.
        {
          code: "const x = np.zeros([3]); cond ? x.add(1) : x.sub(1); x.shape;",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output:
                    "const x = np.zeros([3]); cond ? x.add(1) : x.ref.sub(1); x.shape;",
                },
              ],
            },
          ],
        },

        // Ternary: both branches consume, use .shape after — flagged
        {
          code: [
            "const x = np.zeros([3]);",
            "const y = cond ? x.add(1) : x.mul(2);",
            "console.log(x.shape);",
          ].join("\n"),
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output: [
                    "const x = np.zeros([3]);",
                    "const y = cond ? x.add(1) : x.ref.mul(2);",
                    "console.log(x.shape);",
                  ].join("\n"),
                },
              ],
            },
          ],
        },

        // Use after consuming method with safe callee between
        // (safe callee doesn't reset consumption)
        {
          code: "const x = np.zeros([3]); x.add(1); console.log(x); x.shape;",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output:
                    "const x = np.zeros([3]); x.ref.add(1); console.log(x); x.shape;",
                },
              ],
            },
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output:
                    "const x = np.zeros([3]); x.ref.add(1); console.log(x); x.shape;",
                },
              ],
            },
          ],
        },

        // Loop without reassignment — genuine use-after-consume every iteration
        {
          code: "const x = np.zeros([3]); while (cond) { x.add(1); x.shape; }",
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output:
                    "const x = np.zeros([3]); while (cond) { x.ref.add(1); x.shape; }",
                },
              ],
            },
          ],
        },

        // Loop with write AFTER use — write doesn't help current iteration
        {
          code: [
            "let x = np.zeros([3]);",
            "while (cond) { x.add(1); x.shape; x = np.zeros([3]); }",
          ].join("\n"),
          errors: [
            {
              messageId: "useAfterConsume",
              suggestions: [
                {
                  messageId: "suggestRef",
                  output: [
                    "let x = np.zeros([3]);",
                    "while (cond) { x.ref.add(1); x.shape; x = np.zeros([3]); }",
                  ].join("\n"),
                },
              ],
            },
          ],
        },
      ],
    });
  });
});
