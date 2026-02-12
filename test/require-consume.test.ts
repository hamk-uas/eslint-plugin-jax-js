import { RuleTester } from "eslint";
import { describe, it } from "vitest";

import rule from "../src/rules/require-consume";

const ruleTester = new RuleTester({
  languageOptions: {
    ecmaVersion: 2022,
    sourceType: "module",
  },
});

describe("require-consume", () => {
  it("passes rule tester", () => {
    ruleTester.run("require-consume", rule, {
      valid: [
        // Array consumed by .js()
        {
          code: `
            const x = array([1, 2, 3]);
            const val = x.js();
          `,
        },
        // Array consumed by .dataSync()
        {
          code: `
            const x = array([1, 2, 3]);
            const val = x.dataSync();
          `,
        },
        // Array consumed by .dispose()
        {
          code: `
            const x = array([1, 2, 3]);
            console.log(x.shape);
            x.dispose();
          `,
        },
        // Array consumed by passing to another op
        {
          code: `
            const x = array([1, 2, 3]);
            const val = x.add(z).js();
          `,
        },
        // Array consumed by being passed as function argument
        {
          code: `
            const x = array([1, 2, 3]);
            f(x);
          `,
        },
        // Array consumed by being returned
        {
          code: `
            function f() {
              const x = array([1, 2, 3]);
              return x;
            }
          `,
        },
        // Array consumed by being assigned to another variable
        {
          code: `
            const x = array([1, 2, 3]);
            const y = x;
          `,
        },
        // Not an array-producing call — no warning
        {
          code: `
            const x = someOtherFunction();
            console.log(x.shape);
          `,
        },
        // Array consumed by .item()
        {
          code: `
            const x = array([5]);
            const val = x.item();
          `,
        },
        // Array consumed by method chain (unambiguous jax-js method)
        {
          code: `
            const x = np.zeros([3, 3]);
            const val = x.reshape([9]).js();
          `,
        },
        // Array consumed in array expression
        {
          code: `
            const x = np.array([1]);
            const list = [x];
          `,
        },
        // Array consumed in object property
        {
          code: `
            const x = np.array([1]);
            const obj = { arr: x };
          `,
        },
        // Result of .add() consumed
        {
          code: `
            const y = x.add(z);
            const val = y.js();
          `,
        },
        // Non-array variable — not flagged even if only property-accessed
        {
          code: `
            const x = someOtherFunction();
            console.log(x.shape);
            console.log(x.dtype);
          `,
        },
        // Result of .ref consumed
        {
          code: `
            const y = x.ref;
            y.dispose();
          `,
        },
      ],

      invalid: [
        // Array created but never used
        {
          code: `
            const x = array([1, 2, 3]);
          `,
          errors: [
            {
              messageId: "neverConsumed",
              suggestions: [
                {
                  messageId: "suggestDispose",
                  output: `
            const x = array([1, 2, 3]);
x.dispose();
          `,
                },
              ],
            },
          ],
        },
        // Array used only for .shape
        {
          code: `
            const x = array([1, 2, 3]);
            console.log(x.shape);
          `,
          errors: [
            {
              messageId: "onlyPropertyAccess",
              suggestions: [
                {
                  messageId: "suggestDispose",
                  output: `
            const x = array([1, 2, 3]);
            console.log(x.shape);
x.dispose();
          `,
                },
              ],
            },
          ],
        },
        // Array used only for .dtype and .ndim
        {
          code: `
            const x = array([1, 2, 3]);
            console.log(x.dtype);
            console.log(x.ndim);
          `,
          errors: [
            {
              messageId: "onlyPropertyAccess",
              suggestions: [
                {
                  messageId: "suggestDispose",
                  output: `
            const x = array([1, 2, 3]);
            console.log(x.dtype);
            console.log(x.ndim);
x.dispose();
          `,
                },
              ],
            },
          ],
        },
        // np.zeros never consumed
        {
          code: `
            const x = np.zeros([3, 3]);
            console.log(x.shape);
          `,
          errors: [
            {
              messageId: "onlyPropertyAccess",
              suggestions: [
                {
                  messageId: "suggestDispose",
                  output: `
            const x = np.zeros([3, 3]);
            console.log(x.shape);
x.dispose();
          `,
                },
              ],
            },
          ],
        },
        // np.ones never consumed
        {
          code: `
            const x = np.ones([2]);
          `,
          errors: [
            {
              messageId: "neverConsumed",
              suggestions: [
                {
                  messageId: "suggestDispose",
                  output: `
            const x = np.ones([2]);
x.dispose();
          `,
                },
              ],
            },
          ],
        },
        // np.eye never consumed
        {
          code: `
            const x = np.eye(3);
            console.log(x.size);
          `,
          errors: [
            {
              messageId: "onlyPropertyAccess",
              suggestions: [
                {
                  messageId: "suggestDispose",
                  output: `
            const x = np.eye(3);
            console.log(x.size);
x.dispose();
          `,
                },
              ],
            },
          ],
        },
        // Result of .reshape() never consumed
        {
          code: `
            const y = x.reshape([3, 3]);
          `,
          errors: [
            {
              messageId: "neverConsumed",
              suggestions: [
                {
                  messageId: "suggestDispose",
                  output: `
            const y = x.reshape([3, 3]);
y.dispose();
          `,
                },
              ],
            },
          ],
        },
        // Result of .transpose() never consumed
        {
          code: `
            const y = x.transpose();
            console.log(y.shape);
          `,
          errors: [
            {
              messageId: "onlyPropertyAccess",
              suggestions: [
                {
                  messageId: "suggestDispose",
                  output: `
            const y = x.transpose();
            console.log(y.shape);
y.dispose();
          `,
                },
              ],
            },
          ],
        },
        // Result of .reshape() only accessed for .shape
        {
          code: `
            const y = x.reshape([3, 3]);
            console.log(y.shape);
          `,
          errors: [
            {
              messageId: "onlyPropertyAccess",
              suggestions: [
                {
                  messageId: "suggestDispose",
                  output: `
            const y = x.reshape([3, 3]);
            console.log(y.shape);
y.dispose();
          `,
                },
              ],
            },
          ],
        },
      ],
    });
  });
});
