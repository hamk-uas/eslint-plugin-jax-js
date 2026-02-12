import { RuleTester } from "eslint";
import { describe, it } from "vitest";

import rule from "../src/rules/no-unnecessary-ref";

const ruleTester = new RuleTester({
  languageOptions: {
    ecmaVersion: 2022,
    sourceType: "module",
  },
});

describe("no-unnecessary-ref", () => {
  it("passes rule tester", () => {
    ruleTester.run("no-unnecessary-ref", rule, {
      valid: [
        // .ref needed because x is used again later
        {
          code: `
            const x = array([1, 2, 3]);
            const data = x.ref.dataSync();
            x.dispose();
          `,
        },
        // .ref needed because x is passed to another op
        {
          code: `
            const x = array([1, 2, 3]);
            const y = x.ref.add(other);
            const z = x.mul(other);
          `,
        },
        // .ref on non-identifier (can't track, no warning)
        {
          code: `
            const data = getArray().ref.dataSync();
          `,
        },
        // .ref used, then variable used later in a return
        {
          code: `
            function f() {
              const x = array([1, 2, 3]);
              const y = x.ref.add(other);
              return x;
            }
          `,
        },
        // .ref on a variable that is used later as a function argument
        {
          code: `
            const x = array([1, 2, 3]);
            const y = x.ref.add(z);
            consume(x);
          `,
        },
        // Multiple .ref uses where variable is consumed at the end
        {
          code: `
            const x = array([1, 2, 3]);
            const a = x.ref.add(y);
            const b = x.ref.mul(z);
            const c = x.sub(w);
          `,
        },
        // .ref needed because later use is also .ref (multiple consuming chains)
        {
          code: `
            const x = array([1, 2, 3]);
            const a = x.ref.dataSync();
            const b = x.ref.dataSync();
            x.dispose();
          `,
        },
        // .ref needed: first topK consumes, second uses x again
        {
          code: `
            const x = np.array([[3, 1, 4], [1, 5, 9]]);
            let [v, i] = lax.topK(x.ref, 2);
            [v, i] = lax.topK(x, 1, 0);
          `,
        },
        // .ref needed: init() consumes params, update() uses it again
        {
          code: `
            const params = np.array([1.0, 2.0]);
            const state = transform.init(params.ref);
            const [u, s] = transform.update(updates, state, params);
          `,
        },
        // .ref inside a closure passed to grad — variable captured from outer scope
        // (closure may be invoked multiple times, .ref keeps array alive)
        {
          code: `
            const L = np.array([[2, 0], [1, 3]]);
            const f = (b) => lax.linalg.triangularSolve(L.ref, b);
            const db = grad(f)(b);
          `,
        },
        // .ref inside arrow function capturing outer variable — multi-invocation safety
        {
          code: `
            const a = np.array([[1, 2], [3, 4]]);
            const f = (b) => np.linalg.lstsq(a.ref, b);
            const db = grad(f)(b);
          `,
        },
        // .ref as return value in a function passed as a callback arg — intentional cloning
        {
          code: `
            items.forEach(function(x) {
              results.push(x.ref);
            });
          `,
        },
        // .ref inside .map — intentional cloning per element
        {
          code: `
            const refs = items.map((t) => t.ref);
          `,
        },
        // .ref in for-of loop — intentional rc-bump on borrowed reference
        {
          code: `
            for (const t of consts) t.ref;
          `,
        },
        // .ref in for-in loop — intentional rc-bump on borrowed reference
        {
          code: `
            for (const t in consts) t.ref;
          `,
        },
        // .ref needed because variable is consumed BEFORE .ref in same expression
        // e.g., equal(a, min(a.ref, axis))
        {
          code: `
            const a = array([1, 2, 3]);
            const isMax = equal(a, min(a.ref));
          `,
        },
        // Same pattern — .ref needed for double-consumption in one expression
        {
          code: `
            const x = array([1, 2, 3]);
            const result = subtract(x, remainder(x.ref, y.ref));
          `,
        },
      ],

      invalid: [
        // Standalone .ref on a local variable — x is never consumed
        {
          code: `
            const x = array([1, 2, 3]);
            const y = x.ref;
          `,
          errors: [{ messageId: "unnecessaryRef" }],
          output: `
            const x = array([1, 2, 3]);
            const y = x;
          `,
        },
        // .ref as return value in a function declaration — x leaks
        {
          code: `
            function f(x) {
              return x.ref;
            }
          `,
          errors: [{ messageId: "unnecessaryRef" }],
          output: `
            function f(x) {
              return x;
            }
          `,
        },
        // Real-world bug pattern: owned param, .ref with no subsequent consumption
        {
          code: `
            const fdot = (x) => {
              const y = jvp(f, [x.ref], [1]);
              return y;
            };
          `,
          errors: [{ messageId: "unnecessaryRef" }],
          output: `
            const fdot = (x) => {
              const y = jvp(f, [x], [1]);
              return y;
            };
          `,
        },
        // Simple case: .ref.dataSync() with no subsequent use
        {
          code: `
            const x = array([1, 2, 3]);
            const data = x.ref.dataSync();
          `,
          errors: [{ messageId: "unnecessaryRef" }],
          output: `
            const x = array([1, 2, 3]);
            const data = x.dataSync();
          `,
        },
        // .ref.js() on last use
        {
          code: `
            const x = array([1, 2, 3]);
            const val = x.ref.js();
          `,
          errors: [{ messageId: "unnecessaryRef" }],
          output: `
            const x = array([1, 2, 3]);
            const val = x.js();
          `,
        },
        // .ref.item() on last use
        {
          code: `
            const x = array([1, 2, 3]);
            const val = x.ref.item();
          `,
          errors: [{ messageId: "unnecessaryRef" }],
          output: `
            const x = array([1, 2, 3]);
            const val = x.item();
          `,
        },
        // .ref.add() where x is never used again
        {
          code: `
            const x = array([1, 2, 3]);
            const y = x.ref.add(z);
          `,
          errors: [{ messageId: "unnecessaryRef" }],
          output: `
            const x = array([1, 2, 3]);
            const y = x.add(z);
          `,
        },
        // .ref followed only by non-consuming .shape access
        // (no autofix — removing .ref would break later .shape access)
        {
          code: `
            const x = array([1, 2, 3]);
            const y = x.ref.add(z);
            console.log(x.shape);
          `,
          errors: [
            {
              messageId: "unnecessaryRefOnlyProps",
              suggestions: [
                {
                  messageId: "addDispose",
                  output: `
            const x = array([1, 2, 3]);
            const y = x.ref.add(z);
            console.log(x.shape);
x.dispose();
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
