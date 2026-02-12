# @hamk-uas/eslint-plugin-jax-js

Community ESLint plugin for catching **array memory leaks** in [jax-js](https://github.com/ekzhang/jax-js) applications at edit time. Not affiliated with or endorsed by the jax-js project.

jax-js uses a consuming ownership model: most operations dispose their input arrays automatically.
If you create an array and never pass it to an operation or call `.dispose()`, the underlying
backend memory leaks. Calling `.ref` bumps the reference count so an array survives past a
consuming operation, but forgetting to `.dispose()` the extra reference also leaks.
These lint rules catch the most common leak patterns statically, so you get
red squiggles in your editor instead of discovering leaks at runtime.

## Design philosophy

The plugin warns about ownership violations **everywhere**, including inside `jit()`
callbacks. jax-js's `jit()` traces operations symbolically and does not actually consume
arrays at trace time, so the warnings are technically false positives there. We warn
anyway, on purpose:

> **Write code that is ownership-correct in both jit and eager mode.**

If your code only works under `jit()` tracing but leaks or crashes in eager mode, you
cannot freely switch between the two — and switching is something you do often during
development (debugging, profiling, adding logging). By keeping your code
ownership-correct at all times, `jit()` becomes a pure performance optimization you can
add or remove without changing program semantics.

If a warning doesn't apply to your situation, you have two options:

- **`// @jax-borrow`** — for calls to your own non-consuming helpers
  (see [`no-use-after-consume`](#jax-jsno-use-after-consume)).
- **`eslint-disable`** — for anything else (long-lived caches, intentional leaks, etc.).
  See [Suppressing warnings](#suppressing-warnings-deliberate-exceptions).

> **A note for the jax-js ecosystem:** The ideal long-term solution would be for jax-js
> itself to enforce ownership rules during `jit()` tracing — i.e., raise an error when
> traced code uses an array after it has been consumed, just as eager mode does. That
> would make ownership-correct code a hard requirement rather than a best practice, and
> this lint plugin would then have zero false positives inside `jit()`. Until that
> happens, the plugin fills the gap statically.

## Installation

```bash
npm install --save-dev github:hamk-uas/eslint-plugin-jax-js
```

This installs from the latest commit on `main`. To pin a specific release:

```bash
npm install --save-dev github:hamk-uas/eslint-plugin-jax-js#v0.1.0
```

See [releases](https://github.com/hamk-uas/eslint-plugin-jax-js/releases) for available tags.

The plugin requires **ESLint v9+** (flat config). It ships as TypeScript source
(no build step needed) — ESLint v9 loads it via its built-in `jiti` transpiler.

## Rules

Each rule reports warnings and can offer automatic code changes through ESLint:

- **Autofix** (ESLint *fix*) — applied automatically on save or via `eslint --fix`.
  The rule has verified that the change is safe for the specific code it flagged.
- **Suggestion** (ESLint *suggestion*) — shown as a lightbulb (💡) quick fix in your editor.
  Requires manual confirmation because the change may need review (e.g., inserting `.ref`
  means you also need a matching `.dispose()`).

### `@jax-js/no-unnecessary-ref`

Warns when `.ref` is used on a variable whose last usage is the `.ref` chain itself. This is a
**guaranteed leak**: `.ref` bumps the reference count, the chained consuming method decrements it
back, but the original variable still holds `rc=1` and nobody will ever dispose it.

```ts
// ❌ Bad — x leaks (rc stays at 1 after dataSync disposes the .ref copy)
const x = np.array([1, 2, 3]);
const data = x.ref.dataSync();

// ✅ Good — x is consumed directly
const x = np.array([1, 2, 3]);
const data = x.dataSync();

// ✅ Good — x.ref is needed because x is used meaningfully afterward
const x = np.array([1, 2, 3]);
const data = x.ref.dataSync();
console.log(x.shape);
x.dispose();
```

**Autofix:** Removes `.ref` from the chain.

### `@jax-js/no-use-after-consume`

Warns when a variable holding a jax-js Array is used after being consumed.
Consumption is detected from:

1. **Method calls** on the array: `.add()`, `.dispose()`, etc.
2. **Passing the array to any function**: `np.multiply(x, y)`, `myHelper(x)`,
   `obj.process(x)` — under move semantics, passing an array transfers ownership.

Known non-consuming callees (`console.log`, `expect`, etc.)
are automatically excluded. For your own non-consuming helpers, add a
`// @jax-borrow` comment.

**Suggestion:** Inserts `.ref` before the consuming call
(e.g., `x.add(1)` → `x.ref.add(1)`), so the array stays alive for later use.

```ts
// ❌ Bad — x is consumed by .add(), then used again
const x = np.zeros([3]);
x.add(1);
x.shape; // use-after-consume!

// ❌ Bad — foo(x) consumes x under move semantics
const x = np.zeros([3]);
foo(x);
x.shape; // use-after-consume!

// ✅ Good — use .ref to keep the array alive
const x = np.zeros([3]);
x.ref.add(1);
x.shape;
x.dispose();

// ✅ Good — @jax-borrow marks a non-consuming call
const x = np.zeros([3]);
myLogger(x); // @jax-borrow
x.dispose();
```

### `@jax-js/require-consume`

Warns when an array stored in a variable is never consumed — never passed to a consuming operation,
returned, yielded, or explicitly disposed. Accessing only non-consuming properties like `.shape`,
`.dtype`, `.ndim`, `.size`, `.device`, or `.refCount` does not count as consumption.

```ts
// ❌ Bad — x is created but never consumed or disposed
const x = np.array([1, 2, 3]);
console.log(x.shape);

// ✅ Good
const x = np.array([1, 2, 3]);
console.log(x.shape);
x.dispose();
```

**Suggestion:** Adds `.dispose()` after last use.

> **Note:** All three rules include the hint *(Can be ignored inside jit.)* in their
> messages. See [Design philosophy](#design-philosophy) for why the plugin warns inside
> `jit()` anyway — in short, it encourages code that works in both jit and eager mode.

## Suppressing warnings (deliberate exceptions)

Sometimes a warning is intentional (e.g., you knowingly keep an array alive in a cache).
In those cases, prefer disabling the specific rule as locally as possible, and include a
short reason.

```ts
// eslint-disable-next-line @jax-js/no-use-after-consume -- inside jit(): traced, not executed
const y = jit((x) => x.add(1))(x);
```

Disable for a single line:

```ts
// eslint-disable-next-line @jax-js/require-consume -- intentionally leaked until process exit
const cached = np.zeros([1024]);
```

Disable for a small block:

```ts
/* eslint-disable @jax-js/require-consume -- constructing a global cache */
const CACHE = new Map();
function getOrCreate(key) {
  if (!CACHE.has(key)) CACHE.set(key, np.zeros([3]));
  return CACHE.get(key);
}
/* eslint-enable @jax-js/require-consume */
```

Or turn a rule off (or change severity) in your ESLint config:

```ts
import jaxJs from "@hamk-uas/eslint-plugin-jax-js";

export default [
  jaxJs.configs.recommended,
  {
    rules: {
      "@jax-js/require-consume": "off",
    },
  },
];
```

## Setup

### Recommended config (all rules as warnings)

Add the plugin to your flat ESLint config:

```ts
// eslint.config.ts (or eslint.config.js)
import jaxJs from "@hamk-uas/eslint-plugin-jax-js";

export default [
  // ... your other config
  jaxJs.configs.recommended,
];
```

### Individual rules

Or enable rules individually:

```ts
import jaxJs from "@hamk-uas/eslint-plugin-jax-js";

export default [
  {
    plugins: { "@jax-js": jaxJs },
    rules: {
      "@jax-js/no-unnecessary-ref": "warn",
      "@jax-js/no-use-after-consume": "warn",
      "@jax-js/require-consume": "warn",
    },
  },
];
```

### Limiting to specific files

If your project mixes jax-js code with other code, you can scope the rules to specific directories:

```ts
import jaxJs from "@hamk-uas/eslint-plugin-jax-js";

export default [
  {
    files: ["src/math/**/*.ts", "src/ml/**/*.ts"],
    ...jaxJs.configs.recommended,
  },
];
```

## IDE Integration

These lint rules are designed to give you immediate feedback as you write jax-js code. Any editor or IDE that supports ESLint will show warnings inline (red/yellow squiggles) and offer quick-fix suggestions.

### VS Code

1. Install the [ESLint extension](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint) (`dbaeumer.vscode-eslint`).
2. The extension auto-detects your `eslint.config.ts` — no extra configuration needed.
3. You will see inline warnings for leak patterns, and **Code Actions** (💡 lightbulb) to apply autofixes and suggestions.

Optional settings for a better experience (add to `.vscode/settings.json`):

```jsonc
{
  // Lint on save for fast feedback
  "eslint.run": "onSave",

  // Validate TypeScript and JavaScript files
  "eslint.validate": ["typescript", "javascript"],

  // Auto-fix fixable rules on save (e.g., removes unnecessary .ref)
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit"
  }
}
```

### WebStorm / IntelliJ IDEA

1. Go to **Settings → Languages & Frameworks → JavaScript → Code Quality Tools → ESLint**.
2. Select **Automatic ESLint configuration** (or point to your config manually).
3. Check **Run eslint --fix on save** if you want autofixes applied automatically.
4. Warnings from `@hamk-uas/eslint-plugin-jax-js` will appear inline in the editor.

### Neovim

If you use [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig) with the ESLint language server:

```lua
-- In your Neovim LSP config
require('lspconfig').eslint.setup({
  -- ESLint will automatically pick up eslint.config.ts
})
```

Or with [none-ls](https://github.com/nvimtools/none-ls.nvim) / [efm-langserver](https://github.com/mattn/efm-langserver), configure ESLint as a diagnostics source.

### Sublime Text

Install [SublimeLinter](https://github.com/SublimeLinter/SublimeLinter) and [SublimeLinter-eslint](https://github.com/SublimeLinter/SublimeLinter-eslint). The plugin will be picked up automatically from your ESLint config.

### Command Line

You can also run the linter from the command line as part of your CI/CD pipeline:

```bash
npx eslint .
```

Or add it as a script in your `package.json`:

```json
{
  "scripts": {
    "lint": "eslint ."
  }
}
```

## Using in jax-js itself

This plugin is also useful for developing jax-js itself — it catches leaks in tests, examples,
and higher-level internal code that creates and consumes arrays. Code inside `jit()` callbacks
will also trigger warnings even though tracing doesn't actually consume arrays — this is
intentional (see [Design philosophy](#design-philosophy)).

To enable it in the jax-js monorepo, add to the root `eslint.config.ts`:

```ts
import jaxJs from "@hamk-uas/eslint-plugin-jax-js";

export default defineConfig([
  // ...existing config
  jaxJs.configs.recommended,
]);
```

## How It Works

The plugin uses **heuristic-based static analysis** to identify variables that hold jax-js Arrays.
It recognizes:

- **Factory calls**: `array()`, `np.zeros()`, `np.ones()`, `np.eye()`, `np.arange()`, etc.
- **Array-returning methods**: `.add()`, `.reshape()`, `.transpose()`, `.mul()`, etc.
- **Consuming terminal methods**: `.js()`, `.dataSync()`, `.data()`, `.item()`, `.dispose()`
- **Non-consuming properties**: `.shape`, `.dtype`, `.ndim`, `.size`, `.device`, `.refCount`
- **jax-js namespace calls**: `np.*()`, `lax.*()`, `nn.*()`, `random.*()`, etc.

The rules understand several patterns:
- **`.ref` bumps reference count** — needed when an array must survive past a consuming call.
- **Consume-and-reassign** (`x = x.add(1)`) is recognized and does not trigger false positives.
- **Mutually exclusive if-branches** (e.g., early return) are handled correctly.
- **Closures** (e.g., `expect(() => ...).toThrow()`) are conservatively skipped.
- **Borrowed bindings** (callback params, for-of vars) with `.ref` are treated as intentional cloning.

### Design decisions & known limitations

- **`require-consume` treats `.ref` capture as consuming.** When you write
  `const y = x.ref`, the rule considers `x` consumed — you've opted into manual
  reference counting, and flagging `x` would be noisy. The trade-off: if `y` is
  never disposed, the leak won't be caught by `require-consume` on `x` (though
  it will be caught on `y` if `y` is also tracked).

- **TypeScript source, no build step.** The plugin ships raw `.ts` files and relies
  on ESLint v9's `jiti` transpiler. This simplifies development and contribution
  but means the plugin won't work with ESLint v8 or custom loaders that don't
  support TypeScript.

- **Heuristic-based, no import tracking.** The rules identify jax-js arrays by
  recognizing factory calls (`np.zeros()`), method names (`.add()`, `.reshape()`),
  and namespace prefixes (`np.*`, `lax.*`). They do not resolve imports, so:
  - All three rules treat **any function argument pass as consuming** under
    move semantics. `no-use-after-consume` additionally maintains a safe-list
    of known non-consuming callees (`console.log`, `expect`,
    etc.) to avoid false positives from debugging/testing code.
  - For custom non-consuming helpers, use the `// @jax-borrow` comment directive
    to suppress consumption tracking on that call.

  This is conservative overall — it avoids false positives at the cost of
  occasional false negatives for unusual import patterns.

## API Surface & Compatibility

The plugin ships with an API surface list derived from **jax-js v0.1.9**
([commit f900c28](https://github.com/ekzhang/jax-js/commit/f900c282880d61f1b9694208e731d6726919c5b6)),
plus a conservative set of JAX functions likely to be added soon (marked with
`// Future-proof` comments in `src/shared.ts`).

**Compatibility approach:**

- The plugin uses its own independent semver versioning.
- `peerDependencies` declares the known-compatible jax-js range (`>=0.1.0 <0.3.0`).
  It's marked optional since you may lint jax-js code without `@jax-js/jax` being a
  direct dependency of your lint config.
- **Older jax-js** (back to 0.1.0): fully compatible — the plugin simply knows about
  more functions than exist, which causes no false positives.
- **Newer jax-js** that adds new methods: the plugin won't break, but it won't lint
  code using unknown new functions (missing coverage, not false positives). Update
  `src/api-surface.generated.ts` and the manual sets in `src/shared.ts` to restore
  full coverage.
- **Breaking ownership model changes** in jax-js: would require a new major version of
  this plugin and a bumped peer dep range.

If jax-js adds new methods, update `src/api-surface.generated.ts` accordingly.

## Contributing

Contributions are welcome! Please:

- **Report bugs** by [opening an issue](https://github.com/hamk-uas/eslint-plugin-jax-js/issues/new)
  with a minimal code snippet that triggers the wrong behavior.
- **Submit fixes** as pull requests. Include a test case that reproduces the bug
  (see [Fixing bugs](#fixing-bugs) below).

### Development setup

After cloning the repo, install dependencies and enable the pre-commit hook:

```bash
npm install
git config core.hooksPath .githooks
```

This runs tests and type-checking before every commit. The hook lives in
`.githooks/pre-commit` and requires no extra dependencies.

### Project structure

- `src/rules/` — one file per lint rule.
- `src/shared.ts` — shared detection logic (array-init heuristics, scope helpers, consuming-site detection).
- `src/api-surface.generated.ts` — extracted method/getter lists from jax-js.
- `test/` — one test file per rule, using ESLint's `RuleTester`.

### Testing a fix in your own project

If you notice a bug while using the plugin in another project, you can point that
project at your local clone or development branch without waiting for a published release.

**From a local clone:**

```bash
# In your project (not the plugin repo):
npm install --save-dev /path/to/jax-js-eslint-plugin
```

**From a git branch (e.g., a PR branch):**

```bash
npm install --save-dev github:hamk-uas/eslint-plugin-jax-js#my-fix-branch
```

Both methods make ESLint use the development version immediately. Since the plugin
ships raw TypeScript (loaded via ESLint's `jiti` transpiler), no build step is needed.

Once the fix is released, switch back to the tagged version:

```bash
npm install --save-dev github:hamk-uas/eslint-plugin-jax-js#v0.1.1
```

### Fixing bugs

1. **Reproduce** — add a failing test case to the relevant file in `test/`. Each test
   file uses ESLint's `RuleTester`, so a new entry in `valid` or `invalid` is usually
   enough to capture the bug.
2. **Fix** — the rules live in `src/rules/`. Shared detection logic is in `src/shared.ts`.
3. **Verify** — run `npm test` and `npx tsc --noEmit` to confirm the fix and catch regressions.

#### Common bug categories

| Symptom | Likely location |
|---------|-----------------|
| False positive (warning on valid code) | The rule is too aggressive — check the consuming/non-consuming classification in `shared.ts` or the rule's visitor logic. |
| False negative (no warning on buggy code) | The pattern isn't recognized — likely missing from `isArrayInit()`, factory/method sets, or the namespace list. |
| Wrong autofix / suggestion | The `fix` or `suggest` callback in the rule — test the `output` field in `RuleTester`. |
| Crash / exception in the rule | Usually a missing null-check on an AST node — add a guard and a regression test. |

#### Tips

- **Keep test-first discipline** — always add the failing test *before* writing the fix.
  This ensures the bug is actually reproduced and prevents regressions.
- If a fix changes user-visible behavior (e.g., a rule now warns in a case it previously
  allowed), mention it in the PR description so maintainers know what to expect.

## Maintainer Guide

This section is for maintainers who create releases.

### Releasing a bug fix

1. Merge the PR with the fix.
2. **Version & tag** — bug fixes are always a **patch** bump. See [releasing](#releasing) below.
3. **Create a GitHub release** with notes describing the fix.

### Updating for a New jax-js Version

When a new version of [jax-js](https://github.com/ekzhang/jax-js) is released, follow these steps
to bring the plugin up to date.

#### 1. Review the upstream changes

Open the jax-js release notes or diff the `Array` / `Tracer` classes to identify:

- **New getters** (non-consuming properties like `.shape`).
- **New methods** (consuming operations like `.add()`).
- **Removed or renamed** members.
- **Changes to the ownership model** (rare — would require a major plugin version bump).

#### 2. Update the API surface file

Edit `src/api-surface.generated.ts`:

- Add new getters to `EXTRACTED_GETTERS`.
- Add new methods to `EXTRACTED_METHODS`.
- Remove any members that no longer exist.
- Update the version comment at the top of the file.

#### 3. Update manual sets in `src/shared.ts`

Most sets are derived automatically, but three require manual curation:

| Set | When to update |
|-----|----------------|
| `CONSUMING_TERMINAL_METHODS` | A new method returns a non-Array value (e.g., a new serialization method). |
| `UNAMBIGUOUS_ARRAY_METHODS` | A new method name doesn't collide with standard JS APIs. |
| `NON_CONSUMING_METHODS` | A new method does not consume the array (like `blockUntilReady`). |

Also check `ARRAY_FACTORY_NAMES` if jax-js adds new top-level factory functions.

#### 4. Update peer dependency range

In `package.json`, widen or bump the `@jax-js/jax` peer dependency range:

```jsonc
"peerDependencies": {
  "@jax-js/jax": ">=0.1.0 <0.4.0"  // ← adjust upper bound
}
```

#### 5. Add tests for new patterns

If new methods introduce novel consuming/non-consuming behaviors, add test cases to the
relevant test files in `test/`.

#### 6. Bump version and release

Follow the releasing steps below.

### Releasing

#### Steps

```bash
# 1. Make sure tests pass and there are no type errors
npm test
npx tsc --noEmit

# 2. Bump the version (choose patch / minor / major as appropriate)
npm version patch   # e.g., 0.1.0 → 0.1.1
# This updates package.json and creates a git tag (v0.1.1).
# Remember to also update the version string in src/index.ts to match.

# 3. Push the commit and tag
git push && git push --tags

# 4. Create a GitHub release
#    Go to https://github.com/hamk-uas/eslint-plugin-jax-js/releases/new
#    Select the tag, write release notes summarizing changes.
```

Users install specific tags, so after releasing they can upgrade with:

```bash
npm install --save-dev github:hamk-uas/eslint-plugin-jax-js#v0.1.1
```

#### Version numbering

| Change | Bump |
|--------|------|
| Documentation only (README, comments) | **no bump** — users on `main` get it automatically |
| Bug fix (false positive/negative, crash, wrong autofix) | **patch** |
| New jax-js methods added to API surface | **patch** |
| New lint rule or rule behavior change | **minor** |
| Breaking: removed rule, changed defaults, ownership model change | **major** |

#### Future: npm publishing

If the user base grows, consider publishing to the npm registry for easier version
management. The package.json is already set up for scoped publishing — just add
`"publishConfig": { "access": "public" }` back, run `npm login`, and `npm publish`.

## License

MIT
