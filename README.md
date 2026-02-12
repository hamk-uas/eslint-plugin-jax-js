# eslint-plugin-jax-js

ESLint rules for catching **array memory leaks** in [jax-js](https://github.com/ekzhang/jax-js) applications at edit time.

jax-js uses a consuming ownership model: most operations dispose their input arrays automatically.
If you create an array and never pass it to an operation or call `.dispose()`, the underlying
backend memory leaks. These lint rules catch the most common leak patterns statically, so you get
red squiggles in your editor instead of discovering leaks at runtime with `checkLeaks`.

## Installation

```bash
npm install --save-dev eslint-plugin-jax-js
```

The plugin requires **ESLint v9+** (flat config).

## Rules

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

// ✅ Good — x.ref is needed because x is used again later
const x = np.array([1, 2, 3]);
const data = x.ref.dataSync();
x.dispose();
```

**Autofix:** Removes `.ref` from the chain.

### `@jax-js/no-use-after-consume`

Warns when a variable holding a jax-js Array is used after being consumed by a method call
(like `.add()`, `.dispose()`) or by passing it to a jax-js function (like `np.multiply()`).
This is a **use-after-free bug**.

```ts
// ❌ Bad — x is consumed by .add(), then used again
const x = np.zeros([3]);
x.add(1);
x.shape; // use-after-consume!

// ✅ Good — use .ref to keep the array alive
const x = np.zeros([3]);
x.ref.add(1);
x.shape;
x.dispose();
```

**Suggestion fix:** Inserts `.ref` at the consuming site.

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

**Suggestion fix:** Adds `.dispose()` after last use.

> **Note:** All three rules include the hint *(Can be ignored inside jit.)* since
> jax-js's `jit()` traces array operations symbolically and does not actually consume arrays.

## Suppressing warnings (deliberate exceptions)

Sometimes a warning is intentional (e.g., you knowingly keep an array alive in a cache, or you're
inside a `jit()` callback where ownership rules are symbolic). In those cases, prefer disabling the
specific rule as locally as possible, and include a short reason.

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
import jaxJs from "eslint-plugin-jax-js";

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
import jaxJs from "eslint-plugin-jax-js";

export default [
  // ... your other config
  jaxJs.configs.recommended,
];
```

### Individual rules

Or enable rules individually:

```ts
import jaxJs from "eslint-plugin-jax-js";

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
import jaxJs from "eslint-plugin-jax-js";

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
4. Warnings from `eslint-plugin-jax-js` will appear inline in the editor.

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
and higher-level internal code that creates and consumes arrays. The only exception is code inside
`jit()` callbacks, which traces symbolically rather than actually consuming arrays (all rule
messages include the hint *"Can be ignored inside jit."*).

To enable it in the jax-js monorepo, add to the root `eslint.config.ts`:

```ts
import jaxJs from "eslint-plugin-jax-js";

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

## Maintainer Guide

This section is for maintainers of the plugin itself — if you're just using the plugin
in your project, you can stop reading here.

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

#### 6. Bump version and publish

Follow the publishing steps below.

### Publishing

#### Prerequisites

- [Node.js](https://nodejs.org/) (v18+) and npm.
- An [npm account](https://www.npmjs.com/signup) with publish access to `eslint-plugin-jax-js`.
- Authenticated locally: `npm login`.

#### Steps

```bash
# 1. Make sure tests pass and there are no type errors
npm test
npx tsc --noEmit

# 2. Bump the version (choose patch / minor / major as appropriate)
npm version patch   # e.g., 0.1.0 → 0.1.1
# This updates package.json and creates a git tag.
# Remember to also update the version string in src/index.ts to match.

# 3. Push the commit and tag
git push && git push --tags

# 4. Publish to npm
npm publish

# 5. (Optional) Create a GitHub release
#    Go to https://github.com/hamk-uas/eslint-plugin-jax-js/releases/new
#    Select the tag, write release notes summarizing changes.
```

#### Version numbering

| Change | Bump |
|--------|------|
| New jax-js methods added to API surface | **patch** |
| New lint rule or rule behavior change | **minor** |
| Breaking: removed rule, changed defaults, ownership model change | **major** |

## License

MIT
