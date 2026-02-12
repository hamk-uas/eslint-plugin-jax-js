import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // ESLint plugin tests run in Node.js, not the browser
    browser: { enabled: false },
    setupFiles: [],
    passWithNoTests: true,
  },
});
