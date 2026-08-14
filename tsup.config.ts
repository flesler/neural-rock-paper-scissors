import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/rps.ts"],
  format: ["iife"],
  platform: "browser",
  outDir: "js",
  outExtension: () => ({ js: ".js" }),
  clean: false,
  minify: false,
  sourcemap: false,
  target: "es2018",
  splitting: false,
  treeshake: true,
  external: ["neataptic"],
});
