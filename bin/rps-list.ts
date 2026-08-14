#!/usr/bin/env npx tsx
import "../src/ai/node";
import { BRAIN_IDS } from "../src/ai";
import { FIXTURE_IDS } from "../src/fixtures";

console.log("brains:\n  " + BRAIN_IDS.join("\n  "));
console.log("\nfixtures:\n  " + FIXTURE_IDS.join("\n  "));
