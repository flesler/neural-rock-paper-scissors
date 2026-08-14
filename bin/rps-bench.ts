#!/usr/bin/env npx tsx
import "../src/ai/node";
import { BRAIN_IDS, createBrain, runSeries } from "../src/ai";
import { FIXTURE_IDS, createFixture } from "../src/fixtures";

type Args = {
  rounds: number;
  seed: number;
  brains: string[];
  fixtures: string[];
  help?: boolean;
};

function parseArgs(argv: string[]): Args {
  const args: Args = {
    rounds: 400,
    seed: 1,
    brains: BRAIN_IDS.slice(),
    fixtures: FIXTURE_IDS.slice(),
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === "--rounds" && next) {
      args.rounds = parseInt(next, 10);
      i++;
    } else if (arg === "--seed" && next) {
      args.seed = parseInt(next, 10);
      i++;
    } else if (arg === "--brains" && next) {
      args.brains = next.split(",");
      i++;
    } else if (arg === "--fixtures" && next) {
      args.fixtures = next.split(",");
      i++;
    } else if (arg === "--help" || arg === "-h") {
      args.help = true;
    }
  }
  return args;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hash(text: string) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function pad(text: string, width: number) {
  return text.length >= width ? text : text + " ".repeat(width - text.length);
}

function pct(n: number) {
  return (100 * n).toFixed(1);
}

function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    console.log(
      "Usage: npm run bench -- [--rounds 400] [--seed 1] [--brains neural,patterns] [--fixtures cycle,always-rock]",
    );
    process.exit(0);
  }

  const nameWidth = Math.max(
    8,
    ...["pattern", ...args.fixtures].map((n) => n.length),
  );
  const colWidth = Math.max(12, ...args.brains.map((n) => n.length + 1));

  console.log(`AI win %  (${args.rounds} rounds, seed ${args.seed})\n`);
  let header = pad("pattern", nameWidth);
  for (const id of args.brains) header += "  " + pad(id, colWidth);
  console.log(header);

  const totals: Record<string, { win: number; score: number; n: number }> = {};
  for (const id of args.brains) totals[id] = { win: 0, score: 0, n: 0 };

  for (const fixtureId of args.fixtures) {
    let row = pad(fixtureId, nameWidth);
    for (const brainId of args.brains) {
      const rng = mulberry32(args.seed + hash(fixtureId + ":" + brainId));
      const brain = createBrain(brainId, { rng });
      const opponent = createFixture(fixtureId, { rng });
      const result = runSeries(brain, opponent, args.rounds);
      row += "  " + pad(pct(result.aiWinRate), colWidth);
      totals[brainId].win += result.aiWinRate;
      totals[brainId].score += result.score;
      totals[brainId].n++;
    }
    console.log(row);
  }

  let mean = pad("MEAN", nameWidth);
  for (const id of args.brains) {
    mean += "  " + pad(pct(totals[id].win / totals[id].n), colWidth);
  }
  console.log(mean);

  let score = pad("SCORE*", nameWidth);
  for (const id of args.brains) {
    score += "  " + pad(pct(totals[id].score / totals[id].n), colWidth);
  }
  console.log(score);
  console.log("\n* SCORE = (wins + 0.5 * ties) / rounds");
}

main();
