#!/usr/bin/env npx tsx
import "../src/ai/node";
import {
  BRAIN_IDS,
  SESSION_ROUNDS,
  WARMUP_BUCKETS,
  createBrain,
  runSeries,
} from "../src/ai";
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
    rounds: 80,
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

function printTable(
  title: string,
  fixtures: string[],
  brains: string[],
  cell: (fixture: string, brain: string) => string,
  footer?: Array<[string, (brain: string) => string]>,
) {
  const nameWidth = Math.max(
    8,
    ...["pattern", ...fixtures].map((n) => n.length),
  );
  const colWidth = Math.max(12, ...brains.map((n) => n.length + 1));
  console.log(title);
  let header = pad("pattern", nameWidth);
  for (const id of brains) header += "  " + pad(id, colWidth);
  console.log(header);
  for (const fixtureId of fixtures) {
    let row = pad(fixtureId, nameWidth);
    for (const brainId of brains) row += "  " + cell(fixtureId, brainId);
    console.log(row);
  }
  if (footer) {
    for (const [label, fn] of footer) {
      let row = pad(label, nameWidth);
      for (const brainId of brains) row += "  " + fn(brainId);
      console.log(row);
    }
  }
  console.log("");
}

function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    console.log(
      "Usage: npm run bench -- [--rounds 80] [--seed 1] [--brains iocaine,genetic] [--fixtures cycle,random]",
    );
    process.exit(0);
  }

  type Cell = ReturnType<typeof runSeries>;
  const results: Record<string, Record<string, Cell>> = {};
  const totals: Record<
    string,
    { win: number; score: number; session: number; lock: number[]; n: number }
  > = {};
  for (const id of args.brains) {
    totals[id] = { win: 0, score: 0, session: 0, lock: [], n: 0 };
  }

  for (const fixtureId of args.fixtures) {
    results[fixtureId] = {};
    for (const brainId of args.brains) {
      const rng = mulberry32(args.seed + hash(fixtureId + ":" + brainId));
      const brain = createBrain(brainId, { rng });
      const opponent = createFixture(fixtureId, { rng });
      const result = runSeries(brain, opponent, args.rounds);
      results[fixtureId][brainId] = result;
      totals[brainId].win += result.aiWinRate;
      totals[brainId].score += result.score;
      totals[brainId].session += result.session.aiWinRate;
      if (result.lockIn != null) totals[brainId].lock.push(result.lockIn);
      totals[brainId].n++;
    }
  }

  const colWidth = Math.max(12, ...args.brains.map((n) => n.length + 1));
  const cell = (f: string, b: string, pick: (r: Cell) => number) =>
    pad(pct(pick(results[f][b])), colWidth);

  printTable(
    `SESSION first ${SESSION_ROUNDS}  (AI win %, ${args.rounds} round games, seed ${args.seed})\n`,
    args.fixtures,
    args.brains,
    (f, b) => cell(f, b, (r) => r.session.aiWinRate),
    [["MEAN", (b) => pad(pct(totals[b].session / totals[b].n), colWidth)]],
  );

  for (const bucket of WARMUP_BUCKETS) {
    const means: Record<string, number> = {};
    for (const b of args.brains) means[b] = 0;
    let n = 0;
    for (const f of args.fixtures) {
      for (const b of args.brains) {
        const row = results[f][b].buckets.find((x) => x.id === bucket.id);
        if (row && row.rounds) {
          means[b] += row.aiWinRate;
        }
      }
      n++;
    }
    let line = pad(`bucket ${bucket.id}`, 16);
    for (const b of args.brains) {
      line += "  " + pad(pct(means[b] / n), colWidth);
    }
    if (bucket.id === WARMUP_BUCKETS[0].id) {
      console.log("WARMUP buckets (mean AI win % across fixtures)\n");
      let header = pad("window", 16);
      for (const id of args.brains) header += "  " + pad(id, colWidth);
      console.log(header);
    }
    console.log(line);
  }

  let lockLine = pad("lock-in r", 16);
  for (const b of args.brains) {
    const xs = totals[b].lock;
    const avg = xs.length ? xs.reduce((a, c) => a + c, 0) / xs.length : NaN;
    lockLine +=
      "  " + pad(Number.isFinite(avg) ? avg.toFixed(1) : "—", colWidth);
  }
  console.log(lockLine);
  console.log(
    "\nlock-in = first round where the last 8 games are ≥60% AI wins (mean over fixtures that lock)\n",
  );

  printTable(
    `FULL game AI win %\n`,
    args.fixtures,
    args.brains,
    (f, b) => cell(f, b, (r) => r.aiWinRate),
    [
      ["MEAN", (b) => pad(pct(totals[b].win / totals[b].n), colWidth)],
      ["SCORE*", (b) => pad(pct(totals[b].score / totals[b].n), colWidth)],
    ],
  );
  console.log("* SCORE = (wins + 0.5 * ties) / rounds");
}

main();
