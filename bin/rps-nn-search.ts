#!/usr/bin/env npx tsx
/**
 * Brute-force NN variants vs fixtures.
 * Train = simple stationary patterns. Holdout = hybrids / noisy / random.
 */
import "../src/ai/node";
import {
  OPTIONS,
  coveringThrow,
  createBrain,
  option,
  runSeries,
  type Brain,
  type BrainOpts,
  type Match,
} from "../src/ai";
import { getNeataptic, lstmTrainOptions } from "../src/ai/neataptic";
import { FIXTURE_IDS, createFixture } from "../src/fixtures";

const TRAIN = [
  "always-rock",
  "always-paper",
  "always-scissors",
  "cycle",
  "cycle-rev",
  "repeat",
  "win-stay-lose-shift",
  "copy-ai",
  "beat-ai",
  "switch-5",
  "switch-8",
  "anti-repeat",
  "oscillate",
  "double-cycle",
  "biased-rock",
];

const HOLDOUT = FIXTURE_IDS.filter((id) => !TRAIN.includes(id));

const ROUNDS = 60;
const SEED = 1;

type Spec = {
  id: string;
  kind: "lstm" | "mlp" | "hist";
  cover: boolean;
  window: number;
  combined?: boolean;
  rate: number;
  iterations: number;
  hidden: number;
  hist?: number;
};

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

function oneHot(n: number, i: number) {
  const v = Array(n).fill(0);
  v[i] = 1;
  return v;
}

function trainLast(
  nn: {
    train: (
      data: Array<{ input: number[]; output: number[] }>,
      opts: object,
    ) => unknown;
  },
  pairs: Array<{ input: number[]; output: number[] }>,
  window: number,
  trainOpts: object,
) {
  const data = pairs.slice(-window);
  if (data.length) nn.train(data, trainOpts);
}

function createLstmBrain(spec: Spec, opts?: BrainOpts): Brain {
  const rng = opts?.rng ?? Math.random;
  const neataptic = getNeataptic();
  const combined = !!spec.combined;
  const inputs = combined ? 7 : 7;
  const outputs = combined ? 6 : 3;
  const nn = new neataptic.architect.LSTM(inputs, spec.hidden, outputs);
  const trainOpts = lstmTrainOptions({
    rate: spec.rate,
    iterations: spec.iterations,
  });
  const matches: Match[] = [];
  const pairs: Array<{ input: number[]; output: number[] }> = [];

  function encode(human: number, ai: number, prev?: Match) {
    const from = prev ?? { human, ai };
    const dir = option(human - from.human);
    if (combined) {
      const output = [...oneHot(3, human), ...oneHot(3, dir)];
      return {
        input: [...output, human === ai ? 0.5 : option(human) === ai ? 1 : 0],
        output,
      };
    }
    const output = oneHot(3, dir);
    return {
      input: [
        ...output,
        ...oneHot(3, human),
        human === ai ? 0.5 : option(human) === ai ? 1 : 0,
      ],
      output,
    };
  }

  return {
    id: spec.id,
    decide() {
      if (matches.length < spec.hidden) return Math.floor(rng() * 3);
      const prev = matches[matches.length - 1];
      const out = nn.activate(prev.input!);
      const chances = [0, 0, 0];
      if (spec.combined) {
        for (let i = 0; i < 3; i++) chances[i] += out[i];
        for (let i = 0; i < 3; i++)
          chances[option(prev.human, i - 1)] += out[3 + i];
      } else {
        for (let i = 0; i < 3; i++)
          chances[option(prev.human, i - 1)] += out[i];
      }
      const ranked = [0, 1, 2].sort((a, b) => chances[b] - chances[a]);
      if (spec.cover && chances[ranked[0]] - chances[ranked[1]] < 0.2) {
        return coveringThrow(ranked[0], ranked[1]);
      }
      return option(ranked[0]);
    },
    learn(human, ai) {
      const prev = matches[matches.length - 1];
      const enc = encode(human, ai, prev);
      const match: Match = { human, ai, input: enc.input, output: enc.output };
      matches.push(match);
      if (prev?.input) pairs.push({ input: prev.input, output: enc.output });
      trainLast(nn, pairs, spec.window, trainOpts);
    },
    getMatches: () => matches,
  };
}

function createMlpBrain(spec: Spec, opts?: BrainOpts): Brain {
  const rng = opts?.rng ?? Math.random;
  const neataptic = getNeataptic();
  const inputs = 7;
  const nn = neataptic.architect.Perceptron(inputs, spec.hidden, 3);
  const trainOpts = lstmTrainOptions({
    rate: spec.rate,
    iterations: spec.iterations,
    clear: true,
  });
  const matches: Match[] = [];
  const pairs: Array<{ input: number[]; output: number[] }> = [];

  function encodeIn(human: number, ai: number, prev?: Match) {
    const from = prev ?? { human, ai };
    const dir = option(human - from.human);
    return [
      ...oneHot(3, dir),
      ...oneHot(3, human),
      human === ai ? 0.5 : option(human) === ai ? 1 : 0,
    ];
  }

  return {
    id: spec.id,
    decide() {
      if (matches.length < 2) return Math.floor(rng() * 3);
      const prev = matches[matches.length - 1];
      const out = nn.activate(prev.input!);
      const humanGuess =
        out[0] >= out[1] && out[0] >= out[2] ? 0 : out[1] >= out[2] ? 1 : 2;
      const snd =
        out[humanGuess] === Math.max(...out)
          ? [0, 1, 2]
              .filter((i) => i !== humanGuess)
              .sort((a, b) => out[b] - out[a])[0]
          : humanGuess;
      if (spec.cover && out[humanGuess] - out[snd] < 0.2) {
        return coveringThrow(humanGuess, snd);
      }
      return option(humanGuess);
    },
    learn(human, ai) {
      const prev = matches[matches.length - 1];
      const input = encodeIn(human, ai, prev);
      const output = oneHot(3, human);
      matches.push({ human, ai, input, output });
      if (prev?.input) pairs.push({ input: prev.input, output });
      trainLast(nn, pairs, spec.window, trainOpts);
    },
    getMatches: () => matches,
  };
}

function createHistBrain(spec: Spec, opts?: BrainOpts): Brain {
  const rng = opts?.rng ?? Math.random;
  const k = spec.hist ?? 4;
  const neataptic = getNeataptic();
  const inputs = k * 6;
  const nn = neataptic.architect.Perceptron(inputs, spec.hidden, 3);
  const trainOpts = lstmTrainOptions({
    rate: spec.rate,
    iterations: spec.iterations,
    clear: true,
  });
  const matches: Match[] = [];
  const pairs: Array<{ input: number[]; output: number[] }> = [];

  function histInput() {
    const v: number[] = [];
    for (let i = matches.length - k; i < matches.length; i++) {
      if (i < 0) {
        v.push(0, 0, 0, 0, 0, 0);
        continue;
      }
      v.push(...oneHot(3, matches[i].human), ...oneHot(3, matches[i].ai));
    }
    return v;
  }

  return {
    id: spec.id,
    decide() {
      if (matches.length < k) return Math.floor(rng() * 3);
      const out = nn.activate(histInput());
      let best = 0;
      for (let i = 1; i < 3; i++) if (out[i] > out[best]) best = i;
      return option(best);
    },
    learn(human, ai) {
      if (matches.length >= k) {
        pairs.push({ input: histInput(), output: oneHot(3, human) });
      }
      matches.push({ human, ai });
      trainLast(nn, pairs, spec.window, trainOpts);
    },
    getMatches: () => matches,
  };
}

function makeBrain(spec: Spec, opts?: BrainOpts): Brain {
  if (spec.kind === "lstm") return createLstmBrain(spec, opts);
  if (spec.kind === "mlp") return createMlpBrain(spec, opts);
  return createHistBrain(spec, opts);
}

function specs(): Spec[] {
  const out: Spec[] = [];

  for (const window of [1, 8]) {
    for (const rate of [0.25, 0.5]) {
      for (const cover of [true, false]) {
        out.push({
          id: `lstm-w${window}-r${rate}${cover ? "-c" : ""}`,
          kind: "lstm",
          cover,
          window,
          combined: false,
          rate,
          iterations: 8,
          hidden: 3,
        });
      }
    }
  }

  out.push({
    id: "lstm-w16-r0.25-c-h6",
    kind: "lstm",
    cover: true,
    window: 16,
    rate: 0.25,
    iterations: 8,
    hidden: 6,
  });

  for (const hist of [3, 4, 6]) {
    for (const hidden of [6, 9]) {
      out.push({
        id: `hist-k${hist}-h${hidden}-w8-r0.2`,
        kind: "hist",
        cover: false,
        window: 8,
        rate: 0.2,
        iterations: 8,
        hidden,
        hist,
      });
    }
  }

  for (const window of [8, 16]) {
    out.push({
      id: `mlp-w${window}-r0.25-c`,
      kind: "mlp",
      cover: true,
      window,
      rate: 0.25,
      iterations: 8,
      hidden: 8,
    });
  }

  return out;
}

type Score = {
  id: string;
  train: number;
  hold: number;
  all: number;
  weakTrain: string[];
  weakHold: string[];
  worstHold: string;
  worstHoldPct: number;
};

function mean(xs: number[]) {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function evalBrain(id: string, factory: (opts?: BrainOpts) => Brain): Score {
  const byFix: Record<string, number> = {};
  for (const fixtureId of FIXTURE_IDS) {
    const rng = mulberry32(SEED + hash(fixtureId + ":" + id));
    const result = runSeries(
      factory({ rng }),
      createFixture(fixtureId, { rng, rounds: ROUNDS }),
      ROUNDS,
    );
    byFix[fixtureId] = result.aiWinRate;
  }
  const trainRates = TRAIN.map((f) => byFix[f]);
  const holdRates = HOLDOUT.map((f) => byFix[f]);
  const allRates = FIXTURE_IDS.map((f) => byFix[f]);
  let worstHold = HOLDOUT[0];
  for (const f of HOLDOUT) if (byFix[f] < byFix[worstHold]) worstHold = f;
  return {
    id,
    train: mean(trainRates),
    hold: mean(holdRates),
    all: mean(allRates),
    weakTrain: TRAIN.filter((f) => byFix[f] < 0.45),
    weakHold: HOLDOUT.filter((f) => byFix[f] < 0.45),
    worstHold,
    worstHoldPct: byFix[worstHold],
  };
}

function pct(n: number) {
  return (100 * n).toFixed(1);
}

function main() {
  const baselines = [
    "best-of",
    "iocaine",
    "neural",
    "neural-window",
    "patterns",
    "adaptive",
  ];
  const rows: Score[] = [];
  console.log(
    `train=${TRAIN.length} holdout=${HOLDOUT.length} rounds=${ROUNDS} seed=${SEED}\n`,
  );
  console.log("HOLD:", HOLDOUT.join(", "));
  console.log("TRAIN:", TRAIN.join(", "));
  console.log("");

  for (const id of baselines) {
    const row = evalBrain(id, (opts) => createBrain(id, opts));
    rows.push(row);
    console.log(
      `BASE ${row.id.padEnd(22)} train ${pct(row.train)}  hold ${pct(row.hold)}  all ${pct(row.all)}  weakH ${row.weakHold.length}  worst ${row.worstHold} ${pct(row.worstHoldPct)}`,
    );
  }

  const grid = specs();
  console.log(`\nSearching ${grid.length} NN specs...\n`);
  for (const spec of grid) {
    const row = evalBrain(spec.id, (opts) => makeBrain(spec, opts));
    rows.push(row);
    if (row.train >= 0.7 || row.hold >= 0.55) {
      console.log(
        `HIT  ${row.id.padEnd(32)} train ${pct(row.train)}  hold ${pct(row.hold)}  all ${pct(row.all)}  weakT ${row.weakTrain.join(",") || "-"}  weakH ${row.weakHold.join(",") || "-"}`,
      );
    }
  }

  rows.sort((a, b) => b.hold - a.hold || b.train - a.train);
  console.log("\nTOP 12 by holdout\n");
  for (const row of rows.slice(0, 12)) {
    console.log(
      `${row.id.padEnd(32)} train ${pct(row.train)}  hold ${pct(row.hold)}  all ${pct(row.all)}  weakH ${row.weakHold.length} (${row.weakHold.join(",") || "-"})  worst ${row.worstHold} ${pct(row.worstHoldPct)}`,
    );
  }

  const nnOnly = rows.filter((r) => !baselines.includes(r.id));
  nnOnly.sort((a, b) => b.train - a.train);
  console.log("\nTOP 8 NN by train (watch holdout drop = overfit)\n");
  for (const row of nnOnly.slice(0, 8)) {
    console.log(
      `${row.id.padEnd(32)} train ${pct(row.train)}  hold ${pct(row.hold)}  gap ${pct(row.train - row.hold)}  weakT ${row.weakTrain.join(",") || "-"}`,
    );
  }
}

main();
