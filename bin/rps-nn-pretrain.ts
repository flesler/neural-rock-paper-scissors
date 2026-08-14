#!/usr/bin/env npx tsx
/**
 * Diagnostic: is NN weak because of cold-start, or because it can't
 * retune when the opponent changes?
 *
 * Pretrain neural-window on stationary fixtures, then play frozen vs
 * keep-learning on train / unseen-stationary / switching holdouts.
 */
import "../src/ai/node";
import {
  DIRECTIONS,
  OPTIONS,
  coveringThrow,
  createBrain,
  getWinner,
  option,
  runSeries,
  type Brain,
  type BrainOpts,
  type Match,
} from "../src/ai";
import { getNeataptic, lstmTrainOptions } from "../src/ai/neataptic";
import { createFixture } from "../src/fixtures";

const PRETRAIN = [
  "always-rock",
  "always-paper",
  "always-scissors",
  "cycle",
  "cycle-rev",
  "repeat",
  "copy-ai",
  "beat-ai",
];

const UNSEEN_STATIONARY = [
  "oscillate",
  "double-cycle",
  "win-stay-lose-shift",
  "switch-5",
  "anti-repeat",
  "biased-rock",
];

const SWITCHING = [
  "mix-2-cycle-beat",
  "mix-2-copy-rand",
  "mix-2-wsls-osc",
  "mix-3-cycle-beat-rand",
  "every-10-cycle-beat",
  "every-10-cycle-copy-beat",
  "phase-rand",
  "trickster",
  "bait-flip",
  "whiplash",
  "meta-farm",
];

const NOISY = ["random", "humanish"];
const ROUNDS = 80;
const PRETRAIN_ROUNDS = 80;
const SEED = 1;
const WINDOW = 8;
const MEMORY = 3;

type Net = {
  activate: (input: number[]) => number[];
  train: (
    data: Array<{ input: number[]; output: number[] }>,
    opts: object,
  ) => unknown;
  toJSON: () => unknown;
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

function directionSample(human: number, ai: number, prev?: Match): Match {
  const match: Match = { human, ai };
  const from = prev ?? match;
  const dir = option(human - from.human);
  const output: number[] = [];
  for (let i = 0; i < DIRECTIONS.length; i++) output[i] = i === dir ? 1 : 0;
  const input = output.concat();
  for (let i = 0; i < OPTIONS.length; i++) input.push(i === human ? 1 : 0);
  input.push(getWinner(human, ai));
  match.output = output;
  match.input = input;
  return match;
}

function chancesFromDirections(output: number[], prevHuman: number) {
  const chances = [0, 0, 0];
  for (let i = 0; i < DIRECTIONS.length; i++) {
    chances[option(prevHuman, DIRECTIONS[i])] += output[i];
  }
  return chances;
}

function cloneNet(nn: Net): Net {
  const neataptic = getNeataptic() as unknown as {
    Network: { fromJSON: (json: unknown) => Net };
  };
  return neataptic.Network.fromJSON(nn.toJSON());
}

function newLstm(): Net {
  const neataptic = getNeataptic();
  return new neataptic.architect.LSTM(
    DIRECTIONS.length + OPTIONS.length + 1,
    MEMORY,
    DIRECTIONS.length,
  ) as unknown as Net;
}

function createWindowBrain(
  id: string,
  nn: Net,
  opts?: BrainOpts & { freeze?: boolean },
): Brain {
  const rng = opts?.rng ?? Math.random;
  const freeze = !!opts?.freeze;
  const trainOpts = lstmTrainOptions({ rate: 0.25 });
  const matches: Match[] = [];

  return {
    id,
    decide() {
      if (matches.length - 1 < MEMORY) {
        return Math.floor(rng() * OPTIONS.length);
      }
      const prev = matches[matches.length - 1];
      const output = nn.activate(prev.input!);
      const chances = chancesFromDirections(output, prev.human);
      const ranked = [0, 1, 2].sort((a, b) => chances[b] - chances[a]);
      if (chances[ranked[0]] - chances[ranked[1]] < 0.2) {
        return coveringThrow(ranked[0], ranked[1]);
      }
      return option(ranked[0]);
    },
    learn(human, ai) {
      const prev = matches[matches.length - 1];
      const match = directionSample(human, ai, prev);
      matches.push(match);
      if (freeze) return;
      const data: Array<{ input: number[]; output: number[] }> = [];
      const start = Math.max(1, matches.length - WINDOW);
      for (let i = start; i < matches.length; i++) {
        const p = matches[i - 1];
        const c = matches[i];
        if (p.input && c.output)
          data.push({ input: p.input, output: c.output });
      }
      if (data.length) nn.train(data, trainOpts);
    },
    getMatches: () => matches,
  };
}

function pretrain(): Net {
  const nn = newLstm();
  const rng = mulberry32(SEED);
  for (const fixtureId of PRETRAIN) {
    runSeries(
      createWindowBrain("pre", nn, { rng }),
      createFixture(fixtureId, { rng, rounds: PRETRAIN_ROUNDS }),
      PRETRAIN_ROUNDS,
    );
  }
  return nn;
}

function pct(n: number) {
  return (100 * n).toFixed(1).padStart(5);
}

function mean(xs: number[]) {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function evalSet(
  label: string,
  fixtures: string[],
  factory: (id: string, opts?: BrainOpts) => Brain,
) {
  const rates: number[] = [];
  const early: number[] = [];
  const late: number[] = [];
  const rows: string[] = [];
  for (const fixtureId of fixtures) {
    const rng = mulberry32(SEED + hash(label + fixtureId));
    const result = runSeries(
      factory(fixtureId, { rng }),
      createFixture(fixtureId, { rng, rounds: ROUNDS }),
      ROUNDS,
    );
    rates.push(result.aiWinRate);
    const first = result.payoffByRound.slice(0, 10);
    const last = result.payoffByRound.slice(-20);
    const e = first.filter((p) => p > 0).length / first.length;
    const l = last.filter((p) => p > 0).length / last.length;
    early.push(e);
    late.push(l);
    rows.push(
      `  ${fixtureId.padEnd(26)} ${pct(result.aiWinRate)}   r1-10 ${pct(e)}   last20 ${pct(l)}`,
    );
  }
  console.log(
    `${label.padEnd(22)} mean ${pct(mean(rates))}   r1-10 ${pct(mean(early))}   last20 ${pct(mean(late))}`,
  );
  for (const row of rows) console.log(row);
  console.log("");
  return mean(rates);
}

function main() {
  console.log("Pretraining neural-window on:");
  console.log(" ", PRETRAIN.join(", "));
  const pretrained = pretrain();
  console.log("done.\n");

  const scratch = (id: string, opts?: BrainOpts) =>
    createBrain("neural-window", opts);
  const frozen = (_id: string, opts?: BrainOpts) =>
    createWindowBrain("nn-frozen", cloneNet(pretrained), {
      ...opts,
      freeze: true,
    });
  const live = (_id: string, opts?: BrainOpts) =>
    createWindowBrain("nn-live", cloneNet(pretrained), {
      ...opts,
      freeze: false,
    });
  const iocaine = (_id: string, opts?: BrainOpts) =>
    createBrain("iocaine", opts);

  const groups: Array<[string, string[]]> = [
    ["PRETRAIN seen", PRETRAIN],
    ["UNSEEN stationary", UNSEEN_STATIONARY],
    ["SWITCHING", SWITCHING],
    ["NOISY", NOISY],
  ];

  for (const [name, fixtures] of groups) {
    console.log(`=== ${name} ===`);
    evalSet("scratch neural-window", fixtures, scratch);
    evalSet("pretrained FROZEN", fixtures, frozen);
    evalSet("pretrained + learn", fixtures, live);
    evalSet("iocaine", fixtures, iocaine);
  }
}

main();
