import {
  AI,
  DIRECTIONS,
  HUMAN,
  OPTIONS,
  TIE,
  type Brain,
  type BrainOpts,
  type Match,
  argmax,
  coveringThrow,
  getWinner,
  option,
  payoff,
  rngFrom,
} from "./core";
import {
  createArenaBrain,
  createContestBrain,
  createGeneticBrain,
  createIocaineBrain,
} from "./advanced";
import { createNetwork, trainOptions } from "./brain";

const DIRECTION_INPUTS = DIRECTIONS.length + OPTIONS.length + 1;
const DIRECTION_OUTPUTS = DIRECTIONS.length;
const COMBINED = [...OPTIONS, ...DIRECTIONS] as const;
const COMBINED_OUTPUTS = COMBINED.length;
const COMBINED_INPUTS = COMBINED_OUTPUTS + 1;

function directionSample(human: number, ai: number, prev?: Match): Match {
  const match: Match = { human, ai };
  const from = prev ?? match;
  const dir = option(human - from.human);
  const output: number[] = [];
  for (let i = 0; i < DIRECTIONS.length; i++) {
    output[i] = i === dir ? 1 : 0;
  }
  const input = output.concat();
  for (let i = 0; i < OPTIONS.length; i++) {
    input.push(i === human ? 1 : 0);
  }
  input.push(getWinner(human, ai));
  match.output = output;
  match.input = input;
  return match;
}

function combinedSample(human: number, ai: number, prev?: Match): Match {
  const match: Match = { human, ai };
  const from = prev ?? match;
  const dir = option(human - from.human);
  const output: number[] = [];
  for (let i = 0; i < COMBINED.length; i++) {
    const val = COMBINED[i];
    const ok =
      typeof val === "string"
        ? OPTIONS.indexOf(val) === human
        : DIRECTIONS.indexOf(val) === dir;
    output[i] = ok ? 1 : 0;
  }
  match.output = output;
  match.input = output.concat([getWinner(human, ai)]);
  return match;
}

function trainLast(
  nn: {
    train: (
      data: Array<{ input: number[]; output: number[] }>,
      opts: Record<string, unknown>,
    ) => { error: number };
  },
  matches: Match[],
  window: number,
  trainOpts: Record<string, unknown>,
) {
  const data: Array<{ input: number[]; output: number[] }> = [];
  const start = Math.max(1, matches.length - window);
  for (let i = start; i < matches.length; i++) {
    const prev = matches[i - 1];
    const cur = matches[i];
    if (prev.input && cur.output) {
      data.push({ input: prev.input, output: cur.output });
    }
  }
  if (data.length) nn.train(data, trainOpts);
}

function chancesFromDirections(output: number[], prevHuman: number) {
  const chances = [0, 0, 0];
  for (let i = 0; i < DIRECTIONS.length; i++) {
    chances[option(prevHuman, DIRECTIONS[i])] += output[i];
  }
  return chances;
}

function chancesFromCombined(output: number[], prevHuman: number) {
  const chances = [0, 0, 0];
  for (let i = 0; i < COMBINED.length; i++) {
    const val = COMBINED[i];
    const index =
      typeof val === "string" ? OPTIONS.indexOf(val) : option(prevHuman, val);
    chances[index] += output[i];
  }
  return chances;
}

function aiFromChances(chances: number[], cover: boolean) {
  const ranked = [0, 1, 2].sort((a, b) => chances[b] - chances[a]);
  const best = ranked[0];
  const snd = ranked[1];
  if (cover && chances[best] - chances[snd] < 0.2) {
    return coveringThrow(best, snd);
  }
  return option(best);
}

function createNeuralBrain(
  id: string,
  opts: BrainOpts | undefined,
  spec: {
    cover: boolean;
    window: number;
    combined: boolean;
    memory?: number;
    rate?: number;
    open?: "random" | "beat-last";
  },
): Brain {
  const rng = rngFrom(opts);
  const inputs = spec.combined ? COMBINED_INPUTS : DIRECTION_INPUTS;
  const outputs = spec.combined ? COMBINED_OUTPUTS : DIRECTION_OUTPUTS;
  const memory =
    spec.memory ?? (spec.combined ? OPTIONS.length : DIRECTION_OUTPUTS);
  const nn = createNetwork(inputs, memory, outputs);
  const trainOpts = trainOptions(
    spec.rate != null ? { learningRate: spec.rate } : {},
  );
  const matches: Match[] = [];
  const sample = spec.combined ? combinedSample : directionSample;
  const open = spec.open ?? "random";

  return {
    id,
    decide() {
      if (matches.length - 1 < memory) {
        if (open === "beat-last" && matches.length) {
          return option(matches[matches.length - 1].human);
        }
        return Math.floor(rng() * OPTIONS.length);
      }
      const prev = matches[matches.length - 1];
      const output = nn.activate(prev.input!);
      const chances = spec.combined
        ? chancesFromCombined(output, prev.human)
        : chancesFromDirections(output, prev.human);
      return aiFromChances(chances, spec.cover);
    },
    learn(human, ai) {
      const prev = matches[matches.length - 1];
      const match = sample(human, ai, prev);
      matches.push(match);
      trainLast(nn, matches, spec.window, trainOpts);
    },
    getMatches: () => matches,
  };
}

export function createRandomBrain(opts?: BrainOpts): Brain {
  const rng = rngFrom(opts);
  const matches: Match[] = [];
  return {
    id: "random",
    decide: () => Math.floor(rng() * OPTIONS.length),
    learn(human, ai) {
      matches.push({ human, ai });
    },
    getMatches: () => matches,
  };
}

export function createNeuralBrainClassic(opts?: BrainOpts) {
  return createNeuralBrain("feedforward", opts, {
    cover: false,
    window: 1,
    combined: false,
  });
}

export function createNeuralCoverBrain(opts?: BrainOpts) {
  return createNeuralBrain("neural-cover", opts, {
    cover: true,
    window: 1,
    combined: false,
  });
}

export function createNeural6Brain(opts?: BrainOpts) {
  return createNeuralBrain("neural-6", opts, {
    cover: true,
    window: 1,
    combined: true,
  });
}

export function createNeuralWindowBrain(opts?: BrainOpts) {
  return createNeuralBrain("windowed-net", opts, {
    cover: true,
    window: 8,
    combined: false,
    rate: 0.25,
  });
}

export function createNeuralComboBrain(opts?: BrainOpts) {
  return createNeuralBrain("neural-combo", opts, {
    cover: true,
    window: 8,
    combined: true,
    rate: 0.5,
  });
}

export function createNeuralOpenBrain(opts?: BrainOpts) {
  return createNeuralBrain("neural-open", opts, {
    cover: true,
    window: 8,
    combined: false,
    rate: 0.25,
    open: "beat-last",
  });
}

export function createPatternsBrain(opts?: BrainOpts): Brain {
  const rng = rngFrom(opts);
  const matches: Match[] = [];
  const freq = [0, 0, 0];
  const trans = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const trans2: number[][][] = [];
  for (let a = 0; a < 3; a++) {
    trans2[a] = [];
    for (let b = 0; b < 3; b++) {
      trans2[a][b] = [0, 0, 0];
    }
  }

  function predictHuman() {
    if (!matches.length) return Math.floor(rng() * OPTIONS.length);
    const last = matches[matches.length - 1].human;
    if (matches.length >= 3) {
      const prev = matches[matches.length - 2].human;
      const row = trans2[prev][last];
      if (row[0] + row[1] + row[2] >= 2) return argmax(row);
    }
    const row = trans[last];
    if (row[0] + row[1] + row[2] >= 1) return argmax(row);
    if (freq[0] + freq[1] + freq[2]) return argmax(freq);
    return Math.floor(rng() * OPTIONS.length);
  }

  return {
    id: "patterns",
    decide: () => option(predictHuman()),
    learn(human, ai) {
      if (matches.length) trans[matches[matches.length - 1].human][human]++;
      if (matches.length >= 2) {
        trans2[matches[matches.length - 2].human][
          matches[matches.length - 1].human
        ][human]++;
      }
      freq[human]++;
      matches.push({ human, ai });
    },
    getMatches: () => matches,
  };
}

export function createAdaptiveBrain(opts?: BrainOpts): Brain {
  const rng = rngFrom(opts);
  const matches: Match[] = [];
  const freq = [0, 0, 0];
  const trans = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const scores = { freq: 0, markov: 0, repeat: 0, rotate: 0, copyAi: 0 };
  let losses = 0;

  function expertGuesses() {
    const last = matches[matches.length - 1];
    const guesses = {
      freq: argmax(freq),
      markov: last ? argmax(trans[last.human]) : argmax(freq),
      repeat: last ? last.human : 0,
      rotate: last ? option(last.human) : 1,
      copyAi: last ? last.ai : 0,
    };
    if (freq[0] + freq[1] + freq[2] === 0) {
      guesses.freq = Math.floor(rng() * OPTIONS.length);
    }
    if (last) {
      const row = trans[last.human];
      if (row[0] + row[1] + row[2] === 0) guesses.markov = guesses.freq;
    }
    return guesses;
  }

  function bestExpert(guesses: Record<string, number>) {
    let name: keyof typeof scores = "freq";
    let best = -Infinity;
    for (const key of Object.keys(scores) as Array<keyof typeof scores>) {
      if (scores[key] > best) {
        best = scores[key];
        name = key;
      }
    }
    return guesses[name];
  }

  return {
    id: "adaptive",
    decide() {
      const guesses = expertGuesses();
      const humanGuess = matches.length
        ? bestExpert(guesses)
        : Math.floor(rng() * OPTIONS.length);
      let throw_ = option(humanGuess);
      if (losses >= 3) {
        throw_ =
          rng() < 0.5 ? option(throw_, -1) : Math.floor(rng() * OPTIONS.length);
      }
      return throw_;
    },
    learn(human, ai) {
      const guesses = expertGuesses();
      for (const key of Object.keys(guesses) as Array<keyof typeof scores>) {
        const would = option(guesses[key]);
        const winner = getWinner(human, would);
        if (winner === AI) scores[key] += 1;
        else if (winner === HUMAN) scores[key] -= 1;
      }
      if (matches.length) trans[matches[matches.length - 1].human][human]++;
      freq[human]++;
      const winner = getWinner(human, ai);
      losses = winner === HUMAN ? losses + 1 : 0;
      matches.push({ human, ai });
    },
    getMatches: () => matches,
  };
}

type SpeciesPool = "core" | "nn" | "full";

function speciesPool(kind: SpeciesPool, opts?: BrainOpts): Brain[] {
  const iocaine = createIocaineBrain({
    ...opts,
    decay: 0.9,
    biasLock: 0.5,
    id: "s-iocaine",
  });
  const core = [
    iocaine,
    createContestBrain(opts),
    createPatternsBrain(opts),
    createAdaptiveBrain(opts),
  ];
  if (kind === "core") return core;
  if (kind === "nn") return [...core, createNeuralCoverBrain(opts)];
  return [
    createIocaineBrain({
      ...opts,
      decay: 0.86,
      biasLock: 0.5,
      id: "s-io-fast",
    }),
    createIocaineBrain({
      ...opts,
      decay: 0.94,
      biasLock: 0.55,
      id: "s-io-slow",
    }),
    createPatternsBrain(opts),
    createAdaptiveBrain(opts),
    createContestBrain(opts),
    createNeuralCoverBrain(opts),
    createRandomBrain(opts),
  ];
}

export function createSpeciesBrain(
  opts?: BrainOpts & {
    id?: string;
    speciesDecay?: number;
    defaultUntil?: number;
    pool?: SpeciesPool;
  },
): Brain {
  const pool = speciesPool(opts?.pool ?? "core", opts);
  const scores = pool.map(() => 0);
  const proposed = pool.map(() => 0);
  const matches: Match[] = [];
  const decay = opts?.speciesDecay ?? 1;
  const defaultUntil = opts?.defaultUntil ?? 0;

  function champ() {
    let best = 0;
    for (let i = 1; i < pool.length; i++) {
      if (scores[i] > scores[best]) best = i;
    }
    return best;
  }

  return {
    id: opts?.id ?? "best-of",
    decide() {
      for (let i = 0; i < pool.length; i++) proposed[i] = pool[i].decide();
      if (matches.length < defaultUntil) return proposed[0];
      return proposed[champ()];
    },
    learn(human, ai) {
      for (let i = 0; i < pool.length; i++) {
        scores[i] = scores[i] * decay + payoff(human, proposed[i]);
        pool[i].learn(human, ai);
      }
      matches.push({ human, ai });
    },
    getMatches: () => matches,
  };
}

const FACTORIES: Record<string, (opts?: BrainOpts) => Brain> = {
  random: createRandomBrain,
  feedforward: createNeuralBrainClassic,
  "neural-cover": createNeuralCoverBrain,
  "neural-6": createNeural6Brain,
  "windowed-net": createNeuralWindowBrain,
  "neural-combo": createNeuralComboBrain,
  "neural-open": createNeuralOpenBrain,
  patterns: createPatternsBrain,
  adaptive: createAdaptiveBrain,
  iocaine: (opts) => createIocaineBrain({ ...opts, decay: 0.9, biasLock: 0.5 }),
  "iocaine-plus": createContestBrain,
  "best-of": createSpeciesBrain,
  "best-of-nn": (opts) =>
    createSpeciesBrain({ ...opts, pool: "nn", id: "best-of-nn" }),
  "best-of-decay": (opts) =>
    createSpeciesBrain({
      ...opts,
      pool: "full",
      speciesDecay: 0.85,
      id: "best-of-decay",
    }),
  "best-of-warm": (opts) =>
    createSpeciesBrain({ ...opts, defaultUntil: 6, id: "best-of-warm" }),
  "genetic-mix": createGeneticBrain,
  arena: createArenaBrain,
};

export const BRAIN_IDS = Object.keys(FACTORIES);

/** First bucket (1–5): noisy, still gathering a read. */
export const LEARNING_ROUNDS = 5;

/**
 * Lock-in round from `npm run bench` (first 8-game window ≥60% AI wins).
 * UI: Learning through `LEARNING_ROUNDS`, then Perfecting until this.
 * Random never locks.
 */
export const BRAIN_WARMUP: Record<string, number> = {
  "best-of": 10,
  "best-of-nn": 10,
  "best-of-decay": 10,
  "best-of-warm": 8,
  iocaine: 10,
  "iocaine-plus": 9,
  feedforward: 12,
  "neural-cover": 12,
  "neural-6": 12,
  "windowed-net": 10,
  "neural-combo": 10,
  "neural-open": 8,
  patterns: 14,
  adaptive: 10,
  "genetic-mix": 10,
  arena: 10,
  random: 0,
};

export function warmupRounds(id: string) {
  return BRAIN_WARMUP[id] ?? 10;
}

export function createBrain(id: string, opts?: BrainOpts): Brain {
  const factory = FACTORIES[id];
  if (!factory) throw new Error("unknown brain: " + id);
  return factory(opts);
}

export { AI, HUMAN, OPTIONS, TIE };
