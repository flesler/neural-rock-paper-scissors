export const OPTIONS = ["rock", "paper", "scissors"] as const;
export const DIRECTIONS = [-1, 0, 1] as const;

/** World RPS Society mix: rock 35.4%, paper 29.6%, scissors 35.0%. */
export const HUMAN_THROW_PRIOR = [0.354, 0.296, 0.35] as const;

export const HUMAN = 0;
export const AI = 1;
export const TIE = 0.5;

export type Match = {
  human: number;
  ai: number;
  input?: number[];
  output?: number[];
};

export type Brain = {
  id: string;
  decide: () => number;
  learn: (human: number, ai: number) => void;
  getMatches: () => Match[];
};

export type BrainOpts = {
  rng?: () => number;
};

export function option(choice: number, delta = +1) {
  return (choice + delta + OPTIONS.length) % OPTIONS.length;
}

export function getWinner(human: number, ai: number) {
  switch (ai) {
    case human:
      return TIE;
    case option(human):
      return AI;
    default:
      return HUMAN;
  }
}

export function argmax(values: number[]) {
  let max = values[0];
  let idx = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > max) {
      max = values[i];
      idx = i;
    }
  }
  return idx;
}

export function rngFrom(opts?: BrainOpts) {
  return opts?.rng ?? Math.random;
}

/** Throw that beats `a` and ties `b` when they differ. */
export function coveringThrow(a: number, b: number) {
  if (a === b) return option(a);
  return a === option(b) ? a : b;
}

export function payoff(human: number, ai: number) {
  const winner = getWinner(human, ai);
  if (winner === AI) return 1;
  if (winner === HUMAN) return -1;
  return 0;
}

export type SeriesResult = {
  aiWins: number;
  humanWins: number;
  ties: number;
  rounds: number;
  aiWinRate: number;
  score: number;
  payoffByRound: number[];
  session: { rounds: number; aiWinRate: number; score: number };
  buckets: Array<{
    id: string;
    rounds: number;
    aiWinRate: number;
    score: number;
  }>;
  segments: Array<{
    id: string;
    rounds: number;
    aiWinRate: number;
    score: number;
  }>;
  lockIn: number | null;
};

export const SESSION_ROUNDS = 20;

export const WARMUP_BUCKETS = [
  { id: "1-5", from: 0, to: 5 },
  { id: "6-12", from: 5, to: 12 },
  { id: "13-20", from: 12, to: 20 },
  { id: "21-40", from: 20, to: 40 },
  { id: "41+", from: 40, to: Number.POSITIVE_INFINITY },
] as const;

function summarize(payoffs: number[]) {
  let wins = 0;
  let ties = 0;
  for (const p of payoffs) {
    if (p > 0) wins++;
    else if (p === 0) ties++;
  }
  const rounds = payoffs.length;
  return {
    rounds,
    aiWinRate: rounds ? wins / rounds : 0,
    score: rounds ? (wins + 0.5 * ties) / rounds : 0,
  };
}

function lockInRound(payoffs: number[], window = 8, target = 0.6) {
  if (payoffs.length < window) return null;
  let wins = 0;
  for (let i = 0; i < window; i++) if (payoffs[i] > 0) wins++;
  if (wins / window >= target) return window;
  for (let i = window; i < payoffs.length; i++) {
    if (payoffs[i - window] > 0) wins--;
    if (payoffs[i] > 0) wins++;
    if (wins / window >= target) return i + 1;
  }
  return null;
}

export type Opponent = {
  id?: string;
  reset?: () => void;
  decide: () => number;
  learn?: (human: number, ai: number) => void;
  segment?: (round: number) => string;
};

export function runSeries(
  brain: Brain,
  opponent: Opponent,
  rounds: number,
): SeriesResult {
  opponent.reset?.();
  const payoffByRound: number[] = [];
  let aiWins = 0;
  let humanWins = 0;
  let ties = 0;
  for (let i = 0; i < rounds; i++) {
    const ai = brain.decide();
    const human = opponent.decide();
    const winner = getWinner(human, ai);
    if (winner === AI) aiWins++;
    else if (winner === HUMAN) humanWins++;
    else ties++;
    payoffByRound.push(payoff(human, ai));
    brain.learn(human, ai);
    opponent.learn?.(human, ai);
  }
  const buckets = WARMUP_BUCKETS.map((b) => ({
    id: b.id,
    ...summarize(
      payoffByRound.slice(b.from, Math.min(b.to, payoffByRound.length)),
    ),
  }));
  const bySegment: Record<string, number[]> = {};
  if (opponent.segment) {
    for (let i = 0; i < payoffByRound.length; i++) {
      const id = opponent.segment(i);
      (bySegment[id] ||= []).push(payoffByRound[i]);
    }
  }
  const segments = Object.keys(bySegment).map((id) => ({
    id,
    ...summarize(bySegment[id]),
  }));
  return {
    aiWins,
    humanWins,
    ties,
    rounds,
    aiWinRate: aiWins / rounds,
    score: (aiWins + 0.5 * ties) / rounds,
    payoffByRound,
    session: summarize(payoffByRound.slice(0, SESSION_ROUNDS)),
    buckets,
    segments,
    lockIn: lockInRound(payoffByRound),
  };
}
