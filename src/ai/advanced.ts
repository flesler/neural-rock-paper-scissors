import {
  AI,
  HUMAN,
  HUMAN_THROW_PRIOR,
  OPTIONS,
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

function zeros3() {
  return [0, 0, 0];
}

/** World RPS Society throw mix: rock 35.4%, paper 29.6%, scissors 35.0%. */
const HUMAN_PRIOR_N = 6;

function freqPredict(humans: number[], prior = false) {
  if (!humans.length && !prior) return null;
  const c = zeros3();
  if (prior) {
    for (let i = 0; i < 3; i++) c[i] = HUMAN_THROW_PRIOR[i] * HUMAN_PRIOR_N;
  }
  for (const h of humans) c[h]++;
  return argmax(c);
}

function throwMix(seq: number[], window?: number, prior = false) {
  const c = zeros3();
  if (prior) {
    for (let i = 0; i < 3; i++) c[i] = HUMAN_THROW_PRIOR[i] * HUMAN_PRIOR_N;
  }
  const start = window != null ? Math.max(0, seq.length - window) : 0;
  for (let i = start; i < seq.length; i++) c[seq[i]]++;
  const tot = c[0] + c[1] + c[2];
  if (!tot) return [...HUMAN_THROW_PRIOR];
  return [c[0] / tot, c[1] / tot, c[2] / tot];
}

function bestEvThrow(mix: number[]) {
  let best = 0;
  let bestEv = -Infinity;
  for (let i = 0; i < 3; i++) {
    const ev = mix[(i + 2) % 3] - mix[(i + 1) % 3];
    if (ev > bestEv) {
      bestEv = ev;
      best = i;
    }
  }
  return best;
}

function evHumanPred(seq: number[], window?: number, prior = false) {
  if (!seq.length && !prior) return null;
  return option(bestEvThrow(throwMix(seq, window, prior)), -1);
}

function historyNext(seq: number[], maxK = 12): number | null {
  const n = seq.length;
  if (n < 2) return null;
  for (let k = Math.min(maxK, n - 1); k >= 1; k--) {
    const start = n - k;
    for (let i = start - 1; i >= 0; i--) {
      let ok = true;
      for (let j = 0; j < k; j++) {
        if (seq[i + j] !== seq[start + j]) {
          ok = false;
          break;
        }
      }
      if (ok) return seq[i + k];
    }
  }
  return null;
}

type ExpertFn = (humans: number[], ais: number[]) => number | null;

function recencyPredict(humans: number[]) {
  if (!humans.length) return null;
  const c = [0, 0, 0];
  let w = 1;
  for (let i = humans.length - 1; i >= 0; i--) {
    c[humans[i]] += w;
    w *= 0.92;
  }
  return argmax(c);
}

function markovPredict(humans: number[], order: 1 | 2): number | null {
  if (humans.length < order) return freqPredict(humans);
  if (order === 1) {
    const last = humans[humans.length - 1];
    const c = zeros3();
    for (let i = 0; i < humans.length - 1; i++) {
      if (humans[i] === last) c[humans[i + 1]]++;
    }
    return c[0] + c[1] + c[2] ? argmax(c) : freqPredict(humans);
  }
  const a = humans[humans.length - 2];
  const b = humans[humans.length - 1];
  const c = zeros3();
  for (let i = 0; i < humans.length - 2; i++) {
    if (humans[i] === a && humans[i + 1] === b) c[humans[i + 2]]++;
  }
  return c[0] + c[1] + c[2] ? argmax(c) : markovPredict(humans, 1);
}

const EXPERTS: Record<string, ExpertFn> = {
  histH: (h) => historyNext(h),
  histA: (_h, a) => historyNext(a),
  histPair: (h, a) => {
    if (h.length < 3) return null;
    const seq = h.map((human, i) => human + 3 * a[i]);
    const next = historyNext(seq, 10);
    return next == null ? null : next % 3;
  },
  freq: (h) => freqPredict(h),
  recency: (h) => recencyPredict(h),
  markov1: (h) => markovPredict(h, 1),
  markov2: (h) => markovPredict(h, 2),
  repeat: (h) => (h.length ? h[h.length - 1] : null),
  rotate: (h) => (h.length ? option(h[h.length - 1]) : null),
  copyAi: (_h, a) => (a.length ? a[a.length - 1] : null),
  beatAi: (_h, a) => (a.length ? option(a[a.length - 1]) : null),
};

const EXPERT_NAMES = Object.keys(EXPERTS);
const ROTS = [0, 1, 2];

function collectPreds(humans: number[], ais: number[]) {
  const preds: Record<string, number | null> = {};
  for (const name of EXPERT_NAMES) {
    preds[name] = EXPERTS[name](humans, ais);
  }
  return preds;
}

function aiFor(pred: number, rot: number) {
  return option(pred, 1 - rot);
}

export function createIocaineBrain(
  opts?: BrainOpts & {
    decay?: number;
    biasLock?: number;
    nashMix?: number;
    id?: string;
  },
): Brain {
  const rng = rngFrom(opts);
  const matches: Match[] = [];
  const humans: number[] = [];
  const ais: number[] = [];
  const scores: Record<string, number[]> = {};
  for (const name of EXPERT_NAMES) scores[name] = [0, 0, 0];
  const decay = opts?.decay ?? 0.93;
  const biasLock = opts?.biasLock ?? 0.55;
  const nashMix = opts?.nashMix ?? 0.0;
  let losses = 0;

  function bestMove() {
    if (!humans.length) return Math.floor(rng() * OPTIONS.length);
    if (humans.length >= 8) {
      const c = zeros3();
      for (const h of humans) c[h]++;
      const total = humans.length;
      const top = argmax(c);
      if (c[top] / total >= biasLock) return option(top);
    }
    if (nashMix > 0 && humans.length >= 12) {
      const recent = humans.slice(-12);
      const c = zeros3();
      for (const h of recent) c[h]++;
      if (Math.max(c[0], c[1], c[2]) <= 6 && rng() < nashMix) {
        return Math.floor(rng() * OPTIONS.length);
      }
    }
    const preds = collectPreds(humans, ais);
    let best = -Infinity;
    let move = option(preds.freq ?? 0);
    for (const name of EXPERT_NAMES) {
      const pred = preds[name];
      if (pred == null) continue;
      for (const rot of ROTS) {
        const s = scores[name][rot];
        if (s > best) {
          best = s;
          move = aiFor(pred, rot);
        }
      }
    }
    if (losses >= 4 && rng() < 0.35) {
      return coveringThrow(move, option(move));
    }
    return move;
  }

  return {
    id: opts?.id ?? "iocaine",
    decide: bestMove,
    learn(human, ai) {
      const preds = collectPreds(humans, ais);
      for (const name of EXPERT_NAMES) {
        const pred = preds[name];
        for (const rot of ROTS) {
          scores[name][rot] *= decay;
          if (pred != null) {
            scores[name][rot] += payoff(human, aiFor(pred, rot));
          }
        }
      }
      const winner = getWinner(human, ai);
      losses = winner === HUMAN ? losses + 1 : winner === AI ? 0 : losses;
      humans.push(human);
      ais.push(ai);
      matches.push({ human, ai });
    },
    getMatches: () => matches,
  };
}

function markovOrder(seq: number[], order: number): number | null {
  const n = seq.length;
  if (n <= order) return null;
  const c = zeros3();
  for (let i = 0; i + order < n; i++) {
    let ok = true;
    for (let j = 0; j < order; j++) {
      if (seq[i + j] !== seq[n - order + j]) {
        ok = false;
        break;
      }
    }
    if (ok) c[seq[i + order]]++;
  }
  const tot = c[0] + c[1] + c[2];
  const min = order >= 3 ? 2 : 1;
  return tot >= min ? argmax(c) : null;
}

function freqWindow(seq: number[], window: number): number | null {
  if (!seq.length) return null;
  const start = Math.max(0, seq.length - window);
  const c = zeros3();
  for (let i = start; i < seq.length; i++) c[seq[i]]++;
  return argmax(c);
}

function antiRotate(seq: number[]): number | null {
  if (seq.length < 2) return null;
  const n = seq.length;
  const rot = (seq[n - 1] - seq[n - 2] + 3) % 3;
  if (seq.length >= 3) {
    const prev = (seq[n - 2] - seq[n - 3] + 3) % 3;
    if (prev !== rot) return (seq[n - 1] + rot) % 3;
  }
  return (seq[n - 1] + rot) % 3;
}

function outcomePredict(humans: number[], winners: number[]): number | null {
  if (humans.length < 2) return null;
  const lastH = humans[humans.length - 1];
  const lastW = winners[winners.length - 1];
  const c = zeros3();
  for (let i = 0; i < humans.length - 1; i++) {
    if (humans[i] === lastH && winners[i] === lastW) c[humans[i + 1]]++;
  }
  return c[0] + c[1] + c[2] ? argmax(c) : null;
}

type ContestExpert = {
  name: string;
  kind: "opp" | "self";
  predict: () => number | null;
};

/**
 * Contest ensemble: Iocaine-style 6-way meta-rotation over a richer
 * predictor pool (multi-order Markov, rfind, outcome-conditioned,
 * Greenberg windowed freq) with decayed scoring and a random floor.
 * Opening is best EV vs the population mix (rock), not mode-counter.
 */
export function createContestBrain(
  opts?: BrainOpts & { decay?: number; epsilon?: number; id?: string },
): Brain {
  const rng = rngFrom(opts);
  const matches: Match[] = [];
  const humans: number[] = [];
  const ais: number[] = [];
  const winners: number[] = [];
  const decay = opts?.decay ?? 0.88;
  const epsilon = opts?.epsilon ?? 0;

  const experts: ContestExpert[] = [
    { name: "histH", kind: "opp", predict: () => historyNext(humans) },
    { name: "histA", kind: "self", predict: () => historyNext(ais) },
    {
      name: "histPair",
      kind: "opp",
      predict: () => {
        if (humans.length < 3) return null;
        const seq = humans.map((h, i) => h + 3 * ais[i]);
        const next = historyNext(seq, 10);
        return next == null ? null : next % 3;
      },
    },
    { name: "freq", kind: "opp", predict: () => freqPredict(humans, true) },
    { name: "freq8", kind: "opp", predict: () => freqWindow(humans, 8) },
    { name: "freq16", kind: "opp", predict: () => freqWindow(humans, 16) },
    {
      name: "ev",
      kind: "opp",
      predict: () => evHumanPred(humans, undefined, true),
    },
    {
      name: "ev8",
      kind: "opp",
      predict: () => (humans.length ? evHumanPred(humans, 8) : null),
    },
    {
      name: "ev16",
      kind: "opp",
      predict: () => (humans.length ? evHumanPred(humans, 16) : null),
    },
    { name: "boltz", kind: "opp", predict: () => recencyPredict(humans) },
    { name: "m1", kind: "opp", predict: () => markovOrder(humans, 1) },
    { name: "m2", kind: "opp", predict: () => markovOrder(humans, 2) },
    { name: "m3", kind: "opp", predict: () => markovOrder(humans, 3) },
    { name: "m4", kind: "opp", predict: () => markovOrder(humans, 4) },
    { name: "m5", kind: "opp", predict: () => markovOrder(humans, 5) },
    {
      name: "repeat",
      kind: "opp",
      predict: () => (humans.length ? humans[humans.length - 1] : null),
    },
    {
      name: "rotate",
      kind: "opp",
      predict: () => (humans.length ? option(humans[humans.length - 1]) : null),
    },
    { name: "antiRot", kind: "opp", predict: () => antiRotate(humans) },
    {
      name: "antiStreak",
      kind: "opp",
      predict: () => {
        const n = humans.length;
        if (n < 2 || humans[n - 1] !== humans[n - 2]) return null;
        return option(humans[n - 1]);
      },
    },
    {
      name: "copyAi",
      kind: "opp",
      predict: () => (ais.length ? ais[ais.length - 1] : null),
    },
    {
      name: "beatAi",
      kind: "opp",
      predict: () => (ais.length ? option(ais[ais.length - 1]) : null),
    },
    {
      name: "outcome",
      kind: "opp",
      predict: () => outcomePredict(humans, winners),
    },
    {
      name: "wsls",
      kind: "opp",
      predict: () => {
        if (!humans.length) return null;
        const w = winners[winners.length - 1];
        if (w === HUMAN) return humans[humans.length - 1];
        if (w === AI) return option(ais[ais.length - 1]);
        return option(humans[humans.length - 1]);
      },
    },
    {
      name: "lsws",
      kind: "opp",
      predict: () => {
        if (!humans.length) return null;
        const w = winners[winners.length - 1];
        if (w === AI) return humans[humans.length - 1];
        if (w === HUMAN) return option(ais[ais.length - 1]);
        return humans[humans.length - 1];
      },
    },
  ];

  const scores = experts.map(() => [0, 0, 0]);

  function metaMove(pred: number, kind: "opp" | "self", rot: number) {
    return kind === "opp" ? option(pred, 1 - rot) : option(pred, 2 - rot);
  }

  function bestMove() {
    if (!humans.length) return bestEvThrow([...HUMAN_THROW_PRIOR]);
    let best = -Infinity;
    let move = Math.floor(rng() * OPTIONS.length);
    for (let i = 0; i < experts.length; i++) {
      const pred = experts[i].predict();
      if (pred == null) continue;
      for (let rot = 0; rot < 3; rot++) {
        if (scores[i][rot] > best) {
          best = scores[i][rot];
          move = metaMove(pred, experts[i].kind, rot);
        }
      }
    }
    if (!Number.isFinite(best) || best < -1.5) {
      return Math.floor(rng() * OPTIONS.length);
    }
    if (humans.length >= 8 && rng() < epsilon) {
      return Math.floor(rng() * OPTIONS.length);
    }
    return move;
  }

  return {
    id: opts?.id ?? "iocaine-plus",
    decide: bestMove,
    learn(human, ai) {
      for (let i = 0; i < experts.length; i++) {
        const pred = experts[i].predict();
        for (let rot = 0; rot < 3; rot++) {
          scores[i][rot] *= decay;
          if (pred != null) {
            scores[i][rot] += payoff(
              human,
              metaMove(pred, experts[i].kind, rot),
            );
          }
        }
      }
      humans.push(human);
      ais.push(ai);
      winners.push(getWinner(human, ai));
      matches.push({ human, ai });
    },
    getMatches: () => matches,
  };
}

export function createGeneticBrain(opts?: BrainOpts): Brain {
  const rng = rngFrom(opts);
  const matches: Match[] = [];
  const humans: number[] = [];
  const ais: number[] = [];
  const names = EXPERT_NAMES;
  const popSize = 16;
  const n = names.length * ROTS.length;

  function randomWeights() {
    return Array.from({ length: n }, () => rng());
  }

  let pop = Array.from({ length: popSize }, () => randomWeights());
  let fit = Array.from({ length: popSize }, () => 0);
  let champ = pop[0].slice();
  let gen = 0;

  function moveFromWeights(w: number[], preds: Record<string, number | null>) {
    const votes = zeros3();
    let k = 0;
    for (const name of names) {
      const pred = preds[name];
      for (const rot of ROTS) {
        const weight = w[k++];
        if (pred == null) continue;
        votes[aiFor(pred, rot)] += weight;
      }
    }
    if (votes[0] + votes[1] + votes[2] === 0) {
      return Math.floor(rng() * OPTIONS.length);
    }
    return argmax(votes);
  }

  function evolve() {
    const ranked = pop
      .map((w, i) => ({ w, f: fit[i] }))
      .sort((a, b) => b.f - a.f);
    champ = ranked[0].w.slice();
    const next: number[][] = [];
    for (let i = 0; i < popSize; i++) {
      if (i < 2) {
        next.push(ranked[i].w.slice());
        continue;
      }
      const a = ranked[i % Math.max(2, Math.floor(popSize / 3))].w;
      const b = ranked[Math.floor(rng() * Math.min(6, popSize))].w;
      const child = a.map((v, j) => {
        let x = rng() < 0.5 ? v : b[j];
        if (rng() < 0.12) x = Math.max(0, x + (rng() - 0.5) * 0.8);
        return x;
      });
      next.push(child);
    }
    pop = next;
    fit = fit.map((f) => f * 0.7);
  }

  return {
    id: "genetic-mix",
    decide() {
      if (!humans.length) return Math.floor(rng() * OPTIONS.length);
      return moveFromWeights(champ, collectPreds(humans, ais));
    },
    learn(human, ai) {
      const preds = collectPreds(humans, ais);
      for (let i = 0; i < popSize; i++) {
        const would = moveFromWeights(pop[i], preds);
        fit[i] = fit[i] * 0.92 + payoff(human, would);
      }
      gen++;
      if (gen % 8 === 0) evolve();
      else {
        let bestI = 0;
        for (let i = 1; i < popSize; i++) if (fit[i] > fit[bestI]) bestI = i;
        champ = pop[bestI].slice();
      }
      humans.push(human);
      ais.push(ai);
      matches.push({ human, ai });
    },
    getMatches: () => matches,
  };
}

export function createArenaBrain(opts?: BrainOpts): Brain {
  const rng = rngFrom(opts);
  const inner = createIocaineBrain({
    ...opts,
    decay: 0.9,
    biasLock: 0.5,
    nashMix: 0.08,
  });
  const matches: Match[] = [];
  let streak = 0;

  return {
    id: "arena",
    decide() {
      const move = inner.decide();
      if (streak >= 3 && rng() < 0.4) {
        return option(move, rng() < 0.5 ? 0 : -1);
      }
      return move;
    },
    learn(human, ai) {
      inner.learn(human, ai);
      const winner = getWinner(human, ai);
      streak = winner === HUMAN ? streak + 1 : winner === AI ? 0 : streak;
      matches.push({ human, ai });
    },
    getMatches: () => matches,
  };
}
