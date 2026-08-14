import {
  AI,
  HUMAN,
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

function freqPredict(humans: number[]) {
  if (!humans.length) return null;
  const c = zeros3();
  for (const h of humans) c[h]++;
  return argmax(c);
}

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
  opts?: BrainOpts & { decay?: number; biasLock?: number; nashMix?: number; id?: string },
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
    id: "genetic",
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
