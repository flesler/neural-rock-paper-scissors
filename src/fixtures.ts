import { type Opponent, getWinner, option, HUMAN_THROW_PRIOR } from "./ai/core";

type History = Array<{ human: number; ai: number; winner: number }>;
type FixtureOpts = { rng?: () => number; rounds?: number };
type Play = (history: History, rng: () => number) => number;
type FixtureFactory = (opts?: FixtureOpts) => Opponent;

function createOpponent(
  play: (history: History) => number,
  segment?: (round: number) => string,
): Opponent {
  const history: History = [];
  return {
    reset() {
      history.length = 0;
    },
    decide() {
      return play(history);
    },
    learn(human, ai) {
      history.push({ human, ai, winner: getWinner(human, ai) });
    },
    segment,
  };
}

function last(history: History) {
  return history[history.length - 1];
}

const BASE: Record<string, Play> = {
  rock: () => 0,
  paper: () => 1,
  scissors: () => 2,
  cycle: (h) => h.length % 3,
  "cycle-rev": (h) => (3 - (h.length % 3)) % 3,
  repeat: (h) => (h.length ? last(h).human : 0),
  wsls: (h) => {
    if (!h.length) return 0;
    const prev = last(h);
    if (prev.winner === 0) return prev.human;
    if (prev.winner === 1) return option(prev.ai);
    return option(prev.human);
  },
  "copy-ai": (h) => (h.length ? last(h).ai : 0),
  "beat-ai": (h) => (h.length ? option(last(h).ai) : 0),
  oscillate: (h) => h.length % 2,
  random: (_h, rng) => Math.floor(rng() * 3),
  population: (_h, rng) => {
    const r = rng();
    let acc = 0;
    for (let i = 0; i < HUMAN_THROW_PRIOR.length; i++) {
      acc += HUMAN_THROW_PRIOR[i];
      if (r < acc) return i;
    }
    return 2;
  },
  humanish: (h, rng) => {
    if (!h.length) return Math.floor(rng() * 3);
    const r = rng();
    if (r < 0.5) return last(h).human;
    if (r < 0.75) return option(last(h).human);
    return Math.floor(rng() * 3);
  },
};

function splitSegment(names: string[], rounds: number) {
  const size = Math.max(1, Math.floor(rounds / names.length));
  return (round: number) => {
    const i = Math.min(names.length - 1, Math.floor(round / size));
    return names[i];
  };
}

function intervalSegment(names: string[], every: number) {
  return (round: number) => names[Math.floor(round / every) % names.length];
}

function compose(
  names: string[],
  segment: (round: number) => string,
  rng: () => number,
): Opponent {
  return createOpponent(
    (history) => BASE[segment(history.length)](history, rng),
    segment,
  );
}

const FACTORIES: Record<string, FixtureFactory> = {
  "always-rock": () => createOpponent(() => 0),
  "always-paper": () => createOpponent(() => 1),
  "always-scissors": () => createOpponent(() => 2),
  cycle: () => createOpponent((history) => BASE.cycle(history, Math.random)),
  "cycle-rev": () =>
    createOpponent((history) => BASE["cycle-rev"](history, Math.random)),
  repeat: () => createOpponent((history) => BASE.repeat(history, Math.random)),
  "win-stay-lose-shift": () =>
    createOpponent((history) => BASE.wsls(history, Math.random)),
  "copy-ai": () =>
    createOpponent((history) => BASE["copy-ai"](history, Math.random)),
  "beat-ai": () =>
    createOpponent((history) => BASE["beat-ai"](history, Math.random)),
  "switch-5": () =>
    createOpponent((history) => Math.floor(history.length / 5) % 3),
  "switch-8": () =>
    createOpponent((history) => Math.floor(history.length / 8) % 3),
  "anti-repeat": () =>
    createOpponent((history) => {
      if (history.length < 2) return 0;
      const a = last(history).ai;
      const b = history[history.length - 2].ai;
      return a === b ? option(a) : a;
    }),
  oscillate: () =>
    createOpponent((history) => BASE.oscillate(history, Math.random)),
  "double-cycle": () =>
    createOpponent((history) => Math.floor(history.length / 2) % 3),
  "bait-flip": () =>
    createOpponent((history) => {
      const phase = Math.floor(history.length / 7) % 2;
      if (phase === 0) return 0;
      return history.length ? option(last(history).ai) : 1;
    }),
  trickster: () =>
    createOpponent((history) => {
      const phase = Math.floor(history.length / 6) % 2;
      if (phase === 0) return Math.floor(history.length / 6) % 3;
      return history.length ? option(last(history).ai) : 0;
    }),
  whiplash: () =>
    createOpponent((history) => {
      const block = Math.floor(history.length / 6);
      const i = history.length % 6;
      return block % 2 === 0 ? i % 3 : (3 - (i % 3)) % 3;
    }),
  "meta-farm": () =>
    createOpponent((history) => {
      if (!history.length) return 0;
      return Math.floor(history.length / 5) % 2 === 0
        ? last(history).ai
        : option(last(history).ai);
    }),
  "phase-rand": (opts) => {
    const rng = opts?.rng ?? Math.random;
    return createOpponent((history) => {
      if (history.length % 20 < 10) return history.length % 3;
      return Math.floor(rng() * 3);
    });
  },
  humanish: (opts) => {
    const rng = opts?.rng ?? Math.random;
    return createOpponent((history) => BASE.humanish(history, rng));
  },
  "biased-rock": (opts) => {
    const rng = opts?.rng ?? Math.random;
    return createOpponent(() => {
      const r = rng();
      if (r < 0.7) return 0;
      if (r < 0.85) return 1;
      return 2;
    });
  },
  random: (opts) => {
    const rng = opts?.rng ?? Math.random;
    return createOpponent(() => BASE.random([], rng));
  },
  population: (opts) => {
    const rng = opts?.rng ?? Math.random;
    return createOpponent((history) => BASE.population(history, rng));
  },
  "mix-2-cycle-beat": (opts) => {
    const rng = opts?.rng ?? Math.random;
    const rounds = opts?.rounds ?? 80;
    const names = ["cycle", "beat-ai"];
    return compose(names, splitSegment(names, rounds), rng);
  },
  "mix-2-copy-rand": (opts) => {
    const rng = opts?.rng ?? Math.random;
    const rounds = opts?.rounds ?? 80;
    const names = ["copy-ai", "random"];
    return compose(names, splitSegment(names, rounds), rng);
  },
  "mix-2-wsls-osc": (opts) => {
    const rng = opts?.rng ?? Math.random;
    const rounds = opts?.rounds ?? 80;
    const names = ["wsls", "oscillate"];
    return compose(names, splitSegment(names, rounds), rng);
  },
  "mix-3-cycle-beat-rand": (opts) => {
    const rng = opts?.rng ?? Math.random;
    const rounds = opts?.rounds ?? 80;
    const names = ["cycle", "beat-ai", "random"];
    return compose(names, splitSegment(names, rounds), rng);
  },
  "mix-3-copy-human-wsls": (opts) => {
    const rng = opts?.rng ?? Math.random;
    const rounds = opts?.rounds ?? 80;
    const names = ["copy-ai", "humanish", "wsls"];
    return compose(names, splitSegment(names, rounds), rng);
  },
  "every-10-cycle-beat": (opts) => {
    const rng = opts?.rng ?? Math.random;
    const names = ["cycle", "beat-ai"];
    return compose(names, intervalSegment(names, 10), rng);
  },
  "every-10-cycle-copy-beat": (opts) => {
    const rng = opts?.rng ?? Math.random;
    const names = ["cycle", "copy-ai", "beat-ai"];
    return compose(names, intervalSegment(names, 10), rng);
  },
};

export const FIXTURE_IDS = Object.keys(FACTORIES);

export function createFixture(id: string, opts?: FixtureOpts): Opponent {
  const factory = FACTORIES[id];
  if (!factory) throw new Error("unknown fixture: " + id);
  const opponent = factory(opts);
  opponent.id = id;
  return opponent;
}
