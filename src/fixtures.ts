import { AI, HUMAN, type Opponent, getWinner, option } from "./ai/core";

type History = Array<{ human: number; ai: number; winner: number }>;
type FixtureFactory = (opts?: { rng?: () => number }) => Opponent;

function createOpponent(play: (history: History) => number): Opponent {
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
  };
}

function last(history: History) {
  return history[history.length - 1];
}

const FACTORIES: Record<string, FixtureFactory> = {
  "always-rock": () => createOpponent(() => 0),
  "always-paper": () => createOpponent(() => 1),
  "always-scissors": () => createOpponent(() => 2),
  cycle: () =>
    createOpponent((history) => history.length % 3),
  "cycle-rev": () =>
    createOpponent((history) => (3 - (history.length % 3)) % 3),
  repeat: () =>
    createOpponent((history) => (history.length ? last(history).human : 0)),
  "win-stay-lose-shift": () =>
    createOpponent((history) => {
      if (!history.length) return 0;
      const prev = last(history);
      if (prev.winner === HUMAN) return prev.human;
      if (prev.winner === AI) return option(prev.ai);
      return option(prev.human);
    }),
  "copy-ai": () =>
    createOpponent((history) => (history.length ? last(history).ai : 0)),
  "beat-ai": () =>
    createOpponent((history) =>
      history.length ? option(last(history).ai) : 0,
    ),
  "switch-5": () =>
    createOpponent((history) => Math.floor(history.length / 5) % 3),
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
    return createOpponent(() => Math.floor(rng() * 3));
  },
};

export const FIXTURE_IDS = Object.keys(FACTORIES);

export function createFixture(
  id: string,
  opts?: { rng?: () => number },
): Opponent {
  const factory = FACTORIES[id];
  if (!factory) throw new Error("unknown fixture: " + id);
  const opponent = factory(opts);
  opponent.id = id;
  return opponent;
}
