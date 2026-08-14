export type NeatapticNet = {
  activate: (input: number[]) => number[];
  train: (
    data: Array<{ input: number[]; output: number[] }>,
    options: Record<string, unknown>,
  ) => { error: number };
  evolve?: (
    data: Array<{ input: number[]; output: number[] }>,
    options: Record<string, unknown>,
  ) => Promise<{ error: number }>;
  error?: number;
};

export type NeatapticLib = {
  architect: {
    LSTM: new (
      inputs: number,
      memory: number,
      outputs: number,
    ) => NeatapticNet;
  };
  methods: {
    rate: { FIXED: () => unknown };
    cost: { CROSS_ENTROPY: unknown };
    mutation?: { FFW: unknown };
  };
};

export function getNeataptic(): NeatapticLib {
  const fromGlobal = (globalThis as { neataptic?: NeatapticLib }).neataptic;
  if (fromGlobal) return fromGlobal;
  throw new Error(
    "neataptic is not loaded (set globalThis.neataptic in Node, or include the CDN script in the browser)",
  );
}

export function lstmTrainOptions(overrides: Record<string, unknown> = {}) {
  const neataptic = getNeataptic();
  return {
    log: 0,
    clear: false,
    error: 0.05,
    iterations: 8,
    momentum: 0.3,
    rate: 0.5,
    ratePolicy: neataptic.methods.rate.FIXED(),
    cost: neataptic.methods.cost.CROSS_ENTROPY,
    ...overrides,
  };
}
