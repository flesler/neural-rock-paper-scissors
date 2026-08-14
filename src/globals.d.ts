declare const neataptic: {
  architect: {
    LSTM: new (
      inputs: number,
      memory: number,
      outputs: number,
    ) => {
      activate: (input: number[]) => number[];
      train: (
        data: Array<{ input: number[]; output: number[] }>,
        options: Record<string, unknown>,
      ) => { error: number };
      error: number;
    };
  };
  methods: {
    rate: { FIXED: () => unknown };
    cost: { CROSS_ENTROPY: unknown };
  };
};
