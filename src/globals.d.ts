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

declare const frappe: {
  Chart: new (
    selector: string,
    options: Record<string, unknown>,
  ) => {
    data: {
      labels: string[];
      datasets: Array<{ name: string; values: number[] }>;
      yMarkers: Array<{
        label: string;
        value: number;
        options?: Record<string, unknown>;
      }>;
    };
    removeDataPoint: (index: number) => void;
    addDataPoint: (label: string, values: number[]) => void;
  };
};
