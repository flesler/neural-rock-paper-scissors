export type TrainOptions = {
  iterations?: number;
  errorThresh?: number;
  learningRate?: number;
  momentum?: number;
  log?: boolean | ((status: { iterations: number; error: number }) => void);
};

type BrainNet = {
  initialize: () => void;
  run: (input: number[]) => number[] | Float32Array | Record<string, number>;
  train: (
    data: Array<{ input: number[]; output: number[] }>,
    opts?: TrainOptions,
  ) => { error: number };
  toJSON: () => unknown;
  fromJSON: (json: unknown) => BrainNet;
};

export type RpsNet = {
  activate: (input: number[]) => number[];
  train: (
    data: Array<{ input: number[]; output: number[] }>,
    opts: TrainOptions,
  ) => { error: number };
  toJSON: () => unknown;
};

export type BrainLib = {
  NeuralNetwork: new (options?: {
    inputSize?: number;
    outputSize?: number;
    hiddenLayers?: number[];
  }) => BrainNet;
};

export function getBrain(): BrainLib {
  const fromGlobal = (globalThis as { brain?: BrainLib }).brain;
  if (fromGlobal) return fromGlobal;
  throw new Error(
    "brain.js is not loaded (set globalThis.brain in Node, or include the CDN script in the browser)",
  );
}

function toArray(output: number[] | Float32Array | Record<string, number>) {
  if (output instanceof Float32Array) return Array.from(output);
  if (Array.isArray(output)) return output;
  return Object.values(output);
}

function wrapNet(nn: BrainNet): RpsNet {
  return {
    activate(input) {
      return toArray(nn.run(input));
    },
    train(data, opts) {
      const status = nn.train(
        data.map(({ input, output }) => ({ input, output })),
        opts,
      );
      return { error: status.error };
    },
    toJSON: () => nn.toJSON(),
  };
}

export function createNetwork(
  inputSize: number,
  hiddenSize: number,
  outputSize: number,
): RpsNet {
  const Ctor = getBrain().NeuralNetwork as new (options?: {
    inputSize?: number;
    outputSize?: number;
    hiddenLayers?: number[];
  }) => BrainNet;
  const nn = new Ctor({
    inputSize,
    outputSize,
    hiddenLayers: [hiddenSize],
  });
  nn.initialize();
  return wrapNet(nn);
}

export function cloneNetwork(net: RpsNet): RpsNet {
  const Ctor = getBrain().NeuralNetwork as new () => BrainNet;
  const nn = new Ctor();
  nn.fromJSON(net.toJSON());
  return wrapNet(nn);
}

export function trainOptions(overrides: TrainOptions = {}): TrainOptions {
  return {
    iterations: 8,
    errorThresh: 0.05,
    learningRate: 0.5,
    momentum: 0.3,
    log: false,
    ...overrides,
  };
}
