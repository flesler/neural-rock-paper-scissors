//---------------------- AI --------------------------------//
// TODO: Maybe add an output on how likely to get tricked and do the opposite (2 lossses or 2 wins)
// TODO: Create a test suite that repeats various patterns and evaluates the win rate on each to find weak spots

type Match = {
  human: number;
  ai: number;
  input?: number[];
  output?: number[];
};

const TRAIN_OPTIONS = {
  log: 0,
  clear: false,
  error: 0.05,
  momentum: 0.3,
  rate: 0.5,
  ratePolicy: neataptic.methods.rate.FIXED(),
  cost: neataptic.methods.cost.CROSS_ENTROPY,
};

const OPTIONS = ["rock", "paper", "scissors"] as const;
const DIRECTIONS = [-1, 0, 1] as const;

const INPUTS = DIRECTIONS.length + OPTIONS.length + 1;
const OUTPUTS = DIRECTIONS.length;
const MEMORY_BLOCKS = OUTPUTS;
// Change to invalidate stored state
const VERSION = 2;

// Values
const HUMAN = 0;
const AI = 1;
const TIE = 0.5;

const STORAGE = "brain";

const nn = new neataptic.architect.LSTM(INPUTS, MEMORY_BLOCKS, OUTPUTS);

const matches: Match[] = [];

function record(human: number, ai: number) {
  const prev = matches[matches.length - 1];
  const match = createMatch(human, ai, prev);
  if (prev?.input && match.output) {
    const { error } = nn.train(
      [{ input: prev.input, output: match.output }],
      TRAIN_OPTIONS,
    );
    nn.error = error;
  }
  matches.push(match);
}

function save() {
  const nums = [VERSION];
  for (const match of matches) {
    nums.push(match.human + match.ai * OPTIONS.length);
  }
  localStorage.setItem(STORAGE, nums.join(""));
}

function option(choice: number, delta = +1) {
  return (choice + delta + OPTIONS.length) % OPTIONS.length;
}

function getWinner(human: number, ai: number) {
  switch (ai) {
    case human:
      return TIE;
    case option(human):
      return AI;
    default:
      return HUMAN;
  }
}

function predict() {
  // Do a few random for learning
  if (matches.length - 1 < MEMORY_BLOCKS) {
    return Math.floor(Math.random() * OPTIONS.length);
  }
  const prev = matches[matches.length - 1];
  const output = nn.activate(prev.input!);
  let max = output[0];
  let dir = 0;
  for (let i = 1; i < DIRECTIONS.length; i++) {
    if (output[i] > max) {
      max = output[i];
      dir = i;
    }
  }
  // Try to guess in which direction they'll move from the last one
  return option(prev.human, DIRECTIONS[dir]);
}

function createMatch(human: number, ai: number, prev?: Match) {
  const match: Match = { human, ai };
  if (!prev) {
    // First the 2nd match assume no direction
    prev = match;
  }

  const dir = option(human - prev.human);
  const output: number[] = [];
  for (let i = 0; i < DIRECTIONS.length; i++) {
    output[i] = i === dir ? 1 : 0;
  }
  match.output = output;

  const input = output.concat();
  for (let i = 0; i < OPTIONS.length; i++) {
    input.push(i === human ? 1 : 0);
  }
  input.push(getWinner(human, ai));
  match.input = input;

  if (input.length !== INPUTS) {
    throw new Error("input size mismatch");
  }
  if (output.length !== OUTPUTS) {
    throw new Error("output size mismatch");
  }
  return match;
}

//---------------------- UI --------------------------------//

const MAX_DATAPOINTS = 15;

let prediction: number;

function reset() {
  showChoice("human", 0, false);
  showChoice("ai", 0, false);
}

function startGame() {
  reset();
  setState("thinking");
  prediction = predict();
  setState("ready");

  document
    .getElementById("human")
    ?.scrollIntoView({ block: "end", inline: "end", behavior: "smooth" });
}

function endGame(human: number, ai: number) {
  setState("thinking");
  record(human, ai);
  updateChart();
  save();
  setState("ended");
  setTimeout(startGame, 3000);
}

function setState(state: string) {
  document.body.className = state;
}

const resetButton = document.getElementById("reset");
if (resetButton) {
  resetButton.onclick = function () {
    localStorage.removeItem(STORAGE);
    location.reload();
  };
}

const humanChoices = document.getElementById("human-choices");
OPTIONS.forEach(function (name, choice) {
  const img = document.createElement("img");
  img.className = name;
  img.src = "img/" + name + ".png";
  img.onclick = function () {
    choose(choice);
  };
  humanChoices?.appendChild(img);
});

function choose(human: number) {
  const ai = option(prediction);
  const winner = getWinner(human, ai);
  showChoice("human", human, winner !== HUMAN);
  showChoice("ai", ai, winner !== AI);
  endGame(human, ai);
}

function showChoice(id: string, choice: number, lost: boolean) {
  const elem = document.getElementById(id) as HTMLImageElement | null;
  if (!elem) return;
  elem.src = "img/" + OPTIONS[choice] + ".png";
  elem.classList.toggle("lost", lost);
}

const blankLabels = Array.from({ length: MAX_DATAPOINTS }, function () {
  return " ";
});
const blankValues = Array.from({ length: MAX_DATAPOINTS }, function () {
  return 0;
});

const chart = new frappe.Chart("#chart", {
  type: "bar",
  // Frappe spends ~130px on legend/padding; size the rest from column width
  // so a half-page chart does not flatten into a strip.
  height: (function chartHeight() {
    const el = document.getElementById("chart");
    const width = (el && el.clientWidth) || window.innerWidth / 2;
    const chrome = 130;
    const total = chrome + Math.round(width * 0.4);
    return Math.round(
      Math.min(Math.max(total, 200), Math.min(320, window.innerHeight * 0.36)),
    );
  })(),
  data: {
    labels: blankLabels,
    datasets: [
      { name: "AI", values: blankValues.slice() },
      { name: "Human", values: blankValues.slice() },
      { name: "Tie", values: blankValues.slice() },
    ],
    // Marker 1 gives a non-zero y-scale so empty bars don't produce NaN/Infinity SVG attrs
    yMarkers: [{ label: "", value: 1, options: { labelPos: "right" } }],
  },
  barOptions: {
    spaceRatio: 0.2,
    stacked: 1,
  },
  colors: ["#0F0", "#F00", "#CCC"],
});

function updateChart() {
  let ai = 0;
  let human = 0;
  for (const match of matches) {
    const winner = getWinner(match.human, match.ai);
    if (winner === AI) {
      ai++;
    } else if (winner === HUMAN) {
      human++;
    }
  }
  const total = matches.length;

  if (chart.data.labels.length >= MAX_DATAPOINTS) {
    chart.removeDataPoint(0);
  }
  // The marker at human (from 0) is more indicative than avg(ai, human).
  // Keep a 1-unit scale while there are no matches so Frappe can layout bars.
  chart.data.yMarkers[0].value = total ? human : 1;
  chart.addDataPoint(String(total), [ai, human, total - (ai + human)]);
}

function init() {
  const queue: Array<[number, number]> = [];
  if (localStorage.getItem(STORAGE)) {
    const nums = localStorage
      .getItem(STORAGE)!
      .split("")
      .map(function (n) {
        return parseInt(n, 10);
      });
    // If version doesn't match, ignore it
    if (nums.shift() !== VERSION) {
      nums.length = 0;
    }
    const len = OPTIONS.length;
    for (const num of nums) {
      const human = num % len;
      const ai = (num - human) / len;
      queue.push([human, ai]);
    }
  }

  step(queue);
}

function step(queue: Array<[number, number]>) {
  if (!queue.length) {
    return startGame();
  }
  const [human, ai] = queue.shift()!;
  record(human, ai);
  updateChart();
  setTimeout(step, 100, queue);
}

setState("initializing");
reset();
init();
