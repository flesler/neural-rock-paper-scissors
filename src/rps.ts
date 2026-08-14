import { OPTIONS, createBrain, getWinner, type Brain, type Match } from "./ai";

const VERSION = 2;
const STORAGE = "brain";
const BRAIN_STORAGE = "brainId";
const DEFAULT_BRAIN = "best-of";
const MAX_DATAPOINTS = 15;
const HUMAN = 0;
const AI = 1;
const BRAIN_ALIASES: Record<string, string> = { genetic: "best-of" };

const BRAIN_CHOICES = [
  "best-of",
  "iocaine",
  "neural",
  "neural-window",
  "patterns",
  "adaptive",
  "random",
];

const BRAIN_LABELS: Record<string, string> = {
  "best-of": "Best of",
  iocaine: "Iocaine",
  neural: "Neural",
  "neural-window": "Neural window",
  patterns: "Patterns",
  adaptive: "Adaptive",
  random: "Random",
};

function selectedBrainId() {
  const stored = localStorage.getItem(BRAIN_STORAGE) || DEFAULT_BRAIN;
  return BRAIN_ALIASES[stored] || stored;
}

let brain: Brain = createBrain(selectedBrainId());
let prediction = 0;

function matches(): Match[] {
  return brain.getMatches();
}

function resetHands() {
  showChoice("human", 0, false);
  showChoice("ai", 0, false);
}

function startGame() {
  resetHands();
  setState("thinking");
  prediction = brain.decide();
  setState("ready");
  document
    .getElementById("human")
    ?.scrollIntoView({ block: "end", inline: "end", behavior: "smooth" });
}

function endGame(human: number, ai: number) {
  setState("thinking");
  brain.learn(human, ai);
  updateChart();
  save();
  setState("ended");
  setTimeout(startGame, 3000);
}

function setState(state: string) {
  document.body.className = state;
}

function save() {
  const nums = [VERSION];
  for (const match of matches()) {
    nums.push(match.human + match.ai * OPTIONS.length);
  }
  localStorage.setItem(STORAGE, nums.join(""));
}

function showChoice(id: string, choice: number, lost: boolean) {
  const elem = document.getElementById(id) as HTMLImageElement | null;
  if (!elem) return;
  elem.src = "img/" + OPTIONS[choice] + ".png";
  elem.classList.toggle("lost", lost);
}

function choose(human: number) {
  const ai = prediction;
  const winner = getWinner(human, ai);
  showChoice("human", human, winner !== HUMAN);
  showChoice("ai", ai, winner !== AI);
  endGame(human, ai);
}

const resetButton = document.getElementById("reset");
if (resetButton) {
  resetButton.onclick = function () {
    localStorage.removeItem(STORAGE);
    location.reload();
  };
}

const brainSelect = document.getElementById(
  "brain",
) as HTMLSelectElement | null;
if (brainSelect) {
  for (const id of BRAIN_CHOICES) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = BRAIN_LABELS[id] || id;
    if (id === selectedBrainId()) opt.selected = true;
    brainSelect.appendChild(opt);
  }
  brainSelect.onchange = function () {
    localStorage.setItem(BRAIN_STORAGE, brainSelect.value);
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

const blankLabels = Array.from({ length: MAX_DATAPOINTS }, () => " ");
const blankValues = Array.from({ length: MAX_DATAPOINTS }, () => 0);

function chartHeight() {
  const choices = document.getElementById("human-choices");
  const toolbar = document.querySelector(".ai-toolbar");
  const chartEl = document.getElementById("chart");
  const pickerH = choices?.getBoundingClientRect().height ?? 0;
  const toolH = toolbar?.getBoundingClientRect().height ?? 0;
  const panelChrome = 18;
  const available = pickerH - toolH - panelChrome;
  if (available >= 96) return Math.round(available);
  const width = chartEl?.clientWidth || window.innerWidth / 2;
  return Math.round(Math.min(Math.max(width * 0.3, 100), 180));
}

let chart: InstanceType<typeof frappe.Chart>;

function boot() {
  const chartBox = document.getElementById("chart");
  const measuredChartHeight = chartHeight();
  if (chartBox) chartBox.style.height = measuredChartHeight + "px";

  chart = new frappe.Chart("#chart", {
    type: "bar",
    height: measuredChartHeight,
    data: {
      labels: blankLabels,
      datasets: [
        { name: "AI", values: blankValues.slice() },
        { name: "Human", values: blankValues.slice() },
        { name: "Tie", values: blankValues.slice() },
      ],
      yMarkers: [{ label: "", value: 1, options: { labelPos: "right" } }],
    },
    barOptions: {
      spaceRatio: 0.2,
      stacked: 1,
    },
    colors: ["#0F0", "#F00", "#CCC"],
  });

  setState("initializing");
  resetHands();
  init();
}

if (document.readyState === "complete") boot();
else window.addEventListener("load", boot);

function updateChart() {
  let aiWins = 0;
  let humanWins = 0;
  for (const match of matches()) {
    const winner = getWinner(match.human, match.ai);
    if (winner === AI) aiWins++;
    else if (winner === HUMAN) humanWins++;
  }
  const total = matches().length;
  if (chart.data.labels.length >= MAX_DATAPOINTS) {
    chart.removeDataPoint(0);
  }
  chart.data.yMarkers[0].value = total ? humanWins : 1;
  chart.addDataPoint(String(total), [
    aiWins,
    humanWins,
    total - (aiWins + humanWins),
  ]);
}

function init() {
  const queue: Array<[number, number]> = [];
  const raw = localStorage.getItem(STORAGE);
  if (raw) {
    const nums = raw.split("").map((n) => parseInt(n, 10));
    if (nums.shift() !== VERSION) nums.length = 0;
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
  if (!queue.length) return startGame();
  const [human, ai] = queue.shift()!;
  brain.learn(human, ai);
  updateChart();
  setTimeout(step, 100, queue);
}
