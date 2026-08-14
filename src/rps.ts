import { OPTIONS, createBrain, getWinner, type Brain, type Match } from "./ai";

const VERSION = 2;
const STORAGE = "brain";
const BRAIN_STORAGE = "brainId";
const DEFAULT_BRAIN = "best-of";
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

const aiScoreEl = document.getElementById("ai-score");
const humanScoreEl = document.getElementById("human-score");
const tieScoreEl = document.getElementById("tie-score");
const aiRateEl = document.getElementById("ai-rate");
const scoreHistoryEl = document.getElementById("score-history");

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
  updateScore();
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

const resetButton = document.getElementById(
  "reset",
) as HTMLButtonElement | null;
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

function scoreTotals() {
  let aiWins = 0;
  let humanWins = 0;
  let ties = 0;
  for (const match of matches()) {
    const winner = getWinner(match.human, match.ai);
    if (winner === AI) aiWins++;
    else if (winner === HUMAN) humanWins++;
    else ties++;
  }
  return { aiWins, humanWins, ties, total: matches().length };
}

function syncResetButton() {
  if (resetButton) resetButton.disabled = matches().length === 0;
}

function updateScore() {
  const { aiWins, humanWins, ties, total } = scoreTotals();
  if (aiScoreEl) aiScoreEl.textContent = String(aiWins);
  if (humanScoreEl) humanScoreEl.textContent = String(humanWins);
  if (tieScoreEl) tieScoreEl.textContent = String(ties);
  if (aiRateEl) {
    if (!total) {
      aiRateEl.textContent = "—";
      aiRateEl.className = "rate-neutral";
    } else {
      const pct = Math.round((100 * aiWins) / total);
      aiRateEl.textContent = pct + "%";
      aiRateEl.className =
        pct > 50 ? "rate-win" : pct < 50 ? "rate-lose" : "rate-even";
    }
  }
  if (!scoreHistoryEl) return;
  scoreHistoryEl.replaceChildren();
  for (const match of matches()) {
    const winner = getWinner(match.human, match.ai);
    const dot = document.createElement("span");
    dot.className =
      "score-dot " +
      (winner === AI ? "ai" : winner === HUMAN ? "human" : "tie");
    dot.title = winner === AI ? "AI won" : winner === HUMAN ? "You won" : "Tie";
    scoreHistoryEl.appendChild(dot);
  }
  scoreHistoryEl.scrollTop = scoreHistoryEl.scrollHeight;
  syncResetButton();
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
  if (!queue.length) {
    updateScore();
    return startGame();
  }
  step(queue);
}

function step(queue: Array<[number, number]>) {
  if (!queue.length) return startGame();
  const [human, ai] = queue.shift()!;
  brain.learn(human, ai);
  updateScore();
  setTimeout(step, 100, queue);
}

setState("initializing");
resetHands();
init();
