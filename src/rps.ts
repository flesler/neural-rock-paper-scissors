import {
  OPTIONS,
  createBrain,
  getWinner,
  LEARNING_ROUNDS,
  warmupRounds,
  type Brain,
  type Match,
} from "./ai";
import * as lines from "./status-lines";

const VERSION = 2;
const MATCH_STORAGE = "match";
const BRAIN_STORAGE = "brainId";
const OPEN_STORAGE = "openThrow";
const BRAIN_LABELS = {
  "iocaine-plus": "Iocaine+",
  "best-of": "Best of",
  iocaine: "Iocaine",
  feedforward: "Feedforward",
  "windowed-net": "Windowed net",
  patterns: "Patterns",
  adaptive: "Adaptive",
  random: "Random",
} as const;

type BrainId = keyof typeof BRAIN_LABELS;
const BRAIN_IDS = Object.keys(BRAIN_LABELS) as BrainId[];
const DEFAULT_BRAIN: BrainId = "iocaine-plus";
const HUMAN = 0;
const AI = 1;

function isBrainId(id: string): id is BrainId {
  return Object.prototype.hasOwnProperty.call(BRAIN_LABELS, id);
}

function selectedBrainId(): BrainId {
  const stored = localStorage.getItem(BRAIN_STORAGE);
  return stored && isBrainId(stored) ? stored : DEFAULT_BRAIN;
}

let brain: Brain = createBrain(selectedBrainId());
let prediction = 0;

const aiScoreEl = document.getElementById("ai-score");
const humanScoreEl = document.getElementById("human-score");
const tieScoreEl = document.getElementById("tie-score");
const aiRateEl = document.getElementById("ai-rate");
const scoreHistoryEl = document.getElementById("score-history");
const statusEl = document.getElementById("status-line");

let roundFlashText: string | null = null;
let moodText = "";
const lineIndex: Record<string, number> = {};

function pickLine(key: string, pool: string[]) {
  const i = lineIndex[key] ?? 0;
  lineIndex[key] = (i + 1) % pool.length;
  return pool[i]!;
}

function stageLine(played: number, until: number, verb: string) {
  const left = until - played;
  return left === 1
    ? `${verb} for 1 more turn`
    : `${verb} for ${left} more turns`;
}

function moodLine(
  aiWins: number,
  humanWins: number,
  ties: number,
  total: number,
) {
  const lockIn = warmupRounds(brain.id);
  const learnUntil = Math.min(LEARNING_ROUNDS, lockIn);
  if (lockIn > 0 && total < learnUntil) {
    return stageLine(total, learnUntil, "Learning");
  }
  if (total < lockIn) {
    return stageLine(total, lockIn, "Perfecting");
  }
  if (!total) return pickLine("ready", lines.READY);

  const decided = aiWins + humanWins;
  if (!decided) return pickLine("ready", lines.READY);

  if (total >= 5 && ties / total >= 0.45) {
    return pickLine("tie-heavy", lines.TIE_HEAVY);
  }

  const aiPct = Math.round((100 * aiWins) / decided);
  if (aiPct >= 75) return pickLine("ai-dominate", lines.AI_DOMINATE);
  if (aiPct >= 58) return pickLine("ai-ahead", lines.AI_AHEAD);
  if (aiPct <= 25) return pickLine("human-dominate", lines.HUMAN_DOMINATE);
  if (aiPct <= 42) return pickLine("human-ahead", lines.HUMAN_AHEAD);
  if (aiPct >= 45 && aiPct <= 55) return pickLine("even", lines.EVEN);
  return pickLine("ready", lines.READY);
}

function roundLine(winner: number) {
  if (winner === AI) return pickLine("round-ai", lines.ROUND_AI);
  if (winner === HUMAN) return pickLine("round-human", lines.ROUND_HUMAN);
  return pickLine("round-tie", lines.ROUND_TIE);
}

function refreshMoodText() {
  const { aiWins, humanWins, ties, total } = scoreTotals();
  moodText = moodLine(aiWins, humanWins, ties, total);
}

function syncStatus() {
  if (!statusEl) return;
  const flashing =
    document.body.classList.contains("thinking") ||
    document.body.classList.contains("ended");

  if (flashing && roundFlashText) {
    statusEl.textContent = roundFlashText;
    statusEl.classList.add("blink");
    return;
  }

  statusEl.classList.remove("blink");

  if (document.body.classList.contains("initializing")) {
    statusEl.textContent = "Warming up the AI opponent";
    return;
  }

  statusEl.textContent = moodText;
}

function matches(): Match[] {
  return brain.getMatches();
}

function resetHands() {
  showChoice("human", 0, false);
  showChoice("ai", 0, false);
}

function uiOpeningThrow() {
  const last = parseInt(localStorage.getItem(OPEN_STORAGE) || "", 10);
  const weights = [0.5, 0.32, 0.18];
  if (last >= 0 && last < 3) weights[last]! *= 0.2;
  let sum = 0;
  for (const w of weights) sum += w;
  let r = Math.random() * sum;
  let pick = 0;
  for (let i = 0; i < 3; i++) {
    r -= weights[i]!;
    if (r < 0) {
      pick = i;
      break;
    }
  }
  localStorage.setItem(OPEN_STORAGE, String(pick));
  return pick;
}

function startGame() {
  resetHands();
  setState("thinking");
  prediction =
    matches().length || brain.id === "random"
      ? brain.decide()
      : uiOpeningThrow();
  setState("ready");
  document
    .getElementById("human")
    ?.scrollIntoView({ block: "end", inline: "end", behavior: "smooth" });
}

function endGame(human: number, ai: number) {
  roundFlashText = roundLine(getWinner(human, ai));
  setState("thinking");
  brain.learn(human, ai);
  updateScore();
  save();
  setState("ended");
  setTimeout(startGame, 3000);
}

const GAME_STATES = ["initializing", "ready", "thinking", "ended"];
const CAROUSEL_MS = 750;

let hasPickedOnce = false;
let carouselTimer: ReturnType<typeof setInterval> | null = null;
let carouselIndex = 0;
const pickerImgs: HTMLImageElement[] = [];

function stopCarousel() {
  if (carouselTimer !== null) {
    clearInterval(carouselTimer);
    carouselTimer = null;
  }
  document.body.classList.remove("carousel-pickers");
  for (const img of pickerImgs) {
    img.classList.remove("carousel-highlight");
  }
}

function tickCarousel() {
  for (const img of pickerImgs) {
    img.classList.remove("carousel-highlight");
  }
  const img = pickerImgs[carouselIndex];
  if (img) img.classList.add("carousel-highlight");
  carouselIndex = (carouselIndex + 1) % OPTIONS.length;
}

function startCarousel() {
  if (hasPickedOnce || carouselTimer !== null || !pickerImgs.length) return;
  document.body.classList.add("carousel-pickers");
  carouselIndex = 0;
  tickCarousel();
  carouselTimer = setInterval(tickCarousel, CAROUSEL_MS);
}

function setState(state: string) {
  for (const name of GAME_STATES) {
    document.body.classList.remove(name);
  }
  document.body.classList.add(state);
  if (state === "ready") {
    roundFlashText = null;
    refreshMoodText();
    if (!hasPickedOnce) {
      startCarousel();
    } else {
      stopCarousel();
    }
  } else {
    stopCarousel();
  }
  syncStatus();
}

function save() {
  const nums = [VERSION];
  for (const match of matches()) {
    nums.push(match.human + match.ai * OPTIONS.length);
  }
  localStorage.setItem(MATCH_STORAGE, nums.join(""));
}

function showChoice(id: string, choice: number, lost: boolean) {
  const elem = document.getElementById(id) as HTMLImageElement | null;
  if (!elem) return;
  elem.src = "img/" + OPTIONS[choice] + ".png";
  elem.classList.toggle("lost", lost);
}

function choose(human: number) {
  if (!hasPickedOnce) {
    hasPickedOnce = true;
    stopCarousel();
  }
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
    localStorage.removeItem(MATCH_STORAGE);
    location.reload();
  };
}

const brainSelect = document.getElementById(
  "brain",
) as HTMLSelectElement | null;
if (brainSelect) {
  for (const id of BRAIN_IDS) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = BRAIN_LABELS[id];
    if (id === selectedBrainId()) opt.selected = true;
    brainSelect.appendChild(opt);
  }
  brainSelect.onchange = function () {
    if (!isBrainId(brainSelect.value)) return;
    localStorage.setItem(BRAIN_STORAGE, brainSelect.value);
    localStorage.removeItem(MATCH_STORAGE);
    location.reload();
  };
}

const humanChoices = document.getElementById("human-choices");
OPTIONS.forEach(function (name, choice) {
  const img = document.createElement("img");
  img.className = name;
  img.draggable = false;
  img.src = "img/" + name + ".png";
  img.onclick = function () {
    choose(choice);
  };
  pickerImgs.push(img);
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
  const { aiWins, humanWins, ties } = scoreTotals();
  if (aiScoreEl) aiScoreEl.textContent = String(aiWins);
  if (humanScoreEl) humanScoreEl.textContent = String(humanWins);
  if (tieScoreEl) tieScoreEl.textContent = String(ties);
  if (aiRateEl) {
    const decided = aiWins + humanWins;
    if (!decided) {
      aiRateEl.textContent = "—";
      aiRateEl.className = "rate-neutral";
    } else {
      const pct = Math.round((100 * aiWins) / decided);
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
  refreshMoodText();
  syncStatus();
}

function init() {
  const queue: Array<[number, number]> = [];
  const raw = localStorage.getItem(MATCH_STORAGE);
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
refreshMoodText();
resetHands();
init();
