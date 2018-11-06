//---------------------- AI --------------------------------//
// TODO: Add more inputs, the relative step from previous choice
// TODO: I think I should overhaul so that outputs are delta from last and not options
// TODO: Maybe add an output on how likely to get tricked and do the opposite (2 lossses or 2 wins)
(function() {

// neural network settings

const TRAIN_OPTIONS = {
	log: 0,
	iterations: 6000,
	error: 0.2,
	clear: true,
	rate: 0.3,
	momentum: 0.3
}

const OPTIONS = ['rock', 'paper', 'scissors']
const DIRECTIONS = [-1, 0, 1]

const MEMORY_BLOCKS = 4
const INPUTS = DIRECTIONS.length + 1
const OUTPUTS = DIRECTIONS.length
// Change to invalidate stored state
const VERSION = 2

// Values
const HUMAN = 0
const AI = 1
const TIE = 0.5

const STORAGE = 'brain'

const nn = new neataptic.architect.LSTM(INPUTS, MEMORY_BLOCKS, OUTPUTS);

let matches = []

function record(human, ai) {
	const prev = matches[matches.length - 1]
	const match = createMatch(human, ai, prev)
	if (prev) {
		nn.train([{ input: prev.input, output: match.output }], TRAIN_OPTIONS)
	}
	matches.push(match)
}

function save() {
	const nums = [VERSION]
	for (const match of matches) {
		nums.push(match.human + (match.ai * OPTIONS.length))
	}
	localStorage.setItem(STORAGE, nums.join(''))
}

function option(choice, delta = +1) {
	return (choice + delta + OPTIONS.length) % OPTIONS.length
}

function getWinner(human, ai) {
	switch (ai) {
		case human: return TIE
		case option(human): return AI
		default: return HUMAN
	}
}

function predict() {
	// Do a few random for learning
	if (matches.length - 1 < MEMORY_BLOCKS) {
		return Math.floor(Math.random() * OPTIONS.length)
	}
	const prev = matches[matches.length - 1]
	const output = nn.activate(prev.input)
	let max = output[0]
	let dir = 0
	for (let i = 1; i < DIRECTIONS.length; i++) {
		if (output[i] > max) {
			max = output[i]
			dir = i
		}
	}
	// Try to guess in which direction they'll move from the last one
	return option(prev.human, DIRECTIONS[dir])
}

function createMatch(human, ai, prev) {
	const match = { human: human, ai: ai }
	if (!prev) {
		// First the 2nd match assume no direction
		prev = match
	}

	const dir = option(human - prev.human)
	const output = match.output = []
	for (let i = 0; i < DIRECTIONS.length; i++) {
		output[i] = i === dir ? 1 : 0
	}
	match.input = output.concat(getWinner(human, ai))

	if (match.input.length !== INPUTS) {
		throw new Error('input size mismatch')
	}

	if (output.length !== OUTPUTS) {
		throw new Error('output size mismatch')
	}
	return match
}

//---------------------- UI --------------------------------//

const MAX_DATAPOINTS = 15

// Game flow

let prediction

function startGame() {
	showChoice('human', 0, false)
	showChoice('ai', 0, false)
	setState('thinking')
	prediction = predict()
	setState('ready')
}

function endGame(human, ai) {
	setState('thinking')
	record(human, ai)
	updateChart()
	save()
	setState('ended')
	setTimeout(startGame, 3000)
}

function setState(state) {
	document.body.className = state
}

document.getElementById('reset').onclick = function () {
	localStorage.removeItem(STORAGE)
	location.reload()
}

// Human choices

document.getElementById('human-choices').onclick = function(e) {
	const human = OPTIONS.indexOf(e.target.className)
	if (human === -1) {
		return
	}
	const ai = option(prediction)
	const winner = getWinner(human, ai)
	showChoice('human', human, winner !== HUMAN)
	showChoice('ai', ai, winner !== AI)
	endGame(human, ai)
}

function showChoice(id, choice, lost) {
	const elem = document.getElementById(id)
	OPTIONS.forEach(function(label, i) {
		elem.classList.toggle(label, i === choice)
	})
	elem.classList.toggle('lost', lost)
}

// Chart

const chart = new frappe.Chart('#chart', {
	type: 'bar',
	height: 250,
	data: {
		labels: [],
		datasets: [
			{ name: 'AI', values: [] },
			{ name: 'Tie', values: [] },
			{ name: 'Human', values: [] },
		]
	},
	barOptions: {
		spaceRatio: 0.2,
		stacked: 1,
	},
	colors: ['#0F0', '#CCC', '#F00'],
})

function updateChart() {
	let ai = 0
	let human = 0
	for (const match of matches) {
		const winner = getWinner(match.human, match.ai)
		if (winner === AI) {
			ai++
		} else if (winner === HUMAN) {
			human++
		}
	}
	const total = matches.length
	const tie = total - ai - human

	if (chart.data.labels.length >= MAX_DATAPOINTS) {
		chart.removeDataPoint(0)
	}
	chart.addDataPoint(total || '', [ai, tie, human])
}

function init() {
	const queue = []
	if (localStorage.getItem(STORAGE)) {
		const nums = localStorage.getItem(STORAGE)
			.split('').map(function(n) { return parseInt(n) })
		// If version doesn't match, ignore it
		if (nums.shift() !== VERSION) {
			nums.length = 0
		}
		const len = OPTIONS.length
		for (const num of nums) {
			const human = num % len
			const ai = (num - human) / len
			queue.push([human, ai])
		}
	}

	for (let i = queue.length; i < MAX_DATAPOINTS; i++) {
		updateChart()
	}
	step(queue)
}

function step(queue) {
	if (!queue.length) {
		return startGame()
	}
	const [human, ai] = queue.shift()
	record(human, ai)
	updateChart()
	setTimeout(step, 100, queue)
}

setState('initializing')
init()

})()
