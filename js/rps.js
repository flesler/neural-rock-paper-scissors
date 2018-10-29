//---------------------- AI --------------------------------//
// TODO: Add more inputs, the relative step from previous choice
// TODO: Maybe add an output on how likely to get tricked and do the opposite
(function() {

// neural network settings

const TRAIN_OPTIONS = {
	log: 0,
	iterations: 6000,
	error: 0.03,
	clear: true,
	rate: 0.9,
	momentum: 0.9
}

const MEMORY_BLOCKS = 8
const OUTPUTS = 3
const INPUTS = OUTPUTS + 2
// Change to invalidate stored state
const VERSION = 2

// Values

const HUMAN = 0
const AI = 1
const TIE = 0.5

const STORAGE = 'brain'

const stored = JSON.parse(localStorage.getItem(STORAGE) || '{}')
const ai = stored.version === VERSION ? neataptic.Network.fromJSON(stored) :
		new neataptic.architect.LSTM(INPUTS, MEMORY_BLOCKS, OUTPUTS);

let matches = []

ai.reset = function() {
	localStorage.removeItem(STORAGE)
}

ai.record = function (choice, result) {
	const prev = matches[matches.length - 1]
	const match = createMatch(choice, result, prev)
	if (prev) {
		ai.train([{ input: prev.input, output: match.output }], TRAIN_OPTIONS)

		const state = ai.toJSON()
		state.version = VERSION
		localStorage.setItem(STORAGE, JSON.stringify(state))
	}
	matches.push(match)
}

ai.predict = function () {
	const prev = matches[matches.length - 1]
	if (!prev) {
		return Math.floor(Math.random() * OUTPUTS)
	}
	const output = ai.activate(prev.input)
	let max = output[0]
	let prediction = 0
	for (let i = 1; i < output.length; i++) {
		if (output[i] > max) {
			max = output[i]
			prediction = i
		}
	}

	return prediction
}

function createMatch(choice, result, prev) {
	const match = { choice: choice, result: result, input: [], output: [] }
	const input = match.input
	for (let i = 0; i < OUTPUTS; i++) {
		input[i] = match.output[i] = i === choice ? 1 : 0
	}
	// Track who won
	input.push(result)
	// Track if they picked the same option again
	input.push(prev && prev.choice === choice ? 1 : 0)
	return match
}

//---------------------- UI --------------------------------//

const OPTIONS = ['rock', 'paper', 'scissors']
const MAX_DATAPOINTS = 15

// Game flow

let prediction

function startGame() {
	showChoice('human', 0, false)
	showChoice('ai', 0, false)
	setState('thinking')
	prediction = ai.predict()
	setState('ready')
}

function endGame(choice, result) {
	setState('thinking')
	ai.record(choice, result)
	updateChart()
	setState('ended')
	setTimeout(startGame, 3000)
}

function setState(state) {
	document.body.className = state
}

document.getElementById('reset').onclick = function () {
	ai.reset()
	location.reload()
}

// Human choices

document.getElementById('human-choices').onclick = function(e) {
	const choice = OPTIONS.indexOf(e.target.className)
	if (choice === -1) {
		return
	}
	const result = getResult(choice)
	showChoice('human', choice, result !== HUMAN)
	showChoice('ai', beats(prediction), result !== AI)
	endGame(choice, result)
}

function beats(choice) {
	return (choice + 1) % OPTIONS.length
}

function getResult(choice) {
	switch (choice) {
		case prediction: return AI
		case beats(prediction): return TIE
		default: return HUMAN
	}
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

for (let i = 0; i < MAX_DATAPOINTS; i++) {
	chart.addDataPoint('', [0, 0, 0])
}

function updateChart() {
	let ai = 0
	let human = 0
	for (const match of matches) {
		if (match.result === AI) {
			ai++
		} else if (match.result === HUMAN) {
			human++
		}
	}
	const total = matches.length
	const tie = total - ai - human

	chart.removeDataPoint(0)
	chart.addDataPoint(total.toString(), [ai, tie, human])
}

startGame()

})()
