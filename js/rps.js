//---------------------- AI --------------------------------//
// TODO: Maybe add an output on how likely to get tricked and do the opposite (2 lossses or 2 wins)
// TODO: Create a test suite that repeats various patterns and evaluates the win rate on each to find weak spots
(function() {

// neural network settings

const AI_OPTIONS = {
	log: 0,
	clear: false,
	error: 0.2,
	momentum: 0.3,
	rate: 0.4,
	ratePolicy: neataptic.methods.rate.FIXED(),
	cost: neataptic.methods.cost.CROSS_ENTROPY,
	// Evolving
	mutation: neataptic.methods.mutation.FFW,
	mutationRate: 0.4,
	equal: true,
	elitism: 1,
	population: 100,
	iterations: 1000,
	growth: 0.0001,
	amount: 1,
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

let nn
let matches = []

function train(done) {
	const training = []
	for (let i = 1; i < matches.length; i++) {
		training.push({ input: matches[i - 1].input, output: matches[i].output })
	}

	// Start over on each training session
	nn = new neataptic.architect.LSTM(INPUTS, MEMORY_BLOCKS, OUTPUTS)
	if (!training.length) {
		return done()
	}
	nn.evolve(training, AI_OPTIONS).then(done)
	// nn.train(training, AI_OPTIONS); done()
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

function addMatch(human, ai) {
	const match = { human: human, ai: ai }
	// First the 2nd match assume no direction
	const prev = matches[matches.length - 1] || match
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
	matches.push(match)
}

//---------------------- UI --------------------------------//

const MAX_DATAPOINTS = 15

// Game flow

let prediction

function reset() {
	showChoice('human', 0, false)
	showChoice('ai', 0, false)
}

function startGame() {
	reset()
	setState('thinking')
	prediction = predict()
	setState('ready')
}

function endGame(human, ai) {
	setState('thinking')
	addMatch(human, ai)
	updateChart()
	save()
	train(function() {
		setState('ended')
		setTimeout(startGame, 3000)
	})
}

function setState(state) {
	document.body.className = state
}

document.getElementById('reset').onclick = function () {
	localStorage.removeItem(STORAGE)
	location.reload()
}

// Human choices

const humanChoices = document.getElementById('human-choices')
OPTIONS.forEach(function(option, choice) {
	const img = document.createElement('img')
	img.className = option
	img.src = 'img/' + option + '.png'
	img.onclick = function() {
		choose(choice)
	}
	humanChoices.appendChild(img)
})

function choose(human) {
	const ai = option(prediction)
	const winner = getWinner(human, ai)
	showChoice('human', human, winner !== HUMAN)
	showChoice('ai', ai, winner !== AI)
	endGame(human, ai)
}

function showChoice(id, choice, lost) {
	const elem = document.getElementById(id)
	elem.src = 'img/' + OPTIONS[choice] + '.png'
	elem.classList.toggle('lost', lost)
}

// Chart

const chart = new frappe.Chart('#chart', {
	type: 'bar',
	height: 200,
	data: {
		labels: [],
		datasets: [
			{ name: 'AI', values: [] },
			{ name: 'Tie', values: [] },
			{ name: 'Human', values: [] },
		],
		yMarkers: [{ label: '', value: 0, options: { labelPos: 'right' } }],
	},
	barOptions: {
		spaceRatio: 0.2,
		stacked: 1,
	},
	colors: ['#0F0', '#CCC', '#F00'],
})

function updateChart() {
	// TEMP
	return;

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

	chart.data.yMarkers[0].value = total / 2
	chart.addDataPoint(total || '', [ai, tie, human])
}

function init() {
	const queue = []
	for (let i = 0; i < MAX_DATAPOINTS; i++) {
		updateChart()
	}
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
			addMatch(human, ai)
			updateChart()
		}
	}
	train(startGame)
}

setState('initializing')
reset()
window.onload = init

})()
