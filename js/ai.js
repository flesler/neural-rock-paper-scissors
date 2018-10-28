// neural network settings

const TRAIN_OPTIONS = {
	log: 0,
	iterations: 6000,
	error: 0.005,
	clear: true,
	rate: 0.1,
}

const MEMORY_BLOCKS = 8

// Values

const HUMAN = 0
const AI = 1
const TIE = 0.5

const OUTPUTS = 3

// AI

const ai = localStorage.brain ?
		neataptic.Network.fromJSON(JSON.parse(localStorage.brain)) :
		new neataptic.architect.LSTM(OUTPUTS + 1, MEMORY_BLOCKS, MEMORY_BLOCKS, OUTPUTS);

let matches = []

ai.record = function (choice, result) {
	const match = createMatch(choice, result)
	const prev = matches[matches.length - 1]
	if (prev) {
		ai.train([{ input: prev.input, output: match.output }], TRAIN_OPTIONS)
		localStorage.brain = JSON.stringify(ai.toJSON())
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

function createMatch(choice, result) {
	const match = { choice: choice, result: result, input: [], output: [] }
	for (let i = 0; i < OUTPUTS; i++) {
		match.input[i] = match.output[i] = i === choice ? 1 : 0
	}
	match.input.push(result)
	return match
}
