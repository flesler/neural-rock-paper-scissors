const OPTIONS = ['rock', 'paper', 'scissors']

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
const chartData = {
	labels: ['', ''],
	datasets: [
		{ name: 'Human', values: [0, 0] },
		{ name: 'AI', values: [0, 0] },
	]
}

const chart = new Chart('#chart', {
	title: 'Web-based Rock-Paper-Scissors vs a Neural Network',
	type: 'line',
	height: 100,
	data: chartData
})

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

	chartData.labels.push(matches.length.toString())
	chartData.datasets[0].values.push(human)
	chartData.datasets[1].values.push(ai)

	if (chartData.labels.length > 15) {
		chartData.labels.shift()
		chartData.datasets.forEach(d => d.values.shift())
	}

	chart.update(chartData)
}

startGame()
