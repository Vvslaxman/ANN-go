package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

// neuralNet contains all of the information
// that defines a trained neural network.
type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// neuralNetConfig defines our neural network
// architecture and learning parameters.
type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

func main() {
	// Form the training matrices.
	inputs, labels := makeInputsAndLabels("data/train.csv")

	// Define our network architecture and learning parameters.
	config := neuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs:     10000,
		learningRate:  0.01,
	}

	// Train the neural network.
	network := newNetwork(config)
	trainAcc, trainLoss, err := network.train(inputs, labels)
	if err != nil {
		log.Fatal(err)
	}

	// Form the testing matrices.
	testInputs, testLabels := makeInputsAndLabels("data/test.csv")

	// Make the predictions using the trained model.
	predictions, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate metrics: Accuracy, Precision, Recall, F1-score.
	accuracy := calculateAccuracy(predictions, testLabels)
	precision := calculatePrecision(predictions, testLabels)
	recall := calculateRecall(predictions, testLabels)
	f1Score := calculateF1Score(precision, recall)

	// Output the metrics.
	fmt.Printf("\nMetrics:\n")
	fmt.Printf("Accuracy = %0.2f\n", accuracy)
	fmt.Printf("Precision = %0.2f\n", precision)
	fmt.Printf("Recall = %0.2f\n", recall)
	fmt.Printf("F1-score = %0.2f\n\n", f1Score)

	// Plot training metrics (accuracy and loss).
	plotMetrics(trainAcc, trainLoss)
}

// NewNetwork initializes a new neural network.
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}
// train trains a neural network using backpropagation with L2 regularization.
func (nn *neuralNet) train(x, y *mat.Dense) ([]float64, []float64, error) {
    // Initialize biases/weights.
    randSource := rand.NewSource(time.Now().UnixNano())
    randGen := rand.New(randSource)

    nn.wHidden = mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
    nn.bHidden = mat.NewDense(1, nn.config.hiddenNeurons, nil)
    nn.wOut = mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
    nn.bOut = mat.NewDense(1, nn.config.outputNeurons, nil)

    wHiddenRaw := nn.wHidden.RawMatrix().Data
    bHiddenRaw := nn.bHidden.RawMatrix().Data
    wOutRaw := nn.wOut.RawMatrix().Data
    bOutRaw := nn.bOut.RawMatrix().Data

    for _, param := range [][]float64{
        wHiddenRaw,
        bHiddenRaw,
        wOutRaw,
        bOutRaw,
    } {
        for i := range param {
            param[i] = randGen.Float64()
        }
    }

    // Define the output of the neural network.
    output := new(mat.Dense)

    // Track training metrics.
    var trainAcc []float64
    var trainLoss []float64

    // Regularization parameter lambda (adjust as needed).
    lambda := 0.01

    // Use backpropagation to adjust the weights and biases.
    for epoch := 0; epoch < nn.config.numEpochs; epoch++ {
        // Complete the feed forward process.
        hiddenLayerInput := new(mat.Dense)
        hiddenLayerInput.Mul(x, nn.wHidden)
        addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
        hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

        hiddenLayerActivations := new(mat.Dense)
        applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
        hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

        outputLayerInput := new(mat.Dense)
        outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
        addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
        outputLayerInput.Apply(addBOut, outputLayerInput)
        output.Apply(applySigmoid, outputLayerInput)

        // Calculate loss with L2 regularization.
        loss := nn.calculateLoss(output, y)

        // Calculate L2 regularization terms.
        numSamples, _ := x.Dims()

        // Calculate ||wHidden||^2 and ||wOut||^2
        var wHiddenNorm, wOutNorm float64
        wHiddenSquared := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
        wOutSquared := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
        
        wHiddenSquared.Apply(func(i, j int, v float64) float64 {
            return math.Pow(nn.wHidden.At(i, j), 2)
        }, wHiddenSquared)

        wOutSquared.Apply(func(i, j int, v float64) float64 {
            return math.Pow(nn.wOut.At(i, j), 2)
        }, wOutSquared)

        wHiddenNorm = mat.Sum(wHiddenSquared)
        wOutNorm = mat.Sum(wOutSquared)

        // Regularized loss
        regLoss := loss + (lambda / (2.0 * float64(numSamples))) * (wHiddenNorm + wOutNorm)

        // Calculate accuracy.
        accuracy := calculateAccuracy(output, y)

        // Track metrics.
        trainAcc = append(trainAcc, accuracy)
        trainLoss = append(trainLoss, regLoss)

        // Backpropagation.
        networkError := new(mat.Dense)
        networkError.Sub(y, output)

        slopeOutputLayer := new(mat.Dense)
        applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
        slopeOutputLayer.Apply(applySigmoidPrime, output)
        slopeHiddenLayer := new(mat.Dense)
        slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

        dOutput := new(mat.Dense)
        dOutput.MulElem(networkError, slopeOutputLayer)
        errorAtHiddenLayer := new(mat.Dense)
        errorAtHiddenLayer.Mul(dOutput, nn.wOut.T())

        dHiddenLayer := new(mat.Dense)
        dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

        // Adjust the parameters with L2 regularization.
        wOutAdj := new(mat.Dense)
        wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
        wOutAdj.Scale(nn.config.learningRate, wOutAdj)

        // Regularized adjustment
        wOutAdjReg := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
        wOutAdjReg.Scale(1.0 - nn.config.learningRate*lambda/float64(x.RawMatrix().Cols), nn.wOut)
        nn.wOut.Add(wOutAdjReg, wOutAdj)

        bOutAdj, err := sumAlongAxis(0, dOutput)
        if err != nil {
            return nil, nil, err
        }
        bOutAdj.Scale(nn.config.learningRate, bOutAdj)
        nn.bOut.Add(nn.bOut, bOutAdj)

        wHiddenAdj := new(mat.Dense)
        wHiddenAdj.Mul(x.T(), dHiddenLayer)
        wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)

        // Regularized adjustment
        wHiddenAdjReg := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
        wHiddenAdjReg.Scale(1.0 - nn.config.learningRate*lambda/float64(x.RawMatrix().Cols), nn.wHidden)
        nn.wHidden.Add(wHiddenAdjReg, wHiddenAdj)

        bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
        if err != nil {
            return nil, nil, err
        }
        bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
        nn.bHidden.Add(nn.bHidden, bHiddenAdj)
    }

    // Return the trained network parameters and metrics.
    return trainAcc, trainLoss, nil
}

// calculateLoss calculates the loss (mean squared error).
func (nn *neuralNet) calculateLoss(predictions, labels *mat.Dense) float64 {
	numSamples, _ := predictions.Dims()
	var loss float64

	// Calculate the mean squared error (MSE).
	diff := new(mat.Dense)
	diff.Sub(predictions, labels)
	diff.Apply(func(_, _ int, v float64) float64 {
		return math.Pow(v, 2)
	}, diff)

	loss = mat.Sum(diff) / float64(numSamples)

	return loss
}

// predict makes a prediction based on a trained
// neural network.
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
	// Check to make sure that our neuralNet value
	// represents a trained model.
	if nn.wHidden == nil || nn.wOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Complete the feed forward process.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

// calculateAccuracy calculates the accuracy of predictions.
func calculateAccuracy(predictions, labels *mat.Dense) float64 {
	numPreds, _ := predictions.Dims()
	var correctPredictions float64

	for i := 0; i < numPreds; i++ {
		predicted := mat.Row(nil, i, predictions)
		actual := mat.Row(nil, i, labels)

		predictedIdx := floats.MaxIdx(predicted)
		actualIdx := floats.MaxIdx(actual)

		if predictedIdx == actualIdx {
			correctPredictions++
		}
	}

	accuracy := correctPredictions / float64(numPreds)
	return accuracy
}

// calculatePrecision calculates the precision of predictions.
func calculatePrecision(predictions, labels *mat.Dense) float64 {
	numPreds, _ := predictions.Dims()
	var truePositives float64
	var falsePositives float64

	for i := 0; i < numPreds; i++ {
		predicted := mat.Row(nil, i, predictions)
		actual := mat.Row(nil, i, labels)

		predictedIdx := floats.MaxIdx(predicted)
		actualIdx := floats.MaxIdx(actual)

		if predictedIdx == actualIdx {
			truePositives++
		} else {
			falsePositives++
		}
	}

	if truePositives == 0 {
		return 0.0 // Handle division by zero.
	}

	precision := truePositives / (truePositives + falsePositives)
	return precision
}

// calculateRecall calculates the recall of predictions.
func calculateRecall(predictions, labels *mat.Dense) float64 {
	numPreds, _ := predictions.Dims()
	var truePositives float64
	var falseNegatives float64

	for i := 0; i < numPreds; i++ {
		predicted := mat.Row(nil, i, predictions)
		actual := mat.Row(nil, i, labels)

		predictedIdx := floats.MaxIdx(predicted)
		actualIdx := floats.MaxIdx(actual)

		if predictedIdx == actualIdx {
			truePositives++
		} else {
			falseNegatives++
		}
	}

	if truePositives == 0 {
		return 0.0 // Handle division by zero.
	}

	recall := truePositives / (truePositives + falseNegatives)
	return recall
}

// calculateF1Score calculates the F1-score from precision and recall.
func calculateF1Score(precision, recall float64) float64 {
	if precision == 0 || recall == 0 {
		return 0.0 // Handle division by zero.
	}

	f1Score := 2 * (precision * recall) / (precision + recall)
	return f1Score
}

// sigmoid implements the sigmoid function
// for use in activation functions.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

// sumAlongAxis sums elements along a specified axis of a matrix.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

// makeInputsAndLabels reads CSV file and returns inputs and labels matrices.
func makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Open the dataset file.
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData will hold all the
	// float values that will eventually be
	// used to form matrices.
	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	// Will track the current index of matrix values.
	var inputsIndex int
	var labelsIndex int

	// Sequentially move the rows into a slice of floats.
	for idx, record := range rawCSVData {

		// Skip the header row.
		if idx == 0 {
			continue
		}

		// Loop over the float columns.
		for i, val := range record {

			// Convert the value to a float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)
	return inputs, labels
}
func plotMetrics(accuracy, loss []float64) {
	// Create a plot for accuracy.
	p1 := plot.New()
	p1.Title.Text = "Training Accuracy"
	p1.X.Label.Text = "Epoch"
	p1.Y.Label.Text = "Accuracy"

	pts1 := make(plotter.XYs, len(accuracy))
	for i := range pts1 {
		pts1[i].X = float64(i)
		pts1[i].Y = accuracy[i]
	}

	err := plotutil.AddLinePoints(p1, "Accuracy", pts1)
	if err != nil {
		log.Fatal(err)
	}

	// Save the plot to a file.
	if err := p1.Save(8*vg.Inch, 6*vg.Inch, "accuracy_plot.png"); err != nil {
		log.Fatal(err)
	}

	// Create a plot for loss.
	p2 := plot.New()
	p2.Title.Text = "Training Loss"
	p2.X.Label.Text = "Epoch"
	p2.Y.Label.Text = "Loss"

	pts2 := make(plotter.XYs, len(loss))
	for i := range pts2 {
		pts2[i].X = float64(i)
		pts2[i].Y = loss[i]
	}

	err = plotutil.AddLinePoints(p2, "Loss", pts2)
	if err != nil {
		log.Fatal(err)
	}

	// Save the plot to a file.
	if err := p2.Save(8*vg.Inch, 6*vg.Inch, "loss_plot.png"); err != nil {
		log.Fatal(err)
	}
}
