package main

import (
	"encoding/csv"
	"errors"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type nn struct {
	config  nnConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

type nnConfig struct {
	input_neurons  int
	output_neurons int
	hidden_neurons int
	num_epochs     int
	learning_rate  float64
}

// new_nn creates a new nn object with the given configuration.
func new_nn(config nnConfig) *nn {
	return &nn{config: config}
}

// sigmoid calculates the sigmoid of a given float64 value.
// will be used as the activation function for the hidden layer.
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// sigmoid_prime calculates the derivative of the sigmoid function.
// will be used as the activation function for the hidden layer.
func sigmoid_prime(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

// train trains the neural network with the given input data and labels.
// X is the input data matrix.
// y is the labels matrix.
// Returns an error if there was a problem during training.
func (nn *nn) train(X *mat.Dense, y *mat.Dense) error {
	// Create a new random number generator with a seed based on the current time
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	// Initialize the weight matrices and bias vectors
	wHidden := mat.NewDense(nn.config.hidden_neurons, nn.config.input_neurons, nil)
	bHidden := mat.NewDense(1, nn.config.hidden_neurons, nil)
	wOut := mat.NewDense(nn.config.output_neurons, nn.config.hidden_neurons, nil)
	bOut := mat.NewDense(1, nn.config.output_neurons, nil)

	// Get the underlying data slices of the matrices and vectors
	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	// Initialize the weight matrices and bias vectors with random values
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

	// Create a new dense matrix to store the output
	output := new(mat.Dense)

	// Perform backpropagation to update the weight matrices and bias vectors
	if err := nn.backpropagate(X, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Update the neural network's weight matrices and bias vectors
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

// predict performs a prediction using the given input matrix.
// X: The input matrix to be used for prediction.
// Returns the predicted matrix and an error if any.
func (nn *nn) predict(X *mat.Dense) (*mat.Dense, error) {
	// Check if the weights are empty
	if nn.wHidden == nil || nn.wOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}

	// Check if the biases are empty
	if nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Initialize the output matrix
	output := new(mat.Dense)

	// Calculate the input to the hidden layer
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(X, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	// Apply the sigmoid function to the hidden layer activation
	hiddenLayerActivation := new(mat.Dense)
	applySigmoid := func(_, col int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivation.Apply(applySigmoid, hiddenLayerInput)

	// Calculate the input to the output layer
	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivation, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)

	return output, nil
}

// backpropagate performs backpropagation on the neural network.
//
// x: The input data.
// y: The target output data.
// wHidden: The weight matrix of the hidden layer.
// bHidden: The bias vector of the hidden layer.
// wOut: The weight matrix of the output layer.
// bOut: The bias vector of the output layer.
// output: The output of the neural network.
// error: An error if any occurred during backpropagation.
func (nn *nn) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {
	for i := 0; i < nn.config.num_epochs; i++ {
		// Calculate the input to the hidden layer
		hidden_layer_input := new(mat.Dense)
		hidden_layer_input.Mul(x, wHidden)
		add_b_hidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hidden_layer_input.Apply(add_b_hidden, hidden_layer_input)

		// Apply the sigmoid activation function to the hidden layer
		hidden_layer_activation := new(mat.Dense)
		apply_sigmoid := func(_, col int, v float64) float64 { return sigmoid(v) }
		hidden_layer_activation.Apply(apply_sigmoid, hidden_layer_input)

		// Calculate the input to the output layer
		output_layer_input := new(mat.Dense)
		output_layer_input.Mul(hidden_layer_activation, wOut)
		add_b_out := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		output.Apply(add_b_out, output_layer_input)

		// Calculate the error between the target output and the actual output
		net_err := new(mat.Dense)
		net_err.Sub(y, output_layer_input)

		// Calculate the slope of the output layer
		slope_output_layer := new(mat.Dense)
		apply_sigmoid_prime := func(_, col int, v float64) float64 { return sigmoid_prime(v) }
		slope_output_layer.Apply(apply_sigmoid_prime, output)

		// Calculate the slope of the hidden layer
		slope_hidden_layer := new(mat.Dense)
		slope_hidden_layer.Mul(slope_output_layer, hidden_layer_activation)

		// Calculate the change in the output layer
		d_out := new(mat.Dense)
		d_out.MulElem(slope_output_layer, net_err)

		// Calculate the error at the hidden layer
		err_at_hidden_layer := new(mat.Dense)
		err_at_hidden_layer.MulElem(slope_hidden_layer, err_at_hidden_layer)

		// Calculate the change in the hidden layer
		d_hidden_layer := new(mat.Dense)
		d_hidden_layer.MulElem(err_at_hidden_layer, wOut.T())

		// Update the weights of the output layer
		w_out_adj := new(mat.Dense)
		w_out_adj.Mul(d_out, hidden_layer_activation)
		w_out_adj.Scale(nn.config.learning_rate, w_out_adj)
		wOut.Add(wOut, w_out_adj)

		// Update the biases of the output layer
		b_out_adj, err := sum_along_axis(0, d_out)
		if err != nil {
			return err
		}
		b_out_adj.Scale(nn.config.learning_rate, b_out_adj)
		bOut.Add(bOut, b_out_adj)

		// Update the weights of the hidden layer
		w_hidden_adj := new(mat.Dense)
		w_hidden_adj.Mul(x.T(), d_hidden_layer)
		w_hidden_adj.Scale(nn.config.learning_rate, w_hidden_adj)
		wHidden.Add(wHidden, w_hidden_adj)

		// Update the biases of the hidden layer
		b_hidden_adj, err := sum_along_axis(0, d_hidden_layer)
		if err != nil {
			return err
		}
		b_hidden_adj.Scale(nn.config.learning_rate, b_hidden_adj)
		bHidden.Add(bHidden, b_hidden_adj)

	}
	return nil
}

// sum_along_axis calculates the sum of elements along a specified axis of a matrix.
// Parameters:
// - axis: an integer indicating the axis along which the sum is calculated.
// - mat: a pointer to a mat.Dense matrix.
// Returns:
// - out: a pointer to a mat.Dense matrix containing the sum along the specified axis.
// - error: an error object indicating any error that occurred during the calculation.
func sum_along_axis(axis int, mat *mat.Dense) (*mat.Dense, error) {
	// Get the number of rows and columns in the matrix
	num_rows, num_cols := mat.Dims()

	var out *mat.Dense

	switch axis {
	case 0:
		// Calculate the sum along the rows (axis=0)
		data := make([]float64, num_cols)
		for i := 0; i < num_cols; i++ {
			// Get the i-th column of the matrix
			col := mat.Row(nil, i, data)
			// Calculate the sum of the column
			data[i] = floats.Sum(col)
		}
		// Create a new matrix with the sums along the rows
		out = mat.NewDense(num_rows, 1, data)

	case 1:
		// Calculate the sum along the columns (axis=1)
		data := make([]float64, num_rows)
		for i := 0; i < num_rows; i++ {
			// Get the i-th row of the matrix
			row := mat.Row(nil, i, data)
			// Calculate the sum of the row
			data[i] = floats.Sum(row)
		}
		// Create a new matrix with the sums along the columns
		out = mat.NewDense(num_rows, 1, data)

	default:
		// Invalid axis value
		return nil, errors.New("axis must be 0 or 1")
	}

	return out, nil
}

// makeInputAndLabels reads a CSV file and returns the inputs and labels as matrices.
// The CSV file should have 7 fields per record, with the first record being the header.
// The inputs are the values in the first 4 fields, while the labels are the values in the last 3 fields.
func makeInputAndLabels(filename string) (*mat.Dense, *mat.Dense) {
	f, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Create a CSV reader with 7 fields per record
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Read all records from the CSV file
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// Create slices to store the inputs and labels
	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	var inputsIndex int
	var labelsIndex int

	// Iterate over each record in the CSV data
	for idx, record := range rawCSVData {
		// Skip the header record
		if idx == 0 {
			continue
		}

		// Parse each value in the record as a float
		for i, val := range record {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Store the parsed value in the inputs or labels slice based on the field index
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	// Create matrices from the inputs and labels slices
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)

	return inputs, labels
}
