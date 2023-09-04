package main

import (
	"errors"
	"math"
	"math/rand"
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
	rand_sourse := rand.NewSource(time.Now().UnixNano())
	rand_gen := rand.New(rand_sourse)

	wHidden := mat.NewDense(nn.config.hidden_neurons, nn.config.input_neurons, nil)
	bHidden := mat.NewDense(1, nn.config.hidden_neurons, nil)
	wOut := mat.NewDense(nn.config.output_neurons, nn.config.hidden_neurons, nil)
	bOut := mat.NewDense(1, nn.config.output_neurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = rand_gen.Float64()
		}
	}

	output := new(mat.Dense)

	if err := nn.backpropagate(X, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}
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
	if nn.wHidden == nil || nn.wOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	output := new(mat.Dense)

	hidden_layer_input := new(mat.Dense)
	hidden_layer_input.Mul(X, nn.wHidden)
	add_b_hidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hidden_layer_input.Apply(add_b_hidden, hidden_layer_input)

	hidden_layer_activation := new(mat.Dense)
	apply_sigmoid := func(_, col int, v float64) float64 { return sigmoid(v) }
	hidden_layer_activation.Apply(apply_sigmoid, hidden_layer_input)

	output_layer_input := new(mat.Dense)
	output_layer_input.Mul(hidden_layer_activation, nn.wOut)
	add_b_out := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	output_layer_input.Apply(add_b_out, output_layer_input)

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
		hidden_layer_input := new(mat.Dense)
		hidden_layer_input.Mul(x, wHidden)
		add_b_hidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hidden_layer_input.Apply(add_b_hidden, hidden_layer_input)

		hidden_layer_activation := new(mat.Dense)
		apply_sigmoid := func(_, col int, v float64) float64 { return sigmoid(v) }
		hidden_layer_activation.Apply(apply_sigmoid, hidden_layer_input)

		output_layer_input := new(mat.Dense)
		output_layer_input.Mul(hidden_layer_activation, wOut)
		add_b_out := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		output.Apply(add_b_out, output_layer_input)

		net_err := new(mat.Dense)
		net_err.Sub(y, output_layer_input)

		slope_output_layer := new(mat.Dense)
		apply_sigmoid_prime := func(_, col int, v float64) float64 { return sigmoid_prime(v) }
		slope_output_layer.Apply(apply_sigmoid_prime, output)
		slope_hidden_layer := new(mat.Dense)
		slope_hidden_layer.Mul(slope_output_layer, hidden_layer_activation)

		d_out := new(mat.Dense)
		d_out.MulElem(slope_output_layer, net_err)
		err_at_hidden_layer := new(mat.Dense)
		err_at_hidden_layer.MulElem(slope_hidden_layer, err_at_hidden_layer)

		d_hidden_layer := new(mat.Dense)
		d_hidden_layer.MulElem(err_at_hidden_layer, wOut.T())

		w_out_adj := new(mat.Dense)
		w_out_adj.Mul(d_out, hidden_layer_activation)
		w_out_adj.Scale(nn.config.learning_rate, w_out_adj)
		wOut.Add(wOut, w_out_adj)

		b_out_adj, err := sum_along_axis(0, d_out)
		if err != nil {
			return err
		}
		b_out_adj.Scale(nn.config.learning_rate, b_out_adj)
		bOut.Add(bOut, b_out_adj)

		w_hidden_adj := new(mat.Dense)
		w_hidden_adj.Mul(x.T(), d_hidden_layer)
		w_hidden_adj.Scale(nn.config.learning_rate, w_hidden_adj)
		wHidden.Add(wHidden, w_hidden_adj)

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
	num_rows, num_cols := mat.Dims()

	var out *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, num_cols)
		for i := 0; i < num_cols; i++ {
			col := mat.Row(nil, 1, data)
			data[i] = floats.Sum(col)

		}
		out = mat.NewDense(num_rows, 1, data)

	case 1:
		data := make([]float64, num_rows)
		for i := 0; i < num_rows; i++ {
			row := mat.Row(nil, 1, data)
			data[i] = floats.Sum(row)
		}
		out = mat.NewDense(num_rows, 1, data)

	default:
		return nil, errors.New("axis must be 0 or 1")
	}

	return out, nil
}
