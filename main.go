package main

import (
	"math"
	"math/rand"
	"time"

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

}
