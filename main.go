package main

import (
	"math"

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
