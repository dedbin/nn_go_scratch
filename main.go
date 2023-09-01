package main

import (
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
