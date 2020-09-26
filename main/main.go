package main

import (
	"fmt"
	"github.com/dmitryikh/leaves"
)

func main() {
	// 1. Read model
	model, err := leaves.XGEnsembleFromFile("./model_V1", true)
	if err != nil {
		panic(err)
	}

	// 2. Do predictions!

	input := []float64{
    1,2,3,4,5,
    6,7,8,9,10,
    1,2,3,4,5,
    6,7,8,9,10,
    1,2,3,4,5,
    6,7,8,9,10,
    1,2,3,4,5,
    6,7,8,9,10,
    1,2,3,4,5,
    6,7,8,9,10,
	}
	row_num := 10
	col_num := 5

	preds := make([]float64, row_num)

	err_m := model.PredictDense(input, row_num, col_num, preds, 0, 1)

	fmt.Println(preds)
	fmt.Println(err_m)

}
