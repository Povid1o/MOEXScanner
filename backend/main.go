package main

import (
	"log"

	apicontracts "github.com/Povid1o/MOEXScanner.git/api_contracts"
	"github.com/gin-gonic/gin"
)

func checkError(err error) {
	if err != nil {
		log.Print("[[main]]]: ", err)

	}
}

func main() {

	router := gin.Default()

	handlers := &apicontracts.Handlers{

		Health:     &apicontracts.HealthHandler{},
		Features:   &apicontracts.FeaturesHandler{},
		Prediction: &apicontracts.PredictionHandler{},
		Backtest:   &apicontracts.BacktestHandler{},
		Data:       &apicontracts.DataHandler{},
	}

	apicontracts.SetupRoutes(router, handlers)

	router.Run(":8080")
}
