package apicontracts

import (
	"github.com/gin-gonic/gin"
)

func SetupRoutes(router *gin.Engine, handlers *Handlers) {
	// GET endpoints
	router.GET("/health", handlers.Health.CheckHealth)
	router.GET("/features/:ticker", handlers.Features.GetFeatures)

	// POST endpoints
	router.POST("/predict", handlers.Prediction.Predict)
	router.POST("/backtest", handlers.Backtest.RunBacktest)
	router.POST("/update_data", handlers.Data.UpdateData)
}
