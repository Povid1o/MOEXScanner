package apicontracts

import (
	"github.com/gin-gonic/gin"
)

func SetupRoutes(router *gin.Engine, handlers *Handlers) {
	router.GET("/", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"service": "MOEX Scanner Backend API Gateway",
			"version": "1.0.0",
			"status":  "running",
			"endpoints": gin.H{
				"/health":           "GET - Check server health",
				"/predict":          "POST - Generate predictions (proxies to ML Engine)",
				"/features/:ticker": "GET - Get features for ticker (TODO)",
				"/backtest":         "POST - Run backtest (TODO)",
				"/update_data":      "POST - Update MOEX data (TODO)",
			},
			"connections": gin.H{
				"ml_engine": "http://127.0.0.1:8000",
				"frontend":  "http://localhost:8081",
				"moex_api":  "https://iss.moex.com",
			},
		})
	})
	router.GET("/health", handlers.Health.CheckHealth)
	router.GET("/features/:ticker", handlers.Features.GetFeatures)
	router.POST("/predict", handlers.Prediction.Predict)
	router.POST("/backtest", handlers.Backtest.RunBacktest)
	router.POST("/update_data", handlers.Data.UpdateData)
}
