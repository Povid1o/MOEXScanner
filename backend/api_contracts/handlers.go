package apicontracts

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	src "github.com/Povid1o/MOEXScanner.git/src"
	db "github.com/Povid1o/MOEXScanner.git/src/db"
	"github.com/gin-gonic/gin"
)

// checkError logs handler errors
func checkError(err error) {
	log.Print("[handlers]: ", err)
}

type Handlers struct {
	Health     *HealthHandler
	Features   *FeaturesHandler
	Prediction *PredictionHandler
	Backtest   *BacktestHandler
	Data       *DataHandler
}

type HealthHandler struct{}

// CheckHealth returns service and DB status
func (h *HealthHandler) CheckHealth(c *gin.Context) {
	dbStatus := "connected"
	err := db.Db_connect().Ping()
	if err != nil {
		dbStatus = "disconnected"
	}
	c.JSON(200, gin.H{
		"status":        "healthy",
		"models_loaded": []string{"garch", "lgbm_q16", "lgbm_q50", "lgbm_q84"},
		"cache_status":  "connected",
		"db_status":     dbStatus,
	})
}

type FeaturesHandler struct{}

// GetFeatures returns features for the given ticker
func (h *FeaturesHandler) GetFeatures(c *gin.Context) {
	ticker := c.Param("ticker")

	c.JSON(200, gin.H{
		"ticker":   ticker,
		"features": "реальные данные...",
	})
}

type PredictionHandler struct{}

// Predict handles prediction requests: validate, fetch data, call model
func (h *PredictionHandler) Predict(c *gin.Context) {
	log.Println("[call Predict]")
	var req PredictionRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request format",
			"details": err.Error(),
		})
		return
	}

	endDate := time.Now()
	startDate := endDate.AddDate(0, 0, -365)
	candles, err := src.GetCandles(
		req.Ticker,
		startDate.Format("2006-01-02"),
		endDate.Format("2006-01-02"),
		24,
	)
	if err != nil {
		log.Printf("Error getting candles: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to get data from MOEX",
			"details": err.Error(),
		})
		return
	}
	if len(candles) == 0 {
		c.JSON(http.StatusNotFound, gin.H{
			"error": "No data found for the specified period",
		})
		return
	}

	payload := map[string]interface{}{
		"ticker":    req.Ticker,
		"candles":   candles,
		"timeframe": req.Timeframe,
		"horizon":   req.Horizon,
		"date":      endDate.Format("2006-01-02"),
	}

	log.Println("[local model CLI call]")
	aiResponse, err := src.Ai_send_request_local(payload)
	if err != nil {
		log.Printf("Local model error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to get local model response",
			"details": err.Error(),
			"output":  aiResponse,
		})
		return
	}

	log.Printf("[parsing AI response] Raw response length: %d chars", len(aiResponse))
	cleanedResponse := cleanAIResponse(aiResponse)
	log.Printf("[parsing AI response] Cleaned response length: %d chars", len(cleanedResponse))

	var resp PredictionResponse
	if err := json.Unmarshal([]byte(cleanedResponse), &resp); err != nil {
		log.Printf("Failed to parse AI response: %v", err)
		fixedResponse := fixCommonJSONIssues(cleanedResponse)

		if err2 := json.Unmarshal([]byte(fixedResponse), &resp); err2 != nil {
			log.Printf("Failed to parse fixed response: %v", err2)
			c.JSON(http.StatusOK, getFallbackResponse(req.Ticker, req.Horizon, aiResponse, err2))
			return
		}
	}

	resp.Ticker = req.Ticker
	resp.Horizon = req.Horizon

	log.Printf("[success] Returning response for ticker: %s, horizon: %d", resp.Ticker, resp.Horizon)
	c.JSON(http.StatusOK, resp)
}

type BacktestHandler struct{}

// RunBacktest triggers the backtesting pipeline
func (h *BacktestHandler) RunBacktest(c *gin.Context) {

}

type DataHandler struct{}

// UpdateData initiates data update from MOEX
func (h *DataHandler) UpdateData(c *gin.Context) {

}
