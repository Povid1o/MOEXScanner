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

func (h *HealthHandler) CheckHealth(c *gin.Context) {
	dbStatus := "connected"
	err := db.Db_connect().Ping()
	if err != nil {
		dbStatus = "disconnected"
	}
	c.JSON(200, gin.H{
		"status":        "healthy",                                             //TODO
		"models_loaded": []string{"garch", "lgbm_q16", "lgbm_q50", "lgbm_q84"}, //TODO
		"cache_status":  "connected",
		"db_status":     dbStatus,
	})
}

type FeaturesHandler struct{}

func (h *FeaturesHandler) GetFeatures(c *gin.Context) {
	ticker := c.Param("ticker")

	c.JSON(200, gin.H{
		"ticker":   ticker,
		"features": "реальные данные...", //TODO
	})
}

type PredictionHandler struct{}

func (h *PredictionHandler) Predict(c *gin.Context) {
	log.Println("[call Predict]")
	var req PredictionRequest

	// validation user request
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request format",
			"details": err.Error(),
		})
		return
	}

	// validate data user request
	// use current date as the 'till' date to ensure latest price is included
	endDate := time.Now()
	// Загружаем 365 дней истории для ML-модели
	// Необходимо для расчёта долгосрочных индикаторов (SMA-200, долгосрочная волатильность и т.д.)
	// ML-модель требует минимум 200-250 точек данных для корректной генерации признаков
	startDate := endDate.AddDate(0, 0, -365)
	candles, err := src.GetCandles(
		req.Ticker,
		startDate.Format("2006-01-02"),
		endDate.Format("2006-01-02"),
		24, //24 for day data
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

	// prepare structured payload for local prediction CLI
	// send analysis date as current date to the model so features align with fetched candles
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
		// include output if available
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to get local model response",
			"details": err.Error(),
			"output":  aiResponse,
		})
		return
	}

	// Парсинг JSON ответа от AI
	log.Printf("[parsing AI response] Raw response length: %d chars", len(aiResponse))

	// Очистка ответа
	cleanedResponse := cleanAIResponse(aiResponse)
	log.Printf("[parsing AI response] Cleaned response length: %d chars", len(cleanedResponse))

	var resp PredictionResponse
	if err := json.Unmarshal([]byte(cleanedResponse), &resp); err != nil {
		log.Printf("Failed to parse AI response: %v", err)

		// Попробуем исправить возможные проблемы с JSON
		fixedResponse := fixCommonJSONIssues(cleanedResponse)

		if err2 := json.Unmarshal([]byte(fixedResponse), &resp); err2 != nil {
			log.Printf("Failed to parse fixed response: %v", err2)

			// Возвращаем тестовые данные с ошибкой
			c.JSON(http.StatusOK, getFallbackResponse(req.Ticker, req.Horizon, aiResponse, err2))
			return
		}
	}

	// Убедимся, что тикер и горизонт установлены
	resp.Ticker = req.Ticker
	resp.Horizon = req.Horizon

	// Возвращаем структурированный ответ
	log.Printf("[success] Returning response for ticker: %s, horizon: %d", resp.Ticker, resp.Horizon)
	c.JSON(http.StatusOK, resp)
}

type BacktestHandler struct{}

func (h *BacktestHandler) RunBacktest(c *gin.Context) {
	// TODO
}

type DataHandler struct{}

func (h *DataHandler) UpdateData(c *gin.Context) {
	//TODO
}
