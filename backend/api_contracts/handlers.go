package apicontracts

import (
	"encoding/json"
	"fmt"
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

	//validate data user request
	endDate, _ := time.Parse("2006-01-02", req.Date)
	startDate := endDate.AddDate(0, 0, -60)
	candles, err := src.GetCandles(
		req.Ticker,
		startDate.Format("2006-01-02"),
		endDate.Format("2006-01-02"),
		24, //24 for day data
	)
	if err != nil {
		checkError(err)
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

	//preapre promt for ai
	candlesJSON, _ := json.Marshal(candles)

	prompt := fmt.Sprintf(`
		Ты профессиональный финансовый аналитик. Проанализируй исторические данные акции и верни прогноз в ТОЧНОМ JSON формате ниже.

		ИСТОРИЧЕСКИЕ ДАННЫЕ АКЦИИ %s (в формате JSON):
		%s

		ПАРАМЕТРЫ АНАЛИЗА:
		- Таймфрейм: %s
		- Период прогноза: %d дней
		- Текущая дата анализа: %s

		ТРЕБУЕМЫЙ ФОРМАТ ОТВЕТА (ТОЛЬКО JSON, БЕЗ ЛЮБЫХ ДОПОЛНИТЕЛЬНЫХ ТЕКСТОВЫХ ПОЯСНЕНИЙ):

		{
		"ticker": "%s",
		"horizon": %d,
		"predicted_volatility": {
			"median": число,
			"lower_1sigma": число,
			"upper_1sigma": число,
			"lower_2sigma": число,
			"upper_2sigma": число
		},
		"confidence": число_от_0_до_1,
		"trend": {
			"direction": "uptrend/downtrend/sideways",
			"confidence": "high/medium/low", 
			"strength": число_от_0_до_1
		},
		"channel": {
			"upper_2sigma": число,
			"upper_1sigma": число,
			"current_price": число,
			"lower_1sigma": число,
			"lower_2sigma": число
		},
		"trading_signal": {
			"action": "BUY/SELL/HOLD",
			"entry": число,
			"target": число,
			"stop_loss": число,
			"position_size": число_от_0_до_1,
			"reason": "краткое_обоснование"
		},
		"tail_risk": {
			"warning": true/false,
			"probability": число_от_0_до_1,
			"expected_loss": число_или_null
		},
		"volume_context": {
			"zscore": число,
			"spike_detected": true/false,
			"poc_distance": число,
			"va_position": "inside/above/below"
		},
		"explanation": {
			"text": "аналитический_вывод",
			"top_features": [
			{"name": "название_метрики", "value": число, "contribution": число},
			{"name": "название_метрики", "value": число, "contribution": число},
			{"name": "название_метрики", "value": число, "contribution": число}
			]
		}
		}

		ВАЖНО: 
		1. Верни ТОЛЬКО валидный JSON без каких-либо дополнительных текстовых комментариев
		2. Все числовые значения должны быть реалистичными для финансовых рынков
		3. Убедись, что JSON синтаксически корректен
		`, req.Ticker, string(candlesJSON), req.Timeframe, req.Horizon, req.Date, req.Ticker, req.Horizon)

	//ai call
	log.Println("[ai call]")
	aiResponse, err := src.Ai_send_request("Financial Analyst", prompt)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	//for test
	c.JSON(http.StatusOK, gin.H{
		"raw_ai_response": aiResponse,
		"status":          "success",
		"note":            "test",
	})

}

type BacktestHandler struct{}

func (h *BacktestHandler) RunBacktest(c *gin.Context) {
	// TODO
}

type DataHandler struct{}

func (h *DataHandler) UpdateData(c *gin.Context) {
	//TODO
}
