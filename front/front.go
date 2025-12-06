package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// Структуры для ответа от ML сервера
type PredictedVolatility struct {
	Median      float64 `json:"median"`
	Lower1Sigma float64 `json:"lower_1sigma"`
	Upper1Sigma float64 `json:"upper_1sigma"`
	Lower2Sigma float64 `json:"lower_2sigma"`
	Upper2Sigma float64 `json:"upper_2sigma"`
}

type Trend struct {
	Direction  string  `json:"direction"`
	Confidence string  `json:"confidence"`
	Strength   float64 `json:"strength"`
}

type Channel struct {
	Upper2Sigma  float64 `json:"upper_2sigma"`
	Upper1Sigma  float64 `json:"upper_1sigma"`
	CurrentPrice float64 `json:"current_price"`
	Lower1Sigma  float64 `json:"lower_1sigma"`
	Lower2Sigma  float64 `json:"lower_2sigma"`
}

type TradingSignal struct {
	Action       string  `json:"action"`
	Entry        float64 `json:"entry"`
	Target       float64 `json:"target"`
	StopLoss     float64 `json:"stop_loss"`
	PositionSize float64 `json:"position_size"`
	Reason       string  `json:"reason"`
}

type TailRisk struct {
	Warning      bool     `json:"warning"`
	Probability  float64  `json:"probability"`
	ExpectedLoss *float64 `json:"expected_loss"`
}

type Feature struct {
	Name         string  `json:"name"`
	Value        float64 `json:"value"`
	Contribution float64 `json:"contribution"`
}

type Explanation struct {
	Text        string    `json:"text"`
	TopFeatures []Feature `json:"top_features"`
}

type VolumeContext struct {
	Zscore        float64 `json:"zscore"`
	SpikeDetected bool    `json:"spike_detected"`
	PocDistance   float64 `json:"poc_distance"`
	VaPosition    string  `json:"va_position"`
}

type MLResponse struct {
	Ticker              string              `json:"ticker"`
	Horizon             int                 `json:"horizon"`
	PredictedVolatility PredictedVolatility `json:"predicted_volatility"`
	Confidence          float64             `json:"confidence"`
	Trend               Trend               `json:"trend"`
	Channel             Channel             `json:"channel"`
	TradingSignal       TradingSignal       `json:"trading_signal"`
	TailRisk            TailRisk            `json:"tail_risk"`
	VolumeContext       VolumeContext       `json:"volume_context"`
	Explanation         Explanation         `json:"explanation"`
}

// Структура для запроса от фронта
type UserRequest struct {
	Message string `json:"message"`
}

func main() {
	router := gin.Default()
	router.LoadHTMLGlob("templates/*")

	// Middleware для CORS
	router.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	// Главная страница
	router.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", gin.H{})
	})

	// Эндпоинт для обработки сообщений от пользователя
	router.POST("/api/chat", func(c *gin.Context) {
		var req UserRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		fmt.Printf("Получен запрос: %s\n", req.Message)

		// Здесь можно добавить логику для парсинга сообщения пользователя
		// и формирования запроса к ML серверу

		// Проксируем запрос к ML серверу (127.0.0.1:8080)
		mlResponse, err := forwardToMLServer(req.Message)
		if err != nil {
			// Если ML сервер не доступен, возвращаем тестовые данные
			fmt.Printf("Ошибка подключения к ML серверу: %v\n", err)
			mlResponse = getMockResponse()
		}

		c.JSON(http.StatusOK, mlResponse)
	})

	// Статика для Chart.js
	router.Static("/static", "./static")

	fmt.Println("Сервер запущен на http://localhost:8081")
	router.Run(":8081")
}

// Функция для отправки запроса к ML серверу
func forwardToMLServer(message string) (*MLResponse, error) {
	// Здесь должна быть логика парсинга сообщения и формирования запроса
	// Пока что отправляем простой запрос
	requestBody := map[string]interface{}{
		"message":   message,
		"timestamp": time.Now().Unix(),
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post("http://127.0.0.1:8080/predict",
		"application/json",
		bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var mlResponse MLResponse
	if err := json.Unmarshal(body, &mlResponse); err != nil {
		return nil, err
	}

	return &mlResponse, nil
}

// Функция возвращает тестовые данные, если ML сервер не доступен
func getMockResponse() *MLResponse {
	return &MLResponse{
		Ticker:  "SBER",
		Horizon: 3,
		PredictedVolatility: PredictedVolatility{
			Median:      0.024,
			Lower1Sigma: 0.018,
			Upper1Sigma: 0.031,
			Lower2Sigma: 0.015,
			Upper2Sigma: 0.038,
		},
		Confidence: 0.72,
		Trend: Trend{
			Direction:  "uptrend",
			Confidence: "high",
			Strength:   0.85,
		},
		Channel: Channel{
			Upper2Sigma:  127.5,
			Upper1Sigma:  125.8,
			CurrentPrice: 123.4,
			Lower1Sigma:  121.0,
			Lower2Sigma:  119.3,
		},
		TradingSignal: TradingSignal{
			Action:       "BUY",
			Entry:        121.0,
			Target:       125.8,
			StopLoss:     119.3,
			PositionSize: 0.1,
			Reason:       "Price at lower 1-sigma in uptrend",
		},
		TailRisk: TailRisk{
			Warning:      false,
			Probability:  0.03,
			ExpectedLoss: nil,
		},
		VolumeContext: VolumeContext{
			Zscore:        0.8,
			SpikeDetected: false,
			PocDistance:   -0.02,
			VaPosition:    "inside",
		},
		Explanation: Explanation{
			Text: "Волатильность повышена из-за роста исторической волатильности",
			TopFeatures: []Feature{
				{Name: "realized_vol_20", Value: 0.022, Contribution: 0.008},
				{Name: "beta_to_index", Value: 1.2, Contribution: 0.004},
				{Name: "volume_zscore", Value: 0.8, Contribution: 0.003},
			},
		},
	}
}
