package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
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

// Структура для запроса к ML серверу (как ожидает PredictionHandler)
type MLRequest struct {
	Ticker    string `json:"ticker"`
	Timeframe string `json:"timeframe"`
	Horizon   int    `json:"horizon"`
	Date      string `json:"date"`
}

func main() {
	router := gin.Default()

	// Устанавливаем режим релиза, чтобы убрать предупреждение
	gin.SetMode(gin.ReleaseMode)

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

		// Парсим сообщение пользователя
		ticker, horizon := parseUserMessage(req.Message)

		// Формируем запрос к ML серверу в правильном формате
		mlReq := MLRequest{
			Ticker:    ticker,
			Timeframe: "D", // по умолчанию дневной таймфрейм
			Horizon:   horizon,
			Date:      time.Now().Format("2006-01-02"),
		}

		// Проксируем запрос к ML серверу (127.0.0.1:8080)
		mlResponse, err := forwardToMLServer(mlReq)
		if err != nil {
			// Если ML сервер не доступен, возвращаем тестовые данные
			fmt.Printf("Ошибка подключения к ML серверу: %v\n", err)
			mlResponse = getMockResponse(ticker, horizon)
		}

		c.JSON(http.StatusOK, mlResponse)
	})

	fmt.Println("Фронтенд сервер запущен на http://localhost:8081")
	router.Run(":8081")
}

// Парсим сообщение пользователя для извлечения тикера и горизонта
func parseUserMessage(message string) (string, int) {
	message = strings.ToUpper(message)

	// Список тикеров
	tickers := []string{"SBER", "GAZP", "LKOH", "ROSN", "VTBR", "ALRS", "GMKN", "NVTK", "TATN", "YNDX"}

	// Ищем тикер в сообщении
	ticker := "SBER" // по умолчанию
	for _, t := range tickers {
		if strings.Contains(message, t) {
			ticker = t
			break
		}
	}

	// Определяем горизонт
	horizon := 3 // по умолчанию 3 дня
	if strings.Contains(message, "НЕДЕЛ") || strings.Contains(message, "WEEK") {
		horizon = 7
	} else if strings.Contains(message, "МЕСЯЦ") || strings.Contains(message, "MONTH") {
		horizon = 30
	} else {
		// Пытаемся найти цифры в сообщении
		for i := 1; i <= 365; i++ {
			if strings.Contains(message, fmt.Sprintf("%d", i)) {
				horizon = i
				break
			}
		}
	}

	return ticker, horizon
}

// Функция для отправки запроса к ML серверу в правильном формате
func forwardToMLServer(mlReq MLRequest) (*MLResponse, error) {
	// Формируем правильный запрос для PredictionHandler
	jsonData, err := json.Marshal(mlReq)
	if err != nil {
		return nil, fmt.Errorf("ошибка маршалинга запроса: %v", err)
	}

	fmt.Printf("Отправка запроса к ML серверу: %s\n", string(jsonData))

	// Увеличиваем таймаут до 5 минут (300 секунд)
	client := &http.Client{
		Timeout: 300 * time.Second, // 5 минут
	}

	resp, err := client.Post("http://127.0.0.1:8080/predict",
		"application/json",
		bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("ошибка соединения с ML сервером: %v", err)
	}
	defer resp.Body.Close()

	// Проверяем статус ответа
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML сервер вернул ошибку %d: %s", resp.StatusCode, string(bodyBytes))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("ошибка чтения ответа: %v", err)
	}

	fmt.Printf("Получен ответ от ML сервера (%d байт)\n", len(body))

	var mlResponse MLResponse
	if err := json.Unmarshal(body, &mlResponse); err != nil {
		// Пробуем очистить JSON от лишних символов
		cleanedBody := cleanJSON(string(body))
		if err2 := json.Unmarshal([]byte(cleanedBody), &mlResponse); err2 != nil {
			return nil, fmt.Errorf("ошибка парсинга JSON ответа: %v (после очистки: %v)", err, err2)
		}
	}

	return &mlResponse, nil
}

// Функция для очистки JSON ответа
func cleanJSON(jsonStr string) string {
	// Удаляем markdown блоки кода
	jsonStr = strings.TrimSpace(jsonStr)

	// Удаляем ```json в начале
	if strings.HasPrefix(jsonStr, "```json") {
		jsonStr = strings.TrimPrefix(jsonStr, "```json")
	}

	// Удаляем ``` в начале и конце
	jsonStr = strings.TrimPrefix(jsonStr, "```")
	jsonStr = strings.TrimSuffix(jsonStr, "```")

	// Удаляем лишние пробелы
	jsonStr = strings.TrimSpace(jsonStr)

	return jsonStr
}

// Функция возвращает тестовые данные, если ML сервер не доступен
func getMockResponse(ticker string, horizon int) *MLResponse {
	// Базовые цены для разных тикеров
	priceMap := map[string]float64{
		"SBER": 280.5,
		"GAZP": 160.3,
		"LKOH": 7200.0,
		"ROSN": 550.8,
		"VTBR": 0.026,
		"ALRS": 90.2,
		"GMKN": 26000.0,
		"NVTK": 1200.5,
		"TATN": 380.7,
		"YNDX": 2900.0,
	}

	currentPrice, ok := priceMap[ticker]
	if !ok {
		currentPrice = 123.4
	}

	return &MLResponse{
		Ticker:  ticker,
		Horizon: horizon,
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
			Upper2Sigma:  currentPrice * 1.033,
			Upper1Sigma:  currentPrice * 1.019,
			CurrentPrice: currentPrice,
			Lower1Sigma:  currentPrice * 0.981,
			Lower2Sigma:  currentPrice * 0.967,
		},
		TradingSignal: TradingSignal{
			Action:       "BUY",
			Entry:        currentPrice * 0.981,
			Target:       currentPrice * 1.019,
			StopLoss:     currentPrice * 0.967,
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
			Text: fmt.Sprintf("Тестовые данные для %s на %d дней (ML сервер недоступен)", ticker, horizon),
			TopFeatures: []Feature{
				{Name: "realized_vol_20", Value: 0.022, Contribution: 0.008},
				{Name: "beta_to_index", Value: 1.2, Contribution: 0.004},
				{Name: "volume_zscore", Value: 0.8, Contribution: 0.003},
			},
		},
	}
}
