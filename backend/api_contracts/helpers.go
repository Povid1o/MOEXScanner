package apicontracts

import (
	"regexp"
	"strings"
)

// cleanAIResponse удаляет оболочку и лишние символы вокруг JSON-ответа от AI
func cleanAIResponse(response string) string {
	response = strings.TrimSpace(response)
	if strings.HasPrefix(response, "```json") {
		response = strings.TrimPrefix(response, "```json")
	} else if strings.HasPrefix(response, "```") {
		response = strings.TrimPrefix(response, "```")
	}

	if strings.HasSuffix(response, "```") {
		response = strings.TrimSuffix(response, "```")
	}
	response = strings.TrimSpace(response)
	startIdx := strings.Index(response, "{")
	if startIdx > 0 {
		response = response[startIdx:]
	}
	endIdx := strings.LastIndex(response, "}")
	if endIdx >= 0 && endIdx < len(response)-1 {
		response = response[:endIdx+1]
	}

	return strings.TrimSpace(response)
}

// fixCommonJSONIssues исправляет типичные проблемные паттерны в JSON-строке
func fixCommonJSONIssues(jsonStr string) string {
	rePercent := regexp.MustCompile(`"([^"]+)":\s*"([\d.]+)%"`)
	jsonStr = rePercent.ReplaceAllString(jsonStr, `"$1": $2`)
	reInvalidNumber := regexp.MustCompile(`"([^"]+)":\s*([\d.]+),?\s*%`)
	jsonStr = reInvalidNumber.ReplaceAllString(jsonStr, `"$1": $2`)
	reTrailingComma := regexp.MustCompile(`,(\s*[}\]])`)
	for reTrailingComma.MatchString(jsonStr) {
		jsonStr = reTrailingComma.ReplaceAllString(jsonStr, "$1")
	}

	return jsonStr
}

// getFallbackResponse строит запасной ответ, если парсинг ответа AI не удался
func getFallbackResponse(ticker string, horizon int, aiResponse string, err error) *PredictionResponse {
	currentPrice := 123.4
	if strings.Contains(ticker, "SBER") {
		currentPrice = 280.5
	} else if strings.Contains(ticker, "GAZP") {
		currentPrice = 160.3
	}

	return &PredictionResponse{
		Ticker:  ticker,
		Horizon: horizon,
		PredictedVolatility: PredictedVolatility{
			Median:      0.024,
			Lower1Sigma: 0.018,
			Upper1Sigma: 0.031,
			Lower2Sigma: 0.015,
			Upper2Sigma: 0.038,
		},
		Confidence: 0.68,
		Trend: Trend{
			Direction:  "sideways",
			Confidence: "medium",
			Strength:   0.42,
		},
		Channel: Channel{
			Upper2Sigma:  currentPrice * 1.033,
			Upper1Sigma:  currentPrice * 1.019,
			CurrentPrice: currentPrice,
			Lower1Sigma:  currentPrice * 0.981,
			Lower2Sigma:  currentPrice * 0.967,
		},
		TradingSignal: TradingSignal{
			Action:       "HOLD",
			Entry:        currentPrice * 0.98,
			Target:       currentPrice * 1.02,
			StopLoss:     currentPrice * 0.96,
			PositionSize: 0.35,
			Reason:       "Fallback data - AI response parsing failed: " + err.Error(),
		},
		TailRisk: TailRisk{
			Warning:      false,
			Probability:  0.05,
			ExpectedLoss: nil,
		},
		VolumeContext: VolumeContext{
			Zscore:        0.8,
			SpikeDetected: false,
			PocDistance:   -0.02,
			VaPosition:    "inside",
		},
		Explanation: Explanation{
			Text: "Анализ на основе исторических данных. Обратите внимание: ответ от AI не был корректно распознан, используются базовые данные.",
			TopFeatures: []Feature{
				{Name: "realized_vol_20", Value: 0.022, Contribution: 0.008},
				{Name: "beta_to_index", Value: 1.2, Contribution: 0.004},
				{Name: "volume_zscore", Value: 0.8, Contribution: 0.003},
			},
		},
	}
}
