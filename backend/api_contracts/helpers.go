package apicontracts

import (
	"encoding/json"
	"strings"

	src "github.com/Povid1o/MOEXScanner.git/src"
)

func cleanAIResponse(response string) string {
	cleaned := strings.TrimPrefix(response, "```json")
	cleaned = strings.TrimPrefix(cleaned, "```")
	cleaned = strings.TrimSuffix(cleaned, "```")

	cleaned = strings.TrimSpace(cleaned)

	return cleaned
}

func extractJSONFromText(text string) string {

	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")

	if start == -1 || end == -1 || end < start {
		return ""
	}

	jsonStr := text[start : end+1]

	var js map[string]interface{}
	if json.Unmarshal([]byte(jsonStr), &js) == nil {
		return jsonStr
	}

	return ""
}

func getFallbackResponse(ticker string, horizon int, candles []src.Candle) AIResponse {

	var currentPrice float64
	if len(candles) > 0 {
		currentPrice = candles[len(candles)-1].Close
	} else {
		currentPrice = 123.4
	}

	return AIResponse{
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
			Text: "Волатильность повышена из-за роста исторической волатильности",
			TopFeatures: []Feature{
				{Name: "realized_vol_20", Value: 0.022, Contribution: 0.008},
				{Name: "beta_to_index", Value: 1.2, Contribution: 0.004},
				{Name: "volume_zscore", Value: 0.8, Contribution: 0.003},
			},
		},
	}
}
