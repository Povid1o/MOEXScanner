package src

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

func List() { //топ по ликвидности
	url := "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json?sort_column=VALTODAY&sort_order=desc&limit=30"

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		fmt.Println("err making request:", err)
		os.Exit(1)
	}
	req.Header.Set("Accept", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("error request:", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("error pars answer:", err)
		os.Exit(1)
	}

	var result map[string]any
	if err := json.Unmarshal(body, &result); err != nil {
		fmt.Println("json error:", err)
		os.Exit(1)
	}

	sec := result["securities"].(map[string]any)
	columns := sec["columns"].([]any)
	data := sec["data"].([]any)

	var idxSecid, idxShort, idxValue int
	for i, c := range columns {
		switch c {
		case "SECID":
			idxSecid = i
		case "SHORTNAME":
			idxShort = i
		case "VALTODAY":
			idxValue = i
		}
	}

	fmt.Println("Топ 30 бумаг по ликвидности (VALTODAY):")
	fmt.Println("--------------------------------------")
	for _, row := range data {
		r := row.([]any)
		fmt.Printf("%-10s %-30s %v\n", r[idxSecid], r[idxShort], r[idxValue])
	}
}

type Candle struct {
	Timestamp string  `json:"timestamp"`
	Open      float64 `json:"open"`
	Close     float64 `json:"close"`
	High      float64 `json:"high"`
	Low       float64 `json:"low"`
	Volume    float64 `json:"volume"`
	Ticker    string  `json:"ticker"`
}

// get historical info by switches from moex
func GetCandles(ticker, from, till string, interval int) ([]Candle, error) {
	url := fmt.Sprintf(
		"https://iss.moex.com/iss/engines/stock/markets/shares/securities/%s/candles.json?from=%s&till=%s&interval=%d",
		ticker, from, till, interval,
	)

	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("err request: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("err response: %v", err)
	}

	var result map[string]any
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("err JSON: %v", err)
	}

	candlesData, exists := result["candles"].(map[string]any)
	if !exists {
		return nil, fmt.Errorf("candles data not found in response")
	}

	data, exists := candlesData["data"].([]any)
	if !exists {
		return nil, fmt.Errorf("candles data array not found")
	}

	var candles []Candle
	for _, row := range data {
		r := row.([]any)

		var ts string
		switch v := r[0].(type) {
		case string:
			ts = v
		case float64:
			ts = fmt.Sprintf("%.0f", v)
		default:
			ts = "unknown"
		}

		candle := Candle{
			Timestamp: ts,
			Open:      safeFloat(r[1]),
			Close:     safeFloat(r[2]),
			High:      safeFloat(r[3]),
			Low:       safeFloat(r[4]),
			Volume:    safeFloat(r[6]),
			Ticker:    ticker,
		}
		candles = append(candles, candle)
	}

	return candles, nil
}

func safeFloat(v any) float64 {
	switch val := v.(type) {
	case float64:
		return val
	case int:
		return float64(val)
	default:
		return 0
	}
}
