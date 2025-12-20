package apicontracts

type PredictionRequest struct {
	Ticker      string `json:"ticker" binding:"required"`
	Timeframe   string `json:"timeframe" binding:"required"`
	Horizon     int    `json:"horizon" binding:"required,min=1,max=30"`
	Date        string `json:"date" binding:"required"`
	IncludeSHAP bool   `json:"include_shap"`
}

type BacktestRequest struct {
	Tickers   []string       `json:"tickers" binding:"required"`
	StartDate string         `json:"start_date" binding:"required"`
	EndDate   string         `json:"end_date" binding:"required"`
	Strategy  string         `json:"strategy" binding:"required"`
	Params    BacktestParams `json:"params" binding:"required"`
}

type BacktestParams struct {
	Entry_sigma float32 `json:"entry_sigma" binding:"required"`
	Exit_sigma  float32 `json:"exit_sigma" binding:"required"`
	Stop_sigma  float32 `json:"stop_sigma" binding:"required"`
	Commission  float32 `json:"commission" binding:"required"`
	Slippage    float32 `json:"slippage" binding:"required"`
}

type UpdateData struct {
	Tickers   []string `json:"tickers" binding:"required"`
	StartDate string   `json:"start_date" binding:"required"`
	EndDate   string   `json:"end_date" binding:"required"`
	Source    string   `jsaon:"source" binding:"required"`
}

type PredictionRequest_2 struct {
	Ticker    string `json:"ticker" binding:"required"`
	Timeframe string `json:"timeframe" binding:"required"`
	Horizon   int    `json:"horizon" binding:"required"`
	Date      string `json:"date" binding:"required"`
}
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

type PredictionResponse struct {
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
