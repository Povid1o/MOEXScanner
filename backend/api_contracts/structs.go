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
