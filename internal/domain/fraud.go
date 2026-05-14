package domain

// Request is the POST /fraud-score payload.
type Request struct {
	ID          string      `json:"id"`
	Transaction Transaction `json:"transaction"`
	Customer    Customer    `json:"customer"`
	Merchant    Merchant    `json:"merchant"`
	Terminal    Terminal    `json:"terminal"`
	LastTx      *LastTx     `json:"last_transaction"`
}

type Transaction struct {
	Amount       float64 `json:"amount"`
	Installments int     `json:"installments"`
	RequestedAt  string  `json:"requested_at"`
}

type Customer struct {
	AvgAmount      float64  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type Merchant struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float64 `json:"avg_amount"`
}

type Terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float64 `json:"km_from_home"`
}

type LastTx struct {
	Timestamp     string  `json:"timestamp"`
	KmFromCurrent float64 `json:"km_from_current"`
}

// Response is the POST /fraud-score response body.
type Response struct {
	Approved   bool    `json:"approved"`
	FraudScore float32 `json:"fraud_score"`
}

// NewResponse computes the fraud score from k-NN labels (0=legit, 1=fraud).
// score = fraudCount/5; approved when score < 0.6 (i.e. ≤2 fraud neighbors).
func NewResponse(labels []uint8) Response {
	fraudCount := 0
	for _, l := range labels {
		if l == 1 {
			fraudCount++
		}
	}
	score := float32(fraudCount) / float32(len(labels))
	return Response{Approved: score < 0.6, FraudScore: score}
}
