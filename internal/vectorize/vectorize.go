package vectorize

import (
	"encoding/json"
	"os"
	"time"

	"github.com/daniel-wisky/rinha-backend-2026-go/internal/domain"
)

type config struct {
	MaxAmount            float32 `json:"max_amount"`
	MaxInstallments      float32 `json:"max_installments"`
	AmountVsAvgRatio     float32 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float32 `json:"max_minutes"`
	MaxKm                float32 `json:"max_km"`
	MaxTxCount24h        float32 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float32 `json:"max_merchant_avg_amount"`
}

// Vectorizer converts fraud requests into 14-dimensional float32 vectors.
// Safe for concurrent use after construction.
type Vectorizer struct {
	cfg     config
	mccRisk map[string]float32
}

// NewVectorizer loads normalization.json and mcc_risk.json from dir.
func NewVectorizer(dir string) (*Vectorizer, error) {
	normData, err := os.ReadFile(dir + "/normalization.json")
	if err != nil {
		return nil, err
	}
	var cfg config
	if err := json.Unmarshal(normData, &cfg); err != nil {
		return nil, err
	}

	mccData, err := os.ReadFile(dir + "/mcc_risk.json")
	if err != nil {
		return nil, err
	}
	var mccRisk map[string]float32
	if err := json.Unmarshal(mccData, &mccRisk); err != nil {
		return nil, err
	}

	return &Vectorizer{cfg: cfg, mccRisk: mccRisk}, nil
}

// Vectorize converts a Request into a 14-dimensional float32 vector.
func (vz *Vectorizer) Vectorize(r *domain.Request) [14]float32 {
	var v [14]float32

	// [0] amount
	v[0] = clamp(float32(r.Transaction.Amount) / vz.cfg.MaxAmount)

	// [1] installments
	v[1] = clamp(float32(r.Transaction.Installments) / vz.cfg.MaxInstallments)

	// [2] amount_vs_avg
	if r.Customer.AvgAmount > 0 {
		v[2] = clamp((float32(r.Transaction.Amount) / float32(r.Customer.AvgAmount)) / vz.cfg.AmountVsAvgRatio)
	}

	// Parse requested_at once for dims [3] and [4]
	t, _ := time.Parse(time.RFC3339, r.Transaction.RequestedAt)
	t = t.UTC()

	// [3] hour_of_day (0–23 → 0.0–1.0)
	v[3] = float32(t.Hour()) / 23.0

	// [4] day_of_week: spec wants Mon=0..Sun=6
	// Go's Weekday(): Sun=0, Mon=1..Sat=6 → (weekday+6)%7 gives Mon=0..Sun=6
	v[4] = float32((int(t.Weekday())+6)%7) / 6.0

	// [5] minutes_since_last_tx  |  [6] km_from_last_tx
	if r.LastTx == nil {
		v[5] = -1
		v[6] = -1
	} else {
		lastT, _ := time.Parse(time.RFC3339, r.LastTx.Timestamp)
		minutes := t.Sub(lastT).Minutes()
		v[5] = clamp(float32(minutes) / vz.cfg.MaxMinutes)
		v[6] = clamp(float32(r.LastTx.KmFromCurrent) / vz.cfg.MaxKm)
	}

	// [7] km_from_home
	v[7] = clamp(float32(r.Terminal.KmFromHome) / vz.cfg.MaxKm)

	// [8] tx_count_24h
	v[8] = clamp(float32(r.Customer.TxCount24h) / vz.cfg.MaxTxCount24h)

	// [9] is_online
	if r.Terminal.IsOnline {
		v[9] = 1
	}

	// [10] card_present
	if r.Terminal.CardPresent {
		v[10] = 1
	}

	// [11] unknown_merchant (1 if merchant NOT in known_merchants)
	known := false
	for _, m := range r.Customer.KnownMerchants {
		if m == r.Merchant.ID {
			known = true
			break
		}
	}
	if !known {
		v[11] = 1
	}

	// [12] mcc_risk (default 0.5)
	v[12] = vz.mccRisk[r.Merchant.MCC]
	if v[12] == 0 {
		v[12] = 0.5
	}

	// [13] merchant_avg_amount
	v[13] = clamp(float32(r.Merchant.AvgAmount) / vz.cfg.MaxMerchantAvgAmount)

	return v
}

func clamp(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
