package vectorize

import (
	"encoding/json"
	"fmt"
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
	cfg config

	// mccRisk is a dense lookup keyed by the parsed integer MCC. 10000 floats
	// = 40 KB, fits comfortably in L1. Slot value 0.5 is the default for
	// unknown MCCs; the bitmap below tells which slots came from the JSON.
	mccRisk [10000]float32
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
	var mccMap map[string]float32
	if err := json.Unmarshal(mccData, &mccMap); err != nil {
		return nil, err
	}

	vz := &Vectorizer{cfg: cfg}
	for i := range vz.mccRisk {
		vz.mccRisk[i] = 0.5 // default when MCC isn't in the risk table
	}
	for k, v := range mccMap {
		idx, err := parseMCC(k)
		if err != nil {
			return nil, fmt.Errorf("mcc_risk: bad key %q: %w", k, err)
		}
		vz.mccRisk[idx] = v
	}

	return vz, nil
}

// parseMCC parses a 4-digit MCC string into an int 0-9999. Returns an error if
// the string is malformed; never called on the hot path (used at boot to seed
// the lookup table). The hot-path equivalent is parseMCCFast.
func parseMCC(s string) (int, error) {
	if len(s) == 0 || len(s) > 4 {
		return 0, fmt.Errorf("mcc must be 1-4 digits, got %d", len(s))
	}
	n := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("non-digit %q in mcc", c)
		}
		n = n*10 + int(c-'0')
	}
	return n, nil
}

// parseMCCFast is the hot-path version: assumes a digits-only string of length
// 1-4 and skips error checks. Falls through to a default of 0 on bad input,
// which routes to the 0.5 default slot.
func parseMCCFast(s string) int {
	n := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			return 0
		}
		n = n*10 + int(c-'0')
		if n >= 10000 {
			return 0
		}
	}
	return n
}

// Vectorize converts a Request into a 14-dimensional float32 vector.
// Returns an error if any timestamp field is not valid RFC3339.
func (vz *Vectorizer) Vectorize(r *domain.Request) ([14]float32, error) {
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
	t, err := time.Parse(time.RFC3339, r.Transaction.RequestedAt)
	if err != nil {
		return v, fmt.Errorf("requested_at: %w", err)
	}
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
		lastT, err := time.Parse(time.RFC3339, r.LastTx.Timestamp)
		if err != nil {
			return v, fmt.Errorf("last_transaction.timestamp: %w", err)
		}
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

	// [12] mcc_risk — dense lookup. Unknown MCCs map to slot 0 (parseMCCFast
	// returns 0 on bad input), and `mccRisk[0]` is seeded with 0.5 — so
	// MCC "0" or any malformed code lands on the default.
	v[12] = vz.mccRisk[parseMCCFast(r.Merchant.MCC)]

	// [13] merchant_avg_amount
	v[13] = clamp(float32(r.Merchant.AvgAmount) / vz.cfg.MaxMerchantAvgAmount)

	return v, nil
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
