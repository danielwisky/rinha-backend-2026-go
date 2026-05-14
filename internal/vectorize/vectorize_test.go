package vectorize_test

import (
	"math"
	"os"
	"testing"

	"github.com/daniel-wisky/rinha-backend-2026-go/internal/domain"
	"github.com/daniel-wisky/rinha-backend-2026-go/internal/vectorize"
)

var vz *vectorize.Vectorizer

func TestMain(m *testing.M) {
	var err error
	vz, err = vectorize.NewVectorizer("../../resources")
	if err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}

func approx(a, b float32) bool {
	return math.Abs(float64(a-b)) < 0.001
}

// Smoke test payload from test/smoke.js
var smokeReq = &domain.Request{
	ID: "tx-smoke-001",
	Transaction: domain.Transaction{
		Amount:       384.88,
		Installments: 3,
		RequestedAt:  "2026-03-11T20:23:35Z",
	},
	Customer: domain.Customer{
		AvgAmount:      769.76,
		TxCount24h:     3,
		KnownMerchants: []string{"MERC-009", "MERC-001", "MERC-001"},
	},
	Merchant: domain.Merchant{
		ID:        "MERC-001",
		MCC:       "5912",
		AvgAmount: 298.95,
	},
	Terminal: domain.Terminal{
		IsOnline:    false,
		CardPresent: true,
		KmFromHome:  13.7090520965,
	},
	LastTx: &domain.LastTx{
		Timestamp:     "2026-03-11T14:58:35Z",
		KmFromCurrent: 18.8626479774,
	},
}

func TestVectorizeSmokePayload(t *testing.T) {
	v, err := vz.Vectorize(smokeReq)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	t.Logf("vector: %v", v)

	// [0] amount = 384.88/10000 = 0.038488
	if !approx(v[0], 0.0385) {
		t.Errorf("[0] amount: got %f, want ~0.0385", v[0])
	}
	// [1] installments = 3/12 = 0.25
	if !approx(v[1], 0.25) {
		t.Errorf("[1] installments: got %f, want 0.25", v[1])
	}
	// [2] amount_vs_avg = (384.88/769.76)/10 = 0.05
	if !approx(v[2], 0.05) {
		t.Errorf("[2] amount_vs_avg: got %f, want 0.05", v[2])
	}
	// [3] hour = 20/23 = 0.8696
	if !approx(v[3], 0.8696) {
		t.Errorf("[3] hour_of_day: got %f, want ~0.8696", v[3])
	}
	// [4] 2026-03-11 = Wednesday; Go weekday=3; (3+6)%7=2; 2/6=0.3333
	if !approx(v[4], 0.3333) {
		t.Errorf("[4] day_of_week: got %f, want ~0.3333 (Wednesday)", v[4])
	}
	// [5] minutes_since_last_tx: 20:23:35 - 14:58:35 = 325 min; 325/1440 = 0.2257
	if !approx(v[5], 0.2257) {
		t.Errorf("[5] minutes_since_last_tx: got %f, want ~0.2257", v[5])
	}
	// [6] km_from_last_tx: 18.8626/1000 = 0.01886
	if !approx(v[6], 0.01886) {
		t.Errorf("[6] km_from_last_tx: got %f, want ~0.01886", v[6])
	}
	// [7] km_from_home: 13.709/1000 = 0.01371
	if !approx(v[7], 0.01371) {
		t.Errorf("[7] km_from_home: got %f, want ~0.01371", v[7])
	}
	// [8] tx_count_24h: 3/20 = 0.15
	if !approx(v[8], 0.15) {
		t.Errorf("[8] tx_count_24h: got %f, want 0.15", v[8])
	}
	// [9] is_online = 0
	if v[9] != 0 {
		t.Errorf("[9] is_online: got %f, want 0", v[9])
	}
	// [10] card_present = 1
	if v[10] != 1 {
		t.Errorf("[10] card_present: got %f, want 1", v[10])
	}
	// [11] unknown_merchant: MERC-001 is known → 0
	if v[11] != 0 {
		t.Errorf("[11] unknown_merchant: got %f, want 0 (MERC-001 is known)", v[11])
	}
	// [12] mcc_risk: 5912 → 0.20
	if !approx(v[12], 0.20) {
		t.Errorf("[12] mcc_risk: got %f, want 0.20", v[12])
	}
	// [13] merchant_avg_amount: 298.95/10000 = 0.029895
	if !approx(v[13], 0.02990) {
		t.Errorf("[13] merchant_avg_amount: got %f, want ~0.02990", v[13])
	}
}

func TestVectorizeNullLastTx(t *testing.T) {
	req := *smokeReq
	req.LastTx = nil
	v, err := vz.Vectorize(&req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v[5] != -1 {
		t.Errorf("[5] sentinel: got %f, want -1 when last_transaction is null", v[5])
	}
	if v[6] != -1 {
		t.Errorf("[6] sentinel: got %f, want -1 when last_transaction is null", v[6])
	}
}

func TestVectorizeUnknownMerchant(t *testing.T) {
	req := *smokeReq
	req.Merchant.ID = "MERC-UNKNOWN"
	v, err := vz.Vectorize(&req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v[11] != 1 {
		t.Errorf("[11] unknown_merchant: got %f, want 1 when merchant not in known list", v[11])
	}
}

func TestVectorizeClamp(t *testing.T) {
	req := *smokeReq
	req.Transaction.Amount = 999999 // well above max
	v, err := vz.Vectorize(&req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v[0] != 1.0 {
		t.Errorf("[0] clamp: got %f, want 1.0 for huge amount", v[0])
	}
}

func TestVectorizeInvalidTimestamp(t *testing.T) {
	req := *smokeReq
	req.Transaction.RequestedAt = "not-a-timestamp"
	_, err := vz.Vectorize(&req)
	if err == nil {
		t.Error("expected error for invalid requested_at, got nil")
	}
}

func TestVectorizeInvalidLastTxTimestamp(t *testing.T) {
	req := *smokeReq
	req.LastTx = &domain.LastTx{Timestamp: "bad", KmFromCurrent: 1.0}
	_, err := vz.Vectorize(&req)
	if err == nil {
		t.Error("expected error for invalid last_transaction.timestamp, got nil")
	}
}

func TestVectorizeUnknownMCC(t *testing.T) {
	req := *smokeReq
	req.Merchant.MCC = "9999" // not in mcc_risk.json
	v, err := vz.Vectorize(&req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !approx(v[12], 0.5) {
		t.Errorf("[12] mcc_risk default: got %f, want 0.5", v[12])
	}
}
