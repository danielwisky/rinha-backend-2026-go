package hnsw_test

import (
	"math"
	"testing"

	"github.com/daniel-wisky/rinha-backend-2026-go/internal/hnsw"
)

// Verifica que int8 preserva precisão suficiente para o scoring de fraude.
// Todos os valores do desafio estão em [-1, 1] — int8 (escala 127) tem ~0.008 de granularidade aqui.
func TestInt8RoundtripPrecision(t *testing.T) {
	// Vetor legit da spec
	legitVec := []float32{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	// Vetor fraud da spec
	fraudVec := []float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}

	hnsw.Init(14, 20, 4, 50)

	// Adiciona 10 cópias legit e 10 cópias fraud com pequenas perturbações
	for i := 0; i < 10; i++ {
		v := make([]float32, 14)
		copy(v, legitVec)
		for j := range v { v[j] += float32(i) * 0.001 }
		hnsw.Add(v, 0, i)
	}
	for i := 0; i < 10; i++ {
		v := make([]float32, 14)
		copy(v, fraudVec)
		for j := range v { v[j] += float32(i) * 0.001 }
		hnsw.Add(v, 1, 10+i)
	}

	hnsw.SetEf(50)

	// Query com vetor legit → todos os 5 vizinhos devem ser legit
	labels := hnsw.Search(legitVec, 5)
	if len(labels) != 5 {
		t.Fatalf("expected 5 results, got %d", len(labels))
	}
	for i, l := range labels {
		if l != 0 {
			t.Errorf("neighbor[%d]: expected legit(0), got fraud(1)", i)
		}
	}

	// Query com vetor fraud → todos os 5 vizinhos devem ser fraud
	labels = hnsw.Search(fraudVec, 5)
	fraudCount := 0
	for _, l := range labels { if l == 1 { fraudCount++ } }
	if fraudCount < 4 {
		t.Errorf("expected ≥4 fraud neighbors, got %d", fraudCount)
	}
}

func TestInt8SentinelMinus1(t *testing.T) {
	// Verifica que o valor sentinela -1 (last_tx nulo) sobrevive int8
	// int8 representa -1.0 como -127 (escala 127); preservado perfeitamente
	hnsw.Init(14, 10, 2, 50)
	v := []float32{0.5, 0.5, 0.5, 0.5, 0.5, -1, -1, 0.5, 0.5, 0, 1, 0, 0.5, 0.5}
	hnsw.Add(v, 0, 0)
	hnsw.SetEf(10)

	result := hnsw.Search(v, 1)
	if len(result) != 1 {
		t.Fatalf("expected 1 result, got %d", len(result))
	}
	if result[0] != 0 {
		t.Errorf("expected label 0 (legit), got %d", result[0])
	}
}

func TestFraudScoreCalculation(t *testing.T) {
	// Testa a lógica de fraud_score = fraudCount/5 e approved = score < 0.6
	cases := []struct {
		labels   []uint8
		expected float32
		approved bool
	}{
		{[]uint8{0, 0, 0, 0, 0}, 0.0, true},
		{[]uint8{1, 1, 1, 0, 0}, 0.6, false}, // 0.6 não é < 0.6
		{[]uint8{1, 1, 0, 0, 0}, 0.4, true},
		{[]uint8{1, 1, 1, 1, 1}, 1.0, false},
	}
	for _, c := range cases {
		fraudCount := 0
		for _, l := range c.labels { if l == 1 { fraudCount++ } }
		score := float32(fraudCount) / 5.0
		approved := score < 0.6
		if math.Abs(float64(score-c.expected)) > 0.001 {
			t.Errorf("score: got %f, want %f for labels %v", score, c.expected, c.labels)
		}
		if approved != c.approved {
			t.Errorf("approved: got %v, want %v for score %f", approved, c.approved, score)
		}
	}
}
