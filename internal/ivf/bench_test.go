package ivf

import (
	"testing"
)

// Loads /tmp/ivf-test.bin if present and benchmarks Search throughput.
// Run with: go test -bench=BenchmarkSearch -benchtime=2s ./internal/ivf/
func BenchmarkSearch(b *testing.B) {
	idx, err := LoadMmap("/tmp/ivf-test.bin")
	if err != nil {
		b.Skipf("no index at /tmp/ivf-test.bin: %v", err)
	}
	q := [Dim]float32{0.2, 0.05, 0.5, 0.6, 0.3, 0.2, 0.0, 0.1, 0.05, 0.0, 1.0, 0.0, 0.3, 0.05}

	b.ReportAllocs()
	b.ResetTimer()
	var sink uint8
	for i := 0; i < b.N; i++ {
		top := idx.Search(&q)
		sink ^= top[0]
	}
	_ = sink
}

func BenchmarkNearestCluster(b *testing.B) {
	idx, err := LoadMmap("/tmp/ivf-test.bin")
	if err != nil {
		b.Skipf("no index at /tmp/ivf-test.bin: %v", err)
	}
	q := [Dim]float32{0.2, 0.05, 0.5, 0.6, 0.3, 0.2, 0.0, 0.1, 0.05, 0.0, 1.0, 0.0, 0.3, 0.05}

	b.ReportAllocs()
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		sink ^= idx.NearestCluster(&q)
	}
	_ = sink
}
