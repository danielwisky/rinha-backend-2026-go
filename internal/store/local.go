package store

import "github.com/daniel-wisky/rinha-backend-2026-go/internal/index"

// Local is an in-process VectorStore backed by the HNSW index directly.
// Useful in tests; the production store binary embeds it inside an HTTP server.
type Local struct {
	idx *index.Index
}

// NewLocal wraps an Index as a VectorStore.
func NewLocal(idx *index.Index) *Local {
	return &Local{idx: idx}
}

func (l *Local) Ready() bool { return l.idx.Ready() }

func (l *Local) Search(vec [14]float32) ([]uint8, error) {
	return l.idx.Search(vec)
}
