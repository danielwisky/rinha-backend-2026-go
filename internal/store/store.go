// Package store defines the interface the api uses to query the vector store,
// plus an HTTP-client implementation that talks to the remote store service.
package store

// VectorStore is the contract the api handler depends on. It is intentionally
// minimal: readiness and a 14-dim k-NN search returning labels (0=legit, 1=fraud).
type VectorStore interface {
	Ready() bool
	Search(vec [14]float32) ([]uint8, error)
}
