// Package store defines the interface the api uses to query the vector store.
package store

// K is the number of nearest neighbors the API requests. Mirrors ivf.K but
// kept here to avoid the store interface depending on the ivf package.
const K = 5

// VectorStore is the contract the api handler depends on. It is intentionally
// minimal: readiness and a 14-dim k-NN search returning labels (0=legit, 1=fraud).
//
// Returning [K]uint8 (stack array) instead of []uint8 means the hot path avoids
// the per-request make() allocation.
type VectorStore interface {
	Ready() bool
	Search(vec [14]float32) ([K]uint8, error)
}
