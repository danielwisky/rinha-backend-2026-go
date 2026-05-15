package index

// LocalStore is an in-process VectorStore implementation that wraps an Index
// directly. It is used by the store binary, which embeds the index in its own
// process. cmd/api never imports this — it would force CGO into the api build.
type LocalStore struct {
	idx *Index
}

// NewLocalStore wraps idx as a VectorStore-shaped value (Ready/Search).
func NewLocalStore(idx *Index) *LocalStore {
	return &LocalStore{idx: idx}
}

func (l *LocalStore) Ready() bool { return l.idx.Ready() }

func (l *LocalStore) Search(vec [14]float32) ([]uint8, error) {
	return l.idx.Search(vec)
}
