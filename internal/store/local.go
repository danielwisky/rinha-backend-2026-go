package store

// Intentionally empty. The in-process VectorStore implementation lives in
// internal/index (see local_store.go) to keep this package free of CGO
// transitive imports — cmd/api builds with CGO_ENABLED=0 and only needs the
// HTTPClient from this package.
