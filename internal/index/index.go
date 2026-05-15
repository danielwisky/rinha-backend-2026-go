package index

import (
	"compress/gzip"
	"encoding/json"
	"errors"
	"log"
	"os"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"

	"github.com/daniel-wisky/rinha-backend-2026-go/internal/hnsw"
)

// Config controls HNSW index parameters.
type Config struct {
	Dim         int
	MaxElements int
	M           int
	EfBuild     int
	EfSearch    int
	K           int
}

// DefaultConfig returns the competition-tuned defaults.
func DefaultConfig() Config {
	return Config{
		Dim:         14,
		MaxElements: 3_000_000,
		M:           4,
		EfBuild:     200,
		EfSearch:    200,
		K:           5,
	}
}

// Index wraps the hnswlib CGO index with a readiness flag.
// Safe for concurrent Search calls once Load returns.
type Index struct {
	cfg   Config
	ready atomic.Bool
}

// New initializes a new (empty) HNSW index with the given config.
// Call Load to populate it from references.json.gz, or use NewFromDisk to
// deserialize a pre-built index.
func New(cfg Config) *Index {
	hnsw.Init(cfg.Dim, cfg.MaxElements, cfg.M, cfg.EfBuild)
	return &Index{cfg: cfg}
}

// NewFromDisk loads a previously serialized index from disk. Ready() returns
// true immediately upon successful return — there is no background loading.
func NewFromDisk(cfg Config, path string) (*Index, error) {
	n, err := hnsw.Load(path, cfg.Dim, cfg.MaxElements)
	if err != nil {
		return nil, err
	}
	hnsw.SetEf(cfg.EfSearch)
	idx := &Index{cfg: cfg}
	idx.ready.Store(true)
	log.Printf("index loaded from %s: %d vectors, ef_search=%d", path, n, cfg.EfSearch)
	return idx, nil
}

// Save serializes the index to <path>.graph and <path>.labels.
func (idx *Index) Save(path string) error {
	return hnsw.Save(path)
}

// Load streams reference vectors from a gzipped JSON file into the index.
// Intended to run in a goroutine; sets Ready() = true when done.
func (idx *Index) Load(path string) {
	prevGC := debug.SetGCPercent(20)
	defer debug.SetGCPercent(prevGC)

	log.Printf("streaming %s into HNSW index (M=%d, ef_build=%d, int8)...", path, idx.cfg.M, idx.cfg.EfBuild)

	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("open refs: %v", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		log.Fatalf("gzip reader: %v", err)
	}
	defer gz.Close()

	type entry struct {
		vec   [14]float32
		label uint8
	}

	entryCh := make(chan entry, 64)

	// Producer: single goroutine decodes streaming JSON
	go func() {
		defer close(entryCh)
		dec := json.NewDecoder(gz)

		if _, err := dec.Token(); err != nil { // consume '['
			log.Fatalf("json opening token: %v", err)
		}

		var raw struct {
			Vector []float32 `json:"vector"`
			Label  string    `json:"label"`
		}
		for dec.More() {
			if err := dec.Decode(&raw); err != nil {
				log.Fatalf("decode entry: %v", err)
			}
			var e entry
			copy(e.vec[:], raw.Vector)
			if raw.Label == "fraud" {
				e.label = 1
			}
			entryCh <- e
		}
	}()

	// Workers: parallel insertion into hnswlib (thread-safe addPoint).
	// GOMAXPROCS=1 means only one P; keep workers low to minimize stacks.
	numWorkers := 2

	var (
		total     atomic.Int64
		idCounter atomic.Int32
		wg        sync.WaitGroup
	)

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for e := range entryCh {
				id := int(idCounter.Add(1)) - 1
				hnsw.Add(e.vec[:], int(e.label), id)
				total.Add(1)
			}
		}()
	}
	wg.Wait()

	hnsw.SetEf(idx.cfg.EfSearch)

	runtime.GC()
	debug.FreeOSMemory()

	log.Printf("index ready: %d vectors, ef_search=%d", total.Load(), idx.cfg.EfSearch)
	idx.ready.Store(true)
}

// Ready reports whether the index has finished loading.
func (idx *Index) Ready() bool { return idx.ready.Load() }

// Search returns k-NN labels for the given 14-dim vector.
// Returns an error if the index is not ready.
func (idx *Index) Search(vec [14]float32) ([]uint8, error) {
	if !idx.ready.Load() {
		return nil, errors.New("index not ready")
	}
	return hnsw.Search(vec[:], idx.cfg.K), nil
}
