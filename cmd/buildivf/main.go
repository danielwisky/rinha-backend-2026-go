// buildivf produces an IVF binary index from references.json.gz.
// Replaces the HNSW build pipeline for the exact-within-bucket k-NN approach.
package main

import (
	"flag"
	"log"
	"time"

	"github.com/daniel-wisky/rinha-backend-2026-go/internal/ivf"
)

func main() {
	refs := flag.String("refs", "resources/references.json.gz", "path to references.json.gz")
	out := flag.String("out", "resources/ivf.bin", "output binary path")
	flag.Parse()

	start := time.Now()
	idx, err := ivf.BuildFromRefs(*refs)
	if err != nil {
		log.Fatalf("build: %v", err)
	}
	log.Printf("build complete in %s", time.Since(start))

	if err := idx.Save(*out); err != nil {
		log.Fatalf("save: %v", err)
	}
	log.Printf("ivf written to %s", *out)
}
