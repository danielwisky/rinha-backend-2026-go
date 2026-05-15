// buildindex is a build-time tool that constructs the HNSW index from
// references.json.gz and serializes it to disk. The resulting <out>.graph and
// <out>.labels files are bundled into the store image so the runtime container
// can start instantly via index.NewFromDisk.
package main

import (
	"flag"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/daniel-wisky/rinha-backend-2026-go/internal/index"
)

func main() {
	refs := flag.String("refs", "resources/references.json.gz", "path to references.json.gz")
	out := flag.String("out", "resources/index", "output path prefix (.graph and .labels are appended)")
	flag.Parse()

	cfg := index.DefaultConfig()
	if v := os.Getenv("EF_BUILD"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.EfBuild = n
		}
	}
	if v := os.Getenv("EF_SEARCH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.EfSearch = n
		}
	}
	if v := os.Getenv("M"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.M = n
		}
	}

	idx := index.New(cfg)

	start := time.Now()
	idx.Load(*refs)
	log.Printf("build complete in %s", time.Since(start))

	if err := idx.Save(*out); err != nil {
		log.Fatalf("save: %v", err)
	}
	log.Printf("index written to %s.graph / %s.labels", *out, *out)
}
