package main

import (
	"log"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"time"

	"github.com/daniel-wisky/rinha-backend-2026-go/internal/handler"
	"github.com/daniel-wisky/rinha-backend-2026-go/internal/ivf"
	"github.com/daniel-wisky/rinha-backend-2026-go/internal/store"
	"github.com/daniel-wisky/rinha-backend-2026-go/internal/vectorize"
	"github.com/valyala/fasthttp"
)

func main() {
	resourcesPath := envOr("RESOURCES_PATH", "/app/resources")
	listenAddr := envOr("LISTEN_ADDR", ":8080")
	ivfPath := envOr("IVF_PATH", "/app/resources/ivf.bin")

	vz, err := vectorize.NewVectorizer(resourcesPath)
	if err != nil {
		log.Fatalf("vectorizer: %v", err)
	}

	start := time.Now()
	idx, err := ivf.LoadMmap(ivfPath)
	if err != nil {
		log.Fatalf("load ivf: %v", err)
	}
	log.Printf("ivf index mmap'd in %s", time.Since(start))

	ivfStore := &ivf.Store{Idx: idx}
	var vs store.VectorStore = ivfStore
	h := handler.New(vz, vs)

	srv := &fasthttp.Server{
		Handler:               h.Router,
		Name:                  "api",
		ReadTimeout:           5 * time.Second,
		WriteTimeout:          5 * time.Second,
		IdleTimeout:           120 * time.Second,
		DisableKeepalive:      false,
		TCPKeepalive:          true,
		NoDefaultServerHeader: true,
		NoDefaultDate:         true,
	}

	if os.Getenv("PPROF") == "1" {
		go func() {
			log.Println("pprof on :6060")
			_ = http.ListenAndServe(":6060", nil)
		}()
	}

	// Warmup before announcing /ready=200: walk Search over synthetic queries
	// to fault in pages, fill L1/L2, and let sonic / fasthttp warm their own
	// state when the first real burst arrives. The k6 health check has 60 s
	// margin (20 retries × 3 s) so this is well within budget.
	go func() {
		warmup(idx)
		ivfStore.MarkReady()
	}()

	log.Printf("api listening on %s", listenAddr)
	if err := srv.ListenAndServe(listenAddr); err != nil {
		log.Fatalf("listen: %v", err)
	}
}

// warmup runs ~1000 synthetic Search calls to fault in mmap pages, prime CPU
// caches, and trigger Go runtime warmup before /ready signals OK.
func warmup(idx *ivf.Index) {
	start := time.Now()
	rng := rand.New(rand.NewSource(1))
	var q [ivf.Dim]float32
	var sink uint8
	const n = 1000
	for i := 0; i < n; i++ {
		for j := 0; j < ivf.Dim; j++ {
			q[j] = rng.Float32()*2 - 1
		}
		top := idx.Search(&q)
		sink ^= top[0]
	}
	_ = sink
	log.Printf("warmup: %d queries in %s", n, time.Since(start))
}

func envOr(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}
