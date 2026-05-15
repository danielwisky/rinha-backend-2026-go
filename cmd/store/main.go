package main

import (
	"log"
	"os"
	"strconv"
	"time"

	"github.com/daniel-wisky/rinha-backend-2026-go/internal/index"
	"github.com/daniel-wisky/rinha-backend-2026-go/internal/server"
	"github.com/daniel-wisky/rinha-backend-2026-go/internal/store"
	"github.com/valyala/fasthttp"
)

func main() {
	indexPath := envOr("INDEX_PATH", "/app/resources/index")
	listenAddr := envOr("LISTEN_ADDR", ":9990")

	cfg := index.DefaultConfig()
	if v := os.Getenv("EF_SEARCH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.EfSearch = n
		}
	}
	if v := os.Getenv("MAX_ELEMENTS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.MaxElements = n
		}
	}

	start := time.Now()
	idx, err := index.NewFromDisk(cfg, indexPath)
	if err != nil {
		log.Fatalf("load index: %v", err)
	}
	log.Printf("index ready in %s", time.Since(start))

	handler := server.New(store.NewLocal(idx))

	srv := &fasthttp.Server{
		Handler:               handler.Router,
		Name:                  "store",
		ReadTimeout:           5 * time.Second,
		WriteTimeout:          5 * time.Second,
		TCPKeepalive:          true,
		ReduceMemoryUsage:     true,
		NoDefaultServerHeader: true,
		NoDefaultDate:         true,
	}

	log.Printf("store listening on %s", listenAddr)
	if err := srv.ListenAndServe(listenAddr); err != nil {
		log.Fatalf("listen: %v", err)
	}
}

func envOr(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}
