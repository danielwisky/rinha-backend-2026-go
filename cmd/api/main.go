package main

import (
	"log"
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

	var vs store.VectorStore = &ivf.Store{Idx: idx}
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

	log.Printf("api listening on %s", listenAddr)
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
