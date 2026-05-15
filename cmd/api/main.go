package main

import (
	"log"
	"os"
	"time"

	"github.com/daniel-wisky/rinha-backend-2026-go/internal/handler"
	"github.com/daniel-wisky/rinha-backend-2026-go/internal/store"
	"github.com/daniel-wisky/rinha-backend-2026-go/internal/vectorize"
	"github.com/valyala/fasthttp"
)

func main() {
	storeURL := envOr("STORE_URL", "http://store:9990")
	resourcesPath := envOr("RESOURCES_PATH", "/app/resources")
	listenAddr := envOr("LISTEN_ADDR", ":8080")

	vz, err := vectorize.NewVectorizer(resourcesPath)
	if err != nil {
		log.Fatalf("vectorizer: %v", err)
	}

	client, err := store.NewHTTPClient(storeURL)
	if err != nil {
		log.Fatalf("store client: %v", err)
	}

	h := handler.New(vz, client)

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

	log.Printf("api listening on %s (store=%s)", listenAddr, storeURL)
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
