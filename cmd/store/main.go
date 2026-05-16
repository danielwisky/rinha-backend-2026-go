package main

import (
	"log"
	"net"
	"os"
	"strconv"
	"time"

	"github.com/daniel-wisky/rinha-backend-2026-go/internal/index"
	"github.com/daniel-wisky/rinha-backend-2026-go/internal/server"
	"github.com/valyala/fasthttp"
)

func main() {
	indexPath := envOr("INDEX_PATH", "/app/resources/index")
	listenAddr := envOr("LISTEN_ADDR", ":9990")
	// LISTEN_SOCKET, when set, makes the store listen on a Unix Domain Socket
	// instead of TCP. UDS avoids the TCP/IP stack on localhost — roughly 50–200µs
	// per request, which is significant when chasing sub-millisecond p99.
	listenSocket := os.Getenv("LISTEN_SOCKET")

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

	handler := server.New(index.NewLocalStore(idx))

	srv := &fasthttp.Server{
		Handler:               handler.Router,
		Name:                  "store",
		ReadTimeout:           5 * time.Second,
		WriteTimeout:          5 * time.Second,
		IdleTimeout:           120 * time.Second,
		TCPKeepalive:          true,
		NoDefaultServerHeader: true,
		NoDefaultDate:         true,
	}

	if listenSocket != "" {
		// Clean up any stale socket file from a previous crashed run.
		_ = os.Remove(listenSocket)
		ln, err := net.Listen("unix", listenSocket)
		if err != nil {
			log.Fatalf("listen unix: %v", err)
		}
		// 0666 so the api containers can write to it regardless of UID.
		if err := os.Chmod(listenSocket, 0o666); err != nil {
			log.Fatalf("chmod socket: %v", err)
		}
		log.Printf("store listening on unix:%s", listenSocket)
		if err := srv.Serve(ln); err != nil {
			log.Fatalf("serve unix: %v", err)
		}
		return
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
