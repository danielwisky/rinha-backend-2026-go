// Package server exposes the store service over HTTP using a binary protocol on
// the hot path (POST /search). /ready is a plain text 200/503 check.
package server

import (
	"encoding/binary"
	"math"

	"github.com/daniel-wisky/rinha-backend-2026-go/internal/store"
	"github.com/valyala/fasthttp"
)

// Store handles /ready and /search for a VectorStore.
type Store struct {
	vs store.VectorStore
}

// New constructs a Store handler backed by vs.
func New(vs store.VectorStore) *Store {
	return &Store{vs: vs}
}

// Router dispatches requests.
func (s *Store) Router(ctx *fasthttp.RequestCtx) {
	switch string(ctx.Path()) {
	case "/ready":
		s.ready(ctx)
	case "/search":
		if string(ctx.Method()) == "POST" {
			s.search(ctx)
		} else {
			ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		}
	default:
		ctx.SetStatusCode(fasthttp.StatusNotFound)
	}
}

func (s *Store) ready(ctx *fasthttp.RequestCtx) {
	if !s.vs.Ready() {
		ctx.SetStatusCode(fasthttp.StatusServiceUnavailable)
		return
	}
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetBodyString("ok")
}

func (s *Store) search(ctx *fasthttp.RequestCtx) {
	body := ctx.PostBody()
	if len(body) != 56 {
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		return
	}
	var vec [14]float32
	for i := 0; i < 14; i++ {
		bits := binary.LittleEndian.Uint32(body[i*4:])
		vec[i] = math.Float32frombits(bits)
	}
	labels, err := s.vs.Search(vec)
	if err != nil {
		ctx.SetStatusCode(fasthttp.StatusServiceUnavailable)
		return
	}
	ctx.SetContentType("application/octet-stream")
	ctx.SetBody(labels)
}
