package handler

import (
	"log"

	"github.com/bytedance/sonic"
	"github.com/daniel-wisky/rinha-backend-2026-go/internal/domain"
	"github.com/daniel-wisky/rinha-backend-2026-go/internal/store"
	"github.com/daniel-wisky/rinha-backend-2026-go/internal/vectorize"
	"github.com/valyala/fasthttp"
)

// API handles HTTP requests for the fraud scoring service.
type API struct {
	vz    *vectorize.Vectorizer
	store store.VectorStore
}

// New creates an API handler with the given vectorizer and store.
func New(vz *vectorize.Vectorizer, vs store.VectorStore) *API {
	return &API{vz: vz, store: vs}
}

// Router dispatches requests to the appropriate handler.
func (h *API) Router(ctx *fasthttp.RequestCtx) {
	switch string(ctx.Path()) {
	case "/ready":
		h.ready(ctx)
	case "/fraud-score":
		if string(ctx.Method()) == "POST" {
			h.fraudScore(ctx)
		} else {
			ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		}
	default:
		ctx.SetStatusCode(fasthttp.StatusNotFound)
	}
}

func (h *API) ready(ctx *fasthttp.RequestCtx) {
	if !h.store.Ready() {
		ctx.SetStatusCode(fasthttp.StatusServiceUnavailable)
		return
	}
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetBodyString("ok")
}

func (h *API) fraudScore(ctx *fasthttp.RequestCtx) {
	var req domain.Request
	if err := sonic.Unmarshal(ctx.PostBody(), &req); err != nil {
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		return
	}

	vec, err := h.vz.Vectorize(&req)
	if err != nil {
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		return
	}

	labels, err := h.store.Search(vec)
	if err != nil {
		log.Printf("store error: %v", err)
		ctx.SetStatusCode(fasthttp.StatusInternalServerError)
		return
	}

	resp := domain.NewResponse(labels)

	b, err := sonic.Marshal(resp)
	if err != nil {
		log.Printf("encode: %v", err)
		ctx.SetStatusCode(fasthttp.StatusInternalServerError)
		return
	}
	ctx.SetContentType("application/json")
	ctx.SetBody(b)
}
