package handler

import (
	"log"
	"sync"

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

// Pool of domain.Request structs so each fraud-score request doesn't allocate
// a fresh one (with its nested slice/map fields) on the hot path.
//
// KnownMerchants pre-allocated with cap=8 to cover the typical case without
// triggering a realloc inside sonic.Unmarshal.
var requestPool = sync.Pool{
	New: func() any {
		r := new(domain.Request)
		r.Customer.KnownMerchants = make([]string, 0, 8)
		return r
	},
}

// fastJSON is sonic's "fastest" config: skips UTF-8 validation and key
// sort order. Payloads come from the official k6 load harness, not adversarial
// input.
var fastJSON = sonic.ConfigFastest

// fraud_score = k/5 where k ∈ {0,1,2,3,4,5}. Only 6 possible response bodies —
// precompute both approved variants so the hot path is just a SetBody.
//   approved = score < 0.6, so k=3 (0.6) is the cutoff and below.
var responseBodies = [2][6][]byte{
	// approved=false  (k=3,4,5 → score 0.6, 0.8, 1)
	{
		[]byte(`{"approved":false,"fraud_score":0}`),
		[]byte(`{"approved":false,"fraud_score":0.2}`),
		[]byte(`{"approved":false,"fraud_score":0.4}`),
		[]byte(`{"approved":false,"fraud_score":0.6}`),
		[]byte(`{"approved":false,"fraud_score":0.8}`),
		[]byte(`{"approved":false,"fraud_score":1}`),
	},
	// approved=true  (k=0,1,2 → score 0, 0.2, 0.4)
	{
		[]byte(`{"approved":true,"fraud_score":0}`),
		[]byte(`{"approved":true,"fraud_score":0.2}`),
		[]byte(`{"approved":true,"fraud_score":0.4}`),
		[]byte(`{"approved":true,"fraud_score":0.6}`),
		[]byte(`{"approved":true,"fraud_score":0.8}`),
		[]byte(`{"approved":true,"fraud_score":1}`),
	},
}

func (h *API) fraudScore(ctx *fasthttp.RequestCtx) {
	req := requestPool.Get().(*domain.Request)
	defer func() {
		// Reset slice to avoid retaining large backing arrays in the pool.
		req.Customer.KnownMerchants = req.Customer.KnownMerchants[:0]
		req.LastTx = nil
		requestPool.Put(req)
	}()

	if err := fastJSON.Unmarshal(ctx.PostBody(), req); err != nil {
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		return
	}

	vec, err := h.vz.Vectorize(req)
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

	// Score = fraudCount/5, approved iff < 0.6 (i.e. fraudCount ≤ 2).
	fraudCount := 0
	for _, l := range labels {
		if l == 1 {
			fraudCount++
		}
	}
	approvedIdx := 0
	if fraudCount < 3 {
		approvedIdx = 1
	}
	ctx.SetContentType("application/json")
	ctx.SetBody(responseBodies[approvedIdx][fraudCount])
}
