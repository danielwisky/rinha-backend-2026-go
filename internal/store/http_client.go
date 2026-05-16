package store

import (
	"encoding/binary"
	"errors"
	"math"
	"net"
	"net/url"
	"strings"
	"time"

	"github.com/valyala/fasthttp"
)

// HTTPClient is a VectorStore that calls a remote store service over HTTP using
// a tiny binary protocol on the hot path. /ready uses HTTP GET; /search posts
// 56 bytes (14 × float32 LE) and receives K bytes back, one label per neighbor.
//
// Supports both TCP (http://host:port) and Unix Domain Socket (unix:///path/to/sock)
// transports. UDS is significantly faster on localhost — no TCP/IP stack, no port
// allocation, no connection 3-way handshake.
type HTTPClient struct {
	// hostHeader is the Host header value used in HTTP requests. For TCP it's
	// host:port; for UDS we still need a non-empty Host so fasthttp accepts the URI.
	hostHeader string
	client     *fasthttp.HostClient
}

// NewHTTPClient builds a client targeting baseURL. Scheme can be "http" (TCP) or
// "unix" (Unix Domain Socket — path follows after unix://).
//
//	NewHTTPClient("http://store:9990")
//	NewHTTPClient("unix:///var/run/store.sock")
func NewHTTPClient(baseURL string) (*HTTPClient, error) {
	if strings.HasPrefix(baseURL, "unix://") {
		path := strings.TrimPrefix(baseURL, "unix://")
		if path == "" {
			return nil, errors.New("store: empty unix socket path")
		}
		return &HTTPClient{
			hostHeader: "store",
			client: &fasthttp.HostClient{
				Addr:                path,
				IsTLS:               false,
				MaxConns:            128,
				MaxIdleConnDuration: 60 * time.Second,
				ReadTimeout:         2 * time.Second,
				WriteTimeout:        1 * time.Second,
				Dial: func(_ string) (net.Conn, error) {
					return net.DialTimeout("unix", path, time.Second)
				},
				RetryIf: func(*fasthttp.Request) bool { return true },
			},
		}, nil
	}

	u, err := url.Parse(baseURL)
	if err != nil {
		return nil, err
	}
	host := u.Host
	if host == "" {
		return nil, errors.New("store: empty host in STORE_URL")
	}
	return &HTTPClient{
		hostHeader: host,
		client: &fasthttp.HostClient{
			Addr:                host,
			MaxConns:            512,
			MaxConnWaitTimeout:  500 * time.Millisecond,
			MaxIdleConnDuration: 60 * time.Second,
			ReadTimeout:         2 * time.Second,
			WriteTimeout:        1 * time.Second,
			// /search is idempotent (pure read against the index); retry on
			// transient pool-stale errors instead of bubbling them up as 500s.
			RetryIf: func(*fasthttp.Request) bool { return true },
		},
	}, nil
}

// Ready returns true iff the store responds 200 on GET /ready.
func (c *HTTPClient) Ready() bool {
	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	req.SetRequestURI("http://" + c.hostHeader + "/ready")
	req.Header.SetMethod("GET")
	if err := c.client.DoTimeout(req, resp, 2*time.Second); err != nil {
		return false
	}
	return resp.StatusCode() == fasthttp.StatusOK
}

// Search posts the 14-dim vector as 56 raw bytes and returns the K labels.
func (c *HTTPClient) Search(vec [14]float32) ([]uint8, error) {
	var buf [56]byte
	for i, f := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	req.SetRequestURI("http://" + c.hostHeader + "/search")
	req.Header.SetMethod("POST")
	req.Header.SetContentType("application/octet-stream")
	req.SetBody(buf[:])

	if err := c.client.DoTimeout(req, resp, 2*time.Second); err != nil {
		return nil, err
	}
	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, errors.New("store: non-200 response")
	}
	body := resp.Body()
	out := make([]uint8, len(body))
	copy(out, body)
	return out, nil
}
