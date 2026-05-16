// Package ivf is an exact-within-bucket k-NN search using IVF partitioning by
// 4 binary features (16 buckets). Each search routes the query to one bucket
// and does brute-force distance over ~N/16 vectors with int8 storage.
//
// Why this beats HNSW for fraud-detection:
//
//  1. No graph traversal overhead, no CGO. Pure Go, branch-friendly.
//  2. Per-query data is bounded — one bucket fits in L2/L3 cache. Memory-
//     bandwidth bound, not lock or pointer-chasing bound.
//  3. Exact within the bucket — there's no ef_search/recall tradeoff. The
//     only approximation is the partitioning itself (queries get the
//     5-NN among same-features-class neighbors).
//
// Disk format ("ivf" binary, little-endian):
//   uint32 magic = 0x49564601 ("IVF\x01")
//   uint32 dim    (= 14)
//   uint32 k      (= 5)
//   uint32 nbuckets (= 16)
//   uint32 counts[nbuckets]              — number of vectors in each bucket
//   then for each bucket: count*dim bytes of int8 vectors, then count bytes
//   of labels (0=legit, 1=fraud).
package ivf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"unsafe"
)

const (
	Dim     = 14
	K       = 5
	Buckets = 64
)

var (
	magic = uint32(0x49564601)

	errBadMagic = errors.New("ivf: bad magic")
	errBadDim   = errors.New("ivf: dim mismatch")
)

// Index holds 16 bucket slices of contiguous int8 vectors and uint8 labels.
// The backing arrays may be malloc'd (Load) or mmap'd (LoadMmap).
type Index struct {
	counts  [Buckets]uint32 // number of vectors in each bucket
	vectors [Buckets][]int8 // each: count*Dim bytes
	labels  [Buckets][]byte // each: count bytes

	// Held when the backing is mmap'd so we can munmap on Close (rarely used).
	mmapData []byte
}

// Bucket maps a 14-dim query vector to its bucket index in [0, 32).
//
// 5 binary features chosen to spread the 3M-vector dataset roughly evenly
// while keeping queries near their own bucket's mass (features that are
// stable across "similar" transactions):
//
//	bit 0: last_transaction is null    (sentinel: v[5] == -1)
//	bit 1: terminal.is_online          (v[9])
//	bit 2: terminal.card_present       (v[10])
//	bit 3: unknown_merchant            (v[11])
//	bit 4: hour-of-day in PM half      (v[3] >= 0.5, hour > 11 UTC)
//	bit 5: weekend                     (v[4] > 0.7, day-of-week sat/sun)
//
// Time-based splits (hour, weekday) cleave the dominant in-person bucket
// without correlating tightly with the fraud label, so a query's true
// neighbors usually share these features too.
func Bucket(v *[Dim]float32) int {
	b := 0
	if v[5] < 0 { // -1 sentinel for no last_tx
		b |= 1
	}
	if v[9] >= 0.5 {
		b |= 2
	}
	if v[10] >= 0.5 {
		b |= 4
	}
	if v[11] >= 0.5 {
		b |= 8
	}
	if v[3] >= 0.5 {
		b |= 16
	}
	if v[4] >= 0.7 {
		b |= 32
	}
	return b
}

// f32ToI8 clamps to [-1, 1] then quantizes to int8 with scale 127.
//
// The sentinel value -1 (no last_tx) maps cleanly to -127, preserving
// "absence of last tx" as a far-out distance from any normalized value.
func f32ToI8(f float32) int8 {
	if f < -1 {
		f = -1
	}
	if f > 1 {
		f = 1
	}
	q := int32(f * 127.0)
	if f >= 0 {
		q = int32(f*127.0 + 0.5)
	} else {
		q = int32(f*127.0 - 0.5)
	}
	if q < -127 {
		q = -127
	}
	if q > 127 {
		q = 127
	}
	return int8(q)
}

// Search returns the K labels of the nearest neighbors within the query's bucket.
//
// Total cost is dominated by len(vectors[bucket])/Dim distance computations,
// each ~14 multiplies. For 3M vectors split 16-way, that's ~187k distance
// computations per query — memory-bandwidth bound, ~50-200µs on native amd64.
func (idx *Index) Search(v *[Dim]float32) [K]uint8 {
	bucket := Bucket(v)

	var q [Dim]int8
	for i := 0; i < Dim; i++ {
		q[i] = f32ToI8(v[i])
	}

	vecs := idx.vectors[bucket]
	labs := idx.labels[bucket]
	n := int(idx.counts[bucket])

	// Top-K via insertion-sort over a fixed slice. K=5 — tiny, branchless-ish.
	var topDist [K]int32
	var topLabel [K]uint8
	for i := 0; i < K; i++ {
		topDist[i] = 1 << 30
	}

	// Hoist the slice base pointer once so the inner loop has zero bounds
	// checks. Each element is exactly Dim=14 bytes; we step by 14.
	if n == 0 {
		return topLabel
	}
	base := unsafe.Pointer(&vecs[0])
	const stride = Dim
	for i := 0; i < n; i++ {
		p := (*[Dim]int8)(unsafe.Add(base, uintptr(i*stride)))
		// Unrolled L2² over Dim=14. Straight-line int math the compiler
		// can vectorize (SSE/AVX) on native amd64. Early-exit kernels
		// were tried and proved slower — branches were mispredicted often
		// enough to outweigh the saved multiplies.
		d0 := int32(q[0]) - int32(p[0])
		d1 := int32(q[1]) - int32(p[1])
		d2 := int32(q[2]) - int32(p[2])
		d3 := int32(q[3]) - int32(p[3])
		d4 := int32(q[4]) - int32(p[4])
		d5 := int32(q[5]) - int32(p[5])
		d6 := int32(q[6]) - int32(p[6])
		d7 := int32(q[7]) - int32(p[7])
		d8 := int32(q[8]) - int32(p[8])
		d9 := int32(q[9]) - int32(p[9])
		d10 := int32(q[10]) - int32(p[10])
		d11 := int32(q[11]) - int32(p[11])
		d12 := int32(q[12]) - int32(p[12])
		d13 := int32(q[13]) - int32(p[13])
		dist := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
			d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13

		// Skip if worse than the current worst in top-K.
		if dist >= topDist[K-1] {
			continue
		}
		// Insert into sorted top-K (K=5, so 5-step shift max).
		j := K - 1
		for j > 0 && topDist[j-1] > dist {
			topDist[j] = topDist[j-1]
			topLabel[j] = topLabel[j-1]
			j--
		}
		topDist[j] = dist
		topLabel[j] = labs[i]
	}

	return topLabel
}

// Store wraps Index to satisfy the store.VectorStore interface used by the api
// handler: Ready() bool + Search([14]float32) ([]uint8, error).
type Store struct{ Idx *Index }

func (s *Store) Ready() bool { return s.Idx != nil }

func (s *Store) Search(v [Dim]float32) ([]uint8, error) {
	top := s.Idx.Search(&v)
	out := make([]uint8, K)
	for i := 0; i < K; i++ {
		out[i] = top[i]
	}
	return out, nil
}

// Build constructs an in-memory index from per-bucket vector/label slices.
// Each vecs[b] must be counts[b]*Dim int8 values; labs[b] must be counts[b].
func Build(counts [Buckets]uint32, vecs [Buckets][]int8, labs [Buckets][]byte) *Index {
	idx := &Index{counts: counts, vectors: vecs, labels: labs}
	return idx
}

// Save writes the index to a single binary file. The file is mmap-friendly:
// header is small and pointed-to arrays are contiguous per bucket.
func (idx *Index) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	hdr := make([]byte, 4*4+4*Buckets)
	binary.LittleEndian.PutUint32(hdr[0:], magic)
	binary.LittleEndian.PutUint32(hdr[4:], uint32(Dim))
	binary.LittleEndian.PutUint32(hdr[8:], uint32(K))
	binary.LittleEndian.PutUint32(hdr[12:], uint32(Buckets))
	for b := 0; b < Buckets; b++ {
		binary.LittleEndian.PutUint32(hdr[16+4*b:], idx.counts[b])
	}
	if _, err := f.Write(hdr); err != nil {
		return err
	}

	for b := 0; b < Buckets; b++ {
		n := int(idx.counts[b])
		if n == 0 {
			continue
		}
		// int8 slice → bytes (same memory).
		vbytes := unsafeI8ToBytes(idx.vectors[b])
		if _, err := f.Write(vbytes); err != nil {
			return err
		}
		if _, err := f.Write(idx.labels[b]); err != nil {
			return err
		}
	}
	return nil
}

// Load reads the index into private memory (each process has its own copy).
func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	hdr := make([]byte, 4*4+4*Buckets)
	if _, err := io.ReadFull(f, hdr); err != nil {
		return nil, err
	}
	m := binary.LittleEndian.Uint32(hdr[0:])
	if m != magic {
		return nil, errBadMagic
	}
	if binary.LittleEndian.Uint32(hdr[4:]) != uint32(Dim) {
		return nil, errBadDim
	}
	nb := int(binary.LittleEndian.Uint32(hdr[12:]))
	if nb != Buckets {
		return nil, fmt.Errorf("ivf: expected %d buckets, got %d", Buckets, nb)
	}

	idx := &Index{}
	for b := 0; b < Buckets; b++ {
		idx.counts[b] = binary.LittleEndian.Uint32(hdr[16+4*b:])
	}
	for b := 0; b < Buckets; b++ {
		n := int(idx.counts[b])
		if n == 0 {
			continue
		}
		vbytes := make([]byte, n*Dim)
		if _, err := io.ReadFull(f, vbytes); err != nil {
			return nil, err
		}
		idx.vectors[b] = unsafeBytesToI8(vbytes)
		lbytes := make([]byte, n)
		if _, err := io.ReadFull(f, lbytes); err != nil {
			return nil, err
		}
		idx.labels[b] = lbytes
	}
	return idx, nil
}
