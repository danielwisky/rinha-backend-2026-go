// Package ivf is a k-means IVF (Inverted File) k-NN index with multi-probe
// search. Queries are quantized to int8, routed to the top-NProbe nearest
// centroids by L2 distance, and the K=5 nearest neighbors are picked across
// those clusters.
//
// Disk format ("ivf" binary, little-endian, v2):
//
//	uint32 magic    = 0x49564602 ("IVF\x02")
//	uint32 dim      = 14
//	uint32 k        = 5
//	uint32 nclusters = NClusters (2048)
//	int8   centroids[NClusters * stride]      stride=16, last 2 bytes per centroid zero
//	uint32 counts[NClusters]                  vectors per cluster
//	then per cluster: counts[c]*stride bytes of int8 vectors, then counts[c] bytes of labels.
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
	Dim    = 14
	K      = 5
	NProbe = 4

	stride = 16
)

var (
	magic = uint32(0x49564602)

	errBadMagic    = errors.New("ivf: bad magic")
	errBadDim      = errors.New("ivf: dim mismatch")
	errBadClusters = errors.New("ivf: cluster count mismatch")
)

type Index struct {
	counts    [NClusters]uint32
	centroids []int8 // NClusters * stride int8
	vectors   [NClusters][]int8
	labels    [NClusters][]byte

	mmapData []byte
}

// quantize14 packs a 14-dim float32 query into a stride-16 int8 array.
func quantize14(v *[Dim]float32) [stride]int8 {
	var q [stride]int8
	for i := 0; i < Dim; i++ {
		q[i] = f32ToI8(v[i])
	}
	return q
}

// NearestCluster returns the cluster index whose centroid is L2-closest to v.
func (idx *Index) NearestCluster(v *[Dim]float32) int {
	q := quantize14(v)
	return nearestClusterI8(&q, idx.centroids)
}

// Search returns the K labels of nearest neighbors across the top-NProbe
// nearest clusters (multi-probe IVF).
//
// Pass 1: compute distance to every centroid, keep top-NProbe via insertion-sort.
// Pass 2: scan those NProbe clusters; maintain a global top-K (insertion-sort).
//
// Centroid scan: NClusters * 1 dist = ~2048 distI8x14 calls over 32KB in L1.
// Cluster scans: NProbe * mean_cluster_size ≈ 8 * 1500 = ~12k dist calls.
func (idx *Index) Search(v *[Dim]float32) [K]uint8 {
	q := quantize14(v)

	var probeDist [NProbe]uint32
	var probeIdx [NProbe]int
	for i := 0; i < NProbe; i++ {
		probeDist[i] = 1 << 30
	}
	cbase := unsafe.Pointer(&idx.centroids[0])
	for c := 0; c < NClusters; c++ {
		p := (*[stride]int8)(unsafe.Add(cbase, uintptr(c*stride)))
		d := distI8x14(&q, p)
		if d >= probeDist[NProbe-1] {
			continue
		}
		j := NProbe - 1
		for j > 0 && probeDist[j-1] > d {
			probeDist[j] = probeDist[j-1]
			probeIdx[j] = probeIdx[j-1]
			j--
		}
		probeDist[j] = d
		probeIdx[j] = c
	}

	var topDist [K]uint32
	var topLabel [K]uint8
	for i := 0; i < K; i++ {
		topDist[i] = 1 << 30
	}

	for pi := 0; pi < NProbe; pi++ {
		c := probeIdx[pi]
		n := int(idx.counts[c])
		if n == 0 {
			continue
		}
		vecs := idx.vectors[c]
		labs := idx.labels[c]
		vbase := unsafe.Pointer(&vecs[0])
		for i := 0; i < n; i++ {
			p := (*[stride]int8)(unsafe.Add(vbase, uintptr(i*stride)))
			dist := distI8x14(&q, p)
			if dist >= topDist[K-1] {
				continue
			}
			j := K - 1
			for j > 0 && topDist[j-1] > dist {
				topDist[j] = topDist[j-1]
				topLabel[j] = topLabel[j-1]
				j--
			}
			topDist[j] = dist
			topLabel[j] = labs[i]
		}
	}

	return topLabel
}

// f32ToI8 clamps to [-1, 1] and quantizes with scale 127. The sentinel value
// -1 (no last_tx) maps cleanly to -127.
func f32ToI8(f float32) int8 {
	if f < -1 {
		f = -1
	}
	if f > 1 {
		f = 1
	}
	var q int32
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

func Build(centroids []int8, counts [NClusters]uint32, vecs [NClusters][]int8, labs [NClusters][]byte) *Index {
	return &Index{centroids: centroids, counts: counts, vectors: vecs, labels: labs}
}

const headerSize = 4*4 + NClusters*stride + 4*NClusters

func (idx *Index) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	hdr := make([]byte, headerSize)
	binary.LittleEndian.PutUint32(hdr[0:], magic)
	binary.LittleEndian.PutUint32(hdr[4:], uint32(Dim))
	binary.LittleEndian.PutUint32(hdr[8:], uint32(K))
	binary.LittleEndian.PutUint32(hdr[12:], uint32(NClusters))

	copy(hdr[16:16+NClusters*stride], unsafeI8ToBytes(idx.centroids))
	off := 16 + NClusters*stride
	for c := 0; c < NClusters; c++ {
		binary.LittleEndian.PutUint32(hdr[off+4*c:], idx.counts[c])
	}
	if _, err := f.Write(hdr); err != nil {
		return err
	}

	for c := 0; c < NClusters; c++ {
		n := int(idx.counts[c])
		if n == 0 {
			continue
		}
		if _, err := f.Write(unsafeI8ToBytes(idx.vectors[c])); err != nil {
			return err
		}
		if _, err := f.Write(idx.labels[c]); err != nil {
			return err
		}
	}
	return nil
}

func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	hdr := make([]byte, headerSize)
	if _, err := io.ReadFull(f, hdr); err != nil {
		return nil, err
	}
	if binary.LittleEndian.Uint32(hdr[0:]) != magic {
		return nil, errBadMagic
	}
	if binary.LittleEndian.Uint32(hdr[4:]) != uint32(Dim) {
		return nil, errBadDim
	}
	nc := int(binary.LittleEndian.Uint32(hdr[12:]))
	if nc != NClusters {
		return nil, fmt.Errorf("%w: expected %d, got %d", errBadClusters, NClusters, nc)
	}

	idx := &Index{}
	idx.centroids = make([]int8, NClusters*stride)
	copy(unsafeI8ToBytes(idx.centroids), hdr[16:16+NClusters*stride])
	off := 16 + NClusters*stride
	for c := 0; c < NClusters; c++ {
		idx.counts[c] = binary.LittleEndian.Uint32(hdr[off+4*c:])
	}

	for c := 0; c < NClusters; c++ {
		n := int(idx.counts[c])
		if n == 0 {
			continue
		}
		vbytes := make([]byte, n*stride)
		if _, err := io.ReadFull(f, vbytes); err != nil {
			return nil, err
		}
		idx.vectors[c] = unsafeBytesToI8(vbytes)
		lbytes := make([]byte, n)
		if _, err := io.ReadFull(f, lbytes); err != nil {
			return nil, err
		}
		idx.labels[c] = lbytes
	}
	return idx, nil
}
