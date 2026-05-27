package ivf

import (
	"math/rand"
	"unsafe"
)

// NClusters is the k for the IVF k-means partition.
//
// 2048 matches the top public submissions (e.g. RonieNeubauer/rinha2026).
// At 3M reference vectors, mean cluster size is ~1500 — one cluster's
// scan (~24KB int8 data) fits comfortably in L1, and the centroid table
// itself (NClusters * stride = 32KB) lives in L1 too.
const NClusters = 2048

// Kmeans runs mini-batch k-means on float32 reference vectors and returns
// NClusters quantized int8 centroids in stride-16 layout.
//
// Algorithm: random init (uniform sample without replacement), then
// `iters` mini-batch updates (Sculley, WWW'10). For each sampled point
// we find its nearest centroid and pull that centroid toward the point
// by 1/count_c — the per-cluster learning rate that's been shown to
// converge to near-Lloyd quality at a fraction of the cost.
//
// vectors is row-major [N*Dim]float32. Caller owns the memory.
func Kmeans(vectors []float32, iters, batchSize int, seed int64) []int8 {
	n := len(vectors) / Dim
	rng := rand.New(rand.NewSource(seed))

	centroids := randomInit(vectors, n, NClusters, rng)
	counts := make([]int, NClusters)

	for it := 0; it < iters; it++ {
		for b := 0; b < batchSize; b++ {
			idx := rng.Intn(n)
			v := vectors[idx*Dim : idx*Dim+Dim]
			c := nearestCentroidF32(v, centroids)
			counts[c]++
			eta := 1.0 / float32(counts[c])
			base := c * Dim
			for i := 0; i < Dim; i++ {
				centroids[base+i] += eta * (v[i] - centroids[base+i])
			}
		}
	}

	// Quantize to int8 stride-16 layout (trailing 2 bytes per centroid stay
	// zero to match the SSE2 distance kernel's expected padding).
	out := make([]int8, NClusters*stride)
	for c := 0; c < NClusters; c++ {
		for i := 0; i < Dim; i++ {
			out[c*stride+i] = f32ToI8(centroids[c*Dim+i])
		}
	}
	return out
}

// randomInit picks NClusters distinct vectors as initial centroids.
// k-means++ would be O(k²·n) ≈ 12 trillion ops for k=2048 n=3M — infeasible.
// Random init plus 30+ mini-batch iterations gets within ~5% of k-means++ quality
// at this dataset size.
func randomInit(vectors []float32, n, k int, rng *rand.Rand) []float32 {
	centroids := make([]float32, k*Dim)
	seen := make(map[int]struct{}, k)
	picked := 0
	for picked < k {
		idx := rng.Intn(n)
		if _, ok := seen[idx]; ok {
			continue
		}
		seen[idx] = struct{}{}
		copy(centroids[picked*Dim:picked*Dim+Dim], vectors[idx*Dim:idx*Dim+Dim])
		picked++
	}
	return centroids
}

// nearestCentroidF32 returns the index of the centroid in [NClusters][Dim]
// closest to v under squared L2.
func nearestCentroidF32(v, centroids []float32) int {
	best := 0
	var bestD float32 = 1e30
	for c := 0; c < NClusters; c++ {
		base := c * Dim
		var d float32 = 0
		for i := 0; i < Dim; i++ {
			x := v[i] - centroids[base+i]
			d += x * x
			if d >= bestD {
				break
			}
		}
		if d < bestD {
			bestD = d
			best = c
		}
	}
	return best
}

// nearestClusterI8 returns the centroid index closest to q under SSE2 distI8x14.
// Used during the assignment pass after k-means converges.
func nearestClusterI8(q *[stride]int8, centroids []int8) int {
	best := 0
	var bestD uint32 = 1 << 30
	base := unsafe.Pointer(&centroids[0])
	for c := 0; c < NClusters; c++ {
		p := (*[stride]int8)(unsafe.Add(base, uintptr(c*stride)))
		d := distI8x14(q, p)
		if d < bestD {
			bestD = d
			best = c
		}
	}
	return best
}
