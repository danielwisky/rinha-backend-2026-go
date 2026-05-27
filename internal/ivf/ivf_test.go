package ivf

import (
	"math/rand"
	"testing"
	"unsafe"
)

// TestSearchMatchesBruteForce builds a small index from synthetic data and
// checks Search agrees with an exhaustive brute-force k-NN. Tolerates a small
// number of ties (different vectors at identical distance) but expects
// recall@K to be high.
//
// Sanity check only — runs on uniform-random vectors, so absolute recall
// numbers do not predict production. The real recall will be measured by k6.
func TestSearchMatchesBruteForce(t *testing.T) {
	const (
		n      = 20_000
		nq     = 200
		target = 0.80
	)

	rng := rand.New(rand.NewSource(7))
	vectors := make([]float32, n*Dim)
	labels := make([]byte, n)
	for i := range vectors {
		vectors[i] = rng.Float32()*2 - 1
	}
	for i := range labels {
		if rng.Float32() < 0.4 {
			labels[i] = 1
		}
	}

	centroids := Kmeans(vectors, 30, 3000, 11)
	assignments := make([]uint16, n)
	for i := 0; i < n; i++ {
		var v [Dim]float32
		copy(v[:], vectors[i*Dim:i*Dim+Dim])
		q := quantize14(&v)
		assignments[i] = uint16(nearestClusterI8(&q, centroids))
	}

	var counts [NClusters]uint32
	for _, c := range assignments {
		counts[c]++
	}
	var vecs [NClusters][]int8
	var labs [NClusters][]byte
	for c := 0; c < NClusters; c++ {
		vecs[c] = make([]int8, counts[c]*stride)
		labs[c] = make([]byte, counts[c])
	}
	var cursors [NClusters]uint32
	for i := 0; i < n; i++ {
		c := assignments[i]
		off := cursors[c] * stride
		for j := 0; j < Dim; j++ {
			vecs[c][int(off)+j] = f32ToI8(vectors[i*Dim+j])
		}
		labs[c][cursors[c]] = labels[i]
		cursors[c]++
	}
	idx := Build(centroids, counts, vecs, labs)

	// Brute-force ground truth in int8 space (same quantization Search uses)
	// so we measure recall of the multi-probe approximation, not quantization.
	allVecs := make([]int8, n*stride)
	allLabs := make([]byte, n)
	for i := 0; i < n; i++ {
		for j := 0; j < Dim; j++ {
			allVecs[i*stride+j] = f32ToI8(vectors[i*Dim+j])
		}
		allLabs[i] = labels[i]
	}

	var totalMatch, totalK int
	for qi := 0; qi < nq; qi++ {
		var qf [Dim]float32
		for j := 0; j < Dim; j++ {
			qf[j] = rng.Float32()*2 - 1
		}
		q := quantize14(&qf)

		var bfLabels [K]uint8
		var bfDist [K]uint32
		for i := 0; i < K; i++ {
			bfDist[i] = 1 << 30
		}
		base := unsafe.Pointer(&allVecs[0])
		for i := 0; i < n; i++ {
			p := (*[stride]int8)(unsafe.Add(base, uintptr(i*stride)))
			d := distI8x14(&q, p)
			if d >= bfDist[K-1] {
				continue
			}
			j := K - 1
			for j > 0 && bfDist[j-1] > d {
				bfDist[j] = bfDist[j-1]
				bfLabels[j] = bfLabels[j-1]
				j--
			}
			bfDist[j] = d
			bfLabels[j] = allLabs[i]
		}

		got := idx.Search(&qf)

		// Compare label multisets — Search and brute-force may pick different
		// neighbors at identical distance, but the label outcome (k-NN voting)
		// is what drives accuracy.
		bfBag := bagOfLabels(bfLabels[:])
		gotBag := bagOfLabels(got[:])
		for lbl, c := range gotBag {
			if c > bfBag[lbl] {
				totalMatch += bfBag[lbl]
			} else {
				totalMatch += c
			}
		}
		totalK += K
	}

	recall := float64(totalMatch) / float64(totalK)
	if recall < target {
		t.Fatalf("recall@%d = %.3f below target %.2f", K, recall, target)
	}
	t.Logf("recall@%d = %.3f over %d queries (target %.2f)", K, recall, nq, target)
}

func bagOfLabels(ls []uint8) map[uint8]int {
	m := make(map[uint8]int, 2)
	for _, l := range ls {
		m[l]++
	}
	return m
}
