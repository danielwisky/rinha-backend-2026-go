package ivf

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"
)

// BuildFromRefs streams references.json.gz, runs k-means with NClusters
// centroids on the float32 vectors, then assigns every vector to its nearest
// (quantized) centroid. Returns an in-memory Index ready to Save.
//
// Phases:
//  1. Stream all references into a row-major []float32 + []byte labels.
//  2. Mini-batch k-means → quantized int8 centroids (stride-16 layout).
//  3. Assign each vector to its nearest centroid in int8 space (parallel).
//  4. Allocate and fill per-cluster arrays.
//
// Memory: ~3M * 14 * 4 = 168MB temporary float32 buffer during build.
// This stage runs unconstrained in the buildivf Docker layer.
func BuildFromRefs(refsPath string) (*Index, error) {
	start := time.Now()
	vectors := make([]float32, 0, 3_500_000*Dim)
	labels := make([]byte, 0, 3_500_000)
	err := streamRefs(refsPath, func(vec []float32, label byte) {
		for i := 0; i < Dim; i++ {
			if i < len(vec) {
				vectors = append(vectors, vec[i])
			} else {
				vectors = append(vectors, 0)
			}
		}
		labels = append(labels, label)
	})
	if err != nil {
		return nil, fmt.Errorf("stream refs: %w", err)
	}
	n := len(labels)
	log.Printf("loaded %d reference vectors in %s", n, time.Since(start))

	// 300 iters × 20 000 batch ≈ 6 M points sampled, ~3× the original budget.
	// Build-time only — costs seconds in the ivf-builder Docker stage but
	// shrinks the tail of cluster sizes, which directly reduces worst-case
	// Search latency at runtime.
	start = time.Now()
	centroids := Kmeans(vectors, 300, 20000, 42)
	log.Printf("kmeans (k=%d) complete in %s", NClusters, time.Since(start))

	start = time.Now()
	assignments := make([]uint16, n)
	workers := runtime.NumCPU()
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for vi := lo; vi < hi; vi++ {
				var v [Dim]float32
				copy(v[:], vectors[vi*Dim:vi*Dim+Dim])
				q := quantize14(&v)
				assignments[vi] = uint16(nearestClusterI8(&q, centroids))
			}
		}(lo, hi)
	}
	wg.Wait()
	log.Printf("assignments (%d workers) complete in %s", workers, time.Since(start))

	var counts [NClusters]uint32
	for _, c := range assignments {
		counts[c]++
	}
	logClusterDistribution(counts[:])

	var vecs [NClusters][]int8
	var labs [NClusters][]byte
	for c := 0; c < NClusters; c++ {
		vecs[c] = make([]int8, counts[c]*stride)
		labs[c] = make([]byte, counts[c])
	}

	var cursors [NClusters]uint32
	for vi := 0; vi < n; vi++ {
		c := assignments[vi]
		off := cursors[c] * stride
		for i := 0; i < Dim; i++ {
			vecs[c][int(off)+i] = f32ToI8(vectors[vi*Dim+i])
		}
		labs[c][cursors[c]] = labels[vi]
		cursors[c]++
	}

	return Build(centroids, counts, vecs, labs), nil
}

// logClusterDistribution prints min / p50 / p99 / max / empty-count over the
// cluster sizes. A heavy-tail (p99 >> mean) suggests bumping iters or NClusters;
// many empties suggest reducing NClusters.
func logClusterDistribution(counts []uint32) {
	sorted := make([]uint32, len(counts))
	copy(sorted, counts)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	var total uint64
	for _, c := range sorted {
		total += uint64(c)
	}
	mean := total / uint64(len(sorted))

	var empty int
	for _, c := range sorted {
		if c == 0 {
			empty++
		} else {
			break
		}
	}
	p50 := sorted[len(sorted)/2]
	p99 := sorted[(len(sorted)*99)/100]
	max := sorted[len(sorted)-1]
	log.Printf("cluster sizes: min=%d empty=%d mean=%d p50=%d p99=%d max=%d (over %d clusters)",
		sorted[0], empty, mean, p50, p99, max, len(sorted))
}

// streamRefs decodes the gzipped JSON array of {"vector":[...], "label":"..."}
// objects and invokes onEntry for each.
func streamRefs(path string, onEntry func(vec []float32, label byte)) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	dec := json.NewDecoder(gz)
	if _, err := dec.Token(); err != nil {
		return fmt.Errorf("read opening token: %w", err)
	}

	var raw struct {
		Vector []float32 `json:"vector"`
		Label  string    `json:"label"`
	}
	for dec.More() {
		if err := dec.Decode(&raw); err != nil {
			return fmt.Errorf("decode entry: %w", err)
		}
		var lbl byte
		if raw.Label == "fraud" {
			lbl = 1
		}
		onEntry(raw.Vector, lbl)
	}
	return nil
}
