package ivf

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"os"
)

// BuildFromRefs streams references.json.gz and partitions all 3M vectors
// into the 16 buckets. Returns an in-memory Index ready to Save.
//
// Implementation: two passes.
//   pass 1: count vectors per bucket so we can preallocate exact-sized arrays.
//   pass 2: actually fill the arrays.
//
// Two passes is faster overall than one-pass append because we avoid the
// repeated reallocation of 16 growing slices (each ending up ~187k vectors).
func BuildFromRefs(refsPath string) (*Index, error) {
	counts, err := countBuckets(refsPath)
	if err != nil {
		return nil, fmt.Errorf("count: %w", err)
	}

	var vecs [Buckets][]int8
	var labs [Buckets][]byte
	for b := 0; b < Buckets; b++ {
		vecs[b] = make([]int8, counts[b]*Dim)
		labs[b] = make([]byte, counts[b])
	}

	var cursors [Buckets]uint32
	err = streamRefs(refsPath, func(vec []float32, label byte) {
		var v [Dim]float32
		copy(v[:], vec)
		b := Bucket(&v)
		off := cursors[b] * Dim
		for i := 0; i < Dim; i++ {
			vecs[b][int(off)+i] = f32ToI8(v[i])
		}
		labs[b][cursors[b]] = label
		cursors[b]++
	})
	if err != nil {
		return nil, fmt.Errorf("fill: %w", err)
	}

	return Build(counts, vecs, labs), nil
}

func countBuckets(path string) ([Buckets]uint32, error) {
	var counts [Buckets]uint32
	err := streamRefs(path, func(vec []float32, _ byte) {
		var v [Dim]float32
		copy(v[:], vec)
		counts[Bucket(&v)]++
	})
	return counts, err
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
