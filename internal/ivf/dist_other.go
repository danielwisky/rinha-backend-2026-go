//go:build !amd64

package ivf

// distI8x14 computes the squared L2 distance between two 14-dim int8 vectors
// padded to 16 bytes (positions 14-15 must be zero in both inputs).
//
// Pure-Go fallback for non-amd64 platforms (mainly so the project still
// builds on arm64 dev machines). amd64 uses SSE2 assembly in dist_amd64.s.
func distI8x14(q, p *[16]int8) uint32 {
	// Only the first 14 elements carry signal — the trailing two are zero
	// padding to match the SSE2 layout.
	var sum int32
	for i := 0; i < 14; i++ {
		d := int32(q[i]) - int32(p[i])
		sum += d * d
	}
	return uint32(sum)
}
