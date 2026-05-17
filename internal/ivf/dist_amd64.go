//go:build amd64

package ivf

// distI8x14 computes the squared L2 distance between two int8 vectors stored
// as 16-byte chunks. Only the first 14 bytes carry data; positions 14-15 are
// zero padding so the SSE2 implementation can load full 16-byte XMM registers
// without masking.
//
// Implementation in dist_amd64.s: ~6 SSE2 instructions per vector, no branches,
// no bounds checks. ~5x the throughput of the pure-Go unrolled version.
//
//go:noescape
func distI8x14(q, p *[16]int8) uint32
