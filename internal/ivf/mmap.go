//go:build linux || darwin

package ivf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

// LoadMmap maps the index file read-only. Each bucket's vector/label arrays
// point into the mmap region — no copying, no malloc for the heavy data. The
// kernel's page cache shares pages between processes that mmap the same inode,
// so two api containers reading the same image layer use one physical copy.
func LoadMmap(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := int(stat.Size())
	if size < 4*4+4*Buckets {
		return nil, fmt.Errorf("ivf: file too small (%d bytes)", size)
	}

	data, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_PRIVATE)
	if err != nil {
		return nil, err
	}
	// Bucket scans are linear over contiguous int8 vectors, so we want the
	// kernel to keep these pages resident — opposite of MADV_RANDOM. The
	// k6 ramp-up to 900 RPS only takes 120s, and the first request hitting
	// a cold bucket would otherwise pay disk + TLB cost.
	_ = unix.Madvise(data, unix.MADV_WILLNEED)
	// Touch one byte per page to force the page faults during boot instead
	// of on the hot path. 43 MB / 4 KB ≈ 10 700 iterations — well under 1 s.
	const pageSize = 4096
	var sink byte
	for i := 0; i < size; i += pageSize {
		sink ^= data[i]
	}
	_ = sink

	hdr := data[:4*4+4*Buckets]
	if binary.LittleEndian.Uint32(hdr[0:]) != magic {
		_ = unix.Munmap(data)
		return nil, errBadMagic
	}
	if binary.LittleEndian.Uint32(hdr[4:]) != uint32(Dim) {
		_ = unix.Munmap(data)
		return nil, errBadDim
	}
	if binary.LittleEndian.Uint32(hdr[12:]) != uint32(Buckets) {
		_ = unix.Munmap(data)
		return nil, errors.New("ivf: bucket count mismatch")
	}

	idx := &Index{mmapData: data}
	for b := 0; b < Buckets; b++ {
		idx.counts[b] = binary.LittleEndian.Uint32(hdr[16+4*b:])
	}

	off := len(hdr)
	for b := 0; b < Buckets; b++ {
		n := int(idx.counts[b])
		if n == 0 {
			continue
		}
		vbytes := data[off : off+n*stride : off+n*stride]
		idx.vectors[b] = unsafeBytesToI8(vbytes)
		off += n * stride
		idx.labels[b] = data[off : off+n : off+n]
		off += n
	}
	if off != size {
		_ = unix.Munmap(data)
		return nil, fmt.Errorf("ivf: trailing %d bytes after parsing", size-off)
	}

	return idx, nil
}
