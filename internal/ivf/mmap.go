//go:build linux || darwin

package ivf

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"

	"golang.org/x/sys/unix"
)

// LoadMmap maps the v2 index file read-only. Centroids and per-cluster
// vector/label arrays point into the mmap region — no copying. The kernel
// page cache shares pages between processes mmap'ing the same inode, so
// two api containers reading the same image layer use one physical copy.
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
	if size < headerSize {
		return nil, fmt.Errorf("ivf: file too small (%d bytes, need %d)", size, headerSize)
	}

	data, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_PRIVATE)
	if err != nil {
		return nil, err
	}
	// Linear scans over centroids and clusters benefit from MADV_WILLNEED.
	_ = unix.Madvise(data, unix.MADV_WILLNEED)
	// Touch one byte per page to force the page faults during boot rather
	// than on the hot path. Index is ~50MB / 4KB ≈ 12k iterations, sub-second.
	const pageSize = 4096
	var sink byte
	for i := 0; i < size; i += pageSize {
		sink ^= data[i]
	}
	_ = sink

	// Pin the pages so the kernel never evicts them under RSS pressure. Warning-
	// only on failure — typically requires RLIMIT_MEMLOCK to be raised (see
	// `ulimits: memlock: -1` in docker-compose.yml). Without mlock the
	// MADV_WILLNEED + page-touch above is best-effort.
	if err := unix.Mlock(data); err != nil {
		log.Printf("ivf: mlock failed (continuing without it): %v", err)
	}

	if binary.LittleEndian.Uint32(data[0:]) != magic {
		_ = unix.Munmap(data)
		return nil, errBadMagic
	}
	if binary.LittleEndian.Uint32(data[4:]) != uint32(Dim) {
		_ = unix.Munmap(data)
		return nil, errBadDim
	}
	nc := int(binary.LittleEndian.Uint32(data[12:]))
	if nc != NClusters {
		_ = unix.Munmap(data)
		return nil, fmt.Errorf("%w: expected %d, got %d", errBadClusters, NClusters, nc)
	}

	idx := &Index{mmapData: data}
	// Centroids point directly into mmap.
	idx.centroids = unsafeBytesToI8(data[16 : 16+NClusters*stride : 16+NClusters*stride])
	countsOff := 16 + NClusters*stride
	for c := 0; c < NClusters; c++ {
		idx.counts[c] = binary.LittleEndian.Uint32(data[countsOff+4*c:])
	}

	off := headerSize
	for c := 0; c < NClusters; c++ {
		n := int(idx.counts[c])
		if n == 0 {
			continue
		}
		vbytes := data[off : off+n*stride : off+n*stride]
		idx.vectors[c] = unsafeBytesToI8(vbytes)
		off += n * stride
		idx.labels[c] = data[off : off+n : off+n]
		off += n
	}
	if off != size {
		_ = unix.Munmap(data)
		return nil, fmt.Errorf("ivf: trailing %d bytes after parsing", size-off)
	}

	return idx, nil
}
