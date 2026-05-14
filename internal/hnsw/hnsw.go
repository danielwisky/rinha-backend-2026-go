package hnsw

// #cgo CXXFLAGS: -std=c++17 -O3 -march=native
// #cgo LDFLAGS: -lstdc++ -lm
// #include "hnswlib_wrapper.h"
// #include <stdlib.h>
import "C"
import "unsafe"

// Init initializes the HNSW index.
func Init(dim, maxElements, M, efConstruction int) {
	C.hnsw_init(C.int(dim), C.int(maxElements), C.int(M), C.int(efConstruction))
}

// Add inserts a vector with a label (0=legit, 1=fraud) at the given id.
func Add(vec []float32, label, id int) {
	C.hnsw_add((*C.float)(unsafe.Pointer(&vec[0])), C.int(label), C.int(id))
}

// SetEf sets the ef parameter for search (higher = better recall, slower).
func SetEf(ef int) {
	C.hnsw_set_ef(C.int(ef))
}

// Search returns the labels (0/1) of the k nearest neighbors.
func Search(query []float32, k int) []uint8 {
	labels := make([]uint8, k)
	n := int(C.hnsw_search(
		(*C.float)(unsafe.Pointer(&query[0])),
		C.int(k),
		(*C.uchar)(unsafe.Pointer(&labels[0])),
	))
	return labels[:n]
}
