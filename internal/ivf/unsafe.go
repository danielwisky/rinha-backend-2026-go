package ivf

import "unsafe"

// unsafeI8ToBytes reinterprets a []int8 as []byte without copying. Safe because
// int8 and byte have identical layout (1 byte, no padding).
func unsafeI8ToBytes(s []int8) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s))
}

// unsafeBytesToI8 is the inverse — reinterprets []byte as []int8.
func unsafeBytesToI8(s []byte) []int8 {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*int8)(unsafe.Pointer(&s[0])), len(s))
}
