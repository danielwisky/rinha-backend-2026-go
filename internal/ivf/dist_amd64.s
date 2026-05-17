// SSE2 squared-L2 distance for two 14-dim int8 vectors padded to 16 bytes.
//
// Inputs:
//   q+0(FP):  *[16]int8  — query, last 2 bytes zero
//   p+8(FP):  *[16]int8  — candidate, last 2 bytes zero
// Output:
//   ret+16(FP): uint32   — sum of (q[i] - p[i])^2 over the 14 real dims
//
// Strategy: sign-extend int8 → int16 on both halves, subtract, square via
// PMULLW (low 16 bits — fits because |d| ≤ 254, d² ≤ 64516 < 2^16), then
// promote int16 squares to int32 and accumulate. PMULLW + PUNPCK / PADDD
// rather than the more direct PMADDWD because some Go asm vintages don't
// know PMADDWD with XMM operands.

#include "textflag.h"

// func distI8x14(q, p *[16]int8) uint32
TEXT ·distI8x14(SB), NOSPLIT, $0-20
    MOVQ q+0(FP), AX
    MOVQ p+8(FP), BX

    // Load 16 bytes from each into XMM regs (unaligned).
    MOVOU (AX), X0
    MOVOU (BX), X1

    // Low 8 bytes → 8x int16, subtract, square (int16 result is exact).
    PMOVSXBW X0, X2
    PMOVSXBW X1, X3
    PSUBW X3, X2
    PMULLW X2, X2          // X2 = (q-p)^2 as 8 int16 values

    // High 8 bytes — shift the high halves down to the low half.
    PSRLDQ $8, X0
    PSRLDQ $8, X1
    PMOVSXBW X0, X4
    PMOVSXBW X1, X5
    PSUBW X5, X4
    PMULLW X4, X4          // X4 = (q-p)^2 as 8 int16 values

    // Promote each set of int16 squares to int32 lanes and sum.
    // PMOVZXWD: zero-extend low 4 int16s → 4 int32s.
    // Then PSRLDQ + PMOVZXWD for the high 4.
    PMOVZXWD X2, X5        // X5 = low 4 of X2 as int32
    PSRLDQ $8, X2
    PMOVZXWD X2, X6        // X6 = high 4 of X2 (now low) as int32
    PADDD X6, X5

    PMOVZXWD X4, X6
    PSRLDQ $8, X4
    PMOVZXWD X4, X7
    PADDD X7, X6

    PADDD X6, X5           // X5 = 4 int32 partial sums

    // Horizontal sum the 4 int32 lanes.
    PSHUFD $0x0E, X5, X6
    PADDD X6, X5
    PSHUFD $0x01, X5, X6
    PADDD X6, X5

    MOVD X5, AX
    MOVL AX, ret+16(FP)
    RET
