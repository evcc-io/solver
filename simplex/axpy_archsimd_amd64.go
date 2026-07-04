//go:build goexperiment.simd && amd64

package simplex

import "simd/archsimd"

// axpy computes dst[k] += factor*src[k], using SSE2 directly (no portable
// dispatch stub) via simd/archsimd's Float64x2 — part of the amd64 baseline.
func axpy(dst, src []float64, factor float64) {
	fv := archsimd.BroadcastFloat64x2(factor)
	n := len(dst)
	k := 0
	for ; k+2 <= n; k += 2 {
		d := archsimd.LoadFloat64x2(dst[k : k+2])
		s := archsimd.LoadFloat64x2(src[k : k+2])
		s.MulAdd(fv, d).Store(dst[k : k+2])
	}
	for ; k < n; k++ {
		dst[k] += factor * src[k]
	}
}

// scale computes dst[k] /= divisor for all k, using SSE2 directly.
func scale(dst []float64, divisor float64) {
	dv := archsimd.BroadcastFloat64x2(divisor)
	n := len(dst)
	k := 0
	for ; k+2 <= n; k += 2 {
		archsimd.LoadFloat64x2(dst[k : k+2]).Div(dv).Store(dst[k : k+2])
	}
	for ; k < n; k++ {
		dst[k] /= divisor
	}
}
