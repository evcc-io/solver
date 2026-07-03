//go:build goexperiment.simd && !arm64 && !amd64

package simplex

import "simd"

// axpy computes dst[k] += factor*src[k], vectorized via the portable simd
// package. arm64 and amd64 use simd/archsimd directly (axpy_archsimd_*.go).
func axpy(dst, src []float64, factor float64) {
	fv := simd.BroadcastFloat64s(factor)
	width := fv.Len()
	n := len(dst)
	k := 0
	for ; k+width <= n; k += width {
		d := simd.LoadFloat64s(dst[k : k+width])
		s := simd.LoadFloat64s(src[k : k+width])
		s.MulAdd(fv, d).Store(dst[k : k+width])
	}
	for ; k < n; k++ {
		dst[k] += factor * src[k]
	}
}

// scale computes dst[k] /= divisor for all k, vectorized via the portable
// simd package.
func scale(dst []float64, divisor float64) {
	dv := simd.BroadcastFloat64s(divisor)
	width := dv.Len()
	n := len(dst)
	k := 0
	for ; k+width <= n; k += width {
		simd.LoadFloat64s(dst[k : k+width]).Div(dv).Store(dst[k : k+width])
	}
	for ; k < n; k++ {
		dst[k] /= divisor
	}
}
