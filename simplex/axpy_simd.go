//go:build goexperiment.simd

package simplex

import "simd"

// axpy computes dst[k] += factor*src[k] for all k, vectorized via the
// experimental simd package (Go 1.27+, GOEXPERIMENT=simd).
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
