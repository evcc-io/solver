//go:build goexperiment.simd && !arm64

package simplex

import "simd"

// axpy computes dst[k] += factor*src[k], vectorized via the portable simd
// package. arm64 uses simd/archsimd directly (axpy_archsimd_arm64.go).
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
