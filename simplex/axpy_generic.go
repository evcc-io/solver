//go:build !goexperiment.simd

package simplex

// axpy computes dst[k] += factor*src[k] for all k. Scalar fallback used by
// default builds (Go 1.27+ with GOEXPERIMENT=simd gets a vectorized version).
func axpy(dst, src []float64, factor float64) {
	for k := range dst {
		dst[k] += factor * src[k]
	}
}
