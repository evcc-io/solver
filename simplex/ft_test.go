package simplex

import (
	"math"
	"math/rand"
	"testing"
)

// BenchmarkFTLU: factor + repeated ftran/btran at a representative kernel size,
// the per-solve cost FT would provide in the engine.
func BenchmarkFTLU(b *testing.B) {
	const m = 300
	rng := rand.New(rand.NewSource(7))
	a := make([]float64, m*m)
	for i := range a {
		if rng.Float64() < 0.03 {
			a[i] = rng.NormFloat64()
		}
	}
	for d := range m {
		a[d*m+d] += float64(m)
	}
	rhs := make([]float64, m)
	for i := range rhs {
		rhs[i] = rng.NormFloat64()
	}
	b.ResetTimer()
	for range b.N {
		f := newFTLU(m, a)
		v := append([]float64(nil), rhs...)
		f.ftran(v)
		f.btran(v)
	}
}

// matVec computes B*x for column-major B (a[col*m+row]).
func ftMatVec(m int, a, x []float64) []float64 {
	y := make([]float64, m)
	for c := range m {
		for r := range m {
			y[r] += a[c*m+r] * x[c]
		}
	}
	return y
}

func ftMatVecT(m int, a, x []float64) []float64 { // B^T * x
	y := make([]float64, m)
	for c := range m {
		s := 0.0
		for r := range m {
			s += a[c*m+r] * x[r]
		}
		y[c] = s
	}
	return y
}

// TestFTLUSolve checks ftran/btran against a known x on random nonsingular B.
func TestFTLUSolve(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 200; trial++ {
		m := 1 + rng.Intn(12)
		a := make([]float64, m*m)
		for i := range a {
			a[i] = rng.NormFloat64()
		}
		for d := range m { // diagonal boost keeps it well-conditioned
			a[d*m+d] += float64(m)
		}
		f := newFTLU(m, a)
		if f == nil {
			continue
		}
		// ftran: B x = b, pick x, form b, solve, compare
		x := make([]float64, m)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		b := ftMatVec(m, a, x)
		bb := append([]float64(nil), b...)
		f.ftran(bb)
		for i := range m {
			if math.Abs(bb[i]-x[i]) > 1e-7*(1+math.Abs(x[i])) {
				t.Fatalf("ftran trial %d m=%d col %d: got %g want %g", trial, m, i, bb[i], x[i])
			}
		}
		// btran: B^T y = c, pick y, form c, solve, compare
		y := make([]float64, m)
		for i := range y {
			y[i] = rng.NormFloat64()
		}
		c := ftMatVecT(m, a, y)
		cc := append([]float64(nil), c...)
		f.btran(cc)
		for i := range m {
			if math.Abs(cc[i]-y[i]) > 1e-7*(1+math.Abs(y[i])) {
				t.Fatalf("btran trial %d m=%d row %d: got %g want %g", trial, m, i, cc[i], y[i])
			}
		}
	}
}
