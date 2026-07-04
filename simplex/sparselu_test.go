package simplex

import (
	"math"
	"math/rand"
	"testing"
)

func TestSparseLUSolve(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := 0; trial < 60; trial++ {
		k := 2 + rng.Intn(80)
		// diagonal-dominant-ish sparse kernel: guaranteed diagonal + extras
		colIdx := make([][]int32, k)
		colVal := make([][]float64, k)
		for c := range k {
			colIdx[c] = []int32{int32(c)}
			colVal[c] = []float64{2 + rng.Float64()}
			seen := map[int32]bool{int32(c): true}
			for extra := rng.Intn(4); extra > 0; extra-- {
				r := int32(rng.Intn(k))
				if !seen[r] {
					seen[r] = true
					colIdx[c] = append(colIdx[c], r)
					colVal[c] = append(colVal[c], rng.NormFloat64()*0.5)
				}
			}
		}
		lu := luFactorize(k, colIdx, colVal)
		if lu == nil {
			t.Fatalf("trial %d k=%d: unexpected singular", trial, k)
		}
		mul := func(x []float64) []float64 {
			v := make([]float64, k)
			for c := range k {
				for t2, r := range colIdx[c] {
					v[r] += colVal[c][t2] * x[c]
				}
			}
			return v
		}
		// K*x == v
		v := make([]float64, k)
		want := make([]float64, k)
		for i := range v {
			v[i] = rng.NormFloat64()
			want[i] = v[i]
		}
		lu.solve(v)
		back := mul(v)
		for i := range back {
			if math.Abs(back[i]-want[i]) > 1e-7 {
				t.Fatalf("trial %d k=%d solve: K*x[%d]=%g want %g", trial, k, i, back[i], want[i])
			}
		}
		// K^T*y == w
		w := make([]float64, k)
		wantW := make([]float64, k)
		for i := range w {
			w[i] = rng.NormFloat64()
			wantW[i] = w[i]
		}
		lu.solveT(w)
		for c := range k {
			var s float64
			for t2, r := range colIdx[c] {
				s += colVal[c][t2] * w[r]
			}
			if math.Abs(s-wantW[c]) > 1e-7 {
				t.Fatalf("trial %d k=%d solveT: (K^T y)[%d]=%g want %g", trial, k, c, s, wantW[c])
			}
		}
	}
}
