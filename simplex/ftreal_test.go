package simplex

import (
	"math/rand"
	"testing"
)

// BenchmarkFTReplaceReal: per-pivot cost at the engine's refactor cadence
// (rebuild every maxEtas), unlike BenchmarkFTReplace's unbounded rlist growth.
func BenchmarkFTReplaceReal(b *testing.B) {
	const m = 300
	rng := rand.New(rand.NewSource(9))
	a := make([]float64, m*m)
	for i := range a {
		if rng.Float64() < 0.03 {
			a[i] = rng.NormFloat64()
		}
	}
	for d := range m {
		a[d*m+d] += float64(m)
	}
	cr := make([][]int32, m)
	cv := make([][]float64, m)
	for c := range m {
		for r := range m {
			if v := a[c*m+r]; v != 0 {
				cr[c] = append(cr[c], int32(r))
				cv[c] = append(cv[c], v)
			}
		}
	}
	nv := make([]float64, m)
	for i := range nv {
		nv[i] = rng.NormFloat64()
	}
	rhs := make([]float64, m)
	for i := range rhs {
		rhs[i] = rng.NormFloat64()
	}
	f := newFTLU(m, a)
	v := make([]float64, m)
	n := 0
	for b.Loop() {
		if f.nUpd >= maxEtas {
			f.rebuild(cr, cv)
		}
		col := n % m
		nv[col] += float64(m)
		f.replaceColumn(col, nv)
		nv[col] -= float64(m)
		copy(v, rhs)
		f.ftran(v)
		n++
	}
}

// BenchmarkFTClone: per-B&B-child factor deep copy.
func BenchmarkFTClone(b *testing.B) {
	const m = 300
	rng := rand.New(rand.NewSource(9))
	a := make([]float64, m*m)
	for i := range a {
		if rng.Float64() < 0.03 {
			a[i] = rng.NormFloat64()
		}
	}
	for d := range m {
		a[d*m+d] += float64(m)
	}
	f := newFTLU(m, a)
	for b.Loop() {
		_ = f.clone()
	}
}
