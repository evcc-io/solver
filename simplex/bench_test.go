package simplex

import (
	"math/rand"
	"testing"

	"cbcgo/problem"
)

// randomLP builds an n-column, m-row LP with a handful of random
// coefficients per row — sparse like a real MPS instance.
func randomLP(n, m int) *problem.Problem {
	r := rand.New(rand.NewSource(1))
	p := problem.New()
	cols := make([]int, n)
	for j := range n {
		cols[j] = p.AddCol("", 0, 10, r.Float64()*10, false, nil, nil)
	}
	nnzPerRow := min(8, n)
	for i := range m {
		idx := r.Perm(n)[:nnzPerRow]
		coef := make([]float64, nnzPerRow)
		sum := 0.0
		for k := range coef {
			coef[k] = r.Float64()*4 + 1
			sum += coef[k] * 5
		}
		// alternate LE/GE so x=0 violates the GE rows, forcing phase 1
		// to actually pivot instead of stopping at the trivial origin.
		if i%2 == 0 {
			p.AddRow("", idx, coef, problem.GE, sum*0.6)
		} else {
			p.AddRow("", idx, coef, problem.LE, sum*1.4)
		}
	}
	return p
}

func benchColdSolve(b *testing.B, n, m int) {
	p := randomLP(n, m)
	b.ResetTimer()
	for range b.N {
		lp := Build(p)
		status, _, _ := lp.ColdSolve()
		if status != Optimal {
			b.Fatalf("status = %v, want Optimal", status)
		}
	}
}

func BenchmarkColdSolve_50x50(b *testing.B)   { benchColdSolve(b, 50, 50) }
func BenchmarkColdSolve_150x150(b *testing.B) { benchColdSolve(b, 150, 150) }
func BenchmarkColdSolve_300x300(b *testing.B) { benchColdSolve(b, 300, 300) }
