package mip

import (
	"math"
	"math/rand"
	"testing"

	"cbcgo/problem"
)

// genMultiKnapsack builds a deterministic multi-dimensional knapsack big
// enough to force a real branch-and-bound tree.
func genMultiKnapsack(nCols, nRows int, seed int64) *problem.Problem {
	rng := rand.New(rand.NewSource(seed))
	p := problem.New()
	p.ObjSense = -1
	cols := make([]int, nCols)
	for j := range cols {
		cols[j] = p.AddCol("x", 0, 1, 1+float64(rng.Intn(40)), true, nil, nil)
	}
	for range nRows {
		w := make([]float64, nCols)
		total := 0.0
		for j := range w {
			w[j] = 1 + float64(rng.Intn(20))
			total += w[j]
		}
		p.AddRow("cap", cols, w, problem.LE, math.Floor(total/3))
	}
	return p
}

// TestThreadedParity: a threaded solve must prove the same optimum as the
// serial solve (node order differs; the answer must not).
func TestThreadedParity(t *testing.T) {
	for _, seed := range []int64{1, 7, 42} {
		serial := New(genMultiKnapsack(40, 12, seed)).Solve()
		if serial.Status != Optimal {
			t.Fatalf("seed %d: serial status = %v, want Optimal", seed, serial.Status)
		}

		m := New(genMultiKnapsack(40, 12, seed))
		m.Threads = 4
		par := m.Solve()
		if par.Status != Optimal {
			t.Fatalf("seed %d: threaded status = %v, want Optimal", seed, par.Status)
		}
		if !almost(par.Obj, serial.Obj) {
			t.Fatalf("seed %d: threaded obj = %v, serial obj = %v", seed, par.Obj, serial.Obj)
		}
		for _, v := range par.X {
			if math.Abs(v-math.Round(v)) > 1e-6 {
				t.Fatalf("seed %d: non-integer threaded solution", seed)
			}
		}
	}
}
