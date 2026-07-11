package mip

import (
	"math"
	"math/rand"
	"testing"

	"cbcgo/problem"
)

// TestCglCutsSound: every Cgl cut must hold at every 0/1-feasible point of the
// original problem (soundness — never cut an integer-feasible solution).
func TestCglCutsSound(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for trial := 0; trial < 300; trial++ {
		n := 3 + rng.Intn(5)
		p := problem.New()
		for j := 0; j < n; j++ {
			p.AddCol("", 0, 1, rng.NormFloat64(), true, nil, nil)
		}
		nr := 1 + rng.Intn(3)
		for r := 0; r < nr; r++ {
			var idx []int
			var coef []float64
			for j := 0; j < n; j++ {
				if rng.Float64() < 0.7 {
					idx = append(idx, j)
					coef = append(coef, float64(rng.Intn(7)-3))
				}
			}
			if len(idx) == 0 {
				continue
			}
			rhs := float64(rng.Intn(9) - 2)
			p.AddRow("", idx, coef, problem.LE, rhs)
		}
		origRows := len(p.Rows)
		m := &Model{P: p}
		// LP point: use a random fractional x to drive generation
		x := make([]float64, n)
		for j := range x {
			x[j] = rng.Float64()
		}
		m.knapsackCoverCuts(x)
		m.cliqueCuts(x)
		m.zeroHalfCuts(x)
		m.liftProjectCuts(x)

		// enumerate all 0/1 points feasible for the original rows; each must
		// satisfy every generated cut
		for mask := 0; mask < (1 << n); mask++ {
			pt := make([]float64, n)
			for j := 0; j < n; j++ {
				if mask&(1<<j) != 0 {
					pt[j] = 1
				}
			}
			if !feasible(p.Rows[:origRows], pt) {
				continue
			}
			for ri := origRows; ri < len(p.Rows); ri++ {
				if !satisfies(&p.Rows[ri], pt) {
					t.Fatalf("trial %d: cut %d cuts off feasible point %v", trial, ri, pt)
				}
			}
		}
	}
}

func feasible(rows []problem.Row, x []float64) bool {
	for i := range rows {
		if !satisfies(&rows[i], x) {
			return false
		}
	}
	return true
}

func satisfies(r *problem.Row, x []float64) bool {
	lo, hi := r.Bounds()
	s := 0.0
	for k, j := range r.Idx {
		s += r.Coef[k] * x[j]
	}
	return s <= hi+1e-6 && s >= lo-1e-6
}

var _ = math.Abs
