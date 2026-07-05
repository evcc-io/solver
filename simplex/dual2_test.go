package simplex

import (
	"math"
	"math/rand"
	"testing"

	"cbcgo/problem"
)

func randomBoxedLP(rng *rand.Rand) *problem.Problem {
	p := problem.New()
	nv := 4 + rng.Intn(8)
	nr := 3 + rng.Intn(6)
	for j := 0; j < nv; j++ {
		ub := 1 + rng.Float64()*9
		p.AddCol("x", 0, ub, rng.Float64()*4-2, false, nil, nil)
	}
	for i := 0; i < nr; i++ {
		var idx []int
		var coef []float64
		for j := 0; j < nv; j++ {
			if rng.Float64() < 0.5 {
				idx = append(idx, j)
				coef = append(coef, rng.Float64()*4-2)
			}
		}
		if len(idx) == 0 {
			idx, coef = []int{0}, []float64{1}
		}
		sense := []problem.Sense{problem.LE, problem.GE, problem.EQ}[rng.Intn(3)]
		p.AddRow("r", idx, coef, sense, rng.Float64()*8-2)
	}
	return p
}

// TestDual2WarmResolve checks the Clp-style dual engine against both the
// legacy warm path and a cold reference solve after random bound changes.
func TestDual2WarmResolve(t *testing.T) {
	saved := dual2Enabled
	defer func() { dual2Enabled = saved }()

	rng := rand.New(rand.NewSource(7))
	tested := 0
	for trial := 0; trial < 800; trial++ {
		p := randomBoxedLP(rng)
		lp := Build(p)
		status, st, _ := lp.ColdSolve()
		if status != Optimal {
			continue
		}
		// random tightenings, like branch-and-bound bound fixings
		var touched []int
		for j := 0; j < lp.NumCols(); j++ {
			if rng.Float64() < 0.3 {
				lb, ub := lp.Bound(j)
				mid := lb + rng.Float64()*(ub-lb)
				if rng.Float64() < 0.5 {
					lp.SetBound(j, mid, ub)
				} else {
					lp.SetBound(j, lb, mid)
				}
				touched = append(touched, j)
			}
		}
		if len(touched) == 0 {
			continue
		}
		tested++

		dual2Enabled = false
		s1, _, obj1 := lp.WarmSolve(st.Clone(), touched)
		dual2Enabled = true
		s2, _, obj2 := lp.WarmSolve(st.Clone(), touched)

		if s1 != s2 {
			t.Fatalf("trial %d: legacy status %v, dual2 status %v", trial, s1, s2)
		}
		if s1 == Optimal && math.Abs(obj1-obj2) > 1e-6*(1+math.Abs(obj1)) {
			t.Fatalf("trial %d: legacy obj %v, dual2 obj %v", trial, obj1, obj2)
		}
	}
	if tested < 100 {
		t.Fatalf("only %d effective trials", tested)
	}
}
