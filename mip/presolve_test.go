package mip

import (
	"math/rand"
	"testing"
	"time"

	"cbcgo/problem"
)

// x in [0,4], y in [-1,1], z >= 0 integer; c2+c3 imply z in [7,8],
// then x >= 2 and y in [-0.5, 0.5]. The mixed-integer set must survive.
func TestPresolveBoundTightening(t *testing.T) {
	p := problem.New()
	x := p.AddCol("x", 0, 4, 1, false, nil, nil)
	y := p.AddCol("y", -1, 1, 4, false, nil, nil)
	z := p.AddCol("z", 0, problem.Inf, 9, true, nil, nil)
	p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5)
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7.5)
	presolve(p)
	want := [][2]float64{{2, 4}, {-0.5, 0.5}, {7, 8}}
	for j, w := range want {
		if c := p.Cols[j]; c.LB != w[0] || c.UB != w[1] {
			t.Errorf("col %s: [%v,%v], want [%v,%v]", c.Name, c.LB, c.UB, w[0], w[1])
		}
	}
}

// big-M row e - M*y <= 0 with e <= 3 implied: M must shrink to 3.
func TestPresolveCoefficientTightening(t *testing.T) {
	p := problem.New()
	e := p.AddCol("e", 0, 3, 1, false, nil, nil)
	y := p.AddCol("y", 0, 1, 0, true, nil, nil)
	p.AddRow("bigm", []int{e, y}, []float64{1, -1000}, problem.LE, 0)
	presolve(p)
	if got := p.Rows[0].Coef[1]; got != -3 {
		t.Errorf("big-M coef = %v, want -3", got)
	}
	if got := p.Cols[e].Coef[0]; got != 1 {
		t.Errorf("column-side coef for e = %v, want 1", got)
	}
	if got := p.Cols[y].Coef[0]; got != -3 {
		t.Errorf("column-side coef for y = %v, want -3", got)
	}
}

// TestProbeStrengthenSound: probing coefficient strengthening must preserve
// every integer-feasible point of random binary big-M MIPs.
func TestProbeStrengthenSound(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for trial := 0; trial < 300; trial++ {
		p := problem.New()
		nb := 2 + rng.Intn(4) // binaries
		nc := 1 + rng.Intn(3) // bounded continuous
		for i := 0; i < nb; i++ {
			p.AddCol("b", 0, 1, rng.NormFloat64(), true, nil, nil)
		}
		for i := 0; i < nc; i++ {
			p.AddCol("x", 0, 1+4*rng.Float64(), rng.NormFloat64(), false, nil, nil)
		}
		n := nb + nc
		nr := 2 + rng.Intn(5)
		for i := 0; i < nr; i++ {
			var idx []int
			var coef []float64
			for j := 0; j < n; j++ {
				if rng.Float64() < 0.6 {
					idx = append(idx, j)
					c := rng.NormFloat64() * 3
					if j < nb && rng.Float64() < 0.5 {
						c *= 100 // big-M-ish binary coefficient
					}
					coef = append(coef, c)
				}
			}
			if len(idx) == 0 {
				continue
			}
			kind := problem.LE
			if rng.Float64() < 0.5 {
				kind = problem.GE
			}
			p.AddRow("r", idx, coef, kind, rng.NormFloat64()*5)
		}
		// brute force: per binary assignment, the continuous box corners are
		// NOT sufficient for feasibility in general — instead compare row-wise
		// feasibility of many sampled mixed points before vs after probing.
		type pt []float64
		var pts []pt
		for mask := 0; mask < 1<<nb; mask++ {
			for s := 0; s < 8; s++ {
				x := make(pt, n)
				for j := 0; j < nb; j++ {
					x[j] = float64((mask >> j) & 1)
				}
				for j := nb; j < n; j++ {
					x[j] = p.Cols[j].UB * rng.Float64()
				}
				pts = append(pts, x)
			}
		}
		feas := func(pr *problem.Problem, x pt) bool {
			for j, c := range pr.Cols {
				if x[j] < c.LB-1e-6 || x[j] > c.UB+1e-6 {
					return false
				}
			}
			for ri := range pr.Rows {
				r := &pr.Rows[ri]
				rlb, rub := r.Bounds()
				act := 0.0
				for k, j := range r.Idx {
					act += r.Coef[k] * x[j]
				}
				if rub < problem.Inf && act > rub+1e-6 {
					return false
				}
				if rlb > -problem.Inf && act < rlb-1e-6 {
					return false
				}
			}
			return true
		}
		before := make([]bool, len(pts))
		for i, x := range pts {
			before[i] = feas(p, x)
		}
		probe(p, time.Time{})
		for i, x := range pts {
			if before[i] && !feas(p, x) {
				t.Fatalf("trial %d: integer-feasible point %v cut off by probing", trial, x)
			}
		}
	}
}
