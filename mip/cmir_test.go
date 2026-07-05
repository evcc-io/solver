package mip

import (
	"math/rand"
	"testing"

	"cbcgo/problem"
)

// buildChainMIP constructs a small battery-like chain: s_t = s_{t-1} + c_t
// - d_t with VUB gating c_t <= C(1-z_t), d_t <= D z_t — the structure
// cmirCuts targets.
func buildChainMIP(rng *rand.Rand, T int) *problem.Problem {
	p := problem.New()
	p.ObjSense = 1 // minimize
	C, D, S := 2.0+rng.Float64()*3, 1.0+rng.Float64()*2, 3.0+rng.Float64()*4
	s0 := rng.Float64() * S
	s := make([]int, T)
	c := make([]int, T)
	d := make([]int, T)
	z := make([]int, T)
	for t := range T {
		s[t] = p.AddCol("", 0, S, rng.NormFloat64(), false, nil, nil)
		c[t] = p.AddCol("", 0, C, rng.NormFloat64(), false, nil, nil)
		d[t] = p.AddCol("", 0, D, rng.NormFloat64(), false, nil, nil)
		z[t] = p.AddCol("", 0, 1, rng.NormFloat64(), true, nil, nil)
	}
	for t := range T {
		if t == 0 {
			p.AddRow("", []int{s[0], c[0], d[0]}, []float64{1, -1, 1}, problem.EQ, s0)
		} else {
			p.AddRow("", []int{s[t], s[t-1], c[t], d[t]}, []float64{1, -1, -1, 1}, problem.EQ, 0)
		}
		p.AddRow("", []int{c[t], z[t]}, []float64{1, C}, problem.LE, C)
		p.AddRow("", []int{d[t], z[t]}, []float64{1, -D}, problem.LE, 0)
	}
	return p
}

func copyProblem(p *problem.Problem) *problem.Problem {
	q := problem.New()
	q.ObjSense = p.ObjSense
	for _, c := range p.Cols {
		q.AddCol(c.Name, c.LB, c.UB, c.Obj, c.Integer, nil, nil)
	}
	for ri := range p.Rows {
		r := &p.Rows[ri]
		rl, ru := r.Bounds()
		rhs := ru
		if r.Sense == problem.GE {
			rhs = rl
		}
		q.AddRow(r.Name, append([]int(nil), r.Idx...), append([]float64(nil), r.Coef...), r.Sense, rhs)
	}
	return q
}

// TestCMIRSoundness: every integer-feasible point of random chain MIPs must
// satisfy every generated c-MIR cut.
func TestCMIRSoundness(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := range 80 {
		T := 2 + rng.Intn(4)
		p := buildChainMIP(rng, T)
		var zCols []int
		for j := range p.Cols {
			if p.Cols[j].Integer {
				zCols = append(zCols, j)
			}
		}
		// collect LP-optimal vertices of every feasible z assignment
		var feas [][]float64
		for mask := 0; mask < 1<<len(zCols); mask++ {
			q := copyProblem(p)
			for k, j := range zCols {
				v := float64((mask >> k) & 1)
				q.Cols[j].LB, q.Cols[j].UB = v, v
			}
			if res := SolveRelaxation(q); res.Status == Optimal {
				feas = append(feas, res.X)
			}
		}
		if len(feas) == 0 {
			continue
		}
		// generate cuts at the root LP point
		work := copyProblem(p)
		m := &Model{P: work}
		rel := SolveRelaxation(work)
		if rel.Status != Optimal {
			continue
		}
		orig := len(work.Rows)
		m.cmirCuts(rel.X)
		for ri := orig; ri < len(work.Rows); ri++ {
			r := &work.Rows[ri]
			_, ru := r.Bounds()
			for fi, x := range feas {
				act := 0.0
				for k, j := range r.Idx {
					act += r.Coef[k] * x[j]
				}
				if act > ru+1e-6 {
					t.Fatalf("trial %d: cut row %d (idx %v coef %v rhs %g) violated by feasible point %d (act %g)",
						trial, ri, r.Idx, r.Coef, ru, fi, act)
				}
			}
		}
	}
}
