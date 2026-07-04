// Presolve: iterated column-bound tightening plus big-M coefficient
// tightening for binaries (CBC/CglProbing-style), before the LP is built.
package mip

import (
	"math"
	"time"

	"cbcgo/problem"
)

const presolvePasses = 10

// presolve tightens col bounds and binary big-M coefficients in place; it
// preserves the mixed-integer feasible set while tightening the relaxation.
func presolve(p *problem.Problem) {
	for range presolvePasses {
		if !presolvePass(p) {
			return
		}
	}
}

func presolvePass(p *problem.Problem) (changed bool) {
	inf := problem.Inf
	for ri := range p.Rows {
		r := &p.Rows[ri]
		rlb, rub := r.Bounds()
		if len(r.Idx) == 0 {
			continue
		}
		// contribution range of term k, mapping the finite problem.Inf
		// sentinel onto real infinities so the accounting can't absorb
		contrib := func(k int) (lo, hi float64) {
			a := r.Coef[k]
			c := &p.Cols[r.Idx[k]]
			lb, ub := c.LB, c.UB
			if lb <= -inf {
				lb = math.Inf(-1)
			}
			if ub >= inf {
				ub = math.Inf(1)
			}
			lo, hi = a*lb, a*ub
			if a < 0 {
				lo, hi = hi, lo
			}
			return lo, hi
		}
		// activity bounds with infinity counts; recomputed after any change
		var minSum, maxSum float64
		var minInf, maxInf int
		recompute := func() {
			minSum, maxSum, minInf, maxInf = 0, 0, 0, 0
			for k := range r.Idx {
				lo, hi := contrib(k)
				if math.IsInf(lo, -1) {
					minInf++
				} else {
					minSum += lo
				}
				if math.IsInf(hi, 1) {
					maxInf++
				} else {
					maxSum += hi
				}
			}
		}
		recompute()
		others := func(k int) (omin, omax float64) {
			lo, hi := contrib(k)
			omin, omax = math.Inf(-1), math.Inf(1)
			if minInf == 0 {
				omin = minSum - lo
			} else if minInf == 1 && math.IsInf(lo, -1) {
				omin = minSum
			}
			if maxInf == 0 {
				omax = maxSum - hi
			} else if maxInf == 1 && math.IsInf(hi, 1) {
				omax = maxSum
			}
			return omin, omax
		}

		for k, j := range r.Idx {
			a := r.Coef[k]
			if a == 0 {
				continue
			}
			c := &p.Cols[j]
			omin, omax := others(k)
			newLB, newUB := c.LB, c.UB
			// a*x <= rub - omin ; a*x >= rlb - omax
			if rub < inf && !math.IsInf(omin, -1) {
				if a > 0 {
					newUB = math.Min(newUB, (rub-omin)/a)
				} else {
					newLB = math.Max(newLB, (rub-omin)/a)
				}
			}
			if rlb > -inf && !math.IsInf(omax, 1) {
				if a > 0 {
					newLB = math.Max(newLB, (rlb-omax)/a)
				} else {
					newUB = math.Min(newUB, (rlb-omax)/a)
				}
			}
			if c.Integer {
				newLB = math.Ceil(newLB - 1e-9)
				newUB = math.Floor(newUB + 1e-9)
			}
			// only accept meaningful, non-crossing tightening
			if newLB > newUB {
				continue // genuinely infeasible: let the LP report it
			}
			if newLB > c.LB+1e-9 || newUB < c.UB-1e-9 {
				if newLB > c.LB {
					c.LB = newLB
				}
				if newUB < c.UB {
					c.UB = newUB
				}
				changed = true
				recompute()
			}
		}

		// coefficient tightening: binaries in pure <= rows
		if !(rub < inf && rlb == -inf) {
			continue
		}
		for k, j := range r.Idx {
			a := r.Coef[k]
			c := &p.Cols[j]
			if !c.Integer || c.LB != 0 || c.UB != 1 {
				continue
			}
			_, omax := others(k)
			if math.IsInf(omax, 1) {
				continue
			}
			switch {
			case a < 0 && omax > rub && omax+a < rub:
				// y=1 side is slack: shrink |a| so it just reaches
				setCoef(p, ri, k, rub-omax)
				changed = true
				recompute()
			case a > 0 && omax < rub && omax+a > rub:
				// y=0 side is slack: shift a and the rhs down together
				delta := rub - omax
				setCoef(p, ri, k, a-delta)
				r.RHS -= delta
				rub = r.RHS
				changed = true
				recompute()
			}
		}
	}
	return changed
}

// setCoef updates coefficient k of row ri in both the row and the mirrored
// column representation (the LP is built from the column side).
func setCoef(p *problem.Problem, ri, k int, v float64) {
	r := &p.Rows[ri]
	j := r.Idx[k]
	r.Coef[k] = v
	c := &p.Cols[j]
	for pos, rr := range c.Idx {
		if rr == ri {
			c.Coef[pos] = v
			return
		}
	}
}

// propagate tightens the working bound slices from row activity ranges,
// iterating to a fixpoint; reports false when some row proves infeasible.
func propagate(p *problem.Problem, lb, ub []float64) bool {
	inf := problem.Inf
	for pass := 0; pass < 4; pass++ {
		changed := false
		for ri := range p.Rows {
			r := &p.Rows[ri]
			rlb, rub := r.Bounds()
			var minSum, maxSum float64
			var minInf, maxInf int
			for k, j := range r.Idx {
				a := r.Coef[k]
				l, u := lb[j], ub[j]
				if l <= -inf {
					l = math.Inf(-1)
				}
				if u >= inf {
					u = math.Inf(1)
				}
				lo, hi := a*l, a*u
				if a < 0 {
					lo, hi = hi, lo
				}
				if math.IsInf(lo, -1) {
					minInf++
				} else {
					minSum += lo
				}
				if math.IsInf(hi, 1) {
					maxInf++
				} else {
					maxSum += hi
				}
			}
			// row-level infeasibility against the activity range
			if minInf == 0 && rub < inf && minSum > rub+1e-7 {
				return false
			}
			if maxInf == 0 && rlb > -inf && maxSum < rlb-1e-7 {
				return false
			}
			for k, j := range r.Idx {
				a := r.Coef[k]
				if a == 0 {
					continue
				}
				l, u := lb[j], ub[j]
				lf, uf := l, u
				if lf <= -inf {
					lf = math.Inf(-1)
				}
				if uf >= inf {
					uf = math.Inf(1)
				}
				lo, hi := a*lf, a*uf
				if a < 0 {
					lo, hi = hi, lo
				}
				omin, omax := math.Inf(-1), math.Inf(1)
				if minInf == 0 {
					omin = minSum - lo
				} else if minInf == 1 && math.IsInf(lo, -1) {
					omin = minSum
				}
				if maxInf == 0 {
					omax = maxSum - hi
				} else if maxInf == 1 && math.IsInf(hi, 1) {
					omax = maxSum
				}
				nl, nu := l, u
				if rub < inf && !math.IsInf(omin, -1) {
					if a > 0 {
						nu = math.Min(nu, (rub-omin)/a)
					} else {
						nl = math.Max(nl, (rub-omin)/a)
					}
				}
				if rlb > -inf && !math.IsInf(omax, 1) {
					if a > 0 {
						nl = math.Max(nl, (rlb-omax)/a)
					} else {
						nu = math.Min(nu, (rlb-omax)/a)
					}
				}
				if p.Cols[j].Integer {
					nl = math.Ceil(nl - 1e-9)
					nu = math.Floor(nu + 1e-9)
				}
				if nl > nu+1e-9 {
					return false
				}
				if nl > l+1e-9 || nu < u-1e-9 {
					lb[j], ub[j] = math.Max(l, nl), math.Min(u, nu)
					changed = true
				}
			}
		}
		if !changed {
			return true
		}
	}
	return true
}

// probe tentatively fixes each binary both ways (CglProbing-style): an
// infeasible side fixes the binary; otherwise merged implied bounds apply.
// Effort is time-boxed: partial results are valid tightenings.
func probe(p *problem.Problem, deadline time.Time) {
	n := len(p.Cols)
	base := make([]float64, 2*n)
	lb0, ub0 := make([]float64, n), make([]float64, n)
	lb1, ub1 := make([]float64, n), make([]float64, n)
	for j, c := range p.Cols {
		base[j], base[n+j] = c.LB, c.UB
	}
	for j, c := range p.Cols {
		if !c.Integer || c.LB != 0 || c.UB != 1 {
			continue
		}
		if !deadline.IsZero() && time.Now().After(deadline) {
			return
		}
		copy(lb0, base[:n])
		copy(ub0, base[n:])
		copy(lb1, base[:n])
		copy(ub1, base[n:])
		ub0[j] = 0
		lb1[j] = 1
		feas0 := propagate(p, lb0, ub0)
		feas1 := propagate(p, lb1, ub1)
		switch {
		case !feas0 && !feas1:
			return // infeasible problem: leave it to the LP to report
		case !feas0:
			p.Cols[j].LB = 1
			base[j] = 1
		case !feas1:
			p.Cols[j].UB = 0
			base[n+j] = 0
		default:
			// merged implied bounds are valid regardless of y_j
			for i := range n {
				if l := math.Min(lb0[i], lb1[i]); l > base[i]+1e-9 {
					p.Cols[i].LB = l
					base[i] = l
				}
				if u := math.Max(ub0[i], ub1[i]); u < base[n+i]-1e-9 {
					p.Cols[i].UB = u
					base[n+i] = u
				}
			}
		}
	}
}
