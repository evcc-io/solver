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

		// coefficient tightening: binaries in pure one-sided big-M rows. The
		// >= mirror is CBC-ungated under D1, else gated to large models.
		pureLE := rub < inf && rlb == -inf
		pureGE := rlb > -inf && rub == inf
		if !pureLE && !pureGE {
			continue
		}
		for k, j := range r.Idx {
			a := r.Coef[k]
			c := &p.Cols[j]
			if !c.Integer || c.LB != 0 || c.UB != 1 {
				continue
			}
			omin, omax := others(k)
			if pureLE {
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
			} else { // pureGE: mirror of pureLE under negation
				if math.IsInf(omin, -1) {
					continue
				}
				switch {
				case a > 0 && omin < rlb && omin+a > rlb:
					setCoef(p, ri, k, rlb-omin)
					changed = true
					recompute()
				case a < 0 && omin > rlb && omin+a < rlb:
					delta := omin - rlb
					setCoef(p, ri, k, a+delta)
					r.RHS += delta
					rlb = r.RHS
					changed = true
					recompute()
				}
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

// propagate tightens the working bounds via a row worklist run to fixpoint
// (order-independent: bounds only shrink); false when a row is infeasible.
// propScratch pools propagate's membership flags + FIFO ring so the per-node
// propagatedChild path allocates nothing; nil scratch allocates locally.
type propScratch struct {
	inQ  []bool
	ring []int
}

func propagate(p *problem.Problem, lb, ub []float64, sc *propScratch) bool {
	inf := problem.Inf
	nr := len(p.Rows)
	var inQ []bool
	var ring []int
	if sc != nil {
		if cap(sc.inQ) < nr {
			sc.inQ, sc.ring = make([]bool, nr), make([]int, nr)
		}
		inQ, ring = sc.inQ[:nr], sc.ring[:nr]
		for i := range inQ {
			inQ[i] = false
		}
	} else {
		inQ, ring = make([]bool, nr), make([]int, nr)
	}
	// FIFO ring over row indices, starts full; inQ dedups so at most nr rows
	// are queued at once, making a cap-nr ring exact (no reslice reallocation)
	for ri := range ring {
		ring[ri] = ri
		inQ[ri] = true
	}
	head, count := 0, nr
	// the 1e-9 improvement floor guarantees termination; the cap only
	// guards zeno chains, and a capped exit is still a valid tightening
	for done := 0; count > 0 && done < 64*nr; done++ {
		ri := ring[head]
		head++
		if head == nr {
			head = 0
		}
		count--
		inQ[ri] = false
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
		scale := math.Max(1, math.Max(math.Abs(minSum), math.Abs(maxSum)))
		if minInf == 0 && rub < inf && minSum > rub+1e-7*scale {
			return false
		}
		if maxInf == 0 && rlb > -inf && maxSum < rlb-1e-7*scale {
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
			// derived bounds are rounded OUTWARD by the row's error
			// scale: inward drift compounds along equality chains
			out := 1e-9 * scale / math.Max(math.Abs(a), 1e-12)
			nl, nu := l, u
			if rub < inf && !math.IsInf(omin, -1) {
				if a > 0 {
					nu = math.Min(nu, (rub-omin)/a+out)
				} else {
					nl = math.Max(nl, (rub-omin)/a-out)
				}
			}
			if rlb > -inf && !math.IsInf(omax, 1) {
				if a > 0 {
					nl = math.Max(nl, (rlb-omax)/a-out)
				} else {
					nu = math.Min(nu, (rlb-omax)/a+out)
				}
			}
			if p.Cols[j].Integer {
				s := 1e-7 * math.Max(1, math.Max(math.Abs(nl), math.Abs(nu)))
				nl = math.Ceil(nl - s)
				nu = math.Floor(nu + s)
			}
			if nl > nu+1e-7*math.Max(1, math.Abs(nl)) {
				return false
			}
			if nl > l+1e-9 || nu < u-1e-9 {
				lb[j], ub[j] = math.Max(l, nl), math.Min(u, nu)
				for _, rr := range p.Cols[j].Idx {
					if !inQ[rr] {
						inQ[rr] = true
						tail := head + count
						if tail >= nr {
							tail -= nr
						}
						ring[tail] = rr
						count++
					}
				}
			}
		}
	}
	return true
}

// probe tentatively fixes each binary both ways (CglProbing-style): an
// infeasible side fixes the binary; otherwise merged implied bounds apply and
// one-sided rows are coefficient-strengthened from the propagated activities.
// Effort is time-boxed: partial results are valid tightenings. Returns the
// number of model changes (fixes + strengthened coefficients).
func probe(p *problem.Problem, deadline time.Time) int {
	nFix, nTight, nProbed, nStr := 0, 0, 0, 0
	t0 := time.Now()
	defer func() {
		debugf("probe: %d binaries probed, %d fixed, %d bounds tightened, %d coefs strengthened in %v", nProbed, nFix, nTight, nStr, time.Since(t0))
	}()
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
			return nFix + nStr
		}
		nProbed++
		copy(lb0, base[:n])
		copy(ub0, base[n:])
		copy(lb1, base[:n])
		copy(ub1, base[n:])
		ub0[j] = 0
		lb1[j] = 1
		feas0 := propagate(p, lb0, ub0, nil)
		feas1 := propagate(p, lb1, ub1, nil)
		switch {
		case !feas0 && !feas1:
			return nFix + nStr // infeasible problem: leave it to the LP to report
		case !feas0:
			p.Cols[j].LB = 1
			base[j] = 1
			nFix++
		case !feas1:
			p.Cols[j].UB = 0
			base[n+j] = 0
			nFix++
		default:
			nStr += strengthenFromProbe(p, j, lb0, ub0, lb1, ub1)
			// merged implied bounds only for integer columns: continuous
			// merges compound propagation drift into a collapse cascade
			for i := range n {
				if !p.Cols[i].Integer {
					continue
				}
				if l := math.Min(lb0[i], lb1[i]); l > base[i]+1e-9 {
					p.Cols[i].LB = l
					base[i] = l
					nTight++
				}
				if u := math.Max(ub0[i], ub1[i]); u < base[n+i]-1e-9 {
					p.Cols[i].UB = u
					base[n+i] = u
					nTight++
				}
			}
		}
	}
	return nFix + nStr
}

// rowPos returns j's position in row r's index list, or -1.
func rowPos(r *problem.Row, j int) int {
	for k, jj := range r.Idx {
		if jj == j {
			return k
		}
	}
	return -1
}

// strengthenFromProbe tightens binary j's coefficient in one-sided rows using
// the probing-propagated activities of both sides (CglProbing Cgl0003I).
// Integer-feasible points are preserved; only the LP relaxation tightens.
func strengthenFromProbe(p *problem.Problem, j int, lb0, ub0, lb1, ub1 []float64) int {
	inf := problem.Inf
	nStr := 0
	c := &p.Cols[j]
	for _, ri := range c.Idx {
		r := &p.Rows[ri]
		rlb, rub := r.Bounds()
		pureLE := rub < inf && rlb == -inf
		pureGE := rlb > -inf && rub == inf
		if !pureLE && !pureGE {
			continue
		}
		kk := rowPos(r, j)
		if kk < 0 || r.Coef[kk] == 0 {
			continue
		}
		// activity range of the whole row under one probing side's bounds
		act := func(lb, ub []float64) (lo, hi float64) {
			for k, jj := range r.Idx {
				ak := r.Coef[k]
				l, u := lb[jj], ub[jj]
				if l <= -inf {
					l = math.Inf(-1)
				}
				if u >= inf {
					u = math.Inf(1)
				}
				alo, ahi := ak*l, ak*u
				if ak < 0 {
					alo, ahi = ahi, alo
				}
				lo += alo
				hi += ahi
			}
			return lo, hi
		}
		if pureLE {
			tol := 1e-7 * math.Max(1, math.Abs(rub))
			if _, hi0 := act(lb0, ub0); !math.IsInf(hi0, 1) && hi0 < rub-tol {
				// x_j=0 side slack by delta: shift a_j and the rhs down together
				delta := rub - hi0
				setCoef(p, ri, kk, r.Coef[kk]-delta)
				r.RHS -= delta
				nStr++
				continue
			}
			if _, hi1 := act(lb1, ub1); !math.IsInf(hi1, 1) && hi1 < rub-tol {
				// x_j=1 side slack: raise a_j so that side just reaches the rhs
				setCoef(p, ri, kk, r.Coef[kk]+(rub-hi1))
				nStr++
			}
		} else {
			tol := 1e-7 * math.Max(1, math.Abs(rlb))
			if lo0, _ := act(lb0, ub0); !math.IsInf(lo0, -1) && lo0 > rlb+tol {
				delta := lo0 - rlb
				setCoef(p, ri, kk, r.Coef[kk]+delta)
				r.RHS += delta
				nStr++
				continue
			}
			if lo1, _ := act(lb1, ub1); !math.IsInf(lo1, -1) && lo1 > rlb+tol {
				setCoef(p, ri, kk, r.Coef[kk]-(lo1-rlb))
				nStr++
			}
		}
	}
	return nStr
}
