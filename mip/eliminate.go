// Singleton-column elimination (CglPreProcess-style): costed continuous
// columns appearing in one row are substituted out before the LP is built.
package mip

import (
	"math"

	"cbcgo/problem"
)

type elimKind byte

const (
	elimTight elimKind = iota // cost pins the row: x = (b - rest)/a
	elimFixed                 // row never blocks: x sits at its bound
)

type elimRecord struct {
	col, row int
	kind     elimKind
	a        float64 // column's coefficient in its row
	b        float64 // elimTight: row bound the column pins
	val      float64 // elimFixed: fixed value
	obj      float64 // cost at elimination time (dual postsolve shift)
}

// reduction maps a reduced problem back to the caller's original one.
type reduction struct {
	orig    *problem.Problem
	records []elimRecord // in elimination order; postsolve walks it backwards
	colMap  []int        // original col index -> reduced index, -1 if eliminated
}

// eliminateSingletons returns a reduced copy of p (p itself is untouched)
// with eligible singletons substituted out; (nil, nil) when nothing applies.
func eliminateSingletons(p *problem.Problem) (*problem.Problem, *reduction) {
	if len(p.SOSs) != 0 {
		return nil, nil
	}
	inf := problem.Inf
	obj := make([]float64, len(p.Cols))
	for j := range p.Cols {
		obj[j] = p.Cols[j].Obj
	}
	rlb := make([]float64, len(p.Rows))
	rub := make([]float64, len(p.Rows))
	touched := make([]bool, len(p.Rows))
	for ri := range p.Rows {
		rlb[ri], rub[ri] = p.Rows[ri].Bounds()
	}
	elim := make([]bool, len(p.Cols))
	var records []elimRecord
	nTight, nFixed := 0, 0

	// folding a cost onto row siblings can re-classify them, so iterate
	for changed := true; changed; {
		changed = false
		for j := range p.Cols {
			c := &p.Cols[j]
			if elim[j] || c.Integer || len(c.Idx) != 1 {
				continue
			}
			a := c.Coef[0]
			ri := c.Idx[0]
			cm := obj[j] * p.ObjSense
			if math.Abs(a) < 1e-9 || cm == 0 {
				continue
			}
			d := 1.0 // objective-improving direction for x_j (minimize sense)
			if cm > 0 {
				d = -1
			}
			rowBlocks := (a*d > 0 && rub[ri] < inf) || (a*d < 0 && rlb[ri] > -inf)
			colBlocks := (d > 0 && c.UB < inf) || (d < 0 && c.LB > -inf)
			switch {
			case rowBlocks && !colBlocks:
				// any optimum pins the row: substitute x = (b - rest)/a and
				// carry the column's remaining bound onto the row
				b := rub[ri]
				if a*d < 0 {
					b = rlb[ri]
				}
				lo, hi := c.LB, c.UB
				if lo <= -inf {
					lo = math.Inf(-1)
				}
				if hi >= inf {
					hi = math.Inf(1)
				}
				nlb, nub := b-a*hi, b-a*lo
				if a < 0 {
					nlb, nub = b-a*lo, b-a*hi
				}
				if math.IsInf(nlb, -1) {
					nlb = -inf
				}
				if math.IsInf(nub, 1) {
					nub = inf
				}
				if nlb <= -inf && nub >= inf {
					continue // fully free column: row would become vacuous
				}
				f := obj[j] / a
				r := &p.Rows[ri]
				for k, jj := range r.Idx {
					if jj != j && !elim[jj] {
						obj[jj] -= f * r.Coef[k]
					}
				}
				records = append(records, elimRecord{col: j, row: ri, kind: elimTight, a: a, b: b, obj: obj[j]})
				rlb[ri], rub[ri] = nlb, nub
				touched[ri], elim[j], changed = true, true, true
				nTight++
			case !rowBlocks && !colBlocks:
				continue // unbounded ray: leave it for the solver to report
			case !rowBlocks:
				// the row never resists the cost direction: x sits at its bound
				v := c.UB
				if d < 0 {
					v = c.LB
				}
				if rlb[ri] > -inf {
					rlb[ri] -= a * v
				}
				if rub[ri] < inf {
					rub[ri] -= a * v
				}
				records = append(records, elimRecord{col: j, row: ri, kind: elimFixed, a: a, val: v, obj: obj[j]})
				touched[ri], elim[j], changed = true, true, true
				nFixed++
			}
		}
	}
	if len(records) == 0 {
		return nil, nil
	}
	debugf("eliminate: %d singleton cols removed (%d tight, %d fixed) of %d", len(records), nTight, nFixed, len(p.Cols))

	q := problem.New()
	q.Name, q.ObjSense = p.Name, p.ObjSense
	colMap := make([]int, len(p.Cols))
	for j := range p.Cols {
		if elim[j] {
			colMap[j] = -1
			continue
		}
		c := &p.Cols[j]
		colMap[j] = q.AddCol(c.Name, c.LB, c.UB, obj[j], c.Integer, nil, nil)
	}
	for ri := range p.Rows {
		r := &p.Rows[ri]
		idx := make([]int, 0, len(r.Idx))
		coef := make([]float64, 0, len(r.Idx))
		for k, jj := range r.Idx {
			if colMap[jj] >= 0 {
				idx = append(idx, colMap[jj])
				coef = append(coef, r.Coef[k])
			}
		}
		nri := q.AddRow(r.Name, idx, coef, r.Sense, r.RHS)
		nr := &q.Rows[nri]
		nr.HasRange, nr.Range = r.HasRange, r.Range
		if touched[ri] {
			setRowBounds(nr, rlb[ri], rub[ri])
		}
	}
	return q, &reduction{orig: p, records: records, colMap: colMap}
}

// setRowBounds rewrites a row's sense/rhs/range to represent [lb, ub].
func setRowBounds(r *problem.Row, lb, ub float64) {
	inf := problem.Inf
	r.HasRange, r.Range = false, 0
	switch {
	case lb == ub:
		r.Sense, r.RHS = problem.EQ, lb
	case lb > -inf && ub < inf:
		r.Sense, r.RHS = problem.LE, ub
		r.HasRange, r.Range = true, ub-lb
	case ub < inf:
		r.Sense, r.RHS = problem.LE, ub
	default:
		r.Sense, r.RHS = problem.GE, lb
	}
}

// shrinkX maps an original-space point onto the reduced column space.
func (red *reduction) shrinkX(x []float64) []float64 {
	out := make([]float64, 0, len(red.colMap))
	for j, nj := range red.colMap {
		if nj >= 0 {
			out = append(out, x[j])
		}
	}
	return out
}

// expand rewrites a reduced-space Result in the original column space,
// reconstructing eliminated columns, duals and the pinned row activities.
func (red *reduction) expand(res *Result) {
	if res.X == nil {
		return
	}
	n := len(red.orig.Cols)
	x := make([]float64, n)
	rc := make([]float64, n)
	for j, nj := range red.colMap {
		if nj >= 0 {
			x[j] = res.X[nj]
			if nj < len(res.ReducedCost) {
				rc[j] = res.ReducedCost[nj]
			}
		}
	}
	act, price := res.RowActivity, res.RowPrice
	for i := len(red.records) - 1; i >= 0; i-- {
		rec := &red.records[i]
		c := &red.orig.Cols[rec.col]
		switch rec.kind {
		case elimTight:
			v := (rec.b - act[rec.row]) / rec.a
			x[rec.col] = math.Min(math.Max(v, c.LB), c.UB)
			act[rec.row] = rec.b
			rc[rec.col] = -rec.a * price[rec.row]
			price[rec.row] += rec.obj / rec.a
		case elimFixed:
			x[rec.col] = rec.val
			act[rec.row] += rec.a * rec.val
			rc[rec.col] = rec.obj - rec.a*price[rec.row]
		}
	}
	obj := 0.0
	for j := range red.orig.Cols {
		obj += red.orig.Cols[j].Obj * x[j]
	}
	res.X, res.ReducedCost, res.Obj = x, rc, obj
}
