// Package simplex implements a bounded-variable primal simplex method over
// the standard-form system built from a problem.Problem: structural columns
// plus one logical (slack) variable per row, with row bounds folded onto the
// logical variable so L/G/E/R rows need no special-casing in the core loop.
package simplex

import (
	"math"

	"cbcgo/internal/problem"
)

const (
	inf     = problem.Inf
	eps     = 1e-9
	optEps  = 1e-7
	maxIter = 200000
)

type Status int

const (
	Optimal Status = iota
	Infeasible
	Unbounded
	IterLimit
)

type varStat int8

const (
	atLower varStat = iota
	atUpper
	basic
	free
)

// LP is the immutable (per cold-start) coefficient data for the standard-form
// system. Variables [0,n) are structural columns; variables [n,n+m) are
// logical (row-slack) variables, one per row, with column n+i equal to -e_i.
type LP struct {
	n, m int // structural columns, rows (== logical variables)

	// sparse column data, index by variable (0..n+m-1); logical columns are
	// synthesized on the fly rather than stored (single -1 entry).
	colRow [][]int
	colVal [][]float64

	lb, ub  []float64 // length n+m
	cost    []float64 // length n+m, signed for internal minimization
	rawObj  []float64 // length n, unsigned (as given in the problem)
	objSign float64
}

// State is a mutable basis/solution snapshot, reusable across resolves.
type State struct {
	status  []varStat
	basicOf []int // row -> basic variable index
	value   []float64
	binv    [][]float64
}

func Build(p *problem.Problem) *LP {
	n := p.NumCols()
	m := p.NumRows()
	lp := &LP{
		n: n, m: m,
		colRow:  make([][]int, n+m),
		colVal:  make([][]float64, n+m),
		lb:      make([]float64, n+m),
		ub:      make([]float64, n+m),
		cost:    make([]float64, n+m),
		rawObj:  make([]float64, n),
		objSign: p.ObjSense,
	}
	for j, c := range p.Cols {
		lp.lb[j], lp.ub[j] = c.LB, c.UB
		lp.rawObj[j] = c.Obj
		lp.cost[j] = c.Obj * p.ObjSense
		rows := append([]int(nil), c.Idx...)
		vals := append([]float64(nil), c.Coef...)
		lp.colRow[j] = rows
		lp.colVal[j] = vals
	}
	for i := range p.Rows {
		r := &p.Rows[i]
		lb, ub := r.Bounds()
		lp.lb[n+i], lp.ub[n+i] = lb, ub
		lp.cost[n+i] = 0
	}
	return lp
}

func (lp *LP) nTotal() int { return lp.n + lp.m }

// column returns the (row,val) pairs of variable j's column in the full
// [A | -I] matrix.
func (lp *LP) column(j int) ([]int, []float64) {
	if j < lp.n {
		return lp.colRow[j], lp.colVal[j]
	}
	// logical variable for row j-n: single entry, coefficient -1.
	i := j - lp.n
	return []int{i}, []float64{-1}
}

// initState builds the trivial starting basis: all logical variables basic
// (B = -I, so Binv = -I too), structural variables nonbasic at whichever
// finite bound is available (lower preferred), or free at 0.
func (lp *LP) initState() *State {
	nt := lp.nTotal()
	st := &State{
		status:  make([]varStat, nt),
		basicOf: make([]int, lp.m),
		value:   make([]float64, nt),
		binv:    make([][]float64, lp.m),
	}
	for i := 0; i < lp.m; i++ {
		st.binv[i] = make([]float64, lp.m)
		st.binv[i][i] = -1
		st.basicOf[i] = lp.n + i
		st.status[lp.n+i] = basic
	}
	for j := 0; j < lp.n; j++ {
		lp.resetNonbasic(st, j)
	}
	lp.recomputeBasics(st)
	return st
}

// resetNonbasic snaps variable j onto a valid nonbasic bound (lower
// preferred, then upper, then free); also used after a bound override.
func (lp *LP) resetNonbasic(st *State, j int) {
	switch {
	case lp.lb[j] > -inf:
		st.status[j] = atLower
		st.value[j] = lp.lb[j]
	case lp.ub[j] < inf:
		st.status[j] = atUpper
		st.value[j] = lp.ub[j]
	default:
		st.status[j] = free
		st.value[j] = 0
	}
}

// Clone returns a deep copy, safe to mutate independently of st (used to
// warm-start a branch-and-bound child node from its parent's basis).
func (st *State) Clone() *State {
	c := &State{
		status:  append([]varStat(nil), st.status...),
		basicOf: append([]int(nil), st.basicOf...),
		value:   append([]float64(nil), st.value...),
		binv:    make([][]float64, len(st.binv)),
	}
	for i, row := range st.binv {
		c.binv[i] = append([]float64(nil), row...)
	}
	return c
}

// WarmSolve resolves from st after bound changes on touched, instead of
// resetting to the trivial all-slack basis.
func (lp *LP) WarmSolve(st *State, touched []int) (Status, *State, float64) {
	for _, j := range touched {
		if st.status[j] != basic {
			lp.resetNonbasic(st, j)
		}
	}
	lp.recomputeBasics(st)
	return lp.solveFrom(st)
}

// recomputeBasics sets every basic variable's value from the nonbasic
// values: x_B = -Binv * N * x_N (the system's RHS is always zero).
func (lp *LP) recomputeBasics(st *State) {
	m := lp.m
	residual := make([]float64, m)
	for j := 0; j < lp.nTotal(); j++ {
		if st.status[j] == basic {
			continue
		}
		v := st.value[j]
		if v == 0 {
			continue
		}
		rows, vals := lp.column(j)
		for k, r := range rows {
			residual[r] += vals[k] * v
		}
	}
	for i := 0; i < m; i++ {
		var s float64
		for k := 0; k < m; k++ {
			s += st.binv[i][k] * (-residual[k])
		}
		st.value[st.basicOf[i]] = s
	}
}

func (lp *LP) alpha(st *State, j int) []float64 {
	rows, vals := lp.column(j)
	a := make([]float64, lp.m)
	for k, r := range rows {
		c := vals[k]
		if c == 0 {
			continue
		}
		for i := 0; i < lp.m; i++ {
			a[i] += st.binv[i][r] * c
		}
	}
	return a
}

// duals computes y = cost_B^T * Binv (as a length-m row vector).
func duals(st *State, cost []float64, m int) []float64 {
	y := make([]float64, m)
	for i := 0; i < m; i++ {
		cb := cost[st.basicOf[i]]
		if cb == 0 {
			continue
		}
		for k := 0; k < m; k++ {
			y[k] += cb * st.binv[i][k]
		}
	}
	return y
}

func (lp *LP) reducedCost(y []float64, cost []float64, j int) float64 {
	rows, vals := lp.column(j)
	d := cost[j]
	for k, r := range rows {
		d -= y[r] * vals[k]
	}
	return d
}

// ColdSolve runs Phase 1 (feasibility) then Phase 2 (optimize the real
// objective) from the trivial all-logical-variables basis.
func (lp *LP) ColdSolve() (Status, *State, float64) {
	st := lp.initState()
	return lp.solveFrom(st)
}

func (lp *LP) solveFrom(st *State) (Status, *State, float64) {
	status := lp.run(st)
	if status != Optimal {
		return status, st, 0
	}
	return Optimal, st, lp.objective(st)
}

func (lp *LP) objective(st *State) float64 {
	var s float64
	for j := 0; j < lp.n; j++ {
		s += lp.rawObj[j] * st.value[j]
	}
	return s
}

// InternalObjective returns the signed (always-minimized) objective, so
// branch-and-bound bounds compare consistently regardless of obj sense.
func (lp *LP) InternalObjective(st *State) float64 {
	return lp.objective(st) * lp.objSign
}

// Bound returns variable j's current [lb,ub].
func (lp *LP) Bound(j int) (float64, float64) { return lp.lb[j], lp.ub[j] }

// SetBound overrides variable j's [lb,ub] for subsequent solves.
func (lp *LP) SetBound(j int, lb, ub float64) { lp.lb[j] = lb; lp.ub[j] = ub }

// NumCols returns the number of structural columns.
func (lp *LP) NumCols() int { return lp.n }

// phaseCost recomputes the Phase-1 cost vector: -1 for a basic var below
// its lower bound, +1 above its upper bound; false once all are feasible.
func (lp *LP) phaseCost(st *State) ([]float64, bool) {
	cost := make([]float64, lp.nTotal())
	inPhase1 := false
	for i := 0; i < lp.m; i++ {
		bv := st.basicOf[i]
		v := st.value[bv]
		switch {
		case v < lp.lb[bv]-eps:
			cost[bv] = -1
			inPhase1 = true
		case v > lp.ub[bv]+eps:
			cost[bv] = 1
			inPhase1 = true
		}
	}
	return cost, inPhase1
}

// run recomputes the active cost vector fresh before every pivot, so a
// variable that becomes feasible mid-sequence stops influencing pricing.
func (lp *LP) run(st *State) Status {
	for iter := 0; ; iter++ {
		if iter > maxIter {
			return IterLimit
		}
		cost, inPhase1 := lp.phaseCost(st)
		if !inPhase1 {
			cost = lp.cost
		}
		y := duals(st, cost, lp.m)
		q, dir := lp.chooseEntering(st, y, cost)
		if q < 0 {
			if inPhase1 {
				return Infeasible
			}
			return Optimal
		}
		a := lp.alpha(st, q)
		t, row, isFlip := lp.ratioTest(st, a, q, dir, inPhase1)
		if row < 0 && !isFlip {
			if inPhase1 {
				return Infeasible
			}
			return Unbounded
		}
		lp.pivot(st, q, dir, a, t, row, isFlip)
	}
}

func (lp *LP) chooseEntering(st *State, y, cost []float64) (q int, dir float64) {
	best := optEps
	q = -1
	nt := lp.nTotal()
	for j := 0; j < nt; j++ {
		if st.status[j] == basic {
			continue
		}
		d := lp.reducedCost(y, cost, j)
		switch st.status[j] {
		case atLower:
			if -d > best {
				best, q, dir = -d, j, 1
			}
		case atUpper:
			if d > best {
				best, q, dir = d, j, -1
			}
		case free:
			if math.Abs(d) > best {
				best, q = math.Abs(d), j
				if d < 0 {
					dir = 1
				} else {
					dir = -1
				}
			}
		}
	}
	return q, dir
}

// ratioTest finds how far entering variable q can move in direction dir
// before it flips to its opposite bound or a basic variable hits a bound.
func (lp *LP) ratioTest(st *State, a []float64, q int, dir float64, phase1 bool) (t float64, leaveRow int, isFlip bool) {
	t = math.Inf(1)
	leaveRow = -1
	if lp.lb[q] > -inf && lp.ub[q] < inf {
		t = lp.ub[q] - lp.lb[q]
		isFlip = true
	}
	for i := 0; i < lp.m; i++ {
		if math.Abs(a[i]) < eps {
			continue
		}
		bv := st.basicOf[i]
		delta := -a[i] * dir // rate of change of basic var i per unit t
		v := st.value[bv]
		lb, ub := lp.lb[bv], lp.ub[bv]

		var limit float64
		ok := false
		if phase1 {
			switch {
			case v < lb-eps: // infeasible low: only care about moving up to lb
				if delta > eps {
					limit = (lb - v) / delta
					ok = true
				}
			case v > ub+eps: // infeasible high: only care about moving down to ub
				if delta < -eps {
					limit = (ub - v) / delta
					ok = true
				}
			default: // feasible: normal both-direction limits
				if delta > eps && ub < inf {
					limit = (ub - v) / delta
					ok = true
				} else if delta < -eps && lb > -inf {
					limit = (lb - v) / delta
					ok = true
				}
			}
		} else {
			if delta > eps && ub < inf {
				limit = (ub - v) / delta
				ok = true
			} else if delta < -eps && lb > -inf {
				limit = (lb - v) / delta
				ok = true
			}
		}
		if !ok {
			continue
		}
		if limit < 0 {
			limit = 0
		}
		if limit < t-eps {
			t = limit
			leaveRow = i
			isFlip = false
		}
	}
	if leaveRow < 0 && !isFlip {
		return t, -1, false
	}
	return t, leaveRow, isFlip
}

func (lp *LP) pivot(st *State, q int, dir float64, a []float64, t float64, leaveRow int, isFlip bool) {
	if t < 0 {
		t = 0
	}
	// update all basic values
	for i := 0; i < lp.m; i++ {
		if a[i] == 0 {
			continue
		}
		st.value[st.basicOf[i]] -= a[i] * dir * t
	}
	oldVal := st.value[q]
	st.value[q] = oldVal + dir*t

	if isFlip {
		if dir > 0 {
			st.status[q] = atUpper
			st.value[q] = lp.ub[q]
		} else {
			st.status[q] = atLower
			st.value[q] = lp.lb[q]
		}
		return
	}

	leaving := st.basicOf[leaveRow]
	// snap the leaving variable exactly onto the bound it reached to avoid
	// floating-point drift.
	lb, ub := lp.lb[leaving], lp.ub[leaving]
	if math.Abs(st.value[leaving]-lb) <= math.Abs(st.value[leaving]-ub) {
		st.status[leaving] = atLower
		st.value[leaving] = lb
	} else {
		st.status[leaving] = atUpper
		st.value[leaving] = ub
	}

	pivotVal := a[leaveRow]
	rowR := st.binv[leaveRow]
	for k := 0; k < lp.m; k++ {
		rowR[k] /= pivotVal
	}
	for i := 0; i < lp.m; i++ {
		if i == leaveRow || a[i] == 0 {
			continue
		}
		factor := a[i]
		row := st.binv[i]
		for k := 0; k < lp.m; k++ {
			row[k] -= factor * rowR[k]
		}
	}
	st.basicOf[leaveRow] = q
	st.status[q] = basic
}

// Solution extracts primal values, row activity, reduced costs and row
// duals (prices) from a solved state.
func (lp *LP) Solution(st *State) (x, rowActivity, reducedCost, rowPrice []float64) {
	x = append([]float64(nil), st.value[:lp.n]...)
	rowActivity = make([]float64, lp.m)
	for i := 0; i < lp.m; i++ {
		rowActivity[i] = -st.value[lp.n+i] * -1 // logical var value == row activity (y_i = Ax_i)
	}
	y := duals(st, lp.cost, lp.m)
	rowPrice = make([]float64, lp.m)
	for i := range rowPrice {
		rowPrice[i] = y[i] * lp.objSign
	}
	reducedCost = make([]float64, lp.n)
	for j := 0; j < lp.n; j++ {
		reducedCost[j] = lp.reducedCost(y, lp.cost, j) * lp.objSign
	}
	return
}
