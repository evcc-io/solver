// Package simplex implements a bounded-variable primal simplex method over
// the standard-form system built from a problem.Problem: structural columns
// plus one logical (slack) variable per row, with row bounds folded onto the
// logical variable so L/G/E/R rows need no special-casing in the core loop.
package simplex

import (
	"math"
	"sort"
	"time"

	"cbcgo/problem"
)

const (
	inf     = problem.Inf
	eps     = 1e-9
	optEps  = 1e-7
	maxIter = 200000

	// EXPAND (Gill et al.): working bounds widen by xtolStep per pivot up
	// to feasTol, then snap back; degenerate pivots keep a minimum step
	feasTol  = 1e-6
	xtolInit = 2.5e-7
	xtolStep = 1e-8

	// partial pricing kicks in only above this many total variables, and
	// then scans nt/partialPricingDivisor columns per window.
	partialPricingThreshold = 4000
	partialPricingDivisor   = 8
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

	xtol float64 // current EXPAND working-bound expansion (run resets it)

	// sparse column data, index by variable (0..n+m-1); logical columns are
	// synthesized on the fly rather than stored (single -1 entry).
	colRow [][]int
	colVal [][]float64

	// immutable int32 mirror of every column (incl. logicals), shared by
	// all factorizations instead of being converted per refactorize
	colRow32 [][]int32
	// row-wise mirror of A incl. logical columns: the dual pivot row only
	// touches columns intersecting the BTRAN row's support
	rowCol [][]int32
	rowVal [][]float64
	colVal32 [][]float64

	lb, ub  []float64 // length n+m
	cost    []float64 // length n+m, signed for internal minimization
	rawObj  []float64 // length n, unsigned (as given in the problem)
	objSign float64

	// Deadline, when set, aborts a solve with IterLimit once exceeded
	// (checked periodically inside the pivot loop).
	Deadline time.Time

	// pricingWindow >0 makes chooseEntering scan a rotating column window
	// (partial pricing) instead of all n+m every pivot; 0 means full scan.
	pricingWindow int
	pricingCursor int // scan position for partial pricing

	fws *factorWS // reusable factorization workspace

	// Stats accumulates pivot counts across solves (diagnostics only).
	Stats struct {
		Solves, Phase1, Phase2, Dual               int64
		DualStallQ, DualStallA, DualCap, DualFlips int64
		KernelMax, Bland                           int64
	}
}

// State is a mutable basis/solution snapshot, reusable across resolves.
type State struct {
	status  []varStat
	basicOf []int // row -> basic variable index
	value   []float64
	f       *factor // basis factorization, shared between clones
	etas    []*eta  // product-form updates since f was built
}

// ftranVec computes Binv*v in place (v by row in, by basis position out).
func (st *State) ftranVec(v []float64) {
	st.f.ftran(v)
	applyEtas(st.etas, v)
}

// btranVec computes Binv^T*w in place (w by basis position in, by row out).
func (st *State) btranVec(w []float64) {
	applyEtasT(st.etas, w)
	st.f.btran(w)
}

// refactorize rebuilds st's basis factorization and clears its eta file;
// on (numerical) failure the existing factor+etas stay valid.
func (lp *LP) refactorize(st *State) bool {
	colRow := make([][]int32, lp.m)
	colVal := make([][]float64, lp.m)
	for pos, j := range st.basicOf {
		colRow[pos], colVal[pos] = lp.colRow32[j], lp.colVal32[j]
	}
	if lp.fws == nil {
		lp.fws = newFactorWS(lp.m)
	}
	f := factorize(lp.m, colRow, colVal, lp.fws)
	if f == nil {
		return false
	}
	if k := int64(len(f.kRows)); k > lp.Stats.KernelMax {
		lp.Stats.KernelMax = k
	}
	st.f = f
	st.etas = nil
	return true
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
	lp.colRow32 = make([][]int32, n+m)
	lp.colVal32 = make([][]float64, n+m)
	for j := range n + m {
		rows, vals := lp.column(j)
		cr := make([]int32, len(rows))
		for k, r := range rows {
			cr[k] = int32(r)
		}
		lp.colRow32[j], lp.colVal32[j] = cr, vals
	}
	lp.rowCol = make([][]int32, m)
	lp.rowVal = make([][]float64, m)
	cnt := make([]int, m)
	for j := range n {
		for _, r := range lp.colRow[j] {
			cnt[r]++
		}
	}
	for r := range m {
		lp.rowCol[r] = make([]int32, 0, cnt[r]+1)
		lp.rowVal[r] = make([]float64, 0, cnt[r]+1)
	}
	for j := range n {
		for k, r := range lp.colRow[j] {
			lp.rowCol[r] = append(lp.rowCol[r], int32(j))
			lp.rowVal[r] = append(lp.rowVal[r], lp.colVal[j][k])
		}
	}
	for r := range m {
		lp.rowCol[r] = append(lp.rowCol[r], int32(n+r))
		lp.rowVal[r] = append(lp.rowVal[r], -1)
	}
	// partial pricing disabled: extra pivots from window-local entering
	// choices dominate scan savings when each pivot is O(m^2) on dense binv
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
	m := lp.m
	st := &State{
		status:  make([]varStat, nt),
		basicOf: make([]int, m),
		value:   make([]float64, nt),
	}
	for i := range m {
		st.basicOf[i] = lp.n + i
		st.status[lp.n+i] = basic
	}
	lp.refactorize(st)
	for j := range lp.n {
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
	return &State{
		status:  append([]varStat(nil), st.status...),
		basicOf: append([]int(nil), st.basicOf...),
		value:   append([]float64(nil), st.value...),
		f:       st.f, // immutable, shared
		etas:    append([]*eta(nil), st.etas...),
	}
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
	// after a bound change the basis stays (near) dual feasible: a few dual
	// pivots restore primal feasibility far cheaper than a primal Phase 1
	lp.dualRun(st)
	return lp.solveFrom(st)
}

// dualPivotCap bounds dual pivots per re-solve; a degenerate dual bails to
// the primal run instead of grinding forever.
const dualPivotCap = 1024

// blandAfter switches entering selection to Bland's rule once this many
// consecutive degenerate (zero-step) pivots occur.
const blandAfter = 384

// dualRun is a best-effort bounded-variable dual simplex (CBC re-solves via
// Clp's dual): it bails on any stall and leaves the rest to the primal run.
func (lp *LP) dualRun(st *State) {
	m, nt := lp.m, lp.nTotal()
	a := make([]float64, m)
	y := make([]float64, m)
	rowBuf := make([]float64, m)
	alphaRow := make([]float64, nt)

	// reduced costs computed once, then updated incrementally per pivot
	clear(y)
	duals(st, lp.cost, m, y)
	d := make([]float64, nt)
	for j := range nt {
		if st.status[j] != basic {
			d[j] = lp.reducedCost(y, lp.cost, j)
		}
	}

	var skip map[int]bool // rows whose violation the dual cannot fix
	touched := make([]int, 0, 256)
	seen := make([]bool, nt)
	for iter := range dualPivotCap + 1 {
		if !lp.Deadline.IsZero() && time.Now().After(lp.Deadline) {
			return
		}
		if iter == dualPivotCap {
			lp.Stats.DualCap++
			return
		}
		// leaving variable: the most primal-infeasible basic
		r, bound, worst := -1, 0.0, 100*eps
		for i := range m {
			if skip[i] {
				continue
			}
			bv := st.basicOf[i]
			v := st.value[bv]
			if viol := lp.lb[bv] - v; viol > worst {
				worst, r, bound = viol, i, lp.lb[bv]
			}
			if viol := v - lp.ub[bv]; viol > worst {
				worst, r, bound = viol, i, lp.ub[bv]
			}
		}
		if r < 0 {
			return // primal feasible (or only unfixable rows left): done
		}
		leaving := st.basicOf[r]
		needSign := 1.0 // required sign of alpha[r]*dir so the violation shrinks
		if st.value[leaving] < bound {
			needSign = -1
		}
		// pivot row r: alphaRow[j] = binv_r . col_j for all nonbasic j
		clear(rowBuf)
		rowBuf[r] = 1
		st.btranVec(rowBuf)
		rowR := rowBuf
		for _, j := range touched {
			alphaRow[j] = 0
		}
		touched = touched[:0]
		for r := range m {
			v := rowR[r]
			if math.Abs(v) < 1e-12 {
				continue
			}
			cols, vals := lp.rowCol[r], lp.rowVal[r]
			for k, j32 := range cols {
				j := int(j32)
				if !seen[j] {
					seen[j] = true
					touched = append(touched, j)
				}
				alphaRow[j] += v * vals[k]
			}
		}
		var cands []dualCand
		for _, j := range touched {
			seen[j] = false
			if st.status[j] == basic {
				continue
			}
			alphaJ := alphaRow[j]
			if math.Abs(alphaJ) < 1e-7 {
				continue
			}
			for _, dir := range enterDirs(st.status[j]) {
				if alphaJ*dir*needSign < eps {
					continue
				}
				cands = append(cands, dualCand{j, dir, math.Abs(d[j]) / math.Abs(alphaJ)})
			}
		}
		sort.Slice(cands, func(a, b int) bool { return cands[a].ratio < cands[b].ratio })

		// dual long step (Clp "dual with flips"): boxed candidates that
		// can't absorb the violation get flipped; the overshooter pivots
		q, qdir := -1, 0.0
		viol := math.Abs(st.value[leaving] - bound)
		var flips []int
		for _, cd := range cands {
			jlb, jub := lp.lb[cd.j], lp.ub[cd.j]
			rng := jub - jlb
			aj := math.Abs(alphaRow[cd.j])
			if rng >= inf || rng <= 0 || aj*rng >= viol {
				q, qdir = cd.j, cd.dir
				break
			}
			flips = append(flips, cd.j)
			viol -= aj * rng
		}
		if q >= 0 && len(flips) > 0 {
			// apply flips: one combined column delta, one ftran for basics
			delta := make([]float64, m)
			for _, j := range flips {
				jlb, jub := lp.lb[j], lp.ub[j]
				var dv float64
				if st.status[j] == atLower {
					dv = jub - jlb
					st.status[j] = atUpper
					st.value[j] = jub
				} else {
					dv = jlb - jub
					st.status[j] = atLower
					st.value[j] = jlb
				}
				rows, vals := lp.column(j)
				for k, row := range rows {
					delta[row] += vals[k] * dv
				}
				lp.Stats.DualFlips++
			}
			st.ftranVec(delta)
			for pos := range m {
				st.value[st.basicOf[pos]] -= delta[pos]
			}
		}
		if q < 0 {
			// this row's violation is dual-unfixable: leave it to the
			// primal run, but keep repairing the other rows
			lp.Stats.DualStallQ++
			if skip == nil {
				skip = make(map[int]bool)
			}
			skip[r] = true
			continue
		}
		clear(a)
		lp.alpha(st, q, a)
		if math.Abs(a[r]) < 1e-7 {
			lp.Stats.DualStallA++
			if skip == nil {
				skip = make(map[int]bool)
			}
			skip[r] = true
			continue
		}
		t := (st.value[leaving] - bound) / (a[r] * qdir)
		if t < 0 || math.IsInf(t, 0) || math.IsNaN(t) {
			lp.Stats.DualStallA++
			return
		}
		lp.Stats.Dual++
		lp.pivot(st, q, qdir, a, t, r, false)
		// incremental reduced-cost update from the pivot row
		step := d[q] / alphaRow[q]
		for _, j := range touched {
			if st.status[j] == basic {
				d[j] = 0
			} else if alphaRow[j] != 0 {
				d[j] -= step * alphaRow[j]
			}
		}
		d[leaving] = -step
		d[q] = 0
	}
}

// dualCand is one eligible entering candidate in the dual ratio test.
type dualCand struct {
	j     int
	dir   float64
	ratio float64
}

// enterDirs lists the directions a nonbasic variable may enter the basis in.
func enterDirs(s varStat) []float64 {
	switch s {
	case atLower:
		return dirUp
	case atUpper:
		return dirDown
	default:
		return dirBoth
	}
}

var (
	dirUp   = []float64{1}
	dirDown = []float64{-1}
	dirBoth = []float64{1, -1}
)

// recomputeBasics sets every basic variable's value from the nonbasic
// values: x_B = -Binv * N * x_N (the system's RHS is always zero).
func (lp *LP) recomputeBasics(st *State) {
	m := lp.m
	residual := make([]float64, m)
	for j := range lp.nTotal() {
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
	for i := range m {
		residual[i] = -residual[i]
	}
	st.ftranVec(residual)
	for pos := range m {
		st.value[st.basicOf[pos]] = residual[pos]
	}
}

// alpha fills dst (length lp.m, assumed zeroed) with column j's entries
// against the current basis: Binv * column(j).
func (lp *LP) alpha(st *State, j int, dst []float64) {
	rows, vals := lp.column(j)
	for k, r := range rows {
		dst[r] += vals[k]
	}
	st.ftranVec(dst)
}

// duals fills dst (length m, assumed zeroed) with y = cost_B^T * Binv.
func duals(st *State, cost []float64, m int, dst []float64) {
	for pos := range m {
		dst[pos] = cost[st.basicOf[pos]]
	}
	st.btranVec(dst)
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
// objective) from the trivial all-logical-variables basis. Bounds are
// perturbed outward by tiny per-index jitter for the solve and restored
// for a short warm cleanup: distinct ratios destroy the degenerate-tie
// plateaus that stall Phase 1 on presolve-tightened problems (Clp-style).
func (lp *LP) ColdSolve() (Status, *State, float64) {
	nt := lp.nTotal()
	savedLb := append([]float64(nil), lp.lb...)
	savedUb := append([]float64(nil), lp.ub...)
	touched := make([]int, 0, nt)
	for j := range nt {
		jit := perturbEps * float64(1+(uint32(j)*2654435761)%97) / 97
		changed := false
		if lp.lb[j] > -inf {
			lp.lb[j] -= jit * math.Max(1, math.Abs(lp.lb[j]))
			changed = true
		}
		if lp.ub[j] < inf {
			lp.ub[j] += jit * math.Max(1, math.Abs(lp.ub[j]))
			changed = true
		}
		if changed {
			touched = append(touched, j)
		}
	}
	st := lp.initState()
	status := lp.run(st)
	copy(lp.lb, savedLb)
	copy(lp.ub, savedUb)
	if status != Optimal {
		// perturbed solve failed; retry clean from scratch
		st = lp.initState()
		return lp.solveFrom(st)
	}
	return lp.WarmSolve(st, touched)
}

const perturbEps = 1e-7

func (lp *LP) solveFrom(st *State) (Status, *State, float64) {
	lp.Stats.Solves++
	status := lp.run(st)
	if status != Optimal {
		return status, st, 0
	}
	return Optimal, st, lp.objective(st)
}

func (lp *LP) objective(st *State) float64 {
	var s float64
	for j := range lp.n {
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

// NumRows returns m, the row (and logical-variable) count.
func (lp *LP) NumRows() int { return lp.m }

// BasicVar returns the variable basic in row position r and its value.
func (st *State) BasicVar(r int) (int, float64) {
	j := st.basicOf[r]
	return j, st.value[j]
}

// VarStatusAtUpper reports whether nonbasic variable j sits at its upper
// bound; (false, false) means basic or free.
func (st *State) VarStatusAtUpper(j int) (atUpperBound, nonbasicAtBound bool) {
	switch st.status[j] {
	case atUpper:
		return true, true
	case atLower:
		return false, true
	default:
		return false, false
	}
}

// IsBasic reports whether variable j is basic in st.
func (st *State) IsBasic(j int) bool { return st.status[j] == basic }

// TableauRow fills dst (length n+m, assumed zeroed) with row position r of
// Binv*[A|-I], the simplex tableau row used for cut generation.
func (lp *LP) TableauRow(st *State, r int, dst []float64) {
	rowR := make([]float64, lp.m)
	rowR[r] = 1
	st.btranVec(rowR)
	for j := range lp.nTotal() {
		rows, vals := lp.column(j)
		var s float64
		for k, rr := range rows {
			s += rowR[rr] * vals[k]
		}
		dst[j] = s
	}
}

// ZeroCost returns a zero cost vector of the right length for SwapCost.
func (lp *LP) ZeroCost() []float64 { return make([]float64, len(lp.cost)) }

// SwapCost replaces the internal phase-2 cost vector (e.g. for a feasibility
// pump projection) and returns the previous one; objective() is unaffected.
func (lp *LP) SwapCost(c []float64) []float64 {
	old := lp.cost
	lp.cost = c
	return old
}

// SetBound overrides variable j's [lb,ub] for subsequent solves.
func (lp *LP) SetBound(j int, lb, ub float64) { lp.lb[j] = lb; lp.ub[j] = ub }

// NumCols returns the number of structural columns.
func (lp *LP) NumCols() int { return lp.n }

// phaseCost fills dst (length nTotal, zeroed) with the Phase-1 cost
// vector; returns false once all basic variables are feasible.
func (lp *LP) phaseCost(st *State, dst []float64) bool {
	inPhase1 := false
	for i := range lp.m {
		bv := st.basicOf[i]
		v := st.value[bv]
		switch {
		case v < lp.lb[bv]-feasTol:
			dst[bv] = -1
			inPhase1 = true
		case v > lp.ub[bv]+feasTol:
			dst[bv] = 1
			inPhase1 = true
		}
	}
	return inPhase1
}

// run recomputes the active cost vector fresh before every pivot, so a
// variable that becomes feasible mid-sequence stops influencing pricing.
func (lp *LP) run(st *State) Status {
	phase1Cost := make([]float64, lp.nTotal())
	y := make([]float64, lp.m)
	a := make([]float64, lp.m)
	degen := 0 // consecutive zero-step pivots; Bland's rule breaks cycles
	cleaned := false
	lp.xtol = xtolInit
	for iter := 0; ; iter++ {
		if iter > maxIter {
			return IterLimit
		}
		if iter%1024 == 0 && !lp.Deadline.IsZero() && time.Now().After(lp.Deadline) {
			return IterLimit
		}
		// EXPAND reset: snap the expansion back and restore exact values
		if lp.xtol += xtolStep; lp.xtol > feasTol {
			lp.refactorize(st)
			lp.recomputeBasics(st)
			lp.xtol = xtolInit
		}
		clear(phase1Cost)
		inPhase1 := lp.phaseCost(st, phase1Cost)
		cost := phase1Cost
		if !inPhase1 {
			cost = lp.cost
		}
		clear(y)
		duals(st, cost, lp.m, y)
		var q int
		var dir float64
		if degen > blandAfter {
			q, dir = lp.blandEntering(st, y, cost)
		} else {
			q, dir = lp.chooseEntering(st, y, cost)
		}
		if q < 0 {
			// EXPAND cleanup: restore exact values and re-price before
			// concluding, so the expansion never leaks into the answer
			if !cleaned {
				cleaned = true
				lp.refactorize(st)
				lp.recomputeBasics(st)
				lp.xtol = xtolInit
				continue
			}
			if inPhase1 {
				return Infeasible
			}
			return Optimal
		}
		clear(a)
		lp.alpha(st, q, a)
		t, row, isFlip := lp.ratioTest(st, a, q, dir, inPhase1, degen > blandAfter)
		if row < 0 && !isFlip {
			if inPhase1 {
				return Infeasible
			}
			return Unbounded
		}
		if inPhase1 {
			lp.Stats.Phase1++
		} else {
			lp.Stats.Phase2++
		}
		if t <= eps {
			degen++
		} else {
			degen = 0
		}
		lp.pivot(st, q, dir, a, t, row, isFlip)
		cleaned = false
	}
}

// blandEntering returns the first eligible entering variable (Bland's
// rule): slower per pivot but guaranteed to escape degenerate cycles.
func (lp *LP) blandEntering(st *State, y, cost []float64) (int, float64) {
	lp.Stats.Bland++
	for j := range lp.nTotal() {
		if st.status[j] == basic {
			continue
		}
		d := lp.reducedCost(y, cost, j)
		switch st.status[j] {
		case atLower:
			if -d > optEps {
				return j, 1
			}
		case atUpper:
			if d > optEps {
				return j, -1
			}
		case free:
			if math.Abs(d) > optEps {
				if d < 0 {
					return j, 1
				}
				return j, -1
			}
		}
	}
	return -1, 0
}

// chooseEntering scans a rotating column window per pivot above
// partialPricingThreshold, but always covers everything before declaring optimal.
func (lp *LP) chooseEntering(st *State, y, cost []float64) (q int, dir float64) {
	nt := lp.nTotal()
	if lp.pricingWindow <= 0 {
		return lp.scanEntering(st, y, cost, 0, nt)
	}
	start := lp.pricingCursor
	for scanned := 0; scanned < nt; {
		end := start + lp.pricingWindow
		if end > nt {
			end = nt
		}
		q, dir = lp.scanEntering(st, y, cost, start, end)
		scanned += end - start
		if q >= 0 {
			lp.pricingCursor = end % nt
			return q, dir
		}
		start = end % nt
	}
	lp.pricingCursor = 0
	return -1, 0
}

func (lp *LP) scanEntering(st *State, y, cost []float64, lo, hi int) (q int, dir float64) {
	best := optEps
	q = -1
	for j := lo; j < hi; j++ {
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
func (lp *LP) ratioTest(st *State, a []float64, q int, dir float64, phase1, bland bool) (t float64, leaveRow int, isFlip bool) {
	t = math.Inf(1)
	leaveRow = -1
	if lp.lb[q] > -inf && lp.ub[q] < inf {
		t = lp.ub[q] - lp.lb[q]
		isFlip = true
	}
	for i := range lp.m {
		if math.Abs(a[i]) < eps {
			continue
		}
		bv := st.basicOf[i]
		delta := -a[i] * dir // rate of change of basic var i per unit t
		v := st.value[bv]
		lb, ub := lp.lb[bv], lp.ub[bv]
		wlb, wub := lb-lp.xtol, ub+lp.xtol // EXPAND working bounds

		var limit float64
		ok := false
		if phase1 {
			switch {
			case v < lb-feasTol: // infeasible low: only care about moving up to lb
				if delta > eps {
					limit = (lb - v) / delta
					ok = true
				}
			case v > ub+feasTol: // infeasible high: only care about moving down to ub
				if delta < -eps {
					limit = (ub - v) / delta
					ok = true
				}
			default: // feasible: normal both-direction limits
				if delta > eps && ub < inf {
					limit = (wub - v) / delta
					ok = true
				} else if delta < -eps && lb > -inf {
					limit = (wlb - v) / delta
					ok = true
				}
			}
		} else {
			if delta > eps && ub < inf {
				limit = (wub - v) / delta
				ok = true
			} else if delta < -eps && lb > -inf {
				limit = (wlb - v) / delta
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
		} else if bland && !isFlip && leaveRow >= 0 && limit <= t+eps &&
			st.basicOf[i] < st.basicOf[leaveRow] {
			// Bland's leaving rule on ties: smallest basic variable index.
			// Anti-cycling needs Bland for BOTH entering and leaving.
			t = math.Min(t, limit)
			leaveRow = i
		}
	}
	if leaveRow < 0 && !isFlip {
		return t, -1, false
	}
	// EXPAND minimum step: the overshoot stays inside the next iteration's
	// working bounds (xtol grows by xtolStep), so no seesaw re-flagging
	if leaveRow >= 0 && !isFlip {
		if minT := xtolStep / math.Abs(a[leaveRow]); t < minT {
			t = minT
		}
	}
	return t, leaveRow, isFlip
}

func (lp *LP) pivot(st *State, q int, dir float64, a []float64, t float64, leaveRow int, isFlip bool) {
	if t < 0 {
		t = 0
	}
	// update all basic values
	for i := range lp.m {
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

	// product-form update: alpha is exactly the eta for this basis change
	var idx []int32
	var val []float64
	for i, v := range a {
		if math.Abs(v) > etaDropTol {
			idx = append(idx, int32(i))
			val = append(val, v)
		}
	}
	st.etas = append(st.etas, &eta{r: leaveRow, idx: idx, val: val, ar: a[leaveRow]})
	st.basicOf[leaveRow] = q
	st.status[q] = basic
	if len(st.etas) > maxEtas {
		lp.refactorize(st)
		lp.recomputeBasics(st)
	}
}

// Solution extracts primal values, row activity, reduced costs and row
// duals (prices) from a solved state.
func (lp *LP) Solution(st *State) (x, rowActivity, reducedCost, rowPrice []float64) {
	x = append([]float64(nil), st.value[:lp.n]...)
	rowActivity = make([]float64, lp.m)
	for i := range lp.m {
		rowActivity[i] = -st.value[lp.n+i] * -1 // logical var value == row activity (y_i = Ax_i)
	}
	y := make([]float64, lp.m)
	duals(st, lp.cost, lp.m, y)
	rowPrice = make([]float64, lp.m)
	for i := range rowPrice {
		rowPrice[i] = y[i] * lp.objSign
	}
	reducedCost = make([]float64, lp.n)
	for j := range lp.n {
		reducedCost[j] = lp.reducedCost(y, lp.cost, j) * lp.objSign
	}
	return
}
