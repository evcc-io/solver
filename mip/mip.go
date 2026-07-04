// Package mip implements branch-and-bound over a simplex LP,
// using the integrality and SOS metadata from a problem.Problem.
package mip

import (
	"container/heap"
	"math"
	"time"

	"fmt"
	"os"

	"cbcgo/problem"
	"cbcgo/simplex"
)

type Status int

const (
	Optimal Status = iota
	Infeasible
	Unbounded
	Stopped
)

const intTol = 1e-6

// debugf prints heuristic diagnostics when SOLVER_DEBUG is set.
func debugf(format string, args ...any) {
	if os.Getenv("SOLVER_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "debug: "+format+"\n", args...)
	}
}

type boundOverride struct {
	idx    int
	lb, ub float64
}

type node struct {
	overrides   []boundOverride
	parentState *simplex.State // warm-start basis; nil at the root (cold start)
	bound       float64        // valid lower bound (internal, minimized scale)
	depth       int
}

type nodeHeap []*node

func (h nodeHeap) Len() int           { return len(h) }
func (h nodeHeap) Less(i, j int) bool { return h[i].bound < h[j].bound }
func (h nodeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *nodeHeap) Push(x any)        { *h = append(*h, x.(*node)) }
func (h *nodeHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

type Limits struct {
	MaxTime  time.Duration // <=0 means unlimited
	MaxNodes int           // <=0 means unlimited
	GapRel   float64       // relative MIP gap, e.g. 0.0001; <=0 means 0 (prove optimal)
	GapAbs   float64       // absolute MIP gap; <=0 means 0
}

type Result struct {
	Status                   Status
	HasIncumbent             bool
	X                        []float64 // structural solution values
	RowActivity, ReducedCost []float64
	RowPrice                 []float64
	Obj                      float64 // reported (user-sense) objective
	NodeCount                int
}

type Model struct {
	P         *problem.Problem
	LP        *simplex.LP
	Limits    Limits
	MIPStart  []float64       // optional structural start point; ints get fixed
	live      []boundOverride // bounds currently applied to LP; see solveNode
	rcTouched []int           // columns tightened by reducedCostFix
}

func New(p *problem.Problem) *Model {
	return &Model{P: p, LP: simplex.Build(p), Limits: Limits{GapRel: 1e-9, GapAbs: 1e-9}}
}

// SolveRelaxation solves just the LP relaxation, ignoring integrality and
// SOS constraints entirely (the "-initialSolve" / mip=False CLI contract).
func SolveRelaxation(p *problem.Problem) Result {
	lp := simplex.Build(p)
	status, st, _ := lp.ColdSolve()
	res := Result{}
	switch status {
	case simplex.Infeasible:
		res.Status = Infeasible
	case simplex.Unbounded:
		res.Status = Unbounded
	case simplex.IterLimit:
		res.Status = Stopped
	default:
		res.Status = Optimal
		res.HasIncumbent = true
		res.X, res.RowActivity, res.ReducedCost, res.RowPrice = lp.Solution(st)
		res.Obj = lp.InternalObjective(st) * p.ObjSense
	}
	return res
}

func (m *Model) Solve() Result {
	m.live = nil
	deadline := time.Time{}
	if m.Limits.MaxTime > 0 {
		deadline = time.Now().Add(m.Limits.MaxTime)
		m.LP.Deadline = deadline // abort long LP solves at the deadline too
	}

	pq := &nodeHeap{{bound: math.Inf(-1)}}
	heap.Init(pq)

	bestInternal := math.Inf(1)
	var bestX, bestRowAct, bestRC, bestPrice []float64
	hasIncumbent := false
	nodeCount := 0
	rootDone := false
	rootStatus := simplex.Optimal
	stopped := false
	proofLost := false       // an unresolved (IterLimit) node was pruned
	remaining := math.Inf(1) // best bound among subtrees dropped mid-dive

	var rootObj float64 // root LP data for reduced-cost fixing
	var rootX, rootRC []float64
	rcFixed := false
	diveTries := 0
	newIncumbent := func(obj float64, x, rowAct, rc, price []float64) {
		bestInternal = obj
		bestX, bestRowAct, bestRC, bestPrice = x, rowAct, rc, price
		hasIncumbent = true
		if !rcFixed && rootX != nil {
			rcFixed = true
			m.reducedCostFix(rootX, rootRC, rootObj, bestInternal)
		}
	}

	for pq.Len() > 0 {
		if m.Limits.MaxNodes > 0 && nodeCount >= m.Limits.MaxNodes {
			stopped = true
			break
		}
		nd := heap.Pop(pq).(*node)
		if hasIncumbent && !m.improves(nd.bound, bestInternal) {
			continue
		}

		// plunge: dive depth-first, keeping the rounding-nearer child.
		// backtrack is the current level's untried sibling (try before giving up).
		var backtrack *node
		for nd != nil {
			if !deadline.IsZero() && time.Now().After(deadline) {
				stopped = true
				remaining = math.Min(remaining, nd.bound)
				break
			}
			status, x, rowAct, rc, price, obj, endState := m.solveNode(nd)
			nodeCount++
			if !rootDone {
				rootDone = true
				rootStatus = status
				if status == simplex.Optimal {
					rootObj, rootX, rootRC = obj, x, rc
				}
			}
			if status != simplex.Optimal {
				// an IterLimit node stays unresolved: prune it and search on,
				// but optimality can no longer be proven.
				if status == simplex.IterLimit {
					if !deadline.IsZero() && time.Now().After(deadline) {
						stopped = true
					} else {
						proofLost = true
					}
					remaining = math.Min(remaining, nd.bound)
					break
				}
				nd, backtrack = backtrack, nil
				continue
			}
			if hasIncumbent && !m.improves(obj, bestInternal) {
				nd, backtrack = backtrack, nil
				continue
			}
			// this level solved: the previous level's sibling goes to the heap
			if backtrack != nil {
				heap.Push(pq, backtrack)
				backtrack = nil
			}

			branchCol, sosIdx, sosSplit := m.findBranch(x)
			if branchCol < 0 && sosIdx < 0 {
				newIncumbent(obj, x, rowAct, rc, price)
				break
			}

			var near, far *node
			if branchCol >= 0 {
				lb, ub := m.LP.Bound(branchCol)
				floorV := math.Floor(x[branchCol] + 1e-7)
				ceilV := math.Ceil(x[branchCol] - 1e-7)
				floorChild := childNode(nd, endState, boundOverride{branchCol, lb, floorV}, obj)
				ceilChild := childNode(nd, endState, boundOverride{branchCol, ceilV, ub}, obj)
				if x[branchCol]-floorV >= 0.5 {
					near, far = ceilChild, floorChild
				} else {
					near, far = floorChild, ceilChild
				}
			} else {
				members := m.P.SOSs[sosIdx].Idx
				var loOv, hiOv []boundOverride
				for k, idx := range members {
					if k > sosSplit {
						loOv = append(loOv, boundOverride{idx, 0, 0})
					}
					if k <= sosSplit {
						hiOv = append(hiOv, boundOverride{idx, 0, 0})
					}
				}
				near = childNodeMulti(nd, endState, loOv, obj)
				far = childNodeMulti(nd, endState, hiOv, obj)
			}
			// no incumbent yet: caller's MIP start first, then heuristics
			// (children above already captured their bounds)
			if !hasIncumbent && nodeCount == 1 && len(m.MIPStart) == len(m.P.Cols) {
				if hObj, hx, hAct, hRC, hPrice, ok := m.completePoint(nd, m.MIPStart); ok {
					debugf("mipstart: accepted obj=%g", hObj)
					newIncumbent(hObj, hx, hAct, hRC, hPrice)
				} else {
					debugf("mipstart: rejected")
				}
			}
			if !hasIncumbent && diveTries < 8 && branchCol >= 0 && (nodeCount == 1 || nodeCount%256 == 0) {
				diveTries++
				hObj, hx, hAct, hRC, hPrice, ok := m.feasibilityPump(nd, x, endState)
				if !ok {
					hObj, hx, hAct, hRC, hPrice, ok = m.diveForIncumbent(nd, x, endState)
				}
				if ok {
					newIncumbent(hObj, hx, hAct, hRC, hPrice)
				}
			}
			nd, backtrack = near, far
		}
		if backtrack != nil {
			heap.Push(pq, backtrack)
		}
		if stopped {
			break
		}
	}
	for _, ov := range m.live {
		m.LP.SetBound(ov.idx, m.P.Cols[ov.idx].LB, m.P.Cols[ov.idx].UB)
	}
	m.live = nil

	// the incumbent is proven optimal (within gap) if no unexplored subtree
	// — on the heap or dropped mid-dive — has a bound that could still beat it
	if pq.Len() > 0 {
		remaining = math.Min(remaining, (*pq)[0].bound)
	}
	proven := hasIncumbent && !m.improves(remaining, bestInternal)

	res := Result{HasIncumbent: hasIncumbent, NodeCount: nodeCount}
	switch {
	case !rootDone:
		res.Status = Infeasible
	case rootStatus == simplex.Unbounded:
		res.Status = Unbounded
	case proven:
		res.Status = Optimal
	case stopped || proofLost:
		res.Status = Stopped
	case hasIncumbent:
		res.Status = Optimal
	default:
		res.Status = Infeasible
	}
	if hasIncumbent {
		res.X = bestX
		res.RowActivity, res.ReducedCost, res.RowPrice = bestRowAct, bestRC, bestPrice
		res.Obj = bestInternal * m.P.ObjSense
	}
	return res
}

// improves reports whether an internal-scale bound/objective candidate is
// strictly better than the incumbent, honoring the configured MIP gap.
func (m *Model) improves(candidate, incumbent float64) bool {
	gap := m.Limits.GapAbs
	rel := m.Limits.GapRel * math.Abs(incumbent)
	if rel > gap {
		gap = rel
	}
	return candidate < incumbent-gap-1e-9
}

func childNode(parent *node, parentState *simplex.State, ov boundOverride, bound float64) *node {
	ovs := make([]boundOverride, len(parent.overrides), len(parent.overrides)+1)
	copy(ovs, parent.overrides)
	ovs = append(ovs, ov)
	// bounds only tighten down a branch; max() stops numerical drift from
	// chained warm starts leaking below the true LP bound over deep dives
	return &node{overrides: ovs, parentState: parentState, bound: math.Max(parent.bound, bound), depth: parent.depth + 1}
}

func childNodeMulti(parent *node, parentState *simplex.State, extra []boundOverride, bound float64) *node {
	ovs := make([]boundOverride, len(parent.overrides), len(parent.overrides)+len(extra))
	copy(ovs, parent.overrides)
	ovs = append(ovs, extra...)
	return &node{overrides: ovs, parentState: parentState, bound: math.Max(parent.bound, bound), depth: parent.depth + 1}
}

// solveNode diffs m.live against nd.overrides (touching only changed
// columns), solves, and leaves bounds live as nd.overrides for next time.
func (m *Model) solveNode(nd *node) (status simplex.Status, x, rowAct, rc, price []float64, internalObj float64, endState *simplex.State) {
	common := 0
	for common < len(m.live) && common < len(nd.overrides) && m.live[common] == nd.overrides[common] {
		common++
	}
	for i := len(m.live) - 1; i >= common; i-- {
		idx := m.live[i].idx
		// a column overridden twice down a branch reverts to the still-active
		// prefix override, not to the original problem bounds
		lb, ub := m.P.Cols[idx].LB, m.P.Cols[idx].UB
		for j := common - 1; j >= 0; j-- {
			if m.live[j].idx == idx {
				lb, ub = m.live[j].lb, m.live[j].ub
				break
			}
		}
		m.LP.SetBound(idx, lb, ub)
	}
	for i := common; i < len(nd.overrides); i++ {
		ov := nd.overrides[i]
		m.LP.SetBound(ov.idx, ov.lb, ov.ub)
	}
	m.live = nd.overrides

	// the warm start needs every column whose bounds may differ from the
	// PARENT state's assumptions, not just the diff vs the last-solved node
	touchedIdx := make([]int, 0, len(nd.overrides)+len(m.rcTouched))
	for _, ov := range nd.overrides {
		touchedIdx = append(touchedIdx, ov.idx)
	}
	touchedIdx = append(touchedIdx, m.rcTouched...)

	if nd.parentState != nil {
		status, endState, _ = m.LP.WarmSolve(nd.parentState.Clone(), touchedIdx)
		// a degenerate warm start can stall at the iteration cap; a fresh
		// basis usually solves the same node quickly
		if status == simplex.IterLimit && (m.LP.Deadline.IsZero() || time.Now().Before(m.LP.Deadline)) {
			status, endState, _ = m.LP.ColdSolve()
		}
	} else {
		status, endState, _ = m.LP.ColdSolve()
	}
	if status == simplex.Optimal {
		x, rowAct, rc, price = m.LP.Solution(endState)
		internalObj = m.LP.InternalObjective(endState)
	}
	return status, x, rowAct, rc, price, internalObj, endState
}

// feasibilityPump alternates rounding and L1-projection LP solves (CBC's
// FPump heuristic) to manufacture a first incumbent; fails on cycling.
func (m *Model) feasibilityPump(nd *node, x []float64, endState *simplex.State) (obj float64, dx, rowAct, rc, price []float64, ok bool) {
	fail := func() (float64, []float64, []float64, []float64, []float64, bool) {
		return 0, nil, nil, nil, nil, false
	}
	fpCost := m.LP.ZeroCost()
	orig := m.LP.SwapCost(fpCost)
	defer func() { m.LP.SwapCost(orig) }()

	st := endState.Clone()
	curX := x
	var prev []float64
	for iter := 0; iter < 30; iter++ {
		if !m.LP.Deadline.IsZero() && time.Now().After(m.LP.Deadline) {
			return fail()
		}
		// round the current LP point and build the L1 pull-back objective
		integral := true
		rounded := make([]float64, 0, 64)
		for j, c := range m.P.Cols {
			if !c.Integer {
				continue
			}
			lb, ub := m.LP.Bound(j)
			v := math.Max(lb, math.Min(ub, math.Round(curX[j])))
			if frac := math.Abs(curX[j] - v); frac > intTol {
				integral = false
			}
			switch {
			case v <= lb+intTol:
				fpCost[j] = 1
			case v >= ub-intTol:
				fpCost[j] = -1
			default:
				fpCost[j] = 0
			}
			rounded = append(rounded, v)
		}
		if integral {
			debugf("fp: integral at iter %d", iter)
			// polish: real objective, all integers fixed at their values
			m.LP.SwapCost(orig)
			return m.completePoint(nd, curX)
		}
		if prev != nil && slicesEqual(prev, rounded) {
			debugf("fp: cycle at iter %d", iter)
			return fail() // cycling
		}
		prev = rounded

		status, nst, _ := m.LP.WarmSolve(st, nil)
		if status != simplex.Optimal {
			debugf("fp: projection LP status=%d at iter %d", status, iter)
			return fail()
		}
		st = nst
		curX, _, _, _ = m.LP.Solution(st)
	}
	return fail()
}

func slicesEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// completePoint fixes every integer column at its rounded value from point
// (a structural vector) and LP-solves the continuous completion.
func (m *Model) completePoint(nd *node, point []float64) (obj float64, x, rowAct, rc, price []float64, ok bool) {
	ovs := make([]boundOverride, len(nd.overrides), len(nd.overrides)+len(m.P.Cols))
	copy(ovs, nd.overrides)
	for j, c := range m.P.Cols {
		if !c.Integer {
			continue
		}
		lb, ub := m.LP.Bound(j)
		v := math.Max(lb, math.Min(ub, math.Round(point[j])))
		ovs = append(ovs, boundOverride{j, v, v})
	}
	child := &node{overrides: ovs, bound: nd.bound, depth: nd.depth + 1}
	status, cx, cAct, cRC, cPrice, cObj, _ := m.solveNode(child)
	if status != simplex.Optimal {
		return 0, nil, nil, nil, nil, false
	}
	return cObj, cx, cAct, cRC, cPrice, true
}

// diveForIncumbent fixes the least-fractional column at its nearest integer
// and re-solves warm until integral (CBC diving); gives up when infeasible.
func (m *Model) diveForIncumbent(nd *node, x []float64, endState *simplex.State) (obj float64, dx, rowAct, rc, price []float64, ok bool) {
	cur, curX, curState := nd, x, endState
	for {
		if !m.LP.Deadline.IsZero() && time.Now().After(m.LP.Deadline) {
			return 0, nil, nil, nil, nil, false
		}
		col, sosIdx, _ := m.findBranch(curX)
		if sosIdx >= 0 {
			return 0, nil, nil, nil, nil, false
		}
		if col < 0 {
			dx, rowAct, rc, price = m.LP.Solution(curState)
			return m.LP.InternalObjective(curState), dx, rowAct, rc, price, true
		}
		lb, ub := m.LP.Bound(col)
		v := math.Max(lb, math.Min(ub, math.Round(curX[col])))
		// nearest rounding first; if that goes infeasible, the other side
		alt, hasAlt := 2*math.Floor(curX[col])+1-v, true
		if alt < lb || alt > ub {
			hasAlt = false
		}
		var child *node
		var cx []float64
		var cState *simplex.State
		for {
			ovs := make([]boundOverride, len(cur.overrides)+1)
			copy(ovs, cur.overrides)
			ovs[len(ovs)-1] = boundOverride{col, v, v}
			child = &node{overrides: ovs, parentState: curState, bound: cur.bound, depth: cur.depth + 1}
			status, x, _, _, _, _, cs := m.solveNode(child)
			if status == simplex.Optimal {
				cx, cState = x, cs
				break
			}
			if !hasAlt {
				debugf("dive: dead end at depth %d col %d", len(cur.overrides)-len(nd.overrides), col)
				return 0, nil, nil, nil, nil, false
			}
			v, hasAlt = alt, false
		}
		cur, curX, curState = child, cx, cState
	}
}

// reducedCostFix tightens integer bounds the root reduced costs prove cannot
// beat the incumbent by more than the gap (CBC-style); runs on first incumbent.
func (m *Model) reducedCostFix(rootX, rootRC []float64, rootObj, best float64) {
	gap := math.Max(m.Limits.GapAbs, m.Limits.GapRel*math.Abs(best))
	slack := best - gap - rootObj
	if slack < 0 {
		return
	}
	live := make(map[int]bool, len(m.live))
	for _, ov := range m.live {
		live[ov.idx] = true
	}
	for j := range m.P.Cols {
		c := &m.P.Cols[j]
		if !c.Integer || j >= len(rootRC) {
			continue
		}
		rcInt := rootRC[j] * m.P.ObjSense // undo user-sign back to internal
		lb, ub := c.LB, c.UB
		switch {
		case rcInt > 1e-9:
			ub = math.Min(ub, math.Floor(rootX[j]+slack/rcInt+intTol))
		case rcInt < -1e-9:
			lb = math.Max(lb, math.Ceil(rootX[j]-slack/-rcInt-intTol))
		default:
			continue
		}
		if (lb == c.LB && ub == c.UB) || ub < lb {
			continue
		}
		c.LB, c.UB = lb, ub
		m.rcTouched = append(m.rcTouched, j)
		// overridden columns pick the new bounds up on revert; the rest now
		if !live[j] {
			m.LP.SetBound(j, lb, ub)
		}
	}
}

// findBranch picks the least-fractional integer column (cheap dives: rounding
// near-decided vars rarely hurts the LP), or the first violated SOS constraint.
func (m *Model) findBranch(x []float64) (col, sosIdx, sosSplit int) {
	col, sosIdx = -1, -1
	best := math.Inf(1)
	for j, c := range m.P.Cols {
		if !c.Integer {
			continue
		}
		v := x[j]
		frac := v - math.Floor(v)
		dist := math.Min(frac, 1-frac)
		if dist > intTol && dist < best {
			best, col = dist, j
		}
	}
	if col >= 0 {
		return col, -1, 0
	}
	for si, s := range m.P.SOSs {
		nonzero := 0
		for _, idx := range s.Idx {
			if math.Abs(x[idx]) > intTol {
				nonzero++
			}
		}
		limit := 1
		if s.Type == 2 {
			limit = 2
		}
		if nonzero <= limit {
			continue
		}
		return -1, si, sosSplitPoint(s.Idx, x)
	}
	return -1, -1, 0
}

func sosSplitPoint(members []int, x []float64) int {
	total := 0.0
	for _, idx := range members {
		total += math.Abs(x[idx])
	}
	if total == 0 {
		return len(members) / 2
	}
	half := total / 2
	cum := 0.0
	for r, idx := range members {
		cum += math.Abs(x[idx])
		if cum >= half {
			if r >= len(members)-1 {
				r = len(members) - 2
			}
			return r
		}
	}
	return len(members) - 2
}
