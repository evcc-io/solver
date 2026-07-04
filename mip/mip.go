// Package mip implements branch-and-bound over a simplex LP,
// using the integrality and SOS metadata from a problem.Problem.
package mip

import (
	"container/heap"
	"math"
	"sort"
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

	// pseudocost bookkeeping: which column/direction/fraction created this
	// child; -1 when not a branching child
	brCol  int
	brUp   bool
	brFrac float64
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
	P             *problem.Problem
	LP            *simplex.LP
	Limits        Limits
	MIPStart      []float64       // optional structural start point; ints get fixed
	SkipProbing   bool            // restart passes re-derive identical probe facts
	live          []boundOverride // bounds currently applied to LP; see solveNode
	rcTouched     []int           // columns tightened by reducedCostFix
	bestXSnapshot []float64       // incumbent X for the RINS neighborhood

	// scratch for node-level bound propagation (propagatedChild)
	propLB, propUB, propL0, propU0 []float64

	// per-column pseudocosts: observed bound gain per unit fraction
	psUp, psDn   []float64
	psUpN, psDnN []int
}

// psRecord accumulates one observed branching gain for column j.
func (m *Model) psRecord(j int, up bool, frac, gain float64) {
	if m.psUp == nil {
		n := len(m.P.Cols)
		m.psUp, m.psDn = make([]float64, n), make([]float64, n)
		m.psUpN, m.psDnN = make([]int, n), make([]int, n)
	}
	if gain < 0 {
		gain = 0
	}
	if up {
		if d := 1 - frac; d > intTol {
			m.psUp[j] += gain / d
			m.psUpN[j]++
		}
	} else if frac > intTol {
		m.psDn[j] += gain / frac
		m.psDnN[j]++
	}
}

// psSelect picks the fractional column with the best pseudocost score
// (min of both directions' estimated gains); -1 if none is reliable.
func (m *Model) psSelect(x []float64) int {
	if m.psUp == nil {
		return -1
	}
	best, col := 0.0, -1
	for j, c := range m.P.Cols {
		if !c.Integer {
			continue
		}
		frac := x[j] - math.Floor(x[j])
		if math.Min(frac, 1-frac) <= intTol {
			continue
		}
		if m.psUpN[j] == 0 || m.psDnN[j] == 0 {
			continue
		}
		up := m.psUp[j] / float64(m.psUpN[j]) * (1 - frac)
		dn := m.psDn[j] / float64(m.psDnN[j]) * frac
		if score := math.Min(up, dn); score > best {
			best, col = score, j
		}
	}
	return col
}

func New(p *problem.Problem) *Model {
	presolve(p)
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

	// time-boxed binary probing (CglProbing), then rebuild the LP on the
	// tightened problem
	if len(m.P.SOSs) == 0 && !m.SkipProbing {
		probeDeadline := time.Time{}
		if m.Limits.MaxTime > 0 {
			probeDeadline = time.Now().Add(m.Limits.MaxTime / 6)
		}
		probe(m.P, probeDeadline)
		presolve(m.P)
		m.LP = simplex.Build(m.P)
		m.LP.Deadline = deadline
	}

	// seed the incumbent before cutting: the caller's MIP start plus root
	// reduced-cost fixing shrinks the problem the cuts then work on
	var startObj float64
	var startX, startAct, startRC, startPrice []float64
	haveStart := false
	if len(m.MIPStart) == len(m.P.Cols) && len(m.P.SOSs) == 0 {
		if status, st, _ := m.LP.ColdSolve(); status == simplex.Optimal {
			rx, _, rrc, _ := m.LP.Solution(st)
			robj := m.LP.InternalObjective(st)
			if obj, x, act, rc, price, ok := m.completePoint(&node{brCol: -1}, m.MIPStart); ok {
				if pObj, px, pAct, pRC, pPrice, better := m.polishIncumbent(&node{brCol: -1}, obj, x, deadline); better {
					debugf("polish: mipstart %g -> %g", obj, pObj)
					obj, x, act, rc, price = pObj, px, pAct, pRC, pPrice
				}
				startObj, startX, startAct, startRC, startPrice = obj, x, act, rc, price
				haveStart = true
				debugf("mipstart: pre-cut incumbent obj=%g", obj)
				m.reducedCostFix(rx, rrc, robj, obj)
			}
			for _, ov := range m.live {
				m.LP.SetBound(ov.idx, m.P.Cols[ov.idx].LB, m.P.Cols[ov.idx].UB)
			}
			m.live = nil
		}
	}

	// GMI cut rounds tighten the root while its bound still moves; capped
	// at a fifth of the budget so the tree always gets its time
	origRows := len(m.P.Rows)
	// restart passes inherit the first pass's cuts: go straight to the tree
	if len(m.P.SOSs) == 0 && !m.SkipProbing {
		cutDeadline := deadline
		if m.Limits.MaxTime > 0 {
			cutDeadline = time.Now().Add(m.Limits.MaxTime / 5)
			m.LP.Deadline = cutDeadline
		}
		prevObj := math.Inf(1)
		flat := 0       // bound improvement can pause a round and resume
		lastBatch := -1 // first row index of the previous round's cuts
		poisoned := 0
		for round := range maxCutRound {
			if !cutDeadline.IsZero() && time.Now().After(cutDeadline) {
				break
			}
			status, st, _ := m.LP.ColdSolve()
			if status != simplex.Optimal {
				// the last batch poisoned the LP (degenerate grind):
				// retract it and continue with the last good relaxation
				if lastBatch >= 0 {
					truncateRows(m.P, lastBatch)
					m.LP = simplex.Build(m.P)
					m.LP.Deadline = cutDeadline
					debugf("cuts: retracted poison batch, back to %d rows", lastBatch)
					lastBatch = -1
					if poisoned++; poisoned <= 2 {
						continue // retry from the last good relaxation
					}
				}
				m.LP.Deadline = deadline
				break
			}
			obj := m.LP.InternalObjective(st)
			if round > 0 && obj-prevObj < math.Max(1e-7, 1e-9*math.Abs(obj)) {
				if flat++; flat >= 2 {
					debugf("cuts: bound stalled at %g after %d rounds", obj, round)
					break
				}
			} else {
				flat = 0
			}
			prevObj = obj
			x, _, _, _ := m.LP.Solution(st)
			if col, sos, _ := m.findBranch(x); col < 0 && sos < 0 {
				break // root already integral
			}
			lastBatch = len(m.P.Rows)
			gmi := m.gomoryCuts(st)
			// probing cuts pay off on big fixed-charge instances; on small
			// ones they poison node re-solves that branching closes anyway
			prb := 0
			if m.LP.NumRows() > 1500 {
				prb = m.probingCuts(x, cutDeadline)
			}
			added := gmi + prb
			debugf("cuts: round %d added %d rows (gmi %d, probing %d, bound %g)", round, added, gmi, prb, obj)
			if added == 0 {
				break
			}
			stats := m.LP.Stats
			m.LP = simplex.Build(m.P)
			m.LP.Stats = stats // carry pivot counters across rebuilds
			m.LP.Deadline = cutDeadline
		}
		m.LP.Deadline = deadline
	}

	// keep only cuts tight at the root: slack rows bloat every node re-solve
	if len(m.P.Rows) > origRows {
		if status, st, _ := m.LP.ColdSolve(); status == simplex.Optimal {
			_, rowAct, _, _ := m.LP.Solution(st)
			if dropped := dropSlackCuts(m.P, origRows, rowAct); dropped > 0 {
				stats := m.LP.Stats
				m.LP = simplex.Build(m.P)
				m.LP.Stats = stats
				m.LP.Deadline = deadline
				debugf("cuts: dropped %d slack rows, kept %d", dropped, len(m.P.Rows)-origRows)
			}
		}
	}

	pq := &nodeHeap{{bound: math.Inf(-1), brCol: -1}}
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
	diveTries := 0
	newIncumbent := func(obj float64, x, rowAct, rc, price []float64) {
		bestInternal = obj
		bestX, bestRowAct, bestRC, bestPrice = x, rowAct, rc, price
		m.bestXSnapshot = x
		hasIncumbent = true
		// re-fix on every improvement: tighter cutoffs fix more columns
		if rootX != nil {
			m.reducedCostFix(rootX, rootRC, rootObj, bestInternal)
		}
	}

	if haveStart {
		newIncumbent(startObj, startX, startAct, startRC, startPrice)
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
			if nd.brCol >= 0 {
				m.psRecord(nd.brCol, nd.brUp, nd.brFrac, obj-nd.bound)
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
			// shallow nodes pick the branch variable by probing (strong
			// branching); deeper plunges use pseudocosts once seeded.
			// Probes cost ~a node solve each: big problems get few
			sbDepth := 16
			if m.LP.NumRows() > 1500 {
				sbDepth = 2
			}
			if branchCol >= 0 {
				if nd.depth < sbDepth {
					if sb := m.strongBranch(nd, x, endState, obj); sb >= 0 {
						branchCol = sb
					}
				} else if ps := m.psSelect(x); ps >= 0 {
					branchCol = ps
				}
			}

			var near, far *node
			if branchCol >= 0 {
				lb, ub := m.nodeBound(nd, branchCol)
				floorV := math.Floor(x[branchCol] + 1e-7)
				ceilV := math.Ceil(x[branchCol] - 1e-7)
				floorChild := m.propagatedChild(nd, endState, boundOverride{branchCol, lb, floorV}, obj)
				ceilChild := m.propagatedChild(nd, endState, boundOverride{branchCol, ceilV, ub}, obj)
				frac := x[branchCol] - floorV
				if floorChild != nil {
					floorChild.brCol, floorChild.brUp, floorChild.brFrac = branchCol, false, frac
				}
				if ceilChild != nil {
					ceilChild.brCol, ceilChild.brUp, ceilChild.brFrac = branchCol, true, frac
				}
				if x[branchCol]-floorV >= 0.5 {
					near, far = ceilChild, floorChild
				} else {
					near, far = floorChild, ceilChild
				}
				if near == nil {
					near, far = far, nil
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
			if diveTries < 8 && branchCol >= 0 && (nodeCount == 1 || nodeCount%256 == 0) {
				diveTries++
				// box the heuristic burst: dives must never eat the tree's time.
				// The root burst gets more — success there ends the whole solve
				savedDL := m.LP.Deadline
				if m.Limits.MaxTime > 0 {
					div := time.Duration(8)
					if nodeCount == 1 {
						div = 3
					}
					if hb := time.Now().Add(m.Limits.MaxTime / div); savedDL.IsZero() || hb.Before(savedDL) {
						m.LP.Deadline = hb
					}
				}
				cutoff := math.Inf(1)
				if hasIncumbent {
					cutoff = bestInternal
				}
				hObj, hx, hAct, hRC, hPrice, ok := m.faceWalk(nd, x, endState, obj, cutoff)
				if ok {
					debugf("facewalk: integral vertex on face obj=%g", hObj)
				} else {
					hObj, hx, hAct, hRC, hPrice, ok = m.rensImprove(nd, x)
				}
				if !ok {
					hObj, hx, hAct, hRC, hPrice, ok = m.feasibilityPump(nd, x, endState)
					if !ok {
						hObj, hx, hAct, hRC, hPrice, ok = m.diveForIncumbent(nd, x, endState)
					}
				}
				m.LP.Deadline = savedDL
				if ok && (!hasIncumbent || m.improves(hObj, bestInternal)) {
					debugf("dive: incumbent %g -> %g", bestInternal, hObj)
					newIncumbent(hObj, hx, hAct, hRC, hPrice)
				}
			}
			// RINS-lite: fix integers where incumbent and node LP agree,
			// LP-complete the rest; accept only strict improvements
			if hasIncumbent && nodeCount%64 == 0 {
				if hObj, hx, hAct, hRC, hPrice, ok := m.rinsImprove(nd, x); ok && m.improves(hObj, bestInternal) {
					debugf("rins: improved %g -> %g", bestInternal, hObj)
					if pObj, px, pAct, pRC, pPrice, better := m.polishIncumbent(nd, hObj, hx, deadline); better {
						debugf("polish: rins %g -> %g", hObj, pObj)
						hObj, hx, hAct, hRC, hPrice = pObj, px, pAct, pRC, pPrice
					}
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
	debugf("mip exit: nodes=%d best=%g remaining=%g proven=%v stopped=%v proofLost=%v heap=%d",
		nodeCount, bestInternal, remaining, proven, stopped, proofLost, pq.Len())

	res := Result{HasIncumbent: hasIncumbent, NodeCount: nodeCount}
	switch {
	case !rootDone && stopped:
		res.Status = Stopped // out of time before the root was even solved
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
		// cut rows are internal: report only the caller's original rows
		if len(res.RowActivity) > origRows {
			res.RowActivity = res.RowActivity[:origRows]
		}
		if len(res.RowPrice) > origRows {
			res.RowPrice = res.RowPrice[:origRows]
		}
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
	return &node{overrides: ovs, parentState: parentState, bound: math.Max(parent.bound, bound), depth: parent.depth + 1, brCol: -1}
}

// propagatedChild builds a branch child, propagating the branch bound through
// the rows: nil = pruned infeasible; implied integer fixings become overrides
func (m *Model) propagatedChild(nd *node, st *simplex.State, ov boundOverride, bound float64) *node {
	n := len(m.P.Cols)
	if cap(m.propLB) < n {
		m.propLB, m.propUB = make([]float64, n), make([]float64, n)
		m.propL0, m.propU0 = make([]float64, n), make([]float64, n)
	}
	lb, ub := m.propLB[:n], m.propUB[:n]
	for j := range n {
		lb[j], ub[j] = m.P.Cols[j].LB, m.P.Cols[j].UB
	}
	for _, o := range nd.overrides {
		lb[o.idx], ub[o.idx] = o.lb, o.ub
	}
	lb[ov.idx], ub[ov.idx] = ov.lb, ov.ub
	l0, u0 := m.propL0[:n], m.propU0[:n]
	copy(l0, lb)
	copy(u0, ub)
	if !propagate(m.P, lb, ub) {
		return nil
	}
	extra := make([]boundOverride, 0, 8)
	extra = append(extra, ov)
	for j := range n {
		if !m.P.Cols[j].Integer || j == ov.idx {
			continue
		}
		if lb[j] > l0[j]+1e-9 || ub[j] < u0[j]-1e-9 {
			extra = append(extra, boundOverride{j, lb[j], ub[j]})
		}
	}
	return childNodeMulti(nd, st, extra, bound)
}

func childNodeMulti(parent *node, parentState *simplex.State, extra []boundOverride, bound float64) *node {
	ovs := make([]boundOverride, len(parent.overrides), len(parent.overrides)+len(extra))
	copy(ovs, parent.overrides)
	ovs = append(ovs, extra...)
	return &node{overrides: ovs, parentState: parentState, bound: math.Max(parent.bound, bound), depth: parent.depth + 1, brCol: -1}
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
	for iter := range 30 {
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

// rinsImprove fixes integers where the incumbent and the node LP agree and
// dives the few disagreeing ones (CBC's RINS neighborhood, LP-approximated).
func (m *Model) rinsImprove(nd *node, x []float64) (obj float64, dx, rowAct, rc, price []float64, ok bool) {
	best := m.bestXSnapshot
	if best == nil {
		return 0, nil, nil, nil, nil, false
	}
	ovs := make([]boundOverride, len(nd.overrides), len(nd.overrides)+len(m.P.Cols))
	copy(ovs, nd.overrides)
	for j, c := range m.P.Cols {
		if !c.Integer {
			continue
		}
		bv := math.Round(best[j])
		if math.Abs(x[j]-bv) > 0.1 {
			continue // disagreement: leave free for the sub-solve
		}
		lb, ub := m.LP.Bound(j)
		v := math.Max(lb, math.Min(ub, bv))
		ovs = append(ovs, boundOverride{j, v, v})
	}
	child := &node{overrides: ovs, bound: nd.bound, depth: nd.depth + 1}
	status, cx, _, _, _, _, cState := m.solveNode(child)
	if status != simplex.Optimal {
		return 0, nil, nil, nil, nil, false
	}
	// remaining fractional integers: finish with the rounding dive
	return m.diveForIncumbent(child, cx, cState)
}

// faceWalk dives by fixing fractional ints, preferring fixes that keep the LP
// on its optimal face and lifting the face minimally when neither side fits
func (m *Model) faceWalk(nd *node, x []float64, st *simplex.State, faceObj, cutoff float64) (obj float64, dx, rowAct, rc, price []float64, ok bool) {
	tol := math.Max(m.Limits.GapAbs, 1e-7)
	cur, curX, curState := nd, x, st
	face := faceObj
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
		lb, ub := m.nodeBound(cur, col)
		v := math.Max(lb, math.Min(ub, math.Round(curX[col])))
		alt, hasAlt := 2*math.Floor(curX[col])+1-v, true
		if alt < lb || alt > ub {
			hasAlt = false
		}
		try := func(val float64) (*node, []float64, *simplex.State, float64) {
			child := childNodeMulti(cur, curState, []boundOverride{{col, val, val}}, cur.bound)
			status, x2, _, _, _, cObj, cs := m.solveNode(child)
			if status != simplex.Optimal {
				return child, nil, nil, math.Inf(1)
			}
			return child, x2, cs, cObj
		}
		c1, x1, cs1, o1 := try(v)
		if o1 <= face+tol {
			cur, curX, curState = c1, x1, cs1
			continue
		}
		o2 := math.Inf(1)
		var c2 *node
		var x2 []float64
		var cs2 *simplex.State
		if hasAlt {
			c2, x2, cs2, o2 = try(alt)
			if o2 <= face+tol {
				cur, curX, curState = c2, x2, cs2
				continue
			}
		}
		// neither side on the face: lift it minimally and keep diving
		if math.Min(o1, o2) >= cutoff {
			debugf("facewalk: dead at depth %d col %d lift=%g face=%g cutoff=%g",
				c1.depth-nd.depth, col, math.Min(o1, o2), face, cutoff)
			return 0, nil, nil, nil, nil, false
		}
		if o1 <= o2 {
			cur, curX, curState, face = c1, x1, cs1, o1
		} else {
			cur, curX, curState, face = c2, x2, cs2, o2
		}
	}
}

// rensImprove (CBC RENS) fixes integers already integral in the node LP and
// dives the rest — incumbent-independent, so it can escape a poisoned start
func (m *Model) rensImprove(nd *node, x []float64) (obj float64, dx, rowAct, rc, price []float64, ok bool) {
	ovs := make([]boundOverride, len(nd.overrides), len(nd.overrides)+len(m.P.Cols))
	copy(ovs, nd.overrides)
	for j, c := range m.P.Cols {
		if !c.Integer {
			continue
		}
		v := math.Round(x[j])
		if math.Abs(x[j]-v) > intTol {
			continue // fractional: leave free for the dive
		}
		lb, ub := m.LP.Bound(j)
		v = math.Max(lb, math.Min(ub, v))
		ovs = append(ovs, boundOverride{j, v, v})
	}
	child := &node{overrides: ovs, bound: nd.bound, depth: nd.depth + 1}
	status, cx, _, _, _, _, cState := m.solveNode(child)
	if status != simplex.Optimal {
		return 0, nil, nil, nil, nil, false
	}
	// single-shot rounding first: one warm LP when the vertex is near-integral
	round := append([]boundOverride(nil), child.overrides...)
	for j, c := range m.P.Cols {
		if v := math.Round(cx[j]); c.Integer && math.Abs(cx[j]-v) > intTol {
			lb, ub := m.nodeBound(child, j)
			round = append(round, boundOverride{j, math.Max(lb, math.Min(ub, v)), math.Max(lb, math.Min(ub, v))})
		}
	}
	rchild := &node{overrides: round, parentState: cState, bound: child.bound, depth: child.depth + 1}
	if rStatus, rx, rAct, rRC, rPrice, rObj, _ := m.solveNode(rchild); rStatus == simplex.Optimal {
		return rObj, rx, rAct, rRC, rPrice, true
	}
	return m.diveForIncumbent(child, cx, cState)
}

// polishIncumbent 1-opt flips each binary and LP-completes the rest
// (CBC CbcHeuristicLocal style), keeping strict improvements; time-boxed
func (m *Model) polishIncumbent(nd *node, obj float64, x []float64, deadline time.Time) (float64, []float64, []float64, []float64, []float64, bool) {
	box := time.Now().Add(m.Limits.MaxTime / 8)
	if m.Limits.MaxTime <= 0 || (!deadline.IsZero() && deadline.Before(box)) {
		box = deadline
	}
	// base child: every integer fixed at the incumbent's (rounded) value
	base := append(make([]boundOverride, 0, len(nd.overrides)+len(m.P.Cols)), nd.overrides...)
	pos := make([]int, len(m.P.Cols))
	flippable := make([]bool, len(m.P.Cols))
	for j, c := range m.P.Cols {
		if !c.Integer {
			continue
		}
		lb, ub := m.nodeBound(nd, j)
		v := math.Max(lb, math.Min(ub, math.Round(x[j])))
		pos[j] = len(base)
		flippable[j] = c.UB-c.LB == 1 && ub-lb == 1
		base = append(base, boundOverride{j, v, v})
	}
	status, cx, act, rc, price, curObj, st := m.solveNode(&node{overrides: base, parentState: nd.parentState, bound: nd.bound, depth: nd.depth + 1})
	if status != simplex.Optimal {
		return obj, x, nil, nil, nil, false
	}
	cur := append([]float64(nil), cx...)
	improved := false
	saved := m.LP.Deadline
	defer func() { m.LP.Deadline = saved }()
	for j, c := range m.P.Cols {
		if !flippable[j] {
			continue
		}
		if !box.IsZero() && time.Now().After(box) {
			break
		}
		flip := c.LB + c.UB - base[pos[j]].lb
		cand := append([]boundOverride(nil), base...)
		cand[pos[j]] = boundOverride{j, flip, flip}
		probe := time.Now().Add(50 * time.Millisecond)
		if saved.IsZero() || probe.Before(saved) {
			m.LP.Deadline = probe
		}
		fStatus, fx, fAct, fRC, fPrice, fObj, fEnd := m.solveNode(&node{overrides: cand, parentState: st, bound: nd.bound, depth: nd.depth + 1})
		m.LP.Deadline = saved
		if fStatus == simplex.Optimal && m.improves(fObj, curObj) {
			curObj, act, rc, price = fObj, fAct, fRC, fPrice
			copy(cur, fx)
			base, st = cand, fEnd
			improved = true
		}
	}
	return curObj, cur, act, rc, price, improved
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
	fixed := make([]bool, len(m.P.Cols))
	for _, ov := range cur.overrides {
		if ov.lb == ov.ub {
			fixed[ov.idx] = true
		}
	}
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
		// batch-fix every near-integral int (CBC dives fix many per re-solve)
		var batch []boundOverride
		for j, c := range m.P.Cols {
			if !c.Integer || fixed[j] || j == col {
				continue
			}
			v := math.Round(curX[j])
			if math.Abs(curX[j]-v) > 0.01 {
				continue
			}
			lb, ub := m.nodeBound(cur, j)
			batch = append(batch, boundOverride{j, math.Max(lb, math.Min(ub, v)), math.Max(lb, math.Min(ub, v))})
		}
		lb, ub := m.nodeBound(cur, col)
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
			child = childNodeMulti(cur, curState, append(batch, boundOverride{col, v, v}), cur.bound)
			status, x2, _, _, _, _, cs := m.solveNode(child)
			if status == simplex.Optimal {
				cx, cState = x2, cs
				break
			}
			if len(batch) > 0 {
				batch = nil // batch too greedy: retry this level single-col
				continue
			}
			if !hasAlt {
				debugf("dive: dead end at depth %d col %d status=%d", child.depth-nd.depth, col, status)
				return 0, nil, nil, nil, nil, false
			}
			v, hasAlt = alt, false
		}
		for _, ov := range child.overrides[len(cur.overrides):] {
			fixed[ov.idx] = true
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

// nodeBound returns column j's effective bounds under nd's overrides,
// independent of whatever bounds are currently live on the LP.
func (m *Model) nodeBound(nd *node, j int) (float64, float64) {
	lb, ub := m.P.Cols[j].LB, m.P.Cols[j].UB
	for _, ov := range nd.overrides {
		if ov.idx == j {
			lb, ub = ov.lb, ov.ub
		}
	}
	return lb, ub
}

// strongBranch (CBC-style) probes both children of the top candidates and
// returns the column whose worse child moves the bound most; -1 if none do.
func (m *Model) strongBranch(nd *node, x []float64, endState *simplex.State, obj float64) int {
	type cand struct {
		j    int
		dist float64
	}
	var cands []cand
	for j, c := range m.P.Cols {
		if !c.Integer {
			continue
		}
		frac := x[j] - math.Floor(x[j])
		if dist := math.Min(frac, 1-frac); dist > intTol {
			cands = append(cands, cand{j, dist})
		}
	}
	if len(cands) < 2 {
		return -1
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].dist > cands[b].dist })
	if len(cands) > 8 {
		cands = cands[:8]
	}

	// short per-probe deadline: strong branching must never eat the budget
	saved := m.LP.Deadline
	defer func() { m.LP.Deadline = saved }()

	bestCol, bestScore := -1, math.Inf(-1)
	for _, cd := range cands {
		lb, ub := m.nodeBound(nd, cd.j)
		floorV := math.Floor(x[cd.j])
		score := math.Inf(1)
		for _, b := range [2][2]float64{{lb, floorV}, {floorV + 1, ub}} {
			ovs := make([]boundOverride, len(nd.overrides)+1)
			copy(ovs, nd.overrides)
			ovs[len(ovs)-1] = boundOverride{cd.j, b[0], b[1]}
			child := &node{overrides: ovs, parentState: endState, bound: nd.bound, depth: nd.depth + 1}
			probeDeadline := time.Now().Add(20 * time.Millisecond)
			if saved.IsZero() || probeDeadline.Before(saved) {
				m.LP.Deadline = probeDeadline
			}
			status, _, _, _, _, cObj, _ := m.solveNode(child)
			m.LP.Deadline = saved
			switch status {
			case simplex.Optimal:
				score = math.Min(score, cObj-obj) // bound gain of this side
				m.psRecord(cd.j, b[0] == floorV+1, x[cd.j]-floorV, cObj-obj)
			case simplex.Infeasible:
				// side prunes outright: keep score as-is (excellent)
			default:
				score = math.Min(score, 0) // probe unresolved: no credit
			}
		}
		if score > bestScore {
			bestScore, bestCol = score, cd.j
		}
	}
	if bestScore <= 1e-12 {
		return -1 // nothing moves the bound; fall back to least-fractional
	}
	return bestCol
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
