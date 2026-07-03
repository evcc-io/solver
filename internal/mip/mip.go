// Package mip implements branch-and-bound over an internal/simplex LP,
// using the integrality and SOS metadata from an internal/problem.Problem.
package mip

import (
	"container/heap"
	"math"
	"time"

	"cbcgo/internal/problem"
	"cbcgo/internal/simplex"
)

type Status int

const (
	Optimal Status = iota
	Infeasible
	Unbounded
	Stopped
)

const intTol = 1e-6

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
	MaxSeconds float64 // <=0 means unlimited
	MaxNodes   int     // <=0 means unlimited
	GapRel     float64 // relative MIP gap, e.g. 0.0001; <=0 means 0 (prove optimal)
	GapAbs     float64 // absolute MIP gap; <=0 means 0
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
	P      *problem.Problem
	LP     *simplex.LP
	Limits Limits
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
	deadline := time.Time{}
	if m.Limits.MaxSeconds > 0 {
		deadline = time.Now().Add(time.Duration(m.Limits.MaxSeconds * float64(time.Second)))
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

	for pq.Len() > 0 {
		if !deadline.IsZero() && time.Now().After(deadline) {
			stopped = true
			break
		}
		if m.Limits.MaxNodes > 0 && nodeCount >= m.Limits.MaxNodes {
			stopped = true
			break
		}
		nd := heap.Pop(pq).(*node)
		if hasIncumbent && !m.improves(nd.bound, bestInternal) {
			continue
		}

		status, x, rowAct, rc, price, obj, endState := m.solveNode(nd)
		nodeCount++
		if !rootDone {
			rootDone = true
			rootStatus = status
		}
		if status == simplex.Infeasible {
			continue
		}
		if status == simplex.Unbounded {
			continue
		}
		if hasIncumbent && !m.improves(obj, bestInternal) {
			continue
		}

		branchCol, sosIdx, sosSplit := m.findBranch(x)
		if branchCol < 0 && sosIdx < 0 {
			bestInternal = obj
			bestX, bestRowAct, bestRC, bestPrice = x, rowAct, rc, price
			hasIncumbent = true
			continue
		}

		if branchCol >= 0 {
			lb, ub := m.LP.Bound(branchCol)
			floorV := math.Floor(x[branchCol] + 1e-7)
			ceilV := math.Ceil(x[branchCol] - 1e-7)
			heap.Push(pq, childNode(nd, endState, boundOverride{branchCol, lb, floorV}, obj))
			heap.Push(pq, childNode(nd, endState, boundOverride{branchCol, ceilV, ub}, obj))
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
			heap.Push(pq, childNodeMulti(nd, endState, loOv, obj))
			heap.Push(pq, childNodeMulti(nd, endState, hiOv, obj))
		}
	}

	res := Result{HasIncumbent: hasIncumbent, NodeCount: nodeCount}
	switch {
	case !rootDone:
		res.Status = Infeasible
	case rootStatus == simplex.Unbounded:
		res.Status = Unbounded
	case stopped:
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
	return &node{overrides: ovs, parentState: parentState, bound: bound, depth: parent.depth + 1}
}

func childNodeMulti(parent *node, parentState *simplex.State, extra []boundOverride, bound float64) *node {
	ovs := make([]boundOverride, len(parent.overrides), len(parent.overrides)+len(extra))
	copy(ovs, parent.overrides)
	ovs = append(ovs, extra...)
	return &node{overrides: ovs, parentState: parentState, bound: bound, depth: parent.depth + 1}
}

// solveNode applies nd's overrides, solves (warm from nd.parentState if
// set, else cold), reverts the shared bounds, and returns the end state.
func (m *Model) solveNode(nd *node) (status simplex.Status, x, rowAct, rc, price []float64, internalObj float64, endState *simplex.State) {
	type saved struct{ lb, ub float64 }
	touched := map[int]saved{}
	var touchedIdx []int
	for _, ov := range nd.overrides {
		if _, seen := touched[ov.idx]; !seen {
			lb, ub := m.LP.Bound(ov.idx)
			touched[ov.idx] = saved{lb, ub}
			touchedIdx = append(touchedIdx, ov.idx)
		}
		m.LP.SetBound(ov.idx, ov.lb, ov.ub)
	}
	if nd.parentState != nil {
		status, endState, _ = m.LP.WarmSolve(nd.parentState.Clone(), touchedIdx)
	} else {
		status, endState, _ = m.LP.ColdSolve()
	}
	if status == simplex.Optimal {
		x, rowAct, rc, price = m.LP.Solution(endState)
		internalObj = m.LP.InternalObjective(endState)
	}
	for idx, s := range touched {
		m.LP.SetBound(idx, s.lb, s.ub)
	}
	return status, x, rowAct, rc, price, internalObj, endState
}

// findBranch picks the most-fractional integer column, or failing that the
// first violated SOS constraint (returning its split point).
func (m *Model) findBranch(x []float64) (col, sosIdx, sosSplit int) {
	col, sosIdx = -1, -1
	best := intTol
	for j, c := range m.P.Cols {
		if !c.Integer {
			continue
		}
		v := x[j]
		frac := v - math.Floor(v)
		dist := math.Min(frac, 1-frac)
		if dist > best {
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
