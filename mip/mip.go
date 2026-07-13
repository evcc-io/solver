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
	red           *reduction      // singleton elimination; nil when none applied
	subRed        *reduction      // equality-chain substitution; nil when none applied
	live          []boundOverride // bounds currently applied to LP; see solveNode
	rcTouched     []int           // columns tightened by reducedCostFix
	bestXSnapshot []float64       // incumbent X for the RINS neighborhood

	// scratch for node-level bound propagation (propagatedChild)
	propLB, propUB, propL0, propU0 []float64

	// debug-only pivot attribution per heuristic (SOLVER_DEBUG)
	dbgFW, dbgFP, dbgRINS, dbgDive, dbgNodeCold int64
	dbgNodePiv, dbgNodeN, dbgSBPiv, dbgSBN      int64

	heurStop    int64   // pivot-ledger value where the running heuristic must stop
	subMIP      bool    // this model is a heuristic sub-MIP: no nested sub-solves
	rinsLastObj float64 // incumbent the last RINS sub-MIP ran against

	// per-column pseudocosts: observed bound gain per unit fraction
	psUp, psDn   []float64
	psUpN, psDnN []int

	// continuous-row drop postsolve: keep mask + pre-drop rows for a·x
	rrKeep []bool
	rrRows []problem.Row
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

// cbcScore combines up/down estimated degradations the CBC way: WEIGHT_BEFORE
// min+max blend before an incumbent, product rule after (higher is better).
func cbcScore(estUp, estDn float64, hasIncumbent bool) float64 {
	lo, hi := math.Min(estUp, estDn), math.Max(estUp, estDn)
	if hasIncumbent {
		const eps = 1e-6
		return math.Max(lo, eps) * math.Max(hi, eps)
	}
	const weightBefore = 0.8
	return weightBefore*lo + (1-weightBefore)*hi
}

func New(p *problem.Problem) *Model {
	presolve(p)
	// GapAbs 1e-5 mirrors CBC's default cutoff increment (CbcCutoffIncrement)
	return &Model{P: p, LP: simplex.Build(p), Limits: Limits{GapRel: 1e-9, GapAbs: 1e-5}}
}

// rebuildLP rebuilds the LP with DSE suspended: cuts/root run on the canonical
// dualRun vertex; DSE is re-enabled only for deep node re-solves.
func (m *Model) rebuildLP() {
	m.LP = simplex.Build(m.P)
	m.LP.SuspendDSE(true)
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
	t0 := time.Now()
	// restart calls pass an original-space MIP start; map it down the chain
	if m.subRed != nil && len(m.MIPStart) == len(m.subRed.orig.Cols) {
		m.MIPStart = m.subRed.shrinkX(m.MIPStart)
	}
	if m.red != nil && len(m.MIPStart) == len(m.red.orig.Cols) {
		m.MIPStart = m.red.shrinkX(m.MIPStart)
	}
	mark := func(phase string) {
		st := m.LP.Stats
		debugf("phase: %s at %v (solves %d, pivots %d)", phase, time.Since(t0).Round(time.Millisecond), st.Solves, st.Phase1+st.Phase2+st.Dual)
	}
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
		// multi-pass, as CBC's CglPreProcess: each strengthening pass exposes
		// further fixes/strengthenings until quiet (CBC default: 4 passes)
		for pass := 0; pass < 4; pass++ {
			changed := probe(m.P, probeDeadline)
			presolve(m.P)
			if changed == 0 {
				break
			}
		}
		// CBC CglPreProcess applied uniformly (as CBC/Clp do), not gated on the
		// relaxation: strengthening (presolve) + forcing/redundant-row removal.
		if q, keep := dropRedundantRows(m.P, true); q != nil {
			m.rrKeep, m.rrRows = keep, m.P.Rows
			m.P = q
		}
		// substitute implied-free continuous cols out of equality chains
		// (CglPreProcess), then singleton elimination on the smaller model
		if q, red := substituteChains(m.P); red != nil {
			m.P, m.subRed = q, red
			if len(m.MIPStart) == len(red.orig.Cols) {
				m.MIPStart = red.shrinkX(m.MIPStart)
			}
		}
		if q, red := eliminateSingletons(m.P); red != nil {
			m.P, m.red = q, red
			if len(m.MIPStart) == len(red.orig.Cols) {
				m.MIPStart = red.shrinkX(m.MIPStart)
			}
		}
		m.rebuildLP()
		m.LP.Deadline = deadline
	}

	// seed the incumbent before cutting: the caller's MIP start plus root
	// reduced-cost fixing shrinks the problem the cuts then work on
	var startObj float64
	var startX, startAct, startRC, startPrice []float64
	haveStart := false
	// the mipstart block's root solve doubles as cut round 0's when
	// reducedCostFix left the LP untouched (preSolvedPivots keeps the cut
	// pivot ledger identical to solving it inside the loop)
	var preSolved *simplex.State
	var preSolvedPivots int64
	if len(m.MIPStart) == len(m.P.Cols) && len(m.P.SOSs) == 0 {
		preBase := m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual
		if status, st, _ := m.LP.ColdSolve(); status == simplex.Optimal {
			preSolvedPivots = m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual - preBase
			rx, _, rrc, _ := m.LP.Solution(st)
			robj := m.LP.InternalObjective(st)
			// polishing the trivial start measured as pure waste: it burned
			// 45k pivots on 020 while reducedCostFix fixed 0 columns at any
			// cutoff far above the optimum; tree heuristics polish for real
			if obj, x, act, rc, price, ok := m.completePoint(&node{brCol: -1}, m.MIPStart, st); ok {
				startObj, startX, startAct, startRC, startPrice = obj, x, act, rc, price
				haveStart = true
				debugf("mipstart: pre-cut incumbent obj=%g", obj)
				m.reducedCostFix(rx, rrc, robj, obj)
			}
			for _, ov := range m.live {
				m.LP.SetBound(ov.idx, m.P.Cols[ov.idx].LB, m.P.Cols[ov.idx].UB)
			}
			m.live = nil
			if len(m.rcTouched) == 0 {
				preSolved = st // bounds fully restored: st still solves this LP
			}
		}
	}

	mark("mipstart done")
	// GMI cut rounds tighten the root while its bound still moves; capped
	// at a fifth of the budget so the tree always gets its time
	origRows := len(m.P.Rows)
	// solved state of the CURRENT m.LP, nil whenever the LP was rebuilt
	// after the solve — lets the slack-drop pass skip a duplicate ColdSolve
	var rootSt *simplex.State
	// restart passes inherit the first pass's cuts: go straight to the tree
	if len(m.P.SOSs) == 0 && !m.SkipProbing {
		cutDeadline := deadline
		if m.Limits.MaxTime > 0 {
			cutDeadline = time.Now().Add(m.Limits.MaxTime / 5)
			m.LP.Deadline = cutDeadline
		}
		prevObj := math.Inf(1)
		firstObj := 0.0 // round-0 bound, for the relative diminishing-returns test
		flat := 0       // bound improvement can pause a round and resume
		lastBatch := -1 // first row index of the previous round's cuts
		poisoned := 0
		// carry the prior round's solved state so the next round warm-starts
		// the appended cut rows (dual repair) instead of solving from scratch
		var warmPrev *simplex.State
		var warmPrevM int
		// pivots are speed-invariant work units: budgeting rounds by pivots
		// keeps the cut set deterministic across engine speed changes; the
		// wall-clock box stays as a safety cap only
		pivots := func() int64 { return m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual }
		cutBase := pivots()
		if preSolved != nil {
			cutBase -= preSolvedPivots // the reused solve's work stays on the ledger
		}
		cutPivotBudget := int64(10 * m.LP.NumRows())
		for round := range maxCutRound {
			if !cutDeadline.IsZero() && time.Now().After(cutDeadline) {
				break
			}
			if pivots()-cutBase > cutPivotBudget {
				// the final batch was never validated by a re-solve: drop
				// it, matching the poison path's retraction semantics
				if lastBatch >= 0 {
					truncateRows(m.P, lastBatch)
					stats := m.LP.Stats
					m.rebuildLP()
					m.LP.Stats = stats
					rootSt = nil
				}
				debugf("cuts: pivot budget spent after %d rounds", round)
				break
			}
			var status simplex.Status
			var st *simplex.State
			if round == 0 && preSolved != nil {
				status, st = simplex.Optimal, preSolved
			} else if warmPrev != nil {
				status, st, _ = m.LP.WarmSolveExtended(warmPrev, warmPrevM)
			} else {
				status, st, _ = m.LP.ColdSolve()
			}
			rootSt = nil
			if status != simplex.Optimal {
				// the last batch poisoned the LP (degenerate grind):
				// retract it and continue with the last good relaxation
				if lastBatch >= 0 {
					truncateRows(m.P, lastBatch)
					stats := m.LP.Stats
					m.rebuildLP()
					m.LP.Stats = stats // counters survive the retraction rebuild
					m.LP.Deadline = cutDeadline
					warmPrev = nil // retracted: cold-solve the last good relaxation
					debugf("cuts: retracted poison batch, back to %d rows", lastBatch)
					lastBatch = -1
					if poisoned++; poisoned <= 2 {
						continue // retry from the last good relaxation
					}
				}
				m.LP.Deadline = deadline
				break
			}
			rootSt = st
			obj := m.LP.InternalObjective(st)
			// incumbent already proves optimality within the gap: more cuts
			// only refine a bound the tree closes in a few nodes
			if haveStart && !m.improves(obj, startObj) {
				debugf("cuts: root bound %g proves incumbent %g within gap after %d rounds", obj, startObj, round)
				break
			}
			// stall test is relative to the proof gap: a round that raises the
			// bound by less than the gap we're proving is work the tree absorbs
			gapTol := math.Max(m.Limits.GapAbs, m.Limits.GapRel*math.Abs(obj))
			if round == 0 {
				firstObj = obj
			}
			// diminishing returns vs cumulative gain: warm re-solves are too
			// cheap for the pivot budget to stop over-cutting (weak late cuts)
			total := obj - firstObj
			marginal := round > 0 && total > 0 && obj-prevObj < cutMarginFrac*total
			if marginal || (round > 0 && obj-prevObj < math.Max(1e-7, gapTol)) {
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
			prb, mir := 0, 0
			if m.LP.NumRows() > 1500 {
				prb = m.probingCuts(x, cutDeadline)
				// single-row MIR only seeds rounds 0-1: repeated rounds
				// bloat the rows and blur the face the walk needs
				if round <= 1 {
					mir = m.rowMIRCuts(x, origRows)
				}
			}
			cgl := 0
			if cglEnabled {
				cgl = m.knapsackCoverCuts(x) + m.cliqueCuts(x) + m.zeroHalfCuts(x) +
					m.flowCoverCuts(x) + m.liftProjectCuts(x)
			}
			added := gmi + prb + mir + cgl
			debugf("cuts: round %d added %d rows (gmi %d, probing %d, mir %d, bound %g, pivots %d)",
				round, added, gmi, prb, mir, obj, m.LP.Stats.Phase1+m.LP.Stats.Phase2+m.LP.Stats.Dual)
			if added == 0 {
				break
			}
			stats := m.LP.Stats
			m.rebuildLP()
			m.LP.Stats = stats // carry pivot counters across rebuilds
			m.LP.Deadline = cutDeadline
			rootSt = nil
			// warm re-solve reuses the pre-rebuild basis; invalid once scaling
			// recomputes factors on rebuild, so cold-solve each round when scaled
			if !m.LP.Scaled() {
				warmPrev, warmPrevM = st, lastBatch
			}
		}
		m.LP.Deadline = deadline
	}

	// keep only cuts tight at the root; drop against the cold canonical vertex
	// (a warm re-solve lands elsewhere, keeping a different cut set)
	if len(m.P.Rows) > origRows {
		rootSt = nil
		if status, st, _ := m.LP.ColdSolve(); status == simplex.Optimal {
			rootSt = st
		}
		if rootSt != nil {
			_, rowAct, _, _ := m.LP.Solution(rootSt)
			if dropped := dropSlackCuts(m.P, origRows, rowAct); dropped > 0 {
				stats := m.LP.Stats
				m.rebuildLP()
				m.LP.Stats = stats
				m.LP.Deadline = deadline
				debugf("cuts: dropped %d slack rows, kept %d", dropped, len(m.P.Rows)-origRows)
			}
		}
	}

	mark("cuts+drop done")
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
	lastTreeCut, treeCutRounds := 0, 0
	pendCutActs, pendCutBase, localCutFails := 0, 0, 0
	var pendCutNode *node
	pendCutPre := 0.0
	rinsFails := 0
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
			// DSE only for deep node re-solves; root stays canonical so the
			// incumbent heuristics find their target (CBC solves those separately)
			m.LP.SuspendDSE(nd.depth == 0)
			ndBase := m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual
			status, x, rowAct, rc, price, obj, endState := m.solveNode(nd)
			m.dbgNodePiv += m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual - ndBase
			m.dbgNodeN++
			m.LP.SuspendDSE(true)
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

			// cut-effectiveness check (CBC prunes ineffective cuts): the batch
			// must lift this node's bound, else it is retracted wholesale;
			// accepted batches drop the cuts that don't bind at the new optimum
			if pendCutActs > 0 && nd != pendCutNode {
				pendCutActs = 0 // owner pruned before re-solve: rows stay vacuous
			}
			if pendCutActs > 0 {
				nActs := pendCutActs
				pendCutActs = 0
				if obj < pendCutPre+math.Max(1e-9, 1e-7*math.Abs(pendCutPre)) {
					// no bound gain: retract the local rows and activations
					truncateRows(m.P, pendCutBase)
					stats := m.LP.Stats
					m.rebuildLP()
					m.LP.Stats = stats
					m.LP.Deadline = deadline
					m.live = nil
					nd.overrides = nd.overrides[:len(nd.overrides)-nActs]
					for _, h := range *pq {
						h.parentState = nil
					}
					nd.parentState = nil
					if localCutFails++; localCutFails >= 2 {
						treeCutRounds = 8 // this model rejects local cuts: stop
					}
					debugf("treecuts: retracted %d local rows (no bound gain, fails %d)", nActs, localCutFails)
					continue // re-solve clean
				}
				// keep only the binding cuts active (effectiveness pruning)
				keepActs := nd.overrides[:len(nd.overrides)-nActs]
				dropped := 0
				for _, ov := range nd.overrides[len(nd.overrides)-nActs:] {
					ri := ov.idx - len(m.P.Cols)
					slack := math.Min(rowAct[ri]-ov.lb, ov.ub-rowAct[ri])
					if slack <= 1e-6 {
						keepActs = append(keepActs, ov)
					} else {
						dropped++ // row stays (vacuous NR); activation dies
					}
				}
				nd.overrides = keepActs
				debugf("treecuts: batch kept, bound %g -> %g, %d of %d cuts binding", pendCutPre, obj, nActs-dropped, nActs)
			}

			// in-tree LOCAL cuts (CBC generates cuts throughout the search):
			// GMI cuts from this node's basis are valid only under its bounds,
			// so the rows are stored free (NR: vacuous for every other node and
			// for propagation/heuristics) and activated for this subtree via
			// bound overrides on their logical variables, which children
			// inherit; the CGL families (globally valid) join the same batch.
			if cglEnabled && treeCutRounds < 8 && nodeCount-lastTreeCut >= 256 {
				lastTreeCut = nodeCount
				nGlob := m.knapsackCoverCuts(x) + m.cliqueCuts(x) + m.zeroHalfCuts(x) +
					m.flowCoverCuts(x) + m.liftProjectCuts(x)
				locBase := len(m.P.Rows)
				nLoc := 0
				if localCutsEnabled {
					// node-local probing implication cuts (CBC's in-tree default:
					// cheap 1-2 element rows, no degenerate grind) derived under
					// the node bounds, plus node-basis GMI
					nc := len(m.P.Cols)
					ndLB, ndUB := make([]float64, nc), make([]float64, nc)
					for j := range nc {
						ndLB[j], ndUB[j] = m.LP.Bound(j)
					}
					nLoc = m.probingCutsNode(x, ndLB, ndUB, m.LP.Deadline)
					nLoc += m.gomoryCuts(endState)
				}
				if nGlob+nLoc > 0 {
					treeCutRounds++
					acts := make([]boundOverride, 0, nLoc)
					for ri := locBase; ri < len(m.P.Rows); ri++ {
						r := &m.P.Rows[ri]
						rlb, rub := r.Bounds()
						r.Sense, r.HasRange = problem.NR, false
						acts = append(acts, boundOverride{len(m.P.Cols) + ri, rlb, rub})
					}
					debugf("treecuts: +%d local gmi, +%d global rows at node %d (round %d)", nLoc, nGlob, nodeCount, treeCutRounds)
					stats := m.LP.Stats
					m.rebuildLP()
					m.LP.Stats = stats
					m.LP.Deadline = deadline
					m.live = nil
					for _, h := range *pq {
						h.parentState = nil
					}
					nd.parentState = nil
					nd.brCol = -1 // avoid double pseudocost credit on re-solve
					nd.overrides = append(nd.overrides, acts...)
					if len(acts) > 0 {
						pendCutActs, pendCutBase, pendCutPre, pendCutNode = len(acts), locBase, obj, nd
					}
					if backtrack != nil {
						backtrack.parentState = nil
					}
					continue // re-solve this node on the tightened LP
				}
			}
			branchCol, sosIdx, sosSplit := m.findBranch(x)
			if branchCol < 0 && sosIdx < 0 {
				newIncumbent(obj, x, rowAct, rc, price)
				break
			}
			// CBC reliability branching at every depth: it self-limits by
			// only probing columns whose pseudocosts aren't yet trusted.
			if branchCol >= 0 {
				sbCut := math.Inf(1)
				if hasIncumbent {
					sbCut = bestInternal
				}
				sb, fixed := m.chooseBranch(nd, x, endState, obj, sbCut, hasIncumbent)
				if len(fixed) > 0 {
					// probes fathomed sides: fix here and re-solve the
					// node with the survivors forced (no branch spent)
					debugf("sbfix: %d vars fixed at depth %d", len(fixed), nd.depth)
					nd = childNodeMulti(nd, endState, fixed, obj)
					continue
				}
				if sb >= 0 {
					branchCol = sb
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
				if hObj, hx, hAct, hRC, hPrice, ok := m.completePoint(nd, m.MIPStart, endState); ok {
					debugf("mipstart: accepted obj=%g", hObj)
					newIncumbent(hObj, hx, hAct, hRC, hPrice)
				} else {
					debugf("mipstart: rejected")
				}
			}
			closeGap := hasIncumbent && bestInternal-obj < 0.1*math.Max(1, math.Abs(bestInternal))
			if diveTries < 8 && branchCol >= 0 && !(nodeCount == 1 && closeGap) && (nodeCount == 1 || nodeCount%256 == 0) {
				diveTries++
				// box the heuristic burst: dives must never eat the tree's time.
				// The root burst gets more — success there ends the whole solve
				savedDL := m.LP.Deadline
				burstDL := savedDL
				if m.Limits.MaxTime > 0 {
					div := time.Duration(8)
					if nodeCount == 1 {
						div = 3
					}
					if hb := time.Now().Add(m.Limits.MaxTime / div); savedDL.IsZero() || hb.Before(savedDL) {
						burstDL = hb
					}
				}
				m.LP.Deadline = burstDL
				// per-SOLVE pivot cap (CBC caps heuristic iterations): a warm
				// heuristic LP takes tens of pivots; a grind past 2m+200 is a
				// degenerate cycle and returns IterLimit = heuristic failure
				m.LP.SetIterCap(2*m.LP.NumRows() + 200)
				cutoff := math.Inf(1)
				if hasIncumbent {
					cutoff = bestInternal
				}
				var hObj float64
				var hx, hAct, hRC, hPrice []float64
				ok := false
				keep := func(o float64, ax, aa, ar, ap []float64, k bool) {
					if k && (!ok || m.improves(o, hObj)) {
						hObj, hx, hAct, hRC, hPrice, ok = o, ax, aa, ar, ap, true
					}
				}
				// pump FIRST at the root with its own sub-budget so faceWalk
				// can't starve it (CBC runs the feasibility pump first).
				if nodeCount == 1 {
					if m.Limits.MaxTime > 0 {
						if pd := time.Now().Add(m.Limits.MaxTime / 6); pd.Before(burstDL) {
							m.LP.Deadline = pd
						}
					}
					fpBase := m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual
					keep(m.feasibilityPump(nd, x, endState))
					m.dbgFP += m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual - fpBase
					m.LP.Deadline = burstDL
				}
				// faceWalk stays the primary (its vertex path is load-bearing:
				// four reorderings measured wall-negative on the golden cases);
				// the failure fallbacks below are reduced-sub-problem solves
				fwBase := m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual
				keep(m.faceWalk(nd, x, endState, obj, cutoff))
				m.dbgFW += m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual - fwBase
				if !ok {
					keep(m.rensImprove(nd, x, endState))
				}
				if !ok {
					fpBase := m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual
					keep(m.feasibilityPump(nd, x, endState))
					m.dbgFP += m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual - fpBase
					if !ok {
						dvBase := m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual
						keep(m.diveForIncumbent(nd, x, endState))
						m.dbgDive += m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual - dvBase
					}
				}
				m.LP.SetIterCap(0)
				m.LP.Deadline = savedDL
				if ok && (!hasIncumbent || m.improves(hObj, bestInternal)) {
					debugf("dive: incumbent %g -> %g", bestInternal, hObj)
					newIncumbent(hObj, hx, hAct, hRC, hPrice)
				}
			}
			// RINS-lite: fix integers where incumbent and node LP agree,
			// LP-complete the rest; accept only strict improvements.
			// Failures back the frequency off (CBC adaptive frequency)
			// CBC runs RINS only against a NEW incumbent: the neighborhood is
			// a function of (incumbent, LP) and repeats are mostly redundant
			if hasIncumbent && nodeCount%(64<<min(rinsFails, 3)) == 0 && bestInternal != m.rinsLastObj {
				m.rinsLastObj = bestInternal
				rnBase := m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual
				m.LP.SetIterCap(2*m.LP.NumRows() + 200)
				if hObj, hx, hAct, hRC, hPrice, ok := m.rinsImprove(nd, x, endState); ok && m.improves(hObj, bestInternal) {
					rinsFails = 0
					debugf("rins: improved %g -> %g", bestInternal, hObj)
					if pObj, px, pAct, pRC, pPrice, better := m.polishIncumbent(nd, hObj, hx, deadline); better {
						debugf("polish: rins %g -> %g", hObj, pObj)
						hObj, hx, hAct, hRC, hPrice = pObj, px, pAct, pRC, pPrice
					}
					newIncumbent(hObj, hx, hAct, hRC, hPrice)
				} else {
					rinsFails++
				}
				m.LP.SetIterCap(0)
				m.dbgRINS += m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual - rnBase
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
	debugf("mip exit: nodes=%d pivots=%d best=%g remaining=%g proven=%v stopped=%v proofLost=%v heap=%d elapsed=%v",
		nodeCount, m.LP.Stats.Phase1+m.LP.Stats.Phase2+m.LP.Stats.Dual, bestInternal, remaining, proven, stopped, proofLost, pq.Len(), time.Since(t0).Round(time.Millisecond))
	debugf("pivot split: facewalk %d, fpump %d, dive %d, rins %d, coldfallbacks %d, node %d/%d, sb %d/%d",
		m.dbgFW, m.dbgFP, m.dbgDive, m.dbgRINS, m.dbgNodeCold, m.dbgNodePiv, m.dbgNodeN, m.dbgSBPiv, m.dbgSBN)

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
	if m.red != nil {
		m.red.expand(&res)
	}
	if m.subRed != nil {
		m.subRed.expand(&res)
	}
	if m.rrKeep != nil {
		expandRows(&res, m.rrKeep, m.rrRows)
	}
	return res
}

// baseBound returns an override target's un-overridden bounds: column bounds
// for structurals, row bounds for logical (row-slack) variables — vacuous for
// local-cut rows, which are stored free and activated per subtree.
func (m *Model) baseBound(idx int) (float64, float64) {
	if n := len(m.P.Cols); idx >= n {
		return m.P.Rows[idx-n].Bounds()
	}
	c := &m.P.Cols[idx]
	return c.LB, c.UB
}

// heurOver reports whether the shared heuristic pivot budget is spent; every
// heuristic dive/walk/pump loop checks it (CBC budgets heuristics the same way).
func (m *Model) heurOver() bool {
	return m.heurStop > 0 && m.LP.Stats.Phase1+m.LP.Stats.Phase2+m.LP.Stats.Dual > m.heurStop
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
		if o.idx >= n {
			continue // local-cut activation on a logical var: not a column bound
		}
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
		lb, ub := m.baseBound(idx)
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
			m.dbgNodeCold++
			status, endState, _ = m.LP.ColdSolve()
		}
	} else {
		m.dbgNodeCold++
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
	// work in ORIGINAL space (the LP scales internally, CBC/OSI-style)
	orig := m.LP.ObjectiveOrig()
	fpCost := make([]float64, len(orig))
	m.LP.SetObjectiveOrig(fpCost)
	defer func() { m.LP.SetObjectiveOrig(orig) }()

	// objective feasibility pump: blend the decaying true objective into the L1
	// projection so it lands near good-objective points (Achterberg-Berthold).
	cNorm := 0.0
	for _, v := range orig {
		cNorm += v * v
	}
	cNorm = math.Sqrt(cNorm)
	alpha := 1.0

	ints := make([]int, 0, 64)
	for j, c := range m.P.Cols {
		if c.Integer {
			ints = append(ints, j)
		}
	}
	rounded := make([]float64, len(ints))
	fracOf := make([]float64, len(ints)) // signed distance curX-rounded
	st := endState.Clone()
	curX := x
	var prev []float64
	// pivot-budgeted like CBC's pump (iteration-capped): a projection loop
	// that grinds past this is cycling on a degenerate face, not converging
	fpBase := m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual
	fpBudget := int64(10 * m.LP.NumRows())
	for iter := range 50 {
		if m.LP.Stats.Phase1+m.LP.Stats.Phase2+m.LP.Stats.Dual-fpBase > fpBudget {
			return fail()
		}
		if m.heurOver() {
			return fail()
		}
		if !m.LP.Deadline.IsZero() && time.Now().After(m.LP.Deadline) {
			return fail()
		}
		integral, nFrac := true, 0
		for k, j := range ints {
			lb, ub := m.LP.Bound(j)
			v := math.Max(lb, math.Min(ub, math.Round(curX[j])))
			rounded[k], fracOf[k] = v, curX[j]-v
			if math.Abs(fracOf[k]) > intTol {
				integral = false
				nFrac++
			}
		}
		if integral {
			debugf("fp: integral at iter %d", iter)
			m.LP.SetObjectiveOrig(orig) // polish the real objective, integers fixed
			cx := append([]float64(nil), curX...)
			for k, j := range ints {
				cx[j] = rounded[k]
			}
			return m.completePoint(nd, cx, nil)
		}
		if prev != nil && slicesEqual(prev, rounded) {
			flipMostFractional(rounded, fracOf, ints, m.LP, 10+3*iter)
		}
		prev = append(prev[:0], rounded...)

		// projection cost: unit-normalized L1 pull-back to `rounded` blended
		// with the true objective, objective-heavy (alpha~1) decaying to L1.
		dNorm := math.Sqrt(float64(nFrac))
		w1, wc := 0.0, 0.0
		if dNorm > 0 {
			w1 = (1 - alpha) / dNorm
		}
		if cNorm > 0 {
			wc = alpha / cNorm
		}
		for j := range fpCost {
			fpCost[j] = wc * orig[j]
		}
		for k, j := range ints {
			lb, ub := m.LP.Bound(j)
			switch {
			case rounded[k] <= lb+intTol:
				fpCost[j] += w1
			case rounded[k] >= ub-intTol:
				fpCost[j] -= w1
			}
		}
		alpha *= fpDecay

		m.LP.SetObjectiveOrig(fpCost) // scale the projection cost into the LP
		if status := m.LP.PrimalResolve(st); status != simplex.Optimal {
			debugf("fp: projection LP status=%d at iter %d", status, iter)
			return fail()
		}
		curX, _, _, _ = m.LP.Solution(st)
	}
	return fail()
}

// fpDecay is the objective-pump alpha decay per iteration: slower decay keeps
// the pump objective-guided longer, landing near good-objective incumbents.
const fpDecay = 0.7

// flipMostFractional rounds the T most-fractional entries the other way to
// break a feasibility-pump cycle (CBC's FPump restart perturbation).
func flipMostFractional(rounded, fracOf []float64, ints []int, lp *simplex.LP, T int) {
	order := make([]int, len(fracOf))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return math.Abs(fracOf[order[a]]) > math.Abs(fracOf[order[b]])
	})
	if T > len(order) {
		T = len(order)
	}
	for i := 0; i < T; i++ {
		k := order[i]
		if math.Abs(fracOf[k]) <= intTol {
			break
		}
		v := rounded[k] - 1
		if fracOf[k] > 0 {
			v = rounded[k] + 1
		}
		lb, ub := lp.Bound(ints[k])
		rounded[k] = math.Max(lb, math.Min(ub, v))
	}
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

// rinsImprove (CBC CbcHeuristicRINS): fix integers where the node LP and the
// incumbent agree, then solve the reduced problem as a node-capped sub-MIP —
// far cheaper than walking the full LP and it searches the neighborhood
// exhaustively. The winner is re-completed on the main LP for pricing.
func (m *Model) rinsImprove(nd *node, x []float64, st *simplex.State) (obj float64, dx, rowAct, rc, price []float64, ok bool) {
	best := m.bestXSnapshot
	if best == nil || m.subMIP {
		return 0, nil, nil, nil, nil, false
	}
	q := cloneProblem(m.P)
	nInt, nFix := 0, 0
	for j := range q.Cols {
		c := &q.Cols[j]
		if !c.Integer {
			continue
		}
		nInt++
		bv := math.Round(best[j])
		if math.Abs(x[j]-bv) > 0.1 {
			continue // disagreement: leave free for the sub-MIP
		}
		lb, ub := m.LP.Bound(j)
		v := math.Max(lb, math.Min(ub, bv))
		c.LB, c.UB = v, v
		nFix++
	}
	if nInt == 0 || nFix < nInt/2 {
		return 0, nil, nil, nil, nil, false // neighborhood too large
	}
	return m.solveNeighborhood(q, st)
}

// cloneProblem deep-copies rows/cols/SOS for an independent sub-MIP solve.
func cloneProblem(p *problem.Problem) *problem.Problem {
	q := problem.New()
	q.Name, q.ObjSense = p.Name, p.ObjSense
	q.Cols = make([]problem.Col, len(p.Cols))
	for j, c := range p.Cols {
		c.Idx = append([]int(nil), c.Idx...)
		c.Coef = append([]float64(nil), c.Coef...)
		q.Cols[j] = c
	}
	q.Rows = make([]problem.Row, len(p.Rows))
	for i, r := range p.Rows {
		r.Idx = append([]int(nil), r.Idx...)
		r.Coef = append([]float64(nil), r.Coef...)
		q.Rows[i] = r
	}
	q.SOSs = make([]problem.SOS, len(p.SOSs))
	for i, s := range p.SOSs {
		s.Idx = append([]int(nil), s.Idx...)
		s.Weight = append([]float64(nil), s.Weight...)
		q.SOSs[i] = s
	}
	return q
}

// faceWalk dives by fixing fractional ints, preferring fixes that keep the LP
// on its optimal face and lifting the face minimally when neither side fits
func (m *Model) faceWalk(nd *node, x []float64, st *simplex.State, faceObj, cutoff float64) (obj float64, dx, rowAct, rc, price []float64, ok bool) {
	// LP-noise tolerance, deliberately independent of the proof gap: a wide
	// gap lets per-step drift compound over long walks and miss the vertex
	const tol = 1e-6
	cur, curX, curState := nd, x, st
	face := faceObj
	for {
		if m.heurOver() {
			return 0, nil, nil, nil, nil, false
		}
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
func (m *Model) rensImprove(nd *node, x []float64, st *simplex.State) (obj float64, dx, rowAct, rc, price []float64, ok bool) {
	ovs := make([]boundOverride, len(nd.overrides), len(nd.overrides)+len(m.P.Cols))
	copy(ovs, nd.overrides)
	nInt, nFix := 0, 0
	for j, c := range m.P.Cols {
		if !c.Integer {
			continue
		}
		nInt++
		v := math.Round(x[j])
		if math.Abs(x[j]-v) > intTol {
			continue // fractional: left free for the sub-MIP
		}
		lb, ub := m.LP.Bound(j)
		v = math.Max(lb, math.Min(ub, v))
		ovs = append(ovs, boundOverride{j, v, v})
		nFix++
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
	// the rounded completion failed: solve the integral-fixed neighborhood as
	// a node-capped sub-MIP (CBC RENS / mini branch and bound), not a dive
	if m.subMIP || nInt == 0 || nFix < nInt/2 {
		return m.diveForIncumbent(child, cx, cState)
	}
	q := cloneProblem(m.P)
	for _, ov := range ovs {
		if ov.idx < len(q.Cols) && ov.lb == ov.ub {
			q.Cols[ov.idx].LB, q.Cols[ov.idx].UB = ov.lb, ov.ub
		}
	}
	return m.solveNeighborhood(q, st)
}

// solveNeighborhood solves q (a reduced clone) as a node-capped sub-MIP and
// completes the winner on the main LP (CBC's mini branch and bound).
func (m *Model) solveNeighborhood(q *problem.Problem, st *simplex.State) (obj float64, dx, rowAct, rc, price []float64, ok bool) {
	sub := New(q)
	sub.subMIP = true
	if m.bestXSnapshot != nil {
		sub.MIPStart = append([]float64(nil), m.bestXSnapshot...)
	}
	sub.Limits = Limits{GapRel: m.Limits.GapRel, GapAbs: m.Limits.GapAbs,
		MaxNodes: 50, MaxTime: time.Second}
	if m.Limits.MaxTime > 0 {
		if t := m.Limits.MaxTime / 16; t < sub.Limits.MaxTime {
			sub.Limits.MaxTime = t
		}
	}
	res := sub.Solve()
	if !res.HasIncumbent {
		return 0, nil, nil, nil, nil, false
	}
	// complete globally: the neighborhood winner is a global point
	return m.completePoint(&node{brCol: -1}, res.X, st)
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
	// pivot-budgeted on top of the time box: each flip is one small LP, so a
	// between-flip check binds tightly (1-opt must stay cheap vs the tree)
	polBase := m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual
	polBudget := int64(3 * m.LP.NumRows())
	for j, c := range m.P.Cols {
		if !flippable[j] {
			continue
		}
		if m.LP.Stats.Phase1+m.LP.Stats.Phase2+m.LP.Stats.Dual-polBase > polBudget {
			break
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
func (m *Model) completePoint(nd *node, point []float64, st *simplex.State) (obj float64, x, rowAct, rc, price []float64, ok bool) {
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
	child := &node{overrides: ovs, parentState: st, bound: nd.bound, depth: nd.depth + 1}
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
		if ov.idx < len(fixed) && ov.lb == ov.ub {
			fixed[ov.idx] = true
		}
	}
	for {
		if m.heurOver() {
			return 0, nil, nil, nil, nil, false
		}
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
	if os.Getenv("CBC_NORCFIX") != "" {
		return
	}
	gap := math.Max(m.Limits.GapAbs, m.Limits.GapRel*math.Abs(best))
	slack := best - gap - rootObj
	if slack < 0 {
		return
	}
	nFixed := len(m.rcTouched)
	defer func() {
		debugf("rcfix: cutoff %g fixed %d cols (total %d)", best, len(m.rcTouched)-nFixed, len(m.rcTouched))
	}()
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

// strongProbeCap bounds dual pivots per strong-branch probe: enough for a
// useful bound, cheap enough that probing all shallow nodes stays affordable.
const strongProbeCap = 100

// CBC reliability-branching knobs (the cbc binary's -trust / -strong defaults):
// a column's pseudocost is trusted after numberBeforeTrust observed gains, and
// at most maxStrong untrusted columns are strong-branched per node.
const (
	numberBeforeTrust = 10
	maxStrong         = 5
)

// psEstimate returns column j's up/down predicted degradations at fraction
// frac, and its observation counts. Unseen directions fall back to |obj| —
// CBC's initial dynamic pseudocost (costValue).
func (m *Model) psEstimate(j int, frac float64) (estUp, estDn float64, upN, dnN int) {
	if m.psUpN != nil {
		upN, dnN = m.psUpN[j], m.psDnN[j]
	}
	init := math.Abs(m.P.Cols[j].Obj)
	pu, pd := init, init
	if upN > 0 {
		pu = m.psUp[j] / float64(upN)
	}
	if dnN > 0 {
		pd = m.psDn[j] / float64(dnN)
	}
	return pu * (1 - frac), pd * frac, upN, dnN
}

// chooseBranch is CBC reliability branching: untrusted pseudocosts float up,
// the top maxStrong are probed to seed them, best cbcScore wins (see below).
func (m *Model) chooseBranch(nd *node, x []float64, endState *simplex.State, obj, cutoff float64, hasIncumbent bool) (int, []boundOverride) {
	type cand struct {
		j       int
		frac    float64
		sortKey float64
		trusted bool
	}
	var cands []cand
	for j, c := range m.P.Cols {
		if !c.Integer {
			continue
		}
		frac := x[j] - math.Floor(x[j])
		if math.Min(frac, 1-frac) <= intTol {
			continue
		}
		estUp, estDn, upN, dnN := m.psEstimate(j, frac)
		key := cbcScore(estUp, estDn, hasIncumbent)
		trusted := upN >= numberBeforeTrust && dnN >= numberBeforeTrust
		if !trusted {
			key *= 1e3 // untrusted floats up; never-probed higher still
			if upN == 0 && dnN == 0 {
				key *= 1e10
			}
		}
		cands = append(cands, cand{j, frac, key, trusted})
	}
	if len(cands) == 0 {
		return -1, nil
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].sortKey > cands[b].sortKey })

	// short per-probe deadline + cheap dual-only cap (CBC hot-start iter cap).
	saved := m.LP.Deadline
	defer func() { m.LP.Deadline = saved }()
	m.LP.SetProbe(strongProbeCap)
	defer m.LP.ClearProbe()

	var fixed []boundOverride
	probed := 0
	for _, cd := range cands {
		if cd.trusted || probed >= maxStrong {
			continue // trust the pseudocost; no probe
		}
		probed++
		sbBase := m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual
		m.dbgSBN++
		lb, ub := m.nodeBound(nd, cd.j)
		floorV := math.Floor(x[cd.j])
		dead := [2]bool{}
		sides := [2][2]float64{{lb, floorV}, {floorV + 1, ub}}
		for s, b := range sides {
			ovs := make([]boundOverride, len(nd.overrides)+1)
			copy(ovs, nd.overrides)
			ovs[len(ovs)-1] = boundOverride{cd.j, b[0], b[1]}
			child := &node{overrides: ovs, parentState: endState, bound: nd.bound, depth: nd.depth + 1}
			probeDeadline := time.Now().Add(80 * time.Millisecond)
			if saved.IsZero() || probeDeadline.Before(saved) {
				m.LP.Deadline = probeDeadline
			}
			status, _, _, _, _, cObj, _ := m.solveNode(child)
			m.LP.Deadline = saved
			switch status {
			case simplex.Optimal:
				m.psRecord(cd.j, b[0] == floorV+1, cd.frac, cObj-obj)
				if !m.improves(cObj, cutoff) {
					dead[s] = true // fathomed by the cutoff: as good as pruned
				}
			case simplex.Infeasible:
				dead[s] = true // side prunes outright
			}
		}
		m.dbgSBPiv += m.LP.Stats.Phase1 + m.LP.Stats.Phase2 + m.LP.Stats.Dual - sbBase
		if dead[0] != dead[1] {
			surv := sides[0]
			if dead[0] {
				surv = sides[1]
			}
			fixed = append(fixed, boundOverride{cd.j, surv[0], surv[1]})
		}
	}
	if len(fixed) > 0 {
		return -1, fixed
	}

	// final pick: score every candidate by its (now seeded) pseudocost.
	best, bestCol := 0.0, -1
	for _, cd := range cands {
		estUp, estDn, upN, dnN := m.psEstimate(cd.j, cd.frac)
		if upN == 0 || dnN == 0 {
			continue // still unseeded: no reliable score
		}
		if score := cbcScore(estUp, estDn, hasIncumbent); score > best {
			best, bestCol = score, cd.j
		}
	}
	return bestCol, nil
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
