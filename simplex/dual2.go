package simplex

import (
	"math"
	"os"
	"slices"
	"time"
)

// dual2Enabled gates the Clp-faithful dual engine for A/B measurement.
var (
	dual2Enabled = os.Getenv("CBC_DUAL2") != "0" // Clp DSE dual, on by default
	dual2NoDSE   = os.Getenv("CBC_DUAL2_NODSE") == "1"
	dual2Perturb = os.Getenv("CBC_DUAL2_PERTURB") != "0" // Clp always perturbs
	noDualRepair = os.Getenv("CBC_NODUAL") == "1"
)

const (
	dualTol      = 1e-7    // allowed reduced-cost infeasibility (Harris slop)
	dual2PrimTol = feasTol // violations below LU recompute noise are not chased
	dual2Accept  = 1e-9    // minimum admissible pivot row entry
	dual2Cap     = 20000   // hard pivot cap per re-solve
)

type dual2Result int

const (
	dual2Bail dual2Result = iota
	dual2Optimal
	dual2Infeasible
)

// dual2Cand is one admissible entering candidate in the dual ratio test.
type dual2Cand struct {
	j     int
	dir   float64
	at    float64 // |pivot row entry|, admissible sign guaranteed
	rd    float64 // reduced-cost slack in dir (clamped >= 0)
	ratio float64
}

// dual2Run is a complete bounded dual simplex modeled on ClpSimplexDual:
// DSE pricing, Harris ratio test with bound flips, run to optimality.
func (lp *LP) dual2Run(st *State) dual2Result {
	m, nt := lp.m, lp.nTotal()
	lp.Stats.Dual2Runs++

	if lp.d2ws == nil {
		lp.d2ws = &dual2WS{
			d: make([]float64, nt), w: make([]float64, m),
			rowR: make([]float64, m), tau: make([]float64, m),
			a: make([]float64, m), delta: make([]float64, m),
			alphaRow: make([]float64, nt), costBuf: make([]float64, nt),
			y: make([]float64, m), seen: make([]bool, nt),
			touched: make([]int, 0, 256), cands: make([]dual2Cand, 0, 256),
		}
	}
	ws := lp.d2ws

	// perturbed cost copy (Clp-style anti-degeneracy): jitter favors each
	// variable's entry status; solveFrom's true-cost primal cleanup undoes it
	cost := lp.cost
	if dual2Perturb {
		cost = ws.costBuf
		copy(cost, lp.cost)
		for j := range nt {
			// basic columns stay exact: their cost feeds y and would shift
			// every reduced cost past the phase-0 tolerance
			switch st.status[j] {
			case atLower:
				u := 0.5 + float64((uint32(j)*2654435761)%1024)/1024
				cost[j] += 1e-6 * (1 + math.Abs(cost[j])) * u
			case atUpper:
				u := 0.5 + float64((uint32(j)*2654435761)%1024)/1024
				cost[j] -= 1e-6 * (1 + math.Abs(cost[j])) * u
			}
		}
	}

	d := ws.d
	reprice := func() {
		y := ws.y
		clear(y)
		duals(st, cost, m, y)
		for j := range nt {
			if st.status[j] != basic {
				d[j] = lp.reducedCost(y, cost, j)
			} else {
				d[j] = 0
			}
		}
	}
	reprice()

	// phase 0: boxed nonbasics with wrong-signed reduced cost flip to the
	// other bound; any other dual infeasibility means no warm dual start
	var flip0 []int
	for j := range nt {
		switch st.status[j] {
		case atLower:
			if d[j] < -dualTol {
				if lp.ub[j] >= inf {
					lp.Stats.Dual2P0++
					return dual2Bail
				}
				flip0 = append(flip0, j)
			}
		case atUpper:
			if d[j] > dualTol {
				if lp.lb[j] <= -inf {
					lp.Stats.Dual2P0++
					return dual2Bail
				}
				flip0 = append(flip0, j)
			}
		case free:
			if math.Abs(d[j]) > dualTol {
				lp.Stats.Dual2P0++
				return dual2Bail
			}
		}
	}
	if len(flip0) > 0 {
		for _, j := range flip0 {
			if st.status[j] == atLower {
				st.status[j], st.value[j] = atUpper, lp.ub[j]
			} else {
				st.status[j], st.value[j] = atLower, lp.lb[j]
			}
			lp.Stats.Dual2Flips++
		}
		lp.recomputeBasics(st)
	}

	// DSE weights reset to 1 each solve: Clp mode-3 fresh reference frame.
	// (Persisting them across sibling node bases measured worse — stale.)
	w := ws.w
	for i := range w {
		w[i] = 1
	}

	rowR, tau, a, delta := ws.rowR, ws.tau, ws.a, ws.delta
	alphaRow := ws.alphaRow
	clear(alphaRow) // stale from the prior re-solve; run maintains it via touched
	touched := ws.touched[:0]
	seen := ws.seen
	clear(seen)
	cands := ws.cands[:0]
	verified := false // one refactorize+reprice before trusting a certificate

	refresh := func() {
		lp.refactorize(st)
		lp.recomputeBasics(st)
		reprice()
	}

	cap2 := dual2Cap
	if lp.IterCap > 0 && lp.IterCap < cap2 {
		cap2 = lp.IterCap // per-solve heuristic cap (see SetIterCap)
	}
	for iter := range cap2 {
		if iter%64 == 0 && !lp.Deadline.IsZero() && time.Now().After(lp.Deadline) {
			return dual2Bail
		}

		// leaving row: dual steepest edge, max violation^2 / weight
		r, bound := -1, 0.0
		best := 0.0
		for i := range m {
			bv := st.basicOf[i]
			v := st.value[bv]
			var viol, b float64
			if v < lp.lb[bv]-dual2PrimTol {
				viol, b = lp.lb[bv]-v, lp.lb[bv]
			} else if v > lp.ub[bv]+dual2PrimTol {
				viol, b = v-lp.ub[bv], lp.ub[bv]
			} else {
				continue
			}
			if s := viol * viol / w[i]; s > best {
				best, r, bound = s, i, b
			}
		}
		if r < 0 {
			if !verified {
				verified = true
				refresh()
				continue
			}
			lp.Stats.Dual2Opt++
			return dual2Optimal
		}
		leaving := st.basicOf[r]
		needSign := 1.0
		if st.value[leaving] < bound {
			needSign = -1
		}

		// pivot row r over the BTRAN row's support
		clear(rowR)
		rowR[r] = 1
		st.btranVec(rowR)
		for _, j := range touched {
			alphaRow[j] = 0
		}
		touched = touched[:0]
		for i := range m {
			v := rowR[i]
			if math.Abs(v) < 1e-12 {
				continue
			}
			cols, vals := lp.rowCol[i], lp.rowVal[i]
			for k, j32 := range cols {
				j := int(j32)
				if !seen[j] {
					seen[j] = true
					touched = append(touched, j)
				}
				alphaRow[j] += v * vals[k]
			}
		}
		cands = cands[:0]
		for _, j := range touched {
			seen[j] = false
			if st.status[j] == basic {
				continue
			}
			alphaJ := alphaRow[j]
			if math.Abs(alphaJ) < dual2Accept {
				continue
			}
			for _, dir := range enterDirs(st.status[j]) {
				at := alphaJ * dir * needSign
				if at < dual2Accept {
					continue
				}
				rd := max(d[j]*dir, 0)
				cands = append(cands, dual2Cand{j, dir, at, rd, rd / at})
			}
		}
		slices.SortFunc(cands, func(a, b dual2Cand) int {
			switch {
			case a.ratio < b.ratio:
				return -1
			case a.ratio > b.ratio:
				return 1
			}
			return 0
		})

		// bound-flipping walk, then a Harris window: the largest pivot
		// entry inside the first blocker's relaxed ratio wins
		remaining := math.Abs(st.value[leaving] - bound)
		nFlips, pick := 0, -1
		for k := range cands {
			cd := &cands[k]
			rng := lp.ub[cd.j] - lp.lb[cd.j]
			if st.status[cd.j] != free && rng < inf && cd.at*rng < remaining {
				nFlips = k + 1
				remaining -= cd.at * rng
				continue
			}
			relax := (cd.rd + dualTol) / cd.at
			bestA := 0.0
			for k2 := k; k2 < len(cands); k2++ {
				if cands[k2].ratio > relax {
					break
				}
				if cands[k2].at > bestA {
					bestA, pick = cands[k2].at, k2
				}
			}
			break
		}
		if pick < 0 {
			// no admissible pivot and flips cannot absorb the violation:
			// a dual ray — the LP is primal infeasible (verify first)
			if !verified {
				verified = true
				refresh()
				continue
			}
			lp.Stats.Dual2Inf++
			return dual2Infeasible
		}
		q, qdir := cands[pick].j, cands[pick].dir

		if nFlips > 0 {
			clear(delta)
			for k := range nFlips {
				j := cands[k].j
				var dv float64
				if st.status[j] == atLower {
					dv = lp.ub[j] - lp.lb[j]
					st.status[j], st.value[j] = atUpper, lp.ub[j]
				} else {
					dv = lp.lb[j] - lp.ub[j]
					st.status[j], st.value[j] = atLower, lp.lb[j]
				}
				rows, vals := lp.column(j)
				for k2, row := range rows {
					delta[row] += vals[k2] * dv
				}
				lp.Stats.Dual2Flips++
			}
			st.ftranVec(delta)
			for pos := range m {
				st.value[st.basicOf[pos]] -= delta[pos]
			}
		}

		clear(a)
		lp.alpha(st, q, a)
		// FTRAN/BTRAN agreement guard on the pivot element (Clp-style)
		if math.Abs(a[r]-alphaRow[q]) > 1e-7*(1+math.Abs(a[r])) || math.Abs(a[r]) < dual2Accept {
			if !verified {
				verified = true
				refresh()
				continue
			}
			lp.Stats.Dual2Guard++
			return dual2Bail
		}
		verified = false

		ar := a[r]
		if !dual2NoDSE {
			// Clp updateWeights, fresh norm=||beta_r||^2/ar^2 for gamma_r;
			// gamma_i += alpha_i*(alpha_i*norm - tau_i*2/ar) (canonical sign).
			norm := 0.0
			for i := range m {
				norm += rowR[i] * rowR[i]
			}
			norm /= ar * ar
			copy(tau, rowR)
			st.ftranVec(tau)
			mult := 2.0 / ar
			for i := range m {
				if i == r || a[i] == 0 {
					continue
				}
				theta := a[i]
				w[i] = max(w[i]+theta*(theta*norm-tau[i]*mult), 1e-8)
			}
			w[r] = max(norm, 1e-8)
		}

		t := (st.value[leaving] - bound) / (ar * qdir)
		if t < 0 || math.IsInf(t, 0) || math.IsNaN(t) {
			lp.Stats.Dual2TNeg++
			return dual2Bail
		}
		lp.Stats.Dual2++
		lp.pivot(st, q, qdir, a, t, r, false)

		if len(st.etas) == 0 {
			// pivot refactorized: recompute reduced costs from scratch to
			// stop sparse-row truncation drift from accumulating
			reprice()
			continue
		}
		step := d[q] / alphaRow[q]
		for _, j := range touched {
			if alphaRow[j] != 0 {
				d[j] -= step * alphaRow[j]
			}
		}
		d[leaving] = -step
		d[q] = 0
	}
	lp.Stats.Dual2Cap++
	return dual2Bail
}
