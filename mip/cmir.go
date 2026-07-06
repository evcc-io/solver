// c-MIR cuts (Marchand-Wolsey) from aggregated equality chains with exact
// variable-bound substitution (x = M*z - slack, slack re-substituted after
// MIR). Measured on the golden cases: the sound derivation separates zero
// violated cuts at the root — GMI + probing already dominate this cut
// space — so the generator is not wired into the cut loop. It stays, with
// its soundness property test, as the recipe for future aggregation work
// (an earlier draft that substituted the VUB as an inequality was subtly
// unsound and cut off optima; see TestCMIRSoundness).
package mip

import (
	"math"

	"cbcgo/problem"
)

// vub captures a variable upper bound detected from a 2-var row:
// direct: x <= M*z (row x - M z <= 0); complemented: x <= M*(1-z)
// (row x + M z <= M).
type vub struct {
	z          int
	m          float64
	complement bool
}

// detectVUBs scans 2-var LE rows pairing one continuous with one binary.
func (mo *Model) detectVUBs() map[int]vub {
	p := mo.P
	out := make(map[int]vub)
	for ri := range p.Rows {
		r := &p.Rows[ri]
		if len(r.Idx) != 2 || r.Sense != problem.LE {
			continue
		}
		for k := range 2 {
			cj, zj := r.Idx[k], r.Idx[1-k]
			ac, az := r.Coef[k], r.Coef[1-k]
			cc, cz := &p.Cols[cj], &p.Cols[zj]
			if cc.Integer || !cz.Integer || cz.LB != 0 || cz.UB != 1 || ac <= 0 {
				continue
			}
			rhs := r.RHS
			switch {
			case rhs == 0 && az < 0:
				// ac*x - |az| z <= 0  =>  x <= (|az|/ac) z
				out[cj] = vub{z: zj, m: -az / ac}
			case rhs > 0 && az > 0 && math.Abs(az-rhs) < 1e-9*math.Abs(rhs):
				// ac*x + M z <= M  =>  x <= (M/ac)(1-z)
				out[cj] = vub{z: zj, m: az / ac, complement: true}
			}
		}
	}
	return out
}

// eqChains links equality rows through continuous variables that appear in
// exactly two equality rows, returning maximal chains of row indices.
func (mo *Model) eqChains() [][]int {
	p := mo.P
	eqOf := make(map[int][]int) // continuous col -> EQ rows containing it
	for ri := range p.Rows {
		r := &p.Rows[ri]
		if r.Sense != problem.EQ {
			continue
		}
		for _, cj := range r.Idx {
			if !p.Cols[cj].Integer {
				eqOf[cj] = append(eqOf[cj], ri)
			}
		}
	}
	adj := make(map[int][]int) // EQ row -> neighbor EQ rows
	for _, rows := range eqOf {
		if len(rows) == 2 {
			adj[rows[0]] = append(adj[rows[0]], rows[1])
			adj[rows[1]] = append(adj[rows[1]], rows[0])
		}
	}
	visited := make(map[int]bool)
	extend := func(chain []int) []int {
		cur := chain[len(chain)-1]
		for {
			next := -1
			for _, nb := range adj[cur] {
				if !visited[nb] {
					next = nb
					break
				}
			}
			if next < 0 {
				return chain
			}
			chain = append(chain, next)
			visited[next] = true
			cur = next
		}
	}
	var chains [][]int
	for start := range adj {
		if visited[start] {
			continue
		}
		visited[start] = true
		// grow forward, then reverse and grow the other way: a maximal
		// path through the (multi)graph regardless of node degrees
		chain := extend([]int{start})
		for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
			chain[i], chain[j] = chain[j], chain[i]
		}
		chain = extend(chain)
		if len(chain) >= 2 {
			chains = append(chains, chain)
		}
	}
	return chains
}

// linkVar returns the continuous variable shared by two equality rows.
func linkVar(p *problem.Problem, r1, r2 int) int {
	in1 := make(map[int]bool, len(p.Rows[r1].Idx))
	for _, j := range p.Rows[r1].Idx {
		if !p.Cols[j].Integer {
			in1[j] = true
		}
	}
	for _, j := range p.Rows[r2].Idx {
		if in1[j] {
			return j
		}
	}
	return -1
}

// cmirCuts aggregates chain windows, substitutes VUBs, applies MIR with a
// divisor/complement search, and adds violated cuts. Returns rows added.
var cmirDbg struct{ windows, bails, derives, noBin, f0rej, violrej int }

// rowMIRCuts derives single-row MIR cuts (CglMixedIntegerRounding2-style)
// from the first nRows original rows: no aggregation, just VUB/bound
// substitution and the divisor search on each row in both directions.
func (mo *Model) rowMIRCuts(x []float64, nRows int) int {
	p := mo.P
	vubs := mo.detectVUBs()
	cmirDbg = struct{ windows, bails, derives, noBin, f0rej, violrej int }{}
	added := 0
	agg := map[int]float64{}
	for ri := 0; ri < nRows && added < maxCutsPer; ri++ {
		r := &p.Rows[ri]
		clear(agg)
		for pos, j := range r.Idx {
			agg[j] += r.Coef[pos]
		}
		rl, ru := r.Bounds()
		if ru < problem.Inf {
			if mo.cmirDerive(agg, ru, 1, vubs, x) {
				added++
			}
		}
		if rl > -problem.Inf {
			if mo.cmirDerive(agg, rl, -1, vubs, x) {
				added++
			}
		}
	}
	debugf("rowmir: vubs=%d derives=%d noBin=%d f0rej=%d violrej=%d added=%d",
		len(vubs), cmirDbg.derives, cmirDbg.noBin, cmirDbg.f0rej, cmirDbg.violrej, added)
	return added
}

func (mo *Model) cmirCuts(x []float64) int {
	vubs := mo.detectVUBs()
	if len(vubs) == 0 {
		debugf("cmir: no VUBs detected")
		return 0
	}
	chains := mo.eqChains()
	cmirDbg = struct{ windows, bails, derives, noBin, f0rej, violrej int }{}
	added := 0
	for _, chain := range chains {
		for _, win := range [3]int{2, 4, 8} {
			for at := 0; at+win <= len(chain) && added < maxCutsPer; at += win / 2 {
				cmirDbg.windows++
				added += mo.cmirWindow(chain[at:at+win], vubs, x)
			}
		}
	}
	debugf("cmir: vubs=%d chains=%d windows=%d bails=%d derives=%d noBin=%d f0rej=%d violrej=%d added=%d",
		len(vubs), len(chains), cmirDbg.windows, cmirDbg.bails, cmirDbg.derives, cmirDbg.noBin, cmirDbg.f0rej, cmirDbg.violrej, added)
	return added
}

// cmirWindow aggregates one window of equality rows (cancelling the link
// variables), tries both inequality directions with VUB substitution and a
// small divisor search, and adds the best violated MIR cut, if any.
func (mo *Model) cmirWindow(rows []int, vubs map[int]vub, x []float64) int {
	p := mo.P
	agg := map[int]float64{}
	rhs := 0.0
	mult := 1.0
	for k, ri := range rows {
		r := &p.Rows[ri]
		if k > 0 {
			lv := linkVar(p, rows[k-1], ri)
			if lv < 0 {
				cmirDbg.bails++
				return 0
			}
			cPrev, cCur := agg[lv], 0.0
			for pos, j := range r.Idx {
				if j == lv {
					cCur = r.Coef[pos]
				}
			}
			if cCur == 0 || cPrev == 0 {
				cmirDbg.bails++
				return 0
			}
			mult = -cPrev / cCur
		}
		rl, ru := r.Bounds()
		if rl != ru {
			cmirDbg.bails++
			return 0 // equality expected
		}
		rhs += mult * rl
		for pos, j := range r.Idx {
			agg[j] += mult * r.Coef[pos]
		}
	}
	// two directions: agg <= rhs and -agg <= -rhs
	best := 0
	for _, dir := range [2]float64{1, -1} {
		if mo.cmirDerive(agg, rhs, dir, vubs, x) {
			best++
		}
	}
	return best
}

// cmirDerive substitutes VUBs/bounds to reach a binary knapsack plus bounded
// continuous rest, applies MIR over a divisor search, adds if violated.
// vubSub records one exact substitution x = M*z - s (direct) or
// x = M*(1-z) - s (complemented) applied with row coefficient c > 0;
// the slack s >= 0 rides through MIR and is re-substituted afterwards.
type vubSub struct {
	j int
	v vub
	c float64
}

func (mo *Model) cmirDerive(agg map[int]float64, rhs, dir float64, vubs map[int]vub, x []float64) bool {
	p := mo.P
	// working row: sum coef*x <= b; keep negative-coef continuous terms
	// (they strengthen the MIR); relax positive ones at their lower bound
	b := dir * rhs
	binCoef := map[int]float64{}
	contNeg := map[int]float64{}
	var subs []vubSub
	for j, cRaw := range agg {
		c := dir * cRaw
		if c == 0 {
			continue
		}
		col := &p.Cols[j]
		if col.Integer {
			binCoef[j] += c
			continue
		}
		if v, ok := vubs[j]; ok && c > 0 && col.LB == 0 {
			// exact substitution x = M*z - s (s >= 0 by the VUB row):
			// c*x = c*M*z - c*s; complemented form folds the constant c*M
			if v.complement {
				binCoef[v.z] -= c * v.m
				b -= c * v.m
			} else {
				binCoef[v.z] += c * v.m
			}
			subs = append(subs, vubSub{j: j, v: v, c: c})
			continue
		}
		if c > 0 {
			if col.LB <= -problem.Inf {
				return false
			}
			b -= c * col.LB // relax: c*xj >= c*lb
		} else {
			contNeg[j] += c
		}
	}

	// divisor search over distinct |binary coefficients|
	seen := map[float64]bool{}
	var deltas []float64
	for _, c := range binCoef {
		d := math.Abs(c)
		if d > 1e-9 && !seen[d] {
			seen[d] = true
			deltas = append(deltas, d)
		}
	}
	cmirDbg.derives++
	if len(deltas) == 0 {
		cmirDbg.noBin++
	}
	for _, delta := range deltas {
		if mo.mirMixed(binCoef, contNeg, subs, b, delta, x) {
			return true
		}
	}
	return false
}

// mirMixed applies MIR to sum a_j z_j + sum c_k x_k <= b (z binary, c_k<0,
// x_k >= 0 continuous) with divisor delta, complementing binaries whose LP
// value exceeds one half. The continuous part joins scaled by 1/(1-f0).
func (mo *Model) mirMixed(binCoef, contNeg map[int]float64, subs []vubSub, b, delta float64, x []float64) bool {
	p := mo.P
	// complement set: z' = 1-z for x[z] > 0.5
	work := map[int]float64{}
	rhs := b
	comp := map[int]bool{}
	for j, c := range binCoef {
		if x[j] > 0.5 {
			comp[j] = true
			rhs -= c
			work[j] = -c
		} else {
			work[j] = c
		}
	}
	f0 := rhs/delta - math.Floor(rhs/delta)
	if f0 < 0.05 || f0 > 0.95 {
		cmirDbg.f0rej++
		return false
	}
	oneMinus := 1 - f0
	mirC := map[int]float64{}
	lhsAt := 0.0
	for j, c := range work {
		a := c / delta
		fa := a - math.Floor(a)
		var g float64
		if fa <= f0 {
			g = math.Floor(a)
		} else {
			g = math.Floor(a) + (fa-f0)/oneMinus
		}
		mirC[j] = g
		zv := x[j]
		if comp[j] {
			zv = 1 - zv
		}
		lhsAt += g * zv
	}
	// negative continuous part: coefficient c/(delta*(1-f0)) stays in the cut
	contC := map[int]float64{}
	for k, c := range contNeg {
		g := c / (delta * oneMinus)
		contC[k] += g
		lhsAt += g * x[k]
	}
	// VUB slacks (-c*s terms, s = M*z - x or M*(1-z) - x) join the same
	// way, then re-substitute back into the original variables
	binExtra := map[int]float64{}
	rhsShift := 0.0
	for _, sb := range subs {
		ks := -sb.c / (delta * oneMinus) // slack coefficient in the cut
		var sv float64
		if sb.v.complement {
			// s = M - M*z - x: ks*s = ks*M - ks*M*z - ks*x
			rhsShift -= ks * sb.v.m
			binExtra[sb.v.z] -= ks * sb.v.m
			contC[sb.j] -= ks
			sv = sb.v.m*(1-x[sb.v.z]) - x[sb.j]
		} else {
			// s = M*z - x: ks*s = ks*M*z - ks*x
			binExtra[sb.v.z] += ks * sb.v.m
			contC[sb.j] -= ks
			sv = sb.v.m*x[sb.v.z] - x[sb.j]
		}
		lhsAt += ks * sv
	}
	mirRHS := math.Floor(rhs / delta)
	if lhsAt <= mirRHS+rhsShift+1e-4 {
		cmirDbg.violrej++
		return false // not violated
	}
	// uncomplement back to original variables
	var idx []int
	var vals []float64
	finalRHS := mirRHS + rhsShift
	binOut := map[int]float64{}
	for j, g := range mirC {
		if g == 0 {
			continue
		}
		if comp[j] {
			finalRHS -= g
			binOut[j] -= g
		} else {
			binOut[j] += g
		}
	}
	for j, g := range binExtra {
		binOut[j] += g
	}
	for j, g := range binOut {
		if g != 0 {
			idx = append(idx, j)
			vals = append(vals, g)
		}
	}
	for k, g := range contC {
		if g != 0 {
			idx = append(idx, k)
			vals = append(vals, g)
		}
	}
	if len(idx) == 0 || len(idx) > 64 {
		return false
	}
	minAbs, maxAbs := math.Inf(1), 0.0
	for _, v := range vals {
		minAbs = math.Min(minAbs, math.Abs(v))
		maxAbs = math.Max(maxAbs, math.Abs(v))
	}
	if maxAbs < 1e-9 || maxAbs/minAbs > 1e8 {
		return false
	}
	for k := range vals {
		vals[k] /= maxAbs
	}
	p.AddRow("", idx, vals, problem.LE, finalRHS/maxAbs)
	return true
}
