package mip

import (
	"math"
	"os"

	"cbcgo/problem"
)

// cglEnabled gates the extra Cgl cut generators (knapsack-cover, clique,
// zero-half — all sound; gated as measured low-value here). Component 4.
var cglEnabled = os.Getenv("CBC_CGL") != "0"

// localCutsEnabled gates in-tree node-local GMI cuts (dev): correct via
// NR-row + subtree activation, but measured net-negative on 020 without
// CBC-style cut-effectiveness pruning; off until that lands.
var localCutsEnabled = os.Getenv("CBC_LOCALCUTS") == "1"

// crunchEnabled (CBC_CRUNCH=1) probes on a row-subset LP (Clp crunch):
// exact bounds, ~25% fewer probe rows on the golden cases, but the changed
// probe roundoff re-rolls 020's tree lottery — opt-in until trees are freer
// of basin sensitivity.
var crunchEnabled = os.Getenv("CBC_CRUNCH") == "1"

// coverItem is a knapsack element with its complemented LP value.
type coverItem struct {
	k    int
	xhat float64
}

// binRowLE returns the row as a pure-binary knapsack a·x <= b with a_j > 0,
// complementing negative coefficients (x -> 1-x); ok=false if the row has a
// non-binary term or is not a <= constraint. compl[k] marks complemented vars.
func (m *Model) binRowLE(r *problem.Row) (idx []int, a []float64, b float64, compl []bool, ok bool) {
	_, ub := r.Bounds()
	if math.IsInf(ub, 1) {
		return nil, nil, 0, nil, false // only finite upper bound (<=) rows
	}
	b = ub
	for k, j := range r.Idx {
		c := &m.P.Cols[j]
		if !c.Integer || c.LB != 0 || c.UB != 1 {
			return nil, nil, 0, nil, false
		}
		coef := r.Coef[k]
		if coef == 0 {
			continue
		}
		cm := false
		if coef < 0 { // complement: coef*x = coef*(1-xbar) => -coef*xbar + coef
			b -= coef
			coef = -coef
			cm = true
		}
		idx = append(idx, j)
		a = append(a, coef)
		compl = append(compl, cm)
	}
	return idx, a, b, compl, len(idx) > 0
}

// knapsackCoverCuts finds minimal covers C (sum a_j > b) violated by x and adds
// the cover inequality sum_{j in C} xhat_j <= |C|-1 (xhat = x or 1-x).
func (m *Model) knapsackCoverCuts(x []float64) int {
	added := 0
	origRows := len(m.P.Rows)
	for ri := 0; ri < origRows; ri++ {
		idx, a, b, compl, ok := m.binRowLE(&m.P.Rows[ri])
		if !ok || b < 0 {
			continue
		}
		// greedily build a cover from the most-fractional-toward-1 xhat
		items := make([]coverItem, len(idx))
		for k := range idx {
			xh := x[idx[k]]
			if compl[k] {
				xh = 1 - xh
			}
			items[k] = coverItem{k, xh}
		}
		// pick items by descending xhat until sum a > b
		sortByXhatDesc(items)
		cover := []int{}
		sumA, sumX := 0.0, 0.0
		for _, e := range items {
			if e.xhat <= 1e-9 {
				break
			}
			cover = append(cover, e.k)
			sumA += a[e.k]
			sumX += e.xhat
			if sumA > b+1e-9 {
				break
			}
		}
		if sumA <= b+1e-9 || len(cover) < 2 {
			continue
		}
		// violated if sum xhat > |C|-1
		if sumX <= float64(len(cover)-1)+1e-6 {
			continue
		}
		// build cut sum xhat_j <= |C|-1 in original x: xhat=x or 1-x
		var ci []int
		var cv []float64
		rhs := float64(len(cover) - 1)
		for _, k := range cover {
			if compl[k] {
				ci = append(ci, idx[k]) // (1-x_j): -x_j, rhs -= 1
				cv = append(cv, -1)
				rhs -= 1
			} else {
				ci = append(ci, idx[k])
				cv = append(cv, 1)
			}
		}
		m.P.AddRow("", ci, cv, problem.LE, rhs)
		added++
	}
	return added
}

// cliqueCuts builds a binary conflict graph (pairs i,j with a_i+a_j>b in a
// binary knapsack, so x_i+x_j<=1) and adds sum_{clique} x_j <= 1 for cliques
// violated by x.
func (m *Model) cliqueCuts(x []float64) int {
	// collect conflict pairs over pure-binary rows
	type pair struct{ i, j int }
	conf := map[pair]bool{}
	origRows := len(m.P.Rows)
	nbin := 0
	for j := range m.P.Cols {
		if c := &m.P.Cols[j]; c.Integer && c.LB == 0 && c.UB == 1 {
			nbin++
		}
	}
	if nbin == 0 {
		return 0
	}
	for ri := 0; ri < origRows; ri++ {
		idx, a, b, compl, ok := m.binRowLE(&m.P.Rows[ri])
		if !ok {
			continue
		}
		for p := 0; p < len(idx); p++ {
			for q := p + 1; q < len(idx); q++ {
				if a[p]+a[q] > b+1e-9 && !compl[p] && !compl[q] {
					i, j := idx[p], idx[q]
					if i > j {
						i, j = j, i
					}
					conf[pair{i, j}] = true
				}
			}
		}
	}
	if len(conf) == 0 {
		return 0
	}
	adj := map[int]map[int]bool{}
	for pr := range conf {
		if adj[pr.i] == nil {
			adj[pr.i] = map[int]bool{}
		}
		if adj[pr.j] == nil {
			adj[pr.j] = map[int]bool{}
		}
		adj[pr.i][pr.j] = true
		adj[pr.j][pr.i] = true
	}
	// greedy clique seeded at each fractional binary, extend by conflict+xhat
	added := 0
	seen := map[int]bool{}
	for seed := range adj {
		if seen[seed] || x[seed] < 1e-6 {
			continue
		}
		clique := []int{seed}
		sum := x[seed]
		for cand := range adj[seed] {
			if x[cand] < 1e-6 {
				continue
			}
			all := true
			for _, c := range clique {
				if !adj[cand][c] {
					all = false
					break
				}
			}
			if all {
				clique = append(clique, cand)
				sum += x[cand]
			}
		}
		if len(clique) >= 2 && sum > 1+1e-6 {
			ci := append([]int(nil), clique...)
			cv := make([]float64, len(clique))
			for k := range cv {
				cv[k] = 1
			}
			m.P.AddRow("", ci, cv, problem.LE, 1)
			added++
			for _, c := range clique {
				seen[c] = true
			}
		}
	}
	return added
}

// liftProjectCuts adds lifted cover inequalities (sequential up-lifting of a
// knapsack cover — a sound lift-and-project family cut).
func (m *Model) liftProjectCuts(x []float64) int {
	added := 0
	origRows := len(m.P.Rows)
	for ri := 0; ri < origRows; ri++ {
		idx, a, b, compl, ok := m.binRowLE(&m.P.Rows[ri])
		if !ok || b < 0 {
			continue
		}
		items := make([]coverItem, len(idx))
		for k := range idx {
			xh := x[idx[k]]
			if compl[k] {
				xh = 1 - xh
			}
			items[k] = coverItem{k, xh}
		}
		sortByXhatDesc(items)
		inCover := make([]bool, len(idx))
		var cw []float64
		sumA := 0.0
		ncov := 0
		for _, e := range items {
			if e.xhat <= 1e-9 {
				break
			}
			inCover[e.k] = true
			cw = append(cw, a[e.k])
			sumA += a[e.k]
			ncov++
			if sumA > b+1e-9 {
				break
			}
		}
		if sumA <= b+1e-9 || ncov < 2 {
			continue
		}
		sortFloats(cw)
		card := ncov - 1
		maxfit := func(cap float64) int {
			if cap < -1e-9 {
				return 0
			}
			s, c := 0.0, 0
			for _, w := range cw {
				if s+w <= cap+1e-9 {
					s += w
					c++
				} else {
					break
				}
			}
			return c
		}
		// build lifted cut in xhat: cover coef 1, non-cover lifted coef
		var ci []int
		var cv []float64
		rhs := float64(card)
		viol := 0.0
		for k := range idx {
			coef := 0.0
			if inCover[k] {
				coef = 1
			} else if al := card - maxfit(b-a[k]); al > 0 {
				coef = float64(al)
			}
			if coef == 0 {
				continue
			}
			xh := x[idx[k]]
			if compl[k] {
				ci = append(ci, idx[k]) // coef*(1-x): -coef x, rhs -= coef
				cv = append(cv, -coef)
				rhs -= coef
				viol += coef * (1 - xh)
			} else {
				ci = append(ci, idx[k])
				cv = append(cv, coef)
				viol += coef * xh
			}
		}
		if viol <= float64(card)+1e-6 {
			continue
		}
		m.P.AddRow("", ci, cv, problem.LE, rhs)
		added++
	}
	return added
}

// sortFloats sorts ascending (small n, insertion sort).
func sortFloats(a []float64) {
	for i := 1; i < len(a); i++ {
		v := a[i]
		j := i - 1
		for j >= 0 && a[j] > v {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = v
	}
}

// zeroHalfCuts adds {0,1/2}-Chvatal-Gomory cuts from pairs of integer-coefficient
// <= rows whose coefficient sum is even and RHS sum is odd.
func (m *Model) zeroHalfCuts(x []float64) int {
	origRows := len(m.P.Rows)
	// candidate rows: <= with all-integer coefficients and finite RHS
	type crow struct {
		idx []int
		a   []int64
		b   int64
	}
	var rows []crow
	for ri := 0; ri < origRows; ri++ {
		r := &m.P.Rows[ri]
		_, ub := r.Bounds()
		if math.IsInf(ub, 1) {
			continue
		}
		ok := true
		a := make([]int64, len(r.Idx))
		for k, coef := range r.Coef {
			// all vars must be integer (else the CG floor is invalid) and
			// coefficients integral
			if !m.P.Cols[r.Idx[k]].Integer || math.Abs(coef-math.Round(coef)) > 1e-9 {
				ok = false
				break
			}
			a[k] = int64(math.Round(coef))
		}
		if !ok || math.Abs(ub-math.Round(ub)) > 1e-9 {
			continue
		}
		rows = append(rows, crow{r.Idx, a, int64(math.Round(ub))})
	}
	added := 0
	for i := 0; i < len(rows) && added < 32; i++ {
		for j := i + 1; j < len(rows) && added < 32; j++ {
			if (rows[i].b+rows[j].b)%2 == 0 {
				continue // need odd RHS sum
			}
			// merged coefficients per column
			merged := map[int]int64{}
			for k, c := range rows[i].idx {
				merged[c] += rows[i].a[k]
			}
			for k, c := range rows[j].idx {
				merged[c] += rows[j].a[k]
			}
			allEven := true
			for _, v := range merged {
				if v%2 != 0 {
					allEven = false
					break
				}
			}
			if !allEven {
				continue
			}
			// cut: sum (v/2) x <= (b_i+b_j-1)/2, check violation
			rhs := float64((rows[i].b + rows[j].b - 1) / 2)
			var ci []int
			var cv []float64
			lhs := 0.0
			for c, v := range merged {
				if v == 0 {
					continue
				}
				h := float64(v / 2)
				ci = append(ci, c)
				cv = append(cv, h)
				lhs += h * x[c]
			}
			if len(ci) == 0 || lhs <= rhs+1e-6 {
				continue
			}
			m.P.AddRow("", ci, cv, problem.LE, rhs)
			added++
		}
	}
	return added
}

// flowCoverCuts adds simple flow-cover inequalities (Padberg-Van Roy-Wolsey)
// for capacity rows sum x_j <= b whose x_j carry VUBs x_j <= u_j y_j.
func (m *Model) flowCoverCuts(x []float64) int {
	vubs := m.detectVUBs()
	if len(vubs) == 0 {
		return 0
	}
	added := 0
	origRows := len(m.P.Rows)
	for ri := 0; ri < origRows; ri++ {
		r := &m.P.Rows[ri]
		_, b := r.Bounds()
		if math.IsInf(b, 1) || len(r.Idx) < 2 {
			continue
		}
		// require every term a continuous +1 non-complemented VUB var
		ok := true
		for k, j := range r.Idx {
			v, has := vubs[j]
			if !has || v.complement || math.Abs(r.Coef[k]-1) > 1e-9 {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}
		// cover C: greedily add until sum u_j > b
		type cj struct {
			j int
			u float64
		}
		var cover []cj
		sumU := 0.0
		for _, j := range r.Idx {
			v := vubs[j]
			cover = append(cover, cj{j, v.m})
			sumU += v.m
			if sumU > b+1e-9 {
				break
			}
		}
		lambda := sumU - b
		if lambda <= 1e-9 || len(cover) < 2 {
			continue
		}
		// cut: sum_{C} [ x_j + (u_j-lambda)^+ (1 - y_j) ] <= b
		var ci []int
		var cv []float64
		rhs := b
		viol := 0.0
		for _, e := range cover {
			ci = append(ci, e.j)
			cv = append(cv, 1)
			viol += x[e.j]
			if g := e.u - lambda; g > 0 {
				y := vubs[e.j].z
				ci = append(ci, y) // +g*(1-y): -g*y, rhs -= g
				cv = append(cv, -g)
				rhs -= g
				viol += g * (1 - x[y])
			}
		}
		if viol <= rhs+1e-6 {
			continue
		}
		m.P.AddRow("", ci, cv, problem.LE, rhs)
		added++
	}
	return added
}

// sortByXhatDesc sorts items by descending xhat (small n, insertion sort).
func sortByXhatDesc(items []coverItem) {
	for a := 1; a < len(items); a++ {
		v := items[a]
		b := a - 1
		for b >= 0 && items[b].xhat < v.xhat {
			items[b+1] = items[b]
			b--
		}
		items[b+1] = v
	}
}
