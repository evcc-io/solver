// Gomory mixed-integer (GMI) cuts generated at the root from fractional
// basic integer variables, CBC/CglGomory-style.
package mip

import (
	"math"
	"sort"
	"time"

	"cbcgo/problem"
	"cbcgo/simplex"
)

const (
	cutFracMin  = 0.01 // f0 safety window: skip nearly-integral rows
	cutCoefMax  = 1e7  // discard cuts with wild tableau coefficients
	cutDropTol  = 1e-11
	maxCutsPer  = 32 // cuts per round
	maxCutRound = 30 // hard cap; rounds stop early once the bound stalls

	// cutMarginFrac stops cut rounds once a round's bound gain drops below this
	// fraction of the cumulative gain (warm-invariant diminishing-returns test)
	cutMarginFrac = 0.02
)

// probingCuts probes fractional binaries at x and adds violated implication
// rows (CglProbing cut generator): a bound implied by z=v becomes linear.
func (m *Model) probingCuts(x []float64, deadline time.Time) int {
	p := m.P
	n := len(p.Cols)
	inf := problem.Inf
	wl, wu := make([]float64, n), make([]float64, n)
	added := 0
	for j := range n {
		c := &p.Cols[j]
		if added >= 2*maxCutsPer || (!deadline.IsZero() && time.Now().After(deadline)) {
			break
		}
		if !c.Integer || c.LB != 0 || c.UB != 1 {
			continue
		}
		f := x[j]
		if f < intTol || f > 1-intTol {
			continue
		}
		for side := range 2 {
			for k := range n {
				wl[k], wu[k] = p.Cols[k].LB, p.Cols[k].UB
			}
			wl[j], wu[j] = float64(side), float64(side)
			if !propagate(p, wl, wu) {
				// side infeasible: the binary is fixed the other way
				v := 1 - float64(side)
				p.Cols[j].LB, p.Cols[j].UB = v, v
				added++
				break
			}
			for k := range n {
				if k == j || added >= 2*maxCutsPer {
					continue
				}
				l0, u0 := p.Cols[k].LB, p.Cols[k].UB
				scale := math.Max(1, math.Max(math.Abs(l0), math.Abs(u0)))
				// slack absorbs propagation drift; minTight skips noise-level
				// implications whose drift could cut off the optimum
				slack := 1e-7 * scale
				minTight := 1e-4 * scale
				vtol := 1e-6 * scale
				if u0 < inf && wu[k] < u0-minTight {
					ub := wu[k] + slack
					d := u0 - ub
					if side == 1 && x[k]+d*f > u0+vtol {
						// z=1 forces x_k <= u': x_k + (u0-u')z <= u0
						p.AddRow("", []int{k, j}, []float64{1, d}, problem.LE, u0)
						added++
					} else if side == 0 && x[k]-d*f > ub+vtol {
						// z=0 forces x_k <= u': x_k - (u0-u')z <= u'
						p.AddRow("", []int{k, j}, []float64{1, -d}, problem.LE, ub)
						added++
					}
				}
				if l0 > -inf && wl[k] > l0+minTight {
					lb := wl[k] - slack
					d := lb - l0
					if side == 1 && x[k]-d*f < l0-vtol {
						// z=1 forces x_k >= l': x_k - (l'-l0)z >= l0
						p.AddRow("", []int{k, j}, []float64{1, -d}, problem.GE, l0)
						added++
					} else if side == 0 && x[k]+d*f < lb-vtol {
						// z=0 forces x_k >= l': x_k + (l'-l0)z >= l'
						p.AddRow("", []int{k, j}, []float64{1, d}, problem.GE, lb)
						added++
					}
				}
			}
		}
	}
	return added
}

// dropSlackCuts removes cut rows (index >= orig) with slack at the root:
// inactive cuts only bloat node re-solves (CBC keeps active cuts only).
func dropSlackCuts(p *problem.Problem, orig int, rowAct []float64) int {
	keep := make([]int, len(p.Rows))
	w := orig
	for i := orig; i < len(p.Rows); i++ {
		rlb, rub := p.Rows[i].Bounds()
		tol := 1e-6 * math.Max(1, math.Max(math.Abs(rlb), math.Abs(rub)))
		tight := (rub < problem.Inf && rowAct[i] > rub-tol) ||
			(rlb > -problem.Inf && rowAct[i] < rlb+tol)
		if tight {
			keep[i] = w
			p.Rows[w] = p.Rows[i]
			w++
		} else {
			keep[i] = -1
		}
	}
	dropped := len(p.Rows) - w
	if dropped == 0 {
		return 0
	}
	p.Rows = p.Rows[:w]
	for j := range p.Cols {
		c := &p.Cols[j]
		wr := 0
		for pos, ri := range c.Idx {
			ni := ri
			if ri >= orig {
				ni = keep[ri]
			}
			if ni >= 0 {
				c.Idx[wr], c.Coef[wr] = ni, c.Coef[pos]
				wr++
			}
		}
		c.Idx, c.Coef = c.Idx[:wr], c.Coef[:wr]
	}
	return dropped
}

// truncateRows drops rows k and beyond, including their column mirrors.
func truncateRows(p *problem.Problem, k int) {
	p.Rows = p.Rows[:k]
	for j := range p.Cols {
		c := &p.Cols[j]
		w := 0
		for pos, ri := range c.Idx {
			if ri < k {
				c.Idx[w], c.Coef[w] = c.Idx[pos], c.Coef[pos]
				w++
			}
		}
		c.Idx, c.Coef = c.Idx[:w], c.Coef[:w]
	}
}

// gomoryCuts derives GMI cuts from st (an optimal root LP basis) and adds
// them to the problem as GE rows; returns how many were added.
func (m *Model) gomoryCuts(st *simplex.State) int {
	n := len(m.P.Cols)
	mm := m.LP.NumRows()
	nt := n + mm

	// candidate rows: basic structural integer vars, most fractional first
	type cand struct {
		r    int
		frac float64
	}
	var cands []cand
	for r := range mm {
		j, v := st.BasicVar(r)
		if j >= n || !m.P.Cols[j].Integer {
			continue
		}
		f := v - math.Floor(v)
		if f < cutFracMin || f > 1-cutFracMin {
			continue
		}
		cands = append(cands, cand{r, math.Abs(f - 0.5)})
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].frac < cands[b].frac })
	if len(cands) > maxCutsPer {
		cands = cands[:maxCutsPer]
	}

	// derive builds one GMI/MIR cut from a tableau-space equality with
	// integer LHS and fractional rhs f0; reports whether a row was added
	derive := func(tab []float64, f0 float64, maxSup int) bool {
		// build the cut Σ γ_j s_j >= f0 over shifted nonbasics, then
		// unshift into original variables: coef·x >= rhs
		coef := make([]float64, n)
		rhs := f0
		ok := true
		for j := range nt {
			if !ok {
				break
			}
			a := tab[j]
			upper, nonbasic := st.VarStatusAtUpper(j)
			if !nonbasic {
				if st.IsBasic(j) {
					continue // ~e_k tableau entries; the row's own var is f0
				}
				if math.Abs(a) > cutDropTol {
					ok = false // free nonbasic with weight: cannot shift
				}
				continue
			}
			if math.Abs(a) < cutDropTol {
				continue
			}
			if math.Abs(a) > cutCoefMax {
				ok = false
				continue
			}
			at := a
			if upper {
				at = -a
			}
			var g float64
			if j < n && m.P.Cols[j].Integer {
				fj := at - math.Floor(at)
				if fj <= f0 {
					g = fj
				} else {
					g = f0 * (1 - fj) / (1 - f0)
				}
			} else {
				if at > 0 {
					g = at
				} else {
					g = -at * f0 / (1 - f0)
				}
			}
			if g < cutDropTol {
				continue
			}
			// s_j = x_j - lb (lower) or ub - x_j (upper)
			lb, ub := m.LP.Bound(j)
			var sign, bnd float64
			if upper {
				sign, bnd = -1, ub
			} else {
				sign, bnd = 1, lb
			}
			if j < n {
				coef[j] += g * sign
				rhs += g * sign * bnd
			} else {
				// logical var x_{n+i} = row i activity: expand into structurals
				row := &m.P.Rows[j-n]
				for k, cj := range row.Idx {
					coef[cj] += g * sign * row.Coef[k]
				}
				rhs += g * sign * bnd
			}
		}
		if !ok {
			return false
		}
		var idx []int
		var vals []float64
		minAbs, maxAbs := math.Inf(1), 0.0
		for j, c := range coef {
			if math.Abs(c) > cutDropTol {
				idx = append(idx, j)
				vals = append(vals, c)
				minAbs = math.Min(minAbs, math.Abs(c))
				maxAbs = math.Max(maxAbs, math.Abs(c))
			}
		}
		// CBC-style hygiene: reject dense or ill-conditioned cuts, which
		// send the simplex into degenerate grinding; normalize the rest
		if len(idx) == 0 || len(idx) > maxSup || maxAbs/minAbs > 1e8 {
			return false
		}
		for k := range vals {
			vals[k] /= maxAbs
		}
		m.P.AddRow("", idx, vals, problem.GE, rhs/maxAbs)
		return true
	}

	tab := make([]float64, nt)
	added := 0
	for _, cd := range cands {
		_, v := st.BasicVar(cd.r)
		f0 := v - math.Floor(v)
		clear(tab)
		m.LP.TableauRow(st, cd.r, tab)
		if derive(tab, f0, max(200, n/3)) {
			added++
		}
	}

	// TwoMIR-lite (CglTwomir): pairwise +/- aggregations of the most
	// fractional tableau rows; the aggregate LHS stays integer, so the
	// same MIR derivation applies. Big instances only (cf. probing cuts).
	if m.LP.NumRows() > 1500 {
		// D3: aggregate many tableau pairs toward CBC's TwoMir cut count.
		topCap, pairCap := 24, 64
		topN := len(cands)
		if topN > topCap {
			topN = topCap
		}
		tabs := make([][]float64, topN)
		bval := make([]float64, topN)
		for i := range topN {
			tabs[i] = make([]float64, nt)
			m.LP.TableauRow(st, cands[i].r, tabs[i])
			_, bval[i] = st.BasicVar(cands[i].r)
		}
		agg := make([]float64, nt)
		pairAdded := 0
		for a := 0; a < topN && pairAdded < pairCap; a++ {
			for b := a + 1; b < topN && pairAdded < pairCap; b++ {
				for _, sgn := range [2]float64{1, -1} {
					for j := range nt {
						agg[j] = tabs[a][j] + sgn*tabs[b][j]
					}
					v := bval[a] + sgn*bval[b]
					f0 := v - math.Floor(v)
					if f0 < cutFracMin || f0 > 1-cutFracMin {
						continue
					}
					if derive(agg, f0, 64) {
						pairAdded++
					}
				}
			}
		}
		added += pairAdded
	}
	return added
}
