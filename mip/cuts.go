// Gomory mixed-integer (GMI) cuts generated at the root from fractional
// basic integer variables, CBC/CglGomory-style.
package mip

import (
	"math"
	"sort"

	"cbcgo/problem"
	"cbcgo/simplex"
)

const (
	cutFracMin  = 0.01 // f0 safety window: skip nearly-integral rows
	cutCoefMax  = 1e7  // discard cuts with wild tableau coefficients
	cutDropTol  = 1e-11
	maxCutsPer  = 32 // cuts per round
	maxCutRound = 30 // hard cap; rounds stop early once the bound stalls
)

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

	tab := make([]float64, nt)
	added := 0
	for _, cd := range cands {
		_, v := st.BasicVar(cd.r)
		f0 := v - math.Floor(v)
		clear(tab)
		m.LP.TableauRow(st, cd.r, tab)

		// build the cut Σ γ_j s_j >= f0 over shifted nonbasics, then
		// unshift into original variables: coef·x >= rhs
		coef := make([]float64, n)
		rhs := f0
		ok := true
		for j := 0; j < nt && ok; j++ {
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
			continue
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
		if len(idx) == 0 || len(idx) > max(200, n/3) || maxAbs/minAbs > 1e8 {
			continue
		}
		for k := range vals {
			vals[k] /= maxAbs
		}
		m.P.AddRow("", idx, vals, problem.GE, rhs/maxAbs)
		added++
	}
	return added
}
