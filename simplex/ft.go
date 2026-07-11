package simplex

import "math"

// Forrest-Tomlin basis factorization (Clp CoinFactorization port, component 1;
// see docs/rewrite-clp-core.md). Column swap = shift + Hessenberg elimination.
type ftLU struct {
	m       int
	lmult   []float64 // lmult[origRow*m+step]: unit-lower L multiplier
	ustep   []float64 // ustep[step*m+jcol]: upper triangular in step-col space
	prow    []int     // prow[step] = original basis row
	pcol    []int     // pcol[stepCol] = original basis column at that step-col
	rinvcol []int     // original column -> step-col
	rlist   []ftR     // FT update row-transforms, applied after L^-1 P
	z       []float64 // solve scratch, step-indexed
	spike   []float64 // replaceColumn scratch, pooled (hot path)

	// sparse solve kernel: nonzero indices over the dense factors (O(nnz)
	// solves). L lists built once; U lists rebuilt after each replaceColumn.
	uNZrow [][]int32 // step-row s -> step-cols j>s with ustep[s][j] != 0
	uNZcol [][]int32 // step-col s -> step-rows t<s with ustep[t][s] != 0
	lFwd   [][]int32 // step s -> steps t<s with lmult[prow[s]][t] != 0
	lBack  [][]int32 // step s -> steps t>s with lmult[prow[t]][s] != 0
}

// ftR is one update elimination: step-row s+1 -= mult * step-row s.
type ftR struct {
	s    int
	mult float64
}

// newFTLU factors dense basis a (column-major a[col*m+row]) as PB = LU with
// partial row pivoting; nil if singular.
func newFTLU(m int, a []float64) *ftLU {
	f := &ftLU{
		m: m, lmult: make([]float64, m*m), ustep: make([]float64, m*m),
		prow: make([]int, m), pcol: make([]int, m), rinvcol: make([]int, m),
		z: make([]float64, m), spike: make([]float64, m),
	}
	work := make([]float64, m*m) // work[row*m+col]
	for c := range m {
		for r := range m {
			work[r*m+c] = a[c*m+r]
		}
	}
	used := make([]bool, m)
	for step := range m {
		pr, pv := -1, 0.0
		for r := range m {
			if !used[r] {
				if v := math.Abs(work[r*m+step]); v > pv {
					pr, pv = r, v
				}
			}
		}
		if pr < 0 || pv < 1e-12 {
			return nil
		}
		used[pr] = true
		f.prow[step], f.pcol[step], f.rinvcol[step] = pr, step, step
		piv := work[pr*m+step]
		for j := step; j < m; j++ {
			f.ustep[step*m+j] = work[pr*m+j]
		}
		for r := range m {
			if used[r] {
				continue
			}
			mult := work[r*m+step] / piv
			if mult == 0 {
				continue
			}
			f.lmult[r*m+step] = mult
			for j := step; j < m; j++ {
				work[r*m+j] -= mult * work[pr*m+j]
			}
		}
	}
	f.buildLNZ()
	f.buildUNZ()
	return f
}

// buildLNZ records L's nonzero indices (L is fixed under column swaps).
func (f *ftLU) buildLNZ() {
	m := f.m
	f.lFwd = make([][]int32, m)
	f.lBack = make([][]int32, m)
	for s := range m {
		for t := 0; t < s; t++ {
			if f.lmult[f.prow[s]*m+t] != 0 {
				f.lFwd[s] = append(f.lFwd[s], int32(t))
			}
		}
		for t := s + 1; t < m; t++ {
			if f.lmult[f.prow[t]*m+s] != 0 {
				f.lBack[s] = append(f.lBack[s], int32(t))
			}
		}
	}
}

// buildUNZ (re)builds U's row/column nonzero indices from the dense factor.
func (f *ftLU) buildUNZ() {
	m := f.m
	if f.uNZrow == nil {
		f.uNZrow = make([][]int32, m)
		f.uNZcol = make([][]int32, m)
	}
	for s := range m {
		f.uNZrow[s] = f.uNZrow[s][:0]
		f.uNZcol[s] = f.uNZcol[s][:0]
	}
	for s := range m {
		for j := s + 1; j < m; j++ {
			if f.ustep[s*m+j] != 0 {
				f.uNZrow[s] = append(f.uNZrow[s], int32(j))
				f.uNZcol[j] = append(f.uNZcol[j], int32(s))
			}
		}
	}
}

// forwardLR computes z = R L^-1 P b in step space (shared by ftran).
func (f *ftLU) forwardLR(b []float64) {
	m := f.m
	z := f.z
	for s := range m {
		v := b[f.prow[s]]
		row := f.prow[s] * m
		for _, t := range f.lFwd[s] {
			v -= f.lmult[row+int(t)] * z[t]
		}
		z[s] = v
	}
	for _, r := range f.rlist {
		z[r.s+1] -= r.mult * z[r.s]
	}
}

// ftran solves B*x = b in place: b by original row in, x by basis column out.
func (f *ftLU) ftran(b []float64) {
	m := f.m
	f.forwardLR(b)
	z := f.z
	for s := m - 1; s >= 0; s-- { // U^-1, z becomes x in step space
		v := z[s]
		row := s * m
		for _, j := range f.uNZrow[s] {
			v -= f.ustep[row+int(j)] * z[j]
		}
		z[s] = v / f.ustep[row+s]
	}
	for s := range m {
		b[f.pcol[s]] = z[s]
	}
}

// btran solves B^T*y = c in place: c by basis column in, y by original row out.
func (f *ftLU) btran(c []float64) {
	m := f.m
	z := f.z
	for s := range m { // U^-T (forward), c mapped through pcol
		v := c[f.pcol[s]]
		for _, t := range f.uNZcol[s] {
			v -= f.ustep[int(t)*m+s] * z[t]
		}
		z[s] = v / f.ustep[s*m+s]
	}
	for i := len(f.rlist) - 1; i >= 0; i-- { // R^T, reverse order
		r := f.rlist[i]
		z[r.s] -= r.mult * z[r.s+1]
	}
	for s := m - 1; s >= 0; s-- { // L^-T (back), map through prow
		v := z[s]
		for _, t := range f.lBack[s] {
			v -= f.lmult[f.prow[t]*m+s] * z[t]
		}
		z[s] = v
		c[f.prow[s]] = v
	}
}

// replaceColumn swaps basis column `col` for vector a (original-row indexed),
// updating the factors in place (Forrest-Tomlin); false if it goes singular.
func (f *ftLU) replaceColumn(col int, a []float64) bool {
	m := f.m
	w := f.spike
	for s := range m { // spike w = R L^-1 P a (current U-space)
		v := a[f.prow[s]]
		for t := 0; t < s; t++ {
			v -= f.lmult[f.prow[s]*m+t] * w[t]
		}
		w[s] = v
	}
	for _, r := range f.rlist {
		w[r.s+1] -= r.mult * w[r.s]
	}
	p := f.rinvcol[col]
	for s := range m { // shift step-cols p+1..m-1 left, spike -> last col
		for j := p; j < m-1; j++ {
			f.ustep[s*m+j] = f.ustep[s*m+j+1]
		}
		f.ustep[s*m+(m-1)] = w[s]
	}
	for j := p; j < m-1; j++ {
		f.pcol[j] = f.pcol[j+1]
		f.rinvcol[f.pcol[j]] = j
	}
	f.pcol[m-1], f.rinvcol[col] = col, m-1
	for s := p; s < m-1; s++ { // eliminate Hessenberg subdiagonal
		sub := f.ustep[(s+1)*m+s]
		if sub == 0 {
			continue
		}
		diag := f.ustep[s*m+s]
		if math.Abs(diag) < 1e-12 {
			return false
		}
		mult := sub / diag
		for j := s; j < m; j++ {
			f.ustep[(s+1)*m+j] -= mult * f.ustep[s*m+j]
		}
		f.rlist = append(f.rlist, ftR{s, mult})
	}
	f.buildUNZ() // U changed; L (and its lFwd/lBack) is fixed under swaps
	return math.Abs(f.ustep[(m-1)*m+(m-1)]) >= 1e-12
}
