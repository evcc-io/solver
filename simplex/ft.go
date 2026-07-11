package simplex

import (
	"math"
	"os"
)

// ftEnabled wires the Forrest-Tomlin factor into the solve path (experimental).
var ftEnabled = os.Getenv("CBC_FT") == "1"

// clone deep-copies the mutable factor so a branch-and-bound child updates it
// independently of its parent.
func (f *ftLU) clone() *ftLU {
	c := &ftLU{
		m: f.m, udiag: append([]float64(nil), f.udiag...),
		prow: append([]int(nil), f.prow...), pcol: append([]int(nil), f.pcol...),
		rinvrow: append([]int(nil), f.rinvrow...), rinvcol: append([]int(nil), f.rinvcol...),
		rlist: append([]ftR(nil), f.rlist...),
		z:     make([]float64, f.m), spike: make([]float64, f.m), blk: make([]float64, f.m*f.m),
		colBuf: make([]float64, f.m),
		lCol:   make([][]int32, f.m), lVal: make([][]float64, f.m),
		uCol: make([][]int32, f.m), uVal: make([][]float64, f.m),
	}
	for s := range f.m {
		c.lCol[s] = append([]int32(nil), f.lCol[s]...)
		c.lVal[s] = append([]float64(nil), f.lVal[s]...)
		c.uCol[s] = append([]int32(nil), f.uCol[s]...)
		c.uVal[s] = append([]float64(nil), f.uVal[s]...)
	}
	return c
}

// Forrest-Tomlin basis factorization (Clp CoinFactorization port, components
// 1+2; see docs/rewrite-clp-core.md). Sparse L-columns + U-rows, O(nnz) solves.
type ftLU struct {
	m       int
	lCol    [][]int32   // step s -> original rows eliminated at s
	lVal    [][]float64 // matching L multipliers
	udiag   []float64   // udiag[s] = U[s][s]
	uCol    [][]int32   // step-row s -> step-cols j>s with a nonzero
	uVal    [][]float64
	prow    []int // prow[step] = original basis row
	pcol    []int // pcol[stepCol] = original basis column
	rinvrow []int // original row -> step
	rinvcol []int // original column -> step-col
	rlist   []ftR
	z       []float64 // solve scratch
	spike   []float64 // replaceColumn scratch (pooled)
	blk     []float64 // trailing-block dense scratch (pooled)
	colBuf  []float64 // dense entering-column scatter (pivot wiring)
}

// ftR is one update elimination: step-row s+1 -= mult * step-row s.
type ftR struct {
	s    int
	mult float64
}

// newFTLU factors dense basis a (column-major a[col*m+row]); nil if singular.
func newFTLU(m int, a []float64) *ftLU {
	colRow := make([][]int32, m)
	colVal := make([][]float64, m)
	for c := range m {
		for r := range m {
			if v := a[c*m+r]; v != 0 {
				colRow[c] = append(colRow[c], int32(r))
				colVal[c] = append(colVal[c], v)
			}
		}
	}
	return newFTLUSparse(m, colRow, colVal)
}

// newFTLUSparse factors a basis given as sparse columns (colRow[c]/colVal[c]),
// with shortest-row + largest-entry pivoting; nil if singular.
func newFTLUSparse(m int, colRow [][]int32, colVal [][]float64) *ftLU {
	f := &ftLU{
		m: m, udiag: make([]float64, m),
		lCol: make([][]int32, m), lVal: make([][]float64, m),
		uCol: make([][]int32, m), uVal: make([][]float64, m),
		prow: make([]int, m), pcol: make([]int, m),
		rinvrow: make([]int, m), rinvcol: make([]int, m),
		z: make([]float64, m), spike: make([]float64, m), blk: make([]float64, m*m),
		colBuf: make([]float64, m),
	}
	// working sparse rows (by original row); entries are (origCol, val)
	rowCol := make([][]int32, m)
	rowVal := make([][]float64, m)
	for c := range m {
		for k, r := range colRow[c] {
			rowCol[r] = append(rowCol[r], int32(c))
			rowVal[r] = append(rowVal[r], colVal[c][k])
		}
	}
	rowUsed := make([]bool, m)
	colUsed := make([]bool, m)
	acc := make([]float64, m)
	mark := make([]bool, m)
	for step := range m {
		// pivot row: shortest active row with a usable entry
		pr, best := -1, 1<<30
		for r := range m {
			if rowUsed[r] || len(rowCol[r]) >= best {
				continue
			}
			for k, c := range rowCol[r] {
				if !colUsed[c] && math.Abs(rowVal[r][k]) > 1e-12 {
					pr, best = r, len(rowCol[r])
					break
				}
			}
		}
		if pr < 0 {
			return nil
		}
		// pivot col: largest active entry in the row
		pc, pv := int32(-1), 0.0
		for k, c := range rowCol[pr] {
			if !colUsed[c] && math.Abs(rowVal[pr][k]) > math.Abs(pv) {
				pc, pv = c, rowVal[pr][k]
			}
		}
		if pc < 0 || math.Abs(pv) < 1e-12 {
			return nil
		}
		rowUsed[pr], colUsed[pc] = true, true
		f.prow[step], f.pcol[step] = pr, int(pc)
		f.rinvrow[pr], f.rinvcol[pc] = step, step
		f.udiag[step] = pv
		// scatter the pivot row's other active entries (the U row, by origCol)
		var pcols []int32
		for k, c := range rowCol[pr] {
			if c != pc && !colUsed[c] {
				acc[c] = rowVal[pr][k]
				pcols = append(pcols, c)
			}
		}
		// eliminate pivot col from every active row
		for rr := range m {
			if rowUsed[rr] {
				continue
			}
			hit, at := 0.0, -1
			for k, c := range rowCol[rr] {
				if c == pc {
					hit, at = rowVal[rr][k], k
					break
				}
			}
			if at < 0 || hit == 0 {
				continue
			}
			mult := hit / pv
			f.lCol[step] = append(f.lCol[step], int32(rr))
			f.lVal[step] = append(f.lVal[step], mult)
			rc, rv := rowCol[rr], rowVal[rr]
			w := 0
			for k, c := range rc {
				if c == pc || colUsed[c] {
					continue
				}
				v := rv[k]
				if a := acc[c]; a != 0 {
					v -= mult * a
					mark[c] = true
				}
				if v != 0 {
					rc[w], rv[w] = c, v
					w++
				}
			}
			rc, rv = rc[:w], rv[:w]
			for _, c := range pcols {
				if a := acc[c]; a != 0 && !mark[c] {
					if v := -mult * a; v != 0 {
						rc = append(rc, c)
						rv = append(rv, v)
					}
				}
				mark[c] = false
			}
			rowCol[rr], rowVal[rr] = rc, rv
		}
		for _, c := range pcols {
			acc[c] = 0
		}
	}
	f.buildURows(rowCol, rowVal)
	return f
}

// buildURows converts the eliminated working rows into step-col U rows.
func (f *ftLU) buildURows(rowCol [][]int32, rowVal [][]float64) {
	m := f.m
	for s := range m {
		pr := f.prow[s]
		rc, rv := rowCol[pr], rowVal[pr]
		for k, c := range rc {
			jc := f.rinvcol[c]
			if jc > s { // strictly upper in step space
				f.uCol[s] = append(f.uCol[s], int32(jc))
				f.uVal[s] = append(f.uVal[s], rv[k])
			}
		}
	}
}

// forwardLR computes z = R L^-1 P b in step space.
func (f *ftLU) forwardLR(b []float64) {
	m := f.m
	z := f.z
	for s := range m {
		z[s] = b[f.prow[s]]
	}
	for s := range m { // L^-1 forward by column push
		zs := z[s]
		if zs == 0 {
			continue
		}
		lc, lv := f.lCol[s], f.lVal[s]
		for k, r := range lc {
			z[f.rinvrow[r]] -= lv[k] * zs
		}
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
	for s := m - 1; s >= 0; s-- {
		v := z[s]
		uc, uv := f.uCol[s], f.uVal[s]
		for k, j := range uc {
			v -= uv[k] * z[j]
		}
		z[s] = v / f.udiag[s]
	}
	for s := range m {
		b[f.pcol[s]] = z[s]
	}
}

// btran solves B^T*y = c in place: c by basis column in, y by original row out.
func (f *ftLU) btran(c []float64) {
	m := f.m
	z := f.z
	for s := range m {
		z[s] = c[f.pcol[s]]
	}
	for s := range m { // U^-T forward by push
		zs := z[s] / f.udiag[s]
		z[s] = zs
		uc, uv := f.uCol[s], f.uVal[s]
		for k, j := range uc {
			z[j] -= uv[k] * zs
		}
	}
	for i := len(f.rlist) - 1; i >= 0; i-- {
		r := f.rlist[i]
		z[r.s] -= r.mult * z[r.s+1]
	}
	for s := m - 1; s >= 0; s-- { // L^-T back by column
		v := z[s]
		lc, lv := f.lCol[s], f.lVal[s]
		for k, r := range lc {
			v -= lv[k] * z[f.rinvrow[r]]
		}
		z[s] = v
		c[f.prow[s]] = v
	}
}

// replaceColumn swaps basis column `col` for vector a (Forrest-Tomlin); false
// if singular. Trailing rows [p..m-1] update in a dense block, others renumber.
func (f *ftLU) replaceColumn(col int, a []float64) bool {
	m := f.m
	w := f.spike
	for s := range m {
		w[s] = a[f.prow[s]]
	}
	for s := range m { // w = L^-1 P a by column push
		ws := w[s]
		if ws == 0 {
			continue
		}
		for k, r := range f.lCol[s] {
			w[f.rinvrow[r]] -= f.lVal[s][k] * ws
		}
	}
	for _, r := range f.rlist {
		w[r.s+1] -= r.mult * w[r.s]
	}
	p := f.rinvcol[col]

	for s := 0; s < p; s++ { // rows above p: renumber columns
		uc, uv := f.uCol[s], f.uVal[s]
		nc, nv := uc[:0], uv[:0]
		for k, j := range uc {
			if int(j) == p {
				continue
			}
			jj := j
			if int(j) > p {
				jj--
			}
			nc = append(nc, jj)
			nv = append(nv, uv[k])
		}
		if w[s] != 0 {
			nc = append(nc, int32(m-1))
			nv = append(nv, w[s])
		}
		f.uCol[s], f.uVal[s] = nc, nv
	}

	bm := m - p
	d := f.blk[:bm*bm]
	for i := range d {
		d[i] = 0
	}
	for i := 0; i < bm; i++ {
		s := p + i
		d[i*bm+i] = f.udiag[s]
		for k, j := range f.uCol[s] {
			d[i*bm+(int(j)-p)] = f.uVal[s][k]
		}
	}
	for i := 0; i < bm; i++ { // column shift inside block, spike last
		for j := 0; j < bm-1; j++ {
			d[i*bm+j] = d[i*bm+j+1]
		}
		d[i*bm+(bm-1)] = w[p+i]
	}
	for j := 0; j < bm-1; j++ { // eliminate Hessenberg subdiagonal
		sub := d[(j+1)*bm+j]
		if sub == 0 {
			continue
		}
		diag := d[j*bm+j]
		if math.Abs(diag) < 1e-12 {
			return false
		}
		mult := sub / diag
		if math.Abs(mult) > 1e8 { // unstable update (no pivoting): refactorize
			return false
		}
		for c := j; c < bm; c++ {
			d[(j+1)*bm+c] -= mult * d[j*bm+c]
		}
		f.rlist = append(f.rlist, ftR{p + j, mult})
	}
	if math.Abs(d[(bm-1)*bm+(bm-1)]) < 1e-12 {
		return false
	}
	for i := 0; i < bm; i++ { // re-sparsify trailing rows
		s := p + i
		f.udiag[s] = d[i*bm+i]
		nc, nv := f.uCol[s][:0], f.uVal[s][:0]
		for j := i + 1; j < bm; j++ {
			if v := d[i*bm+j]; v != 0 {
				nc = append(nc, int32(p+j))
				nv = append(nv, v)
			}
		}
		f.uCol[s], f.uVal[s] = nc, nv
	}
	for j := p; j < m-1; j++ {
		f.pcol[j] = f.pcol[j+1]
		f.rinvcol[f.pcol[j]] = j
	}
	f.pcol[m-1], f.rinvcol[col] = col, m-1
	return true
}
