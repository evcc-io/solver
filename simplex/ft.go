package simplex

import "math"

// Forrest-Tomlin basis factorization (Clp CoinFactorization port, component 1;
// see docs/rewrite-clp-core.md). Sparse U-rows, O(nnz) solves and column swap.
type ftLU struct {
	m       int
	lmult   []float64 // lmult[origRow*m+step]: unit-lower L multiplier (dense, fixed)
	lFwd    [][]int32 // step s -> steps t<s with lmult[prow[s]][t] != 0
	lBack   [][]int32 // step s -> steps t>s with lmult[prow[t]][s] != 0
	udiag   []float64 // udiag[s] = U[s][s]
	uCol    [][]int32 // step-row s -> step-cols j>s with a nonzero
	uVal    [][]float64
	prow    []int // prow[step] = original basis row
	pcol    []int // pcol[stepCol] = original basis column
	rinvcol []int // original column -> step-col
	rlist   []ftR
	z       []float64 // solve scratch
	spike   []float64 // replaceColumn scratch (pooled)
	blk     []float64 // trailing-block dense scratch (pooled)
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
		m: m, lmult: make([]float64, m*m), udiag: make([]float64, m),
		uCol: make([][]int32, m), uVal: make([][]float64, m),
		prow: make([]int, m), pcol: make([]int, m), rinvcol: make([]int, m),
		z: make([]float64, m), spike: make([]float64, m), blk: make([]float64, m*m),
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
		f.udiag[step] = piv
		for j := step + 1; j < m; j++ {
			if v := work[pr*m+j]; v != 0 {
				f.uCol[step] = append(f.uCol[step], int32(j))
				f.uVal[step] = append(f.uVal[step], v)
			}
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

// forwardLR computes z = R L^-1 P b in step space.
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
	for s := m - 1; s >= 0; s-- { // U^-1
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
	for s := range m { // U^-T forward by push (row storage)
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
	for s := m - 1; s >= 0; s-- { // L^-T back
		v := z[s]
		for _, t := range f.lBack[s] {
			v -= f.lmult[f.prow[t]*m+s] * z[t]
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
	for s := range m { // spike w = R L^-1 P a
		v := a[f.prow[s]]
		for _, t := range f.lFwd[s] {
			v -= f.lmult[f.prow[s]*m+int(t)] * w[t]
		}
		w[s] = v
	}
	for _, r := range f.rlist {
		w[r.s+1] -= r.mult * w[r.s]
	}
	p := f.rinvcol[col]

	// rows above p: renumber columns (drop p, shift >p down, append spike col)
	for s := 0; s < p; s++ {
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

	// trailing block d[i*bm+j], i,j in [0..bm), global step = p+i / col = p+j
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
	// column shift inside the block: drop col 0 (global p), shift left, spike last
	for i := 0; i < bm; i++ {
		for j := 0; j < bm-1; j++ {
			d[i*bm+j] = d[i*bm+j+1]
		}
		d[i*bm+(bm-1)] = w[p+i]
	}
	// eliminate the Hessenberg subdiagonal top-down
	for j := 0; j < bm-1; j++ {
		sub := d[(j+1)*bm+j]
		if sub == 0 {
			continue
		}
		diag := d[j*bm+j]
		if math.Abs(diag) < 1e-12 {
			return false
		}
		mult := sub / diag
		for c := j; c < bm; c++ {
			d[(j+1)*bm+c] -= mult * d[j*bm+c]
		}
		f.rlist = append(f.rlist, ftR{p + j, mult})
	}
	if math.Abs(d[(bm-1)*bm+(bm-1)]) < 1e-12 {
		return false
	}
	// re-sparsify trailing rows
	for i := 0; i < bm; i++ {
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

	// pcol / rinvcol
	for j := p; j < m-1; j++ {
		f.pcol[j] = f.pcol[j+1]
		f.rinvcol[f.pcol[j]] = j
	}
	f.pcol[m-1], f.rinvcol[col] = col, m-1
	return true
}
