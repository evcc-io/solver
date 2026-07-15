package simplex

import (
	"math"
	"os"
)

// ftEnabled wires the Forrest-Tomlin factor into the solve path (opt-in:
// per-pivot parity with the eta path, but node-path variance loses wall-clock).
var ftEnabled = os.Getenv("CBC_FT") == "1"

// ftPivTol gates Markowitz pivot-column choice: candidate within this fraction
// of the row max. 1.0 = sparsest among max-abs entries (no stability loss).
const ftPivTol = 1.0

// clone deep-copies the mutable factor so a branch-and-bound child updates it
// independently of its parent.
func (f *ftLU) clone() *ftLU {
	c := &ftLU{
		m: f.m, udiag: append([]float64(nil), f.udiag...),
		prow: append([]int(nil), f.prow...), pcol: append([]int(nil), f.pcol...),
		rinvrow: append([]int(nil), f.rinvrow...), rinvcol: append([]int(nil), f.rinvcol...),
		rlist: append([]ftR(nil), f.rlist...),
		nUpd:  f.nUpd,
		lPr:   append([]int(nil), f.lPr...),
		lFR: append([]int32(nil), f.lFR...), lFV: append([]float64(nil), f.lFV...),
		lPtr: append([]int32(nil), f.lPtr...),
		uCol: make([][]int32, f.m), uVal: make([][]float64, f.m),
		ucRows: make([][]int32, f.m),
		// transient scratch is shared (single-threaded, sequential solves);
		// only the factor data above needs an independent copy
		z: f.z, spike: f.spike, colBuf: f.colBuf,
		bacc: f.bacc, bmark: f.bmark, bTouch: f.bTouch,
		wRowCol: f.wRowCol, wRowVal: f.wRowVal, wColRows: f.wColRows, wBuck: f.wBuck,
		wRowUsed: f.wRowUsed, wColUsed: f.wColUsed, wMrk: f.wMrk, wAcc: f.wAcc, wPcols: f.wPcols,
		wColLen: f.wColLen,
	}
	for s := range f.m {
		c.uCol[s] = append([]int32(nil), f.uCol[s]...)
		c.uVal[s] = append([]float64(nil), f.uVal[s]...)
		c.ucRows[s] = append([]int32(nil), f.ucRows[s]...)
	}
	return c
}

// Forrest-Tomlin factorization (Clp port; docs/rewrite-clp-core.md). L/R-file
// keyed by stable orig rows, U by stable basis cols; "step" order remaps freely.
type ftLU struct {
	m       int
	lFR     []int32     // flat L: step s eliminated rows, range [lPtr[s],lPtr[s+1])
	lFV     []float64   // matching L multipliers (contiguous, cache-friendly)
	lPtr    []int32     // step s -> start offset into lFR/lFV (len m+1)
	lPr     []int       // factorize step s -> its pivot orig row (immutable)
	udiag   []float64   // udiag[s] = U[s][s]
	uCol    [][]int32   // step-row s -> basis cols c with rinvcol[c] > s
	uVal    [][]float64
	ucRows  [][]int32 // basis col -> orig rows whose U row may hold it (lazy)
	prow    []int     // prow[step] = orig row
	pcol    []int     // pcol[step] = basis col
	rinvrow []int     // orig row -> step
	rinvcol []int     // basis col -> step
	rlist   []ftR
	nUpd    int       // column updates since the last (re)factorize
	z       []float64 // solve scratch
	spike   []float64 // replaceColumn scratch (pooled)
	colBuf  []float64 // dense entering-column scatter (pivot wiring)

	// row-spike elimination scratch, keyed by basis col (pooled)
	bacc   []float64
	bmark  []bool
	bTouch []int32

	// factorize working scratch, reused across in-place rebuilds
	wRowCol, wColRows, wBuck [][]int32
	wRowVal                  [][]float64
	wRowUsed, wColUsed, wMrk []bool
	wAcc                     []float64
	wPcols                   []int32
	wColLen                  []int // active-row count per column (Markowitz)
}

// ftR is one R-file op in orig-row space: row r -= mult * row s. Orig-row ids
// stay valid across step renumbering, so the list never needs rewriting.
type ftR struct {
	r, s int32
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
// with shortest-row + Markowitz least-fill pivoting; nil if singular.
func newFTLUSparse(m int, colRow [][]int32, colVal [][]float64) *ftLU {
	f := &ftLU{
		m: m, udiag: make([]float64, m),
		lPr:  make([]int, m),
		uCol: make([][]int32, m), uVal: make([][]float64, m),
		ucRows: make([][]int32, m),
		prow:   make([]int, m), pcol: make([]int, m),
		rinvrow: make([]int, m), rinvcol: make([]int, m),
		z: make([]float64, m), spike: make([]float64, m),
		colBuf: make([]float64, m),
		bacc:   make([]float64, m), bmark: make([]bool, m), bTouch: make([]int32, 0, m),
		wRowCol: make([][]int32, m), wRowVal: make([][]float64, m),
		wColRows: make([][]int32, m), wBuck: make([][]int32, m+1),
		wRowUsed: make([]bool, m), wColUsed: make([]bool, m), wMrk: make([]bool, m),
		wAcc: make([]float64, m), wPcols: make([]int32, 0, m), wColLen: make([]int, m),
	}
	if !f.rebuild(colRow, colVal) {
		return nil
	}
	return f
}

// rebuild refactorizes in place, reusing all working buffers (no allocation).
func (f *ftLU) rebuild(colRow [][]int32, colVal [][]float64) bool {
	m := f.m
	f.rlist = f.rlist[:0]
	f.nUpd = 0
	f.lFR, f.lFV, f.lPtr = f.lFR[:0], f.lFV[:0], f.lPtr[:0]
	rowCol, rowVal, colRows := f.wRowCol, f.wRowVal, f.wColRows
	buckets := f.wBuck
	rowUsed, colUsed, mark, acc := f.wRowUsed, f.wColUsed, f.wMrk, f.wAcc
	for r := range m {
		rowCol[r], rowVal[r], colRows[r] = rowCol[r][:0], rowVal[r][:0], colRows[r][:0]
		rowUsed[r], colUsed[r], mark[r], acc[r] = false, false, false, 0
		f.uCol[r], f.uVal[r] = f.uCol[r][:0], f.uVal[r][:0]
		f.ucRows[r] = f.ucRows[r][:0]
		buckets[r] = buckets[r][:0]
	}
	buckets[m] = buckets[m][:0]
	for c := range m {
		for k, r := range colRow[c] {
			rowCol[r] = append(rowCol[r], int32(c))
			rowVal[r] = append(rowVal[r], colVal[c][k])
			colRows[c] = append(colRows[c], r)
		}
	}
	for r := range m {
		buckets[len(rowCol[r])] = append(buckets[len(rowCol[r])], int32(r))
	}
	colLen := f.wColLen
	for c := range m {
		colLen[c] = len(colRows[c])
	}
	minL := 0
	for step := range m {
		f.lPtr = append(f.lPtr, int32(len(f.lFR)))
		// pivot row: shortest active row (via buckets)
		pr := -1
		for minL <= m {
			for len(buckets[minL]) > 0 {
				r := int(buckets[minL][len(buckets[minL])-1])
				buckets[minL] = buckets[minL][:len(buckets[minL])-1]
				if !rowUsed[r] && len(rowCol[r]) == minL {
					pr = r
					break
				}
			}
			if pr >= 0 {
				break
			}
			minL++
		}
		if pr < 0 {
			return false
		}
		// pivot col: among the numerically strongest entries in the row (within
		// ftPivTol of the max), take the sparsest column — Markowitz least-fill.
		maxAbs := 0.0
		for k, c := range rowCol[pr] {
			if !colUsed[c] {
				if av := math.Abs(rowVal[pr][k]); av > maxAbs {
					maxAbs = av
				}
			}
		}
		pc, pv, bestLen := int32(-1), 0.0, 1<<62
		for k, c := range rowCol[pr] {
			if colUsed[c] {
				continue
			}
			av := math.Abs(rowVal[pr][k])
			if av < ftPivTol*maxAbs {
				continue
			}
			if cl := colLen[c]; cl < bestLen || (cl == bestLen && av > math.Abs(pv)) {
				pc, pv, bestLen = c, rowVal[pr][k], cl
			}
		}
		if pc < 0 || math.Abs(pv) < 1e-12 {
			return false
		}
		rowUsed[pr], colUsed[pc] = true, true
		f.prow[step], f.pcol[step] = pr, int(pc)
		f.rinvrow[pr], f.rinvcol[pc] = step, step
		f.udiag[step] = pv
		// scatter the pivot row's other active entries (the U row, by basis col)
		pcols := f.wPcols[:0]
		for k, c := range rowCol[pr] {
			if c != pc && !colUsed[c] {
				acc[c] = rowVal[pr][k]
				pcols = append(pcols, c)
			}
		}
		f.wPcols = pcols
		for _, c := range pcols { // pivot row leaves: its columns lose a row
			colLen[c]--
		}
		// eliminate pivot col only from rows that contain it (lazy colRows:
		// stale rows are skipped when they no longer hold pc)
		for _, rr32 := range colRows[pc] {
			rr := int(rr32)
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
			f.lFR = append(f.lFR, int32(rr))
			f.lFV = append(f.lFV, mult)
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
				} else {
					colLen[c]-- // column cancelled out of this row
				}
			}
			rc, rv = rc[:w], rv[:w]
			for _, c := range pcols {
				if a := acc[c]; a != 0 && !mark[c] {
					if v := -mult * a; v != 0 {
						rc = append(rc, c)
						rv = append(rv, v)
						colRows[c] = append(colRows[c], int32(rr)) // fill
						colLen[c]++
					}
				}
				mark[c] = false
			}
			rowCol[rr], rowVal[rr] = rc, rv
			nl := len(rc)
			buckets[nl] = append(buckets[nl], int32(rr))
			if nl < minL {
				minL = nl
			}
		}
		for _, c := range pcols {
			acc[c] = 0
		}
	}
	f.lPtr = append(f.lPtr, int32(len(f.lFR)))
	copy(f.lPr, f.prow) // L stays keyed by factorize-time pivot rows
	f.buildURows(rowCol, rowVal)
	return true
}

// buildURows converts the eliminated working rows into U rows keyed by basis
// col, and indexes them column-wise in ucRows.
func (f *ftLU) buildURows(rowCol [][]int32, rowVal [][]float64) {
	for s := range f.m {
		pr := f.prow[s]
		rc, rv := rowCol[pr], rowVal[pr]
		for k, c := range rc {
			if f.rinvcol[c] > s { // strictly upper in step space
				f.uCol[s] = append(f.uCol[s], c)
				f.uVal[s] = append(f.uVal[s], rv[k])
				f.ucRows[c] = append(f.ucRows[c], int32(pr))
			}
		}
	}
}

// forwardLR applies R L^-1 P to b in place, in orig-row space.
func (f *ftLU) forwardLR(b []float64) {
	for s := range f.m { // L^-1 forward by column push
		v := b[f.lPr[s]]
		if v == 0 {
			continue
		}
		for k := f.lPtr[s]; k < f.lPtr[s+1]; k++ {
			b[f.lFR[k]] -= f.lFV[k] * v
		}
	}
	for _, op := range f.rlist {
		b[op.r] -= op.mult * b[op.s]
	}
}

// ftran solves B*x = b in place: b by orig row in, x by basis column out.
func (f *ftLU) ftran(b []float64) {
	m := f.m
	f.forwardLR(b)
	x := f.z
	for s := m - 1; s >= 0; s-- { // U backward; x keyed by basis col
		v := b[f.prow[s]]
		uc, uv := f.uCol[s], f.uVal[s]
		for k, c := range uc {
			v -= uv[k] * x[c]
		}
		x[f.pcol[s]] = v / f.udiag[s]
	}
	copy(b, x[:m])
}

// btran solves B^T*y = c in place: c by basis column in, y by orig row out.
func (f *ftLU) btran(c []float64) {
	m := f.m
	for s := range m { // U^-T forward by push, in place on c
		cc := f.pcol[s]
		v := c[cc]
		if v == 0 { // sparse RHS (dual pricing): skip empty steps
			continue
		}
		v /= f.udiag[s]
		c[cc] = v
		uc, uv := f.uCol[s], f.uVal[s]
		for k, j := range uc {
			c[j] -= uv[k] * v
		}
	}
	z := f.z
	for s := range m { // move to orig-row space for the R/L passes
		z[f.prow[s]] = c[f.pcol[s]]
	}
	for i := len(f.rlist) - 1; i >= 0; i-- {
		op := f.rlist[i]
		z[op.s] -= op.mult * z[op.r]
	}
	for s := m - 1; s >= 0; s-- { // L^-T back by column
		pr := f.lPr[s]
		v := z[pr]
		for k := f.lPtr[s]; k < f.lPtr[s+1]; k++ {
			v -= f.lFV[k] * z[f.lFR[k]]
		}
		z[pr] = v
		c[pr] = v
	}
}

// replaceColumn swaps basis column `col` for vector a: the pivot cycles to the
// last step, its row spike eliminated in O(nnz(row)); false = refactorize.
func (f *ftLU) replaceColumn(col int, a []float64) bool {
	m := f.m
	w := f.spike
	copy(w, a[:m])
	f.forwardLR(w) // w = R L^-1 P a, orig-row space

	p := f.rinvcol[col]
	rp := f.prow[p]

	// stash row p (the row spike) into acc, keyed by basis col
	acc, mark, touch := f.bacc, f.bmark, f.bTouch[:0]
	for k, c := range f.uCol[p] {
		acc[c], mark[c] = f.uVal[p][k], true
		touch = append(touch, c)
	}
	savedC, savedV := f.uCol[p][:0], f.uVal[p][:0]

	// drop the leaving column's entries from rows above p (lazy ucRows:
	// stale rows no longer holding col are skipped)
	for _, r32 := range f.ucRows[col] {
		s := f.rinvrow[r32]
		if s == p {
			continue // row p already extracted
		}
		uc, uv := f.uCol[s], f.uVal[s]
		for k, c := range uc {
			if int(c) == col {
				last := len(uc) - 1
				uc[k], uv[k] = uc[last], uv[last]
				f.uCol[s], f.uVal[s] = uc[:last], uv[:last]
				break
			}
		}
	}

	// cycle steps p+1..m-1 up one; the spike takes the last step
	for s := p; s < m-1; s++ {
		f.prow[s] = f.prow[s+1]
		f.rinvrow[f.prow[s]] = s
		f.pcol[s] = f.pcol[s+1]
		f.rinvcol[f.pcol[s]] = s
		f.udiag[s] = f.udiag[s+1]
		f.uCol[s], f.uVal[s] = f.uCol[s+1], f.uVal[s+1]
	}
	f.prow[m-1], f.rinvrow[rp] = rp, m-1
	f.pcol[m-1], f.rinvcol[col] = col, m-1
	f.uCol[m-1], f.uVal[m-1] = savedC, savedV // empty; reuses capacity

	// scatter spike w as the new column: above-diagonal entries into U rows
	rows := f.ucRows[col][:0]
	for r := range m {
		wv := w[r]
		if wv == 0 || r == rp {
			continue
		}
		s := f.rinvrow[r]
		f.uCol[s] = append(f.uCol[s], int32(col))
		f.uVal[s] = append(f.uVal[s], wv)
		rows = append(rows, int32(r))
	}
	f.ucRows[col] = rows
	if !mark[col] {
		mark[col] = true
		touch = append(touch, int32(col))
	}
	acc[col] = w[rp]

	// eliminate the row spike's below-diagonal entries in ascending step
	// order; fill stays inside the row, the R-file records the multipliers
	ok := true
	for s := p; s < m-1; s++ {
		c := f.pcol[s]
		if !mark[c] {
			continue
		}
		v := acc[c]
		acc[c] = 0
		if v == 0 {
			continue
		}
		mult := v / f.udiag[s]
		if math.Abs(mult) > 1e8 { // unstable update: refactorize instead
			ok = false
			break
		}
		f.rlist = append(f.rlist, ftR{r: int32(rp), s: int32(f.prow[s]), mult: mult})
		uc, uv := f.uCol[s], f.uVal[s]
		for k, c2 := range uc {
			if !mark[c2] {
				mark[c2] = true
				touch = append(touch, c2)
			}
			acc[c2] -= mult * uv[k]
		}
	}
	d := acc[col]
	for _, c := range touch {
		acc[c], mark[c] = 0, false
	}
	f.bTouch = touch[:0]
	if !ok || math.Abs(d) < 1e-12 {
		return false
	}
	f.udiag[m-1] = d
	f.nUpd++
	return true
}
