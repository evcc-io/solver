// Basis factorization: singleton triangularization + dense kernel inverse,
// with product-form (eta) updates so pivots cost O(nnz) instead of O(m^2).
package simplex

import "math"

const (
	pivotTol   = 1e-10 // singleton pivots below this go to the kernel
	etaDropTol = 1e-12 // eta entries below this are dropped
	maxEtas    = 32    // refactorize when the eta file grows past this
)

// triPivot resolves one variable by substitution: basis position pos pivots
// on row row with diagonal a.
type triPivot struct {
	pos, row int
	a        float64
}

// eta is one product-form update: basis position r was replaced, alpha is
// Binv_old * newcol (sparse), ar its pivot element.
type eta struct {
	r   int
	idx []int32
	val []float64
	ar  float64
}

// factor is an immutable factorization of a basis matrix; shared between
// cloned states (etas live on the State, not here).
type factor struct {
	m    int
	cols []int32 // basis variable per position, indexes tblRow/tblVal
	// shared immutable column tables (owned by the LP, indexed by variable)
	tblRow  [][]int32
	tblVal  [][]float64
	fwd     []triPivot // row-singleton pivots, forward order
	bwd     []triPivot // col-singleton pivots, applied in reverse
	kRows   []int      // kernel rows (original indices)
	kPos    []int      // kernel basis positions
	klu     *sparseLU  // sparse LU of the kernel
	rowKIdx []int      // row -> kernel row index, -1 otherwise

	// solve scratch, reused across calls (solver is single-threaded);
	// per-call make() showed up as ~8% madvise in profiles
	xScratch, kScratch []float64
}

// factorWS holds factorize's reusable working arrays; nil means allocate
// fresh (the solver is single-threaded, so one per LP suffices).
type factorWS struct {
	rowCount, colCount   []int
	rowActive, colActive []bool
	rowCols              [][]int32
	queue                []int
}

func newFactorWS(m int) *factorWS {
	return &factorWS{
		rowCount: make([]int, m), colCount: make([]int, m),
		rowActive: make([]bool, m), colActive: make([]bool, m),
		rowCols: make([][]int32, m), queue: make([]int, 0, m),
	}
}

// factorize builds a factorization of the basis whose pos-th column is
// (tblRow[cols[pos]], tblVal[cols[pos]]); returns nil if the kernel is
// singular. cols is retained by the factor; the tables are shared.
func factorize(m int, cols []int32, tblRow [][]int32, tblVal [][]float64, ws *factorWS) *factor {
	f := &factor{m: m, cols: cols, tblRow: tblRow, tblVal: tblVal}
	// one triPivot arena: fwd fills from the front, bwd continues after it
	tri := make([]triPivot, 0, m)
	f.fwd = tri

	if ws == nil {
		ws = newFactorWS(m)
	}
	rowCount, colCount := ws.rowCount, ws.colCount
	rowActive, colActive := ws.rowActive, ws.colActive
	rowCols := ws.rowCols
	clear(rowCount)
	clear(colCount)
	for i := range rowActive {
		rowActive[i], colActive[i] = true, true
		rowCols[i] = rowCols[i][:0]
	}
	for pos := range m {
		colRow, colVal := tblRow[cols[pos]], tblVal[cols[pos]]
		for k, r := range colRow {
			if math.Abs(colVal[k]) < pivotTol {
				continue
			}
			rowCount[r]++
			colCount[pos]++
			rowCols[r] = append(rowCols[r], int32(pos))
		}
	}
	colEntry := func(pos int, row int) float64 {
		colRow, colVal := tblRow[cols[pos]], tblVal[cols[pos]]
		for k, r := range colRow {
			if int(r) == row {
				return colVal[k]
			}
		}
		return 0
	}

	// phase 1: row singletons resolve forward
	queue := ws.queue[:0]
	for r := range m {
		if rowCount[r] == 1 {
			queue = append(queue, r)
		}
	}
	for len(queue) > 0 {
		r := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		if !rowActive[r] || rowCount[r] != 1 {
			continue
		}
		// find the single active column in row r
		pos := -1
		for _, p := range rowCols[r] {
			if colActive[p] {
				pos = int(p)
				break
			}
		}
		if pos < 0 {
			continue
		}
		a := colEntry(pos, r)
		if math.Abs(a) < pivotTol {
			continue // leave to the kernel
		}
		f.fwd = append(f.fwd, triPivot{pos, r, a})
		rowActive[r], colActive[pos] = false, false
		colRow, colVal := tblRow[cols[pos]], tblVal[cols[pos]]
		for k, rr := range colRow {
			if rowActive[rr] && math.Abs(colVal[k]) >= pivotTol {
				rowCount[rr]--
				if rowCount[rr] == 1 {
					queue = append(queue, int(rr))
				}
			}
		}
	}

	// bwd continues in the tri arena right after fwd's final extent
	f.bwd = f.fwd[len(f.fwd):len(f.fwd):cap(f.fwd)]

	// recount active cols, then phase 2: column singletons resolve backward
	for pos := range m {
		if !colActive[pos] {
			continue
		}
		n := 0
		colRow, colVal := tblRow[cols[pos]], tblVal[cols[pos]]
		for k, r := range colRow {
			if rowActive[r] && math.Abs(colVal[k]) >= pivotTol {
				n++
			}
		}
		colCount[pos] = n
		if n == 1 {
			queue = append(queue, pos)
		}
	}
	for len(queue) > 0 {
		pos := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		if !colActive[pos] || colCount[pos] != 1 {
			continue
		}
		row, a := -1, 0.0
		colRow, colVal := tblRow[cols[pos]], tblVal[cols[pos]]
		for k, r := range colRow {
			if rowActive[r] && math.Abs(colVal[k]) >= pivotTol {
				row, a = int(r), colVal[k]
				break
			}
		}
		if row < 0 {
			continue
		}
		f.bwd = append(f.bwd, triPivot{pos, row, a})
		rowActive[row], colActive[pos] = false, false
		for _, p := range rowCols[row] {
			if colActive[p] {
				colCount[p]--
				if colCount[p] == 1 {
					queue = append(queue, int(p))
				}
			}
		}
	}

	// kernel: whatever remains, inverted densely
	f.rowKIdx = make([]int, m)
	for i := range f.rowKIdx {
		f.rowKIdx[i] = -1
	}
	for r := range m {
		if rowActive[r] {
			f.rowKIdx[r] = len(f.kRows)
			f.kRows = append(f.kRows, r)
		}
	}
	for pos := range m {
		if colActive[pos] {
			f.kPos = append(f.kPos, pos)
		}
	}
	k := len(f.kRows)
	if k != len(f.kPos) {
		return nil // structurally singular
	}
	if k > 0 {
		nnz := 0
		for _, pos := range f.kPos {
			for _, r := range tblRow[cols[pos]] {
				if f.rowKIdx[r] >= 0 {
					nnz++
				}
			}
		}
		idxArena := make([]int32, 0, nnz)
		valArena := make([]float64, 0, nnz)
		colIdx := make([][]int32, k)
		colVal := make([][]float64, k)
		for kc, pos := range f.kPos {
			lo := len(idxArena)
			cr, cv := tblRow[cols[pos]], tblVal[cols[pos]]
			for kk, r := range cr {
				if ki := f.rowKIdx[r]; ki >= 0 {
					idxArena = append(idxArena, int32(ki))
					valArena = append(valArena, cv[kk])
				}
			}
			colIdx[kc] = idxArena[lo:len(idxArena):len(idxArena)]
			colVal[kc] = valArena[lo:len(valArena):len(valArena)]
		}
		f.klu = luFactorize(k, colIdx, colVal)
		if f.klu == nil {
			return nil
		}
	}
	buf := make([]float64, m+k)
	f.xScratch, f.kScratch = buf[:m], buf[m:]
	return f
}

// ftranFactor solves B*x = v in place: v becomes x indexed by basis
// position (x[pos] = multiplier of basic column pos).
func (f *factor) ftran(v []float64) {
	x := f.xScratch // by basis position
	clear(x)
	// forward: row singletons
	for _, tp := range f.fwd {
		xc := v[tp.row] / tp.a
		x[tp.pos] = xc
		if xc != 0 {
			cr, cv := f.tblRow[f.cols[tp.pos]], f.tblVal[f.cols[tp.pos]]
			for k, r := range cr {
				v[r] -= cv[k] * xc
			}
			v[tp.row] = 0 // exactly resolved
		}
	}
	// kernel
	k := len(f.kRows)
	if k > 0 {
		kv := f.kScratch
		for ki, r := range f.kRows {
			kv[ki] = v[r]
		}
		f.klu.solve(kv) // kv now holds x by kernel-local col
		for ki := range k {
			s := kv[ki]
			pos := f.kPos[ki]
			x[pos] = s
			if s != 0 {
				cr, cv := f.tblRow[f.cols[pos]], f.tblVal[f.cols[pos]]
				for kk, r := range cr {
					v[r] -= cv[kk] * s
				}
			}
		}
	}
	// backward: col singletons in reverse
	for i := len(f.bwd) - 1; i >= 0; i-- {
		tp := f.bwd[i]
		xc := v[tp.row] / tp.a
		x[tp.pos] = xc
		if xc != 0 {
			cr, cv := f.tblRow[f.cols[tp.pos]], f.tblVal[f.cols[tp.pos]]
			for k, r := range cr {
				v[r] -= cv[k] * xc
			}
		}
	}
	copy(v, x)
}

// btran solves B^T*y = w in place: w is indexed by basis position on entry,
// y by row on exit. Ops are the adjoints of ftran's, in reverse order.
func (f *factor) btran(w []float64) {
	y := f.xScratch // by row
	clear(y)
	// adjoint of backward pass, in forward order
	for _, tp := range f.bwd {
		s := w[tp.pos]
		cr, cv := f.tblRow[f.cols[tp.pos]], f.tblVal[f.cols[tp.pos]]
		for k, r := range cr {
			if int(r) != tp.row {
				s -= cv[k] * y[r]
			}
		}
		y[tp.row] = s / tp.a
	}
	// adjoint of kernel: solve K^T y_K = (w_K - cols^T y_known)
	k := len(f.kRows)
	if k > 0 {
		kw := f.kScratch
		for ki, pos := range f.kPos {
			s := w[pos]
			cr, cv := f.tblRow[f.cols[pos]], f.tblVal[f.cols[pos]]
			for kk, r := range cr {
				if f.rowKIdx[r] < 0 {
					s -= cv[kk] * y[r]
				}
			}
			kw[ki] = s
		}
		f.klu.solveT(kw) // kw now holds y by kernel-local row
		for ki, r := range f.kRows {
			y[r] = kw[ki]
		}
	}
	// adjoint of forward pass, in reverse order
	for i := len(f.fwd) - 1; i >= 0; i-- {
		tp := f.fwd[i]
		s := w[tp.pos]
		cr, cv := f.tblRow[f.cols[tp.pos]], f.tblVal[f.cols[tp.pos]]
		for k, r := range cr {
			if int(r) != tp.row {
				s -= cv[k] * y[r]
			}
		}
		y[tp.row] = s / tp.a
	}
	copy(w, y)
}

// applyEtas maps x = Binv_factor*v to Binv_current*v (FTRAN direction).
func applyEtas(etas []*eta, x []float64) {
	for _, e := range etas {
		xr := x[e.r] / e.ar
		x[e.r] = xr
		if xr == 0 {
			continue
		}
		for k, i := range e.idx {
			if int(i) != e.r {
				x[i] -= e.val[k] * xr
			}
		}
	}
}

// applyEtasT pre-transforms w for BTRAN: adjoints in reverse order.
func applyEtasT(etas []*eta, w []float64) {
	for i := len(etas) - 1; i >= 0; i-- {
		e := etas[i]
		s := w[e.r]
		for k, j := range e.idx {
			if int(j) != e.r {
				s -= e.val[k] * w[j]
			}
		}
		w[e.r] = s / e.ar
	}
}
