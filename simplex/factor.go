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
	m       int
	colRow  [][]int32   // basis column sparsity, by basis position
	colVal  [][]float64 //
	fwd     []triPivot  // row-singleton pivots, forward order
	bwd     []triPivot  // col-singleton pivots, applied in reverse
	kRows   []int       // kernel rows (original indices)
	kPos    []int       // kernel basis positions
	klu     *sparseLU   // sparse LU of the kernel
	rowKIdx []int       // row -> kernel row index, -1 otherwise

	// solve scratch, reused across calls (solver is single-threaded);
	// per-call make() showed up as ~8% madvise in profiles
	xScratch, kScratch []float64
}

// factorize builds a factorization of the basis whose pos-th column is
// (colRow[pos], colVal[pos]); returns nil if the kernel is singular.
func factorize(m int, colRow [][]int32, colVal [][]float64) *factor {
	f := &factor{m: m, colRow: colRow, colVal: colVal}

	rowCount := make([]int, m) // active nonzeros per row
	colCount := make([]int, m) // active nonzeros per col (basis position)
	rowActive := make([]bool, m)
	colActive := make([]bool, m)
	// rowCols[r] lists (pos) with a structural entry in row r
	rowCols := make([][]int32, m)
	for i := range rowActive {
		rowActive[i], colActive[i] = true, true
	}
	for pos := range m {
		for k, r := range colRow[pos] {
			if math.Abs(colVal[pos][k]) < pivotTol {
				continue
			}
			rowCount[r]++
			colCount[pos]++
			rowCols[r] = append(rowCols[r], int32(pos))
		}
	}
	colEntry := func(pos int, row int) float64 {
		for k, r := range f.colRow[pos] {
			if int(r) == row {
				return f.colVal[pos][k]
			}
		}
		return 0
	}

	// phase 1: row singletons resolve forward
	queue := make([]int, 0, m)
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
		for k, rr := range f.colRow[pos] {
			if rowActive[rr] && math.Abs(f.colVal[pos][k]) >= pivotTol {
				rowCount[rr]--
				if rowCount[rr] == 1 {
					queue = append(queue, int(rr))
				}
			}
		}
	}

	// recount active cols, then phase 2: column singletons resolve backward
	for pos := range m {
		if !colActive[pos] {
			continue
		}
		n := 0
		for k, r := range f.colRow[pos] {
			if rowActive[r] && math.Abs(f.colVal[pos][k]) >= pivotTol {
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
		for k, r := range f.colRow[pos] {
			if rowActive[r] && math.Abs(f.colVal[pos][k]) >= pivotTol {
				row, a = int(r), f.colVal[pos][k]
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
		colIdx := make([][]int32, k)
		colVal := make([][]float64, k)
		for kc, pos := range f.kPos {
			for kk, r := range f.colRow[pos] {
				if ki := f.rowKIdx[r]; ki >= 0 {
					colIdx[kc] = append(colIdx[kc], int32(ki))
					colVal[kc] = append(colVal[kc], f.colVal[pos][kk])
				}
			}
		}
		f.klu = luFactorize(k, colIdx, colVal)
		if f.klu == nil {
			return nil
		}
	}
	return f
}

// ftranFactor solves B*x = v in place: v becomes x indexed by basis
// position (x[pos] = multiplier of basic column pos).
func (f *factor) ftran(v []float64) {
	if f.xScratch == nil {
		f.xScratch = make([]float64, f.m)
		f.kScratch = make([]float64, len(f.kRows))
	}
	x := f.xScratch // by basis position
	clear(x)
	// forward: row singletons
	for _, tp := range f.fwd {
		xc := v[tp.row] / tp.a
		x[tp.pos] = xc
		if xc != 0 {
			for k, r := range f.colRow[tp.pos] {
				v[r] -= f.colVal[tp.pos][k] * xc
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
				for kk, r := range f.colRow[pos] {
					v[r] -= f.colVal[pos][kk] * s
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
			for k, r := range f.colRow[tp.pos] {
				v[r] -= f.colVal[tp.pos][k] * xc
			}
		}
	}
	copy(v, x)
}

// btran solves B^T*y = w in place: w is indexed by basis position on entry,
// y by row on exit. Ops are the adjoints of ftran's, in reverse order.
func (f *factor) btran(w []float64) {
	if f.xScratch == nil {
		f.xScratch = make([]float64, f.m)
		f.kScratch = make([]float64, len(f.kRows))
	}
	y := f.xScratch // by row
	clear(y)
	// adjoint of backward pass, in forward order
	for _, tp := range f.bwd {
		s := w[tp.pos]
		for k, r := range f.colRow[tp.pos] {
			if int(r) != tp.row {
				s -= f.colVal[tp.pos][k] * y[r]
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
			for kk, r := range f.colRow[pos] {
				if f.rowKIdx[r] < 0 {
					s -= f.colVal[pos][kk] * y[r]
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
		for k, r := range f.colRow[tp.pos] {
			if int(r) != tp.row {
				s -= f.colVal[tp.pos][k] * y[r]
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
