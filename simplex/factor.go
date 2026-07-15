// Basis factorization: singleton triangularization + dense kernel inverse,
// with product-form (eta) updates so pivots cost O(nnz) instead of O(m^2).
package simplex

import (
	"math"
	"os"
	"strconv"
)

const (
	pivotTol   = 1e-10 // singleton pivots below this go to the kernel
	etaDropTol = 1e-12 // eta entries below this are dropped
)

// 64 (was 32): with scaling default-on higher intervals are 021-safe, and 64
// lands 020's fixed-column-eliminated model in a proven 15s basin (32 does not)
var maxEtas = envInt("CBC_MAXETAS", 64)

func envInt(k string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil && v > 0 {
		return v
	}
	return def
}

// triPivot resolves one variable by substitution: basis position pos pivots
// on row row with diagonal a.
type triPivot struct {
	pos, row int32
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
	rowKIdx []int32    // row -> kernel row index, -1 otherwise

	// btran activation graph in solve order (bwd, kernel, reversed fwd):
	// a pivot runs only when its w entry or an earlier written row feeds it
	posSeq  []int32 // basis position -> pivot seq
	rdrHead []int32 // row -> [rdrHead[r], rdrHead[r+1]) into rdrList
	rdrList []int32 // pivot seqs whose column reads that row

	// ftran activation: a written row activates the one pivot that reads it
	// as its pivot row; ftranOwner maps row -> that ftran seq (kernel shares one).
	ftranOwner []int32

	ws *factorWS // shared per-LP solve scratch (solver is single-threaded)
}

// factorWS holds factorize's reusable working arrays plus the solve-time
// scratch shared by every factor of one LP; nil means allocate fresh (the
// solver is single-threaded, so one per LP suffices).
type factorWS struct {
	rowCount, colCount   []int
	rowActive, colActive []bool
	rowCols              [][]int32
	queue                []int

	// solve scratch: xScratch/kScratch are per-call temporaries; ySolve
	// stays all-zero between btranSparse calls (written-rows restore);
	// actGen+gen are the epoch-stamped activation flags
	xScratch, kScratch, ySolve []float64
	actGen                     []int32
	written                    []int32
	gen                        int32

	lu *luWS // kernel LU scratch, reused across refactorizations

	// kernel-column feed for luFactorize (transient: LU copies it out)
	kIdxArena []int32
	kValArena []float64
	kColIdx   [][]int32
	kColVal   [][]float64
}

func newFactorWS(m int) *factorWS {
	fbuf := make([]float64, 3*m)
	ibuf := make([]int32, 2*m+1)
	return &factorWS{
		rowCount: make([]int, m), colCount: make([]int, m),
		rowActive: make([]bool, m), colActive: make([]bool, m),
		rowCols: make([][]int32, m), queue: make([]int, 0, m),
		xScratch: fbuf[:m], kScratch: fbuf[m : 2*m], ySolve: fbuf[2*m:],
		actGen: ibuf[: m+1 : m+1], written: ibuf[m+1:][:0],
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
		f.fwd = append(f.fwd, triPivot{int32(pos), int32(r), a})
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
		f.bwd = append(f.bwd, triPivot{int32(pos), int32(row), a})
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
	f.rowKIdx = make([]int32, m)
	for i := range f.rowKIdx {
		f.rowKIdx[i] = -1
	}
	for r := range m {
		if rowActive[r] {
			f.rowKIdx[r] = int32(len(f.kRows))
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
		if cap(ws.kIdxArena) < nnz {
			ws.kIdxArena = make([]int32, 0, nnz)
			ws.kValArena = make([]float64, 0, nnz)
		}
		if cap(ws.kColIdx) < k {
			ws.kColIdx = make([][]int32, k)
			ws.kColVal = make([][]float64, k)
		}
		idxArena, valArena := ws.kIdxArena[:0], ws.kValArena[:0]
		colIdx, colVal := ws.kColIdx[:k], ws.kColVal[:k]
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
		if ws.lu == nil {
			ws.lu = &luWS{}
		}
		f.klu = luFactorize(k, colIdx, colVal, ws.lu)
		if f.klu == nil {
			return nil
		}
	}
	f.ws = ws
	f.buildBtranGraph()
	return f
}

// buildBtranGraph assigns each pivot its btran solve-order seq and builds
// the row -> reader-pivots CSR used to activate only reachable pivots.
func (f *factor) buildBtranGraph() {
	m, nb, nf := f.m, len(f.bwd), len(f.fwd)
	kernelSeq := int32(nb)
	// one arena for the graph's fixed-size int32 arrays
	arena := make([]int32, m+(m+1))
	f.posSeq = arena[:m:m]
	head := arena[m:]

	for i, tp := range f.bwd {
		f.posSeq[tp.pos] = int32(i)
	}
	for _, pos := range f.kPos {
		f.posSeq[pos] = kernelSeq
	}
	for i, tp := range f.fwd {
		f.posSeq[tp.pos] = int32(nb + 1 + (nf - 1 - i))
	}

	countTri := func(tri []triPivot) {
		for _, tp := range tri {
			for _, r := range f.tblRow[f.cols[tp.pos]] {
				if r != tp.row {
					head[r+1]++
				}
			}
		}
	}
	countTri(f.fwd)
	countTri(f.bwd)
	for _, pos := range f.kPos {
		for _, r := range f.tblRow[f.cols[pos]] {
			if f.rowKIdx[r] < 0 {
				head[r+1]++
			}
		}
	}
	for r := range m {
		head[r+1] += head[r]
	}
	// cursorless CSR fill: advance head[r] itself, then shift it back
	list := make([]int32, head[m])
	fillTri := func(tri []triPivot, seqOf func(i int) int32) {
		for i, tp := range tri {
			s := seqOf(i)
			for _, r := range f.tblRow[f.cols[tp.pos]] {
				if r != tp.row {
					list[head[r]] = s
					head[r]++
				}
			}
		}
	}
	fillTri(f.bwd, func(i int) int32 { return int32(i) })
	fillTri(f.fwd, func(i int) int32 { return int32(nb + 1 + (nf - 1 - i)) })
	for _, pos := range f.kPos {
		for _, r := range f.tblRow[f.cols[pos]] {
			if f.rowKIdx[r] < 0 {
				list[head[r]] = kernelSeq
				head[r]++
			}
		}
	}
	for r := m; r > 0; r-- {
		head[r] = head[r-1]
	}
	head[0] = 0
	f.rdrHead, f.rdrList = head, list

	// ftran solve order: fwd forward (seq i), kernel (seq nf), bwd reversed
	// (bwd[i] fires at seq nf+1+(nb-1-i)); own each row by its resolving seq.
	owner := make([]int32, m)
	for i, tp := range f.fwd {
		owner[tp.row] = int32(i)
	}
	for _, r := range f.kRows {
		owner[r] = int32(nf)
	}
	for i, tp := range f.bwd {
		owner[tp.row] = int32(nf + 1 + (nb - 1 - i))
	}
	f.ftranOwner = owner
}

// ftranFactor solves B*x = v in place: v becomes x indexed by basis
// position (x[pos] = multiplier of basic column pos).
func (f *factor) ftran(v []float64) {
	nnz := 0
	for _, val := range v[:f.m] {
		if val != 0 {
			nnz++
		}
	}
	// same activation-graph payoff threshold as btran: mostly-zero rhs
	if nnz*10 <= f.m {
		f.ftranSparse(v)
		return
	}
	x := f.ws.xScratch // by basis position
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
		kv := f.ws.kScratch[:k]
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

// ftranSparse is ftran firing only activation-reachable pivots; skipped
// pivots would compute exactly 0, so results match the dense path bitwise.
func (f *factor) ftranSparse(v []float64) {
	nf, nb := len(f.fwd), len(f.bwd)
	kernelSeq := int32(nf)
	x := f.ws.xScratch // by basis position
	clear(x)
	f.ws.gen++
	gen := f.ws.gen
	act := f.ws.actGen
	for r := range f.m {
		if v[r] != 0 {
			act[f.ftranOwner[r]] = gen
		}
	}
	// a fired pivot writes its column's rows; activate each written row's owner
	mark := func(pos int32) {
		for _, r := range f.tblRow[f.cols[pos]] {
			act[f.ftranOwner[r]] = gen
		}
	}
	// forward: row singletons
	for i, tp := range f.fwd {
		if act[i] != gen {
			continue
		}
		xc := v[tp.row] / tp.a
		x[tp.pos] = xc
		if xc != 0 {
			cr, cv := f.tblRow[f.cols[tp.pos]], f.tblVal[f.cols[tp.pos]]
			for k, r := range cr {
				v[r] -= cv[k] * xc
			}
			v[tp.row] = 0 // exactly resolved
			mark(tp.pos)
		}
	}
	// kernel: dense solve when reachable (matches btranSparse's solveT)
	if k := len(f.kRows); k > 0 && act[kernelSeq] == gen {
		kv := f.ws.kScratch[:k]
		for ki, r := range f.kRows {
			kv[ki] = v[r]
		}
		f.klu.solve(kv)
		for ki := range k {
			s := kv[ki]
			pos := f.kPos[ki]
			x[pos] = s
			if s != 0 {
				cr, cv := f.tblRow[f.cols[pos]], f.tblVal[f.cols[pos]]
				for kk, r := range cr {
					v[r] -= cv[kk] * s
				}
				mark(int32(pos))
			}
		}
	}
	// backward: col singletons in reverse
	for i := nb - 1; i >= 0; i-- {
		seq := int32(nf + 1 + (nb - 1 - i))
		if act[seq] != gen {
			continue
		}
		tp := f.bwd[i]
		xc := v[tp.row] / tp.a
		x[tp.pos] = xc
		if xc != 0 {
			cr, cv := f.tblRow[f.cols[tp.pos]], f.tblVal[f.cols[tp.pos]]
			for k, r := range cr {
				v[r] -= cv[k] * xc
			}
			mark(tp.pos)
		}
	}
	copy(v, x)
}

// btran solves B^T*y = w in place: w is indexed by basis position on entry,
// y by row on exit. Ops are the adjoints of ftran's, in reverse order.
func (f *factor) btran(w []float64) {
	nnz := 0
	for _, v := range w[:f.m] {
		if v != 0 {
			nnz++
		}
	}
	// hypersparse pay-off needs a mostly-zero rhs; activation bookkeeping
	// costs about one extra column pass per active pivot
	if nnz*10 <= f.m {
		f.btranSparse(w)
		return
	}
	y := f.ws.xScratch // by row
	clear(y)
	// adjoint of backward pass, in forward order
	for _, tp := range f.bwd {
		s := w[tp.pos]
		cr, cv := f.tblRow[f.cols[tp.pos]], f.tblVal[f.cols[tp.pos]]
		for k, r := range cr {
			if r != tp.row {
				s -= cv[k] * y[r]
			}
		}
		y[tp.row] = s / tp.a
	}
	// adjoint of kernel: solve K^T y_K = (w_K - cols^T y_known)
	k := len(f.kRows)
	if k > 0 {
		kw := f.ws.kScratch[:k]
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
			if r != tp.row {
				s -= cv[k] * y[r]
			}
		}
		y[tp.row] = s / tp.a
	}
	copy(w, y)
}

// btranSparse is btran gathering only activation-reachable pivots; skipped
// pivots would compute exactly 0, so results match the dense path bitwise.
func (f *factor) btranSparse(w []float64) {
	y := f.ws.ySolve // by row, all-zero on entry (restored at exit)
	f.ws.gen++
	gen := f.ws.gen
	act := f.ws.actGen
	written := f.ws.written[:0]
	for pos := range f.m {
		if w[pos] != 0 {
			act[f.posSeq[pos]] = gen
		}
	}
	mark := func(row int32) {
		for _, t := range f.rdrList[f.rdrHead[row]:f.rdrHead[row+1]] {
			act[t] = gen
		}
	}
	for i, tp := range f.bwd {
		if act[i] != gen {
			continue
		}
		s := w[tp.pos]
		cr, cv := f.tblRow[f.cols[tp.pos]], f.tblVal[f.cols[tp.pos]]
		for k, r := range cr {
			if r != tp.row {
				s -= cv[k] * y[r]
			}
		}
		if yr := s / tp.a; yr != 0 {
			y[tp.row] = yr
			written = append(written, tp.row)
			mark(tp.row)
		}
	}
	nb := len(f.bwd)
	if k := len(f.kRows); k > 0 && act[nb] == gen {
		kw := f.ws.kScratch[:k]
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
		f.klu.solveT(kw)
		for ki, r := range f.kRows {
			if yr := kw[ki]; yr != 0 {
				y[r] = yr
				written = append(written, int32(r))
				mark(int32(r))
			}
		}
	}
	nf := len(f.fwd)
	for si := range nf {
		if act[nb+1+si] != gen {
			continue
		}
		tp := f.fwd[nf-1-si]
		s := w[tp.pos]
		cr, cv := f.tblRow[f.cols[tp.pos]], f.tblVal[f.cols[tp.pos]]
		for k, r := range cr {
			if r != tp.row {
				s -= cv[k] * y[r]
			}
		}
		if yr := s / tp.a; yr != 0 {
			y[tp.row] = yr
			written = append(written, tp.row)
			mark(tp.row)
		}
	}
	copy(w, y)
	for _, r := range written {
		y[r] = 0
	}
	f.ws.written = written
}

// applyEtas maps x = Binv_factor*v to Binv_current*v (FTRAN direction).
func applyEtas(etas []eta, x []float64) {
	for i := range etas {
		e := &etas[i]
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
func applyEtasT(etas []eta, w []float64) {
	for i := len(etas) - 1; i >= 0; i-- {
		e := &etas[i]
		s := w[e.r]
		for k, j := range e.idx {
			if int(j) != e.r {
				s -= e.val[k] * w[j]
			}
		}
		w[e.r] = s / e.ar
	}
}
