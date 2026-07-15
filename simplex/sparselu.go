// Sparse LU factorization for the kernel block: right-looking elimination
// with threshold pivoting and shortest-row (Markowitz-lite) selection.
package simplex

import "math"

// sparseLU holds PKQ = LU in elimination-step space: L unit-lower stored as
// per-step multiplier columns, U upper stored as per-step rows.
type sparseLU struct {
	k         int
	rowPerm   []int32   // step -> kernel-local row
	colPerm   []int32   // step -> kernel-local col
	stepOfRow []int32   // kernel-local row -> step
	lFlatIdx  []int32   // flat L: step s multipliers in [lPtr[s],lPtr[s+1])
	lFlatVal  []float64 // matching L values, kernel-local row ids
	lPtr      []int32   // step -> start offset into lFlat* (len k+1)
	uDiag     []float64
	uFlatIdx  []int32 // flat U: step s entries in [uPtr[s],uPtr[s+1])
	uFlatVal  []float64
	uPtr      []int32 // step -> start offset into uFlat* (len k+1)
	work      []float64
}

// luWS holds luFactorize's transient scratch, reused across kernel
// refactorizations (one per LP; none of it escapes the call).
type luWS struct {
	cnt     []int
	rowUsed []bool
	colUsed []bool
	colPos  []int32
	acc     []float64
	mark    []bool
	touched []int32
	rowIdx  [][]int32
	rowVal  [][]float64
}

func (w *luWS) reset(k int) {
	if cap(w.cnt) < k {
		w.cnt = make([]int, k)
		w.rowUsed = make([]bool, k)
		w.colUsed = make([]bool, k)
		w.colPos = make([]int32, k)
		w.acc = make([]float64, k)
		w.mark = make([]bool, k)
		w.touched = make([]int32, 0, k)
		w.rowIdx = make([][]int32, k)
		w.rowVal = make([][]float64, k)
	}
	w.cnt, w.rowUsed, w.colUsed = w.cnt[:k], w.rowUsed[:k], w.colUsed[:k]
	w.colPos, w.acc, w.mark = w.colPos[:k], w.acc[:k], w.mark[:k]
	w.rowIdx, w.rowVal = w.rowIdx[:k], w.rowVal[:k]
	clear(w.cnt)
	clear(w.rowUsed)
	clear(w.colUsed)
	clear(w.acc)
	clear(w.mark)
}

// luFactorize factors the k*k kernel given as sparse columns in
// kernel-local indices; returns nil when singular. w reuses scratch across
// calls (pass nil to allocate fresh).
func luFactorize(k int, colIdx [][]int32, colVal [][]float64, w *luWS) *sparseLU {
	if w == nil {
		w = &luWS{}
	}
	w.reset(k)
	rowIdx, rowVal, cnt := w.rowIdx, w.rowVal, w.cnt
	for c := range k {
		for _, r := range colIdx[c] {
			cnt[r]++
		}
	}
	for r := range k {
		rowIdx[r] = rowIdx[r][:0]
		rowVal[r] = rowVal[r][:0]
	}
	for c := range k {
		for t, r := range colIdx[c] {
			rowIdx[r] = append(rowIdx[r], int32(c))
			rowVal[r] = append(rowVal[r], colVal[c][t])
		}
	}
	rowUsed := w.rowUsed
	colUsed := w.colUsed
	colPos := w.colPos

	lu := &sparseLU{
		k:         k,
		rowPerm:   make([]int32, k),
		colPerm:   make([]int32, k),
		stepOfRow: make([]int32, k),
		uDiag:     make([]float64, k),
		lPtr:      make([]int32, k+1),
		uPtr:      make([]int32, k+1),
		work:      make([]float64, k),
	}

	acc := w.acc             // zeroed by reset; restored to zero each step
	mark := w.mark           // pivot-row cols seen in the current row's scan
	touched := w.touched[:0] // grows, backing kept across calls

	for step := range k {
		// pivot row: shortest active row that still has a usable entry
		bestRow, bestLen := int32(-1), 1<<30
		for r := range k {
			if rowUsed[r] || len(rowIdx[r]) >= bestLen {
				continue
			}
			for t, c := range rowIdx[r] {
				if !colUsed[c] && math.Abs(rowVal[r][t]) > pivotTol {
					bestRow, bestLen = int32(r), len(rowIdx[r])
					break
				}
			}
		}
		if bestRow < 0 {
			return nil // singular
		}
		r := bestRow
		// pivot col: largest usable entry in the chosen row
		var pc int32 = -1
		pv := 0.0
		for t, c := range rowIdx[r] {
			if !colUsed[c] && math.Abs(rowVal[r][t]) > math.Abs(pv) {
				pc, pv = c, rowVal[r][t]
			}
		}
		if pc < 0 || math.Abs(pv) < pivotTol {
			return nil
		}
		rowUsed[r], colUsed[pc] = true, true
		lu.rowPerm[step], lu.colPerm[step] = r, pc
		lu.stepOfRow[r] = int32(step)
		colPos[pc] = int32(step)
		lu.uDiag[step] = pv

		// scatter the pivot row's remaining entries
		touched = touched[:0]
		for t, c := range rowIdx[r] {
			if c != pc && !colUsed[c] {
				acc[c] = rowVal[r][t]
				touched = append(touched, c)
			}
		}
		lu.lPtr[step] = int32(len(lu.lFlatIdx))
		lu.uPtr[step] = int32(len(lu.uFlatIdx))
		for _, c := range touched {
			lu.uFlatIdx = append(lu.uFlatIdx, c) // remapped to steps below
			lu.uFlatVal = append(lu.uFlatVal, acc[c])
		}

		// eliminate the pivot column from every remaining row
		for rr := range k {
			if rowUsed[rr] {
				continue
			}
			hit, hitAt := 0.0, -1
			for t, c := range rowIdx[rr] {
				if c == pc {
					hit, hitAt = rowVal[rr][t], t
					break
				}
			}
			if hitAt < 0 || hit == 0 {
				continue
			}
			mult := hit / pv
			lu.lFlatIdx = append(lu.lFlatIdx, int32(rr))
			lu.lFlatVal = append(lu.lFlatVal, mult)

			// rewrite the row in place: the write index never passes the
			// read index, and fill-in only appends after the scan
			row, val := rowIdx[rr], rowVal[rr]
			w := 0
			for t, c := range row {
				if c == pc || colUsed[c] {
					continue
				}
				v := val[t]
				if a := acc[c]; a != 0 {
					v -= mult * a
					mark[c] = true
				}
				if math.Abs(v) > etaDropTol {
					row[w], val[w] = c, v
					w++
				}
			}
			row, val = row[:w], val[:w]
			// fill-in: pivot-row cols absent from the original row rr
			for _, c := range touched {
				if a := acc[c]; a != 0 && !mark[c] {
					if v := -mult * a; math.Abs(v) > etaDropTol {
						row = append(row, c)
						val = append(val, v)
					}
				}
				mark[c] = false
			}
			rowIdx[rr], rowVal[rr] = row, val
		}
		for _, c := range touched {
			acc[c] = 0
		}
	}
	lu.lPtr[k] = int32(len(lu.lFlatIdx))
	lu.uPtr[k] = int32(len(lu.uFlatIdx))
	for t := range lu.uFlatIdx {
		lu.uFlatIdx[t] = colPos[lu.uFlatIdx[t]]
	}
	return lu
}

// solve computes K*x = v in place: v by kernel-local row in, x by
// kernel-local col out.
func (lu *sparseLU) solve(v []float64) {
	k := lu.k
	y := lu.work
	for s := range k {
		ys := v[lu.rowPerm[s]]
		y[s] = ys
		if ys != 0 {
			for t := lu.lPtr[s]; t < lu.lPtr[s+1]; t++ {
				v[lu.lFlatIdx[t]] -= lu.lFlatVal[t] * ys
			}
		}
	}
	for s := k - 1; s >= 0; s-- {
		x := y[s]
		for t := lu.uPtr[s]; t < lu.uPtr[s+1]; t++ {
			x -= lu.uFlatVal[t] * y[lu.uFlatIdx[t]]
		}
		y[s] = x / lu.uDiag[s]
	}
	for s := range k {
		v[lu.colPerm[s]] = y[s]
	}
}

// solveT computes K^T*y = w in place: w by kernel-local col in, y by
// kernel-local row out.
func (lu *sparseLU) solveT(w []float64) {
	k := lu.k
	y := lu.work
	// U^T forward, scattering into later steps' inputs
	for s := range k {
		x := w[lu.colPerm[s]] / lu.uDiag[s]
		y[s] = x
		if x != 0 {
			for t := lu.uPtr[s]; t < lu.uPtr[s+1]; t++ {
				w[lu.colPerm[lu.uFlatIdx[t]]] -= lu.uFlatVal[t] * x
			}
		}
	}
	// L^T backward, gathering from later steps
	for s := k - 1; s >= 0; s-- {
		x := y[s]
		for t := lu.lPtr[s]; t < lu.lPtr[s+1]; t++ {
			x -= lu.lFlatVal[t] * y[lu.stepOfRow[lu.lFlatIdx[t]]]
		}
		y[s] = x
	}
	for s := range k {
		w[lu.rowPerm[s]] = y[s]
	}
}
