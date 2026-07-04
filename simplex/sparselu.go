// Sparse LU factorization for the kernel block: right-looking elimination
// with threshold pivoting and shortest-row (Markowitz-lite) selection.
package simplex

import "math"

// sparseLU holds PKQ = LU in elimination-step space: L unit-lower stored as
// per-step multiplier columns, U upper stored as per-step rows.
type sparseLU struct {
	k         int
	rowPerm   []int32 // step -> kernel-local row
	colPerm   []int32 // step -> kernel-local col
	stepOfRow []int32 // kernel-local row -> step
	lIdx      [][]int32
	lVal      [][]float64 // L[row, step] multipliers, rows in kernel-local ids
	uDiag     []float64
	uIdx      [][]int32 // U[step, step'] entries, step' > step
	uVal      [][]float64
	work      []float64
}

// luFactorize factors the k*k kernel given as sparse columns in
// kernel-local indices; returns nil when singular.
func luFactorize(k int, colIdx [][]int32, colVal [][]float64) *sparseLU {
	rowIdx := make([][]int32, k)
	rowVal := make([][]float64, k)
	for c := range k {
		for t, r := range colIdx[c] {
			rowIdx[r] = append(rowIdx[r], int32(c))
			rowVal[r] = append(rowVal[r], colVal[c][t])
		}
	}
	rowUsed := make([]bool, k)
	colUsed := make([]bool, k)
	colPos := make([]int32, k)

	lu := &sparseLU{
		k:         k,
		rowPerm:   make([]int32, k),
		colPerm:   make([]int32, k),
		stepOfRow: make([]int32, k),
		lIdx:      make([][]int32, k),
		lVal:      make([][]float64, k),
		uDiag:     make([]float64, k),
		uIdx:      make([][]int32, k),
		uVal:      make([][]float64, k),
		work:      make([]float64, k),
	}

	acc := make([]float64, k)
	touched := make([]int32, 0, k)

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
		uIdx := make([]int32, len(touched))
		uVal := make([]float64, len(touched))
		for t, c := range touched {
			uIdx[t] = c // remapped to steps after the loop
			uVal[t] = acc[c]
		}
		lu.uIdx[step], lu.uVal[step] = uIdx, uVal

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
			lu.lIdx[step] = append(lu.lIdx[step], int32(rr))
			lu.lVal[step] = append(lu.lVal[step], mult)

			nIdx := make([]int32, 0, len(rowIdx[rr])+len(touched))
			nVal := make([]float64, 0, len(rowIdx[rr])+len(touched))
			for t, c := range rowIdx[rr] {
				if c == pc || colUsed[c] {
					continue
				}
				v := rowVal[rr][t]
				if a := acc[c]; a != 0 {
					v -= mult * a
				}
				if math.Abs(v) > etaDropTol {
					nIdx = append(nIdx, c)
					nVal = append(nVal, v)
				}
			}
			// fill-in: pivot-row cols absent from row rr
			for _, c := range touched {
				if a := acc[c]; a != 0 {
					found := false
					for _, cc := range rowIdx[rr] {
						if cc == c {
							found = true
							break
						}
					}
					if !found {
						if v := -mult * a; math.Abs(v) > etaDropTol {
							nIdx = append(nIdx, c)
							nVal = append(nVal, v)
						}
					}
				}
			}
			rowIdx[rr], rowVal[rr] = nIdx, nVal
		}
		for _, c := range touched {
			acc[c] = 0
		}
	}
	for s := range k {
		for t, c := range lu.uIdx[s] {
			lu.uIdx[s][t] = colPos[c]
		}
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
			for t, rr := range lu.lIdx[s] {
				v[rr] -= lu.lVal[s][t] * ys
			}
		}
	}
	for s := k - 1; s >= 0; s-- {
		x := y[s]
		for t, cs := range lu.uIdx[s] {
			x -= lu.uVal[s][t] * y[cs]
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
			for t, cs := range lu.uIdx[s] {
				w[lu.colPerm[cs]] -= lu.uVal[s][t] * x
			}
		}
	}
	// L^T backward, gathering from later steps
	for s := k - 1; s >= 0; s-- {
		x := y[s]
		for t, rr := range lu.lIdx[s] {
			x -= lu.lVal[s][t] * y[lu.stepOfRow[rr]]
		}
		y[s] = x
	}
	for s := range k {
		w[lu.rowPerm[s]] = y[s]
	}
}
