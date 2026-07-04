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
	// L and U in flat CSR-style storage: contiguous solve loops instead of
	// per-step slice-header chasing
	lStart []int32
	lIdx   []int32 // L[row, step] multipliers, rows in kernel-local ids
	lVal   []float64
	uDiag  []float64
	uStart []int32
	uIdx   []int32 // U[step, step'] entries in step space, step' > step
	uVal   []float64
	work   []float64
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
		uDiag:     make([]float64, k),
		work:      make([]float64, k),
	}
	lIdxS := make([][]int32, k) // per-step, flattened after elimination
	lValS := make([][]float64, k)
	uIdxS := make([][]int32, k)
	uValS := make([][]float64, k)

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
		uIdxS[step], uValS[step] = uIdx, uVal

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
			lIdxS[step] = append(lIdxS[step], int32(rr))
			lValS[step] = append(lValS[step], mult)

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
	// flatten into CSR-style arrays; remap U column ids to step space
	lu.lStart = make([]int32, k+1)
	lu.uStart = make([]int32, k+1)
	var ln, un int
	for s := range k {
		ln += len(lIdxS[s])
		un += len(uIdxS[s])
	}
	lu.lIdx, lu.lVal = make([]int32, 0, ln), make([]float64, 0, ln)
	lu.uIdx, lu.uVal = make([]int32, 0, un), make([]float64, 0, un)
	for s := range k {
		lu.lStart[s] = int32(len(lu.lIdx))
		lu.lIdx = append(lu.lIdx, lIdxS[s]...)
		lu.lVal = append(lu.lVal, lValS[s]...)
		lu.uStart[s] = int32(len(lu.uIdx))
		for t, c := range uIdxS[s] {
			lu.uIdx = append(lu.uIdx, colPos[c])
			lu.uVal = append(lu.uVal, uValS[s][t])
		}
	}
	lu.lStart[k] = int32(len(lu.lIdx))
	lu.uStart[k] = int32(len(lu.uIdx))
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
			for t := lu.lStart[s]; t < lu.lStart[s+1]; t++ {
				v[lu.lIdx[t]] -= lu.lVal[t] * ys
			}
		}
	}
	for s := k - 1; s >= 0; s-- {
		x := y[s]
		for t := lu.uStart[s]; t < lu.uStart[s+1]; t++ {
			x -= lu.uVal[t] * y[lu.uIdx[t]]
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
			for t := lu.uStart[s]; t < lu.uStart[s+1]; t++ {
				w[lu.colPerm[lu.uIdx[t]]] -= lu.uVal[t] * x
			}
		}
	}
	// L^T backward, gathering from later steps
	for s := k - 1; s >= 0; s-- {
		x := y[s]
		for t := lu.lStart[s]; t < lu.lStart[s+1]; t++ {
			x -= lu.lVal[t] * y[lu.stepOfRow[lu.lIdx[t]]]
		}
		y[s] = x
	}
	for s := range k {
		w[lu.rowPerm[s]] = y[s]
	}
}
