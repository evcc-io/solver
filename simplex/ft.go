package simplex

import "math"

// Forrest-Tomlin basis factorization (Clp CoinFactorization port, component 1;
// see docs/rewrite-clp-core.md). Dense first-cut; sparse kernel + FT update next.
type ftLU struct {
	m     int
	lmult []float64 // lmult[origRow*m+step]: unit-lower L multiplier (step space)
	u     []float64 // u[origRow*m+col]: U[s][c] = u[prow[s]*m+c], upper for c>=s
	prow  []int     // prow[step] = original basis row pivoted at that step
	z     []float64 // solve scratch, indexed by step
}

// newFTLU factors dense basis a (column-major a[col*m+row]) as PB = LU with
// partial row pivoting; nil if singular.
func newFTLU(m int, a []float64) *ftLU {
	f := &ftLU{
		m: m, lmult: make([]float64, m*m), u: make([]float64, m*m),
		prow: make([]int, m), z: make([]float64, m),
	}
	for c := range m {
		for r := range m {
			f.u[r*m+c] = a[c*m+r]
		}
	}
	used := make([]bool, m)
	for step := range m {
		pr, pv := -1, 0.0
		for r := range m {
			if !used[r] {
				if v := math.Abs(f.u[r*m+step]); v > pv {
					pr, pv = r, v
				}
			}
		}
		if pr < 0 || pv < 1e-12 {
			return nil
		}
		used[pr] = true
		f.prow[step] = pr
		piv := f.u[pr*m+step]
		for r := range m {
			if used[r] {
				continue
			}
			mult := f.u[r*m+step] / piv
			if mult == 0 {
				continue
			}
			f.lmult[r*m+step] = mult
			for c := step; c < m; c++ {
				f.u[r*m+c] -= mult * f.u[pr*m+c]
			}
		}
	}
	return f
}

// ftran solves B*x = b in place: b indexed by original row in, x by column out.
func (f *ftLU) ftran(b []float64) {
	m := f.m
	z := f.z
	for s := range m { // z = L^-1 P b
		v := b[f.prow[s]]
		for t := 0; t < s; t++ {
			v -= f.lmult[f.prow[s]*m+t] * z[t]
		}
		z[s] = v
	}
	for s := m - 1; s >= 0; s-- { // x = U^-1 z; b[c] holds x[c] for c>s
		pr := f.prow[s]
		v := z[s]
		for c := s + 1; c < m; c++ {
			v -= f.u[pr*m+c] * b[c]
		}
		b[s] = v / f.u[pr*m+s]
	}
}

// btran solves B^T*y = c in place: c indexed by column in, y by original row out.
func (f *ftLU) btran(c []float64) {
	m := f.m
	w := f.z
	for i := range m { // U^T w = c (lower in step space)
		v := c[i]
		for j := 0; j < i; j++ {
			v -= f.u[f.prow[j]*m+i] * w[j]
		}
		w[i] = v / f.u[f.prow[i]*m+i]
	}
	for s := m - 1; s >= 0; s-- { // L^T v = w, back; y = P^T v (w reused as v)
		v := w[s]
		for t := s + 1; t < m; t++ {
			v -= f.lmult[f.prow[t]*m+s] * w[t]
		}
		w[s] = v
		c[f.prow[s]] = v
	}
}
