package simplex

import (
	"math"
	"math/rand"
	"testing"
)

// randomBasis makes a nonsingular sparse m*m matrix shaped like real bases:
// mostly singleton/chain columns plus a denser kernel block.
func randomBasis(rng *rand.Rand, m int) ([][]int32, [][]float64) {
	colRow := make([][]int32, m)
	colVal := make([][]float64, m)
	perm := rng.Perm(m)
	for pos := 0; pos < m; pos++ {
		r := perm[pos]
		// guaranteed diagonal under permutation keeps it nonsingular-ish
		rows := []int32{int32(r)}
		vals := []float64{1 + rng.Float64()}
		seen := map[int32]bool{int32(r): true}
		for extra := rng.Intn(3); extra > 0; extra-- {
			rr := int32(rng.Intn(m))
			if !seen[rr] {
				seen[rr] = true
				rows = append(rows, rr)
				vals = append(vals, rng.NormFloat64())
			}
		}
		colRow[pos], colVal[pos] = rows, vals
	}
	return colRow, colVal
}

func multiply(colRow [][]int32, colVal [][]float64, m int, x []float64) []float64 {
	v := make([]float64, m)
	for pos := 0; pos < m; pos++ {
		if x[pos] == 0 {
			continue
		}
		for k, r := range colRow[pos] {
			v[r] += colVal[pos][k] * x[pos]
		}
	}
	return v
}

func TestFactorFtranBtran(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 50; trial++ {
		m := 5 + rng.Intn(60)
		colRow, colVal := randomBasis(rng, m)
		f := factorize(m, colRow, colVal)
		if f == nil {
			continue // singular draw: skip
		}
		// FTRAN: B*x == v
		v := make([]float64, m)
		want := make([]float64, m)
		for i := range v {
			v[i] = rng.NormFloat64()
			want[i] = v[i]
		}
		f.ftran(v)
		back := multiply(colRow, colVal, m, v)
		for i := range back {
			if math.Abs(back[i]-want[i]) > 1e-7 {
				t.Fatalf("trial %d m=%d ftran: B*x[%d]=%g want %g (kernel %d fwd %d bwd %d)",
					trial, m, i, back[i], want[i], len(f.kRows), len(f.fwd), len(f.bwd))
			}
		}
		// BTRAN: B^T*y == w
		w := make([]float64, m)
		wantW := make([]float64, m)
		for i := range w {
			w[i] = rng.NormFloat64()
			wantW[i] = w[i]
		}
		f.btran(w)
		// (B^T y)[pos] = col_pos . y
		for pos := 0; pos < m; pos++ {
			var s float64
			for k, r := range colRow[pos] {
				s += colVal[pos][k] * w[r]
			}
			if math.Abs(s-wantW[pos]) > 1e-7 {
				t.Fatalf("trial %d m=%d btran: (B^T y)[%d]=%g want %g", trial, m, pos, s, wantW[pos])
			}
		}
	}
}

func TestFactorEtaUpdate(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for trial := 0; trial < 30; trial++ {
		m := 5 + rng.Intn(40)
		colRow, colVal := randomBasis(rng, m)
		f := factorize(m, colRow, colVal)
		if f == nil {
			continue
		}
		// replace basis position r with a random new column
		r := rng.Intn(m)
		r1 := rng.Intn(m)
		r2 := (r1 + 1 + rng.Intn(m-1)) % m
		newRows := []int32{int32(r1), int32(r2)}
		newVals := []float64{2 + rng.Float64(), rng.NormFloat64()}
		// alpha = Binv * newcol
		alpha := make([]float64, m)
		for k, rr := range newRows {
			alpha[rr] += newVals[k]
		}
		f.ftran(alpha)
		if math.Abs(alpha[r]) < 1e-6 {
			continue // would be a singular update
		}
		var idx []int32
		var val []float64
		for i, v := range alpha {
			if math.Abs(v) > etaDropTol {
				idx = append(idx, int32(i))
				val = append(val, v)
			}
		}
		etas := []*eta{{r: r, idx: idx, val: val, ar: alpha[r]}}

		// updated basis: column r swapped
		colRow2 := append([][]int32{}, colRow...)
		colVal2 := append([][]float64{}, colVal...)
		colRow2[r], colVal2[r] = newRows, newVals

		// FTRAN through factor+eta must satisfy B_new * x == v
		v := make([]float64, m)
		want := make([]float64, m)
		for i := range v {
			v[i] = rng.NormFloat64()
			want[i] = v[i]
		}
		f.ftran(v)
		applyEtas(etas, v)
		back := multiply(colRow2, colVal2, m, v)
		for i := range back {
			if math.Abs(back[i]-want[i]) > 1e-6 {
				t.Fatalf("trial %d m=%d eta-ftran: B_new*x[%d]=%g want %g", trial, m, i, back[i], want[i])
			}
		}
		// BTRAN: B_new^T * y == w
		w := make([]float64, m)
		wantW := make([]float64, m)
		for i := range w {
			w[i] = rng.NormFloat64()
			wantW[i] = w[i]
		}
		applyEtasT(etas, w)
		f.btran(w)
		for pos := 0; pos < m; pos++ {
			var s float64
			for k, rr := range colRow2[pos] {
				s += colVal2[pos][k] * w[rr]
			}
			if math.Abs(s-wantW[pos]) > 1e-6 {
				t.Fatalf("trial %d m=%d eta-btran: (B_new^T y)[%d]=%g want %g", trial, m, pos, s, wantW[pos])
			}
		}
	}
}
