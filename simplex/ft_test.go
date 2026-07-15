package simplex

import (
	"math"
	"math/rand"
	"testing"
)

// BenchmarkFTLU: factor + repeated ftran/btran at a representative kernel size,
// the per-solve cost FT would provide in the engine.
func BenchmarkFTLU(b *testing.B) {
	const m = 300
	rng := rand.New(rand.NewSource(7))
	a := make([]float64, m*m)
	for i := range a {
		if rng.Float64() < 0.03 {
			a[i] = rng.NormFloat64()
		}
	}
	for d := range m {
		a[d*m+d] += float64(m)
	}
	rhs := make([]float64, m)
	for i := range rhs {
		rhs[i] = rng.NormFloat64()
	}
	b.ResetTimer()
	for range b.N {
		f := newFTLU(m, a)
		v := append([]float64(nil), rhs...)
		f.ftran(v)
		f.btran(v)
	}
}

// BenchmarkFTReplace: one FT column update + solve (the per-pivot engine cost).
func BenchmarkFTReplace(b *testing.B) {
	const m = 300
	rng := rand.New(rand.NewSource(9))
	a := make([]float64, m*m)
	for i := range a {
		if rng.Float64() < 0.03 {
			a[i] = rng.NormFloat64()
		}
	}
	for d := range m {
		a[d*m+d] += float64(m)
	}
	nv := make([]float64, m)
	for i := range nv {
		nv[i] = rng.NormFloat64()
	}
	rhs := make([]float64, m)
	for i := range rhs {
		rhs[i] = rng.NormFloat64()
	}
	f := newFTLU(m, a)
	v := make([]float64, m)
	b.ResetTimer()
	for n := range b.N {
		col := n % m
		nv[col] += float64(m)
		f.replaceColumn(col, nv)
		nv[col] -= float64(m)
		copy(v, rhs)
		f.ftran(v)
	}
}

// matVec computes B*x for column-major B (a[col*m+row]).
func ftMatVec(m int, a, x []float64) []float64 {
	y := make([]float64, m)
	for c := range m {
		for r := range m {
			y[r] += a[c*m+r] * x[c]
		}
	}
	return y
}

func ftMatVecT(m int, a, x []float64) []float64 { // B^T * x
	y := make([]float64, m)
	for c := range m {
		s := 0.0
		for r := range m {
			s += a[c*m+r] * x[r]
		}
		y[c] = s
	}
	return y
}

// TestFTReplaceColumn applies random column swaps and checks ftran/btran
// against the rebuilt basis (Forrest-Tomlin update correctness).
func TestFTReplaceColumn(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := 0; trial < 150; trial++ {
		m := 2 + rng.Intn(10)
		a := make([]float64, m*m)
		for i := range a {
			a[i] = rng.NormFloat64()
		}
		for d := range m {
			a[d*m+d] += float64(2 * m)
		}
		f := newFTLU(m, a)
		if f == nil {
			continue
		}
		for upd := 0; upd < 5; upd++ {
			col := rng.Intn(m)
			nv := make([]float64, m)
			for i := range nv {
				nv[i] = rng.NormFloat64()
			}
			nv[col] += float64(2 * m) // keep well-conditioned
			if !f.replaceColumn(col, nv) {
				break
			}
			for r := range m { // apply the swap to the reference basis
				a[col*m+r] = nv[r]
			}
			x := make([]float64, m)
			for i := range x {
				x[i] = rng.NormFloat64()
			}
			b := ftMatVec(m, a, x)
			bb := append([]float64(nil), b...)
			f.ftran(bb)
			for i := range m {
				if math.Abs(bb[i]-x[i]) > 1e-6*(1+math.Abs(x[i])) {
					t.Fatalf("ftran trial %d upd %d m=%d col %d: got %g want %g", trial, upd, m, i, bb[i], x[i])
				}
			}
			y := make([]float64, m)
			for i := range y {
				y[i] = rng.NormFloat64()
			}
			c := ftMatVecT(m, a, y)
			cc := append([]float64(nil), c...)
			f.btran(cc)
			for i := range m {
				if math.Abs(cc[i]-y[i]) > 1e-6*(1+math.Abs(y[i])) {
					t.Fatalf("btran trial %d upd %d m=%d row %d: got %g want %g", trial, upd, m, i, cc[i], y[i])
				}
			}
		}
	}
}

// TestFTReplaceColumnLarge exercises the sparse Hessenberg update on blocks far
// past the old dense cap, checking ftran/btran against the rebuilt basis.
func TestFTReplaceColumnLarge(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, m := range []int{100, 300, 400} {
		a := make([]float64, m*m)
		for i := range a {
			if rng.Float64() < 0.04 {
				a[i] = rng.NormFloat64()
			}
		}
		for d := range m {
			a[d*m+d] += float64(2 * m)
		}
		f := newFTLU(m, a)
		if f == nil {
			t.Fatalf("m=%d: singular factor", m)
		}
		for upd := 0; upd < 40; upd++ {
			col := rng.Intn(m) // early cols force a large trailing block
			nv := make([]float64, m)
			for i := range nv {
				if rng.Float64() < 0.04 {
					nv[i] = rng.NormFloat64()
				}
			}
			nv[col] += float64(2 * m)
			ok := f.replaceColumn(col, nv)
			for r := range m {
				a[col*m+r] = nv[r]
			}
			if !ok { // refactorize path: rebuild from the updated reference
				if f = newFTLU(m, a); f == nil {
					t.Fatalf("m=%d upd=%d: refactorize singular", m, upd)
				}
				continue
			}
			x := make([]float64, m)
			for i := range x {
				x[i] = rng.NormFloat64()
			}
			b := ftMatVec(m, a, x)
			f.ftran(b)
			for i := range m {
				if math.Abs(b[i]-x[i]) > 1e-5*(1+math.Abs(x[i])) {
					t.Fatalf("ftran m=%d upd=%d i=%d: got %g want %g", m, upd, i, b[i], x[i])
				}
			}
			y := make([]float64, m)
			for i := range y {
				y[i] = rng.NormFloat64()
			}
			c := ftMatVecT(m, a, y)
			f.btran(c)
			for i := range m {
				if math.Abs(c[i]-y[i]) > 1e-5*(1+math.Abs(y[i])) {
					t.Fatalf("btran m=%d upd=%d i=%d: got %g want %g", m, upd, i, c[i], y[i])
				}
			}
		}
	}
}

// TestFTLUSolve checks ftran/btran against a known x on random nonsingular B.
func TestFTLUSolve(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 200; trial++ {
		m := 1 + rng.Intn(12)
		a := make([]float64, m*m)
		for i := range a {
			a[i] = rng.NormFloat64()
		}
		for d := range m { // diagonal boost keeps it well-conditioned
			a[d*m+d] += float64(m)
		}
		f := newFTLU(m, a)
		if f == nil {
			continue
		}
		// ftran: B x = b, pick x, form b, solve, compare
		x := make([]float64, m)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		b := ftMatVec(m, a, x)
		bb := append([]float64(nil), b...)
		f.ftran(bb)
		for i := range m {
			if math.Abs(bb[i]-x[i]) > 1e-7*(1+math.Abs(x[i])) {
				t.Fatalf("ftran trial %d m=%d col %d: got %g want %g", trial, m, i, bb[i], x[i])
			}
		}
		// btran: B^T y = c, pick y, form c, solve, compare
		y := make([]float64, m)
		for i := range y {
			y[i] = rng.NormFloat64()
		}
		c := ftMatVecT(m, a, y)
		cc := append([]float64(nil), c...)
		f.btran(cc)
		for i := range m {
			if math.Abs(cc[i]-y[i]) > 1e-7*(1+math.Abs(y[i])) {
				t.Fatalf("btran trial %d m=%d row %d: got %g want %g", trial, m, i, cc[i], y[i])
			}
		}
	}
}
