package simplex

import (
	"encoding/gob"
	"math/rand"
	"os"
	"testing"
)

// synthKernel builds a random diagonally-dominant sparse k*k kernel as sparse
// columns; representative of a mid-size refactorized basis block.
func synthKernel(k int, density float64) ([][]int32, [][]float64) {
	rng := rand.New(rand.NewSource(7))
	colIdx := make([][]int32, k)
	colVal := make([][]float64, k)
	for c := range k {
		for r := range k {
			if r == c {
				colIdx[c] = append(colIdx[c], int32(r))
				colVal[c] = append(colVal[c], float64(k))
			} else if rng.Float64() < density {
				colIdx[c] = append(colIdx[c], int32(r))
				colVal[c] = append(colVal[c], rng.NormFloat64())
			}
		}
	}
	return colIdx, colVal
}

// BenchmarkLUFactorizeSynth: self-contained factor + solve + solveT on a
// synthetic kernel; -benchmem tracks the per-factorize allocation count.
func BenchmarkLUFactorizeSynth(b *testing.B) {
	const k = 300
	colIdx, colVal := synthKernel(k, 0.03)
	w := &luWS{}
	v := make([]float64, k)
	for b.Loop() {
		lu := luFactorize(k, colIdx, colVal, w)
		if lu == nil {
			b.Fatal("singular")
		}
		for j := range v {
			v[j] = float64(j%17) - 8
		}
		lu.solve(v)
		lu.solveT(v)
	}
}

func loadKernel(b *testing.B) (int, [][]int32, [][]float64) {
	path := os.Getenv("SOLVER_KERNEL_GOB")
	if path == "" {
		b.Skip("SOLVER_KERNEL_GOB not set")
	}
	fh, err := os.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	defer fh.Close()
	dec := gob.NewDecoder(fh)
	var k int
	var colIdx [][]int32
	var colVal [][]float64
	if err := dec.Decode(&k); err != nil {
		b.Fatal(err)
	}
	if err := dec.Decode(&colIdx); err != nil {
		b.Fatal(err)
	}
	if err := dec.Decode(&colVal); err != nil {
		b.Fatal(err)
	}
	return k, colIdx, colVal
}

func BenchmarkLUFactorize(b *testing.B) {
	k, colIdx, colVal := loadKernel(b)
	b.ResetTimer()
	for range b.N {
		if lu := luFactorize(k, colIdx, colVal, nil); lu == nil {
			b.Fatal("singular")
		}
	}
}

func BenchmarkLUSolve(b *testing.B) {
	k, colIdx, colVal := loadKernel(b)
	lu := luFactorize(k, colIdx, colVal, nil)
	if lu == nil {
		b.Fatal("singular")
	}
	v := make([]float64, k)
	b.ResetTimer()
	for i := range b.N {
		for j := range v {
			v[j] = float64((i+j)%17) - 8
		}
		lu.solve(v)
	}
}

func BenchmarkLUSolveT(b *testing.B) {
	k, colIdx, colVal := loadKernel(b)
	lu := luFactorize(k, colIdx, colVal, nil)
	if lu == nil {
		b.Fatal("singular")
	}
	w := make([]float64, k)
	b.ResetTimer()
	for i := range b.N {
		for j := range w {
			w[j] = float64((i+j)%13) - 6
		}
		lu.solveT(w)
	}
}
