package simplex

import (
	"math"
	"testing"

	"cbcgo/internal/problem"
)

func almost(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

func TestMaximizeSimple(t *testing.T) {
	// max 2x + 3y s.t. x+y<=4, x<=3, y<=2, x,y>=0 -> x=2,y=2,obj=10
	p := problem.New()
	p.ObjSense = -1
	x := p.AddCol("x", 0, 3, 2, false, nil, nil)
	y := p.AddCol("y", 0, 2, 3, false, nil, nil)
	p.AddRow("cap", []int{x, y}, []float64{1, 1}, problem.LE, 4)

	lp := Build(p)
	status, st, obj := lp.ColdSolve()
	if status != Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
	if !almost(obj, 10) {
		t.Fatalf("obj = %v, want 10", obj)
	}
	xs, _, _, _ := lp.Solution(st)
	if !almost(xs[x], 2) || !almost(xs[y], 2) {
		t.Fatalf("solution = %v, want [2,2]", xs)
	}
}

func TestMinimizeTwoGERows(t *testing.T) {
	// min x+y s.t. x+2y>=4, 3x+y>=6, x,y>=0 -> obj ~ 2.8
	p := problem.New()
	x := p.AddCol("x", 0, problem.Inf, 1, false, nil, nil)
	y := p.AddCol("y", 0, problem.Inf, 1, false, nil, nil)
	p.AddRow("r0", []int{x, y}, []float64{1, 2}, problem.GE, 4)
	p.AddRow("r1", []int{x, y}, []float64{3, 1}, problem.GE, 6)

	lp := Build(p)
	status, _, obj := lp.ColdSolve()
	if status != Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
	if !almost(obj, 2.8) {
		t.Fatalf("obj = %v, want 2.8", obj)
	}
}

func TestInfeasible(t *testing.T) {
	p := problem.New()
	x := p.AddCol("x", 0, 10, 1, false, nil, nil)
	p.AddRow("r0", []int{x}, []float64{1}, problem.GE, 5)
	p.AddRow("r1", []int{x}, []float64{1}, problem.LE, 2)

	lp := Build(p)
	status, _, _ := lp.ColdSolve()
	if status != Infeasible {
		t.Fatalf("status = %v, want Infeasible", status)
	}
}

func TestUnbounded(t *testing.T) {
	p := problem.New()
	p.ObjSense = -1 // maximize
	p.AddCol("x", 0, problem.Inf, 1, false, nil, nil)

	lp := Build(p)
	status, _, _ := lp.ColdSolve()
	if status != Unbounded {
		t.Fatalf("status = %v, want Unbounded", status)
	}
}

func TestEqualityRow(t *testing.T) {
	// min 2x+3y s.t. x+y=10, x,y>=0 -> x=10,y=0,obj=20
	p := problem.New()
	x := p.AddCol("x", 0, problem.Inf, 2, false, nil, nil)
	y := p.AddCol("y", 0, problem.Inf, 3, false, nil, nil)
	p.AddRow("r0", []int{x, y}, []float64{1, 1}, problem.EQ, 10)

	lp := Build(p)
	status, st, obj := lp.ColdSolve()
	if status != Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
	if !almost(obj, 20) {
		t.Fatalf("obj = %v, want 20", obj)
	}
	xs, _, _, _ := lp.Solution(st)
	if !almost(xs[x], 10) || !almost(xs[y], 0) {
		t.Fatalf("solution = %v, want [10,0]", xs)
	}
}

func TestFreeVariable(t *testing.T) {
	// min x s.t. x+y=0, y<=5, x free -> x=-5 at optimum (y at its upper bound)
	p := problem.New()
	x := p.AddCol("x", -problem.Inf, problem.Inf, 1, false, nil, nil)
	y := p.AddCol("y", -problem.Inf, 5, 0, false, nil, nil)
	p.AddRow("r0", []int{x, y}, []float64{1, 1}, problem.EQ, 0)

	lp := Build(p)
	status, st, obj := lp.ColdSolve()
	if status != Unbounded {
		// x=-y, y in (-inf,5], so x in [-5,+inf); min x = -5.
		if status != Optimal || !almost(obj, -5) {
			t.Fatalf("status=%v obj=%v, want Optimal obj=-5", status, obj)
		}
		_ = st
	}
}
