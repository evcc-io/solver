package mip

import (
	"math"
	"testing"

	"cbcgo/internal/problem"
)

func almost(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

func TestKnapsack(t *testing.T) {
	// max 10x0+13x1+18x2+31x3+7x4+15x5 s.t. 11x0+15x1+20x2+35x3+10x4+33x5<=47, binary
	p := problem.New()
	p.ObjSense = -1
	profit := []float64{10, 13, 18, 31, 7, 15}
	weight := []float64{11, 15, 20, 35, 10, 33}
	cols := make([]int, len(profit))
	for i := range profit {
		cols[i] = p.AddCol("x", 0, 1, profit[i], true, nil, nil)
	}
	p.AddRow("cap", cols, weight, problem.LE, 47)

	res := New(p).Solve()
	if res.Status != Optimal {
		t.Fatalf("status = %v, want Optimal", res.Status)
	}
	if !almost(res.Obj, 41) {
		t.Fatalf("obj = %v, want 41", res.Obj)
	}
}

func TestRequiresBranching(t *testing.T) {
	// max x+y s.t. 2x+y<=4, x+2y<=4, x,y>=0 integer. LP relax = (4/3,4/3),
	// integer optimum = 2 (e.g. x=2,y=0 or x=0,y=2 or x=1,y=1).
	p := problem.New()
	p.ObjSense = -1
	x := p.AddCol("x", 0, problem.Inf, 1, true, nil, nil)
	y := p.AddCol("y", 0, problem.Inf, 1, true, nil, nil)
	p.AddRow("r0", []int{x, y}, []float64{2, 1}, problem.LE, 4)
	p.AddRow("r1", []int{x, y}, []float64{1, 2}, problem.LE, 4)

	res := New(p).Solve()
	if res.Status != Optimal {
		t.Fatalf("status = %v, want Optimal", res.Status)
	}
	if !almost(res.Obj, 2) {
		t.Fatalf("obj = %v, want 2", res.Obj)
	}
	for _, v := range res.X {
		if math.Abs(v-math.Round(v)) > 1e-6 {
			t.Fatalf("non-integer solution: %v", res.X)
		}
	}
}

func TestInfeasibleMIP(t *testing.T) {
	p := problem.New()
	x := p.AddCol("x", 0, 10, 1, true, nil, nil)
	p.AddRow("r0", []int{x}, []float64{1}, problem.GE, 5)
	p.AddRow("r1", []int{x}, []float64{1}, problem.LE, 2)

	res := New(p).Solve()
	if res.Status != Infeasible {
		t.Fatalf("status = %v, want Infeasible", res.Status)
	}
}

func TestSOS1(t *testing.T) {
	// max 5a+4b+3c s.t. a+b+c<=10, SOS1(a,b,c) -> pick the single best: a=10,obj=50
	p := problem.New()
	p.ObjSense = -1
	a := p.AddCol("a", 0, 10, 5, false, nil, nil)
	b := p.AddCol("b", 0, 10, 4, false, nil, nil)
	c := p.AddCol("c", 0, 10, 3, false, nil, nil)
	p.AddRow("cap", []int{a, b, c}, []float64{1, 1, 1}, problem.LE, 10)
	p.SOSs = append(p.SOSs, problem.SOS{Type: 1, Idx: []int{a, b, c}, Weight: []float64{1, 2, 3}})

	res := New(p).Solve()
	if res.Status != Optimal {
		t.Fatalf("status = %v, want Optimal", res.Status)
	}
	if !almost(res.Obj, 50) {
		t.Fatalf("obj = %v, want 50", res.Obj)
	}
	nonzero := 0
	for _, v := range res.X {
		if v > 1e-6 {
			nonzero++
		}
	}
	if nonzero > 1 {
		t.Fatalf("SOS1 violated: %v", res.X)
	}
}
