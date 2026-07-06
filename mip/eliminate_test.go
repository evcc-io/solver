package mip

import (
	"math"
	"testing"

	"cbcgo/problem"
)

// solveReducedAndExpand solves the reduced problem and maps the result back.
func solveReducedAndExpand(t *testing.T, q *problem.Problem, red *reduction) Result {
	t.Helper()
	res := New(q).Solve()
	if res.Status != Optimal {
		t.Fatalf("reduced solve status = %v", res.Status)
	}
	red.expand(&res)
	return res
}

// checkDualIdentity asserts rc_j == c_j - sum_i price_i * a_ij in the
// original space for every column.
func checkDualIdentity(t *testing.T, p *problem.Problem, res Result) {
	t.Helper()
	for j := range p.Cols {
		c := &p.Cols[j]
		want := c.Obj
		for k, ri := range c.Idx {
			want -= res.RowPrice[ri] * c.Coef[k]
		}
		if math.Abs(res.ReducedCost[j]-want) > 1e-7 {
			t.Errorf("col %s: rc = %g, want %g", c.Name, res.ReducedCost[j], want)
		}
	}
}

func TestEliminateTightSingleton(t *testing.T) {
	// min -2e + g: e in [0,inf) only in e+g<=10, so any optimum pins the row
	p := problem.New()
	e := p.AddCol("e", 0, problem.Inf, -2, false, nil, nil)
	g := p.AddCol("g", 0, 5, 1, true, nil, nil)
	r1 := p.AddRow("cap", []int{e, g}, []float64{1, 1}, problem.LE, 10)
	r2 := p.AddRow("dem", []int{g}, []float64{1}, problem.GE, 2)

	q, red := eliminateSingletons(p)
	if red == nil || len(q.Cols) != 1 || red.colMap[e] != -1 {
		t.Fatalf("expected e eliminated: %+v", red)
	}
	if got := q.Cols[red.colMap[g]].Obj; got != 3 {
		t.Fatalf("folded g cost = %g, want 3", got)
	}
	res := solveReducedAndExpand(t, q, red)
	if math.Abs(res.Obj-(-14)) > 1e-7 || math.Abs(res.X[e]-8) > 1e-7 || math.Abs(res.X[g]-2) > 1e-7 {
		t.Fatalf("obj=%g x=%v, want obj=-14 x=[8 2]", res.Obj, res.X)
	}
	if math.Abs(res.RowActivity[r1]-10) > 1e-7 || math.Abs(res.RowActivity[r2]-2) > 1e-7 {
		t.Fatalf("row activity = %v, want [10 2]", res.RowActivity)
	}
	checkDualIdentity(t, p, res)
}

func TestEliminateFixedSingleton(t *testing.T) {
	// min -x + s: s in [0,3] costs but its <= row never pushes it up: s=0
	p := problem.New()
	x := p.AddCol("x", 0, 7, -1, true, nil, nil)
	s := p.AddCol("s", 0, 3, 1, false, nil, nil)
	r1 := p.AddRow("cap", []int{x, s}, []float64{1, 1}, problem.LE, 10)

	q, red := eliminateSingletons(p)
	if red == nil || len(q.Cols) != 1 || red.colMap[s] != -1 {
		t.Fatalf("expected s eliminated: %+v", red)
	}
	if len(red.records) != 1 || red.records[0].kind != elimFixed || red.records[0].val != 0 {
		t.Fatalf("expected fixed-at-0 record: %+v", red.records)
	}
	res := solveReducedAndExpand(t, q, red)
	if math.Abs(res.Obj-(-7)) > 1e-7 || math.Abs(res.X[x]-7) > 1e-7 || res.X[s] != 0 {
		t.Fatalf("obj=%g x=%v, want obj=-7 x=[7 0]", res.Obj, res.X)
	}
	if math.Abs(res.RowActivity[r1]-7) > 1e-7 {
		t.Fatalf("row activity = %v, want [7]", res.RowActivity)
	}
	checkDualIdentity(t, p, res)
}

func TestEliminatePartialChain(t *testing.T) {
	// two exports share one row; folding e2's cost flips e1 into a bounded
	// penalty that must stay in the problem
	p := problem.New()
	e1 := p.AddCol("e1", 0, 4, -3, false, nil, nil)
	e2 := p.AddCol("e2", 0, problem.Inf, -2, false, nil, nil)
	g := p.AddCol("g", 0, 2, 5, true, nil, nil)
	r1 := p.AddRow("cap", []int{e1, e2, g}, []float64{1, 1, 1}, problem.LE, 10)

	q, red := eliminateSingletons(p)
	if red == nil || len(q.Cols) != 2 || red.colMap[e2] != -1 || red.colMap[e1] < 0 {
		t.Fatalf("expected only e2 eliminated: %+v", red)
	}
	if got := q.Cols[red.colMap[e1]].Obj; got != -1 {
		t.Fatalf("folded e1 cost = %g, want -1", got)
	}
	res := solveReducedAndExpand(t, q, red)
	if math.Abs(res.Obj-(-24)) > 1e-7 || math.Abs(res.X[e1]-4) > 1e-7 || math.Abs(res.X[e2]-6) > 1e-7 || res.X[g] != 0 {
		t.Fatalf("obj=%g x=%v, want obj=-24 x=[4 6 0]", res.Obj, res.X)
	}
	if math.Abs(res.RowActivity[r1]-10) > 1e-7 {
		t.Fatalf("row activity = %v, want [10]", res.RowActivity)
	}
	checkDualIdentity(t, p, res)
}
