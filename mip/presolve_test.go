package mip

import (
	"testing"

	"cbcgo/problem"
)

// x in [0,4], y in [-1,1], z >= 0 integer; c2+c3 imply z in [7,8],
// then x >= 2 and y in [-0.5, 0.5]. The mixed-integer set must survive.
func TestPresolveBoundTightening(t *testing.T) {
	p := problem.New()
	x := p.AddCol("x", 0, 4, 1, false, nil, nil)
	y := p.AddCol("y", -1, 1, 4, false, nil, nil)
	z := p.AddCol("z", 0, problem.Inf, 9, true, nil, nil)
	p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5)
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7.5)
	presolve(p)
	want := [][2]float64{{2, 4}, {-0.5, 0.5}, {7, 8}}
	for j, w := range want {
		if c := p.Cols[j]; c.LB != w[0] || c.UB != w[1] {
			t.Errorf("col %s: [%v,%v], want [%v,%v]", c.Name, c.LB, c.UB, w[0], w[1])
		}
	}
}

// big-M row e - M*y <= 0 with e <= 3 implied: M must shrink to 3.
func TestPresolveCoefficientTightening(t *testing.T) {
	p := problem.New()
	e := p.AddCol("e", 0, 3, 1, false, nil, nil)
	y := p.AddCol("y", 0, 1, 0, true, nil, nil)
	p.AddRow("bigm", []int{e, y}, []float64{1, -1000}, problem.LE, 0)
	presolve(p)
	if got := p.Rows[0].Coef[1]; got != -3 {
		t.Errorf("big-M coef = %v, want -3", got)
	}
	if got := p.Cols[e].Coef[0]; got != 1 {
		t.Errorf("column-side coef for e = %v, want 1", got)
	}
	if got := p.Cols[y].Coef[0]; got != -3 {
		t.Errorf("column-side coef for y = %v, want -3", got)
	}
}
