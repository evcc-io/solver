package mps

import (
	"strings"
	"testing"

	"cbcgo/internal/mip"
	"cbcgo/internal/problem"
)

// lpText mirrors PuLP's writeLP shape for test_infeasible: an empty
// constraint gets a "_dummy: __dummy = 0" row plus itself rewritten.
const lpText = `\* infeas *\
Minimize
OBJ: x + 4 y + 9 z
Subject To
_dummy: __dummy = 0
c1: __dummy >= 5
c2: x + z >= 10
c3: - y + z = 7
c4: w >= 0
Bounds
__dummy = 0
x <= 4
-1 <= y <= 1
End
`

func TestReadLPInfeasible(t *testing.T) {
	p, err := ReadLP(strings.NewReader(lpText))
	if err != nil {
		t.Fatalf("ReadLP: %v", err)
	}
	if p.NumRows() != 5 {
		t.Fatalf("rows = %d, want 5", p.NumRows())
	}
	res := mip.New(p).Solve()
	if res.Status != mip.Infeasible {
		t.Fatalf("status = %v, want Infeasible", res.Status)
	}
}

const lpBasic = `\* t *\
Minimize
OBJ: x + 4 y + 9 z
Subject To
c2: x + z >= 10
c3: - y + z = 7
Bounds
x <= 4
-1 <= y <= 1
End
`

func TestReadLPBasic(t *testing.T) {
	p, err := ReadLP(strings.NewReader(lpBasic))
	if err != nil {
		t.Fatalf("ReadLP: %v", err)
	}
	xi, _ := p.ColIndex("x")
	yi, _ := p.ColIndex("y")
	zi, _ := p.ColIndex("z")
	if p.Cols[xi].UB != 4 || p.Cols[yi].LB != -1 || p.Cols[yi].UB != 1 {
		t.Fatalf("bounds not parsed: x=%+v y=%+v", p.Cols[xi], p.Cols[yi])
	}
	if p.Cols[zi].LB != 0 || p.Cols[zi].UB != problem.Inf {
		t.Fatalf("z default bounds wrong: %+v", p.Cols[zi])
	}
	res := mip.New(p).Solve()
	if res.Status != mip.Optimal {
		t.Fatalf("status = %v, want Optimal", res.Status)
	}
}
