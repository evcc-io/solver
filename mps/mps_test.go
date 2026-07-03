package mps

import (
	"math"
	"strings"
	"testing"

	"cbcgo/mip"
)

const knapsackMPS = `NAME          TEST
ROWS
 N  COST
 L  CAP
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    x0        COST            10.0   CAP             11.0
    x1        COST            13.0   CAP             15.0
    x2        COST            18.0   CAP             20.0
    x3        COST            31.0   CAP             35.0
    x4        COST             7.0   CAP             10.0
    x5        COST            15.0   CAP             33.0
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS       CAP             47.0
BOUNDS
 BV BND       x0
 BV BND       x1
 BV BND       x2
 BV BND       x3
 BV BND       x4
 BV BND       x5
ENDATA
`

func TestReadKnapsack(t *testing.T) {
	p, err := Read(strings.NewReader(knapsackMPS))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if p.NumCols() != 6 || p.NumRows() != 1 {
		t.Fatalf("cols=%d rows=%d, want 6/1", p.NumCols(), p.NumRows())
	}
	for _, c := range p.Cols {
		if !c.Integer || c.LB != 0 || c.UB != 1 {
			t.Fatalf("col %+v not parsed as binary", c)
		}
	}
	p.ObjSense = -1 // as the CLI's -max flag would set
	res := mip.New(p).Solve()
	if res.Status != mip.Optimal {
		t.Fatalf("status = %v, want Optimal", res.Status)
	}
	if math.Abs(res.Obj-41) > 1e-6 {
		t.Fatalf("obj = %v, want 41", res.Obj)
	}
}

const rangedMPS = `NAME
ROWS
 N  OBJ
 L  R1
COLUMNS
    x1        OBJ              1.0   R1               1.0
RHS
    RHS       R1              10.0
RANGES
    RNG       R1               4.0
ENDATA
`

func TestRanges(t *testing.T) {
	p, err := Read(strings.NewReader(rangedMPS))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	lb, ub := p.Rows[0].Bounds()
	if lb != 6 || ub != 10 {
		t.Fatalf("ranged bounds = [%v,%v], want [6,10]", lb, ub)
	}
}
