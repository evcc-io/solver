package problem

import "testing"

func TestAddColThenRow(t *testing.T) {
	p := New()
	c0 := p.AddCol("x0", 0, 1, 10, true, nil, nil)
	c1 := p.AddCol("x1", 0, 1, 13, true, nil, nil)
	p.AddRow("cap", []int{c0, c1}, []float64{11, 15}, LE, 20)

	if len(p.Rows) != 1 || len(p.Cols) != 2 {
		t.Fatalf("unexpected sizes: rows=%d cols=%d", len(p.Rows), len(p.Cols))
	}
	if len(p.Cols[c0].Idx) != 1 || p.Cols[c0].Idx[0] != 0 {
		t.Fatalf("col0 row refs not updated: %+v", p.Cols[c0])
	}
	if len(p.Cols[c1].Idx) != 1 {
		t.Fatalf("col1 row refs not updated: %+v", p.Cols[c1])
	}
}

func TestAddColWithExistingRowRefs(t *testing.T) {
	p := New()
	c0 := p.AddCol("x0", 0, Inf, 1, false, nil, nil)
	p.AddRow("r0", []int{c0}, []float64{1}, GE, 5)
	// add a new column that references the already-existing row
	c1 := p.AddCol("x1", 0, Inf, 1, false, []int{0}, []float64{2})
	_ = c1

	r := p.Rows[0]
	if len(r.Idx) != 2 || r.Idx[1] != c1 || r.Coef[1] != 2 {
		t.Fatalf("row not updated with new column ref: %+v", r)
	}
}

func TestDeleteRowsRemapsColumns(t *testing.T) {
	p := New()
	c0 := p.AddCol("x0", 0, Inf, 1, false, nil, nil)
	p.AddRow("r0", []int{c0}, []float64{1}, LE, 10)
	p.AddRow("r1", []int{c0}, []float64{2}, LE, 20)
	p.DeleteRows([]int{0})

	if len(p.Rows) != 1 || p.Rows[0].Name != "r1" {
		t.Fatalf("row not deleted correctly: %+v", p.Rows)
	}
	if len(p.Cols[c0].Idx) != 1 || p.Cols[c0].Idx[0] != 0 {
		t.Fatalf("column refs not remapped after row delete: %+v", p.Cols[c0])
	}
}

func TestRowBounds(t *testing.T) {
	cases := []struct {
		r      Row
		lb, ub float64
	}{
		{Row{Sense: LE, RHS: 5}, -Inf, 5},
		{Row{Sense: GE, RHS: 5}, 5, Inf},
		{Row{Sense: EQ, RHS: 5}, 5, 5},
		{Row{Sense: GE, RHS: 5, HasRange: true, Range: 3}, 5, 8},
		{Row{Sense: LE, RHS: 5, HasRange: true, Range: -3}, 2, 5},
		{Row{Sense: EQ, RHS: 5, HasRange: true, Range: 3}, 5, 8},
		{Row{Sense: EQ, RHS: 5, HasRange: true, Range: -3}, 2, 5},
	}
	for _, c := range cases {
		lb, ub := c.r.Bounds()
		if lb != c.lb || ub != c.ub {
			t.Errorf("Bounds(%+v) = (%v,%v), want (%v,%v)", c.r, lb, ub, c.lb, c.ub)
		}
	}
}
