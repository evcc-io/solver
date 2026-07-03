// Package problem holds the mutable LP/MIP model: rows, columns, bounds,
// integrality and SOS constraints.
package problem

import "math"

const Inf = 1e30 // matches CBC's convention for "infinite" bound sentinel

type Sense byte

const (
	LE Sense = 'L'
	GE Sense = 'G'
	EQ Sense = 'E'
	NR Sense = 'N' // free row (only valid for the objective row)
)

type Row struct {
	Name     string
	Idx      []int // column indices, insertion order
	Coef     []float64
	Sense    Sense // original L/G/E/N sense; RANGES overlays via HasRange
	RHS      float64
	HasRange bool
	Range    float64
}

// Bounds returns the row's logical-variable [lb,ub], applying the MPS
// RANGES overlay (meaning depends on original sense) when present.
func (r *Row) Bounds() (lb, ub float64) {
	if r.HasRange {
		rng := math.Abs(r.Range)
		switch r.Sense {
		case LE:
			return r.RHS - rng, r.RHS
		case GE:
			return r.RHS, r.RHS + rng
		default: // EQ
			if r.Range >= 0 {
				return r.RHS, r.RHS + r.Range
			}
			return r.RHS + r.Range, r.RHS
		}
	}
	switch r.Sense {
	case LE:
		return -Inf, r.RHS
	case GE:
		return r.RHS, Inf
	case EQ:
		return r.RHS, r.RHS
	default: // NR
		return -Inf, Inf
	}
}

type Col struct {
	Name    string
	LB, UB  float64
	Obj     float64
	Integer bool
	Idx     []int // row indices this column appears in, insertion order
	Coef    []float64
}

type SOS struct {
	Type   int // 1 or 2
	Idx    []int
	Weight []float64
}

type Problem struct {
	Name     string
	Rows     []Row
	Cols     []Col
	ObjSense float64 // 1 = minimize, -1 = maximize
	SOSs     []SOS

	rowIdx map[string]int
	colIdx map[string]int
}

func New() *Problem {
	return &Problem{
		ObjSense: 1,
		rowIdx:   map[string]int{},
		colIdx:   map[string]int{},
	}
}

func (p *Problem) NumRows() int { return len(p.Rows) }
func (p *Problem) NumCols() int { return len(p.Cols) }

func (p *Problem) RowIndex(name string) (int, bool) { i, ok := p.rowIdx[name]; return i, ok }
func (p *Problem) ColIndex(name string) (int, bool) { i, ok := p.colIdx[name]; return i, ok }

// AddRow appends a new row. cols/coefs must reference existing column indices.
func (p *Problem) AddRow(name string, cols []int, coefs []float64, sense Sense, rhs float64) int {
	ri := len(p.Rows)
	p.Rows = append(p.Rows, Row{
		Name:  name,
		Idx:   append([]int(nil), cols...),
		Coef:  append([]float64(nil), coefs...),
		Sense: sense,
		RHS:   rhs,
	})
	if name != "" {
		p.rowIdx[name] = ri
	}
	for k, ci := range cols {
		c := &p.Cols[ci]
		c.Idx = append(c.Idx, ri)
		c.Coef = append(c.Coef, coefs[k])
	}
	return ri
}

// AddCol appends a new column. rows/coefs (may be nil) reference existing row indices.
func (p *Problem) AddCol(name string, lb, ub, obj float64, integer bool, rows []int, coefs []float64) int {
	ci := len(p.Cols)
	p.Cols = append(p.Cols, Col{
		Name:    name,
		LB:      lb,
		UB:      ub,
		Obj:     obj,
		Integer: integer,
		Idx:     append([]int(nil), rows...),
		Coef:    append([]float64(nil), coefs...),
	})
	if name != "" {
		p.colIdx[name] = ci
	}
	for k, ri := range rows {
		r := &p.Rows[ri]
		r.Idx = append(r.Idx, ci)
		r.Coef = append(r.Coef, coefs[k])
	}
	return ci
}

// DeleteRows removes the rows at the given indices (any order, duplicates ignored)
// and remaps all column row-references accordingly.
func (p *Problem) DeleteRows(idx []int) {
	drop := make(map[int]bool, len(idx))
	for _, i := range idx {
		drop[i] = true
	}
	remap := make([]int, len(p.Rows))
	newRows := p.Rows[:0]
	for i, r := range p.Rows {
		if drop[i] {
			remap[i] = -1
			continue
		}
		remap[i] = len(newRows)
		newRows = append(newRows, r)
	}
	p.Rows = newRows
	for ci := range p.Cols {
		c := &p.Cols[ci]
		newIdx := c.Idx[:0]
		newCoef := c.Coef[:0]
		for k, ri := range c.Idx {
			if remap[ri] < 0 {
				continue
			}
			newIdx = append(newIdx, remap[ri])
			newCoef = append(newCoef, c.Coef[k])
		}
		c.Idx, c.Coef = newIdx, newCoef
	}
	p.rowIdx = map[string]int{}
	for i, r := range p.Rows {
		if r.Name != "" {
			p.rowIdx[r.Name] = i
		}
	}
}

// DeleteCols removes the columns at the given indices and remaps all row column-references.
func (p *Problem) DeleteCols(idx []int) {
	drop := make(map[int]bool, len(idx))
	for _, i := range idx {
		drop[i] = true
	}
	remap := make([]int, len(p.Cols))
	newCols := p.Cols[:0]
	for i, c := range p.Cols {
		if drop[i] {
			remap[i] = -1
			continue
		}
		remap[i] = len(newCols)
		newCols = append(newCols, c)
	}
	p.Cols = newCols
	for ri := range p.Rows {
		r := &p.Rows[ri]
		newIdx := r.Idx[:0]
		newCoef := r.Coef[:0]
		for k, ci := range r.Idx {
			if remap[ci] < 0 {
				continue
			}
			newIdx = append(newIdx, remap[ci])
			newCoef = append(newCoef, r.Coef[k])
		}
		r.Idx, r.Coef = newIdx, newCoef
	}
	p.colIdx = map[string]int{}
	for i, c := range p.Cols {
		if c.Name != "" {
			p.colIdx[c.Name] = i
		}
	}
}

func (p *Problem) StoreNameIndexes() {
	p.rowIdx = make(map[string]int, len(p.Rows))
	for i, r := range p.Rows {
		if r.Name != "" {
			p.rowIdx[r.Name] = i
		}
	}
	p.colIdx = make(map[string]int, len(p.Cols))
	for i, c := range p.Cols {
		if c.Name != "" {
			p.colIdx[c.Name] = i
		}
	}
}
