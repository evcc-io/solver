// Package mps reads free-format MPS files, the format PuLP writes by
// default before invoking the cbc executable.
package mps

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"

	"cbcgo/internal/problem"
)

var sectionNames = map[string]bool{
	"NAME": true, "ROWS": true, "COLUMNS": true, "RHS": true,
	"RANGES": true, "BOUNDS": true, "SOS": true, "ENDATA": true,
}

func Read(r io.Reader) (*problem.Problem, error) {
	p := problem.New()
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)

	var section string
	rowSense := map[string]problem.Sense{}
	rowOfName := map[string]int{}
	objRow := ""
	inInteger := false
	var curSOS *problem.SOS

	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "*") || strings.TrimSpace(line) == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		// Headers start in column 1; data lines are indented (a data line's
		// label may otherwise spell a section name, e.g. RHS vector "RHS").
		unindented := line[0] != ' ' && line[0] != '\t'
		if unindented && sectionNames[fields[0]] {
			section = fields[0]
			if section == "SOS" {
				curSOS = nil
			}
			continue
		}

		switch section {
		case "NAME":
			// nothing to do; problem name is optional metadata

		case "ROWS":
			if len(fields) < 2 {
				continue
			}
			sense := problem.Sense(strings.ToUpper(fields[0])[0])
			name := fields[1]
			if sense == problem.NR {
				if objRow == "" {
					objRow = name
				}
				continue // objective row (or an extra free row) isn't stored as a Row
			}
			rowSense[name] = sense
			ri := p.AddRow(name, nil, nil, sense, 0)
			rowOfName[name] = ri

		case "COLUMNS":
			if strings.Contains(line, "'MARKER'") {
				if strings.Contains(line, "'INTORG'") {
					inInteger = true
				} else if strings.Contains(line, "'INTEND'") {
					inInteger = false
				}
				continue
			}
			if len(fields) < 3 {
				continue
			}
			colName := fields[0]
			ci, ok := p.ColIndex(colName)
			if !ok {
				ci = p.AddCol(colName, 0, problem.Inf, 0, inInteger, nil, nil)
			}
			pairs := fields[1:]
			for i := 0; i+1 < len(pairs); i += 2 {
				rowName, valStr := pairs[i], pairs[i+1]
				val, err := strconv.ParseFloat(valStr, 64)
				if err != nil {
					return nil, fmt.Errorf("mps: bad COLUMNS value %q: %w", valStr, err)
				}
				if rowName == objRow {
					p.Cols[ci].Obj = val
					continue
				}
				ri, ok := rowOfName[rowName]
				if !ok {
					return nil, fmt.Errorf("mps: COLUMNS references unknown row %q", rowName)
				}
				appendCoef(p, ri, ci, val)
			}

		case "RHS":
			if len(fields) < 3 {
				continue
			}
			pairs := fields[1:]
			for i := 0; i+1 < len(pairs); i += 2 {
				rowName, valStr := pairs[i], pairs[i+1]
				val, err := strconv.ParseFloat(valStr, 64)
				if err != nil {
					return nil, fmt.Errorf("mps: bad RHS value %q: %w", valStr, err)
				}
				if ri, ok := rowOfName[rowName]; ok {
					p.Rows[ri].RHS = val
				}
			}

		case "RANGES":
			if len(fields) < 3 {
				continue
			}
			pairs := fields[1:]
			for i := 0; i+1 < len(pairs); i += 2 {
				rowName, valStr := pairs[i], pairs[i+1]
				val, err := strconv.ParseFloat(valStr, 64)
				if err != nil {
					return nil, fmt.Errorf("mps: bad RANGES value %q: %w", valStr, err)
				}
				if ri, ok := rowOfName[rowName]; ok {
					p.Rows[ri].HasRange = true
					p.Rows[ri].Range = val
				}
			}

		case "BOUNDS":
			if len(fields) < 2 {
				continue
			}
			if err := applyBound(p, fields); err != nil {
				return nil, err
			}

		case "SOS":
			if len(fields) >= 2 && (fields[0] == "S1" || fields[0] == "S2") {
				typ := 1
				if fields[0] == "S2" {
					typ = 2
				}
				p.SOSs = append(p.SOSs, problem.SOS{Type: typ})
				curSOS = &p.SOSs[len(p.SOSs)-1]
				continue
			}
			if curSOS != nil && len(fields) >= 2 {
				ci, ok := p.ColIndex(fields[0])
				if !ok {
					continue
				}
				w, err := strconv.ParseFloat(fields[1], 64)
				if err != nil {
					return nil, fmt.Errorf("mps: bad SOS weight %q: %w", fields[1], err)
				}
				curSOS.Idx = append(curSOS.Idx, ci)
				curSOS.Weight = append(curSOS.Weight, w)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return p, nil
}

// appendCoef adds a (row,col,val) nonzero, keeping Row/Col adjacency in
// sync when both already exist (AddRow/AddCol only do this at creation).
func appendCoef(p *problem.Problem, ri, ci int, val float64) {
	r := &p.Rows[ri]
	r.Idx = append(r.Idx, ci)
	r.Coef = append(r.Coef, val)
	c := &p.Cols[ci]
	c.Idx = append(c.Idx, ri)
	c.Coef = append(c.Coef, val)
}

// applyBound parses one BOUNDS line: [type, boundName, colName, value?].
func applyBound(p *problem.Problem, fields []string) error {
	typ := strings.ToUpper(fields[0])
	if len(fields) < 3 {
		return fmt.Errorf("mps: malformed BOUNDS line %v", fields)
	}
	col := fields[2]
	ci, ok := p.ColIndex(col)
	if !ok {
		return fmt.Errorf("mps: BOUNDS references unknown column %q", col)
	}
	c := &p.Cols[ci]
	var val float64
	if len(fields) >= 4 {
		v, err := strconv.ParseFloat(fields[3], 64)
		if err != nil {
			return fmt.Errorf("mps: bad BOUNDS value %q: %w", fields[3], err)
		}
		val = v
	}
	switch typ {
	case "UP":
		c.UB = val
	case "LO":
		c.LB = val
	case "FX":
		c.LB, c.UB = val, val
	case "FR":
		c.LB, c.UB = -problem.Inf, problem.Inf
	case "MI":
		c.LB = -problem.Inf
	case "PL":
		c.UB = problem.Inf
	case "BV":
		c.LB, c.UB = 0, 1
		c.Integer = true
	case "LI":
		c.LB = val
		c.Integer = true
	case "UI":
		c.UB = val
		c.Integer = true
	}
	return nil
}
