package mps

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"

	"cbcgo/internal/problem"
)

// ReadLP reads PuLP's CPLEX-style .lp format (pulp/mps_lp.py's writeLP),
// used whenever PuLP is invoked with use_mps=False.
func ReadLP(r io.Reader) (*problem.Problem, error) {
	p := problem.New()

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	var lines []string
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(strings.TrimSpace(line), `\*`) {
			continue
		}
		if strings.TrimSpace(line) == "" {
			continue
		}
		lines = append(lines, line)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}

	col := func(name string) int {
		if ci, ok := p.ColIndex(name); ok {
			return ci
		}
		return p.AddCol(name, 0, problem.Inf, 0, false, nil, nil)
	}

	state := ""
	var buf []string // accumulated lines for the current objective/constraint

	flushConstraint := func() {
		if len(buf) == 0 {
			return
		}
		name, terms, sense, rhs, err := parseConstraint(strings.Join(buf, " "))
		buf = nil
		if err != nil {
			return
		}
		var idx []int
		var coef []float64
		for _, t := range terms {
			idx = append(idx, col(t.name))
			coef = append(coef, t.coef)
		}
		p.AddRow(name, idx, coef, sense, rhs)
	}
	flushObjective := func() {
		if len(buf) == 0 {
			return
		}
		_, terms := stripLabel(strings.Join(buf, " "))
		buf = nil
		for _, t := range parseTerms(strings.Fields(terms)) {
			p.Cols[col(t.name)].Obj = t.coef
		}
	}

	var curSOS *problem.SOS

	for _, raw := range lines {
		line := strings.TrimSpace(raw)
		lower := strings.ToLower(line)
		switch lower {
		case "minimize", "min":
			flushObjective()
			p.ObjSense = 1
			state = "obj"
			continue
		case "maximize", "max":
			flushObjective()
			p.ObjSense = -1
			state = "obj"
			continue
		case "subject to", "such that", "st":
			flushObjective()
			state = "rows"
			continue
		case "bounds":
			flushConstraint()
			state = "bounds"
			continue
		case "generals", "general", "integers", "integer":
			flushConstraint()
			state = "generals"
			continue
		case "binaries", "binary":
			flushConstraint()
			state = "binaries"
			continue
		case "sos":
			flushConstraint()
			state = "sos"
			continue
		case "end":
			flushConstraint()
			state = ""
			continue
		}

		switch state {
		case "obj":
			buf = append(buf, line)
		case "rows":
			if isNewLabeledStatement(line) {
				flushConstraint()
			}
			buf = append(buf, line)
		case "bounds":
			applyLPBound(p, col, line)
		case "generals":
			c := &p.Cols[col(line)]
			c.Integer = true
		case "binaries":
			c := &p.Cols[col(line)]
			c.Integer = true
			c.LB, c.UB = 0, 1
		case "sos":
			if strings.HasPrefix(line, "S1") {
				p.SOSs = append(p.SOSs, problem.SOS{Type: 1})
				curSOS = &p.SOSs[len(p.SOSs)-1]
				continue
			}
			if strings.HasPrefix(line, "S2") {
				p.SOSs = append(p.SOSs, problem.SOS{Type: 2})
				curSOS = &p.SOSs[len(p.SOSs)-1]
				continue
			}
			if curSOS != nil {
				name, wstr, ok := strings.Cut(line, ":")
				if !ok {
					continue
				}
				w, err := strconv.ParseFloat(strings.TrimSpace(wstr), 64)
				if err != nil {
					continue
				}
				curSOS.Idx = append(curSOS.Idx, col(strings.TrimSpace(name)))
				curSOS.Weight = append(curSOS.Weight, w)
			}
		}
	}
	flushConstraint()
	return p, nil
}

// isNewLabeledStatement reports whether line starts a new "name: ..."
// constraint, as opposed to a wrapped continuation of the previous one.
func isNewLabeledStatement(line string) bool {
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return false
	}
	return strings.HasSuffix(fields[0], ":") && len(fields[0]) > 1
}

func stripLabel(s string) (label, rest string) {
	fields := strings.Fields(s)
	if len(fields) > 0 && strings.HasSuffix(fields[0], ":") {
		return strings.TrimSuffix(fields[0], ":"), strings.Join(fields[1:], " ")
	}
	return "", s
}

type term struct {
	name string
	coef float64
}

// parseConstraint splits a joined "name: terms... <sense> <rhs>" statement.
func parseConstraint(s string) (name string, terms []term, sense problem.Sense, rhs float64, err error) {
	name, rest := stripLabel(s)
	fields := strings.Fields(rest)
	senseIdx := -1
	for i, f := range fields {
		if f == "<=" || f == ">=" || f == "=" {
			senseIdx = i
			break
		}
	}
	if senseIdx < 0 || senseIdx+1 >= len(fields) {
		return "", nil, 0, 0, fmt.Errorf("mps/lp: malformed constraint %q", s)
	}
	rhs, err = strconv.ParseFloat(fields[senseIdx+1], 64)
	if err != nil {
		return "", nil, 0, 0, err
	}
	switch fields[senseIdx] {
	case "<=":
		sense = problem.LE
	case ">=":
		sense = problem.GE
	default:
		sense = problem.EQ
	}
	terms = parseTerms(fields[:senseIdx])
	return name, terms, sense, rhs, nil
}

// parseTerms reads a signed sum-of-terms token stream, e.g.
// ["-", "y", "+", "2", "z"] or ["x0", "+", "13", "x1"].
func parseTerms(fields []string) []term {
	var terms []term
	sign := 1.0
	i := 0
	for i < len(fields) {
		switch fields[i] {
		case "+":
			sign = 1
			i++
			continue
		case "-":
			sign = -1
			i++
			continue
		}
		coef := 1.0
		if v, err := strconv.ParseFloat(fields[i], 64); err == nil {
			coef = v
			i++
			if i >= len(fields) {
				break
			}
		}
		terms = append(terms, term{name: fields[i], coef: sign * coef})
		i++
		sign = 1
	}
	return terms
}

// applyLPBound parses one Bounds-section line: "name free", "name = v",
// "-inf <= name [<= v]", "v <= name [<= v2]", or "name <= v".
func applyLPBound(p *problem.Problem, col func(string) int, line string) {
	fields := strings.Fields(line)
	num := func(s string) (float64, bool) {
		v, err := strconv.ParseFloat(s, 64)
		return v, err == nil
	}
	switch {
	case len(fields) == 2 && fields[1] == "free":
		c := &p.Cols[col(fields[0])]
		c.LB, c.UB = -problem.Inf, problem.Inf
	case len(fields) == 3 && fields[1] == "=":
		if v, ok := num(fields[2]); ok {
			c := &p.Cols[col(fields[0])]
			c.LB, c.UB = v, v
		}
	case len(fields) == 5 && fields[1] == "<=" && fields[3] == "<=":
		lb, ok1 := num(fields[0])
		ub, ok2 := num(fields[4])
		if ok1 && ok2 {
			c := &p.Cols[col(fields[2])]
			c.LB, c.UB = lb, ub
		}
	case len(fields) == 3 && fields[0] == "-inf" && fields[1] == "<=":
		c := &p.Cols[col(fields[2])]
		c.LB = -problem.Inf
	case len(fields) == 3 && fields[1] == "<=":
		if v, ok := num(fields[0]); ok { // "v <= name"
			c := &p.Cols[col(fields[2])]
			c.LB = v
		} else if v, ok := num(fields[2]); ok { // "name <= v"
			c := &p.Cols[col(fields[0])]
			c.UB = v
		}
	}
}
