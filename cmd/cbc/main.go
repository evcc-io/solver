// Command cbc drop-in-replaces the compiled CBC binary PuLP invokes: reads
// an MPS file, solves it, writes a CBC-compatible .sol file.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"cbcgo/mip"
	"cbcgo/mps"
	"cbcgo/problem"
	"cbcgo/solfile"
)

// valueOnlyFlags are accepted (with their following value consumed) but
// have no effect on correctness — see the plan's "explicitly out of scope"
// section: cuts/presolve/threads only affect performance, not the answer.
var valueOnlyFlags = map[string]bool{
	"presolve": true, "gomory": true, "knapsack": true, "probing": true,
	"cuts": true, "threads": true, "strong": true, "timemode": true,
	"printingoptions": true,
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: cbc <file.mps> [options] -solve -solution <file>")
	}
	mpsFile := args[0]
	var solutionFile, mipsFile string
	maximize := false
	lpOnly := false
	limits := mip.Limits{GapRel: 1e-9, GapAbs: 1e-9}

	rest := args[1:]
	for i := 0; i < len(rest); i++ {
		key := strings.ToLower(strings.TrimPrefix(rest[i], "-"))
		next := func() string {
			if i+1 < len(rest) {
				i++
				return rest[i]
			}
			return ""
		}
		switch key {
		case "max":
			maximize = true
		case "solve":
			lpOnly = false
		case "initialsolve":
			lpOnly = true
		case "mips":
			mipsFile = next()
		case "sec":
			if v, err := strconv.ParseFloat(next(), 64); err == nil {
				limits.MaxSeconds = v
			}
		case "ratio":
			if v, err := strconv.ParseFloat(next(), 64); err == nil {
				limits.GapRel = v
			}
		case "allow":
			if v, err := strconv.ParseFloat(next(), 64); err == nil {
				limits.GapAbs = v
			}
		case "maxnodes":
			if v, err := strconv.Atoi(next()); err == nil {
				limits.MaxNodes = v
			}
		case "solution":
			solutionFile = next()
		default:
			if valueOnlyFlags[key] {
				next()
			} else if i+1 < len(rest) && looksLikeValue(rest[i+1]) {
				i++ // tolerate an unrecognized flag's value token
			}
		}
	}

	f, err := os.Open(mpsFile)
	if err != nil {
		return err
	}
	var p *problem.Problem
	if strings.EqualFold(filepath.Ext(mpsFile), ".lp") {
		p, err = mps.ReadLP(f)
	} else {
		p, err = mps.Read(f)
	}
	f.Close()
	if err != nil {
		return err
	}
	if maximize {
		p.ObjSense = -1
	}
	if mipsFile != "" {
		if mf, err := os.Open(mipsFile); err == nil {
			solfile.ReadMIPStart(mf) // parsed for compatibility; not yet used to seed B&B
			mf.Close()
		}
	}

	fmt.Printf("cbcgo: solving %s (%d cols, %d rows)\n", mpsFile, p.NumCols(), p.NumRows())

	var res mip.Result
	if lpOnly {
		res = mip.SolveRelaxation(p)
	} else {
		model := mip.New(p)
		model.Limits = limits
		res = model.Solve()
	}

	statusToken := statusToken(res)
	colNames := make([]string, len(p.Cols))
	for i, c := range p.Cols {
		colNames[i] = c.Name
	}
	rowNames := make([]string, len(p.Rows))
	for i, r := range p.Rows {
		rowNames[i] = r.Name
	}

	out := os.Stdout
	if solutionFile != "" {
		sf, err := os.Create(solutionFile)
		if err != nil {
			return err
		}
		defer sf.Close()
		return solfile.Write(sf, statusToken, res.Obj, colNames, res.X, res.ReducedCost,
			rowNames, res.RowActivity, res.RowPrice)
	}
	return solfile.Write(out, statusToken, res.Obj, colNames, res.X, res.ReducedCost,
		rowNames, res.RowActivity, res.RowPrice)
}

// statusToken mirrors real CBC's tokens, including the no-incumbent
// parenthetical that keeps PuLP's token-4 check from reclassifying status.
func statusToken(res mip.Result) string {
	switch res.Status {
	case mip.Optimal:
		return "Optimal"
	case mip.Infeasible:
		return "Infeasible"
	case mip.Unbounded:
		return "Unbounded"
	default:
		if res.HasIncumbent {
			return "Stopped on time"
		}
		return "Stopped on time (no integer solution - continuous used)"
	}
}

func looksLikeValue(tok string) bool {
	switch strings.ToLower(tok) {
	case "on", "off", "elapsed", "cpu":
		return true
	}
	_, err := strconv.ParseFloat(tok, 64)
	return err == nil
}
