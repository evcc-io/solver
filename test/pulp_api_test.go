// Go port of PuLP's COIN_CMDTest solve-behavior cases, built directly
// against problem/simplex/mip instead of LP/MPS files + bin/cbc.
package integration

import (
	"math"
	"testing"

	"cbcgo/mip"
	"cbcgo/problem"
	"cbcgo/simplex"
)

func almostEqual(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

func checkVal(t *testing.T, name string, got, want float64) {
	t.Helper()
	if !almostEqual(got, want) {
		t.Errorf("%s = %v, want %v", name, got, want)
	}
}

// newXYZW builds PuLP's recurring fixture: x in [0,4], y in [-1,1], z >= 0,
// w >= 0. Callers add whichever rows their test needs.
func newXYZW(objX, objY, objZ, objW float64, maximize bool) (p *problem.Problem, x, y, z, w int) {
	p = problem.New()
	if maximize {
		p.ObjSense = -1
	}
	x = p.AddCol("x", 0, 4, objX, false, nil, nil)
	y = p.AddCol("y", -1, 1, objY, false, nil, nil)
	z = p.AddCol("z", 0, problem.Inf, objZ, false, nil, nil)
	w = p.AddCol("w", 0, problem.Inf, objW, false, nil, nil)
	return
}

// test_infeasible
func TestInfeasible(t *testing.T) {
	p, x, y, z, w := newXYZW(1, 4, 9, 0, false)
	p.AddRow("c1", nil, nil, problem.GE, 5) // 0 >= 5: always false
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7)
	p.AddRow("c4", []int{w}, []float64{1}, problem.GE, 0)

	status, _, _ := simplex.Build(p).ColdSolve()
	if status != simplex.Infeasible {
		t.Fatalf("status = %v, want Infeasible", status)
	}
}

// test_continuous (also covers test_zero_constraint, test_divide,
// test_timeLimit — all the identical model with a no-op addition).
func TestContinuousMin(t *testing.T) {
	p, x, y, z, w := newXYZW(1, 4, 9, 0, false)
	p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5)
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7)

	lp := simplex.Build(p)
	status, st, _ := lp.ColdSolve()
	if status != simplex.Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
	sol, _, _, _ := lp.Solution(st)
	checkVal(t, "x", sol[x], 4)
	checkVal(t, "y", sol[y], -1)
	checkVal(t, "z", sol[z], 6)
	checkVal(t, "w", sol[w], 0)
}

// test_continuous_max
func TestContinuousMax(t *testing.T) {
	p, x, y, z, w := newXYZW(1, 4, 9, 0, true)
	p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5)
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7)

	lp := simplex.Build(p)
	status, st, _ := lp.ColdSolve()
	if status != simplex.Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
	sol, _, _, _ := lp.Solution(st)
	checkVal(t, "x", sol[x], 4)
	checkVal(t, "y", sol[y], 1)
	checkVal(t, "z", sol[z], 8)
	checkVal(t, "w", sol[w], 0)
}

// test_unbounded
func TestUnbounded(t *testing.T) {
	p, x, y, z, _ := newXYZW(1, 4, 9, 1, true)
	p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5)
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7)

	status, _, _ := simplex.Build(p).ColdSolve()
	if status != simplex.Unbounded {
		t.Fatalf("status = %v, want Unbounded", status)
	}
}

// test_no_objective
func TestNoObjective(t *testing.T) {
	p, x, y, z, _ := newXYZW(0, 0, 0, 0, false)
	p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5)
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7)

	status, _, _ := simplex.Build(p).ColdSolve()
	if status != simplex.Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
}

// test_variable_as_objective
func TestVariableAsObjective(t *testing.T) {
	p, x, y, z, _ := newXYZW(1, 0, 0, 0, false)
	p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5)
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7)

	status, _, _ := simplex.Build(p).ColdSolve()
	if status != simplex.Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
}

// newMIPXYZ builds the mip fixture: x in [0,4], y in [-1,1], z >= 0
// integer, common to test_mip and its variants.
func newMIPXYZ(objX, objY, objZ float64) (p *problem.Problem, x, y, z int) {
	p = problem.New()
	x = p.AddCol("x", 0, 4, objX, false, nil, nil)
	y = p.AddCol("y", -1, 1, objY, false, nil, nil)
	z = p.AddCol("z", 0, problem.Inf, objZ, true, nil, nil)
	p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5)
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7.5)
	return
}

// test_mip (also covers test_initial_value — same optimum, MIP-start
// hints don't change the result our API would return).
func TestMIP(t *testing.T) {
	p, x, y, z := newMIPXYZ(1, 4, 9)
	res := mip.New(p).Solve()
	if res.Status != mip.Optimal {
		t.Fatalf("status = %v, want Optimal", res.Status)
	}
	checkVal(t, "x", res.X[x], 3)
	checkVal(t, "y", res.X[y], -0.5)
	checkVal(t, "z", res.X[z], 7)
}

// test_mip_floats_objective
func TestMIPFloatsObjective(t *testing.T) {
	p, x, y, z := newMIPXYZ(1.1, 4.1, 9.1)
	res := mip.New(p).Solve()
	if res.Status != mip.Optimal {
		t.Fatalf("status = %v, want Optimal", res.Status)
	}
	checkVal(t, "x", res.X[x], 3)
	checkVal(t, "y", res.X[y], -0.5)
	checkVal(t, "z", res.X[z], 7)
	checkVal(t, "objective", res.Obj, 64.95)
}

// test_fixed_value
func TestFixedValue(t *testing.T) {
	p, x, y, z := newMIPXYZ(1, 4, 9)
	p.Cols[x].LB, p.Cols[x].UB = 4, 4
	p.Cols[y].LB, p.Cols[y].UB = -0.5, -0.5
	p.Cols[z].LB, p.Cols[z].UB = 7, 7

	res := mip.New(p).Solve()
	if res.Status != mip.Optimal {
		t.Fatalf("status = %v, want Optimal", res.Status)
	}
	checkVal(t, "x", res.X[x], 4)
	checkVal(t, "y", res.X[y], -0.5)
	checkVal(t, "z", res.X[z], 7)
}

// test_relaxed_mip
func TestRelaxedMIP(t *testing.T) {
	p, x, y, z := newMIPXYZ(1, 4, 9)
	res := mip.SolveRelaxation(p)
	if res.Status != mip.Optimal {
		t.Fatalf("status = %v, want Optimal", res.Status)
	}
	checkVal(t, "x", res.X[x], 3.5)
	checkVal(t, "y", res.X[y], -1)
	checkVal(t, "z", res.X[z], 6.5)
}

// test_feasibility_only
func TestFeasibilityOnly(t *testing.T) {
	p, _, _, _ := newMIPXYZ(0, 0, 0)
	res := mip.New(p).Solve()
	if res.Status != mip.Optimal {
		t.Fatalf("status = %v, want Optimal", res.Status)
	}
}

// test_infeasible_2
func TestInfeasible2(t *testing.T) {
	p := problem.New()
	x := p.AddCol("x", 0, 4, 0, false, nil, nil)
	y := p.AddCol("y", -1, 1, 0, false, nil, nil)
	z := p.AddCol("z", 0, 10, 0, false, nil, nil)
	p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5.2)
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10.3)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 17.5)

	status, _, _ := simplex.Build(p).ColdSolve()
	if status != simplex.Infeasible {
		t.Fatalf("status = %v, want Infeasible", status)
	}
}

// test_integer_infeasible
func TestIntegerInfeasible(t *testing.T) {
	p := problem.New()
	x := p.AddCol("x", 0, 4, 0, true, nil, nil)
	y := p.AddCol("y", -1, 1, 0, true, nil, nil)
	z := p.AddCol("z", 0, 10, 0, true, nil, nil)
	p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5.2)
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10.3)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7.4)

	res := mip.New(p).Solve()
	if res.Status != mip.Infeasible {
		t.Fatalf("status = %v, want Infeasible", res.Status)
	}
}

// test_integer_infeasible_2
func TestIntegerInfeasible2(t *testing.T) {
	p := problem.New()
	p.ObjSense = -1
	dummy := p.AddCol("dummy", -problem.Inf, problem.Inf, 1, false, nil, nil)
	c1 := p.AddCol("c1", 0, 1, 0, true, nil, nil)
	c2 := p.AddCol("c2", 0, 1, 0, true, nil, nil)
	_ = dummy
	p.AddRow("r1", []int{c1, c2}, []float64{1, 1}, problem.EQ, 2)
	p.AddRow("r2", []int{c1}, []float64{1}, problem.LE, 0)

	res := mip.New(p).Solve()
	if res.Status != mip.Infeasible {
		t.Fatalf("status = %v, want Infeasible", res.Status)
	}
}

// test_dual_variables_reduced_costs
func TestDualVariablesReducedCosts(t *testing.T) {
	p := problem.New()
	x := p.AddCol("x", 0, 5, 1, false, nil, nil)
	y := p.AddCol("y", -1, 1, 4, false, nil, nil)
	z := p.AddCol("z", 0, problem.Inf, 9, false, nil, nil)
	c1 := p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5)
	c2 := p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10)
	c3 := p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7)

	lp := simplex.Build(p)
	status, st, _ := lp.ColdSolve()
	if status != simplex.Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
	sol, rowAct, rc, price := lp.Solution(st)
	checkVal(t, "x", sol[x], 4)
	checkVal(t, "y", sol[y], -1)
	checkVal(t, "z", sol[z], 6)
	checkVal(t, "reducedCost x", rc[x], 0)
	checkVal(t, "reducedCost y", rc[y], 12)
	checkVal(t, "reducedCost z", rc[z], 0)
	checkVal(t, "dual c1", price[c1], 0)
	checkVal(t, "dual c2", price[c2], 1)
	checkVal(t, "dual c3", price[c3], 8)
	checkVal(t, "slack c1", 5-rowAct[c1], 2)
	checkVal(t, "slack c2", rowAct[c2]-10, 0)
	checkVal(t, "slack c3", rowAct[c3]-7, 0)
}

// test_sequential_solve: min x, then pin x at that optimum and maximize y,
// as two independent cold solves rather than PuLP's same-instance resolve.
func TestSequentialSolve(t *testing.T) {
	p1 := problem.New()
	x1 := p1.AddCol("x", 0, 1, 1, false, nil, nil)
	y1 := p1.AddCol("y", 0, 1, 0, false, nil, nil)
	z1 := p1.AddCol("z", 0, 1, 0, false, nil, nil)
	_ = y1
	_ = z1
	status, st, _ := simplex.Build(p1).ColdSolve()
	if status != simplex.Optimal {
		t.Fatalf("stage 1 status = %v, want Optimal", status)
	}
	sol1, _, _, _ := simplex.Build(p1).Solution(st)
	xStar := sol1[x1]

	p2 := problem.New()
	p2.ObjSense = -1
	x2 := p2.AddCol("x", 0, xStar, 0, false, nil, nil)
	y2 := p2.AddCol("y", 0, 1, 1, false, nil, nil)
	p2.AddCol("z", 0, 1, 0, false, nil, nil)

	status2, st2, _ := simplex.Build(p2).ColdSolve()
	if status2 != simplex.Optimal {
		t.Fatalf("stage 2 status = %v, want Optimal", status2)
	}
	sol2, _, _, _ := simplex.Build(p2).Solution(st2)
	checkVal(t, "x", sol2[x2], 0)
	checkVal(t, "y", sol2[y2], 1)
}

// test_fractional_constraints
func TestFractionalConstraints(t *testing.T) {
	p, x, y, z, w := newXYZW(1, 4, 9, 0, false)
	p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5)
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7)
	p.AddRow("c5", []int{x, z}, []float64{1, -0.5}, problem.EQ, 0)

	lp := simplex.Build(p)
	status, st, _ := lp.ColdSolve()
	if status != simplex.Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
	sol, _, _, _ := lp.Solution(st)
	checkVal(t, "x", sol[x], 10.0/3.0)
	checkVal(t, "y", sol[y], -1.0/3.0)
	checkVal(t, "z", sol[z], 20.0/3.0)
	checkVal(t, "w", sol[w], 0)
}

// newElasticBase builds the elastic-constraint fixture: w is free (no
// bounds) with objective coefficient 1, unlike the w>=0 used elsewhere.
func newElasticBase() (p *problem.Problem, x, y, z, w int) {
	p = problem.New()
	x = p.AddCol("x", 0, 4, 1, false, nil, nil)
	y = p.AddCol("y", -1, 1, 4, false, nil, nil)
	z = p.AddCol("z", 0, problem.Inf, 9, false, nil, nil)
	w = p.AddCol("w", -problem.Inf, problem.Inf, 1, false, nil, nil)
	p.AddRow("c1", []int{x, y}, []float64{1, 1}, problem.LE, 5)
	p.AddRow("c2", []int{x, z}, []float64{1, 1}, problem.GE, 10)
	p.AddRow("c3", []int{y, z}, []float64{-1, 1}, problem.EQ, 7)
	return
}

// test_elastic_constraints: no penalty/free-bound, so the elastic
// subproblem collapses to a plain hard constraint w >= -1.
func TestElasticConstraints(t *testing.T) {
	p, x, y, z, w := newElasticBase()
	p.AddRow("wbound", []int{w}, []float64{1}, problem.GE, -1)

	lp := simplex.Build(p)
	status, st, _ := lp.ColdSolve()
	if status != simplex.Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
	sol, _, _, _ := lp.Solution(st)
	checkVal(t, "x", sol[x], 4)
	checkVal(t, "y", sol[y], -1)
	checkVal(t, "z", sol[z], 6)
	checkVal(t, "w", sol[w], -1)
}

// test_elastic_constraints_2: proportionFreeBound=0.1 widens the
// unpenalized slack by 10% of |RHS|, i.e. w >= -1.1 at zero cost.
func TestElasticConstraints2(t *testing.T) {
	p, x, y, z, w := newElasticBase()
	p.AddRow("wbound", []int{w}, []float64{1}, problem.GE, -1.1)

	lp := simplex.Build(p)
	status, st, _ := lp.ColdSolve()
	if status != simplex.Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
	sol, _, _, _ := lp.Solution(st)
	checkVal(t, "x", sol[x], 4)
	checkVal(t, "y", sol[y], -1)
	checkVal(t, "z", sol[z], 6)
	checkVal(t, "w", sol[w], -1.1)
}

// test_elastic_constraints_penalty_unchanged: penalty=1.1 exceeds w's
// per-unit objective benefit of 1, so the penalty variable stays at 0.
func TestElasticConstraintsPenaltyUnchanged(t *testing.T) {
	p, x, y, z, w := newElasticBase()
	d := p.AddCol("d", 0, problem.Inf, 1.1, false, nil, nil)
	p.AddRow("wbound", []int{w, d}, []float64{1, 1}, problem.GE, -1)

	lp := simplex.Build(p)
	status, st, _ := lp.ColdSolve()
	if status != simplex.Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
	sol, _, _, _ := lp.Solution(st)
	checkVal(t, "x", sol[x], 4)
	checkVal(t, "y", sol[y], -1)
	checkVal(t, "z", sol[z], 6)
	checkVal(t, "w", sol[w], -1)
	checkVal(t, "d", sol[d], 0)
}

// test_elastic_constraints_penalty_unbounded: penalty=0.9 is cheaper than
// w's per-unit benefit of 1, so it pays to push the penalty var to
// infinity — the problem becomes unbounded.
func TestElasticConstraintsPenaltyUnbounded(t *testing.T) {
	p, _, _, _, w := newElasticBase()
	d := p.AddCol("d", 0, problem.Inf, 0.9, false, nil, nil)
	p.AddRow("wbound", []int{w, d}, []float64{1, 1}, problem.GE, -1)

	status, _, _ := simplex.Build(p).ColdSolve()
	if status != simplex.Unbounded {
		t.Fatalf("status = %v, want Unbounded", status)
	}
}

// test_unset_objective_value__is_valid
func TestUnsetObjectiveValueIsValid(t *testing.T) {
	p := problem.New()
	p.ObjSense = -1
	x := p.AddCol("x", -problem.Inf, problem.Inf, 0, false, nil, nil)
	p.AddRow("c1", []int{x}, []float64{1}, problem.GE, 1)

	status, _, _ := simplex.Build(p).ColdSolve()
	if status != simplex.Optimal {
		t.Fatalf("status = %v, want Optimal", status)
	}
}

// test_infeasible_problem__is_not_valid
func TestInfeasibleProblemIsNotValid(t *testing.T) {
	p := problem.New()
	p.ObjSense = -1
	x := p.AddCol("x", -problem.Inf, problem.Inf, 1, false, nil, nil)
	p.AddRow("c1", []int{x}, []float64{1}, problem.GE, 2)
	p.AddRow("c2", []int{x}, []float64{1}, problem.LE, 1)

	status, _, _ := simplex.Build(p).ColdSolve()
	if status != simplex.Infeasible {
		t.Fatalf("status = %v, want Infeasible", status)
	}
}
