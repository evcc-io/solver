package simplex

// CrunchProbe builds a row-subset copy of lp for strong-branch probing —
// Clp's crunch/smallModel behind solveFromHotStart. It drops rows that are
// provably redundant (interval activity within the row bounds: they cannot
// bind for any point the probe children admit, Clp's crunch test), so probe
// objectives on the small LP EQUAL the full-LP probe objectives, while the
// dropped rows no longer host dual pivots or FTRAN/BTRAN work. Only
// slack-basic rows drop: removing a row with its basic slack column (-e_i)
// keeps the reduced basis square and nonsingular by cofactor expansion.
// ok=false when too few rows drop or the reduced basis fails to factorize.
func (lp *LP) CrunchProbe(st *State) (*LP, *State, bool) {
	n, m := lp.n, lp.m
	// interval activity of every row over the CURRENT column bounds: a row
	// whose range fits inside its own bounds can never bind for any point
	// the probe children admit, so dropping it changes no probe optimum
	// (Clp crunch redundancy test); ±inf column bounds poison the range.
	maxUp := make([]float64, m)
	maxDn := make([]float64, m)
	for j := range n {
		l, u := lp.lb[j], lp.ub[j]
		for k, r := range lp.colRow[j] {
			v := lp.colVal[j][k]
			hi, lo := v*u, v*l
			if v < 0 {
				hi, lo = lo, hi
			}
			maxUp[r] += hi
			maxDn[r] += lo
		}
	}
	newRow := make([]int, m) // orig row id -> crunched row id, -1 = dropped
	keep := make([]int, 0, m)
	for i := range m {
		// slack row bounds mirror the row bounds: activity within
		// [lb,ub] of the logical variable means the row cannot bind.
		// slack-basic keeps the reduced basis square (see below).
		redundant := st.status[n+i] == basic &&
			maxUp[i] <= lp.ub[n+i]+1e-9 && maxDn[i] >= lp.lb[n+i]-1e-9
		if !redundant {
			newRow[i] = len(keep)
			keep = append(keep, i)
		} else {
			newRow[i] = -1
		}
	}
	mm := len(keep)
	if m-mm < m/8 { // barely anything drops: not worth the copies
		return nil, nil, false
	}

	c := &LP{
		n: n, m: mm,
		colRow:  make([][]int, n+mm),
		colVal:  make([][]float64, n+mm),
		lb:      make([]float64, n+mm),
		ub:      make([]float64, n+mm),
		cost:    make([]float64, n+mm),
		rawObj:  lp.rawObj, // immutable, shared
		objSign: lp.objSign,
	}
	copy(c.lb[:n], lp.lb[:n])
	copy(c.ub[:n], lp.ub[:n])
	copy(c.cost[:n], lp.cost[:n])
	for j := range n {
		rows, vals := lp.colRow[j], lp.colVal[j]
		nr := make([]int, 0, len(rows))
		nv := make([]float64, 0, len(rows))
		for k, r := range rows {
			if rr := newRow[r]; rr >= 0 {
				nr = append(nr, rr)
				nv = append(nv, vals[k])
			}
		}
		c.colRow[j], c.colVal[j] = nr, nv
	}
	for k, i := range keep {
		c.lb[n+k], c.ub[n+k] = lp.lb[n+i], lp.ub[n+i]
	}
	if lp.colScale != nil {
		c.colScale = lp.colScale // shared: column space is unchanged
		c.rowScale = make([]float64, mm)
		for k, i := range keep {
			c.rowScale[k] = lp.rowScale[i]
		}
	}

	// int32 + row-wise mirrors, same layout Build produces
	c.colRow32 = make([][]int32, n+mm)
	c.colVal32 = make([][]float64, n+mm)
	for j := range n + mm {
		rows, vals := c.column(j)
		cr := make([]int32, len(rows))
		for k, r := range rows {
			cr[k] = int32(r)
		}
		c.colRow32[j], c.colVal32[j] = cr, vals
	}
	c.rowCol = make([][]int32, mm)
	c.rowVal = make([][]float64, mm)
	for j := range n {
		for k, r := range c.colRow[j] {
			c.rowCol[r] = append(c.rowCol[r], int32(j))
			c.rowVal[r] = append(c.rowVal[r], c.colVal[j][k])
		}
	}
	for r := range mm {
		c.rowCol[r] = append(c.rowCol[r], int32(n+r))
		c.rowVal[r] = append(c.rowVal[r], -1)
	}

	cst := &State{
		status:  make([]varStat, n+mm),
		basicOf: make([]int, 0, mm),
		value:   make([]float64, n+mm),
	}
	copy(cst.status[:n], st.status[:n])
	copy(cst.value[:n], st.value[:n])
	for k, i := range keep {
		cst.status[n+k] = st.status[n+i]
		cst.value[n+k] = st.value[n+i]
	}
	for i := range m {
		bv := st.basicOf[i]
		if bv >= n {
			if rr := newRow[bv-n]; rr >= 0 {
				cst.basicOf = append(cst.basicOf, n+rr)
			}
			continue // dropped slack leaves with its row
		}
		cst.basicOf = append(cst.basicOf, bv)
	}
	// a basic structural supported only by dropped rows leaves a zero
	// column: refactorize fails and the caller probes on the full LP
	if len(cst.basicOf) != mm || !c.refactorize(cst) {
		return nil, nil, false
	}
	c.recomputeBasics(cst)
	return c, cst, true
}
