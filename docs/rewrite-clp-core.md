# Rewrite: Clp-faithful engine core

Goal: reach CBC/Clp wall-clock and pivot-count parity. The existing feature
ports (reliability branching, DSE `updateWeights`, warm cut re-solves, TwoMir)
are matched to source and active, but KPI/time parity is blocked: CBC wins
because its engine is **co-designed** on a reduced model where no vertex is a
razor edge. Retrofitting single features onto cbcgo's full-size model perturbs
the fragile 020 optimum (proven: DSE-everywhere, D1, Markowitz LU, warm cut
re-solves each regress 020 in isolation).

So the parity path is a coordinated core rewrite, not more feature flags.

References (local): `../Clp` (ClpSimplexDual, ClpFactorization,
ClpDualRowSteepest), `../CoinUtils` (CoinFactorization = Forrest-Tomlin LU),
`../cbc` (CbcNode, CglPreProcess wiring).

## Components, in dependency order

1. **Forrest-Tomlin factorization** (`CoinFactorization`) — replace cbcgo's
   product-form eta updates with FT (`replaceColumn` / `updateColumnFT`).
   Cheaper, more stable updates; enables sparse two-column FTRAN so the DSE
   `tau` solve is nearly free. This is the per-pivot cost lever.
2. **Full Markowitz + threshold pivoting** in the LU — sparser kernel, but only
   safe once (1) gives FT stability; on the current engine it perturbs 020.
3. **`CglPreProcess` model reduction** — coefficient strengthening + forcing/
   redundant-row removal to the ~1375-row model CBC solves. Requires (4) so the
   reduced model keeps its cut strength.
4. **Full `Cgl` suite** — Gomory, Probing, Knapsack, Clique, MixedIntegerRound2,
   FlowCover, TwoMir, ZeroHalf run every node (CBC frequency), so the reduced
   model's bound holds.
5. **`ClpSimplexDual` dual loop** — incremental infeasibility list + partial
   pricing (`numberWanted`), Harris ratio, proper degeneracy/perturbation.

Parity is expected only once 1+3+4 land together (the co-design). Each lands
behind a measurement gate and is benchmarked against `../optimizer` golden +
real CBC (`cbc_run.py`) before activation.

## Status

- [x] 1. Forrest-Tomlin factorization — sparse U-row LU + O(nnz) solves +
      fully-sparse alloc-free `replaceColumn` (trailing-block Bartels-Golub),
      property-tested. Wiring into the engine waits on (2): `newFTLU` is still
      dense-factorize (O(m^3)), too slow for the 2666-row basis.
- [~] 2. Sparse factorize — shortest-row + largest-entry pivoting, sparse
      L-columns + U-rows, no dense work matrix, property-tested. Not yet fast:
      needs a singleton pre-pass and bucketed pivot search (currently O(m^2)
      scans), and full Markowitz; not yet faster than the tuned dense path.
- [ ] 3. CglPreProcess (large subsystem — not started)
- [ ] 4. Cgl 9-generator suite (large subsystem — not started)
- [~] 5. Dual loop — `dual2.go` DSE dual is built and active via the mixed
      engine (shipped); Clp partial-pricing / infeasibility-list not ported.
