# cbcgo

A from-scratch Go reimplementation of enough of [COIN-OR CBC](https://github.com/coin-or/Cbc)
to be a drop-in replacement for the `cbc` binary that
[PuLP](https://github.com/coin-or/pulp)'s `COIN_CMD`/`PULP_CBC_CMD` solver
classes shell out to. It reads an MPS or CPLEX-LP file, solves the LP/MIP,
and writes a CBC-compatible `.sol` file. The `mip`/`simplex` packages are
also usable directly as a Go library (see `mip.Model`).

The scope is bounded to what PuLP actually needs: PuLP talks to CBC purely
via subprocess — write an MPS file, run `cbc <file> [flags] -solve
-solution <out>`, parse the `.sol`. No C API, no callbacks, no cgo.
Validation is PuLP's own test suite (`COIN_CMDTest`, 80 tests) run against
this binary, not a self-authored proxy suite.

## Status

`go test ./...` passes across all packages. Against PuLP 3.3.2's
`COIN_CMDTest`: **78 pass, 2 fail** (see [Missing](#missing-vs-real-cbc)).

## Layout

```
cmd/cbc/     CLI entry point: flag parsing, orchestration
mps/         MPS reader (free format) and CPLEX-LP reader
problem/     mutable LP/MIP model: rows, cols, bounds, SOS
simplex/     bounded-variable primal + dual simplex, factorized basis
mip/         branch-and-bound, presolve, cuts, heuristics
solfile/     .sol writer, .mst warm-start reader
test/        opt-in wrapper for the PuLP compatibility suite
```

## Building and testing

```
go build ./...
go test ./...
go build -o bin/cbc ./cmd/cbc
```

The PuLP suite needs `python3` (and network on first run):
`RUN_PULP_TESTS=1 go test ./test/...` or `./scripts/run-pulp-tests.sh`.
It fails on any failure not listed in `testdata/pulp_known_failures.txt`.

## Implemented (CBC/Clp features and capabilities)

- **Formats**: free-format MPS (incl. `RANGES`, `SOS`, `MARKER` blocks),
  PuLP's CPLEX-style `.lp` quirks, CBC's exact `.sol` status/data lines
  (verified against PuLP's parser and CBC's `CbcSolver.cpp` output code).
- **LP**: bounded-variable primal simplex with composite Phase 1; dual
  simplex for warm re-solves after bound changes — sparse pivot row via
  a row-wise mirror of A, incrementally maintained duals with lazily
  materialized reduced costs, long-step bound flips, and a short tabu
  on just-fixed rows (on degenerate cut faces 95% of dual pivots were
  re-fixes of the last three rows; the tabu breaks the ping-pong);
  wall-clock deadline checked inside the pivot loop. Warm starts keep each touched
  nonbasic on its current bound side (Clp semantics) instead of
  re-snapping to the lower bound. A second, fuller dual engine
  (`simplex/dual2.go`: dual steepest edge, Harris two-pass ratio test,
  status-favoring cost perturbation, run to optimality) is
  property-tested but gated off behind `CBC_DUAL2=1`: measured on the
  target instances it trades the tuned root trajectory for no bound
  gain.
- **Basis factorization**: singleton triangularization with a sparse-LU
  kernel (in-place row elimination, counted arenas — allocation-free on
  the hot path) and product-form (eta) updates — O(nnz) pivots, periodic
  refactorization. No dense inverse. Per-factor data is int32-compacted
  and arena-consolidated, and all solve scratch is shared per LP, so a
  refactorization costs a handful of allocations instead of a dozen.
- **Presolve**: activity-based bound tightening run to fixpoint via a
  row worklist, big-M coefficient tightening for binaries,
  CglProbing-style binary probing (infeasibility fixing plus
  integer-only merged implied bounds), and singleton-column elimination
  (costed continuous singletons that pin their row or sit at a bound
  are substituted out and reconstructed exactly at postsolve).
- **Cuts**: Gomory mixed-integer cuts at the root, with support/dynamism
  hygiene, and retraction (with retries) of batches that degrade the LP
  numerically; rounds are budgeted in pivots (speed-invariant work
  units), so the cut set is a deterministic function of the problem and
  survives engine speed changes — wall clock only remains as a safety
  cap; probing implication cuts
  (CglProbing as a cut generator) on large instances, with slackened
  implied bounds so propagation drift can never cut off the optimum;
  TwoMIR-lite cuts (sparse pairwise tableau-row aggregations through the
  MIR derivation) on large instances; single-row MIR cuts
  (CglMixedIntegerRounding2-style, VUB/bound substitution with a divisor
  search) seed the first two rounds on large instances — later MIR
  rounds are measured-negative (row bloat, face drift); slack cuts are
  dropped before the
  tree so only root-active rows ride into node re-solves.
- **Branch and bound**: best-first with depth-first plunging, warm-started
  child bases, node-level bound propagation on branching, monotone bound
  propagation, bound-based optimality proof at exit, reduced-cost fixing
  re-run on every improving incumbent, SOS1/SOS2 via Beale–Tomlin
  splitting; a failed pass restarts once on the same model, inheriting
  cuts, fixings and probe facts.
- **Branching**: CBC reliability branching — a column whose pseudocost
  isn't yet trusted (fewer than `numberBeforeTrust`=10 observed gains
  either way) is strong-branched with cheap capped dual probes to seed
  it; trusted columns are scored straight from their pseudocost. The
  branch variable is the best CBC score (max-weighted blend before an
  incumbent, product rule after). Untrusted columns float to the top so
  the per-node strong-branch budget (`maxStrong`=5) is spent where the
  pseudocosts are least reliable — so it runs at every depth yet
  self-limits once columns are seeded. CBC-style strong-branch fixing: a
  probe side that is infeasible or cannot beat the incumbent fixes the
  variable at the node without spending a branch.
- **Heuristics**: caller-provided MIP start (`mip.Model.MIPStart`,
  completed via a warm child solve before the cut loop so reduced-cost
  fixing bites; the trivial start is deliberately not polished — measured
  as pure pivot burn), 1-opt incumbent polish on real tree incumbents
  (CbcHeuristicLocal-style binary flips via warm dual re-solves), face walk (least-degradation dive along the LP-optimal
  face — proves optimality outright on degenerate alternate-optima
  instances), RENS, feasibility pump, batch rounding dive, RINS-lite
  warm-started from the node's own basis, with exponential failure
  backoff; heuristic bursts are time-boxed
  (root MaxTime/3, deeper MaxTime/8) and skipped once the incumbent
  sits near the node bound, so the tree keeps its budget.
- **Anti-degeneracy**: EXPAND (Gill et al.) on the primal ratio test —
  expanding working bounds with a guaranteed minimum step, exact
  snap-back at refactorization and a cleanup re-price before concluding;
  full Bland's rule (entering and leaving) after a degenerate streak;
  Clp-style bound perturbation in cold solves with clean-bound restore.
- **CLI**: honors the flags PuLP sends (`-max`, `-sec`, `-ratio`,
  `-allow`, `-maxNodes`, `-solve`, `-initialSolve`, `-solution`);
  unrecognized flags are consumed tolerantly.

## Missing vs. real CBC

- **Cut families beyond GMI, probing, single-row MIR and pairwise
  TwoMIR**: no knapsack cover, clique, flow-cover, or lift-and-project
  cuts; no cuts below the root. A sound multi-row c-MIR generator (equality-chain
  aggregation with exact variable-bound substitution, property-tested)
  exists in `mip/cmir.go` but stays unwired: measured on the target
  instances it separates nothing that GMI + probing have not already
  found.
- **Simpler dual pricing than Clp**: the production dual picks the most
  violated row (plus the revisit tabu) instead of dual steepest edge,
  and runs unperturbed. The Clp-style engine (DSE, Harris ratio test,
  cost perturbation — `simplex/dual2.go`) is property-tested but gated
  off as measured-negative on the target instances. BTRAN is hypersparse
  (activation-graph guarded, bitwise-identical to the dense path) for
  mostly-zero right-hand sides and FTRAN skips zero pivots naturally,
  but there is no Clp-style hypersparse bookkeeping across the eta file
  (measured ~20% result density bounds the further payoff).
- **Partial CglPreProcess-style reductions**: singleton columns are
  eliminated, but rows never are, and the evcc instances' singletons are
  penalty slacks (cost fights the row — a `max(0, ·)` term no linear
  presolve can remove), so node LPs stay large on them (real CBC works
  on a ~4x smaller reduced model via row aggregation).
- **`-mips <file>` warm start is parsed but not wired** to
  `Model.MIPStart`, so `warmStart=True` in PuLP buys nothing yet.
- **No multi-threaded search.** `-threads N` is accepted, ignored.
- **Format gaps**: free-format MPS only; no `OBJSENSE` section (sense
  comes from `-max`, as PuLP conveys it); the negative-`UP`-bound MPS
  convention is not implemented. PuLP never exercises any of these.
- **Two failing PuLP tests**: `test_measuring_solving_time` (a
  ~16,500-variable bin-packing instance must find an incumbent within 10s)
  and `test_infeasible` (expects the specific variable values real CBC's
  pivoting lands on for a degenerate infeasible LP — reproducing them would
  mean cloning CBC's tie-breaking, not fixing a correctness bug).
