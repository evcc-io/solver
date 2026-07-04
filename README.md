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
  simplex for warm re-solves after bound changes (incremental reduced
  costs, long-step bound flips); wall-clock deadline checked inside the
  pivot loop.
- **Basis factorization**: singleton triangularization with a sparse-LU
  kernel and product-form (eta) updates — O(nnz) pivots, periodic
  refactorization. No dense inverse.
- **Presolve**: iterated activity-based bound tightening, big-M
  coefficient tightening for binaries, and time-boxed binary probing
  (CglProbing-style: fix each binary both ways, propagate, fix binaries
  with an infeasible side, merge implied bounds).
- **Cuts**: Gomory mixed-integer cuts at the root, with support/dynamism
  hygiene, bound-driven round control, and retraction of batches that
  degrade the LP numerically.
- **Branch and bound**: best-first with depth-first plunging, warm-started
  child bases, monotone bound propagation, bound-based optimality proof at
  exit, root reduced-cost fixing, SOS1/SOS2 via Beale–Tomlin splitting.
- **Branching**: strong branching at shallow depths seeding pseudocosts;
  pseudocost selection deeper (reliability-branching shape).
- **Heuristics**: caller-provided MIP start (`mip.Model.MIPStart`,
  completed before the cut loop so reduced-cost fixing bites), feasibility
  pump, two-sided rounding dive, RINS-lite improvement.
- **CLI**: honors the flags PuLP sends (`-max`, `-sec`, `-ratio`,
  `-allow`, `-maxNodes`, `-solve`, `-initialSolve`, `-solution`);
  unrecognized flags are consumed tolerantly.

## Missing vs. real CBC

- **Cut families beyond GMI**: no knapsack cover, probing, clique, MIR, or
  lift-and-project cuts; no cuts below the root.
- **`-mips <file>` warm start is parsed but not wired** to
  `Model.MIPStart`, so `warmStart=True` in PuLP buys nothing yet.
- **No multi-threaded search.** `-threads N` is accepted, ignored.
- **No anti-cycling rule** (Dantzig pricing with an iteration cap; no
  Bland fallback, no Harris ratio test, no perturbation).
- **Format gaps**: free-format MPS only; no `OBJSENSE` section (sense
  comes from `-max`, as PuLP conveys it); the negative-`UP`-bound MPS
  convention is not implemented. PuLP never exercises any of these.
- **Two failing PuLP tests**: `test_measuring_solving_time` (a
  ~16,500-variable bin-packing instance must find an incumbent within 10s)
  and `test_infeasible` (expects the specific variable values real CBC's
  pivoting lands on for a degenerate infeasible LP — reproducing them would
  mean cloning CBC's tie-breaking, not fixing a correctness bug).
