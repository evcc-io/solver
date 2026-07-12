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

## Implemented (CBC/Clp features)

- **Formats**: free-format MPS (`RANGES`/`SOS`/`MARKER`), PuLP's CPLEX `.lp`
  quirks, CBC's exact `.sol` status/data lines.
- **LP**: bounded-variable primal simplex (composite Phase 1) + dual simplex
  for warm re-solves (row-wise pivot row, incremental duals, long-step bound
  flips, just-fixed-row tabu, in-loop deadline). A fuller DSE dual
  (`simplex/dual2.go`: steepest edge, Harris ratio test, perturbation) drives
  deep node re-solves.
- **Scaling** (on by default, as Clp; `CBC_SCALE=0` disables): Clp geometric
  row/column scaling, source-matched to `ClpPackedMatrix::scale` / `ClpSimplex`
  — internal to the LP, with bounds/solution/duals/tableau unscaled at the
  boundary. Like Clp, a well-conditioned matrix is left unscaled (byte-identical),
  so scaling only bites on ill-conditioned models like the evcc big-M cases (1e6
  coefficient range): 018 ~29× faster, 021 5.8× and now matches CBC's optimum
  exactly, 020 near-proven (see Benchmarks).
- **Factorization**: singleton triangularization + sparse-LU kernel,
  product-form (eta) updates, periodic refactorize; no dense inverse;
  int32-compacted arenas, per-LP scratch. True Forrest-Tomlin (`simplex/ft.go`,
  `CBC_FT=1`): row-spike update, O(nnz) R-file, per-pivot parity with the eta
  path on the large cases; off by default (see below).
- **Presolve**: activity-based bound tightening to fixpoint, big-M binary
  coefficient tightening, CglProbing binary probing, singleton-column
  elimination (exact postsolve).
- **Cuts**: root Gomory (GMI) with hygiene + numeric retraction, CglProbing
  implication cuts, TwoMIR-lite and single-row MIR on large instances;
  pivot-budgeted rounds; slack cuts dropped before the tree.
- **Branch & bound**: best-first + depth-first plunging, warm child bases,
  node bound propagation, reduced-cost fixing per incumbent, SOS1/2, one
  restart inheriting cuts/fixings/probes.
- **Branching**: CBC reliability branching (pseudocosts, `numberBeforeTrust`=10,
  capped strong-branch probes, `maxStrong`=5, CBC score) + strong-branch fixing.
- **Heuristics**: MIP-start completion, 1-opt polish, face walk, RENS,
  objective feasibility pump, rounding dive, RINS-lite; time-boxed bursts.
- **Anti-degeneracy**: EXPAND ratio test, full Bland's after a degenerate
  streak, Clp bound perturbation in cold solves.
- **CLI**: PuLP's flags (`-max`/`-sec`/`-ratio`/`-allow`/`-maxNodes`/`-solve`/
  `-initialSolve`/`-solution`); unknown flags tolerated.

## Benchmarks (evcc golden cases, Apple M4)

Objective + wall-clock; CBC's optima shown for reference.

| case | main (pre-rewrite) | this branch (scaling default) | CBC |
|---|---|---|---|
| 018 | 4.9s, 18291.45 | **0.3s, 18291.46** | correct |
| 021 | 5.8s, **8.6901 (wrong)** | **2.6s, 8.70087** | 8.70087 |
| 020 | 60s, **−140 (garbage)** | 60s, **0.558**, gap 6e-4 | 0.5583 |

The 1e6-range big-M coefficients make these models ill-conditioned; scaling
(on by default, as Clp) is the lever — correct on all three and far faster
than main, which is both slower and unsound here. Proving 020 is the remaining
perf gap. Hot-path FTRAN/BTRAN is sparse-gather bound (compute, not GC/locality
— measured: `GOGC=off` moves wall <0.5%), so further speed is algorithmic
(scaling, Forrest-Tomlin), not micro-optimization.

## Missing vs. real CBC

- **Forrest-Tomlin gated off** (`CBC_FT=1`): the only gated feature not on by
  default. Its old 2–79× regression was a Bartels-Golub-style trailing-block
  update growing the R-file by Θ(m−p) ops per pivot; rewritten as true FT
  (single sparse row-spike elimination, O(nnz(row)) R-ops) it reaches per-pivot
  parity with the eta path (021: 50µs vs 52µs). Whole-solve can still lose on
  the golden suite because tiny solve-roundoff differences land B&B in a
  different (larger) node basin — measured chaos, not an FT defect: the eta
  path swings the same way across refactorize intervals (021: 3 nodes/2.7s at
  `CBC_MAXETAS=16` vs 131 nodes/31s at 24). Stays off until branching/
  degeneracy handling stops amplifying solve roundoff into node-count swings.
  Scaling, the extra cut families (`CBC_CGL`) and the DSE dual (`CBC_DUAL2`)
  are all on by default and pass the golden suite.
- **DSE dual is ~4× CBC wall** on the hardest case (pivot counts already 4–33×
  down toward CBC's).
- **Proving 020** is the open perf gap (incumbent ≈ CBC under scaling; proof
  needs CBC-quality node throughput / degeneracy handling).
- **No multi-threaded search** (`-threads` accepted, ignored); **`-mips` warm
  start** parsed but not wired; **format gaps** (free MPS only, no `OBJSENSE`,
  no negative-`UP` — PuLP never exercises these).
- **Two failing PuLP tests**: `test_measuring_solving_time` (16.5k-var
  bin-packing, incumbent within 10s) and `test_infeasible` (expects CBC's exact
  tie-breaking on a degenerate infeasible LP — not a correctness bug).
