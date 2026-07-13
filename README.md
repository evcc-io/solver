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
  boundary. Like Clp, a well-conditioned matrix is left unscaled
  (byte-identical); it bites on ill-conditioned models like the evcc big-M
  cases (see [Benchmarks](#benchmarks-evcc-golden-cases-apple-m4)).
- **Factorization**: singleton triangularization + sparse-LU kernel,
  product-form (eta) updates, periodic refactorize; no dense inverse;
  int32-compacted arenas, per-LP scratch. True Forrest-Tomlin (`simplex/ft.go`,
  `CBC_FT=1`): row-spike update, O(nnz) R-file, per-pivot parity with the eta
  path on the large cases; off by default (see below).
- **Presolve**: activity-based bound tightening to fixpoint, big-M binary
  coefficient tightening, multi-pass CglProbing binary probing with
  probing-propagated coefficient strengthening (CBC's Cgl0003I "strengthened
  rows": one-sided big-M rows tighten against each probe side's implied
  activity), singleton-column elimination (exact postsolve).
- **Cuts**: root Gomory (GMI) with hygiene + numeric retraction, CglProbing
  implication cuts, knapsack cover, clique, zero-half, flow cover, c-MIR
  (Marchand-Wolsey), TwoMIR-lite and single-row MIR on large instances;
  pivot-budgeted rounds; slack cuts dropped before the tree.
- **Branch & bound**: best-first + depth-first plunging, warm child bases,
  node bound propagation, reduced-cost fixing per incumbent, SOS1/2, one
  restart inheriting cuts/fixings/probes; periodic in-tree rounds of the
  globally-valid cut families with cold-restarted open nodes.
- **Branching**: CBC reliability branching (pseudocosts, `numberBeforeTrust`=10,
  capped strong-branch probes, `maxStrong`=5, CBC score) + strong-branch fixing.
- **Heuristics**: MIP-start completion, pivot-budgeted 1-opt polish, face
  walk, RENS with a node-capped sub-MIP fallback on the integral-fixed
  neighborhood (CBC mini branch and bound), objective feasibility pump
  (pivot-budgeted), rounding dive, RINS as a
  node-capped sub-MIP on the agree-fixed neighborhood (CbcHeuristicRINS),
  run per new incumbent; time-boxed bursts.
- **Anti-degeneracy**: EXPAND ratio test, full Bland's after a degenerate
  streak, Clp bound perturbation in cold solves.
- **CLI**: PuLP's flags (`-max`/`-sec`/`-ratio`/`-allow`/`-maxNodes`/`-solve`/
  `-initialSolve`/`-solution`); unknown flags tolerated.

## Missing vs. real CBC

- **Forrest-Tomlin gated off** (`CBC_FT=1`): the only gated feature not on by
  default. Its old 2–79× regression was a Bartels-Golub-style trailing-block
  update growing the R-file by Θ(m−p) ops per pivot; rewritten as true FT
  (single sparse row-spike elimination, O(nnz(row)) R-ops) it reaches per-pivot
  parity with the eta path (021: 50µs vs 52µs), but whole-solve still tends to
  land in larger node basins on the golden suite. Scaling, the extra cut
  families (`CBC_CGL`) and the DSE dual (`CBC_DUAL2`) are all on by default
  and pass the golden suite.
- **DSE dual is ~4× CBC wall** on the hardest case (pivot counts already 4–33×
  down toward CBC's).
- **Node-local in-tree cuts implemented but gated** (`CBC_LOCALCUTS=1`): GMI
  cuts from a node's basis are stored as free rows (vacuous globally) and
  activated per subtree via bound overrides on their logical variables — the
  full local lifecycle without LP row add/remove. Correct, but net-negative
  on 020 (degenerate node grind) until CBC-style cut-effectiveness pruning
  lands; the globally-valid in-tree round stays on.
- **Fixed-column elimination implemented but gated** (`CBC_ELIMFIX=1`):
  LB==UB columns fold into row bounds with exact primal/dual postsolve
  (020: −364 cols, 021: −643). Objectives verified exact; measured 021 2.9s
  → 1.7s but 018/020 regress via a feasibility-pump grind on the scaled
  reduced model — gated until single-solve preemption exists. Deeper
  CglPreProcess pieces (doubleton substitution, duplicates) remain missing.
- **020 heuristic pivot spend**: faceWalk + strong-branch probes are ~80% of
  its pivots and resist capping (five reorder/budget variants measured
  wall-negative — the walk's vertex path pins the good tree). Node re-solves
  themselves are already cheap: 12.6 pivots/node vs CBC's 53 iterations/node.
- **Gap semantics**: default absolute gap is 1e-5, mirroring CBC's default
  cutoff increment (`CbcCutoffIncrement`); `-allow`/`-ratio` override it.
- **No multi-threaded search** (`-threads` accepted, ignored); **`-mips` warm
  start** parsed but not wired; **format gaps** (free MPS only, no `OBJSENSE`,
  no negative-`UP` — PuLP never exercises these).
- **Two failing PuLP tests**: `test_measuring_solving_time` (16.5k-var
  bin-packing, incumbent within 10s) and `test_infeasible` (expects CBC's exact
  tie-breaking on a degenerate infeasible LP — not a correctness bug).

## Benchmarks (evcc golden cases, Apple M4)

The evcc battery-optimizer models: 1e6-range big-M coefficients make them
ill-conditioned; the two levers are Clp scaling and CglProbing coefficient
strengthening (both on by default, as CBC/Clp). Reference solver: real CBC
2.10.3 (PuLP's bundled binary), defaults on both sides.

Wall-clock, nodes, objective:

| case | main (pre-rewrite) | this branch | real CBC |
|---|---|---|---|
| 018 | 4.9s, 18291.45 | **0.14s, 7 nodes, 18291.4519** | 0.04s, 0 nodes, 18291.4519 |
| 021 | 5.8s, **8.6901 (wrong)** | **3.0s, 3 nodes, 8.70087** | 0.09s, 0 nodes, 8.70083 |
| 020 | 60s, **−140 (garbage)** | **59s, 1.9k nodes, 0.55835 proven** | 3.6s, 833 nodes, 0.55835 |

Tree robustness — nodes (and wall) as solve roundoff is perturbed via the
refactorize interval `CBC_MAXETAS` ∈ {24, 32, 48, 64, 100}:

| case | before strengthening | after | real CBC (any seed) |
|---|---|---|---|
| 018 | 27–336 nodes, 0.3–6.4s | **6–8 nodes, 0.14–0.21s** | 0 nodes |
| 021 | 3–291 nodes, 2.6–31s | **3 nodes every run, 2.9–5.5s** | 0 nodes |
| 020 | never proven | **proven 4/5 (45–94s); miss ends 1.5e-5 off** | 833 nodes, 3.6s |

Probing coefficient strengthening lifts 018's pre-cut root bound from 18092
to 18291.44 and stops node counts swinging with roundoff. 020 root parity:
preprocessing fixes 105 = CBC's 105 (~350 rows strengthened vs CBC's 501),
root cut bound within 1% of CBC's closed distance (−0.664 vs −0.658 from
−0.885). The remaining wall gap vs CBC is root-closing power (CBC ends
018/021 at 0 nodes via CglPreProcess + pump/mini-B&B) and 020 tree cost
(CBC's ~950 node-local probing cuts and ~4× cheaper node re-solves) — engine
constants and local-cut architecture, not correctness.
