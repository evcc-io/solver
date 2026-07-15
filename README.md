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
  activity), fixed-column (LB==UB) elimination, implied-free equality-chain
  substitution (CglPreProcess doubletons, `CBC_SUBST=0` disables) and
  singleton-column elimination — all with exact primal/dual postsolve.
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
- **Heuristics** (reduced-sub-problem first, as CBC's mini branch and bound):
  RENS — integral-fix + single-shot rounding + node-capped sub-MIP on the
  integral-fixed neighborhood — leads each burst, faceWalk is the fallback;
  RINS as a node-capped sub-MIP on the agree-fixed neighborhood
  (CbcHeuristicRINS), run per new incumbent; MIP-start completion,
  pivot-budgeted 1-opt polish and feasibility pump, rounding dive;
  time-boxed, pivot-capped bursts.
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
  full local lifecycle without LP row add/remove — with CBC-style
  effectiveness pruning (a batch that doesn't lift the node bound is
  retracted; kept batches drop non-binding cuts; two failures disable local
  cuts for the model). With pruning it proves 020 (57s, 1.1k nodes — was a
  timeout unpruned) but still loses to the 16s default, so it stays opt-in.
- **Per-solve pivot caps**: heuristic windows cap every LP at 2m+200 pivots
  (`SetIterCap`, honored by the primal loop and the DSE dual) — CBC's
  hot-start/heuristic iteration caps. This is what made fixed-column
  elimination shippable: single degenerate LP grinds (018: 192k pivots in
  one pump projection) now fail fast instead of eating the budget.
- **Strong-branch probe cost is at CBC's *share*, not CBC's *wall***: real
  CBC 2.10.3 on 020 runs 2032 probes / 55k hot-start iterations (27/probe,
  56% of all its simplex work); cbcgo runs 1022 probes / 82k pivots
  (80/probe, 50% of total) — the probe *economics* match, the absolute gap
  is per-pivot engine constants. Every CBC hot-start mechanism was built and
  measured on 020: DSE/perturbed probe duals (+0.7s), probe stall-exit
  (weak pseudocost seeds, tree 774→1567 nodes), higher probe caps (pure
  degenerate grind), pre-probe refactorize (neutral), and Clp's crunch —
  which ships opt-in (`CBC_CRUNCH=1`): probes solve a row-subset LP with
  provably-redundant rows dropped (~24% of 020's rows, bounds exactly equal;
  021 probe pivots 128→28), but the changed probe roundoff re-rolls 020's
  tree lottery (one refactorize interval loses its proof), so defaults stay
  byte-identical. Probes also skip solution extraction (312MB of the 020
  alloc profile was probe-side arrays nobody read).
- **Gap-scheduled diving** (CBC runs heuristics only while the gap is open):
  cbcgo used to burst diving heuristics at every 256th node unconditionally.
  On 020 the lone mid-tree burst fired with the incumbent already within 0.02
  of the optimum, found nothing, and cost ~26k pivots down the full
  rens→faceWalk→fpump→dive chain. Gating the burst on the open-gap condition
  skips it: 020 208k→183k pivots, 12.9→10.9s, **tree identical** (774 nodes),
  robustness 5/5.
- **Presolve reduction gap is CoinPresolve column elimination, not
  doubletons.** A structural census of the golden models finds **0 duplicate
  rows, 0 duplicate columns, 0 equality doubletons** (every equality row has
  3+ nonzeros; the abundant 2-term rows are all inequalities) — matching real
  CBC, which prints 0 substitutions. Yet CBC reduces 020 to 1375 rows / 1399
  cols where cbcgo reaches 2032 / 2421. The ~1000-column gap is 260 free
  columns + 721 continuous singletons (the `rowBlocks&&colBlocks` case).
  Free-column removal ships opt-in (`CBC_FREECOL=1`, CoinPresolve empty-column
  removal): optimum-preserving and ~2× on 018 (31746→13917 pivots), but the
  perturbed model re-rolls 020's proof-fragile tree (774→3400 nodes), so it
  stays off by default. Fixing (105 = CBC's 105) and big-M coefficient
  strengthening (301 ≈ CBC's 304, `Cgl0010I`-style) already match and ship on.
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
| 018 | 4.9s, 18291.45 | **0.34s, 7 nodes, 18291.4519** | 0.04s, 0 nodes, 18291.4519 |
| 021 | 5.8s, **8.6901 (wrong)** | **3.2s, 3 nodes, 8.70087** | 0.09s, 0 nodes, 8.70083 |
| 020 | 60s, **−140 (garbage)** | **10.9s, 774 nodes, 0.55835 proven** | 3.6s, 833 nodes, 0.55835 |

Tree robustness — nodes (and wall) as solve roundoff is perturbed via the
refactorize interval `CBC_MAXETAS` ∈ {24, 32, 48, 64, 100}:

| case | before strengthening | after | real CBC (any seed) |
|---|---|---|---|
| 018 | 27–336 nodes, 0.3–6.4s | **7–10 nodes, 0.20–0.65s** | 0 nodes |
| 021 | 3–291 nodes, 2.6–31s | **3 nodes every run, 2.7–5.7s** | 0 nodes |
| 020 | never proven | **proven 5/5, 10.3–24.6s** | 833 nodes, 3.6s |

Probing coefficient strengthening lifts 018's pre-cut root bound from 18092
to 18291.44 and stops node counts swinging with roundoff; fixed-column
elimination (020: −364 cols), equality-chain substitution and per-solve
pivot caps plus gap-scheduled diving take 020 from 59.8s to 10.9s with every refactorize interval
proving. 020 root parity: preprocessing fixes 105 = CBC's 105
(~350 rows strengthened vs CBC's 501), root cut bound within 1% of CBC's
closed distance (−0.664 vs −0.658 from −0.885).

**Why 020 is ~3× CBC's wall, decomposed.** The tree is already at parity —
774 nodes vs CBC's 833. The 10.9s-vs-3.6s gap factors cleanly into pivot
*volume* × per-pivot *cost*: cbcgo runs 182619 pivots at 59.7 µs each, CBC
99465 iterations at 36.2 µs — **1.84× volume × 1.65× per-pivot = 3.03×**, the
observed wall ratio. The per-pivot 1.65× is the Go sparse-triangular-solve
(memory-bound `s -= cv[k]·y[r]` gather; ~45% of the solve is the dual BTRAN)
vs Clp's C++ kernel — a codegen floor, not an algorithm gap. The volume 1.84×
is strong-branch probes (81748 pivots vs CBC's 55266 strong-branch iterations,
1.5×, on these 1e6-conditioned models) plus the root incumbent dive: faceWalk fires once at the root and
spends 37711 pivots to find the first feasible point, where CBC's
DiveCoefficient does it in 706 — a 53× gap. The model forces incremental
per-variable diving (batch rounding via RENS fails on 020), so faceWalk
re-solves once per fixed integer; CBC's cheaper (crunched / hot-started)
re-solves and coefficient selection are exactly the two things that re-roll
020's roundoff-fragile tree. Every volume-cutting lever was measured and
re-rolls that tree: faceWalk pivot budget → timeout (loses the root incumbent),
`CBC_CRUNCH`
row-subset probes → 846 nodes (slower), `CBC_FREECOL` → 3400 nodes, probe
stall-exit / higher caps / DSE probes → all worse. The tree-safe wins that
ship (gap-scheduled diving, concrete-sort ratio test, pooled propagate) took
020 from 12.6s to 10.9s without moving a single node. Reaching CBC's 3.6s
needs a C++-class kernel, not a correctness or search-quality fix.
