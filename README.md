# cbcgo

A from-scratch Go reimplementation of just enough of [COIN-OR CBC](https://github.com/coin-or/Cbc)
to be a drop-in replacement for the `cbc` binary that
[PuLP](https://github.com/coin-or/pulp)'s `COIN_CMD`/`PULP_CBC_CMD` solver
classes shell out to. It reads an MPS or CPLEX-LP file, solves the LP/MIP,
and writes a CBC-compatible `.sol` file — nothing more.

## How this came to be

The original ask was much bigger: "reimplement the CBC solver as a Go
server, maintain the same API, pass all tests," pointed at an actual
checkout of upstream COIN-OR CBC (`~177K` lines of C++). That repo exposes
at least three different "APIs" — a ~200-function C API
(`Cbc_C_Interface.h`), the `cbc` CLI/interactive command language, and a C++
Osi interface — each with its own test suite, and the real solving engine
depends on three more COIN-OR libraries (Clp, Cgl, CoinUtils) that aren't
even vendored in that checkout. Reimplementing all of that from scratch was
not a realistic scope for a single session.

The scope was then bounded to "whatever PuLP actually needs," which turned
out to be dramatically smaller: PuLP's `COIN_CMD` talks to CBC purely via
**subprocess** — it writes an MPS (or, in a few cases, CPLEX-LP) file, runs
`cbc <file> [flags] -solve -solution <out>`, and parses the resulting
`.sol` file. No C API, no callbacks, no column generation, no cut-callback
mechanism. That meant no cgo, no C ABI matching, no callback trampolines —
just a plain Go CLI binary.

Two design passes (a C-API/cgo shim design and a from-scratch solver-engine
design) were done *before* that scope-narrowing research came back; the cgo
work was shelved once the PuLP research confirmed a pure-CLI target. The
engine design (simplex + branch-and-bound + SOS branching) carried over
almost unchanged, minus the C-API-only mechanisms (cut/incumbent callbacks,
live Osi/OsiCuts views, dynamic column-generation API) that PuLP never
exercises.

Everything here was written test-first against hand-verifiable LPs/MIPs,
then validated end-to-end by installing PuLP 3.3.2 in a virtualenv, pointing
its `COIN_CMD(path=...)` at this binary, and running PuLP's own `pytest`
suite (`pulp/tests/test_pulp.py::COIN_CMDTest`) — not a self-authored proxy
suite. That's what "passes tests" means for this project: PuLP's tests are
the acceptance bar, not ours.

## Current status

`go test ./...` passes across all packages. Against PuLP 3.3.2's real
`COIN_CMDTest` suite (80 tests): **78 pass, 2 fail** — see
[Known shortcomings](#known-shortcomings) below for what those two are and
why they haven't been fixed.

## Layout

```
cmd/cbc/            CLI entry point: flag parsing, orchestration
mps/                 MPS reader (free format) and CPLEX-LP reader
solfile/             .sol file writer, .mst warm-start file reader
problem/             mutable LP/MIP model: rows, cols, bounds, SOS
simplex/             bounded-variable primal simplex, dense Binv
mip/                 branch-and-bound: best-first search, SOS branching
test/                 Go wrapper for the PuLP compatibility suite (below)
scripts/               run-pulp-tests.sh: the PuLP suite runner itself
testdata/              pulp_known_failures.txt: the documented-failure allowlist
```

Package dependency order (no cycles): `problem` and nothing else at the
bottom; `simplex` depends on `problem`; `mip` depends on `problem` +
`simplex`; `mps`/`solfile` depend on `problem` only; `cmd/cbc` wires
everything together.

## Building and testing

```
go build ./...
go test ./...
go build -o bin/cbc ./cmd/cbc
```

### Verifying against real PuLP

`go test ./...` alone does **not** run the PuLP suite, since it needs
`python3` and (on first run) network access to `pip install`. Opt in with:

```
RUN_PULP_TESTS=1 go test ./test/...
```

or run the underlying script directly for more output:

```
./scripts/run-pulp-tests.sh
```

Either way this creates/reuses a venv at `.pulpenv/` (gitignored), installs
`pulp==3.3.2` and `pytest` if not already present, builds `bin/cbc`, symlinks
it onto `PATH` as `cbc` (`COIN_CMD`'s default path is the bare string `"cbc"`,
resolved via `shutil.which`), and runs PuLP's own
`pulp/tests/test_pulp.py::COIN_CMDTest`.

The test **passes** as long as no failure appears beyond the ones listed in
`testdata/pulp_known_failures.txt` (currently `test_infeasible` and
`test_measuring_solving_time` — see
[Known shortcomings](#known-shortcomings)). It fails loudly on any new,
undocumented failure — a real regression. If a listed failure starts
passing (e.g. after implementing a fix), the script reports it as
resolved and you should remove it from that file.

(`COIN_CMD`'s default path is the bare string `"cbc"`, resolved via
`shutil.which` — hence the `PATH` symlink trick rather than passing `path=`
directly, though that works too.)

## What it does

- Reads free-format MPS (`ROWS`/`COLUMNS`/`RHS`/`RANGES`/`BOUNDS`/`SOS`,
  `MARKER INTORG`/`INTEND` blocks) and PuLP's CPLEX-style `.lp` format
  (including its wrapped-line and empty-constraint-dummy-variable quirks).
- Solves LPs and MIPs: bounded-variable primal simplex with a
  composite-objective Phase 1, best-first branch-and-bound with
  most-fractional branching, SOS1/SOS2 via Beale–Tomlin splitting.
- Warm-starts each B&B child node from a clone of its parent's basis rather
  than re-deriving the basis from scratch at every node.
- Writes CBC's exact `.sol` status-line/data-line format, verified directly
  against PuLP's `readsol_MPS`/`get_status` parser source and real CBC's
  C++ output code (`src/CbcSolver.cpp`) — including the specific quirk where
  a "no incumbent found" stop must NOT look like `"... - objective value
  X"` at token position 4, or PuLP's parser silently reclassifies it as
  solved.
- Accepts (and where meaningful, honors) the CLI flags PuLP actually sends:
  `-max`, `-mips`, `-sec`, `-ratio`, `-allow`, `-maxNodes`, `-solve`,
  `-initialSolve` (LP-relaxation-only), `-solution`. Flags that only affect
  *performance* in real CBC (`-presolve`, `-gomory`/`-cuts`, `-threads`,
  `-strong`, `-timeMode`) are parsed and accepted but have no effect, since
  none of them change the correct answer. Unrecognized flags are tolerantly
  consumed rather than rejected.

## Known shortcomings

### Deliberately out of scope (correctness doesn't need them)

- **No presolve.** No problem-size reduction before solving.
- **No cutting planes** (Gomory, knapsack cover, probing, clique). Flags
  like `-gomory on` are accepted and ignored. These only tighten the LP
  relaxation for speed; branch-and-bound alone still reaches the correct
  answer, just by exploring more nodes.
- **No primal heuristics** (rounding, diving, feasibility pump). This is
  the one gap that has an actual test consequence: `test_measuring_solving_time`
  expects a feasible incumbent for a ~16,500-variable bin-packing instance
  within 10 seconds, which realistically needs a heuristic to manufacture a
  feasible point quickly rather than waiting for branch-and-bound to stumble
  onto one. This is the cause of one of the two failing PuLP tests.
- **No strong/pseudocost branching.** Always most-fractional. `-strong N`
  is accepted, ignored.
- **No multi-threaded search.** `-threads N` is accepted, ignored.

### Implemented but not fully wired up

- **MIP warm-start (`-mips <file>`) is parsed but not used.** The file is
  read (so it doesn't error out) but never seeds the initial incumbent.
  `warmStart=True` in PuLP currently buys nothing here.

### Numerical robustness

- **No Bland's-rule anti-cycling fallback.** Entering-variable selection is
  plain Dantzig's rule (most negative reduced cost) throughout; a
  sufficiently degenerate LP could in principle cycle, guarded only by a
  hard iteration cap (200,000), not a real anti-cycling rule.
- **No periodic basis refactorization.** `Binv` is updated incrementally
  via rank-1 product-form updates for the life of a solve and never
  recomputed from scratch mid-solve, so floating-point error could
  accumulate on a very long-running solve.
- **Dense `Binv`.** O(m²) memory and per-pivot work regardless of
  sparsity — fine at the scale typical PuLP-authored models hit, but part
  of why the 256-row/16,512-column bin-packing stress case is slow even
  with warm-starting.
- **Time limits are only checked between B&B nodes**, not inside a single
  simplex solve. A single very large/slow node's LP resolve could overrun
  `-sec` before the limit is ever checked.

### Format-level gaps (not needed for PuLP, but real gaps vs. general MPS/LP)

- **Free-format MPS only.** No fixed-column-position parsing. PuLP always
  writes free-format, so this has never mattered in practice.
- **No MPS `OBJSENSE` section.** Objective sense comes from the `-max` CLI
  flag (how PuLP conveys it), not from the file. A generic MPS file that
  encodes sense via `OBJSENSE` directly would be read as minimize
  regardless.
- **The "negative `UP` bound implies free lower bound" MPS convention is
  not implemented.** PuLP always writes an explicit `LO` bound to sidestep
  this exact ambiguity itself, so it's never been exercised.

### Dropped entirely when scope narrowed to PuLP

Before the PuLP-bounding decision, two design passes were done against the
*full* CBC C API — a cgo/C-ABI shim (using `cgo.Handle` for `Cbc_Model*`,
C trampolines for stored callback function pointers, call-scoped handles
for cut-callback live views) and a matching solver-engine design including
cut callbacks, incumbent callbacks, and a dynamic column-generation-capable
model. None of that is reachable from PuLP's CLI-only integration, so all of
the following were cut outright, not merely deprioritized:

- The C API / cgo shared-library surface itself.
- Cut callbacks and incumbent callbacks (`Cbc_addCutCallback`/`Cbc_addIncCallback`
  in the original C API).
- The live Osi/OsiCuts "view into the current relaxation" mechanism.
- A first-class dynamic column-generation API (PuLP's own column-generation
  patterns, if any, would just be repeated ordinary `-solve` invocations
  from the Python side — nothing library-side needs to know about it).
- A retained solution pool (`Cbc_numberSavedSolutions` equivalent).
- The `Cbc_addLazyConstraint` mechanism (lazy rows enforced only during
  B&B, not in every relaxation).

Reviving any of these would mean resurrecting the shelved cgo design rather
than extending what's here.

### The one failing test with no clear fix

`test_infeasible` expects specific variable values (`x=4, y=-1, z=6, w=0`)
on a deliberately infeasible LP (it includes an empty `0 >= 5` row). The
*status* (infeasible) is detected correctly; the specific variable values
are not, because they depend on which vertex of the (degenerate, partially
feasible) sub-problem the simplex implementation happens to land on when it
gives up — an implementation-detail of CBC's own internal pivoting sequence
that this project doesn't reproduce. Matching it exactly would mean
replicating CBC's specific tie-breaking rules on a historical regression
test, not fixing a correctness bug.
