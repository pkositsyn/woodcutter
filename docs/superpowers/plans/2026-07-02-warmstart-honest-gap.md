# Warm-start + honest-gap — Implementation Plan (continuation)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Deliver a cutting solver that proves the optimum where it fits the time budget and otherwise returns a near-optimal plan with an honest reported gap (the Python/CBC `timeLimit` contract), then add a bounded-variable dual-simplex warm-start engine — behind an early validation gate — to prove more inputs within budget.

**Architecture:** Builds on the merged Tasks 1–4 (`solveTableau`, `gomoryCuts`, best-bound `SolveMILP`, root cut loop + rounding heuristic). Task 1 here adds a dive heuristic + honest-gap plumbing on the existing engine (immediate, low-risk value). Tasks 2–3 add a bounded-variable dual-simplex engine and a DFS branch-and-cut driver that warm-starts each node from its parent basis, validated by differential testing and gated by a hard-input performance checkpoint.

**Tech Stack:** Go 1.25, stdlib only.

## Global Constraints

- Go 1.25; stdlib only, no external deps.
- User-facing strings in Russian (match `report.go`).
- All LP/solver work stays in `internal/lp`; `cutting` reads only `lp.Solution`.
- **Revised contract:** prove optimum when achievable within the deadline/node backstop → `Proven=true`, `LowerBound==Objective`. Otherwise return the best feasible incumbent with `Proven=false` and `LowerBound` = the global lower bound over open nodes; gap = `(Objective-LowerBound)/Objective`. NO requirement to prove the adversarial 8-coprime input within any fixed small time.
- Existing correctness tests must stay green: `lp_test.go`, `milp_test.go`, `cuts_test.go`, all 7 `TestGoldenCrossCheck` cases, `report_test.go`.
- The objective is non-negative integer (stock lengths) — the dual-simplex engine may rely on `c_j ≥ 0` for a dual-feasible start.
- Numerical tolerances consistent with the package: `eps = 1e-9`; integrality `1e-6`.
- Differential testing is the correctness oracle for the new engine: it must agree with the existing `Solve` (LP) and brute force / existing `SolveMILP` (MILP).

---

### Task 1: Dive heuristic + honest-gap plumbing (current engine)

Deliver immediate, low-risk value on the already-working best-bound engine: a rounding-dive heuristic that seeds a strong incumbent, plus `Plan.Proven`/`Plan.LowerBound` and a report warning with the gap. Revise the acceptance test to the honest-gap contract.

**Files:**
- Modify: `internal/lp/milp.go` (add `diveHeuristic`; call it once after the root cut loop)
- Modify: `internal/cutting/types.go` (`Plan.Proven`, `Plan.LowerBound`)
- Modify: `internal/cutting/solve.go` (propagate; `Proven:true` on trivial feasible returns)
- Modify: `internal/cutting/report.go` (gap warning when `!Proven`)
- Test: `internal/lp/milp_test.go`, `internal/cutting/report_test.go`, `internal/cutting/terminate_test.go`

**Interfaces:**
- Consumes: `Solve`, `mostFractional`, `unit`, `Solution.Proven/LowerBound` (Task 1 of prior plan)
- Produces: `func diveHeuristic(rp Problem, integer []bool) (Solution, bool)`; `Plan.Proven bool`, `Plan.LowerBound int`

- [ ] **Step 1: Failing test for `diveHeuristic`**

In `internal/lp/milp_test.go` add:

```go
// diveHeuristic returns a feasible integer solution by pinning the most-
// fractional variable up until integral. On a covering LP it must beat naive
// one-shot round-up.
func TestDiveHeuristic(t *testing.T) {
	p := Problem{
		Objective: []float64{1, 1},
		Constraints: []Constraint{
			{Coeffs: []float64{2, 3}, Type: GreaterEqual, RHS: 7},
			{Coeffs: []float64{3, 2}, Type: GreaterEqual, RHS: 7},
		},
	}
	sol, ok := diveHeuristic(p, []bool{true, true})
	if !ok {
		t.Fatalf("dive failed to produce a feasible incumbent")
	}
	// Must be integer-feasible for p.
	if !feasiblePoint(p.Constraints, sol.X) {
		t.Fatalf("dive result infeasible: %v", sol.X)
	}
	for j, v := range sol.X {
		if math.Abs(v-math.Round(v)) > 1e-6 {
			t.Fatalf("dive x[%d]=%v not integer", j, v)
		}
	}
	// brute-force optimum is 3 (e.g. (1,2) or (2,1)); dive should be feasible and finite.
	if sol.Objective < 3-1e-9 {
		t.Fatalf("dive obj %v below true optimum 3 (infeasible-optimal)", sol.Objective)
	}
}
```

- [ ] **Step 2: Run — fails (undefined `diveHeuristic`)**

Run: `go test ./internal/lp/ -run TestDiveHeuristic -v`
Expected: FAIL (compile: undefined `diveHeuristic`)

- [ ] **Step 3: Implement `diveHeuristic` and call it in `SolveMILP`**

Add to `internal/lp/milp.go`:

```go
// diveHeuristic performs a rounding dive: repeatedly solve the LP relaxation and
// pin the most-fractional integer variable up to its ceiling until the LP is
// integer-feasible or becomes infeasible. Returns a feasible incumbent (usually
// far stronger than a single round-up) or ok=false if the dive hits infeasibility.
func diveHeuristic(rp Problem, integer []bool) (Solution, bool) {
	var extra []Constraint
	for i := 0; i < 500; i++ {
		cons := make([]Constraint, 0, len(rp.Constraints)+len(extra))
		cons = append(cons, rp.Constraints...)
		cons = append(cons, extra...)
		sol := Solve(Problem{Objective: rp.Objective, Constraints: cons})
		if sol.Status != Optimal {
			return Solution{}, false
		}
		frac := mostFractional(sol.X, integer)
		if frac == -1 {
			sol.Status = Optimal
			return sol, true
		}
		extra = append(extra, Constraint{
			Coeffs: unit(frac, len(rp.Objective)),
			Type:   GreaterEqual,
			RHS:    math.Ceil(sol.X[frac]),
		})
	}
	return Solution{}, false
}
```

In `SolveMILP`, immediately after the root cut loop block (right before `nodes := 0`), seed the incumbent with a dive:

```go
	if allIntegerVars(integer, len(p.Objective)) {
		if d, ok := diveHeuristic(rp, integer); ok && d.Objective < incumbent.Objective {
			incumbent = d
		}
	}
```

- [ ] **Step 4: Run — passes**

Run: `go test ./internal/lp/ -run 'TestDiveHeuristic|TestMILP|TestRoundUp' -v`
Expected: PASS (dive test + all existing MILP tests)

- [ ] **Step 5: Add `Plan.Proven`/`Plan.LowerBound`**

In `internal/cutting/types.go`, extend `Plan`:

```go
	// Proven is true when the integer optimum was proven within the budget.
	// False means a backstop returned a best-effort incumbent.
	Proven bool
	// LowerBound is a proven lower bound on TotalMaterial. When Proven it equals
	// TotalMaterial; otherwise gap = (TotalMaterial - LowerBound) / TotalMaterial.
	LowerBound int
```

- [ ] **Step 6: Propagate in `solve.go`**

In `internal/cutting/solve.go`: set `Proven: true` on the two trivial feasible early returns (`Plan{Feasible: true, TotalMaterial: 0}` and the `...Dropped: dropped` one). Replace the final return with:

```go
	return Plan{
		Feasible:      true,
		TotalMaterial: total,
		Boards:        boards,
		Dropped:       dropped,
		Proven:        sol.Proven,
		LowerBound:    int(math.Round(sol.LowerBound)),
	}
```

(`math` is already imported.)

- [ ] **Step 7: Gap warning in `report.go`**

In `WriteReport`, after the `plan.Dropped` loop and before `Детальный план раскроя`:

```go
	if !plan.Proven {
		gap := 0.0
		if plan.TotalMaterial > 0 {
			gap = float64(plan.TotalMaterial-plan.LowerBound) / float64(plan.TotalMaterial) * 100
		}
		fmt.Fprintf(w, "⚠ Оптимальность не доказана (лимит времени/узлов); верхняя оценка отклонения от оптимума: %.2f%%\n", gap)
	}
```

- [ ] **Step 8: Report tests**

In `internal/cutting/report_test.go` add `TestWriteReportUnproven` and `TestWriteReportProvenNoWarning`:

```go
func TestWriteReportUnproven(t *testing.T) {
	plan := Plan{
		Feasible: true, TotalMaterial: 10000, LowerBound: 9800, Proven: false,
		Boards: []Board{{StockLength: 10000, Pieces: []Piece{{Length: 10000, Start: 0, End: 10000, Kind: PieceCut}}}},
	}
	var sb strings.Builder
	WriteReport(&sb, plan, []Pair{{10000, 1}}, Options{Padding: 5})
	if !strings.Contains(sb.String(), "не доказана") {
		t.Fatalf("expected unproven warning, got:\n%s", sb.String())
	}
}

func TestWriteReportProvenNoWarning(t *testing.T) {
	plan := Plan{
		Feasible: true, TotalMaterial: 10000, LowerBound: 10000, Proven: true,
		Boards: []Board{{StockLength: 10000, Pieces: []Piece{{Length: 10000, Start: 0, End: 10000, Kind: PieceCut}}}},
	}
	var sb strings.Builder
	WriteReport(&sb, plan, []Pair{{10000, 1}}, Options{Padding: 5})
	if strings.Contains(sb.String(), "не доказана") {
		t.Fatalf("proven plan must not print warning:\n%s", sb.String())
	}
}
```

- [ ] **Step 9: Revise `terminate_test.go` to the honest-gap contract**

Replace `TestSolveHardInputTerminates` body. The contract is now: terminate quickly under the backstop, return a feasible plan, and if not proven, report a bounded gap. Do NOT assert a specific optimum or proof.

```go
func TestSolveHardInputTerminates(t *testing.T) {
	stock := []Pair{{6100, 30}, {5900, 30}, {5300, 30}, {4700, 30}, {4100, 30}}
	reqs := []Pair{
		{2333, 13}, {1777, 17}, {1301, 19}, {997, 23},
		{733, 29}, {511, 31}, {389, 37}, {271, 41},
	}
	opts := Options{Padding: 4, MILPTimeout: 3 * time.Second}

	start := time.Now()
	plan := Solve(stock, reqs, opts)
	elapsed := time.Since(start)
	t.Logf("elapsed=%v total=%d proven=%v lb=%d", elapsed, plan.TotalMaterial, plan.Proven, plan.LowerBound)

	if elapsed >= 8*time.Second {
		t.Fatalf("solve took %v, want prompt termination under a 3s backstop", elapsed)
	}
	if !plan.Feasible {
		t.Fatalf("expected a feasible plan, got Feasible=false")
	}
	assertPlanValid(t, plan, stock, reqs, 4)
	// Honest-gap contract: if not proven, the incumbent must be within a sane
	// gap of the reported lower bound (dive heuristic guarantees a decent plan).
	if !plan.Proven {
		if plan.LowerBound <= 0 || plan.LowerBound > plan.TotalMaterial {
			t.Fatalf("invalid lower bound %d for total %d", plan.LowerBound, plan.TotalMaterial)
		}
		gap := float64(plan.TotalMaterial-plan.LowerBound) / float64(plan.TotalMaterial)
		if gap > 0.05 {
			t.Fatalf("gap %.3f exceeds 5%%; dive heuristic too weak", gap)
		}
	}
}
```

Update the doc comment above the test to describe the honest-gap contract (remove the old NOTE).

- [ ] **Step 10: Full suite green**

Run: `go build ./... && go test ./... -v`
Expected: PASS — all `internal/lp`, all golden cross-checks, report tests, and the revised `TestSolveHardInputTerminates` (feasible, prompt, gap ≤ 5%).

Note for the controller: capture the logged `elapsed/total/proven/lb` for the hard input from this run — it is the baseline the warm-start checkpoint (Task 3) is compared against.

- [ ] **Step 11: Commit**

```bash
git add internal/lp/milp.go internal/lp/milp_test.go internal/cutting/types.go \
        internal/cutting/solve.go internal/cutting/report.go \
        internal/cutting/report_test.go internal/cutting/terminate_test.go
git commit -m "$(cat <<'EOF'
feat: rounding-dive heuristic + honest-gap reporting

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Bounded-variable dual-simplex engine

Add a bounded-variable dual-simplex LP solver in a new file. It is the warm-start foundation. This task delivers ONLY the from-scratch solve (no warm-start yet); correctness is gated by differential testing against the existing `Solve`.

**Files:**
- Create: `internal/lp/dualsimplex.go`
- Test: `internal/lp/dualsimplex_test.go`

**Interfaces:**
- Produces: a `bvSolver` type and `func newBVSolver(p Problem) *bvSolver`; method `func (s *bvSolver) solve() Status` (runs dual simplex to optimality/infeasible/unbounded); `func (s *bvSolver) solution() Solution` (objective, X over structural vars). All unexported.

**Algorithm (implement exactly this model):**
- Structural variables `0..n-1` with bounds `[0, +Inf)`. One logical variable per constraint `i`, column `n+i`, with type-dependent bounds: `LessEqual` → `[0, +Inf)`; `GreaterEqual` → `(-Inf, 0]`; `Equal` → `[0, 0]`. Row `i`: `A_i·x + s_i = b_i`.
- Costs: structural `c_j`; logical `0`.
- Initial basis: the logical variable of each row (identity). Nonbasic = structural at lower bound `0`. This basis is dual-feasible because reduced costs equal `c_j ≥ 0` (required by Global Constraints) at lower bound; it is primal-infeasible where `b_i` violates the logical's bounds (`>=`, `=`).
- Maintain a dense tableau `T` (m rows × (total) cols) of `B^{-1}A`, the basic values `xB` (= `B^{-1}(b - N x_N)`), and the reduced-cost row `d` (= `c - c_B B^{-1}A`). Track `atUpper []bool` for nonbasic status and `lo,hi []float64` bounds.
- **Dual simplex loop:** pick leaving row `r` = a basic variable most-violating its bound (value `< lo-1e-9` or `> hi+1e-9`); direction = toward the violated bound. Dual ratio test over eligible nonbasic columns (eligible depends on sign of `T[r][j]` and each nonbasic's at-lower/at-upper status), choosing the entering column that keeps all reduced costs dual-feasible (minimum ratio `|d_j / T[r][j]|`); tie-break by smallest column index (Bland, anti-cycling). If no eligible column → **Infeasible**. Pivot; the leaving variable becomes nonbasic at the bound it hit. Repeat until no violation → **Optimal**. (Unbounded cannot occur for this problem class with `c ≥ 0` and finite optimum; if the ratio test degenerates to unbounded, return `Unbounded`.)
- `solution()`: `X[j]` for structural `j` = its value (basic value or its bound); objective = `c·X`.

- [ ] **Step 1: Differential + unit tests (failing)**

Create `internal/lp/dualsimplex_test.go`:

```go
package lp

import (
	"math"
	"testing"
)

// The bounded-variable dual simplex must agree with the two-phase Solve on the
// same LPs (objective and feasibility status).
func bvSolve(p Problem) Solution {
	s := newBVSolver(p)
	st := s.solve()
	if st != Optimal {
		return Solution{Status: st}
	}
	return s.solution()
}

func TestBVMatchesSolveKnown(t *testing.T) {
	cases := []Problem{
		{Objective: []float64{2, 3}, Constraints: []Constraint{
			{Coeffs: []float64{1, 1}, Type: GreaterEqual, RHS: 10},
			{Coeffs: []float64{1, 0}, Type: GreaterEqual, RHS: 3}}},
		{Objective: []float64{5, 4}, Constraints: []Constraint{
			{Coeffs: []float64{2, 1}, Type: GreaterEqual, RHS: 3},
			{Coeffs: []float64{1, 3}, Type: GreaterEqual, RHS: 4}}},
		{Objective: []float64{1, 1, 1}, Constraints: []Constraint{
			{Coeffs: []float64{3, 2, 1}, Type: GreaterEqual, RHS: 10},
			{Coeffs: []float64{1, 1, 1}, Type: LessEqual, RHS: 8}}},
	}
	for i, p := range cases {
		want := Solve(p)
		got := bvSolve(p)
		if got.Status != want.Status {
			t.Fatalf("case %d: status %v != %v", i, got.Status, want.Status)
		}
		if want.Status == Optimal && math.Abs(got.Objective-want.Objective) > 1e-6 {
			t.Fatalf("case %d: obj %v != %v", i, got.Objective, want.Objective)
		}
	}
}

// Randomized differential test: many small covering LPs, bvSolve vs Solve.
func TestBVMatchesSolveRandom(t *testing.T) {
	rng := newTestRand(1) // deterministic; see helper below
	for iter := 0; iter < 500; iter++ {
		n := 2 + rng.intn(3)
		m := 1 + rng.intn(3)
		obj := make([]float64, n)
		for j := range obj {
			obj[j] = float64(1 + rng.intn(9)) // non-negative costs (problem class)
		}
		cons := make([]Constraint, m)
		for i := range cons {
			row := make([]float64, n)
			for j := range row {
				row[j] = float64(rng.intn(5))
			}
			cons[i] = Constraint{Coeffs: row, Type: GreaterEqual, RHS: float64(1 + rng.intn(15))}
		}
		p := Problem{Objective: obj, Constraints: cons}
		want := Solve(p)
		got := bvSolve(p)
		if got.Status != want.Status {
			t.Fatalf("iter %d status %v != %v (p=%+v)", iter, got.Status, want.Status, p)
		}
		if want.Status == Optimal && math.Abs(got.Objective-want.Objective) > 1e-6 {
			t.Fatalf("iter %d obj %v != %v (p=%+v)", iter, got.Objective, want.Objective, p)
		}
	}
}
```

Add a tiny deterministic RNG helper in the same test file (avoids `math/rand` seeding nondeterminism and the disallowed `Math.random`-style calls in production):

```go
type testRand struct{ s uint64 }

func newTestRand(seed uint64) *testRand { return &testRand{s: seed*2862933555777941757 + 1} }
func (r *testRand) next() uint64 {
	r.s = r.s*6364136223846793005 + 1442695040888963407
	return r.s
}
func (r *testRand) intn(n int) int { return int(r.next() >> 33 % uint64(n)) }
```

- [ ] **Step 2: Run — fails (undefined `newBVSolver`)**

Run: `go test ./internal/lp/ -run TestBV -v`
Expected: FAIL (compile: undefined `newBVSolver`)

- [ ] **Step 3: Implement `internal/lp/dualsimplex.go`**

Implement `bvSolver`, `newBVSolver`, `solve`, `solution` following the Algorithm section above. Keep the tableau dense. Use `1e-9` bound-violation tolerance, Bland tie-breaking on the entering column. Guard divisions by `|T[r][j]| > 1e-9`.

(No verbatim code is prescribed here: the dual simplex is intricate and the differential tests in Step 1 are the correctness oracle. The implementer writes the engine to satisfy the Algorithm spec and pass the tests. Dispatch this task on the most capable model.)

- [ ] **Step 4: Run — differential tests pass**

Run: `go test ./internal/lp/ -run TestBV -v`
Expected: PASS (known cases + 500 random LPs agree with `Solve`)

- [ ] **Step 5: Full lp suite green**

Run: `go build ./... && go test ./internal/lp/ -race`
Expected: PASS (new engine does not perturb existing solver/tests)

- [ ] **Step 6: Commit**

```bash
git add internal/lp/dualsimplex.go internal/lp/dualsimplex_test.go
git commit -m "$(cat <<'EOF'
feat(lp): bounded-variable dual-simplex engine (differential-tested vs Solve)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Warm-start DFS branch-and-cut + performance checkpoint

Add a warm-start API to `bvSolver` (apply a variable-bound change or an added cut row, then re-optimize via dual simplex from the current basis) and a DFS branch-and-cut driver that uses it. Then run the performance checkpoint on the hard input. If the checkpoint shows insufficient payoff, STOP and report to the controller (fallback: Task 1's honest-gap engine already ships).

**Files:**
- Modify: `internal/lp/dualsimplex.go` (warm-start methods)
- Create: `internal/lp/branchcut.go` (DFS branch-and-cut driver)
- Test: `internal/lp/dualsimplex_test.go`, `internal/lp/branchcut_test.go`

**Interfaces:**
- Consumes: `bvSolver`, `gomoryCuts` (root cuts; optional in-tree later), `diveHeuristic`, `mostFractional`
- Produces:
  - warm-start methods: `func (s *bvSolver) setLower(j int, lo float64) Status`, `func (s *bvSolver) setUpper(j int, hi float64) Status`, `func (s *bvSolver) snapshot() bvState`, `func (s *bvSolver) restore(st bvState)` (each mutating method re-optimizes via dual simplex and returns status)
  - `func solveMILPWarm(p Problem, integer []bool, deadline time.Time) Solution` — DFS branch-and-cut, same result contract as `SolveMILP` (Proven/LowerBound/backstop)

- [ ] **Step 1: Warm-start correctness test (failing)**

In `internal/lp/dualsimplex_test.go` add: after solving, tightening a variable's bound and re-optimizing must match a from-scratch solve with that bound added as a constraint.

```go
func TestBVWarmStartMatchesFresh(t *testing.T) {
	p := Problem{Objective: []float64{5, 4}, Constraints: []Constraint{
		{Coeffs: []float64{2, 1}, Type: GreaterEqual, RHS: 3},
		{Coeffs: []float64{1, 3}, Type: GreaterEqual, RHS: 4}}}
	s := newBVSolver(p)
	if s.solve() != Optimal {
		t.Fatal("initial solve not optimal")
	}
	// Warm-start: force x0 >= 2.
	if st := s.setLower(0, 2); st != Optimal {
		t.Fatalf("warm re-solve status %v", st)
	}
	got := s.solution()
	// Fresh solve of the same restricted problem.
	restricted := p
	restricted.Constraints = append(append([]Constraint{}, p.Constraints...),
		Constraint{Coeffs: []float64{1, 0}, Type: GreaterEqual, RHS: 2})
	want := Solve(restricted)
	if math.Abs(got.Objective-want.Objective) > 1e-6 {
		t.Fatalf("warm obj %v != fresh %v", got.Objective, want.Objective)
	}
}
```

- [ ] **Step 2: Run — fails (undefined `setLower`)**

Run: `go test ./internal/lp/ -run TestBVWarmStart -v`
Expected: FAIL (compile)

- [ ] **Step 3: Implement warm-start methods**

Add `setLower`/`setUpper` (change a bound; if the variable is nonbasic reset its at-bound value, if basic the violation triggers dual simplex; then call the dual-simplex loop) and `snapshot`/`restore` (copy/assign the compact state needed to resume: `basis`, `atUpper`, `lo`, `hi`, and the dense tableau + `xB` + reduced-cost row). Re-optimize after each bound change via the existing dual-simplex loop.

- [ ] **Step 4: Run — warm-start test passes**

Run: `go test ./internal/lp/ -run TestBV -v`
Expected: PASS

- [ ] **Step 5: DFS branch-and-cut driver + brute-force tests (failing first)**

Create `internal/lp/branchcut_test.go` with brute-force agreement + proven-flag + backstop-gap tests mirroring `milp_test.go`'s style but calling `solveMILPWarm`. Include the larger covering ILP from `TestMILPBruteAgreeLarger`.

Then create `internal/lp/branchcut.go`: `solveMILPWarm` builds a `bvSolver` from the root problem (optionally with root Gomory cuts added as rows), seeds an incumbent via `diveHeuristic`, then DFS: at each node solve/warm-solve the LP; prune if `ceil(LP) ≥ incumbent`; if integer, update incumbent; else branch on `mostFractional` by tightening bounds (down: `setUpper(j, floor)`; up: `setLower(j, ceil)`), using `snapshot`/`restore` to backtrack. Maintain a global lower bound = min LP bound over open (un-pruned, un-expanded) sibling frontier for `LowerBound`/gap. Honor the deadline/`milpNodeLimit` backstop, setting `Proven=false` + `LowerBound` on early exit; `Proven=true` on full exhaustion.

- [ ] **Step 6: Run — driver tests pass**

Run: `go test ./internal/lp/ -run 'TestBranchCut|TestBV|TestMILP' -race -v`
Expected: PASS

- [ ] **Step 7: PERFORMANCE CHECKPOINT (report to controller; not a unit test)**

Add a temporary, `t.Skip`-guarded benchmark-style test (or run via a scratch harness) that calls `solveMILPWarm` on the hard input's generated pattern MILP through `cutting.Solve` wired to the new driver, at `MILPTimeout` of 5s and 30s. Report: elapsed, incumbent total, proven, lower bound, gap, and nodes/sec.

**Decision gate:** compare against Task 1's baseline (best-bound + dive). Proceed to wire `solveMILPWarm` in as the default (replace the `SolveMILP` call in `cutting/solve.go`) ONLY if it strictly improves the hard-input gap or proves it within ≤30s AND keeps every golden case proven quickly. Otherwise STOP: leave `SolveMILP` (Task 1 engine) as the default, keep `solveMILPWarm` behind an unused/experimental path, and report the checkpoint numbers to the controller for a go/no-go decision. Do not silently ship a slower engine.

- [ ] **Step 8: Wire in (only if the gate passes) + full suite**

If the gate passes: change `internal/cutting/solve.go` to call `solveMILPWarm` instead of `SolveMILP`. Run `go test ./... -race`. Expected: PASS; hard input improved.

If the gate fails: skip the wiring; record the numbers.

- [ ] **Step 9: Commit**

```bash
git add internal/lp/dualsimplex.go internal/lp/dualsimplex_test.go \
        internal/lp/branchcut.go internal/lp/branchcut_test.go internal/cutting/solve.go
git commit -m "$(cat <<'EOF'
feat(lp): warm-start DFS branch-and-cut over the dual-simplex engine

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage (revised spec addendum):**
- Revised contract (proven-where-feasible + honest gap) → Task 1 (Plan fields, report, revised acceptance test). ✔
- Dive heuristic (strong incumbent) → Task 1. ✔
- Bounded-variable dual-simplex engine (dual-feasible start, no phase 1) → Task 2, per the Algorithm section. ✔
- Warm-start (branching = bound change; re-optimize via dual simplex) → Task 3. ✔
- DFS branch-and-cut + global lower bound + backstop → Task 3. ✔
- Differential-testing correctness oracle → Task 2 (vs `Solve`), Task 3 (warm vs fresh; vs brute force). ✔
- Early validation gate (fail-fast) → Task 3 Step 7 checkpoint with explicit go/no-go. ✔
- Honest-gap fallback if warm-start underdelivers → Task 1 ships independently; Task 3 gate protects against shipping a slower engine. ✔

**Placeholder scan:** Task 2 Step 3 and Task 3 Steps 3/5 intentionally specify the algorithm + oracle tests rather than verbatim code — this is deliberate for the intricate dual simplex, with differential tests as the executable spec. Not a placeholder: the behavior, interfaces, and acceptance are fully pinned. All other steps carry concrete code/commands.

**Type consistency:**
- `diveHeuristic(rp Problem, integer []bool) (Solution, bool)` — defined and called in Task 1; reused in Task 3. ✔
- `Plan.Proven bool` / `Plan.LowerBound int` — Task 1 defines; report/solve use. ✔
- `bvSolver`, `newBVSolver`, `solve`, `solution` — Task 2 defines; Task 3 extends (`setLower/setUpper/snapshot/restore`) and consumes. ✔
- `solveMILPWarm(p Problem, integer []bool, deadline time.Time) Solution` — Task 3 defines; wired in `cutting/solve.go` only if the gate passes. ✔
- `feasiblePoint`, `mostFractional`, `unit`, `gomoryCuts` — from prior plan; reused. ✔

**Scope check:** Three tasks; Task 1 is independently shippable value; Tasks 2–3 add the warm-start engine behind a validation gate. Focused and ordered by risk.
