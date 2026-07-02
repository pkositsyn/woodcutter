# woodcutter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Переписать `cutting.py` (1D cutting stock, генерация столбцов Гилмора-Гомори) на Go со стандартной раскладкой cmd/internal, без внешних LP-библиотек, с тестами, доказывающими эквивалентность Python-эталону.

**Architecture:** Собственный двухфазный tableau-симплекс (`internal/lp`) даёт LP-релаксацию и двойственные цены; branch & bound поверх него решает ILP. `internal/knapsack` — bounded knapsack DP (порт 1:1). `internal/cutting` оркестрирует генерацию столбцов, строит план. `cmd/woodcutter` читает stdin, печатает отчёт. Эквивалентность проверяется golden-файлами, сгенерированными вендорной копией Python через pulp.

**Tech Stack:** Go 1.25, стандартная библиотека только. Python 3.10 + pulp 3.0.2 — офлайн, для генерации golden.

## Global Constraints

- Module path: `github.com/pkositsyn/woodcutter`.
- Go stdlib only в продакшн-коде (никаких сторонних зависимостей в go.mod).
- Padding по умолчанию = 5 (совпадает с `min_material_cutting(..., padding=5)`).
- Формат stdin идентичен оригиналу: секция склада, строка `---`, секция требований; строки `длина[, количество]` (разделитель `,` или таб; пробелы в длине игнорируются; количество по умолчанию 1).
- Критерий эквивалентности: `total_material` совпадает с golden **и** план валиден (спрос удовлетворён, запас не превышен, куски+пропилы заполняют доску ровно).
- Недопустимая ILP → `TotalMaterial = -1`, `Feasible = false` (аналог Python `(-1, [])`). Пустой склад или нет валидных требований → `TotalMaterial = 0`, `Feasible = true`.

---

### Task 1: Module init + bounded knapsack

**Files:**
- Create: `go.mod`
- Create: `internal/knapsack/knapsack.go`
- Test: `internal/knapsack/knapsack_test.go`

**Interfaces:**
- Produces: `knapsack.Solve(values []float64, weights []int, counts []int, capacity int) (float64, []int)` — максимальная ценность и выбранное количество каждого типа. Порт Python `solve_bounded_knapsack_optimized`: бинарное разложение, DP по **точному** весу, при недостижимой ёмкости — лучший достижимый вес.

- [ ] **Step 1: Init module**

Run:
```bash
cd /home/kositsyn-pa/work/woodcutter
go mod init github.com/pkositsyn/woodcutter
```
Expected: создан `go.mod` с `module github.com/pkositsyn/woodcutter` и `go 1.25`.

- [ ] **Step 2: Write the failing test**

Create `internal/knapsack/knapsack_test.go`:
```go
package knapsack

import (
	"math"
	"math/rand"
	"testing"
)

// brute enumerates every count combination within limits and returns the best
// value whose exact total weight is <= capacity (mirrors the reference DP's
// exact-weight semantics is NOT required here: brute is a looser oracle used
// only to sanity-check that Solve never exceeds capacity and never beats the
// true optimum). We therefore only assert Solve's value <= brute optimum and
// that the returned vector is self-consistent.
func bruteBestValue(values []float64, weights, counts []int, capacity int) float64 {
	n := len(values)
	best := 0.0
	var rec func(i, w int, v float64)
	rec = func(i, w int, v float64) {
		if i == n {
			if v > best {
				best = v
			}
			return
		}
		for c := 0; c <= counts[i]; c++ {
			nw := w + c*weights[i]
			if nw > capacity {
				break
			}
			rec(i+1, nw, v+float64(c)*values[i])
		}
	}
	rec(0, 0, 0)
	return best
}

func TestSolveExactSmall(t *testing.T) {
	// One item type, weight 3, value 10, up to 3 copies, capacity 10.
	// Exact-weight DP: best exact weight <= 10 is 9 (three copies), value 30.
	val, vec := Solve([]float64{10}, []int{3}, []int{3}, 10)
	if val != 30 {
		t.Fatalf("value = %v, want 30", val)
	}
	if vec[0] != 3 {
		t.Fatalf("vec = %v, want [3]", vec)
	}
}

func TestSolveVectorConsistent(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for iter := 0; iter < 200; iter++ {
		n := 1 + rng.Intn(4)
		values := make([]float64, n)
		weights := make([]int, n)
		counts := make([]int, n)
		for i := 0; i < n; i++ {
			values[i] = float64(1 + rng.Intn(20))
			weights[i] = 1 + rng.Intn(6)
			counts[i] = 1 + rng.Intn(4)
		}
		capacity := 1 + rng.Intn(25)
		val, vec := Solve(values, weights, counts, capacity)
		// Reconstructed value matches vec dotted with values.
		var got, wsum float64
		for i := 0; i < n; i++ {
			got += float64(vec[i]) * values[i]
			wsum += float64(vec[i] * weights[i])
		}
		if math.Abs(got-val) > 1e-9 {
			t.Fatalf("value %v != reconstructed %v (vec %v)", val, got, vec)
		}
		if int(wsum) > capacity {
			t.Fatalf("weight %v exceeds capacity %d (vec %v)", wsum, capacity, vec)
		}
		if val > bruteBestValue(values, weights, counts, capacity)+1e-9 {
			t.Fatalf("value %v beats brute optimum", val)
		}
	}
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `go test ./internal/knapsack/`
Expected: FAIL — `undefined: Solve`.

- [ ] **Step 4: Write minimal implementation**

Create `internal/knapsack/knapsack.go`:
```go
// Package knapsack solves the bounded knapsack problem via binary
// decomposition, a direct port of the reference Python implementation.
package knapsack

// Solve returns the maximum achievable value and the chosen count of each item
// type. It reproduces the reference DP exactly: items are binary-decomposed,
// dp[w] is the best value whose chosen weights sum to EXACTLY w, and when the
// full capacity is not exactly composable the best reachable weight is used.
func Solve(values []float64, weights []int, counts []int, capacity int) (float64, []int) {
	n := len(values)

	var binVal []float64
	var binWt []int
	type mapEntry struct{ orig, cnt int }
	var mapping []mapEntry

	for i := 0; i < n; i++ {
		j := 1
		remaining := counts[i]
		for j <= remaining {
			binVal = append(binVal, values[i]*float64(j))
			binWt = append(binWt, weights[i]*j)
			mapping = append(mapping, mapEntry{i, j})
			remaining -= j
			j *= 2
		}
		if remaining > 0 {
			binVal = append(binVal, values[i]*float64(remaining))
			binWt = append(binWt, weights[i]*remaining)
			mapping = append(mapping, mapEntry{i, remaining})
		}
	}

	binN := len(binVal)
	dp := make([]float64, capacity+1)
	pick := make([]int, capacity+1) // binary item chosen to reach w, or -1
	prev := make([]int, capacity+1) // predecessor weight, or -1
	for w := range pick {
		pick[w] = -1
		prev[w] = -1
	}
	maxWeight := 0

	for w := 1; w <= capacity; w++ {
		for i := 0; i < binN; i++ {
			if binWt[i] <= w && dp[w-binWt[i]]+binVal[i] > dp[w] {
				dp[w] = dp[w-binWt[i]] + binVal[i]
				pick[w] = i
				prev[w] = w - binWt[i]
				if dp[w] > 0 && w > maxWeight {
					maxWeight = w
				}
			}
		}
	}

	target := capacity
	if dp[capacity] == 0 {
		target = maxWeight
	}

	result := make([]int, n)
	for w := target; w > 0 && pick[w] != -1; w = prev[w] {
		m := mapping[pick[w]]
		result[m.orig] += m.cnt
	}
	return dp[target], result
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `go test ./internal/knapsack/`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add go.mod internal/knapsack/
git commit -m "feat: bounded knapsack solver (port of reference DP)"
```

---

### Task 2: LP solver — two-phase tableau simplex

**Files:**
- Create: `internal/lp/lp.go`
- Test: `internal/lp/lp_test.go`

**Interfaces:**
- Produces:
  - `lp.ConstraintType` with `lp.LessEqual`, `lp.GreaterEqual`, `lp.Equal`.
  - `lp.Constraint{ Coeffs []float64; Type ConstraintType; RHS float64 }`.
  - `lp.Problem{ Objective []float64; Constraints []Constraint }` (minimize `Objective·x`, `x >= 0`).
  - `lp.Status` with `lp.Optimal`, `lp.Infeasible`, `lp.Unbounded`.
  - `lp.Solution{ Status Status; Objective float64; X []float64; Duals []float64 }`.
  - `lp.Solve(p Problem) Solution` — duals follow standard min-LP convention: `>=` constraint dual `>= 0`, `<=` constraint dual `<= 0`. Assumes all `RHS >= 0` (true for cutting).

- [ ] **Step 1: Write the failing test**

Create `internal/lp/lp_test.go`:
```go
package lp

import (
	"math"
	"testing"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

// Classic LP: maximize 3x+5y  =>  minimize -3x-5y
// s.t. x <= 4 ; 2y <= 12 ; 3x+2y <= 18 ; x,y >= 0. Optimum (2,6), obj -36.
func TestSolveMaxProfit(t *testing.T) {
	p := Problem{
		Objective: []float64{-3, -5},
		Constraints: []Constraint{
			{Coeffs: []float64{1, 0}, Type: LessEqual, RHS: 4},
			{Coeffs: []float64{0, 2}, Type: LessEqual, RHS: 12},
			{Coeffs: []float64{3, 2}, Type: LessEqual, RHS: 18},
		},
	}
	s := Solve(p)
	if s.Status != Optimal {
		t.Fatalf("status = %v, want Optimal", s.Status)
	}
	if !approx(s.Objective, -36) {
		t.Fatalf("obj = %v, want -36", s.Objective)
	}
	if !approx(s.X[0], 2) || !approx(s.X[1], 6) {
		t.Fatalf("x = %v, want [2 6]", s.X)
	}
}

// Min with >= constraints, checks duals sign/values.
// minimize 2a+3b  s.t. a+b >= 10 ; a >= 3 ; a,b >= 0.
// Optimum a=3? cost pushes b up: a+b>=10, min 2a+3b. Cheapest is maximize a.
// a can be large but a only lower-bounded; b fills rest. Min at a=10,b=0 -> 20.
// Constraint a+b>=10 binding (dual 2), a>=3 slack (dual 0).
func TestSolveMinDuals(t *testing.T) {
	p := Problem{
		Objective: []float64{2, 3},
		Constraints: []Constraint{
			{Coeffs: []float64{1, 1}, Type: GreaterEqual, RHS: 10},
			{Coeffs: []float64{1, 0}, Type: GreaterEqual, RHS: 3},
		},
	}
	s := Solve(p)
	if s.Status != Optimal {
		t.Fatalf("status = %v", s.Status)
	}
	if !approx(s.Objective, 20) {
		t.Fatalf("obj = %v, want 20", s.Objective)
	}
	// >= duals must be non-negative; binding row dual = 2, slack row dual = 0.
	if s.Duals[0] < -1e-9 || s.Duals[1] < -1e-9 {
		t.Fatalf("duals must be >= 0 for >=: %v", s.Duals)
	}
	if !approx(s.Duals[0], 2) || !approx(s.Duals[1], 0) {
		t.Fatalf("duals = %v, want [2 0]", s.Duals)
	}
}

func TestSolveInfeasible(t *testing.T) {
	p := Problem{
		Objective: []float64{1},
		Constraints: []Constraint{
			{Coeffs: []float64{1}, Type: GreaterEqual, RHS: 5},
			{Coeffs: []float64{1}, Type: LessEqual, RHS: 2},
		},
	}
	s := Solve(p)
	if s.Status != Infeasible {
		t.Fatalf("status = %v, want Infeasible", s.Status)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/lp/`
Expected: FAIL — `undefined: Solve`.

- [ ] **Step 3: Write the implementation**

Create `internal/lp/lp.go`:
```go
// Package lp implements a small dense two-phase tableau simplex for linear
// programs (minimize c·x, x >= 0) plus dual-price extraction. No external deps.
package lp

import "math"

type ConstraintType int

const (
	LessEqual ConstraintType = iota
	GreaterEqual
	Equal
)

type Constraint struct {
	Coeffs []float64
	Type   ConstraintType
	RHS    float64
}

type Problem struct {
	Objective   []float64
	Constraints []Constraint
}

type Status int

const (
	Optimal Status = iota
	Infeasible
	Unbounded
)

type Solution struct {
	Status    Status
	Objective float64
	X         []float64
	Duals     []float64
}

const eps = 1e-9

// Solve minimizes p.Objective·x subject to the constraints with x >= 0.
// Requires RHS >= 0 for correct dual signs (all cutting RHS are non-negative).
func Solve(p Problem) Solution {
	m := len(p.Constraints)
	n := len(p.Objective)

	// Count extra columns: <= -> slack; >= -> surplus + artificial; == -> artificial.
	extra := 0
	for _, c := range p.Constraints {
		switch c.Type {
		case LessEqual:
			extra++
		case GreaterEqual:
			extra += 2
		case Equal:
			extra++
		}
	}
	total := n + extra

	rows := make([][]float64, m)
	cost := make([]float64, total) // original cost; extras are 0
	copy(cost, p.Objective)
	artificial := make([]bool, total)
	logical := make([]int, m) // identity column of constraint i (slack or artificial)
	basis := make([]int, m)

	col := n
	for i, c := range p.Constraints {
		row := make([]float64, total+1)
		rhs := c.RHS
		sign := 1.0
		if rhs < 0 {
			sign, rhs = -1.0, -rhs
		}
		for j := 0; j < n; j++ {
			if j < len(c.Coeffs) {
				row[j] = sign * c.Coeffs[j]
			}
		}
		row[total] = rhs
		ctype := c.Type
		if sign < 0 {
			if ctype == LessEqual {
				ctype = GreaterEqual
			} else if ctype == GreaterEqual {
				ctype = LessEqual
			}
		}
		switch ctype {
		case LessEqual:
			row[col] = 1
			logical[i], basis[i] = col, col
			col++
		case GreaterEqual:
			row[col] = -1 // surplus
			col++
			row[col] = 1 // artificial
			artificial[col] = true
			logical[i], basis[i] = col, col
			col++
		case Equal:
			row[col] = 1
			artificial[col] = true
			logical[i], basis[i] = col, col
			col++
		}
		rows[i] = row
	}

	// Phase 1: minimize sum of artificials.
	phase1 := make([]float64, total)
	for j := 0; j < total; j++ {
		if artificial[j] {
			phase1[j] = 1
		}
	}
	forbid := make([]bool, total)
	optimize(rows, basis, phase1, forbid, total)

	// Feasibility: phase-1 objective must be ~0.
	infeas := 0.0
	for r := 0; r < m; r++ {
		infeas += phase1[basis[r]] * rows[r][total]
	}
	if infeas > 1e-7 {
		return Solution{Status: Infeasible}
	}

	// Phase 2: minimize original cost; forbid artificials from re-entering.
	for j := 0; j < total; j++ {
		if artificial[j] {
			forbid[j] = true
		}
	}
	if optimize(rows, basis, cost, forbid, total) == Unbounded {
		return Solution{Status: Unbounded}
	}

	x := make([]float64, n)
	for r := 0; r < m; r++ {
		if basis[r] < n {
			x[basis[r]] = rows[r][total]
		}
	}
	obj := 0.0
	for r := 0; r < m; r++ {
		obj += cost[basis[r]] * rows[r][total]
	}
	// Duals: y = c_B · B^{-1}; B^{-1} column i sits under logical[i].
	duals := make([]float64, m)
	for i := 0; i < m; i++ {
		var y float64
		for r := 0; r < m; r++ {
			y += cost[basis[r]] * rows[r][logical[i]]
		}
		duals[i] = y
	}
	return Solution{Status: Optimal, Objective: obj, X: x, Duals: duals}
}

// optimize runs primal simplex (minimization) on the tableau in place using
// Bland's rule for anti-cycling. Returns Optimal or Unbounded.
func optimize(rows [][]float64, basis []int, cost, forbid []bool0, total int) Status { return 0 }
```

Note: the last line above is a deliberate placeholder to be replaced in Step 4 (the signature has a typo `[]bool0`). Do not keep it.

- [ ] **Step 4: Replace the placeholder `optimize` with the real one**

Replace the final `optimize` stub in `internal/lp/lp.go` with:
```go
// optimize runs primal simplex (minimization) on the tableau in place using
// Bland's rule for anti-cycling. Returns Optimal or Unbounded.
func optimize(rows [][]float64, basis []int, cost []float64, forbid []bool, total int) Status {
	m := len(rows)
	for {
		entering := -1
		for j := 0; j < total; j++ {
			if forbid[j] {
				continue
			}
			rc := cost[j]
			for r := 0; r < m; r++ {
				rc -= cost[basis[r]] * rows[r][j]
			}
			if rc < -eps {
				entering = j // Bland: first improving column
				break
			}
		}
		if entering == -1 {
			return Optimal
		}
		leaving := -1
		best := math.Inf(1)
		for r := 0; r < m; r++ {
			a := rows[r][entering]
			if a <= eps {
				continue
			}
			ratio := rows[r][total] / a
			if leaving == -1 || ratio < best-eps ||
				(ratio <= best+eps && basis[r] < basis[leaving]) {
				best, leaving = ratio, r
			}
		}
		if leaving == -1 {
			return Unbounded
		}
		pivot(rows, basis, leaving, entering, total)
	}
}

func pivot(rows [][]float64, basis []int, pr, pc, total int) {
	piv := rows[pr][pc]
	for j := 0; j <= total; j++ {
		rows[pr][j] /= piv
	}
	for r := 0; r < len(rows); r++ {
		if r == pr {
			continue
		}
		f := rows[r][pc]
		if f == 0 {
			continue
		}
		for j := 0; j <= total; j++ {
			rows[r][j] -= f * rows[pr][j]
		}
	}
	basis[pr] = pc
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `go test ./internal/lp/ -run 'TestSolve'`
Expected: PASS (all three). If `TestSolveMinDuals` fails on sign, the dual convention is inverted — negate `duals[i]`; but with the derivation above it should be correct.

- [ ] **Step 6: Commit**

```bash
git add internal/lp/lp.go internal/lp/lp_test.go
git commit -m "feat: two-phase tableau simplex with dual prices"
```

---

### Task 3: MILP branch & bound

**Files:**
- Modify: `internal/lp/lp.go` (append)
- Test: `internal/lp/milp_test.go`

**Interfaces:**
- Consumes: `Solve`, `Problem`, `Constraint`, `Solution`, `Optimal`, `Infeasible`, `LessEqual`, `GreaterEqual`.
- Produces: `lp.SolveMILP(p Problem, integer []bool) Solution` — minimizes over `x >= 0` with `x[j]` integer where `integer[j]`. Returns `Status Optimal` with integer `X`, or `Infeasible`.

- [ ] **Step 1: Write the failing test**

Create `internal/lp/milp_test.go`:
```go
package lp

import (
	"math"
	"testing"
)

// minimize x+y s.t. x+y >= 3.5 ; integers -> optimum 4 (e.g. x=4,y=0 or 2,2).
func TestMILPRounds(t *testing.T) {
	p := Problem{
		Objective: []float64{1, 1},
		Constraints: []Constraint{
			{Coeffs: []float64{1, 1}, Type: GreaterEqual, RHS: 3.5},
		},
	}
	s := SolveMILP(p, []bool{true, true})
	if s.Status != Optimal {
		t.Fatalf("status = %v", s.Status)
	}
	if !approx(s.Objective, 4) {
		t.Fatalf("obj = %v, want 4", s.Objective)
	}
	for j, v := range s.X {
		if math.Abs(v-math.Round(v)) > 1e-6 {
			t.Fatalf("x[%d]=%v not integer", j, v)
		}
	}
}

// A tiny covering ILP against brute force.
func TestMILPBruteAgree(t *testing.T) {
	// minimize 5a+4b s.t. 2a+b >= 3 ; a+3b >= 4 ; a,b >= 0 integer.
	p := Problem{
		Objective: []float64{5, 4},
		Constraints: []Constraint{
			{Coeffs: []float64{2, 1}, Type: GreaterEqual, RHS: 3},
			{Coeffs: []float64{1, 3}, Type: GreaterEqual, RHS: 4},
		},
	}
	s := SolveMILP(p, []bool{true, true})
	// brute force over a,b in 0..6
	best := math.Inf(1)
	for a := 0; a <= 6; a++ {
		for b := 0; b <= 6; b++ {
			if 2*a+b >= 3 && a+3*b >= 4 {
				c := float64(5*a + 4*b)
				if c < best {
					best = c
				}
			}
		}
	}
	if s.Status != Optimal || !approx(s.Objective, best) {
		t.Fatalf("milp obj %v (status %v), brute %v", s.Objective, s.Status, best)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/lp/ -run TestMILP`
Expected: FAIL — `undefined: SolveMILP`.

- [ ] **Step 3: Write the implementation**

Append to `internal/lp/lp.go`:
```go
// SolveMILP solves the MILP by LP-relaxation branch & bound (DFS with bounding).
// integer[j]==true forces x[j] to an integer. Minimization only.
func SolveMILP(p Problem, integer []bool) Solution {
	best := Solution{Status: Infeasible, Objective: math.Inf(1)}

	var rec func(extra []Constraint)
	rec = func(extra []Constraint) {
		cons := make([]Constraint, 0, len(p.Constraints)+len(extra))
		cons = append(cons, p.Constraints...)
		cons = append(cons, extra...)
		sol := Solve(Problem{Objective: p.Objective, Constraints: cons})
		if sol.Status != Optimal {
			return
		}
		if sol.Objective >= best.Objective-eps {
			return // LP bound cannot beat incumbent
		}
		frac := -1
		for j := range sol.X {
			if j < len(integer) && integer[j] {
				if d := sol.X[j] - math.Floor(sol.X[j]); d > 1e-6 && d < 1-1e-6 {
					frac = j
					break
				}
			}
		}
		if frac == -1 {
			// integer-feasible and strictly better
			sol.Status = Optimal
			best = sol
			return
		}
		v := sol.X[frac]
		down := Constraint{Coeffs: unit(frac, len(p.Objective)), Type: LessEqual, RHS: math.Floor(v)}
		up := Constraint{Coeffs: unit(frac, len(p.Objective)), Type: GreaterEqual, RHS: math.Ceil(v)}
		rec(appendCons(extra, down))
		rec(appendCons(extra, up))
	}
	rec(nil)

	if math.IsInf(best.Objective, 1) {
		return Solution{Status: Infeasible}
	}
	return best
}

func unit(idx, n int) []float64 {
	v := make([]float64, n)
	v[idx] = 1
	return v
}

func appendCons(base []Constraint, c Constraint) []Constraint {
	out := make([]Constraint, len(base)+1)
	copy(out, base)
	out[len(base)] = c
	return out
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./internal/lp/`
Expected: PASS (LP + MILP tests).

- [ ] **Step 5: Commit**

```bash
git add internal/lp/lp.go internal/lp/milp_test.go
git commit -m "feat: MILP branch and bound on top of simplex"
```

---

### Task 4: cutting types + stdin parsing

**Files:**
- Create: `internal/cutting/types.go`
- Create: `internal/cutting/parse.go`
- Test: `internal/cutting/parse_test.go`

**Interfaces:**
- Produces:
  - `cutting.Pair{ Length, Count int }`.
  - `cutting.PieceKind` with `cutting.PieceCut`, `cutting.PiecePadding`, `cutting.PieceWaste`.
  - `cutting.Piece{ Length, Start, End int; Kind PieceKind }`.
  - `cutting.Board{ StockLength int; Pieces []Piece }`.
  - `cutting.Plan{ TotalMaterial int; Feasible bool; Boards []Board; Dropped []Pair }`.
  - `cutting.Options{ Padding int }`.
  - `cutting.ParseInput(r io.Reader) (stock, requirements []Pair, err error)` — requirements aggregated by length in first-seen order; stock raw. Error if no `---` separator.

- [ ] **Step 1: Write the failing test**

Create `internal/cutting/parse_test.go`:
```go
package cutting

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseInputBasic(t *testing.T) {
	in := "6000, 10\n3000\n---\n2000, 3\n2000, 2\n1500\n"
	stock, reqs, err := ParseInput(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	wantStock := []Pair{{6000, 10}, {3000, 1}}
	if !reflect.DeepEqual(stock, wantStock) {
		t.Fatalf("stock = %v, want %v", stock, wantStock)
	}
	// 2000 aggregated to 5, first-seen order: 2000 then 1500.
	wantReq := []Pair{{2000, 5}, {1500, 1}}
	if !reflect.DeepEqual(reqs, wantReq) {
		t.Fatalf("reqs = %v, want %v", reqs, wantReq)
	}
}

func TestParseInputNoSeparator(t *testing.T) {
	_, _, err := ParseInput(strings.NewReader("6000, 10\n3000\n"))
	if err == nil {
		t.Fatal("expected error for missing '---'")
	}
}

func TestParseLineTab(t *testing.T) {
	p, err := parseLine("1 500\t4")
	if err != nil {
		t.Fatal(err)
	}
	if p != (Pair{1500, 4}) {
		t.Fatalf("got %v, want {1500 4}", p)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/cutting/`
Expected: FAIL — undefined `ParseInput`, `Pair`, `parseLine`.

- [ ] **Step 3: Write the types**

Create `internal/cutting/types.go`:
```go
package cutting

// Pair is a (length, count) tuple used for both stock and requirements.
type Pair struct {
	Length int
	Count  int
}

// PieceKind classifies a segment of a cut board.
type PieceKind int

const (
	PieceCut PieceKind = iota
	PiecePadding
	PieceWaste
)

// Piece is one segment on a board: a produced cut, a saw-kerf padding, or waste.
type Piece struct {
	Length int
	Start  int
	End    int
	Kind   PieceKind
}

// Board is a single physical board with its cutting layout.
type Board struct {
	StockLength int
	Pieces      []Piece
}

// Plan is the full cutting result.
type Plan struct {
	TotalMaterial int // sum of used board lengths; -1 when infeasible
	Feasible      bool
	Boards        []Board
	Dropped       []Pair // requirements longer than the longest stock board
}

// Options configures the solver.
type Options struct {
	Padding int
}
```

- [ ] **Step 4: Write the parser**

Create `internal/cutting/parse.go`:
```go
package cutting

import (
	"bufio"
	"errors"
	"io"
	"regexp"
	"strconv"
	"strings"
)

var sepRE = regexp.MustCompile(`[,\t]`)

// ParseInput reads the "stock --- requirements" format. Requirements are
// aggregated by length preserving first-seen order; stock is returned raw.
func ParseInput(r io.Reader) (stock, requirements []Pair, err error) {
	sc := bufio.NewScanner(r)
	var stockLines, reqLines []string
	cur := &stockLines
	seenSep := false
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if line == "---" {
			cur = &reqLines
			seenSep = true
			continue
		}
		*cur = append(*cur, line)
	}
	if e := sc.Err(); e != nil {
		return nil, nil, e
	}
	if !seenSep {
		return nil, nil, errors.New("не найден разделитель '---' между секцией склада и секцией требований")
	}
	for _, l := range stockLines {
		p, e := parseLine(l)
		if e != nil {
			return nil, nil, e
		}
		stock = append(stock, p)
	}
	agg := map[int]int{}
	var order []int
	for _, l := range reqLines {
		p, e := parseLine(l)
		if e != nil {
			return nil, nil, e
		}
		if _, ok := agg[p.Length]; !ok {
			order = append(order, p.Length)
		}
		agg[p.Length] += p.Count
	}
	for _, L := range order {
		requirements = append(requirements, Pair{Length: L, Count: agg[L]})
	}
	return stock, requirements, nil
}

func parseLine(s string) (Pair, error) {
	parts := sepRE.Split(s, -1)
	length, err := strconv.Atoi(strings.ReplaceAll(strings.TrimSpace(parts[0]), " ", ""))
	if err != nil {
		return Pair{}, err
	}
	count := 1
	if len(parts) > 1 {
		count, err = strconv.Atoi(strings.TrimSpace(parts[1]))
		if err != nil {
			return Pair{}, err
		}
	}
	return Pair{Length: length, Count: count}, nil
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `go test ./internal/cutting/ -run TestParse`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/cutting/types.go internal/cutting/parse.go internal/cutting/parse_test.go
git commit -m "feat: cutting types and stdin parser"
```

---

### Task 5: cutting solver — column generation + plan

**Files:**
- Create: `internal/cutting/solve.go`
- Test: `internal/cutting/solve_test.go`

**Interfaces:**
- Consumes: `Pair`, `Plan`, `Board`, `Piece`, `PieceCut/PiecePadding/PieceWaste`, `Options`; `lp.Solve`, `lp.SolveMILP`, `lp.Problem`, `lp.Constraint`, `lp.GreaterEqual`, `lp.LessEqual`, `lp.Optimal`; `knapsack.Solve`.
- Produces: `cutting.Solve(stock, requirements []Pair, opts Options) Plan` — port of `min_material_cutting`.

- [ ] **Step 1: Write the failing test**

Create `internal/cutting/solve_test.go`:
```go
package cutting

import "testing"

// One 6000mm board type, unlimited-ish supply, need three 2000mm pieces.
// With padding 5, two 2000mm+kerf pieces fit one board (2*2005=4010 <= 6005),
// three pieces need two boards -> total_material 12000.
func TestSolveSimple(t *testing.T) {
	stock := []Pair{{6000, 100}}
	reqs := []Pair{{2000, 3}}
	plan := Solve(stock, reqs, Options{Padding: 5})
	if !plan.Feasible {
		t.Fatal("expected feasible")
	}
	if plan.TotalMaterial != 12000 {
		t.Fatalf("total = %d, want 12000", plan.TotalMaterial)
	}
	assertPlanValid(t, plan, stock, reqs, 5)
}

func TestSolveInfeasibleSupply(t *testing.T) {
	// Need three 2000mm pieces but only one 6000mm board -> cannot satisfy.
	stock := []Pair{{6000, 1}}
	reqs := []Pair{{2000, 3}}
	plan := Solve(stock, reqs, Options{Padding: 5})
	if plan.Feasible || plan.TotalMaterial != -1 {
		t.Fatalf("expected infeasible (-1), got feasible=%v total=%d", plan.Feasible, plan.TotalMaterial)
	}
}

func TestSolveDropsTooLong(t *testing.T) {
	stock := []Pair{{3000, 10}}
	reqs := []Pair{{5000, 1}, {1000, 2}}
	plan := Solve(stock, reqs, Options{Padding: 5})
	if !plan.Feasible {
		t.Fatal("expected feasible")
	}
	if len(plan.Dropped) != 1 || plan.Dropped[0].Length != 5000 {
		t.Fatalf("dropped = %v, want one 5000", plan.Dropped)
	}
}

func TestSolveEmptyStock(t *testing.T) {
	plan := Solve(nil, []Pair{{1000, 1}}, Options{Padding: 5})
	if !plan.Feasible || plan.TotalMaterial != 0 {
		t.Fatalf("empty stock: feasible=%v total=%d", plan.Feasible, plan.TotalMaterial)
	}
}
```

Note: `assertPlanValid` is defined in Task 7's `validate_test.go`. To keep this task independently runnable, temporarily add a local stub at the bottom of `solve_test.go`, then DELETE it in Task 7 Step 1:
```go
// TEMP stub — remove in Task 7 when validate_test.go lands.
func assertPlanValid(t *testing.T, plan Plan, stock, reqs []Pair, padding int) {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/cutting/ -run TestSolveSimple`
Expected: FAIL — `undefined: Solve`.

- [ ] **Step 3: Write the implementation**

Create `internal/cutting/solve.go`:
```go
package cutting

import (
	"fmt"
	"math"
	"sort"

	"github.com/pkositsyn/woodcutter/internal/knapsack"
	"github.com/pkositsyn/woodcutter/internal/lp"
)

const tolerance = 1e-6

// Solve is a Go port of the reference min_material_cutting: Gilmore-Gomory
// column generation over multiple stock lengths with limited supply, then an
// integer solve over the generated pattern pool.
func Solve(stock, requirements []Pair, opts Options) Plan {
	padding := opts.Padding

	// Aggregate stock by length, sort descending.
	stockAgg := map[int]int{}
	for _, s := range stock {
		stockAgg[s.Length] += s.Count
	}
	if len(stockAgg) == 0 {
		return Plan{Feasible: true, TotalMaterial: 0}
	}
	stockLen := make([]int, 0, len(stockAgg))
	for L := range stockAgg {
		stockLen = append(stockLen, L)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(stockLen)))
	nStock := len(stockLen)
	stockSup := make([]int, nStock)
	capacity := make([]int, nStock)
	for k, L := range stockLen {
		stockSup[k] = stockAgg[L]
		capacity[k] = L + padding
	}
	maxStock := stockLen[0]

	// Filter requirements longer than the longest board; sort desc by length.
	var valid, dropped []Pair
	for _, r := range requirements {
		if r.Length <= maxStock {
			valid = append(valid, r)
		} else {
			dropped = append(dropped, r)
		}
	}
	sort.Slice(valid, func(i, j int) bool { return valid[i].Length > valid[j].Length })
	if len(valid) == 0 {
		return Plan{Feasible: true, TotalMaterial: 0, Dropped: dropped}
	}

	nReq := len(valid)
	reqLen := make([]int, nReq)
	reqCnt := make([]int, nReq)
	reqW := make([]int, nReq)
	for i, r := range valid {
		reqLen[i] = r.Length
		reqCnt[i] = r.Count
		reqW[i] = r.Length + padding
	}

	// Pattern pool.
	var patterns [][]int
	var patStock []int
	existing := map[string]bool{}
	addPattern := func(k int, vec []int) bool {
		key := fmt.Sprintf("%d|%v", k, vec)
		if existing[key] {
			return false
		}
		existing[key] = true
		cp := append([]int(nil), vec...)
		patterns = append(patterns, cp)
		patStock = append(patStock, k)
		return true
	}

	// Initial greedy single-type patterns.
	for k := 0; k < nStock; k++ {
		for i := 0; i < nReq; i++ {
			if reqW[i] <= capacity[k] {
				vec := make([]int, nReq)
				vec[i] = capacity[k] / reqW[i]
				addPattern(k, vec)
			}
		}
	}

	buildConstraints := func() []lp.Constraint {
		np := len(patterns)
		cons := make([]lp.Constraint, 0, nReq+nStock)
		for i := 0; i < nReq; i++ {
			row := make([]float64, np)
			for j := 0; j < np; j++ {
				row[j] = float64(patterns[j][i])
			}
			cons = append(cons, lp.Constraint{Coeffs: row, Type: lp.GreaterEqual, RHS: float64(reqCnt[i])})
		}
		for k := 0; k < nStock; k++ {
			row := make([]float64, np)
			for j := 0; j < np; j++ {
				if patStock[j] == k {
					row[j] = 1
				}
			}
			cons = append(cons, lp.Constraint{Coeffs: row, Type: lp.LessEqual, RHS: float64(stockSup[k])})
		}
		return cons
	}
	objective := func() []float64 {
		np := len(patterns)
		obj := make([]float64, np)
		for j := 0; j < np; j++ {
			obj[j] = float64(stockLen[patStock[j]])
		}
		return obj
	}

	// Column generation.
	const maxIter = 1000
	for iter := 0; iter < maxIter; iter++ {
		sol := lp.Solve(lp.Problem{Objective: objective(), Constraints: buildConstraints()})
		if sol.Status != lp.Optimal {
			break
		}
		y := sol.Duals[:nReq]
		z := sol.Duals[nReq : nReq+nStock]
		added := false
		for k := 0; k < nStock; k++ {
			bestVal, bestVec := knapsack.Solve(y, reqW, reqCnt, capacity[k])
			reduced := float64(stockLen[k]) - bestVal - z[k]
			if reduced < -tolerance {
				if addPattern(k, bestVec) {
					added = true
				}
			}
		}
		if !added {
			break
		}
	}

	// Integer solve over the pool.
	np := len(patterns)
	integer := make([]bool, np)
	for j := range integer {
		integer[j] = true
	}
	sol := lp.SolveMILP(lp.Problem{Objective: objective(), Constraints: buildConstraints()}, integer)
	if sol.Status != lp.Optimal {
		return Plan{Feasible: false, TotalMaterial: -1, Dropped: dropped}
	}
	usage := make([]int, np)
	for j := 0; j < np; j++ {
		usage[j] = int(math.Round(sol.X[j]))
	}

	// Build the detailed plan.
	var boards []Board
	total := 0
	for j := 0; j < np; j++ {
		u := usage[j]
		if u <= 0 {
			continue
		}
		k := patStock[j]
		L := stockLen[k]
		total += u * L
		for b := 0; b < u; b++ {
			var pieces []Piece
			pos := 0
			for i := 0; i < nReq; i++ {
				for c := 0; c < patterns[j][i]; c++ {
					pieces = append(pieces, Piece{Length: reqLen[i], Start: pos, End: pos + reqLen[i], Kind: PieceCut})
					pos += reqLen[i]
					if padding > 0 {
						pieces = append(pieces, Piece{Length: padding, Start: pos, End: pos + padding, Kind: PiecePadding})
						pos += padding
					}
				}
			}
			if padding > 0 && len(pieces) > 0 && pieces[len(pieces)-1].Kind == PiecePadding {
				pieces = pieces[:len(pieces)-1]
				pos -= padding
			}
			if waste := L - pos; waste > 0 {
				pieces = append(pieces, Piece{Length: waste, Start: pos, End: L, Kind: PieceWaste})
			}
			boards = append(boards, Board{StockLength: L, Pieces: pieces})
		}
	}
	return Plan{Feasible: true, TotalMaterial: total, Boards: boards, Dropped: dropped}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./internal/cutting/ -run TestSolve`
Expected: PASS (all four). If `TestSolveSimple` total differs, re-check dual signs in Task 2.

- [ ] **Step 5: Commit**

```bash
git add internal/cutting/solve.go internal/cutting/solve_test.go
git commit -m "feat: cutting solver via column generation + integer solve"
```

---

### Task 6: report + CLI

**Files:**
- Create: `internal/cutting/report.go`
- Create: `cmd/woodcutter/main.go`
- Test: `internal/cutting/report_test.go`

**Interfaces:**
- Consumes: `Plan`, `Board`, `Piece`, `PieceCut/PiecePadding/PieceWaste`, `Pair`, `Options`, `ParseInput`, `Solve`.
- Produces: `cutting.WriteReport(w io.Writer, plan Plan, requirements []Pair, opts Options)` — clean human report. `cmd/woodcutter` wires stdin → parse → solve → report.

- [ ] **Step 1: Write the failing test**

Create `internal/cutting/report_test.go`:
```go
package cutting

import (
	"strings"
	"testing"
)

func TestWriteReportSmoke(t *testing.T) {
	plan := Plan{
		Feasible:      true,
		TotalMaterial: 6000,
		Boards: []Board{{
			StockLength: 6000,
			Pieces: []Piece{
				{Length: 2000, Start: 0, End: 2000, Kind: PieceCut},
				{Length: 5, Start: 2000, End: 2005, Kind: PiecePadding},
				{Length: 3995, Start: 2005, End: 6000, Kind: PieceWaste},
			},
		}},
	}
	var sb strings.Builder
	WriteReport(&sb, plan, []Pair{{2000, 1}}, Options{Padding: 5})
	out := sb.String()
	if !strings.Contains(out, "6000") || !strings.Contains(out, "2000") {
		t.Fatalf("report missing expected content:\n%s", out)
	}
}

func TestWriteReportInfeasible(t *testing.T) {
	var sb strings.Builder
	WriteReport(&sb, Plan{Feasible: false, TotalMaterial: -1}, nil, Options{Padding: 5})
	if !strings.Contains(sb.String(), "недопуст") {
		t.Fatalf("infeasible report unexpected:\n%s", sb.String())
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/cutting/ -run TestWriteReport`
Expected: FAIL — `undefined: WriteReport`.

- [ ] **Step 3: Write the report**

Create `internal/cutting/report.go`:
```go
package cutting

import (
	"fmt"
	"io"
	"sort"
)

// WriteReport prints a clean cutting report: per-board layout, boards by stock
// type, requirement fulfilment, and waste summary.
func WriteReport(w io.Writer, plan Plan, requirements []Pair, opts Options) {
	if !plan.Feasible {
		fmt.Fprintln(w, "Задача недопустима: запасов склада не хватает под спрос.")
		return
	}

	for _, d := range plan.Dropped {
		fmt.Fprintf(w, "Пропущено (длиннее максимальной доски): %d мм × %d\n", d.Length, d.Count)
	}

	boardsByStock := map[int]int{}
	produced := map[int]int{}
	for _, b := range plan.Boards {
		boardsByStock[b.StockLength]++
		for _, p := range b.Pieces {
			if p.Kind == PieceCut {
				produced[p.Length]++
			}
		}
	}

	fmt.Fprintf(w, "Детальный план раскроя (%d досок, %d мм материала):\n", len(plan.Boards), plan.TotalMaterial)
	for idx, b := range plan.Boards {
		fmt.Fprintf(w, "\nДоска #%d [%d мм]:\n", idx+1, b.StockLength)
		var cuts []string
		waste := 0
		for _, p := range b.Pieces {
			switch p.Kind {
			case PieceCut:
				cuts = append(cuts, fmt.Sprintf("%d мм (%d-%d)", p.Length, p.Start, p.End))
			case PieceWaste:
				waste += p.Length
			}
		}
		fmt.Fprintf(w, "  Куски: %s\n", joinComma(cuts))
		fmt.Fprintf(w, "  Отходы: %d мм\n", waste)
	}

	fmt.Fprintln(w, "\nИспользовано досок по типам:")
	types := make([]int, 0, len(boardsByStock))
	for L := range boardsByStock {
		types = append(types, L)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(types)))
	for _, L := range types {
		fmt.Fprintf(w, "  %d мм: %d шт\n", L, boardsByStock[L])
	}

	fmt.Fprintln(w, "\nПроверка выполнения требований:")
	for _, r := range requirements {
		status := "OK"
		if produced[r.Length] < r.Count {
			status = "НЕ ВЫПОЛНЕНО"
		}
		fmt.Fprintf(w, "  [%s] требуется %d × %d мм, произведено %d\n", status, r.Count, r.Length, produced[r.Length])
	}
}

func joinComma(parts []string) string {
	out := ""
	for i, p := range parts {
		if i > 0 {
			out += ", "
		}
		out += p
	}
	return out
}
```

- [ ] **Step 4: Run report tests**

Run: `go test ./internal/cutting/ -run TestWriteReport`
Expected: PASS.

- [ ] **Step 5: Write the CLI**

Create `cmd/woodcutter/main.go`:
```go
// Command woodcutter reads a "stock --- requirements" spec on stdin and prints
// an optimal 1D cutting plan. Go port of cutting.py.
package main

import (
	"fmt"
	"os"

	"github.com/pkositsyn/woodcutter/internal/cutting"
)

func main() {
	stock, requirements, err := cutting.ParseInput(os.Stdin)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}
	if len(stock) == 0 {
		fmt.Fprintln(os.Stderr, "Ошибка: пустая секция склада")
		os.Exit(1)
	}

	totalStock := 0
	for _, s := range stock {
		totalStock += s.Count
	}
	totalPieces := 0
	for _, r := range requirements {
		totalPieces += r.Count
	}
	fmt.Fprintf(os.Stderr, "Типов досок на складе: %d; всего досок: %d\n", len(stock), totalStock)
	fmt.Fprintf(os.Stderr, "Различных требуемых длин: %d; всего кусков: %d\n", len(requirements), totalPieces)

	opts := cutting.Options{Padding: 5}
	plan := cutting.Solve(stock, requirements, opts)
	cutting.WriteReport(os.Stdout, plan, requirements, opts)
}
```

- [ ] **Step 6: Build and smoke-run the CLI**

Run:
```bash
go build ./... && printf '6000, 100\n---\n2000, 3\n' | go run ./cmd/woodcutter
```
Expected: stderr shows stock/piece counts; stdout shows a plan with `12000 мм материала`.

- [ ] **Step 7: Commit**

```bash
git add internal/cutting/report.go internal/cutting/report_test.go cmd/woodcutter/main.go
git commit -m "feat: clean report and CLI entrypoint"
```

---

### Task 7: golden cross-check against Python

**Files:**
- Create: `reference/cutting.py` (vendored copy of the original, entry point guarded under `__main__`)
- Create: `reference/gen_golden.py`
- Create: `internal/cutting/testdata/*.txt` (input scenarios)
- Create: `internal/cutting/testdata/*.golden.json` (generated)
- Create: `internal/cutting/validate_test.go` (validity checks + golden runner)
- Modify: `internal/cutting/solve_test.go` (remove the temporary `assertPlanValid` stub)

**Interfaces:**
- Consumes: `ParseInput`, `Solve`, `Plan`, `Board`, `Piece`, `PieceCut`, `Options`.
- Produces: `assertPlanValid(t *testing.T, plan Plan, stock, reqs []Pair, padding int)` — validates supply, demand, and board geometry. `TestGoldenCrossCheck` compares `TotalMaterial`/`Feasible` against `*.golden.json`.

- [ ] **Step 1: Remove the temporary stub**

In `internal/cutting/solve_test.go`, delete the temporary block:
```go
// TEMP stub — remove in Task 7 when validate_test.go lands.
func assertPlanValid(t *testing.T, plan Plan, stock, reqs []Pair, padding int) {}
```

- [ ] **Step 2: Vendor the reference and guard its entry point**

Copy the original into `reference/cutting.py`:
```bash
mkdir -p reference
cp "/home/kositsyn-pa/Downloads/Telegram Desktop/cutting.py" reference/cutting.py
```
Then edit `reference/cutting.py`: wrap the module-level entry point (everything from `# --- Точка входа ---` / `stock_raw, req_raw = parse_input(sys.stdin)` to the end) inside a guard so importing the module does not read stdin:
```python
def main():
    stock_raw, req_raw = parse_input(sys.stdin)
    stock = [parse_line(l) for l in stock_raw]
    req_counter = Counter()
    for length, count in (parse_line(l) for l in req_raw):
        req_counter[length] += count
    requirements = list(req_counter.items())
    if not stock:
        print("Ошибка: пустая секция склада", file=sys.stderr)
        sys.exit(1)
    print(f"Типов досок на складе: {len(stock)}; всего досок: {sum(c for _, c in stock)}", file=sys.stderr)
    print(f"Различных требуемых длин: {len(requirements)}; всего кусков: {sum(c for _, c in requirements)}", file=sys.stderr)
    total_material, cutting_plan = min_material_cutting(requirements, stock, show_paddings=False, verbose=True)


if __name__ == "__main__":
    main()
```
The two functions `solve_bounded_knapsack_optimized` and `min_material_cutting` stay untouched — only the top-level entry lines move into `main()`.

- [ ] **Step 3: Create input scenarios**

Create these files under `internal/cutting/testdata/`:

`single_type.txt`:
```
6000, 100
---
2000, 3
1500, 4
800, 5
```

`multi_type.txt`:
```
6000, 20
4000, 20
3000, 20
---
2500, 6
1800, 8
1200, 10
700, 12
```

`limited_supply.txt`:
```
6000, 3
---
2000, 6
1500, 3
```

`infeasible.txt`:
```
6000, 1
---
2000, 3
```

`too_long.txt`:
```
3000, 10
---
5000, 1
1000, 4
2500, 3
```

`no_padding_case.txt`:
```
5000, 10
---
2500, 4
1000, 6
```

- [ ] **Step 4: Write the golden generator**

Create `reference/gen_golden.py`:
```python
"""Regenerate golden files from the vendored reference (pulp) solver.

Run from the repository root:
    python3 reference/gen_golden.py
Requires python3 + pulp. Padding is fixed at 5 to match cutting.Options.
"""
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from cutting import parse_line, min_material_cutting  # noqa: E402

TESTDATA = os.path.join(os.path.dirname(__file__), "..", "internal", "cutting", "testdata")


def load(path):
    stock_lines, req_lines, section, seen = [], [], "stock", False
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line == "---":
                section, seen = "req", True
                continue
            (stock_lines if section == "stock" else req_lines).append(line)
    stock = [parse_line(l) for l in stock_lines]
    counter = Counter()
    for length, count in (parse_line(l) for l in req_lines):
        counter[length] += count
    return stock, list(counter.items())


def main():
    for path in sorted(glob.glob(os.path.join(TESTDATA, "*.txt"))):
        stock, requirements = load(path)
        total, _ = min_material_cutting(requirements, stock, padding=5, verbose=False)
        golden = os.path.splitext(path)[0] + ".golden.json"
        with open(golden, "w", encoding="utf-8") as fh:
            json.dump({"total_material": total, "feasible": total != -1}, fh)
        print(f"{os.path.basename(path)} -> total_material={total}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Generate the golden files**

Run:
```bash
cd /home/kositsyn-pa/work/woodcutter && python3 reference/gen_golden.py
```
Expected: prints `<case> -> total_material=N` per scenario; creates `internal/cutting/testdata/*.golden.json`. `infeasible.txt` should yield `total_material=-1`.

- [ ] **Step 6: Write the validator + golden runner**

Create `internal/cutting/validate_test.go`:
```go
package cutting

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// assertPlanValid checks supply limits, demand fulfilment, and board geometry.
func assertPlanValid(t *testing.T, plan Plan, stock, reqs []Pair, padding int) {
	t.Helper()

	supply := map[int]int{}
	for _, s := range stock {
		supply[s.Length] += s.Count
	}
	maxStock := 0
	for L := range supply {
		if L > maxStock {
			maxStock = L
		}
	}

	used := map[int]int{}
	produced := map[int]int{}
	total := 0
	for _, b := range plan.Boards {
		used[b.StockLength]++
		total += b.StockLength

		pos := 0
		for _, p := range b.Pieces {
			if p.Start != pos {
				t.Fatalf("piece start %d != running pos %d", p.Start, pos)
			}
			if p.End != p.Start+p.Length {
				t.Fatalf("piece end %d != start+length", p.End)
			}
			pos = p.End
			if p.Kind == PieceCut {
				produced[p.Length]++
			}
		}
		if pos != b.StockLength {
			t.Fatalf("board fills %d, want %d", pos, b.StockLength)
		}
	}
	if total != plan.TotalMaterial {
		t.Fatalf("sum of boards %d != TotalMaterial %d", total, plan.TotalMaterial)
	}
	for L, u := range used {
		if u > supply[L] {
			t.Fatalf("stock %d used %d > supply %d", L, u, supply[L])
		}
	}
	for _, r := range reqs {
		if r.Length > maxStock {
			continue // legitimately dropped
		}
		if produced[r.Length] < r.Count {
			t.Fatalf("demand %d mm: produced %d < required %d", r.Length, produced[r.Length], r.Count)
		}
	}
}

func TestGoldenCrossCheck(t *testing.T) {
	files, err := filepath.Glob("testdata/*.txt")
	if err != nil || len(files) == 0 {
		t.Fatalf("no testdata: %v", err)
	}
	for _, f := range files {
		name := strings.TrimSuffix(filepath.Base(f), ".txt")
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			stock, reqs, err := ParseInput(strings.NewReader(string(data)))
			if err != nil {
				t.Fatal(err)
			}
			plan := Solve(stock, reqs, Options{Padding: 5})

			gb, err := os.ReadFile("testdata/" + name + ".golden.json")
			if err != nil {
				t.Fatalf("missing golden (run reference/gen_golden.py): %v", err)
			}
			var g struct {
				TotalMaterial int  `json:"total_material"`
				Feasible      bool `json:"feasible"`
			}
			if err := json.Unmarshal(gb, &g); err != nil {
				t.Fatal(err)
			}
			if plan.Feasible != g.Feasible {
				t.Fatalf("feasible %v != golden %v", plan.Feasible, g.Feasible)
			}
			if plan.TotalMaterial != g.TotalMaterial {
				t.Fatalf("total_material %d != golden %d", plan.TotalMaterial, g.TotalMaterial)
			}
			if plan.Feasible {
				assertPlanValid(t, plan, stock, reqs, 5)
			}
		})
	}
}
```

- [ ] **Step 7: Run the full suite**

Run: `go test ./...`
Expected: PASS across knapsack, lp, cutting (including every golden scenario). If a golden `total_material` mismatches, that is a real divergence — debug the column-generation control flow against `reference/cutting.py` before touching the test (per the spec's risk note).

- [ ] **Step 8: Commit**

```bash
git add reference/ internal/cutting/testdata/ internal/cutting/validate_test.go internal/cutting/solve_test.go
git commit -m "test: golden cross-check against Python reference"
```

---

## Self-Review notes

- **Spec coverage:** simplex+duals (Task 2), branch&bound ILP (Task 3), knapsack port (Task 1), column-generation orchestration (Task 5), parse (Task 4), clean report+CLI (Task 6), golden cross-check on total_material + validity across all required case types — single type, multi type, unlimited (large supply), insufficient supply, too-long drop, padding-varied (Task 7). All spec sections map to a task.
- **Equivalence criterion:** enforced in Task 7 (`total_material` == golden AND `assertPlanValid`).
- **Type consistency:** `Solve`, `SolveMILP`, `Problem`, `Constraint`, `Solution`, `Plan`, `Pair`, `Piece`, `Board`, `Options`, `PieceCut/PiecePadding/PieceWaste` used identically across tasks.
- **Placeholder note:** Task 2 Step 3 intentionally ends with a typo'd `optimize` stub that Step 4 replaces — this is a compile-forcing scaffold, not a real placeholder, and is called out explicitly.
```
