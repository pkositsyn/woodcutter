# MILP branch-and-cut с best-bound — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заставить целочисленный solver доказывать оптимальность и завершаться сам (best-bound B&B + корневые Gomory-отсечения + primal-эвристика), устранив «зависание» на hard-входе.

**Architecture:** Внутри пакета `internal/lp` выделяем tableau из симплекса, генерируем Gomory-отсечения в структурном пространстве, гоняем корневой цикл отсечений, затем best-bound branch & bound через min-heap с primal-эвристикой. Пакет `internal/cutting` только читает новые флаги результата (`Proven`, `LowerBound`) и печатает предупреждение при срабатывании backstop. Warm-start и per-node отсечения — вне объёма.

**Tech Stack:** Go 1.25, только stdlib (`container/heap`, `math`, `time`). Без внешних LP-библиотек.

## Global Constraints

- Go 1.25; только стандартная библиотека, без внешних зависимостей (`container/heap` допустим).
- Пользовательские строки — на русском (как в существующем `report.go`).
- Раскладка `internal/lp` (generic MILP) и `internal/cutting` (домен). Вся работа с tableau/отсечениями — внутри `lp`; `cutting` читает только `lp.Solution`.
- Контракт: доказанный оптимум + backstop (дедлайн `Options.MILPTimeout` / `milpNodeLimit`). При срабатывании backstop — вернуть лучший инкумбент, пометить `Proven=false`.
- Отсекающие плоскости — только корневые (задел под per-node), только для чисто целочисленных задач (все структурные переменные integer).
- Существующие тесты (`lp_test.go`, `milp_test.go`, golden cross-check) должны продолжать проходить.
- Числовые допуски: `eps = 1e-9` (уже в пакете), целочисленность LP-значений — `1e-6`.

---

### Task 1: Выделить tableau из симплекса (без изменения поведения) + поля результата

Рефакторинг: `Solve` начинает строиться поверх внутреннего `solveTableau`, который сохраняет финальный tableau с метаданными колонок (нужно генератору отсечений в Task 2). Внешнее поведение `Solve` не меняется. Добавляем поля `Proven`/`LowerBound` в `Solution` (заполняются позже).

**Files:**
- Modify: `internal/lp/lp.go` (заменить тело `Solve`, добавить типы `colKind`/`tableau`, функции `solveTableau` и метод `tableau.solution`; поля в `Solution`)
- Test: `internal/lp/lp_test.go` (существующие тесты — регрессия, не меняем)

**Interfaces:**
- Produces:
  - `type colKind uint8` со значениями `colStructural, colSlack, colSurplus, colArtificial`
  - `type tableau struct { rows [][]float64; basis, logical []int; kind []colKind; conOf []int; cons []Constraint; n, total int }`
  - `func solveTableau(p Problem) (*tableau, Status)`
  - `func (t *tableau) solution(objective []float64) Solution`
  - Поля `Solution.Proven bool`, `Solution.LowerBound float64`

- [ ] **Step 1: Прогнать существующие lp-тесты (базовая линия — зелёные)**

Run: `go test ./internal/lp/ -run 'TestSolve' -v`
Expected: PASS (TestSolveMaxProfit, TestSolveMinDuals, TestSolveInfeasible, TestSolveDegenerate*)

- [ ] **Step 2: Добавить поля в `Solution`**

В `internal/lp/lp.go` в структуру `Solution` добавить два поля:

```go
type Solution struct {
	Status    Status
	Objective float64
	X         []float64
	Duals     []float64

	// Proven is true when optimality was proven (normal case). It is false
	// only when a backstop (deadline / node cap) cut the search short and the
	// returned solution is a best-effort incumbent.
	Proven bool
	// LowerBound is the best proven lower bound at termination. When Proven,
	// LowerBound == Objective; otherwise LowerBound < Objective and the gap is
	// (Objective - LowerBound) / Objective.
	LowerBound float64
}
```

- [ ] **Step 3: Добавить типы tableau и `solveTableau`; переписать `Solve` поверх него**

В `internal/lp/lp.go` добавить (перед `Solve`) типы и функцию, затем заменить тело `Solve`. `solveTableau` — это текущее тело `Solve` до извлечения решения, но с записью метаданных колонок (`kind`, `conOf`) и возвратом tableau:

```go
type colKind uint8

const (
	colStructural colKind = iota
	colSlack
	colSurplus
	colArtificial
)

// tableau is the final simplex tableau plus column metadata, retained so the
// cut generator can translate Gomory cuts back into structural-variable space.
type tableau struct {
	rows    [][]float64 // m rows, each len total+1 (last column is RHS)
	basis   []int       // basic column index per row
	logical []int       // initial-identity column per constraint (slack/artificial), for duals
	kind    []colKind   // len total
	conOf   []int       // constraint index for slack/surplus columns; -1 otherwise
	cons    []Constraint
	n       int // number of structural variables
	total   int // number of columns (structural + slack/surplus/artificial)
}

// solveTableau runs the two-phase simplex and returns the final tableau on
// success. Requires RHS >= 0 (all cutting-stock RHS are non-negative).
func solveTableau(p Problem) (*tableau, Status) {
	m := len(p.Constraints)
	n := len(p.Objective)

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
	artificial := make([]bool, total)
	kind := make([]colKind, total)
	conOf := make([]int, total)
	for j := range conOf {
		conOf[j] = -1
	}
	logical := make([]int, m)
	basis := make([]int, m)

	col := n
	for i, c := range p.Constraints {
		row := make([]float64, total+1)
		for j := 0; j < n; j++ {
			if j < len(c.Coeffs) {
				row[j] = c.Coeffs[j]
			}
		}
		row[total] = c.RHS
		switch c.Type {
		case LessEqual:
			row[col] = 1
			kind[col] = colSlack
			conOf[col] = i
			logical[i], basis[i] = col, col
			col++
		case GreaterEqual:
			row[col] = -1
			kind[col] = colSurplus
			conOf[col] = i
			col++
			row[col] = 1
			kind[col] = colArtificial
			artificial[col] = true
			logical[i], basis[i] = col, col
			col++
		case Equal:
			row[col] = 1
			kind[col] = colArtificial
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

	infeas := 0.0
	for r := 0; r < m; r++ {
		infeas += phase1[basis[r]] * rows[r][total]
	}
	if infeas > 1e-7 {
		return nil, Infeasible
	}

	// Drive out artificials basic at zero (see Solve's original comment).
	for r := 0; r < m; r++ {
		if !artificial[basis[r]] {
			continue
		}
		pc := -1
		var bestMag float64
		for j := 0; j < total; j++ {
			if artificial[j] {
				continue
			}
			if mag := math.Abs(rows[r][j]); mag > eps && mag > bestMag {
				bestMag, pc = mag, j
			}
		}
		if pc != -1 {
			pivot(rows, basis, r, pc, total)
		}
	}

	// Phase 2: minimize original cost; forbid artificials from re-entering.
	cost := make([]float64, total)
	copy(cost, p.Objective)
	for j := 0; j < total; j++ {
		if artificial[j] {
			forbid[j] = true
		}
	}
	if optimize(rows, basis, cost, forbid, total) == Unbounded {
		return nil, Unbounded
	}

	return &tableau{
		rows: rows, basis: basis, logical: logical,
		kind: kind, conOf: conOf, cons: p.Constraints,
		n: n, total: total,
	}, Optimal
}

// solution extracts primal values, objective, and dual prices from the tableau.
func (t *tableau) solution(objective []float64) Solution {
	m := len(t.rows)
	cost := make([]float64, t.total)
	copy(cost, objective)

	x := make([]float64, t.n)
	for r := 0; r < m; r++ {
		if t.basis[r] < t.n {
			x[t.basis[r]] = t.rows[r][t.total]
		}
	}
	obj := 0.0
	for r := 0; r < m; r++ {
		obj += cost[t.basis[r]] * t.rows[r][t.total]
	}
	// Duals: y = c_B · B^{-1}; B^{-1} column i sits under logical[i].
	duals := make([]float64, m)
	for i := 0; i < m; i++ {
		var y float64
		for r := 0; r < m; r++ {
			y += cost[t.basis[r]] * t.rows[r][t.logical[i]]
		}
		duals[i] = y
	}
	return Solution{Status: Optimal, Objective: obj, X: x, Duals: duals}
}
```

Заменить всё текущее тело `Solve` (строки от `m := len(p.Constraints)` до финального `return Solution{...}`) на:

```go
func Solve(p Problem) Solution {
	tab, st := solveTableau(p)
	if st != Optimal {
		return Solution{Status: st}
	}
	return tab.solution(p.Objective)
}
```

- [ ] **Step 4: Прогнать lp-тесты — поведение не изменилось**

Run: `go test ./internal/lp/ -run 'TestSolve' -v`
Expected: PASS (все 5 тестов `Solve` зелёные; duals/x/obj прежние)

- [ ] **Step 5: Прогнать весь пакет lp (SolveMILP пока старый, использует Solve)**

Run: `go build ./... && go test ./internal/lp/ -run 'TestSolve|TestMILP' -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add internal/lp/lp.go
git commit -m "$(cat <<'EOF'
refactor(lp): extract solveTableau; add Solution.Proven/LowerBound

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Генерация Gomory-отсечений в структурном пространстве

Новый файл `cuts.go`: для дробной базисной целочисленной переменной вывести дробное Gomory-отсечение, подставить slack/surplus обратно в структурные переменные, применить численные фильтры. Валидно только для чисто целочисленных задач.

**Files:**
- Create: `internal/lp/cuts.go`
- Test: `internal/lp/cuts_test.go`

**Interfaces:**
- Consumes (из Task 1): `*tableau`, `colKind`, `Constraint`
- Produces:
  - `func gomoryCuts(t *tableau, integer []bool) []Constraint`
  - `func allIntegerVars(integer []bool, n int) bool`
  - `func frac(v float64) float64` (со снапом near-integer к 0)

- [ ] **Step 1: Написать падающий тест валидности отсечения**

Создать `internal/lp/cuts_test.go`:

```go
package lp

import (
	"math"
	"testing"
)

// A Gomory cut must (a) separate the fractional LP optimum and (b) never cut
// off any integer-feasible point of the original problem.
// Deterministic 1-var case: min -x s.t. 2x <= 3, x integer. LP opt x=1.5.
// Expected cut is equivalent to x <= 1.
func TestGomoryCut1D(t *testing.T) {
	p := Problem{
		Objective:   []float64{-1},
		Constraints: []Constraint{{Coeffs: []float64{2}, Type: LessEqual, RHS: 3}},
	}
	tab, st := solveTableau(p)
	if st != Optimal {
		t.Fatalf("status = %v, want Optimal", st)
	}
	sol := tab.solution(p.Objective)
	cuts := gomoryCuts(tab, []bool{true})
	if len(cuts) == 0 {
		t.Fatalf("no cut generated for fractional LP opt x=%v", sol.X)
	}
	// (a) separates the LP optimum: some cut is violated at x*.
	separated := false
	for _, c := range cuts {
		if dot(c.Coeffs, sol.X) < c.RHS-1e-7 {
			separated = true
		}
	}
	if !separated {
		t.Fatalf("no cut separates LP opt x=%v; cuts=%v", sol.X, cuts)
	}
	// (b) valid: every integer point feasible for the original is feasible for every cut.
	for x := 0; x <= 3; x++ {
		if 2*x > 3 {
			continue // infeasible for original
		}
		for _, c := range cuts {
			if dot(c.Coeffs, []float64{float64(x)}) < c.RHS-1e-7 {
				t.Fatalf("cut %v wrongly cuts off integer x=%d", c, x)
			}
		}
	}
}

// Property test on a genuine 2-var fractional vertex:
// min -x - y s.t. 4x+5y <= 20, x <= 3, y <= 3, integers. LP opt x=3, y=1.6.
func TestGomoryCut2D(t *testing.T) {
	p := Problem{
		Objective: []float64{-1, -1},
		Constraints: []Constraint{
			{Coeffs: []float64{4, 5}, Type: LessEqual, RHS: 20},
			{Coeffs: []float64{1, 0}, Type: LessEqual, RHS: 3},
			{Coeffs: []float64{0, 1}, Type: LessEqual, RHS: 3},
		},
	}
	tab, st := solveTableau(p)
	if st != Optimal {
		t.Fatalf("status = %v", st)
	}
	sol := tab.solution(p.Objective)
	cuts := gomoryCuts(tab, []bool{true, true})
	if len(cuts) == 0 {
		t.Fatalf("no cut generated for fractional LP opt %v", sol.X)
	}
	separated := false
	for _, c := range cuts {
		if dot(c.Coeffs, sol.X) < c.RHS-1e-7 {
			separated = true
		}
	}
	if !separated {
		t.Fatalf("no cut separates LP opt %v", sol.X)
	}
	for x := 0; x <= 3; x++ {
		for y := 0; y <= 3; y++ {
			if 4*x+5*y > 20 {
				continue
			}
			for _, c := range cuts {
				if dot(c.Coeffs, []float64{float64(x), float64(y)}) < c.RHS-1e-7 {
					t.Fatalf("cut %v wrongly cuts off integer (%d,%d)", c, x, y)
				}
			}
		}
	}
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		if i < len(b) {
			s += a[i] * b[i]
		}
	}
	return s
}
```

- [ ] **Step 2: Прогнать тест — должен упасть (нет `gomoryCuts`)**

Run: `go test ./internal/lp/ -run TestGomoryCut -v`
Expected: FAIL (compile error: undefined `gomoryCuts` / `allIntegerVars`)

- [ ] **Step 3: Реализовать `cuts.go`**

Создать `internal/lp/cuts.go`:

```go
package lp

import "math"

const (
	cutFracTol  = 0.01   // ignore basic rows whose fractionality is ~0 or ~1
	cutCoeffEps = 1e-9   // drop cut coefficients below this magnitude
	cutMaxCoeff = 1e7    // drop ill-conditioned cuts (huge coefficients)
	fracSnap    = 1e-9   // snap near-integer tableau entries to integer
)

// frac returns the fractional part in [0,1), snapping near-integer values
// (from either side) to 0 so float noise cannot masquerade as a fraction.
func frac(v float64) float64 {
	f := v - math.Floor(v)
	if f < fracSnap || f > 1-fracSnap {
		return 0
	}
	return f
}

// allIntegerVars reports whether every structural variable is integer.
// Pure Gomory fractional cuts are only valid for pure integer programs.
func allIntegerVars(integer []bool, n int) bool {
	if len(integer) < n {
		return false
	}
	for j := 0; j < n; j++ {
		if !integer[j] {
			return false
		}
	}
	return true
}

// gomoryCuts derives Gomory fractional cuts from fractional integer basic
// variables and translates them into structural-variable space (substituting
// slack/surplus columns via their constraint definitions). Callers must ensure
// the program is pure-integer (allIntegerVars); otherwise the cuts are invalid.
func gomoryCuts(t *tableau, integer []bool) []Constraint {
	isBasic := make([]bool, t.total)
	for _, b := range t.basis {
		isBasic[b] = true
	}

	var cuts []Constraint
	for r := 0; r < len(t.rows); r++ {
		bv := t.basis[r]
		if bv >= t.n || bv >= len(integer) || !integer[bv] {
			continue
		}
		fb := frac(t.rows[r][t.total])
		if fb < cutFracTol || fb > 1-cutFracTol {
			continue
		}

		g := make([]float64, t.n)
		R := fb
		for j := 0; j < t.total; j++ {
			if isBasic[j] {
				continue // basic columns contribute 0 in exact arithmetic
			}
			fa := frac(t.rows[r][j])
			if fa == 0 {
				continue
			}
			switch t.kind[j] {
			case colStructural:
				g[j] += fa
			case colSlack: // s_k = RHS_k - coeffs_k·x
				c := t.cons[t.conOf[j]]
				for i := 0; i < t.n && i < len(c.Coeffs); i++ {
					g[i] -= fa * c.Coeffs[i]
				}
				R -= fa * c.RHS
			case colSurplus: // e_k = coeffs_k·x - RHS_k
				c := t.cons[t.conOf[j]]
				for i := 0; i < t.n && i < len(c.Coeffs); i++ {
					g[i] += fa * c.Coeffs[i]
				}
				R += fa * c.RHS
			case colArtificial:
				// nonbasic at zero, forbidden -> contributes nothing
			}
		}

		// Numerical hygiene: drop tiny coefficients and ill-conditioned cuts.
		maxC, nonzero := 0.0, false
		for i := range g {
			if math.Abs(g[i]) < cutCoeffEps {
				g[i] = 0
				continue
			}
			nonzero = true
			if a := math.Abs(g[i]); a > maxC {
				maxC = a
			}
		}
		if !nonzero || maxC > cutMaxCoeff {
			continue
		}
		cuts = append(cuts, Constraint{Coeffs: g, Type: GreaterEqual, RHS: R})
	}
	return cuts
}
```

- [ ] **Step 4: Прогнать тесты — зелёные**

Run: `go test ./internal/lp/ -run TestGomoryCut -v`
Expected: PASS (TestGomoryCut1D, TestGomoryCut2D)

- [ ] **Step 5: Commit**

```bash
git add internal/lp/cuts.go internal/lp/cuts_test.go
git commit -m "$(cat <<'EOF'
feat(lp): Gomory fractional cut generation in structural space

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Заменить DFS branch & bound на best-bound (min-heap), без отсечений

Переносим `SolveMILP` из `lp.go` в новый `milp.go` и переписываем: best-bound обход через `container/heap`, целочисленно-объектная граница `ceil(LP)`, backstop по дедлайну/узлам, установка `Proven`/`LowerBound`. Отсечения и эвристику добавит Task 4. `unit` переезжает сюда, `appendCons` удаляется.

**Files:**
- Modify: `internal/lp/lp.go` (удалить старый `SolveMILP`, `unit`, `appendCons`, `milpNodeLimit` и их комментарии)
- Create: `internal/lp/milp.go`
- Test: `internal/lp/milp_test.go` (добавить тесты best-bound/Proven; существующие остаются)

**Interfaces:**
- Consumes: `Solve`, `Problem`, `Constraint`, `Solution`, `eps`
- Produces:
  - `func SolveMILP(p Problem, integer []bool, deadline time.Time) Solution` (та же сигнатура)
  - `func mostFractional(x []float64, integer []bool) int`
  - `func unit(idx, n int) []float64`
  - `type bbNode`, `type nodeHeap` (внутренние)

- [ ] **Step 1: Удалить старый MILP-код из `lp.go`**

В `internal/lp/lp.go` удалить целиком: комментарий и константу `milpNodeLimit` (строки ~241-245), функцию `SolveMILP` (строки ~247-315), функции `unit` (~317-321) и `appendCons` (~323-328). Также удалить теперь неиспользуемый импорт `"time"` из `lp.go`, если он остался только ради `SolveMILP` (проверить: `time` больше нигде в `lp.go` не используется — удалить из import-блока).

- [ ] **Step 2: Написать падающие тесты best-bound в `milp_test.go`**

В `internal/lp/milp_test.go` добавить (не трогая существующие `TestMILPRounds`, `TestMILPBruteAgree`):

```go
// Best-bound must PROVE optimality (Proven=true) with no deadline.
func TestMILPProvenOptimal(t *testing.T) {
	p := Problem{
		Objective: []float64{5, 4},
		Constraints: []Constraint{
			{Coeffs: []float64{2, 1}, Type: GreaterEqual, RHS: 3},
			{Coeffs: []float64{1, 3}, Type: GreaterEqual, RHS: 4},
		},
	}
	s := SolveMILP(p, []bool{true, true}, time.Time{})
	if s.Status != Optimal || !s.Proven {
		t.Fatalf("status=%v proven=%v, want Optimal+proven", s.Status, s.Proven)
	}
	if !approx(s.LowerBound, s.Objective) {
		t.Fatalf("proven LowerBound=%v must equal Objective=%v", s.LowerBound, s.Objective)
	}
}

// Larger covering ILP vs brute force, proven.
func TestMILPBruteAgreeLarger(t *testing.T) {
	p := Problem{
		Objective: []float64{7, 5, 9},
		Constraints: []Constraint{
			{Coeffs: []float64{3, 2, 4}, Type: GreaterEqual, RHS: 9},
			{Coeffs: []float64{1, 4, 2}, Type: GreaterEqual, RHS: 8},
		},
	}
	s := SolveMILP(p, []bool{true, true, true}, time.Time{})
	best := math.Inf(1)
	for a := 0; a <= 5; a++ {
		for b := 0; b <= 5; b++ {
			for c := 0; c <= 5; c++ {
				if 3*a+2*b+4*c >= 9 && a+4*b+2*c >= 8 {
					if v := float64(7*a + 5*b + 9*c); v < best {
						best = v
					}
				}
			}
		}
	}
	if s.Status != Optimal || !approx(s.Objective, best) {
		t.Fatalf("milp obj %v (status %v), brute %v", s.Objective, s.Status, best)
	}
}

// A past deadline must terminate promptly (no hang), not spin.
func TestMILPBackstopTerminates(t *testing.T) {
	p := Problem{
		Objective:   []float64{1, 1},
		Constraints: []Constraint{{Coeffs: []float64{1, 1}, Type: GreaterEqual, RHS: 3.5}},
	}
	done := make(chan Solution, 1)
	go func() { done <- SolveMILP(p, []bool{true, true}, time.Now().Add(-time.Second)) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("SolveMILP did not terminate under a past deadline")
	}
}
```

- [ ] **Step 3: Прогнать — падает (нет `SolveMILP` после удаления / нет новых символов)**

Run: `go test ./internal/lp/ -run TestMILP -v`
Expected: FAIL (compile error: undefined `SolveMILP`)

- [ ] **Step 4: Реализовать `milp.go` (best-bound, без отсечений)**

Создать `internal/lp/milp.go`:

```go
package lp

import (
	"container/heap"
	"math"
	"time"
)

// milpNodeLimit caps branch-and-bound nodes so pathological inputs cannot hang.
// Hitting it yields a best-effort incumbent (Proven=false).
const milpNodeLimit = 5_000_000

// bbNode is a branch & bound node: branch-bound constraints layered on the root
// problem, plus a lower bound (its parent's LP objective) used as the heap key.
type bbNode struct {
	extra []Constraint
	bound float64
}

func (n bbNode) child(c Constraint, bound float64) bbNode {
	ex := make([]Constraint, len(n.extra)+1)
	copy(ex, n.extra)
	ex[len(n.extra)] = c
	return bbNode{extra: ex, bound: bound}
}

// nodeHeap is a min-heap of nodes keyed by lower bound (best-bound search).
type nodeHeap []bbNode

func (h nodeHeap) Len() int           { return len(h) }
func (h nodeHeap) Less(i, j int) bool { return h[i].bound < h[j].bound }
func (h nodeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *nodeHeap) Push(x any)        { *h = append(*h, x.(bbNode)) }
func (h *nodeHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// SolveMILP solves the MILP by best-bound LP-relaxation branch & bound.
// integer[j]==true forces x[j] integer. Minimization only; the objective is
// assumed integer-valued (used for the ceil bound). A non-zero deadline (or the
// node cap) bounds wall-clock time: on backstop the best incumbent so far is
// returned with Proven=false. Optimality is proven when the smallest remaining
// node bound cannot beat the incumbent.
func SolveMILP(p Problem, integer []bool, deadline time.Time) Solution {
	rp := Problem{Objective: p.Objective, Constraints: append([]Constraint(nil), p.Constraints...)}

	incumbent := Solution{Status: Infeasible, Objective: math.Inf(1)}
	nodes := 0
	proven := true
	lowerBound := math.Inf(-1)

	h := &nodeHeap{{extra: nil, bound: math.Inf(-1)}}
	for h.Len() > 0 {
		if nodes >= milpNodeLimit || (!deadline.IsZero() && time.Now().After(deadline)) {
			proven = false
			lowerBound = (*h)[0].bound // smallest remaining bound (min-heap root)
			break
		}
		node := heap.Pop(h).(bbNode)
		if node.bound >= incumbent.Objective-eps {
			// Best-bound: every remaining node is >= this -> incumbent optimal.
			break
		}
		nodes++

		cons := make([]Constraint, 0, len(rp.Constraints)+len(node.extra))
		cons = append(cons, rp.Constraints...)
		cons = append(cons, node.extra...)
		sol := Solve(Problem{Objective: rp.Objective, Constraints: cons})
		if sol.Status != Optimal {
			continue
		}
		lb := math.Ceil(sol.Objective - 1e-9) // integer-objective lower bound
		if lb >= incumbent.Objective-eps {
			continue // cannot beat incumbent
		}
		frac := mostFractional(sol.X, integer)
		if frac == -1 {
			sol.Status = Optimal
			incumbent = sol
			continue
		}
		v := sol.X[frac]
		down := node.child(Constraint{Coeffs: unit(frac, len(rp.Objective)), Type: LessEqual, RHS: math.Floor(v)}, lb)
		up := node.child(Constraint{Coeffs: unit(frac, len(rp.Objective)), Type: GreaterEqual, RHS: math.Ceil(v)}, lb)
		heap.Push(h, up)
		heap.Push(h, down)
	}

	if math.IsInf(incumbent.Objective, 1) {
		return Solution{Status: Infeasible}
	}
	incumbent.Proven = proven
	if proven {
		incumbent.LowerBound = incumbent.Objective
	} else if math.IsInf(lowerBound, -1) {
		incumbent.LowerBound = incumbent.Objective // no better info
	} else {
		incumbent.LowerBound = lowerBound
	}
	return incumbent
}

// mostFractional returns the integer-constrained variable closest to 0.5, or -1
// if the point is integer-feasible.
func mostFractional(x []float64, integer []bool) int {
	frac := -1
	bestDist := math.Inf(1)
	for j := range x {
		if j < len(integer) && integer[j] {
			d := x[j] - math.Floor(x[j])
			if d > 1e-6 && d < 1-1e-6 {
				if dist := math.Abs(d - 0.5); dist < bestDist {
					bestDist, frac = dist, j
				}
			}
		}
	}
	return frac
}

func unit(idx, n int) []float64 {
	v := make([]float64, n)
	v[idx] = 1
	return v
}
```

- [ ] **Step 5: Прогнать все lp-тесты — зелёные**

Run: `go build ./... && go test ./internal/lp/ -v`
Expected: PASS (TestSolve*, TestGomoryCut*, TestMILPRounds, TestMILPBruteAgree, TestMILPProvenOptimal, TestMILPBruteAgreeLarger, TestMILPBackstopTerminates)

- [ ] **Step 6: Commit**

```bash
git add internal/lp/lp.go internal/lp/milp.go internal/lp/milp_test.go
git commit -m "$(cat <<'EOF'
feat(lp): best-bound branch & bound with proven/backstop semantics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Корневой цикл отсечений + primal-эвристика в `SolveMILP`

Встроить в `SolveMILP` (перед B&B): корневой цикл Gomory-отсечений (усиливает границу) и rounding-эвристику (сеет инкумбент). Только для чисто целочисленных задач.

**Files:**
- Modify: `internal/lp/milp.go`
- Test: `internal/lp/milp_test.go` (добавить тест эвристики/Proven=false с инкумбентом)

**Interfaces:**
- Consumes (Task 2): `gomoryCuts`, `allIntegerVars`; (Task 1): `solveTableau`, `tableau.solution`
- Produces:
  - `func roundUpHeuristic(cons, obj, x []float64-подобные, integer []bool) (Solution, bool)` →
    точная сигнатура `func roundUpHeuristic(cons []Constraint, obj, x []float64, integer []bool) (Solution, bool)`
  - `func feasiblePoint(cons []Constraint, x []float64) bool`
  - Константы `maxCutRounds`, `cutStallEps`

- [ ] **Step 1: Написать падающий тест эвристики + Proven=false с инкумбентом**

В `internal/lp/milp_test.go` добавить:

```go
// roundUpHeuristic rounds integer vars up and accepts only feasible points.
func TestRoundUpHeuristic(t *testing.T) {
	cons := []Constraint{
		{Coeffs: []float64{1, 1}, Type: GreaterEqual, RHS: 3},
		{Coeffs: []float64{1, 0}, Type: LessEqual, RHS: 10},
	}
	obj := []float64{1, 1}
	// x=(1.4, 1.6) rounds up to (2,2): satisfies both -> feasible, obj 4.
	s, ok := roundUpHeuristic(cons, obj, []float64{1.4, 1.6}, []bool{true, true})
	if !ok || !approx(s.Objective, 4) {
		t.Fatalf("heuristic ok=%v obj=%v, want ok+obj 4", ok, s.Objective)
	}
	// Rounding up violates a tight <= cap -> rejected.
	tight := []Constraint{{Coeffs: []float64{1, 0}, Type: LessEqual, RHS: 1}}
	if _, ok := roundUpHeuristic(tight, obj, []float64{0.9, 0}, []bool{true, true}); ok {
		t.Fatalf("heuristic must reject infeasible round-up (x0 -> 1 > cap 1? equals ok); use 1.9")
	}
}

// Under a past deadline but with a heuristic incumbent available, SolveMILP
// returns a feasible incumbent flagged Proven=false with LowerBound < Objective.
func TestMILPBackstopGap(t *testing.T) {
	// Covering LP with fractional root so the root heuristic seeds an incumbent
	// before the (already-passed) deadline stops branching.
	p := Problem{
		Objective: []float64{1, 1},
		Constraints: []Constraint{
			{Coeffs: []float64{2, 3}, Type: GreaterEqual, RHS: 7},
			{Coeffs: []float64{3, 2}, Type: GreaterEqual, RHS: 7},
		},
	}
	s := SolveMILP(p, []bool{true, true}, time.Now().Add(-time.Second))
	if s.Status != Optimal {
		t.Fatalf("status=%v, want Optimal best-effort incumbent", s.Status)
	}
	if s.Proven {
		t.Fatalf("Proven must be false under a past deadline")
	}
	if s.LowerBound > s.Objective+1e-9 {
		t.Fatalf("LowerBound %v must be <= Objective %v", s.LowerBound, s.Objective)
	}
}
```

Note: во втором под-кейсе `TestRoundUpHeuristic` использовать `x0=1.9` (округление вверх → 2 > cap 1 → отклонение). Исправить входной вектор на `[]float64{1.9, 0}` при написании.

- [ ] **Step 2: Прогнать — падает (нет `roundUpHeuristic`)**

Run: `go test ./internal/lp/ -run 'TestRoundUpHeuristic|TestMILPBackstopGap' -v`
Expected: FAIL (compile error: undefined `roundUpHeuristic`)

- [ ] **Step 3: Добавить эвристику/фильтр и корневой цикл в `milp.go`**

В `internal/lp/milp.go` добавить константы и функции:

```go
const (
	maxCutRounds = 15   // cap root cutting-plane rounds
	cutStallEps  = 1e-6 // stop cutting when the LP bound stops improving
)

// feasiblePoint reports whether x satisfies every constraint and x >= 0.
func feasiblePoint(cons []Constraint, x []float64) bool {
	for _, c := range cons {
		lhs := 0.0
		for j := range c.Coeffs {
			if j < len(x) {
				lhs += c.Coeffs[j] * x[j]
			}
		}
		switch c.Type {
		case LessEqual:
			if lhs > c.RHS+1e-6 {
				return false
			}
		case GreaterEqual:
			if lhs < c.RHS-1e-6 {
				return false
			}
		case Equal:
			if math.Abs(lhs-c.RHS) > 1e-6 {
				return false
			}
		}
	}
	for _, v := range x {
		if v < -1e-6 {
			return false
		}
	}
	return true
}

// roundUpHeuristic rounds integer variables of a fractional LP point up and
// returns the resulting solution if it is feasible for cons. Rounding up keeps
// covering (>=) demand satisfied; it may violate supply (<=) caps, in which
// case the point is rejected (no repair).
func roundUpHeuristic(cons []Constraint, obj, x []float64, integer []bool) (Solution, bool) {
	sol := make([]float64, len(x))
	for j := range x {
		if j < len(integer) && integer[j] {
			sol[j] = math.Ceil(x[j] - 1e-9)
		} else {
			sol[j] = x[j]
		}
	}
	if !feasiblePoint(cons, sol) {
		return Solution{}, false
	}
	o := 0.0
	for j := range obj {
		if j < len(sol) {
			o += obj[j] * sol[j]
		}
	}
	return Solution{Status: Optimal, Objective: o, X: sol}, true
}
```

Затем в `SolveMILP` заменить инициализацию `rp`/`incumbent` и вставить корневой цикл **перед** циклом B&B. Заменить блок:

```go
	rp := Problem{Objective: p.Objective, Constraints: append([]Constraint(nil), p.Constraints...)}

	incumbent := Solution{Status: Infeasible, Objective: math.Inf(1)}
	nodes := 0
```

на:

```go
	rp := Problem{Objective: p.Objective, Constraints: append([]Constraint(nil), p.Constraints...)}
	incumbent := Solution{Status: Infeasible, Objective: math.Inf(1)}

	// Root cutting-plane loop (pure integer programs only): tighten the LP
	// bound with Gomory cuts and seed an incumbent via a rounding heuristic.
	if allIntegerVars(integer, len(p.Objective)) {
		prev := math.Inf(-1)
		for round := 0; round < maxCutRounds; round++ {
			tab, st := solveTableau(rp)
			if st != Optimal {
				return Solution{Status: st}
			}
			sol := tab.solution(rp.Objective)
			if mostFractional(sol.X, integer) == -1 {
				sol.Status = Optimal
				sol.Proven = true
				sol.LowerBound = sol.Objective
				return sol // integer LP optimum is the global optimum
			}
			if h, ok := roundUpHeuristic(p.Constraints, p.Objective, sol.X, integer); ok && h.Objective < incumbent.Objective {
				incumbent = h
			}
			if sol.Objective <= prev+cutStallEps {
				break // bound stalled; stop cutting
			}
			prev = sol.Objective
			cuts := gomoryCuts(tab, integer)
			if len(cuts) == 0 {
				break
			}
			rp.Constraints = append(rp.Constraints, cuts...)
		}
	}

	nodes := 0
```

(Остальная часть `SolveMILP` — цикл B&B — без изменений; он уже использует `rp` и `incumbent`.)

Also: при обновлении инкумбента внутри B&B на целочисленном узле — если корневая эвристика уже дала инкумбент, узловой апдейт `incumbent = sol` должен срабатывать только если строго лучше. Это уже гарантировано прунингом `lb >= incumbent.Objective-eps` (узел с не-лучшей границей отсекается до ветвления), поэтому любой достигнутый целочисленный `sol` строго лучше инкумбента. Изменений не требуется.

- [ ] **Step 4: Прогнать новые и все lp-тесты**

Run: `go test ./internal/lp/ -v`
Expected: PASS (включая TestRoundUpHeuristic, TestMILPBackstopGap; все прежние зелёные)

- [ ] **Step 5: Commit**

```bash
git add internal/lp/milp.go internal/lp/milp_test.go
git commit -m "$(cat <<'EOF'
feat(lp): root Gomory cut loop + rounding heuristic in SolveMILP

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Интеграция в cutting + отчёт + усиление приёмочного теста

Прокинуть `Proven`/`LowerBound` в `Plan`, печатать предупреждение при backstop, и усилить `TestSolveHardInputTerminates` до приёмочного критерия (доказанный оптимум 171600, быстро).

**Files:**
- Modify: `internal/cutting/types.go` (поля `Plan.Proven`, `Plan.LowerBound`)
- Modify: `internal/cutting/solve.go` (прокинуть флаги; `Proven:true` в тривиальных ветках)
- Modify: `internal/cutting/report.go` (строка предупреждения)
- Modify: `internal/cutting/terminate_test.go` (усилить)
- Test: `internal/cutting/report_test.go` (тест предупреждения)

**Interfaces:**
- Consumes (Task 3/4): `lp.Solution.Proven`, `lp.Solution.LowerBound`
- Produces: `Plan.Proven bool`, `Plan.LowerBound int`

- [ ] **Step 1: Добавить поля в `Plan`**

В `internal/cutting/types.go` в структуру `Plan` добавить:

```go
// Plan is the full cutting result.
type Plan struct {
	TotalMaterial int // sum of used board lengths; -1 when infeasible
	Feasible      bool
	Boards        []Board
	Dropped       []Pair // requirements longer than the longest stock board

	// Proven is true when the integer optimum was proven. It is false only when
	// a backstop (time/node limit) returned a best-effort incumbent.
	Proven bool
	// LowerBound is a proven lower bound on TotalMaterial. When Proven, it
	// equals TotalMaterial; otherwise the optimality gap is
	// (TotalMaterial - LowerBound) / TotalMaterial.
	LowerBound int
}
```

- [ ] **Step 2: Прокинуть флаги в `solve.go`**

В `internal/cutting/solve.go`:

Тривиальные допустимые ветки помечать доказанными. Заменить `return Plan{Feasible: true, TotalMaterial: 0}` на:

```go
		return Plan{Feasible: true, TotalMaterial: 0, Proven: true}
```

и `return Plan{Feasible: true, TotalMaterial: 0, Dropped: dropped}` на:

```go
		return Plan{Feasible: true, TotalMaterial: 0, Dropped: dropped, Proven: true}
```

В финале функции заменить последний `return`:

```go
	return Plan{Feasible: true, TotalMaterial: total, Boards: boards, Dropped: dropped}
```

на:

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

(`math` уже импортирован в `solve.go`.)

- [ ] **Step 3: Печать предупреждения в `report.go`**

В `internal/cutting/report.go`, в `WriteReport`, сразу после цикла печати `plan.Dropped` и **перед** строкой `Детальный план раскроя`, вставить:

```go
	if !plan.Proven {
		gap := 0.0
		if plan.TotalMaterial > 0 {
			gap = float64(plan.TotalMaterial-plan.LowerBound) / float64(plan.TotalMaterial) * 100
		}
		fmt.Fprintf(w, "⚠ Оптимальность не доказана (лимит времени/узлов); верхняя оценка отклонения от оптимума: %.2f%%\n", gap)
	}
```

- [ ] **Step 4: Написать тест предупреждения в `report_test.go`**

В `internal/cutting/report_test.go` добавить:

```go
func TestWriteReportUnproven(t *testing.T) {
	plan := Plan{
		Feasible:      true,
		TotalMaterial: 10000,
		LowerBound:    9800,
		Proven:        false,
		Boards: []Board{{
			StockLength: 10000,
			Pieces:      []Piece{{Length: 10000, Start: 0, End: 10000, Kind: PieceCut}},
		}},
	}
	var sb strings.Builder
	WriteReport(&sb, plan, []Pair{{10000, 1}}, Options{Padding: 5})
	out := sb.String()
	if !strings.Contains(out, "не доказана") {
		t.Fatalf("expected unproven warning, got:\n%s", out)
	}
}

func TestWriteReportProvenNoWarning(t *testing.T) {
	plan := Plan{
		Feasible:      true,
		TotalMaterial: 10000,
		LowerBound:    10000,
		Proven:        true,
		Boards: []Board{{
			StockLength: 10000,
			Pieces:      []Piece{{Length: 10000, Start: 0, End: 10000, Kind: PieceCut}},
		}},
	}
	var sb strings.Builder
	WriteReport(&sb, plan, []Pair{{10000, 1}}, Options{Padding: 5})
	if strings.Contains(sb.String(), "не доказана") {
		t.Fatalf("proven plan must not print warning:\n%s", sb.String())
	}
}
```

- [ ] **Step 5: Усилить приёмочный тест `terminate_test.go`**

Заменить тело `TestSolveHardInputTerminates` в `internal/cutting/terminate_test.go`. Теперь ожидаем доказанный оптимум, равный Python-референсу (171600), и быстрое завершение (backstop не должен срабатывать):

```go
func TestSolveHardInputTerminates(t *testing.T) {
	stock := []Pair{{6100, 30}, {5900, 30}, {5300, 30}, {4700, 30}, {4100, 30}}
	reqs := []Pair{
		{2333, 13}, {1777, 17}, {1301, 19}, {997, 23},
		{733, 29}, {511, 31}, {389, 37}, {271, 41},
	}
	// MILPTimeout is only a backstop; the solver must prove optimality well
	// before it, so elapsed stays far under the budget.
	opts := Options{Padding: 4, MILPTimeout: 10 * time.Second}

	start := time.Now()
	plan := Solve(stock, reqs, opts)
	elapsed := time.Since(start)

	t.Logf("elapsed=%v totalMaterial=%d proven=%v lowerBound=%d",
		elapsed, plan.TotalMaterial, plan.Proven, plan.LowerBound)

	if elapsed >= 5*time.Second {
		t.Fatalf("solve took %v, want < 5s (should prove optimum, not hit backstop)", elapsed)
	}
	if !plan.Feasible {
		t.Fatalf("expected feasible plan, got Feasible=false")
	}
	if !plan.Proven {
		t.Fatalf("expected proven optimum, got Proven=false (backstop fired)")
	}
	if plan.TotalMaterial != 171600 {
		t.Fatalf("TotalMaterial = %d, want 171600 (Python reference optimum)", plan.TotalMaterial)
	}
	assertPlanValid(t, plan, stock, reqs, 4)
}
```

Обновить doc-комментарий над тестом: теперь контракт — доказанный оптимум, а не только термингация (убрать NOTE про «optimality is not guaranteed»).

- [ ] **Step 6: Прогнать весь тестовый набор проекта**

Run: `go build ./... && go test ./... -v`
Expected: PASS — в частности:
- `TestSolveHardInputTerminates` (proven, 171600, < 5s)
- `TestGoldenCrossCheck/*` (все golden — те же оптимумы, `assertPlanValid` ок)
- `TestWriteReportUnproven`, `TestWriteReportProvenNoWarning`
- все `internal/lp` тесты

Если `TestSolveHardInputTerminates` не проходит по времени/доказательству — см. раздел «Риски» спеки: сначала поднять `maxCutRounds`, затем рассмотреть per-node отсечения. Это тюнинг на этапе исполнения, не переписывание.

- [ ] **Step 7: Прогнать race-детектор на солвере (heap + отсутствие гонок)**

Run: `go test ./internal/lp/ ./internal/cutting/ -race`
Expected: PASS без предупреждений гонок.

- [ ] **Step 8: Обновить README (раздел про доказанную оптимальность)**

В `README.md`, в разделе про алгоритм (пункт 3 «Целочисленная задача»), уточнить, что целочисленный solve теперь доказывает оптимальность (branch-and-cut с best-bound), а при срабатывании лимита времени печатает предупреждение с оценкой отклонения. Добавить 1-2 предложения; не переписывать раздел.

Пример вставки после списка шагов:

```markdown
Целочисленный этап использует branch-and-cut (best-bound + корневые
Gomory-отсечения) и **доказывает** оптимальность. На патологических входах, если
срабатывает лимит времени, возвращается лучший найденный план с пометкой о
недоказанной оптимальности и верхней оценкой отклонения.
```

- [ ] **Step 9: Commit**

```bash
git add internal/cutting/types.go internal/cutting/solve.go internal/cutting/report.go \
        internal/cutting/report_test.go internal/cutting/terminate_test.go README.md
git commit -m "$(cat <<'EOF'
feat(cutting): surface proven/gap; prove optimum on hard input

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Контракт (доказ. оптимум + backstop) → Task 3 (`Proven`/`LowerBound`, backstop), Task 5 (проброс/отчёт). ✔
- Экспонирование tableau (Компонент 1 спеки) → Task 1. ✔
- Gomory-отсечения + численная защита (Компонент 2) → Task 2. ✔
- Корневой цикл отсечений (Компонент 3) → Task 4. ✔
- Best-bound B&B (Компонент 4) → Task 3. ✔
- Primal-эвристика (Компонент 5) → Task 4. ✔
- Сигнал результата (Компонент 6) → Task 1 (поля), Task 3/4 (заполнение). ✔
- Интеграция cutting + отчёт (Компонент 7) → Task 5. ✔
- Тестирование (усиленный terminate, golden, unit lp) → Task 2/3/4/5. ✔
- Задел под per-node отсечения → `gomoryCuts(t *tableau, ...)` работает от любого узлового tableau (структурная подстановка обобщается на добавленные ограничения). ✔

**Placeholder scan:** нет TBD/TODO; весь код приведён. Один явно помеченный нюанс — во втором под-кейсе `TestRoundUpHeuristic` входной вектор `[]float64{1.9, 0}` (указано в Step 1 Task 4). ✔

**Type consistency:**
- `tableau` поля (`rows/basis/logical/kind/conOf/cons/n/total`) — одинаковы в Task 1 (объявление) и Task 2 (использование `t.rows/t.basis/t.kind/t.conOf/t.cons/t.n/t.total`). ✔
- `gomoryCuts(t *tableau, integer []bool) []Constraint`, `allIntegerVars(integer []bool, n int) bool` — сигнатуры совпадают в Task 2 и вызовах Task 4. ✔
- `SolveMILP(p Problem, integer []bool, deadline time.Time) Solution` — сигнатура неизменна (Task 3), вызов в `solve.go` не меняется. ✔
- `Solution.Proven bool` / `Solution.LowerBound float64` (Task 1) → читаются в Task 5 (`sol.Proven`, `sol.LowerBound`). ✔
- `Plan.Proven bool` / `Plan.LowerBound int` (Task 5) → читаются в `report.go` (Task 5). ✔
- `roundUpHeuristic(cons []Constraint, obj, x []float64, integer []bool)` / `feasiblePoint(cons []Constraint, x []float64)` — объявление и вызовы в Task 4 согласованы. ✔
- `mostFractional` / `unit` — объявлены в Task 3 (milp.go), используются в Task 3 и Task 4. ✔
