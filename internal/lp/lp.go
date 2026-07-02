// Package lp implements a small dense two-phase tableau simplex for linear
// programs (minimize c·x, x >= 0) plus dual-price extraction. No external deps.
package lp

import (
	"math"
	"time"
)

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

	// Proven is true when optimality was proven (normal case). It is false
	// only when a backstop (deadline / node cap) cut the search short and the
	// returned solution is a best-effort incumbent.
	Proven bool
	// LowerBound is the best proven lower bound at termination. When Proven,
	// LowerBound == Objective; otherwise LowerBound < Objective and the gap is
	// (Objective - LowerBound) / Objective.
	LowerBound float64
}

const eps = 1e-9

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

// Solve minimizes p.Objective·x subject to the constraints with x >= 0.
// Requires RHS >= 0 for correct behavior (all cutting-stock RHS are non-negative).
func Solve(p Problem) Solution {
	tab, st := solveTableau(p)
	if st != Optimal {
		return Solution{Status: st}
	}
	return tab.solution(p.Objective)
}

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

// milpNodeLimit caps the number of branch-and-bound nodes explored so that
// pathological inputs (e.g. many requirement types) can never hang. Hitting
// the cap yields a best-effort (possibly suboptimal) incumbent, mirroring how
// the Python reference solver (CBC) is itself run under a time limit.
const milpNodeLimit = 5_000_000

// SolveMILP solves the MILP by LP-relaxation branch & bound (DFS with bounding).
// integer[j]==true forces x[j] to an integer. Minimization only.
//
// A non-zero deadline bounds wall-clock time: once passed, the search stops
// and returns the best incumbent found so far, which may be suboptimal on
// hard inputs. A zero deadline means no time limit (bounded only by the node
// cap milpNodeLimit).
func SolveMILP(p Problem, integer []bool, deadline time.Time) Solution {
	best := Solution{Status: Infeasible, Objective: math.Inf(1)}
	nodes := 0

	var rec func(extra []Constraint)
	rec = func(extra []Constraint) {
		if nodes >= milpNodeLimit || (!deadline.IsZero() && time.Now().After(deadline)) {
			return
		}
		nodes++
		cons := make([]Constraint, 0, len(p.Constraints)+len(extra))
		cons = append(cons, p.Constraints...)
		cons = append(cons, extra...)
		sol := Solve(Problem{Objective: p.Objective, Constraints: cons})
		if sol.Status != Optimal {
			return
		}
		// The cutting-stock objective (sum of integer stock lengths times
		// integer counts) is always integer-valued in this project, so the
		// best integer objective reachable from this node is at least
		// ceil(sol.Objective). If that cannot beat the incumbent, prune.
		// This bound is only valid because the objective is guaranteed
		// integer; SolveMILP must not be reused where that does not hold.
		if math.Ceil(sol.Objective-1e-9) >= best.Objective-eps {
			return // integer LP bound cannot beat incumbent
		}
		frac := -1
		bestDist := math.Inf(1)
		for j := range sol.X {
			if j < len(integer) && integer[j] {
				if d := sol.X[j] - math.Floor(sol.X[j]); d > 1e-6 && d < 1-1e-6 {
					// Most-fractional branching: pick the variable closest
					// to 0.5 to find a strong incumbent quickly.
					dist := math.Abs(d - 0.5)
					if dist < bestDist {
						bestDist, frac = dist, j
					}
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
		// Explore ceil first: for covering-style problems rounding up tends
		// to reach feasibility sooner, producing an early strong incumbent
		// that prunes the rest of the tree aggressively.
		rec(appendCons(extra, up))
		rec(appendCons(extra, down))
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
