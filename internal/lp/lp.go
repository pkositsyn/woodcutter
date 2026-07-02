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
// Requires RHS >= 0 for correct behavior (all cutting-stock RHS are non-negative).
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
		for j := 0; j < n; j++ {
			if j < len(c.Coeffs) {
				row[j] = c.Coeffs[j]
			}
		}
		row[total] = c.RHS
		switch c.Type {
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
