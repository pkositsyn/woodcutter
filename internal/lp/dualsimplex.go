package lp

import "math"

// bvSolver is a bounded-variable dual-simplex LP engine. It minimizes c·x for
// the same problem class as Solve (structural x >= 0, non-negative costs), and
// is differential-tested against Solve. This task delivers only the
// from-scratch solve; warm-start support arrives later.
//
// Model: structural variables 0..n-1 with bounds [0, +Inf). One logical
// variable per constraint i in column n+i with type-dependent bounds
// (LessEqual -> [0,+Inf), GreaterEqual -> (-Inf,0], Equal -> [0,0]). Each row i
// is A_i·x + s_i = b_i, so the logical columns form an identity. The initial
// basis is the logicals: dual-feasible (reduced costs equal c_j >= 0 with
// nonbasic structurals at their lower bound 0) but primal-infeasible wherever
// b_i violates the logical's bounds.
type bvSolver struct {
	n     int // structural variable count
	m     int // constraint count
	total int // n + m columns

	cost    []float64   // per-column cost; logicals are 0
	lo, hi  []float64   // per-column bounds (may be +/-Inf)
	tab     [][]float64 // m x total tableau = B^{-1}A
	binvb   []float64   // B^{-1} b (RHS transformed like the tableau)
	xB      []float64   // basic value per row (derived from binvb and nonbasics)
	basis   []int       // basic column index per row
	inBasis []bool      // whether a column is basic
	atUpper []bool      // nonbasic status: at upper bound (true) or lower (false)
}

// newBVSolver builds the initial dual-feasible basis for p.
func newBVSolver(p Problem) *bvSolver {
	n := len(p.Objective)
	m := len(p.Constraints)
	total := n + m

	s := &bvSolver{
		n: n, m: m, total: total,
		cost:    make([]float64, total),
		lo:      make([]float64, total),
		hi:      make([]float64, total),
		tab:     make([][]float64, m),
		binvb:   make([]float64, m),
		xB:      make([]float64, m),
		basis:   make([]int, m),
		inBasis: make([]bool, total),
		atUpper: make([]bool, total),
	}

	// Structural columns: cost c_j, bounds [0, +Inf).
	for j := 0; j < n; j++ {
		s.cost[j] = p.Objective[j]
		s.lo[j] = 0
		s.hi[j] = math.Inf(1)
	}

	// Logical columns: cost 0, type-dependent bounds.
	for i, c := range p.Constraints {
		col := n + i
		s.cost[col] = 0
		switch c.Type {
		case LessEqual:
			s.lo[col] = 0
			s.hi[col] = math.Inf(1)
		case GreaterEqual:
			s.lo[col] = math.Inf(-1)
			s.hi[col] = 0
		case Equal:
			s.lo[col] = 0
			s.hi[col] = 0
		}
	}

	// Tableau rows = A with the logical identity appended; binvb = b (B = I).
	for i, c := range p.Constraints {
		row := make([]float64, total)
		for j := 0; j < n; j++ {
			if j < len(c.Coeffs) {
				row[j] = c.Coeffs[j]
			}
		}
		row[n+i] = 1 // logical coefficient in its own row
		s.tab[i] = row
		s.binvb[i] = c.RHS

		// Initial basis: logical n+i, nonbasic structurals at lower bound 0.
		s.basis[i] = n + i
		s.inBasis[n+i] = true
	}

	// Nonbasic structurals start at their lower bound (0), atUpper=false (zero
	// value). Compute the initial basic values.
	s.recomputeBasics()
	return s
}

// nonbasicValue returns the current value of nonbasic column j.
func (s *bvSolver) nonbasicValue(j int) float64 {
	if s.atUpper[j] {
		return s.hi[j]
	}
	return s.lo[j]
}

// recomputeBasics sets xB[r] = binvb[r] - sum_{nonbasic j} T[r][j]*value(j).
func (s *bvSolver) recomputeBasics() {
	for r := 0; r < s.m; r++ {
		v := s.binvb[r]
		row := s.tab[r]
		for j := 0; j < s.total; j++ {
			if s.inBasis[j] {
				continue
			}
			val := s.nonbasicValue(j)
			if val != 0 {
				v -= row[j] * val
			}
		}
		s.xB[r] = v
	}
}

// reducedCost computes d_j = c_j - c_B · T[:,j] for a nonbasic column j.
func (s *bvSolver) reducedCost(j int) float64 {
	d := s.cost[j]
	for r := 0; r < s.m; r++ {
		cb := s.cost[s.basis[r]]
		if cb != 0 {
			d -= cb * s.tab[r][j]
		}
	}
	return d
}

// solve runs the dual simplex to optimality, infeasibility, or unboundedness.
func (s *bvSolver) solve() Status {
	const tol = 1e-9
	// Iteration guard: dual simplex with Bland tie-breaking terminates, but cap
	// defensively against any pathological loop.
	maxIter := 1000 * (s.total + s.m + 1)

	for iter := 0; iter < maxIter; iter++ {
		// Choose leaving row: basic var most violating its bound.
		leave := -1
		var worst float64
		belowLo := false // true if xB below lo (must increase), else above hi
		for r := 0; r < s.m; r++ {
			b := s.basis[r]
			v := s.xB[r]
			if v < s.lo[b]-tol {
				if viol := s.lo[b] - v; leave == -1 || viol > worst {
					worst, leave, belowLo = viol, r, true
				}
			} else if v > s.hi[b]+tol {
				if viol := v - s.hi[b]; leave == -1 || viol > worst {
					worst, leave, belowLo = viol, r, false
				}
			}
		}
		if leave == -1 {
			return Optimal // primal-feasible => optimal (dual-feasible throughout)
		}

		// Dual ratio test. The leaving basic variable x_B[leave] must move
		// toward the violated bound: increase if below lower (dir=+1), decrease
		// if above upper (dir=-1). Along the pivot on nonbasic column j,
		//   delta(x_B[leave]) = -T[leave][j] * delta(x_j),
		// and a nonbasic at its lower bound may only increase (delta x_j >= 0),
		// at its upper bound may only decrease (delta x_j <= 0). Column j is
		// eligible when its permitted move produces delta(x_B[leave]) of sign
		// dir. Among eligible columns pick the minimum |d_j / T[leave][j]|;
		// Bland tie-break on smallest column index.
		dir := 1.0
		if !belowLo {
			dir = -1.0
		}

		enter := -1
		var bestRatio float64
		for j := 0; j < s.total; j++ {
			if s.inBasis[j] {
				continue
			}
			a := s.tab[leave][j]
			if math.Abs(a) <= tol {
				continue
			}
			// delta x_j = (-dir / a) * t, t>0 => x_j must move up iff -dir/a>0.
			needUp := (-dir * a) > 0
			if needUp {
				if s.atUpper[j] { // must increase but sits at upper bound
					continue
				}
			} else {
				if !s.atUpper[j] { // must decrease but sits at lower bound
					continue
				}
			}
			ratio := math.Abs(s.reducedCost(j) / a)
			if enter == -1 || ratio < bestRatio-tol ||
				(ratio <= bestRatio+tol && j < enter) {
				bestRatio, enter = ratio, j
			}
		}

		if enter == -1 {
			return Infeasible // no eligible entering column
		}

		// Pivot: `enter` becomes basic in row `leave`; the leaving variable
		// becomes nonbasic at the bound it hit.
		leavingCol := s.basis[leave]
		leaveAtUpper := !belowLo // hit its upper bound iff it was above it

		s.pivotBV(leave, enter)

		s.inBasis[leavingCol] = false
		s.inBasis[enter] = true
		s.atUpper[leavingCol] = leaveAtUpper
		s.basis[leave] = enter

		s.recomputeBasics()
	}
	return Unbounded // unreachable for this problem class
}

// pivotBV row-reduces the tableau (and binvb) so column pc becomes a unit
// vector on row pr.
func (s *bvSolver) pivotBV(pr, pc int) {
	piv := s.tab[pr][pc]

	// Snapshot the entering column before mutating the tableau.
	col := make([]float64, s.m)
	for r := 0; r < s.m; r++ {
		col[r] = s.tab[r][pc]
	}

	prow := s.tab[pr]
	for j := 0; j < s.total; j++ {
		prow[j] /= piv
	}
	s.binvb[pr] /= piv

	for r := 0; r < s.m; r++ {
		if r == pr {
			continue
		}
		f := col[r]
		if f == 0 {
			continue
		}
		row := s.tab[r]
		for j := 0; j < s.total; j++ {
			row[j] -= f * prow[j]
		}
		s.binvb[r] -= f * s.binvb[pr]
	}
}

// solution reports structural X values and the objective c·X.
func (s *bvSolver) solution() Solution {
	x := make([]float64, s.n)
	// Basic structurals take their basic value; nonbasic structurals their bound.
	for j := 0; j < s.n; j++ {
		if !s.inBasis[j] {
			x[j] = s.nonbasicValue(j)
		}
	}
	for r := 0; r < s.m; r++ {
		b := s.basis[r]
		if b < s.n {
			x[b] = s.xB[r]
		}
	}
	obj := 0.0
	for j := 0; j < s.n; j++ {
		obj += s.cost[j] * x[j]
	}
	return Solution{Status: Optimal, Objective: obj, X: x}
}
