package lp

import (
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
